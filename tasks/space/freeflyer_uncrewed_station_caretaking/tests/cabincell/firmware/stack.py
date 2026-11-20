"""The robot side of the cell's wiring: localiser stack, rig, firmware layer with every option of this task, the task
firmware.  ``build_stack`` takes explicit parts (a bench scene); ``stack_for_site`` derives them from a site built by
the station package (``station.sites.build_site`` with the robot, ``station.site_rig.station_components``).

Localiser stack (bottom to top): ``spacekit.sensors.Localizer`` (the map estimate, consistent error model, richness =
``localisation.CabinRichness`` or full) -> ``localisation.GatedFiducialLocalizer`` (fiducial-relative mode, gated by
the plate's visibility) -> ``localisation.RailLocalizer`` (rail-relative mode of capture).  The camera-visibility tests
(the richness, the plate gate, the API's evidence rays) share one ``localisation.RobotOccluder``: the robot as the
frames show it (the maker's mesh, the frame recipe's near clip ``render.near_clip_m``), ``tf.occluder``.  Rig: ``spacekit.robot.
FreeFlyerRig`` with the reference configuration, the IMU sampled, ``power.TaskBattery``, the station frame, the cabin
air for the robot's drag, the dock model of each berth (``spacekit.dock.BerthDock``, fast source), the caller's
fast / slow sources (suction at 50 Hz is a slow source).  Layer: ``spacekit.primitives.FirmwareLayer`` with
``ZoneMap(inflation_arm=...)``, ``fiducial_scope='action'``, ``holding_interlock=True``, ``zone_band_exit=True``.
``spec.apply_dock_defaults`` at build time.  Seeds: one ``numpy.random.SeedSequence(seed)`` spawns the streams
(rig / IMU, map, fiducial, rail, bar detector) in a fixed order.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from spacekit import dock as DK
from spacekit import gripper as GR
from spacekit import primitives as PR
from spacekit.robot import FreeFlyerAdapter, FreeFlyerRig, FreeFlyerSpec
from spacekit.sensors import Localizer, camera_model
from spacekit.cameras import IMAGE_SIZE

from . import bars as BR
from . import localisation as LO
from .firmware import TaskFirmware
from .power import TaskBattery, Telemetry
from .spec import apply_dock_defaults, default_spec, load_robot_values
from .zonecheck import ApproachZone, ZoneRules, box_zone

STREAMS = ("rig", "map", "fiducial", "rail", "detector")


def plate_record(pid: str, name: str, points, normal, t1, owner: str, skip) -> dict:
    """A fiducial plate as the firmware uses it: marker points, normal, plate frame (x = t1, z = normal), owner (the
    approach zone that exempts fly_rel to it), own geoms (transparent to its rays)."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    x = np.asarray(t1, float) - (np.asarray(t1, float) @ n) * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    P = np.atleast_2d(np.asarray(points, float))
    return dict(id=pid, name=name, points=P, normal=n, center=P.mean(axis=0), R=np.column_stack([x, y, n]), owner=owner,
                skip=tuple(int(g) for g in skip), t1=x)


def geoms_with_prefix(model, prefix: str) -> list:
    return [g for g in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(prefix)]


UNIT_PLATES = (("F1", "hs+b", "fid_top"), ("F2", "hs-b", "fid_bottom"), ("F3", "pp0", "pp0_fid"), ("F4", "pp1", "pp1_fid"))


def unit_plates(model, U: dict, scene_prefix: str = "su_") -> dict:
    """Plates of one stowage unit (an ``info['units']`` entry): SB-<M>-F1 = the plate on the screen's +b side,
    -F2 = -b side, -F3 / -F4 = the plates of pouch pads P1 / P2 (the station data's pp0 at -a / pp1 at +a), at the unit's pad side."""
    part, mod = U["part"], U["module"].upper()
    out = {}
    for fid, which, nm in UNIT_PLATES:
        pts, n = part.fiducial_points(which)
        t1, _ = part.fiducial_axes(which)
        name = f"{scene_prefix}{part.name}_{nm}"
        out[f"SB-{mod}-{fid}"] = plate_record(f"SB-{mod}-{fid}", name, pts, n, t1, owner=f"SB-{mod}",
                                              skip=geoms_with_prefix(model, name))
    return out


def dock_plate(model, data, dock_quat, berth_axis=None) -> dict:
    """DOCK-F1: the dock's target face (``dock_c_dock_target_face``), normal = the dock's egress axis."""
    x = DK.quat_to_mat(np.asarray(dock_quat, float))[:, 0] if berth_axis is None else np.asarray(berth_axis, float)
    pts, nrm = DK.plate_points(model, data, "dock_c_dock_target_face", x)
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "dock_c_dock_target_face")
    R = data.geom_xmat[g].reshape(3, 3)
    h = model.geom_aabb[g][3:]
    inplane = [k for k in range(3) if abs(float(R[:, k] @ nrm)) < 0.5]
    k_long = max(inplane, key=lambda k: h[k])
    skip = geoms_with_prefix(model, "dock_c_dock_target") + geoms_with_prefix(model, "dock_v_dock_target")
    return plate_record("DOCK-F1", "dock_target", pts, nrm, R[:, k_long], owner="DOCK", skip=skip)


def dock_zone(feats: dict, spec) -> ApproachZone:
    """The dock's approach zone (firmware.dock.approach_zone, the unit rule of world.stowage_unit.approach_zone): its
    collision box outline grown by the zone margin, from the dock face to the zone depth along the egress axis."""
    D = feats["dock"]
    R = DK.quat_to_mat(np.asarray(D["quat_wxyz"], float))
    cb = D["collision_box"]
    face = np.asarray(cb["pos"], float) + R[:, 0] * float(cb["half"][0])
    return box_zone("DOCK", face, (R[:, 1], R[:, 2], R[:, 0]), (float(cb["half"][1]), float(cb["half"][2])),
                    spec.dock_approach_grow_m, spec.dock_approach_depth_m)


def build_stack(model, data, *, zones: list, plates: dict, approach_zones: dict, bars: list, berths: dict | None = None,
                features: dict | None = None, lighting: dict | None = None, flow=None, station=None, suction=None,
                extra_sources=(), extra_slow_sources=(), seed: int = 0, soc0: float = 1.0, docked: bool = False,
                berth: int | None = None, mode: str | None = None, spec=None, robot_values: dict | None = None,
                frame: bool = True, clock=None, capture_budget=None, loc_noise: bool = True) -> TaskFirmware:
    """Wire the robot of a compiled scene (module doc).  features: map feature points (None = full richness);
    lighting: {module: state}; berths: {1: pose, 2: pose} (robot body pose on each berth; None = no dock);
    docked: the robot starts on ``berth``; mode: flight mode at the start (default 'off' docked, else 'standard')."""
    from spacekit.frames import RotatingStationFrame

    spec = spec or default_spec()
    robot_values = robot_values or load_robot_values()
    apply_dock_defaults(spec)
    adapter = FreeFlyerAdapter(model)
    body = adapter.body
    ss = np.random.SeedSequence(int(seed))
    r_rig, r_map, r_fid, r_rail, r_det = (np.random.default_rng(s) for s in ss.spawn(len(STREAMS)))
    occluder = LO.RobotOccluder(model, int(model.body_rootid[body]), spec.near_clip_m)
    richness = None
    if features is not None:
        richness = LO.CabinRichness(model, data, body, features, lighting or {}, spec.lighting(), LO.lamp_positions(robot_values),
                                    rate_hz=spec.sparse_map_hz, occluder=occluder)
    base = Localizer(r_map, richness=richness) if loc_noise else None
    K = np.asarray(camera_model("aft_wide")["K"], float)
    cam = LO.PlateCamera(model, body, K, IMAGE_SIZE["aft_wide"], spec.view_cone_deg)
    tester = LO.RayTester(model, model.body_rootid[body], occluder)
    gate = dict(cam=cam, tester=tester, min_points=spec.min_visible_markers, rate_hz=spec.fiducial_check_hz)
    fid = LO.GatedFiducialLocalizer(r_fid, base=base, gate=gate, half_fov_deg=spec.view_cone_deg,
                                    hold_s=spec.loss_hold_s)
    top = LO.RailLocalizer(fid, r_rail)
    rc = FreeFlyerSpec.reference_configuration()
    battery = TaskBattery(rc.battery_count, rc.battery_capacity_wh, soc0=soc0, other_w=spec.other_w, lamp_w=spec.lamp_power_w)
    magnets = {}
    if berths:
        for k, pose in berths.items():
            magnets[int(k)] = DK.BerthDock(body, (np.asarray(pose[0], float), np.asarray(pose[1], float)))
    fast = list(magnets.values()) + list(extra_sources)
    slow = list(extra_slow_sources)
    if suction is not None:
        slow.append(suction)
    st_frame = RotatingStationFrame(model) if frame else None
    start_mode = mode or ("off" if docked else "standard")
    rig = FreeFlyerRig(model, data, spec=rc, mode=start_mode, rng=r_rig, localizer=top, spin_up=not docked,
                       station_frame=st_frame, flow=flow, drag=True, battery=battery, extra_sources=fast,
                       extra_slow_sources=slow, slow_every=10)
    rules = ZoneRules(zones, spec, approach_zones)
    b0 = int(berth) if berth is not None else (min(magnets) if magnets else None)
    layer = PR.FirmwareLayer(rig, GR.ARM_NAMES, berth_pose=(berths[b0] if berths else None), station=station, suction=suction,
                             zones=rules.map, magnets=(magnets.get(b0) if magnets else None), zone_band_exit=True,
                             fiducial_scope="action", holding_interlock=True)
    layer.docked = bool(docked)
    _check = layer.check_segment

    def check_segment(seg, arm_exempt: bool = False):
        """The layer's validation; a move with no segment (already at the goal) has nothing to validate (spacekit 0.4.0
        ``ZoneMap.check_positions`` indexes the last sample of an empty path)."""
        if not seg:
            return "ok", -1
        return _check(seg, arm_exempt)
    layer.check_segment = check_segment
    off, ax = DK.camera_in_body(model, LO.AWC, body)
    if "DOCK-F1" in plates:
        P = plates["DOCK-F1"]
        layer.dock_fiducial = dict(name=P["id"], points=P["points"], normal=P["normal"], cam_offset_body=off, cam_axis_body=ax,
                                   skip=P["skip"])
    det = BR.BarDetector(model, bars, body, r_det, spec.detector_sigma_pos_m, spec.detector_sigma_axis_deg)
    tel = Telemetry(rig, layer, spec)
    tf = TaskFirmware(rig, layer, spec, rules, det, top, fid, battery, tel, plates, berths=berths, magnets=magnets,
                      richness=richness, robot_values=robot_values, awc=(off, ax), clock=clock, capture_budget=capture_budget)
    if b0 is not None:
        tf.berth = b0
    tf.streams = dict(zip(STREAMS, (r_rig, r_map, r_fid, r_rail, r_det)))
    tf.occluder = occluder
    return tf


def stack_for_site(model, data, info: dict, comps, seed: int = 0, soc0: float = 1.0, item_aero: bool = True, **kw) -> TaskFirmware:
    """``build_stack`` for a site built by the station package with the robot on its berth: zones, plates (every
    stowage unit's four plates and the dock's), approach zones (each unit's, the dock's), bars (the items' handles and
    the map's handrails), berths, map features and lighting of the site, the cabin + unit air, the station model and the
    suction of ``comps`` (``station.site_rig.station_components``), the robot's jets on the items (``item_aero``)."""
    from spacekit import aero as AE
    from station import data as SD
    from station import mapfeat as MF

    spec = kw.pop("spec", None) or default_spec()
    scene = info.get("scene", "mission")
    feats = SD.features(scene)
    zones = SD.zones(scene)
    plates, approach = {}, {}
    for U in info.get("units", []):
        plates.update(unit_plates(model, U))
        approach[f"SB-{U['module'].upper()}"] = ApproachZone.from_dict(f"SB-{U['module'].upper()}", U["part"].approach_zone())
    D = feats["dock"]
    berths = {k: DK.berth_contact_pose(D["pos"], D["quat_wxyz"], k) for k in (1, 2)}
    if info.get("dock") is not None or any((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "") == "dock_c_dock_target_face"
                                           for g in range(model.ngeom)):
        plates["DOCK-F1"] = dock_plate(model, data, D["quat_wxyz"])
        approach["DOCK"] = dock_zone(feats, spec)
    bars = BR.item_bars(model, info) + BR.map_bars(model, feats)
    tag = SD.SCENES[scene]["tag"]
    features = MF.from_json(SD.load_json("map_features.json")[tag])
    lighting = dict(info.get("lighting") or {})
    extra_slow = list(kw.pop("extra_slow_sources", ()))
    holder = {}
    jets = None
    if item_aero and comps is not None and getattr(comps, "shapes", None):
        ad = FreeFlyerAdapter(model)
        jets = AE.NozzleJetField(ad.body, ad.nozzle_pos, ad.nozzle_dir, thrust_fn=lambda: holder["tf"].rig.plant.thrust,
                                 dp_fn=lambda: [mm.dp for mm in holder["tf"].rig.plant.modules]).bind(data)
        aero = AE.ItemAero(model, comps.shapes, flow=comps.flow, jets=jets, shelters=comps.shelters)
        extra_slow.append(aero)
    berth = int((info.get("robot") or {}).get("berth", 1))
    tf = build_stack(model, data, zones=zones, plates=plates, approach_zones=approach, bars=bars, berths=berths,
                     features=features, lighting=lighting, flow=(comps.flow if comps is not None else None),
                     station=(comps.station if comps is not None else None), suction=(comps.suction if comps is not None else None),
                     extra_slow_sources=extra_slow, seed=seed, soc0=soc0, docked=kw.pop("docked", True), berth=berth,
                     spec=spec, **kw)
    holder["tf"] = tf
    tf.jets = jets
    tf.site_info = info
    tf.comps = comps
    tf.features_scene = feats
    return tf
