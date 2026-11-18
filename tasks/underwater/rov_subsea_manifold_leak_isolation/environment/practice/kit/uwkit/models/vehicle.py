"""Vehicle MJCF builder: reference-vehicle digest (spec_values + manifest + mesh library) -> MjSpec / XML.

    vb = build_vehicle()                                    # the mission configuration, fresh-water ballast
    vb = build_vehicle("mission", ballast="sea")            # digest v3: the sea-water trim (fresh layout + added units, its battery position), rho_build = sea
    vb = build_vehicle("heavy"), build_vehicle("std")       # the other two digest configurations
    vb = build_vehicle("mission_skid")                      # task 02 host: mission + the second (skid) bottom plate, DVL moved there
    m  = vb.compile();  xml = vb.to_xml();  vb.export(dir)  # MjModel / self-contained XML / XML + neutral mesh copies

Bodies of the reference model (B = FLU body frame, origin = CAD origin; every body frame is B-aligned):
  hull        free joint; all rigid parts; mass / CG / inertia = digest totals minus the moving bodies below, so that
              the whole model reproduces the digest exactly; merged added mass (+ gravcomp); hydro record "full"
  cam_tilt    camera module + tilt bracket (inside the dry enclosure), hinge about -y through the tilt pivot:
              joint angle = optical axis UP positive (decision U8); position servo "cam_tilt"; camera "cam_front"
  jaw_upper,  the two gripper jaws (mission), each on a hinge about its own pin, coupled 1:1 by an equality constraint
  jaw_lower   (the pusher linkage) -> 1 DOF; one position servo "gripper" on jaw_upper; torque-limited so that the
              squeeze force at the TCP equals the official grip force; wet "simple" hydro bodies
  sonar, dvl  fixed (jointless, massless) bodies carrying the acoustic sensors' own housings, so that each acoustic
              sensor can exclude exactly its own housing and still see the rest of the vehicle (self-occlusion)

Geom groups: 1 = vehicle visual meshes the forward camera can never see (inside / behind the dome, enclosure parts),
2 = vehicle visual meshes that can appear in the forward camera (gripper, front of the frame, lights ...),
3 = vehicle collision (digest convex pieces, merged hulls, or none), 0 = world.  Visual geoms do not collide.
Collision proxies (collision=): "mesh" = the digest's convex pieces (501 for mission sea, default); "merged" = one
convex hull per assembly (34 for mission, models/collision_proxy.py); "none".  Jaws (jaw_collision=): "shell" = one box
on each gripping face + two convex pieces per jaw cut from the jaw hull (throat / knuckles behind the pads, the
tapered back), never inside the gripping gap (default since 0.3.0); "pads" = the boxes only (0.2.0 default).
Collision bit masks: vehicle geoms (contype 2, conaffinity 1) collide with ordinary geoms (1, 1) but not with each
other; a grasped object is switched to (4, 1).

The legacy single-box builder (vehicle=VehicleSpec, or config="placeholder") is kept for the analytic tests and for
reproducing the pre-digest numbers.

All uwkit data (hydro records, thruster model, battery, hotel loads, articulation, provenance) travel inside the model
as <custom><text name="uwkit"> JSON, so a saved XML is self-contained.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mujoco

from ..frames import mat_to_quat, quat_from_x_axis, quat_to_mat
from ..params import (G, HydroCoeffs, VehicleSpec, ThrusterSpec, ThrusterModelSpec, BatterySpec, load_hydro,
                      load_vehicle_spec, default_hydro, RHO_FRESH, MODELS_DIR)
from ..dynamics.hydro import HydroBodyParams, UWKIT_TEXT, iso_split
from .. import refdata
from .. import hydrostatics as hs
from . import collision_proxy as cp

HULL = "hull"
THRUSTER_SITE_PREFIX = "thr_"
VEHICLE_CONTYPE, VEHICLE_CONAFFINITY = 2, 1
GRASPED_CONTYPE, GRASPED_CONAFFINITY = 4, 1
SENSOR_SITES = ("cam_front", "sonar_head", "dvl", "depth", "imu", "light_1", "light_2", "light_3",
                "light_4", "gripper_tcp")
GROUP_VIS_HIDDEN, GROUP_VIS_CAMERA, GROUP_COLLISION = 1, 2, 3
REFERENCE_CONFIGS = ("mission", "heavy", "std", "mission_skid")
DEFAULT_JAW_COLLISION = "shell"            # 0.3.0 (0.2.0: "pads")


@dataclass
class VehicleBuild:
    spec: mujoco.MjSpec
    vehicle: VehicleSpec
    hydro: HydroCoeffs
    meta: dict
    rho_build: float
    objects: list = field(default_factory=list)

    def write_meta(self):
        _set_meta(self.spec, self.meta)

    def compile(self) -> mujoco.MjModel:
        self.write_meta()
        return self.spec.compile()

    def to_xml(self) -> str:
        self.write_meta()
        self.spec.compile()
        return self.spec.to_xml()

    def export(self, out_dir: str, model_name: str = "vehicle.xml", keep_sources: bool = False) -> str:
        """Write a self-contained model directory: <out_dir>/<model_name> + meshes/m###.stl (neutral file names,
        meshdir relative to the XML).  Provenance strings and the mesh-library path are dropped unless keep_sources
        (they name maintainer-side files)."""
        out_dir = os.path.abspath(out_dir)
        mdir = os.path.join(out_dir, "meshes")
        os.makedirs(mdir, exist_ok=True)
        self.write_meta()
        # MjSpec.copy() corrupts memory (MuJoCo 3.14.0) when a material's texture list is shorter than mjNTEXROLE
        # (e.g. textures=["", "sand"] in a task scene); padding with "" is a no-op for the compiled model
        pad_material_textures(self.spec)
        s = self.spec.copy()
        mapping = {}
        for mesh in s.meshes:
            if not mesh.file:                     # vertex-only mesh (merged hulls, jaw shells): stays inline in the XML
                continue
            src = mesh.file if os.path.isabs(mesh.file) else os.path.join(self.spec.meshdir or "", mesh.file)
            if src not in mapping:
                dst = f"m{len(mapping):03d}{os.path.splitext(src)[1].lower()}"
                shutil.copyfile(src, os.path.join(mdir, dst))
                mapping[src] = dst
            mesh.file = mapping[src]
        s.meshdir = mdir
        meta = copy.deepcopy(self.meta)
        if not keep_sources:
            meta["sources"] = {}
            meta.get("reference", {}).pop("mesh_root", None)
            # coefficient-set labels are file stems of the grader-side sets; the numbers stay (the model needs them),
            # so a model built with drawn coefficients remains verifier / service side (decision K5)
            meta["hydro_set"] = "coefficients"
            meta["hydro_coeffs"]["name"] = "coefficients"
        _set_meta(s, meta)
        s.compile()
        xml = s.to_xml().replace(f'meshdir="{mdir}/"', 'meshdir="meshes/"').replace(f'meshdir="{mdir}"', 'meshdir="meshes"')
        xml = _full_precision_inertials(xml, self.spec)
        path = os.path.join(out_dir, model_name)
        with open(path, "w") as f:
            f.write(xml)
        return path


# --------------------------------------------------------------------------- helpers
def _full_precision_inertials(xml: str, spec: mujoco.MjSpec) -> str:
    """MjSpec.to_xml writes about 6 significant digits and diagonalised inertias; rewrite the explicit inertial of
    every body (mass, CoM, full inertia, gravcomp) with full precision so that an exported model keeps the digest's
    mass properties to ~1e-12 instead of ~3e-6."""
    by_name = {b.name: b for b in spec.bodies if b.name}
    pat = re.compile(r'(<body name="([^"]+)"[^>]*?)>(\s*)<inertial [^>]*/>')

    def sub(mt):
        b = by_name.get(mt.group(2))
        if b is None or not b.explicitinertial:
            return mt.group(0)
        tag = re.sub(r'\s+gravcomp="[^"]*"', "", mt.group(1))
        if b.gravcomp:
            tag += f' gravcomp="{float(b.gravcomp)!r}"'
        inr = (f'<inertial pos="{" ".join(repr(float(x)) for x in b.ipos)}" mass="{float(b.mass)!r}" '
               f'fullinertia="{" ".join(repr(float(x)) for x in b.fullinertia)}"/>')
        return f"{tag}>{mt.group(3)}{inr}"
    return pat.sub(sub, xml)


N_TEXROLE = int(mujoco.mjtTextureRole.mjNTEXROLE)


def pad_material_textures(spec: mujoco.MjSpec) -> int:
    """Pad every material's texture-role list to mjNTEXROLE entries (empty = no texture); returns the number padded.
    Needed before MjSpec.copy() in MuJoCo 3.14.0 (a shorter list corrupts memory); no effect on the compiled model."""
    n = 0
    for m in spec.materials:
        t = list(m.textures)
        if len(t) < N_TEXROLE:
            m.textures = t + [""] * (N_TEXROLE - len(t))
            n += 1
    return n


def _set_meta(spec: mujoco.MjSpec, meta: dict):
    txt = None
    for t in spec.texts:
        if t.name == UWKIT_TEXT:
            txt = t
    if txt is None:
        txt = spec.add_text(name=UWKIT_TEXT)
    txt.data = json.dumps(meta, separators=(",", ":"))


def _fullinertia(I):
    I = np.asarray(I, float)
    return [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]


def hull_hydro_record(v: VehicleSpec, h: HydroCoeffs, rho_build: float, volume=None, cob=None, mass=None,
                      com=None, inertia=None) -> HydroBodyParams:
    A = h.added_mass * (rho_build / h.rho_ref)
    m_iso, I_A = iso_split(A)
    return HydroBodyParams(kind="full", volume=v.volume_m3 if volume is None else volume,
                           cob=v.cob_m if cob is None else cob, added_mass=A,
                           merged=np.concatenate([[m_iso], I_A]), damping_linear=h.damping_linear,
                           damping_quadratic=h.damping_quadratic, rho_ref=h.rho_ref,
                           half_height=v.half_height_m, rigid_mass=v.mass_kg if mass is None else mass,
                           rigid_com=v.com_m if com is None else com,
                           rigid_inertia=v.inertia_com_kgm2 if inertia is None else inertia)


def simple_hydro_record(volume, cob, rigid_mass, rigid_com, rigid_inertia, rho_build, ca=0.5,
                        ca_rot=0.0, d_lin=(0,) * 6, d_quad=(0,) * 6, half_height=0.05):
    """Isotropic added mass ca * rho V (sphere: 0.5), all of it merged into the body inertia."""
    m_a = ca * rho_build * volume
    Ia = np.full(3, ca_rot)
    return HydroBodyParams(kind="simple", volume=volume, cob=cob,
                           added_mass=np.concatenate([np.full(3, m_a), Ia]),
                           merged=np.concatenate([[m_a], Ia]), damping_linear=d_lin,
                           damping_quadratic=d_quad, rho_ref=rho_build, half_height=half_height,
                           rigid_mass=rigid_mass, rigid_com=rigid_com, rigid_inertia=rigid_inertia)


def _apply_merged_inertia(body, rec: HydroBodyParams):
    m_iso = rec.merged[0]
    body.explicitinertial = True
    body.mass = rec.rigid_mass + m_iso
    body.ipos = rec.rigid_com
    body.fullinertia = _fullinertia(rec.rigid_inertia + np.diag(rec.merged[1:]))
    body.gravcomp = m_iso / (rec.rigid_mass + m_iso) if (rec.rigid_mass + m_iso) > 0 else 0.0


def _load_params(path=None) -> dict:
    with open(path or os.path.join(MODELS_DIR, "vehicle_model_params.json")) as f:
        return json.load(f)


def _load_materials() -> dict:
    with open(os.path.join(MODELS_DIR, "render_materials.json")) as f:
        return json.load(f)


def _T(item) -> np.ndarray:
    return np.asarray(item["T_B"], float)


def _part_of(item, mass=None, volume=None, inertia=None) -> hs.Part:
    return hs.Part(item["id"], item["mass_kg"] if mass is None else mass, item["com_B_m"],
                   item["displacement_m3"] if volume is None else volume, item["cb_B_m"],
                   item.get("inertia_about_com_B_kgm2") if inertia is None else inertia, item.get("category", ""))


def _box_inertia(mass, bbox):
    lo, hi = np.asarray(bbox[0], float), np.asarray(bbox[1], float)
    a, b, c = hi - lo
    return mass / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])


def _plate_dz(h) -> float:
    """Plate offset of a hole tuple (x, y[, dz]): 0 = the vehicle bottom plate, -0.144 = the skid plate (mission_skid)."""
    return float(h[2]) if len(h) > 2 else 0.0


def _same_hole(a, b, tol=1e-6, tol_z=0.02):
    """Same ballast hole: (x, y) within tol and the same plate (dz within tol_z; a missing third element = 0)."""
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol and abs(_plate_dz(a) - _plate_dz(b)) < tol_z


def _hole_tuple(h) -> tuple:
    return tuple(float(v) for v in h[:3]) if len(h) > 2 else (float(h[0]), float(h[1]))


# --------------------------------------------------------------------------- reference vehicle spec
def _is_unit_item(it):
    return it.get("category") == "ballast" or bool(it.get("ballast_station"))


def _art_items(ids, by_id, config):
    """Articulation item ids of the digest (written for 'mission') mapped to this configuration's instances: an id that
    is not in the configuration is looked up with the configuration's own prefix (mission_skid: 'mission.dvl...' ->
    'mission_skid.dvl...')."""
    out = []
    for i in ids:
        if i in by_id:
            out.append(i)
        elif i.startswith("mission.") and f"{config}.{i[len('mission.'):]}" in by_id:
            out.append(f"{config}.{i[len('mission.'):]}")
    return out


def _battery_item(items):
    b = [i for i in items if i["id"] == "battery" or i["id"].endswith(".battery")]
    return b[0] if b else None


def reference_vehicle_spec(config="mission", ballast="fresh", spec_path=None, manifest=None,
                           with_gripper=True, with_tilt=True, params=None, battery_offset=None):
    """VehicleSpec of the WHOLE vehicle (all bodies) from the digest, plus the body layout used by the builder.

    ballast: 'fresh' (instanced layout, fresh water 20 C), 'sea' (mission: the digest's sea-water trim -- the fresh layout +
    added units and its battery position), or a list of hole centres [(x, y) or (x, y, dz) m] replacing the instanced
    ballast units (digest v3: unit = weight + screw; dz = plate offset, mission_skid: -0.144 m on the skid plate, as
    hydrostatics.ballast_layout('mission_skid') writes it; holes with the same (x, y) on different plates are different). battery_offset: absolute axial battery position [m, +x from the centred position];
    None = the digest value of the chosen state (a custom hole list keeps the instanced position).
    Returns (VehicleSpec, layout) with layout = dict(bodies=..., totals=Hydrostatics, ballast_holes=..., ...)."""
    if config not in REFERENCE_CONFIGS:
        raise ValueError(f"config must be one of {REFERENCE_CONFIGS} (or 'placeholder')")
    S = lambda k, d=KeyError: refdata.spec(k, spec_path, d)          # noqa: E731
    man = manifest if manifest is not None else refdata.load_manifest()
    P = params or _load_params()
    items = refdata.manifest_items(config, man, include_derived=False)
    by_id = {i["id"]: i for i in items}
    art = S("uwkit.mission.articulation")
    # ---------------------------------------------------------------- totals (+ ballast change)
    lump = hs.lumped_from_spec(config, "fresh_20C", spec_path)
    inst = hs.ballast_layout(config, "fresh_20C", spec_path)
    if isinstance(ballast, str) and ballast in ("fresh", "fresh_20C", "instanced"):
        target, ballast_name = inst, "fresh_20C"
    elif isinstance(ballast, str) and ballast in ("sea", "sea_SP35", "sea_SP35_20C"):
        target, ballast_name = hs.ballast_layout(config, "sea_SP35", spec_path), "sea_SP35_20C"
    else:
        target, ballast_name = [_hole_tuple(h) for h in ballast], "custom"
    if inst is None and target is not None:
        raise ValueError(f"the {config} configuration keeps the vendor default ballast; no layout change supported")
    unit = hs.ballast_unit(spec_path)
    removed = [] if inst is None else [h for h in inst if not any(_same_hole(h, t) for t in target)]
    added = [] if inst is None else [h for h in target if not any(_same_hole(h, i) for i in inst)]
    parts = [lump]
    for h in removed:
        p = hs.PartTable().add_ballast(h, unit)
        parts.append(hs.Part(p.name, -p.mass_kg, p.com_m, -p.displacement_m3, p.cb_m, -p.inertia_com_kgm2, "ballast"))
    for h in added:
        parts.append(hs.PartTable().add_ballast(h, unit))
    # battery position (digest v3: a trim setting); the lump carries the instanced position
    d_inst = hs.battery_offset(config, "fresh_20C", spec_path)
    if battery_offset is None:
        battery_offset = hs.battery_offset(config, "sea_SP35", spec_path) if ballast_name == "sea_SP35_20C" else d_inst
    lo_b, hi_b = hs.battery_travel(spec_path)
    if not (lo_b - 1e-9 <= battery_offset <= hi_b + 1e-9):
        raise ValueError(f"battery offset {battery_offset:+.4f} m outside its travel [{lo_b:+.4f}, {hi_b:+.4f}] m")
    bat_item = _battery_item(items)
    bat_delta = float(battery_offset - d_inst)
    if abs(bat_delta) > 0.0:
        bp = _part_of(bat_item)
        parts.append(hs.Part(bp.name, -bp.mass_kg, bp.com_m, 0.0, bp.cb_m, -bp.inertia_com_kgm2, bp.category))
        parts.append(hs.Part(bp.name, bp.mass_kg, bp.com_m + np.array([bat_delta, 0, 0]), 0.0, bp.cb_m + np.array([bat_delta, 0, 0]),
                             bp.inertia_com_kgm2, bp.category))
    tot = hs.compute(parts, refdata.water_density(ballast_name if ballast_name != "custom" else "fresh_20C", spec_path),
                     refdata.gravity(spec_path))
    # ---------------------------------------------------------------- moving / sensor bodies
    bodies = {}                                    # name -> dict(items=[ids], parts=[hs.Part], joint=...)
    tilt_ids = _art_items(art["camera_tilt"]["items"], by_id, config) if with_tilt else []
    if tilt_ids:
        bodies["cam_tilt"] = dict(items=tilt_ids, parts=[_part_of(by_id[i], volume=0.0) for i in tilt_ids],
                                  pos=np.asarray(art["camera_tilt"]["pivot_B_m"], float),
                                  axis=np.asarray(art["camera_tilt"]["axis_B"], float), wet=False)
    gr = art["gripper"]
    has_gripper = gr["jaw_upper"] in by_id and gr["jaw_lower"] in by_id
    if has_gripper and with_gripper:
        for side in ("upper", "lower"):
            it = by_id[gr[f"jaw_{side}"]]
            m = float(it["solid_volume_m3"]) * float(it["density_kg_m3"])
            V = float(it["solid_volume_m3"])
            p = hs.Part(it["id"], m, it["com_B_m"], V, it["com_B_m"], _box_inertia(m, it["bbox_B_m"]), "actuator")
            bodies[f"jaw_{side}"] = dict(items=[it["id"]], parts=[p], pos=np.asarray(gr[f"pivot_{side}_B_m"], float),
                                         axis=np.asarray(gr[f"open_axis_{side}_B"], float), wet=True)
    for sname in ("sonar", "dvl"):
        ids = _art_items(art[sname]["items"], by_id, config)
        if ids:
            bodies[sname] = dict(items=ids, parts=[], pos=np.zeros(3), axis=None, wet=False, fixed=True)
    moving = [p for b in bodies.values() for p in b["parts"]]
    neg = [hs.Part(p.name, -p.mass_kg, p.com_m, -p.displacement_m3, p.cb_m, -p.inertia_com_kgm2) for p in moving]
    hull = hs.compute([hs.Part("total", tot.mass_kg, tot.cg_m, tot.displacement_m3, tot.cb_m, tot.inertia_cg_kgm2)]
                      + neg, tot.rho, tot.g) if moving else tot
    # ---------------------------------------------------------------- thrusters, sites
    thr = [ThrusterSpec(t["site"], t["pos_m"], t["axis_unit"]) for t in S(f"{config}.thrusters")]
    thr_q = {t["site"]: t["quat_wxyz"] for t in S(f"{config}.thrusters")}
    sites_all = S(f"{config}.sites")
    sites = {k: v["pos_m"] for k, v in sites_all.items() if not k.startswith(THRUSTER_SITE_PREFIX)}
    squats = {k: v["quat_wxyz"] for k, v in sites_all.items() if not k.startswith(THRUSTER_SITE_PREFIX)}
    squats.update(thr_q)
    bb = S(f"{config}.bbox_B_m")
    sspec = S("sensor_specs")
    ct = P["camera_tilt"]
    cam_tilt = {}
    if tilt_ids:
        cam_tilt = dict(pivot_m=list(art["camera_tilt"]["pivot_B_m"]), axis=list(art["camera_tilt"]["axis_B"]),
                        range_rad=[math.radians(x) for x in sspec["cam_front"]["tilt_range_deg"]],
                        rate_rad_s=math.radians(ct["rate_deg_s"]), kp=ct["kp_N_m_per_rad"], kv=ct["kv_N_m_s_per_rad"],
                        damping=ct["damping_N_m_s_per_rad"], armature=ct["armature_kg_m2"],
                        mass_kg=sum(p.mass_kg for p in bodies["cam_tilt"]["parts"]), sign="up_positive",
                        vfov_deg=sspec["cam_front"]["vfov_deg_air"], hfov_deg=sspec["cam_front"]["hfov_deg_air"])
        domes = [i for i in items if i.get("generic_name", "").endswith("acrylic dome")]
        dc = np.asarray(sspec["cam_front"]["dome_centre_m"], float)
        lens_off = sspec["cam_front"]["lens_ahead_of_dome_centre_mm"] * 1e-3
        if domes:
            R_out = float(domes[0]["bbox_B_m"][1][0] - dc[0])          # dome apex ahead of the sphere centre
            cam_tilt["port"] = dict(type="dome", centre_B_m=dc.tolist(), outer_radius_m=R_out, lens_offset_m=lens_off,
                                    clearance_m=R_out - lens_off,
                                    note="entrance pupil modelled at the lens (site cam_front); in-water FOV = in-air FOV")
    gripper = {}
    if has_gripper and with_gripper:
        gp, gs = P["gripper"], sspec["gripper_tcp"]
        ax = np.asarray(gr["gripper_axis_B"], float)
        oz = np.asarray(gr["opening_dir_B"], float)
        piv_u = np.asarray(gr["pivot_upper_B_m"], float)
        tcp = np.asarray(gr["tcp_B_m"], float)
        tip = np.asarray(gr["jaw_tip_B_m"], float)
        x_piv, x_tcp, x_tip = (float((p - tcp) @ ax) for p in (piv_u, tcp, tip))
        z_piv = float((piv_u - tcp) @ oz)
        face = gp["pad_face_offset_m"]
        dx, dz = x_tip - x_piv, face - z_piv
        half_open = 0.5 * gs["jaw_opening_mm"] * 1e-3

        def z_tip(th):
            return z_piv + dx * math.sin(th) + dz * math.cos(th)
        lo, hi = 0.0, 1.2
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if z_tip(mid) < half_open else (lo, mid)
        th_open = 0.5 * (lo + hi)
        cal = gp["force_calibration_point"]
        F = gs["grip_force_middle_N"] if cal == "tcp" else gs["grip_force_tip_N"]
        lever_cal = (x_tcp if cal == "tcp" else x_tip) - x_piv
        tau = 2.0 * F * lever_cal
        gripper = dict(type="hinge_pair", theta_open_rad=th_open, torque_max_N_m=tau, kp=gp["kp_N_m_per_rad"],
                       kv=gp["kv_N_m_s_per_rad"], armature=gp["armature_kg_m2"], damping=gp["joint_damping_N_m_s_per_rad"],
                       closed_margin_rad=gp["closed_margin_rad"],
                       close_open_s=gp["close_open_s"] or gs["open_close_s"], opening_m=gs["jaw_opening_mm"] * 1e-3,
                       lever_tcp_m=x_tcp - x_piv, lever_tip_m=x_tip - x_piv,
                       grip_force_tcp_N=tau / (2 * (x_tcp - x_piv)), grip_force_tip_N=tau / (2 * (x_tip - x_piv)),
                       official_grip_force_N=dict(tip=gs["grip_force_tip_N"], middle=gs["grip_force_middle_N"]),
                       calibration=cal, tcp_B_m=tcp.tolist(), axis_B=ax.tolist(), opening_dir_B=oz.tolist(),
                       pad=dict(face=face, thickness=gp["pad_thickness_m"], half_width=gp["pad_half_width_m"],
                                x_from=-gp["pad_back_from_tcp_m"], x_to=x_tip - x_tcp, friction=gp["pad_friction"]),
                       ca=gp["added_mass_coefficient"])
    tm = P["thruster_model"]
    v = VehicleSpec(
        name=f"reference_{config}", mass_kg=tot.mass_kg, com_m=tot.cg_m, inertia_com_kgm2=tot.inertia_cg_kgm2,
        volume_m3=tot.displacement_m3, cob_m=tot.cb_m, half_height_m=0.5 * (bb[1][2] - bb[0][2]), thrusters=thr,
        thruster_model=ThrusterModelSpec(**tm), collision_boxes=[], sites=sites, camera_tilt=cam_tilt,
        gripper=gripper, battery=BatterySpec(**{k: (tuple(x) if isinstance(x, list) else x) for k, x in P["battery"].items()}),
        hotel_W=dict(P["hotel_W"]), site_quats=squats,
        sources={"mass_properties": f"spec_values uwkit.{config}.fresh_20C (+ ballast {ballast_name})",
                 "thrusters": f"spec_values {config}.thrusters", "sites": f"spec_values {config}.sites",
                 "articulation": "spec_values uwkit.mission.articulation",
                 "model_params": "uwkit/models/vehicle_model_params.json"})
    layout = dict(bodies=bodies, totals=tot, hull=hull, ballast_name=ballast_name, ballast_holes=target,
                  ballast_removed=removed, ballast_added=added, items=items, by_id=by_id, articulation=art,
                  battery_offset=float(battery_offset), battery_delta=bat_delta, battery_item=None if bat_item is None else bat_item["id"])
    return v, layout


def _camera_may_see(bbox, cam_pos, pivot, tilt_range, hfov_deg, vfov_deg, n=4):
    """True if any point of a bbox grid lies inside the forward camera's frustum for some tilt angle."""
    lo, hi = np.asarray(bbox[0], float), np.asarray(bbox[1], float)
    g = [np.linspace(lo[k], hi[k], n) for k in range(3)]
    P = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    P = np.vstack([P, 0.5 * (lo + hi)])
    th_x, th_z = math.tan(math.radians(hfov_deg / 2)), math.tan(math.radians(vfov_deg / 2))
    for th in np.radians(np.arange(tilt_range[0], tilt_range[1] + 1e-9, 5.0)):
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])          # rotation about -y by th (optical axis up)
        cp = pivot + R @ (cam_pos - pivot)
        d = (P - cp) @ R                                           # camera frame: x fwd, y left, z up
        ok = (d[:, 0] > 0.005) & (np.abs(d[:, 1]) < th_x * d[:, 0]) & (np.abs(d[:, 2]) < th_z * d[:, 0])
        if ok.any():
            return True
    return False


def _build_reference(config, hydro, ballast, collision, visual, mesh_dir, spec_values, manifest, world, rho_build,
                     pos, quat, timestep, floor_depth, with_gripper, with_tilt, jaw_collision="pads", battery_offset=None) -> VehicleBuild:
    spec_path = spec_values if isinstance(spec_values, str) and spec_values != "auto" else None
    man = manifest if isinstance(manifest, dict) else refdata.load_manifest(manifest if isinstance(manifest, str) else None)
    root = refdata.mesh_root(man, mesh_dir)
    v, L = reference_vehicle_spec(config, ballast, spec_path, man, with_gripper, with_tilt, battery_offset=battery_offset)
    h = default_hydro() if hydro is None else (hydro if isinstance(hydro, HydroCoeffs) else load_hydro(hydro))
    if rho_build is None:
        rho_build = L["totals"].rho
    P = _load_params()
    MAT = _load_materials()
    s = world if world is not None else mujoco.MjSpec()
    if world is None:
        s.modelname = f"uwkit_reference_{config}"
        s.compiler.degree = False
        s.option.timestep = timestep
        s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        s.option.gravity = [0.0, 0.0, -G]
        s.option.density = 0.0
        s.option.viscosity = 0.0
        # grasping: elliptic friction cones with impratio 10 keep a clamped object from creeping out of the jaws
        # (pyramidal / impratio 1: ~8 mm/s creep under a 9 N m clamp; elliptic + 10: < 0.5 mm/s)
        s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        s.option.impratio = 10.0
        s.visual.global_.offwidth = 2400
        s.visual.global_.offheight = 1800
        s.visual.map.znear = 0.005
        s.stat.extent = 2.0
        s.worldbody.add_light(name="sun", pos=[0, 0, 5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        if floor_depth is not None:
            s.worldbody.add_geom(name="seabed", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, 0.1],
                                 pos=[0, 0, -float(floor_depth)], rgba=[0.45, 0.42, 0.33, 1],
                                 friction=[0.8, 0.02, 0.001])
    if s.option.density != 0 or s.option.viscosity != 0:
        raise ValueError("the world spec must have option.density = option.viscosity = 0")
    tot, hullp = L["totals"], L["hull"]
    bodies = L["bodies"]
    # ---------------------------------------------------------------- hull body
    rec_hull = hull_hydro_record(v, h, rho_build, volume=hullp.displacement_m3, cob=hullp.cb_m, mass=hullp.mass_kg,
                                 com=hullp.cg_m, inertia=hullp.inertia_cg_kgm2)
    hull = s.worldbody.add_body(name=HULL, pos=list(pos), quat=list(quat))
    hull.add_freejoint(name="hull_root")
    _apply_merged_inertia(hull, rec_hull)
    mjb = {HULL: hull}
    hydro_bodies = {HULL: rec_hull.to_dict()}
    joints = {}
    for name, b in bodies.items():
        body = hull.add_body(name=name, pos=list(b["pos"]))
        mjb[name] = body
        if b.get("fixed"):
            continue
        pt = hs.compute(b["parts"], tot.rho, tot.g)
        com_rel = pt.cg_m - b["pos"]
        if b["wet"]:
            rec = simple_hydro_record(pt.displacement_m3, pt.cb_m - b["pos"], pt.mass_kg, com_rel,
                                      pt.inertia_cg_kgm2, rho_build, ca=v.gripper.get("ca", 0.5),
                                      half_height=0.02)
            _apply_merged_inertia(body, rec)
            hydro_bodies[name] = rec.to_dict()
        else:
            body.explicitinertial = True
            body.mass = pt.mass_kg
            body.ipos = list(com_rel)
            body.fullinertia = _fullinertia(pt.inertia_cg_kgm2 + np.eye(3) * 1e-9)
    # ---------------------------------------------------------------- materials, meshes, geoms
    item_body = {i: name for name, b in bodies.items() for i in b["items"]}
    used_mats = {}
    mesh_names = {}

    def material(key):
        key = key or "plastic_grey"
        if key not in used_mats:
            rgba, spec_, shin = MAT["materials"].get(key, MAT["default"])
            nm = f"rov_{key}"
            s.add_material(name=nm, rgba=list(rgba), specular=float(spec_), shininess=float(shin))
            used_mats[key] = nm
        return used_mats[key]

    def mesh(rel, prefix):
        path = os.path.join(root, rel)
        if path not in mesh_names:
            nm = f"{prefix}{len(mesh_names):04d}"
            s.add_mesh(name=nm, file=path)
            mesh_names[path] = nm
        return mesh_names[path]

    cam_pos = np.asarray(v.sites.get("cam_front", [0, 0, 0]), float)
    piv = np.asarray(v.camera_tilt.get("pivot_m", cam_pos), float)
    sspec = refdata.spec("sensor_specs", spec_path)
    hf, vf = sspec["cam_front"]["hfov_deg_air"], sspec["cam_front"]["vfov_deg_air"]
    trange = sspec["cam_front"]["tilt_range_deg"]
    # ballast: geometry of removed / added units (digest v3: weight + its screw, the items of a unit share `ballast_station`)
    bal_items = [i for i in L["items"] if _is_unit_item(i)]
    unit_z = float(hs.ballast_unit(spec_path)["com_z_B_m"])
    plates = [0.0]
    if config == "mission_skid":
        plates.append(-float(refdata.spec("mission_skid.datum", spec_path)["plates_apart_m"]))

    def unit_hole(w):
        """(x, y, plate dz) of a ballast weight item (centroid on the hole; dz snapped to the plate offsets)."""
        dz = float(w["com_B_m"][2]) - unit_z
        return (float(w["com_B_m"][0]), float(w["com_B_m"][1]), min(plates, key=lambda z: abs(z - dz)))

    def unit_members(w):
        st = w.get("ballast_station")
        return [i for i in bal_items if st and i.get("ballast_station") == st] if st else [w]
    drop = set()
    weights = [i for i in bal_items if i["category"] == "ballast"]
    for hle in L["ballast_removed"]:
        for w in weights:
            if _same_hole(unit_hole(w), hle, 1e-4):
                drop.update(i["id"] for i in unit_members(w))
    # loose items without a station (pre-v3 manifests): match by (x, y) on the same plate as before
    for hle in L["ballast_removed"]:
        for i in bal_items:
            if not i.get("ballast_station") and i["category"] != "ballast" and \
                    _same_hole((i["com_B_m"][0], i["com_B_m"][1]), (hle[0], hle[1]), 1e-4):
                drop.add(i["id"])
    extra = []
    if L["ballast_added"]:
        w0 = next(i for i in bal_items if i["category"] == "ballast")
        tmpl = [i for i in bal_items if i.get("ballast_station") == w0.get("ballast_station") and
                abs(i["com_B_m"][2] - w0["com_B_m"][2]) < 0.02] if w0.get("ballast_station") else [w0]
        h0 = unit_hole(w0)
        for hle in L["ballast_added"]:
            for t_ in tmpl:
                it = copy.deepcopy(t_)
                T = _T(t_)
                dxy = np.array([hle[0] - h0[0], hle[1] - h0[1], _plate_dz(hle) - h0[2]])
                T[:3, 3] += dxy
                it["T_B"] = T.tolist()
                it["com_B_m"] = (np.asarray(t_["com_B_m"]) + dxy).tolist()
                it["id"] = f"ballast_added@{hle[0]:.5f},{hle[1]:.5f}" + (f",{_plate_dz(hle):.4f}" if _plate_dz(hle) else "") \
                    + ("" if t_["category"] == "ballast" else ".screw")
                lo_, hi_ = np.asarray(t_["bbox_B_m"][0]), np.asarray(t_["bbox_B_m"][1])
                it["bbox_B_m"] = [(lo_ + dxy).tolist(), (hi_ + dxy).tolist()]
                extra.append(it)
    if L["battery_delta"] and L["battery_item"]:
        # battery moved inside its tube: shift its geometry (mass properties are already in the totals)
        drop.add(L["battery_item"])
        it = copy.deepcopy(L["by_id"][L["battery_item"]])
        T = _T(it); T[0, 3] += L["battery_delta"]; it["T_B"] = T.tolist()
        it["bbox_B_m"] = [(np.asarray(it["bbox_B_m"][k]) + np.array([L["battery_delta"], 0, 0])).tolist() for k in (0, 1)]
        extra.append(it)
    n_vis = n_col = 0
    geom_parts = {}                                  # geom name -> generic part name (diagnostics, neutral names)
    thr_part = {t["site"]: t.get("part") for t in refdata.spec(f"{config}.thrusters", spec_path)}
    part_site = {p: s_ for s_, p in thr_part.items() if p}
    thruster_geoms = {s_: [] for s_ in thr_part}     # site -> visual geom names of that thruster (stable mapping)
    collision_groups = {}                            # merged proxy: geom name -> group key (neutral)
    jaw_ids = {bodies[k]["items"][0]: k for k in ("jaw_upper", "jaw_lower") if k in bodies}
    eff_items = [i for i in L["items"] if i["id"] not in drop] + extra
    for it in eff_items:
        if not it.get("mesh"):
            continue
        bname = item_body.get(it["id"], HULL)
        body = mjb[bname]
        T = _T(it)
        off = bodies[bname]["pos"] if bname != HULL else np.zeros(3)
        p_rel = T[:3, 3] - off
        q = mat_to_quat(T[:3, :3])
        if visual:
            # hidden from the forward camera: pressure-housing shells (tubes, dome, flanges, caps), everything inside a
            # housing, the camera's own tilt body; every other part only if no tilt angle can bring it into view
            enclosed = str(it.get("location", "")).startswith("inside") or it.get("location") == "housing_shell" \
                or bname == "cam_tilt"
            grp = GROUP_VIS_HIDDEN if enclosed or not _camera_may_see(it["bbox_B_m"], cam_pos, piv, trange, hf, vf) \
                else GROUP_VIS_CAMERA
            body.add_geom(name=f"rov_v{n_vis:03d}", type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname=mesh(it["mesh"]["visual_lo"], "vm"), pos=list(p_rel), quat=list(q),
                          contype=0, conaffinity=0, group=grp, mass=0.0, material=material(it.get("render_material")))
            geom_parts[f"rov_v{n_vis:03d}"] = it.get("generic_name") or it.get("category", "")
            if it["id"] in part_site:
                thruster_geoms[part_site[it["id"]]].append(f"rov_v{n_vis:03d}")
            n_vis += 1
        if collision == "mesh" and it["id"] not in jaw_ids:
            for c in it["mesh"]["collision"]:
                body.add_geom(name=f"rov_c{n_col:03d}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh(c, "cm"),
                              pos=list(p_rel), quat=list(q), group=GROUP_COLLISION, mass=0.0,
                              friction=P["collision"]["friction"], contype=VEHICLE_CONTYPE,
                              conaffinity=VEHICLE_CONAFFINITY, rgba=[0.15, 0.35, 0.75, 0.5])
                n_col += 1
    if collision == "merged":
        # one convex hull per assembly, all on the hull body (the sonar / dvl bodies are fixed to it, B-aligned, at
        # the origin); MuJoCo builds the hull from the union of the pieces' vertices (vertex-only mesh)
        groups = cp.merged_groups(eff_items, root, exclude_ids=set(jaw_ids))
        for key, g in groups.items():
            V, _ = cp.convex_hull(g["points"])
            mname = f"ch{n_col:03d}"
            s.add_mesh(name=mname, uservert=V.ravel().tolist())
            gname = f"rov_c{n_col:03d}"
            hull.add_geom(name=gname, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname, group=GROUP_COLLISION,
                          mass=0.0, friction=P["collision"]["friction"], contype=VEHICLE_CONTYPE,
                          conaffinity=VEHICLE_CONAFFINITY, rgba=[0.15, 0.35, 0.75, 0.5])
            collision_groups[gname] = key
            n_col += 1
    # ---------------------------------------------------------------- sites
    for t in v.thrusters:
        hull.add_site(name=t.name, pos=list(t.pos), quat=list(v.site_quats.get(t.name, quat_from_x_axis(t.dir))),
                      size=[0.015, 0, 0], rgba=[1, 0.35, 0, 1], group=4)
    site_body = {"cam_front": "cam_tilt", "sonar_head": "sonar", "dvl": "dvl"}
    for name, p in v.sites.items():
        bname = site_body.get(name, HULL) if site_body.get(name) in mjb else HULL
        off = bodies[bname]["pos"] if bname != HULL else np.zeros(3)
        mjb[bname].add_site(name=name, pos=list(np.asarray(p, float) - off), quat=list(v.site_quats.get(name, [1, 0, 0, 0])),
                            size=[0.01, 0, 0], rgba=[0, 1, 0.3, 1], group=4)
    hull.add_site(name="com", pos=list(tot.cg_m), size=[0.008, 0, 0], rgba=[1, 1, 0, 1], group=5)
    hull.add_site(name="cob", pos=list(tot.cb_m), size=[0.008, 0, 0], rgba=[0, 0.8, 1, 1], group=5)
    # ---------------------------------------------------------------- camera tilt joint, servo, camera
    if "cam_tilt" in bodies:
        ct = v.camera_tilt
        tb = mjb["cam_tilt"]
        tb.add_joint(name="cam_tilt", type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(ct["axis"]),
                     range=list(ct["range_rad"]), limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=ct["damping"],
                     armature=ct["armature"])
        cam_p = cam_pos - piv
        Rcam = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])   # -z_cam = +x, +y_cam = +z
        tb.add_camera(name="cam_front", pos=list(cam_p), quat=list(mat_to_quat(Rcam)), fovy=float(ct["vfov_deg"]))
        a = s.add_actuator(name="cam_tilt", target="cam_tilt", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=float(ct["kp"]), kv=float(ct["kv"]))
        a.ctrlrange = list(ct["range_rad"])
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    # ---------------------------------------------------------------- gripper
    if v.gripper:
        gp = v.gripper
        Rg = np.column_stack([gp["axis_B"], np.cross(gp["opening_dir_B"], gp["axis_B"]), gp["opening_dir_B"]])
        qg = mat_to_quat(Rg)
        tcp = np.asarray(gp["tcp_B_m"], float)
        pad = gp["pad"]
        xc = 0.5 * (pad["x_from"] + pad["x_to"])
        for side, sgn in (("upper", 1.0), ("lower", -1.0)):
            jb = mjb[f"jaw_{side}"]
            off = bodies[f"jaw_{side}"]["pos"]
            jb.add_joint(name=f"jaw_{side}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(bodies[f"jaw_{side}"]["axis"]),
                         range=[-gp["closed_margin_rad"], gp["theta_open_rad"] + gp["closed_margin_rad"]],
                         limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=gp["damping"], armature=gp["armature"])
            c_pad = tcp + Rg @ np.array([xc, 0.0, sgn * (pad["face"] + 0.5 * pad["thickness"])])
            jb.add_geom(name=f"jaw_{side}_pad", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.5 * (pad["x_to"] - pad["x_from"]), pad["half_width"], 0.5 * pad["thickness"]],
                        pos=list(c_pad - off), quat=list(qg), condim=4, friction=list(pad["friction"]),
                        solref=[0.004, 1.0], group=GROUP_COLLISION, mass=0.0, contype=VEHICLE_CONTYPE,
                        conaffinity=VEHICLE_CONAFFINITY, rgba=[0.3, 0.3, 0.3, 0.6])
            if jaw_collision == "shell" and collision != "none":
                it = L["by_id"][bodies[f"jaw_{side}"]["items"][0]]
                pts = np.vstack([cp.transform(cp.read_stl_vertices(os.path.join(root, c)), it["T_B"])
                                 for c in it["mesh"]["collision"]])
                for part, q in cp.jaw_shell_pieces(pts, tcp, Rg, sgn, pad).items():
                    if len(q) < 4:
                        continue
                    V, _ = cp.convex_hull(q - off)
                    mname = f"jaw_{side}_{part}"
                    s.add_mesh(name=mname, uservert=V.ravel().tolist())
                    jb.add_geom(name=f"jaw_{side}_{part}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname,
                                condim=4, friction=list(pad["friction"]), solref=[0.004, 1.0], group=GROUP_COLLISION,
                                mass=0.0, contype=VEHICLE_CONTYPE, conaffinity=VEHICLE_CONAFFINITY,
                                rgba=[0.3, 0.3, 0.3, 0.4])
            tip = tcp + Rg @ np.array([pad["x_to"], 0.0, sgn * pad["face"]])
            jb.add_site(name=f"jaw_{side}_tip", pos=list(tip - off), size=[0.004, 0, 0], rgba=[1, 0, 1, 1], group=5)
            s.add_exclude(bodyname1=HULL, bodyname2=f"jaw_{side}")
        s.add_exclude(bodyname1="jaw_upper", bodyname2="jaw_lower")
        e = s.add_equality(name="gripper_coupling", type=mujoco.mjtEq.mjEQ_JOINT, name1="jaw_lower", name2="jaw_upper")
        e.data[0:5] = [0.0, 1.0, 0.0, 0.0, 0.0]
        e.solref = [0.005, 1.0]
        a = s.add_actuator(name="gripper", target="jaw_upper", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=float(gp["kp"]), kv=float(gp["kv"]))
        a.ctrlrange = [0.0, gp["theta_open_rad"]]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-gp["torque_max_N_m"], gp["torque_max_N_m"]]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    # ---------------------------------------------------------------- metadata
    hydro_set_name = h.name
    meta = dict(version=3, vehicle_body=HULL, vehicle_name=v.name, hydro_set=hydro_set_name,
                hydro_coeffs=h.to_dict(), rho_build=rho_build, hydro_bodies=hydro_bodies,
                thrusters=[t.name for t in v.thrusters], thruster_model=v.thruster_model.__dict__,
                battery={k: (list(x) if isinstance(x, tuple) else x) for k, x in v.battery.__dict__.items()},
                hotel_W=v.hotel_W, sources=dict(v.sources), free_objects={},
                gripper=dict(v.gripper) if v.gripper else None,
                camera_tilt=dict(v.camera_tilt) if v.camera_tilt else None,
                reference=dict(config=config, ballast=L["ballast_name"],
                               ballast_holes=None if L["ballast_holes"] is None else [list(x) for x in L["ballast_holes"]],
                               battery_offset_m=L["battery_offset"], battery_item=L["battery_item"],
                               collision=collision, jaw_collision=jaw_collision if v.gripper else None,
                               visual=bool(visual), mesh_root=root, n_visual_geoms=n_vis,
                               n_collision_geoms=n_col, geom_parts=geom_parts, collision_groups=collision_groups,
                               thruster_parts=thr_part, thruster_geoms=thruster_geoms,
                               totals=dict(mass_kg=tot.mass_kg, cg_B_m=tot.cg_m.tolist(), displacement_m3=tot.displacement_m3,
                                           cb_B_m=tot.cb_m.tolist(), inertia_cg_kgm2=tot.inertia_cg_kgm2.tolist(),
                                           rho_ballast=tot.rho),
                               acoustic_self_bodies={k: k for k in ("sonar", "dvl") if k in bodies},
                               geom_groups=dict(vis_hidden=GROUP_VIS_HIDDEN, vis_camera=GROUP_VIS_CAMERA,
                                                collision=GROUP_COLLISION)))
    vb = VehicleBuild(spec=s, vehicle=v, hydro=h, meta=meta, rho_build=rho_build)
    vb.write_meta()
    return vb


# --------------------------------------------------------------------------- main entry
def build_vehicle(config="mission", hydro=None, ballast="fresh", collision="mesh", visual=True,
                  manifest=None, spec_values="auto", vehicle: Optional[VehicleSpec] = None,
                  world: Optional[mujoco.MjSpec] = None, rho_build=None, pos=(0.0, 0.0, -2.0),
                  quat=(1.0, 0.0, 0.0, 0.0), timestep=0.002, floor_depth=None, with_gripper=True,
                  with_tilt=True, keep_placeholder_collision=None, mesh_dir=None,
                  jaw_collision=None, battery_offset=None) -> VehicleBuild:
    """Build the vehicle into `world` (a new MjSpec if None).

    config      : 'mission' (default) | 'heavy' | 'std' | 'mission_skid' (task 02 host: mission + skid, DVL on the skid
                  plate) from the digest; 'placeholder' = the pre-digest box vehicle
    hydro       : name of a coefficient set, a HydroCoeffs (e.g. a per-world draw), or None (nominal set A)
    ballast     : 'fresh' (instanced, fresh water 20 C) | 'sea' (mission sea-water trim: the fresh layout + added units, its battery
                  position; digest v3) | list of hole centres (ballast units = weight + screw)
    battery_offset : absolute axial battery position [m] along +x from the centred position (the rear stop); None = the digest
                  value of the chosen state; must lie in spec 'battery_axial_travel'
    collision   : 'mesh' (digest convex pieces, default) | 'merged' (one convex hull per assembly, 34 for mission;
                  see models/collision_proxy.py) | 'none' (vehicle body without collision; the jaws keep theirs)
    jaw_collision : None = DEFAULT_JAW_COLLISION ('shell' since 0.3.0: pads + two convex pieces per jaw cut from the
                  jaw's hull: throat / pin knuckles behind the pads and the tapered back; never in the gap) | 'pads'
                  (one box per gripping face only; the 0.2.0 default)
    visual      : add the visual_lo meshes (False: no visual geoms; physics unchanged)
    rho_build   : density used to scale and merge the added mass (default: the ballast layout's water)
    mesh_dir    : mesh library root (default: $UWKIT_MESHLIB or the manifest's mesh_root)
    spec_values : 'auto' (default file or $UWKIT_SPEC_VALUES) or a path; manifest: None / path / dict
    vehicle     : a VehicleSpec -> legacy single-body builder (analytic tests)
    floor_depth : add a flat seabed plane at this depth [m] (None = no floor)
    """
    if vehicle is not None or config in ("placeholder", "default"):
        return _build_legacy(config="placeholder" if config in ("placeholder", "default") else config, hydro=hydro,
                             vehicle=vehicle, world=world, rho_build=RHO_FRESH if rho_build is None else rho_build,
                             pos=pos, quat=quat, timestep=timestep, floor_depth=floor_depth,
                             with_gripper=with_gripper, with_tilt=with_tilt,
                             keep_placeholder_collision=keep_placeholder_collision, mesh_dir=mesh_dir)
    if collision not in ("mesh", "merged", "none"):
        raise ValueError("collision must be 'mesh', 'merged' or 'none'")
    if jaw_collision is None:
        jaw_collision = DEFAULT_JAW_COLLISION
    if jaw_collision not in ("pads", "shell"):
        raise ValueError("jaw_collision must be 'pads' or 'shell'")
    return _build_reference(config, hydro, ballast, collision, visual, mesh_dir, spec_values, manifest, world,
                            rho_build, pos, quat, timestep, floor_depth, with_gripper, with_tilt, jaw_collision, battery_offset)


# --------------------------------------------------------------------------- legacy (pre-digest) builder
def _build_legacy(config="placeholder", hydro=None, vehicle: Optional[VehicleSpec] = None,
                  world: Optional[mujoco.MjSpec] = None, rho_build=RHO_FRESH, pos=(0.0, 0.0, -2.0),
                  quat=(1.0, 0.0, 0.0, 0.0), timestep=0.002, floor_depth=None, with_gripper=True,
                  with_tilt=True, keep_placeholder_collision=None, mesh_dir=None) -> VehicleBuild:
    """The single-body box vehicle of the dynamics lane (placeholder spec or a literature test vehicle)."""
    v = copy.deepcopy(vehicle) if vehicle is not None else load_vehicle_spec(config, spec_values=None)
    h = default_hydro() if hydro is None else (hydro if isinstance(hydro, HydroCoeffs) else load_hydro(hydro))
    sources = dict(v.sources)
    s = world if world is not None else mujoco.MjSpec()
    if world is None:
        s.modelname = "uwkit_vehicle"
        s.compiler.degree = False
        s.option.timestep = timestep
        s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        s.option.gravity = [0.0, 0.0, -G]
        s.option.density = 0.0
        s.option.viscosity = 0.0
        s.visual.global_.offwidth = 1280
        s.visual.global_.offheight = 960
        s.stat.extent = 2.0
        s.worldbody.add_light(name="sun", pos=[0, 0, 5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        if floor_depth is not None:
            s.worldbody.add_geom(name="seabed", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, 0.1],
                                 pos=[0, 0, -float(floor_depth)], rgba=[0.45, 0.42, 0.33, 1],
                                 friction=[0.8, 0.02, 0.001])
    if mesh_dir:
        s.meshdir = mesh_dir
    rec_hull = hull_hydro_record(v, h, rho_build)
    hull = s.worldbody.add_body(name=HULL, pos=list(pos), quat=list(quat))
    hull.add_freejoint(name="hull_root")
    _apply_merged_inertia(hull, rec_hull)
    use_boxes = True if keep_placeholder_collision is None else keep_placeholder_collision
    if use_boxes:
        for b in v.collision_boxes:
            hull.add_geom(name=f"hull_{b['name']}", type=mujoco.mjtGeom.mjGEOM_BOX, size=b["half_m"],
                          pos=b["pos_m"], rgba=[0.15, 0.35, 0.75, 0.6], friction=[0.8, 0.02, 0.001],
                          group=3, contype=VEHICLE_CONTYPE, conaffinity=VEHICLE_CONAFFINITY)
    gr = v.gripper if with_gripper else {}
    if gr:
        hull.add_geom(name="gripper_housing", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=list(gr["housing_from_m"]) + list(gr["housing_to_m"]),
                      size=[gr["housing_radius_m"], 0, 0], rgba=[0.2, 0.2, 0.2, 1], group=3,
                      contype=VEHICLE_CONTYPE, conaffinity=VEHICLE_CONAFFINITY)
    for t in v.thrusters:
        hull.add_site(name=t.name, pos=t.pos, quat=quat_from_x_axis(t.dir), size=[0.015, 0, 0],
                      rgba=[1, 0.35, 0, 1], group=4)
    for name, p in v.sites.items():
        if name == "cam_front" and with_tilt and v.camera_tilt:
            continue
        hull.add_site(name=name, pos=p, size=[0.01, 0, 0], rgba=[0, 1, 0.3, 1], group=4)
    hull.add_site(name="com", pos=v.com_m, size=[0.008, 0, 0], rgba=[1, 1, 0, 1], group=5)
    hull.add_site(name="cob", pos=v.cob_m, size=[0.008, 0, 0], rgba=[0, 0.8, 1, 1], group=5)
    hydro_bodies = {HULL: rec_hull.to_dict()}
    cam_meta = None
    if with_tilt and v.camera_tilt:
        ct = v.camera_tilt
        piv = np.asarray(ct["pivot_m"], float)
        tb = hull.add_body(name="cam_tilt", pos=piv)
        tb.explicitinertial = True
        tb.mass = float(ct["mass_kg"])
        tb.ipos = [0.0, 0.0, 0.0]
        tb.inertia = [2e-5, 2e-5, 2e-5]
        # hinge about -y: joint angle = optical axis up (decision U8)
        tb.add_joint(name="cam_tilt", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, -1, 0],
                     range=ct["range_rad"], limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=0.02,
                     armature=1e-4)
        cam_p = np.asarray(v.sites["cam_front"], float) - piv
        tb.add_site(name="cam_front", pos=cam_p, size=[0.01, 0, 0], rgba=[0, 1, 0.3, 1], group=4)
        Rcam = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        tb.add_camera(name="cam_front", pos=cam_p, quat=mat_to_quat(Rcam), fovy=64.0)
        a = s.add_actuator(name="cam_tilt", target="cam_tilt", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=float(ct.get("kp", 2.0)), kv=0.05)
        a.ctrlrange = ct["range_rad"]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        cam_meta = dict(range_rad=list(ct["range_rad"]), rate_rad_s=math.radians(60.0), sign="up_positive")
    if gr:
        x0, L = float(gr["jaw_base_x_m"]), float(gr["jaw_length_m"])
        z0 = float(gr["housing_from_m"][2])
        ht, hz = float(gr["jaw_half_thickness_m"]), float(gr["jaw_half_height_m"])
        mj, Vj = float(gr["jaw_mass_kg"]), float(gr["jaw_volume_m3"])
        Ij = mj / 12.0 * np.diag([(2 * ht) ** 2 + (2 * hz) ** 2, L ** 2 + (2 * hz) ** 2, L ** 2 + (2 * ht) ** 2])
        for side, sgn in (("l", 1.0), ("r", -1.0)):
            jb = hull.add_body(name=f"jaw_{side}", pos=[x0 + 0.5 * L, 0.0, z0])
            jb.add_joint(name=f"jaw_{side}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, sgn, 0],
                         range=[-0.002, float(gr["open_half_gap_m"])], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                         damping=5.0, armature=0.01)
            jb.add_geom(name=f"jaw_{side}_pad", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5 * L, ht, hz],
                        pos=[0.0, sgn * ht, 0.0], rgba=[0.3, 0.3, 0.3, 1], condim=4,
                        friction=[1.2, 0.02, 0.002], solref=[0.004, 1.0], group=3,
                        contype=VEHICLE_CONTYPE, conaffinity=VEHICLE_CONAFFINITY)
            rec = simple_hydro_record(Vj, [0.0, sgn * ht, 0.0], mj, [0.0, sgn * ht, 0.0], Ij, rho_build,
                                      half_height=hz)
            _apply_merged_inertia(jb, rec)
            hydro_bodies[f"jaw_{side}"] = rec.to_dict()
            a = s.add_actuator(name=f"grip_{side}", target=f"jaw_{side}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=float(gr.get("kp", 3000.0)), kv=20.0)
            a.ctrlrange = [0.0, float(gr["open_half_gap_m"])]
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.forcerange = [-0.5 * float(gr["force_limit_N"]), 0.5 * float(gr["force_limit_N"])]
            a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            s.add_exclude(bodyname1=HULL, bodyname2=f"jaw_{side}")
        s.add_exclude(bodyname1="jaw_l", bodyname2="jaw_r")
    meta = dict(version=1, vehicle_body=HULL, vehicle_name=v.name, hydro_set=h.name,
                hydro_coeffs=h.to_dict(), rho_build=rho_build, hydro_bodies=hydro_bodies,
                thrusters=[t.name for t in v.thrusters], thruster_model=v.thruster_model.__dict__,
                battery={k: (list(x) if isinstance(x, tuple) else x) for k, x in v.battery.__dict__.items()},
                hotel_W=v.hotel_W, sources=sources, free_objects={},
                gripper=dict(type="slide_pair", open_half_gap_m=float(gr["open_half_gap_m"])) if gr else None,
                camera_tilt=cam_meta)
    vb = VehicleBuild(spec=s, vehicle=v, hydro=h, meta=meta, rho_build=rho_build)
    vb.write_meta()
    return vb


# --------------------------------------------------------------------------- free objects
SHAPES = ("box", "sphere", "cylinder")


def shape_volume_inertia(shape, size, mass):
    """Volume [m^3] and inertia (3x3, about the centroid) of a solid primitive (MuJoCo half-sizes)."""
    if shape == "sphere":
        r = size[0]
        V = 4.0 / 3.0 * math.pi * r ** 3
        I = np.eye(3) * 0.4 * mass * r * r
    elif shape == "box":
        a, b, c = (2 * x for x in size[:3])
        V = a * b * c
        I = mass / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
    elif shape == "cylinder":                   # axis along local z, size = (radius, half-length)
        r, hl = size[0], size[1]
        V = math.pi * r * r * 2 * hl
        I = np.diag([mass * (3 * r * r + (2 * hl) ** 2) / 12.0] * 2 + [0.5 * mass * r * r])
    else:
        raise ValueError(shape)
    return V, I


CONTACT_PRESETS = {
    # MuJoCo defaults (solref 0.02, solimp 0.9 / 0.95, condim 4 here)
    "default": dict(condim=4),
    # stiff, nearly hard contact with torsional / rolling friction: less creep of an object resting under a steady
    # tangential load (task 07 G6: a pipe leaning on a rail 0.13 -> 0.023 mm/s; a box on the seabed pushed with 0.3 x
    # its friction limit 0.015 -> 0.004 mm/s, uwkit_tests/integration/test_maintenance_030.py)
    "resting": dict(condim=6, solref=(0.004, 1.0), solimp=(0.999, 0.9999, 0.001, 0.5, 2.0)),
}


def add_free_object(vb: VehicleBuild, name, shape="box", size=(0.05, 0.05, 0.05), mass=1.0,
                    volume=None, pos=(1.0, 0.0, -2.0), quat=(1, 0, 0, 0), rgba=(0.8, 0.5, 0.1, 1),
                    friction=(1.0, 0.02, 0.002), cd=1.0, ca=None, weldable=True, cob=(0, 0, 0), contact="default"):
    """Add a free rigid object with the simple hydro model (buoyancy + merged isotropic added mass +
    quadratic drag 0.5 rho Cd A per axis).  volume defaults to the primitive's volume (solid object);
    pass a smaller volume for hollow / flooded objects.  weldable adds an inactive weld to the hull
    that UnderwaterSim.attach() switches on (rigid grasp).

    contact: 'default' (condim 4, MuJoCo's soft default solref / solimp) or 'resting' (condim 6, solref 0.004,
    solimp 0.999 / 0.9999): MuJoCo contacts are soft, so an object at rest under a persistent tangential load creeps
    (a pipe leaning on a rail: 0.13 mm/s default, 0.023 mm/s 'resting'); use 'resting' for objects that must stay put
    for a whole episode (and prove it with a static-hold check in the world generator).  A dict with condim / solref /
    solimp is also accepted."""
    s = vb.spec
    V_shape, I = shape_volume_inertia(shape, size, mass)
    V = float(volume if volume is not None else V_shape)
    gtype = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
             "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}[shape]
    if shape == "sphere":
        areas = [math.pi * size[0] ** 2] * 3
        hh = size[0]
        ca = 0.5 if ca is None else ca
    elif shape == "box":
        a, b, c = (2 * x for x in size[:3])
        areas = [b * c, a * c, a * b]
        hh = size[2]
        ca = 0.7 if ca is None else ca
    else:
        r, hl = size[0], size[1]
        areas = [2 * r * 2 * hl, 2 * r * 2 * hl, math.pi * r * r]
        hh = hl
        ca = 0.8 if ca is None else ca
    rho = vb.rho_build
    dq = [0.5 * rho * cd * A for A in areas]
    Lc = (V_shape ** (1.0 / 3.0))
    dq_rot = [0.5 * rho * cd * A * (0.5 * Lc) ** 3 for A in areas]    # crude rotational drag
    rec = simple_hydro_record(V, cob, mass, [0, 0, 0], I, rho, ca=ca, ca_rot=0.0,
                              d_lin=(0,) * 6, d_quad=tuple(dq) + tuple(dq_rot), half_height=hh)
    b = s.worldbody.add_body(name=name, pos=list(pos), quat=list(quat))
    b.add_freejoint(name=f"{name}_root")
    cp_ = dict(CONTACT_PRESETS[contact]) if isinstance(contact, str) else dict(contact)
    gkw = dict(condim=int(cp_.pop("condim", 4)))
    if "solref" in cp_:
        gkw["solref"] = list(cp_.pop("solref"))
    if "solimp" in cp_:
        gkw["solimp"] = list(cp_.pop("solimp"))
    if cp_:
        raise ValueError(f"unknown contact attributes {sorted(cp_)}")
    b.add_geom(name=f"{name}_geom", type=gtype, size=list(size) + [0] * (3 - len(size)), rgba=list(rgba),
               friction=list(friction), mass=0.0, **gkw)
    _apply_merged_inertia(b, rec)
    vb.meta["hydro_bodies"][name] = rec.to_dict()
    eq_name = None
    if weldable:
        eq_name = f"grasp_{name}"
        e = s.add_equality(name=eq_name, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                           name1=name, name2=HULL, active=False)
        e.solref = [0.004, 1.0]
        e.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
    vb.meta["free_objects"][name] = dict(shape=shape, size=list(size), mass=mass, volume=V,
                                         wet_weight_N=(mass - rho * V) * G, weld=eq_name,
                                         contact=contact if isinstance(contact, str) else "custom")
    vb.objects.append(name)
    vb.write_meta()
    return rec
