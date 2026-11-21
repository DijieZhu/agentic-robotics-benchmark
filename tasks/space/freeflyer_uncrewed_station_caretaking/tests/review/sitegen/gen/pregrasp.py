"""Pre-grasp reach of a site (the site rule 'Reach'; site_spec ``world.pregrasp``).

Every grasp feature (the blocker's carry strap and handle loop, each pouch's handle) must have at least one pre-grasp
pose that the task firmware's arm-deployed zone check accepts: the bar ``dist`` in front of the palm, dist in
world.pregrasp.distance_m (0.35-0.55 m, sampled at ``DISTANCES``), both cage rolls (palm +z = +-bar axis,
world.pregrasp.palm_orientations 2), reached by a straight fly_to from the module's centreline (the point of the
module axis at the bar's axial position) with the arm deployed: the segment of ``spacekit.control.plan_trapezoid``
(STANDARD) validated by the firmware's ``ZoneMap.check_segment`` with the arm-deployed inflation and the band rule
(``firmware.zonecheck.ZoneRules`` on the station's as-surveyed zones -- the same map and rule as the firmware's
fly_to).  The records (``pregrasp_records``) list every accepted pose.

Per bar, every stand-off and palm-z sense is checked through the firmware's arm map (the whole segment, not a
point); ``robot_pose_for_bar`` is the construction of ``spacekit.primitives.GraspFeature._target``; the palm
transform and its plane offset are read once from a task firmware layer (they are constants of the reference
build).
"""
from __future__ import annotations

import functools
import warnings

import numpy as np

from cabincell.site import spec as SPEC

DISTANCES = (0.35, 0.40, 0.45, 0.50, 0.55)        # [C] samples of world.pregrasp.distance_m
ZSIGNS = (1.0, -1.0)


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@functools.lru_cache(maxsize=1)
def palm_geometry() -> dict:
    """The deployed palm in the body frame (T_body_palm_deployed = (p, R)) and the palm plane offset z_plane, read
    from a task firmware layer built on a robot site (spacekit FirmwareLayer: arm tilt 0, pan 0)."""
    from station import sites as SI
    from station import site_rig as SR
    from cabincell.site import build as BU
    import firmware as FW
    S = SI.example_site("P")
    S["robot"] = dict(present=True, berth=1)
    S["blocker"] = None
    S["small_items"] = []
    P = BU.robot_paths()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, info = SI.build_site(S, robot_xml=P["robot_xml"], dock_xml=P["dock_xml"])
    d = SI.initial_data(m, info)
    comps = SR.station_components(m, d, info)
    tf = FW.stack_for_site(m, d, info, comps, seed=0, soc0=0.9, item_aero=False)
    pb, Rb = tf.layer.T_body_palm_deployed
    return dict(pb=np.asarray(pb, float).copy(), Rb=np.asarray(Rb, float).copy(), z_plane=float(tf.layer.z_plane))


def palm_z_pref(n) -> np.ndarray:
    """spacekit FirmwareLayer.palm_z_pref (the robot's top toward the module's overhead when possible)."""
    return np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])


def robot_pose_for_bar(geom: dict, c, a, n, depth: float, extra: float = 0.0, zsign: float | None = None):
    """(body position, quaternion wxyz, palm +z) that put a bar (centre c, axis a, face normal n toward the robot) at
    palm depth ``depth`` plus ``extra`` out along n; depth = -dist puts the bar dist in front of the palm."""
    from spacekit import primitives as PR
    a = unit(a)
    n = unit(np.asarray(n, float) - (np.asarray(n, float) @ a) * a)
    if zsign is None:
        z = a if (a @ palm_z_pref(n)) >= 0 else -a
    else:
        z = float(zsign) * a
    x = n
    y = np.cross(z, x)
    Rp = np.column_stack([x, y, z])
    pp = np.asarray(c, float) - Rp @ np.array([depth, 0.0, geom["z_plane"]]) + n * extra
    p, q = PR.pose_for_palm((geom["pb"], geom["Rb"]), pp, Rp)
    return np.asarray(p, float), np.asarray(q, float), z


@functools.lru_cache(maxsize=1)
def zone_rules():
    """The task firmware's zone rules on the mission scene's as-surveyed zones (stowed + arm-deployed maps)."""
    from station import data as SD
    from firmware.spec import default_spec
    from firmware.zonecheck import ZoneRules
    return ZoneRules(SD.zones("mission"), default_spec(), {})


@functools.lru_cache(maxsize=1)
def standard_mode():
    from spacekit.control import load_flight_modes
    return load_flight_modes()["standard"]


def centreline_point(feats: dict, module: str, p) -> np.ndarray:
    M = feats["modules"][module]
    o, ax = np.asarray(M["axis_origin"], float), unit(M["axis_dir"])
    return o + ax * float((np.asarray(p, float) - o) @ ax)


def check_pose(start, pos, quat) -> str:
    """The firmware's arm-deployed fly_to validation of the straight move start -> (pos, quat) ('ok' or the code)."""
    from spacekit.control import plan_trapezoid
    seg = plan_trapezoid([(np.asarray(start, float), np.asarray(quat, float)), (np.asarray(pos, float), np.asarray(quat, float))],
                         standard_mode(), False)
    if not seg:
        return "ok"
    R = zone_rules()
    r, _k = R.map.check_segment(seg, band_exit=True, arm=True)
    return r


def feature_poses(geom: dict, feats: dict, module: str, c, a, n) -> list:
    """Every sampled pre-grasp pose of one bar with its check: [dict(pos, quat, palm_dir, dist_m, zsign, check)]."""
    out = []
    for dist in DISTANCES:
        for zs in ZSIGNS:
            p, q, z = robot_pose_for_bar(geom, c, a, n, -float(dist), 0.0, zs)
            start = centreline_point(feats, module, c)
            out.append(dict(pos=[float(x) for x in p], quat=[float(x) for x in q], palm_dir=[float(x) for x in z], dist_m=float(dist),
                            zsign=float(zs), check=check_pose(start, p, q)))
    return out


def records(model, data_, info, feats: dict) -> tuple:
    """(pregrasp_records {feature: [accepted poses]}, counts {feature: dict(accepted, evaluated)}) of a built site in
    its current state (truth bar poses, ``firmware.bars.item_bars``)."""
    from firmware import bars as BR
    from cabincell.site import placement as PL
    geom = palm_geometry()
    recs, counts = {}, {}
    for b in BR.item_bars(model, info):
        c, a, n = b.pose(data_)
        iid = info["blocker"]["inlet"] if b.name.startswith("blocker") else next(
            s["inlet"] for s in info["small_items"] if b.name.startswith(s["id"] + ":"))
        module = PL.inlet(feats, iid)["module"]
        rows = feature_poses(geom, feats, module, c, a, n)
        ok = [r for r in rows if r["check"] == "ok"]
        recs[b.name] = [dict(pos=r["pos"], quat=r["quat"], palm_dir=r["palm_dir"], dist_m=r["dist_m"], zone_ok=True) for r in ok]
        counts[b.name] = dict(accepted=len(ok), evaluated=len(rows),
                              refused={r["check"]: sum(1 for x in rows if x["check"] == r["check"]) for r in rows if r["check"] != "ok"})
    return recs, counts


def check_against_spec(values=None) -> list:
    """Problems of the sampled distances and rolls against world.pregrasp (empty = consistent)."""
    v = values or SPEC.values()
    W = v["world.pregrasp"]
    lo, hi = (float(x) for x in W["distance_m"])
    out = []
    if min(DISTANCES) != lo or max(DISTANCES) != hi or any(not (lo <= d <= hi) for d in DISTANCES):
        out.append(f"distances {DISTANCES} do not span {lo}-{hi}")
    if len(ZSIGNS) != int(W["palm_orientations"]):
        out.append("palm orientations")
    return out
