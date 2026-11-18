"""The shop's bench plan of the final-assembly cell (grader side): how the episode starts before anything moves.

Source (the form): docs/plan/driving/tasks/racecar_full_assembly_tenth_scale_ur10e/tests/arm/assembly_car/bench_plan.py
sha256 4f983682e8d3141f35ef59cf759c08d86ecf15cd0650a5b3b9f748a67a3fa497 (the race-car plan put five loose units on the
bare bench). Here every loose unit starts in a station (`fixtures.py` gives the rest poses, `layout.py` the episode's
draws), so the plan is:
* `READY_Q`: the arm's ready posture at the start of every episode (the cell's default `robot_home_q`): the base
  turned so that the arm points along the cell's -x (away from the positioner), the upper arm up, the forearm level,
  the tool down. `check_ready(cell)` measures it on a built cell: the smallest distance from any robot geom to the
  bench, every station, the positioner (base and carrier), the vehicle, every loose unit and the turn column, at the
  index position the cell is in; the plan requires `READY_CLEAR_M` from each and the robot outside the column;
* `START`: the positioner's start index (`keel_up`: the ballast goes on first), the gripper open, its force register at
  the cell's default, the clock at 0 after the cell's settling;
* `rest_plan(kit, fx, lay)`: per loose unit, the station and the place in it (pocket / nest / rack), the surface it
  rests on (its height in the cell at the layout's station pose), and the lowest point of the unit's KIT collision
  geometry at the layout's pose above that surface: the layout lifts every unit `fixtures.PLAYS["settle_lift_m"]`
  above the support the fixture facts give, and the kit's collision pieces may sit a little lower or higher than the
  fine mesh the facts were measured on (the enclosure's 36-sided tube prism, the thrusters' and guards' feet, the
  ballast block's inner boxes); `drop_mm` is how far the unit falls when the cell settles. The floating check
  (`-dev/float_check.py`) prints this table and checks that every unit then rests (<= 0.05 mm/s).

    python -I -B bench_plan.py --kit <kit dir> --out <json> [--seed N] [--stratum dome0|dome180] [--nominal]
"""
import json
import math
import os
import sys

import numpy as np

FORMAT = "rov-bench-plan/1"
# the arm's ready posture (rad): q1 0 = the arm along the cell's -x, upper arm up, forearm level, tool down
READY_Q = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
READY_CLEAR_M = 0.05
START = {"index": "keel_up", "gripper": "open", "clock_s": 0.0}
# where each kind of unit rests: the station, and the support's name (the station's box the unit sits on)
SUPPORT = {"ballast_block": ("ballast_box", "pocket floor"), "thruster_a": ("thruster_tray", "nest support pads"),
           "thruster_b": ("thruster_tray", "nest support pads"), "electronics_enclosure": ("enclosure_rack", "V blocks"),
           "thruster_guard": (None, "guard shelf")}


def _quat_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _read_stl(path):
    import struct
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n, offset=84)
    return np.unique(rec["v"].astype(np.float64).reshape(-1, 3), axis=0)


def collision_points(kit, body, roles=None):
    """((n, 3) m, [(centre, radius)]): the body's collision geometry in its body frame: mesh vertices and box corners,
    and the spheres apart; roles: only the geoms of these kit roles (e.g. ("seat",): a unit's feet)"""
    pts, spheres = [], []
    for g in kit["bodies"][body]["collision"]:
        if roles is not None and g.get("role", "col") not in roles:
            continue
        R = _quat_R(g.get("quat", [1, 0, 0, 0]))
        p = np.asarray(g.get("pos", [0, 0, 0]), float)
        if g["type"] == "mesh":
            V = _read_stl(os.path.join(kit["dir"], g["file"]))
        elif g["type"] == "box":
            s = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
            V = s * np.asarray(g["size"], float)
        else:
            spheres.append((p, float(g["size"][0])))
            continue
        pts.append(V @ R.T + p)
    return (np.concatenate(pts) if pts else np.zeros((0, 3))), spheres


def lowest_point_z(kit, body, T, roles=None):
    """the lowest z (m) of the body's collision geometry (or of its geoms of `roles`) at the world pose T"""
    P, S = collision_points(kit, body, roles)
    z = []
    if len(P):
        z.append(float((P @ T[:3, :3].T + T[:3, 3])[:, 2].min()))
    for p, r in S:
        z.append(float((T[:3, :3] @ p + T[:3, 3])[2] - r))
    return min(z)


def _support_z(fx, lay, kind, unit):
    """the support surface's height (m, cell frame) under a unit at the layout's station pose"""
    from . import fixtures as FX
    from . import layout as LY
    st = SUPPORT[kind][0]
    if kind == "thruster_guard":
        st = lay["guards"][unit]["rack"]
    Tw = LY.T_world_station(lay["stations"][st])
    if kind == "ballast_block":
        z = fx.st["ballast_box"]["floor_z_mm"] / 1000.0
    elif kind in ("thruster_a", "thruster_b"):
        z = fx.st["thruster_tray"]["top_z_mm"] / 1000.0 + fx.st["thruster_tray"]["rim_above_top_mm"] / 1000.0
    elif kind == "thruster_guard":
        z = fx.st["guard_rack"]["shelf_top_z_mm"] / 1000.0
    else:
        z = None                       # the V blocks: tangent support, reported by the tube axis height instead
    return st, (None if z is None else float(Tw[2, 3] + z)), FX


def rest_plan(kit, fx, lay):
    """the plan (module docstring) for a loaded kit (with "dir"), fixtures.Fixtures and an episode layout"""
    from . import layout as LY
    units = {}
    lift = None
    for u in kit["loose"]:
        kind = kit["bodies"][u]["type"]
        st, zs, FX = _support_z(fx, lay, kind, u)
        lift = float(FX.PLAYS["settle_lift_m"])
        pos, quat = lay["parts"][u]
        T = np.eye(4)
        T[:3, :3] = _quat_R(quat)
        T[:3, 3] = np.asarray(pos, float)
        e = {"type": kind, "station": st, "support": SUPPORT[kind][1], "pose": {"pos": [round(float(v), 7) for v in pos],
                                                                                "quat": [round(float(v), 9) for v in quat]}}
        # a unit with support feet (role 'seat': the thrusters' duct feet, the guards' feet) rests on them; the
        # thruster's nose cone hangs below its support, through the nest's hole
        feet = ("seat",) if any(g.get("role") == "seat" for g in kit["bodies"][u]["collision"]) else None
        low = lowest_point_z(kit, u, T, feet)
        e["rests_on"] = "its feet (role seat)" if feet else "its collision geometry"
        e["lowest_collision_z_m"] = round(low, 7)
        if zs is not None:
            e["support_z_m"] = round(zs, 7)
            e["lowest_collision_above_support_mm"] = round((low - zs) * 1000.0, 4)
        else:
            Tw = LY.T_world_station(lay["stations"][st])
            ax = fx.st["enclosure_rack"]["tube_axis_z_mm"] / 1000.0 + float(Tw[2, 3])
            e["tube_axis_z_m"] = round(ax, 7)
            e["tube_axis_above_rack_axis_mm"] = round((float(T[2, 3]) - ax) * 1000.0, 4)
        units[u] = e
    for p in lay.get("pockets") or []:
        units[p["unit"]]["place"] = p["pocket"]
    for n in lay.get("nests") or []:
        units[n["unit"]].update({"place": n["nest"], "mount_deg": n["mount_deg"], "prop_deg": round(float(n["prop_deg"]), 6)})
    return {"format": FORMAT, "ready_q": list(READY_Q), "ready_clear_m": READY_CLEAR_M, "start": dict(START),
            "settle_lift_m": lift, "units": units, "layout_digest": LY.layout_digest(lay), "seed": lay.get("seed"),
            "stratum": lay.get("stratum"), "nominal": lay.get("nominal"),
            "source": "arm/assembly_rov/bench_plan.py rest_plan() on the kit's collision (variant %s)" % kit.get("variant")}


def check_ready(cell, q=None, clear_m=READY_CLEAR_M):
    """the ready posture on a built cell (the arm's joints set to q, default READY_Q; the cell's state restored):
    {"min_m": {what: smallest distance}, "in_column": bool, "ok": bool}"""
    import mujoco
    m, d = cell.m, cell.d
    rb = cell.robot
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in rb["joints"]]
    keep = d.qpos.copy()
    d.qpos[qadr] = np.asarray(READY_Q if q is None else q, float)
    mujoco.mj_kinematics(m, d)
    tmp = np.zeros(6)
    # the arm's base and its shoulder stand on the bench in every posture: not part of the check against the bench
    fixed = set(m.body(b).id for b in rb["arm_bodies"][:2])
    rgs = [g for g in cell.robot_geoms if m.geom_contype[g] or m.geom_conaffinity[g]]
    groups = {"bench": [m.geom("bench_top").id]}
    for st, gn in cell.names["stations"].items():
        groups["station:" + st] = [m.geom(n).id for n in gn]
    groups["positioner"] = [m.geom(n).id for n in (cell.H.base_geoms or []) + (cell.H.carrier_geoms or [])]
    groups["vehicle"] = [m.geom(n).id for n in cell.names["vehicle"]["collision"]]
    for u in cell.loose:
        groups["unit:" + u] = [m.geom(n).id for n in cell.names["units"][u]["collision"]]
    out = {}
    cutoff = 1.0
    for k, gs in groups.items():
        best = cutoff
        for a in rgs:
            if k == "bench" and int(m.geom_bodyid[a]) in fixed:
                continue
            for b in gs:
                dist = float(mujoco.mj_geomDistance(m, d, int(a), int(b), cutoff, tmp))
                if dist < best:
                    best = dist
        out[k] = round(best, 4)
    inside, near = cell.P.robot_in_column()
    col = min(near.values()) if near else None
    d.qpos[:] = keep
    mujoco.mj_forward(m, d)
    ok = all(v >= clear_m for v in out.values()) and not inside
    return {"q": list(READY_Q if q is None else q), "min_m": out, "in_column": bool(inside),
            "column_min_m": None if col is None else round(col, 4), "required_m": clear_m, "ok": bool(ok)}


def write(plan, path):
    with open(path, "w") as f:
        json.dump(plan, f, indent=1, sort_keys=True)
        f.write("\n")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stratum", default="dome0")
    ap.add_argument("--nominal", action="store_true")
    a = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(here)))
    from arm.assembly_rov import bench_plan as BP
    from arm.assembly_rov import cell as C
    from arm.assembly_rov import fixtures as FX
    from arm.assembly_rov import layout as LY
    kit = C.load_kit(a.kit)
    fx = FX.Fixtures(FX.load_cell_layout(), FX.load_facts())
    lay = LY.episode_layout(a.seed, fx=fx, stratum=a.stratum, nominal=a.nominal)
    BP.write(BP.rest_plan(kit, fx, lay), a.out)


if __name__ == "__main__":
    main()
