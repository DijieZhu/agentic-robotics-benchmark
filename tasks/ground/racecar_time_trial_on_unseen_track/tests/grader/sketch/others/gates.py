"""Gates of the obstacle model: procedures and lines, fixed in this text before the first run.

    python -B -m sketch.others.gates {beams,wall90,compare,determinism,all} [--out DIR] [--quick]

The lines below were written before any gate was run (the sha256 of this text was recorded at that
moment); a line is never moved after seeing a result.

G1  beams at 3 m
    Scene: a flat floor (base floor kind of the example venue), nothing else; the ego car (the drive model's car, default tyre model) at
    rest at the origin facing +x, settled 0.5 s; the sensor model's SensorSuite with its frozen parameters (noise on).
    A target is put so that the horizontal distance from the ego's scan origin (site car/lidar) to the target's
    vertical centre axis is 3.00 m, on the bearings -90, -45, 0, +45, +90 deg from the ego's heading, and turned to
    the 24 aspects 0, 15, ..., 345 deg about its own vertical axis (aspect 0: the target's +x axis points away from the
    ego). For each configuration 2 ticks are run and the second is counted: beams with `returned` whose hit geom is
    the target's LiDAR geom.
    Targets (a) boxes: the 8 corner sizes of the placement rule's size ranges (others_params.json box.size_range_m)
    and the three boxes of the example world; (b) the tail box of a slow car (the dynamic model, settled 0.5 s).
    Line: the smallest count over all configurations is >= 8, for (a) and for (b) separately.
    Reported only: counts on the whole slow car; counts against distance 1..8 m at aspects 0, 90, 180 deg.

G2  90 s without touching a wall
    Venues: the example world and the generated test venues of testbeds.py (TESTBED_SEEDS, at least 8 layouts), each
    at the three grip tiers (base grip at the tier's midpoint, its floor patches as generated), with the avoidance set
    of placement.place_avoidance (3 boxes, 2 slow cars).
    Run: SlowCars.enter() at t = 0, then 3600 ticks of 10 physics steps in the cell's order (ego car pre_step, slow
    cars pre_step, mj_step, mj_kinematics refresh); the ego car parked outside the bounds (it cannot interact).
    Both modes (dynamic, mocap).
    Line: in no physics step is there a contact (dist <= 0) between a geom of a slow car and a duct `col` geom or a
    box `col` geom, for every venue, tier, slow car and mode.
    Sub-line (same run): every slow car covers at least 85 % of the distance of its planned speed (speed_factor x
    reference speed, starting from rest) over the 90 s, and no slow car is reset by the staff rule.
    Reported only: smallest clearance to walls and boxes, largest lateral error from the path, resets, distance.

G3  dynamic against replayed mocap; the choice
    Determinism: a 90 s run on the example world and two test venues with an ego driven by a privileged path
    follower that overtakes; sha256 over (qpos, qvel, mocap_pos, mocap_quat) after every tick: twice in one process,
    once in a fresh process (must be identical), and on a LAN machine when one is idle (reported).
    Cost: thread CPU time per physics step of the avoid model (venue + ego + 3 boxes + 2 slow cars) with the slow cars
    staged (static phase) and on track (dynamic phase), slow-car pre_step included, sensors excluded; per mode.
    Contact: C1 rear-end on a straight at closing speeds 1, 2, 3 m/s; C2 side swipe at 10 deg and 1.5 m/s closing;
    C3 the ego standing on the slow car's path; C4 the ego cutting in 0.5 m ahead and braking. Measured: largest
    penetration, ego speed change, slow car speed, yaw and lateral response, momentum ratio, slow-car wall and box
    contacts in the 3 s after, solver warnings.
    Choice rule: choose `dynamic` unless (i) it fails same-machine determinism, (ii) it fails G2 where mocap passes,
    (iii) two dynamic slow cars cost more than two mocap slow cars by over 0.05 s CPU per simulated second (14 % of
    the scoring module's 0.35 s line), or (iv) a contact scenario gives a non-physical dynamic result (NaN or a solver warning, a body
    with |vz| > 1 m/s or lifted > 5 cm, or a penetration above 30 mm). If one of (i)-(iv) holds and mocap has none of
    them, choose `mocap`; if both fail, the package is blocked. That mocap needs a contract change (a mocap chassis
    has no `root` joint, the contract and check_model) is recorded as a cost, not as a veto.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import mujoco
import numpy as np

from sketch.car import CarRuntime, add_car
from sketch.examples import check
from sketch.examples import constants as K

from .params import load_params
from .path import REF_X, Path, car_extent, reference_speed
from .placement import with_avoidance
from .sim import AvoidSim, PathDriver
from .slowcars import add_slow_cars
from .boxes import add_boxes
from .testbeds import TESTBED_SEEDS, make_stadium, make_testbed

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(os.path.dirname(HERE))
MIN_WORLD = os.path.join(DEV, "sketch", "examples", "min_world.json")
SCRATCH = "gates_out"
TIERS = ("low", "mid", "high")
MODES = ("dynamic", "mocap")


def _min_world():
    with open(MIN_WORLD, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def gate_layouts():
    """The G2 layouts: TESTBED_SEEDS, classic for odd seeds and featured for even ones."""
    return [(s, "classic" if s % 2 else "featured") for s in TESTBED_SEEDS]


# ============================================================================================== G1 beams at 3 m
def _g1_world():
    w = copy.deepcopy(_min_world())
    t = w["truth"]
    t["floor"]["patches"] = []
    t["ducts"], t["walls"], t["gaps"], t["open_walls"], t["clutter"] = [], [], [], [], []
    return w


def _g1_model(world, boxes, slow_pose):
    spec = mujoco.MjSpec.from_string(
        '<mujoco><option timestep="%s"/><worldbody><geom name="floor" type="plane" size="0 0 0.05" group="%d" '
        'contype="%d" conaffinity="%d"/></worldbody></mujoco>' % (K.PHYSICS_DT_S, K.GROUP_FLOOR, *K.CONTACT["floor"]))
    add_car(spec, prefix=K.EGO_PREFIX, pose=(0.0, 0.0, 0.0))
    w = copy.deepcopy(world)
    w["truth"]["avoidance"] = {"boxes": boxes, "slow_cars": [{"id": "S0", "staging_pose": list(slow_pose)}]}
    add_boxes(spec, w)
    add_slow_cars(spec, w, mode="dynamic")
    m = spec.compile()
    m.opt.timestep = K.PHYSICS_DT_S
    return m, w


def g1_beams(out_dir, quick=False):
    from sketch.sensors import SensorSuite
    p = load_params()
    world = _g1_world()
    rng_sizes = p["box"]["size_range_m"]
    sizes = [[a, b, c] for a in rng_sizes[0] for b in rng_sizes[1] for c in rng_sizes[2]]
    sizes += [bx["size_m"] for bx in _min_world()["truth"]["avoidance"]["boxes"]]
    park = [-40.0, 0.0]
    boxes = [{"id": "T%d" % i, "size_m": s, "mass_kg": 0.45, "pose": [park[0] - 2.0 * i, park[1], 0.0], "s_m": 0.0,
              "free_passage_m": 1.0} for i, s in enumerate(sizes)]
    m, w = _g1_model(world, boxes, [-40.0, 10.0, 0.0])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ego = CarRuntime(m, d, prefix=K.EGO_PREFIX)
    ego.place([0.0, 0.0, 0.0])
    slow = CarRuntime(m, d, prefix="slow0/")
    slow.place([-40.0, 10.0, 0.0])
    suite = SensorSuite(m, d, w, ego, prefix=K.EGO_PREFIX, seed=11)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "car/lidar")

    def step():
        ego.pre_step()
        slow.pre_step()
        mujoco.mj_step(m, d)
        mujoco.mj_kinematics(m, d)

    for _ in range(200):                                      # 0.5 s: the ego car settles
        step()
    for _ in range(4):
        suite.tick(step)
    origin = d.site_xpos[sid].copy()
    bearings = [-90, -45, 0, 45, 90] if not quick else [0, 90]
    aspects = list(range(0, 360, 15)) if not quick else [0, 45, 90, 180]
    box_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, K.box_geom(b["id"], "lidar")) for b in boxes]
    box_jq = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, K.box_body(b["id"]) + "/root")]) for b in boxes]
    tail_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "slow0/tail_box/lidar")
    slow_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "slow0/chassis")
    slow_gids = set(g for g in range(m.ngeom) if int(m.body_rootid[int(m.geom_bodyid[g])]) == slow_root)

    def count(gids):
        last = suite.lidar.last
        hit = np.isin(last["geomid"], list(gids)) & last["returned"]
        return int(hit.sum())

    def scan_twice():
        suite.tick(step)
        suite.tick(step)

    rows_box = []
    for i, b in enumerate(boxes):
        for br in bearings:
            for asp in aspects:
                bx = origin[0] + 3.0 * math.cos(math.radians(br))
                by = origin[1] + 3.0 * math.sin(math.radians(br))
                yaw = math.radians(br) + math.radians(asp)
                qa = box_jq[i]
                d.qpos[qa:qa + 7] = [bx, by, 0.5 * b["size_m"][2], math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]
                mujoco.mj_forward(m, d)
                scan_twice()
                rows_box.append({"box": b["id"], "size_m": b["size_m"], "bearing_deg": br, "aspect_deg": asp,
                                 "beams": count([box_gid[i]])})
                d.qpos[qa:qa + 7] = [park[0] - 2.0 * i, park[1], 0.5 * b["size_m"][2], 1.0, 0.0, 0.0, 0.0]
                mujoco.mj_forward(m, d)
    tb = p["tail_box_centre_m"]
    rows_tail = []

    def put_slow(dist, br, asp):
        # the tail box's vertical axis at `dist` from the scan origin on bearing br; the car's +x at br + asp
        cx = origin[0] + dist * math.cos(math.radians(br))
        cy = origin[1] + dist * math.sin(math.radians(br))
        yaw = math.radians(br) + math.radians(asp)
        slow.place([cx - tb[0] * math.cos(yaw), cy - tb[0] * math.sin(yaw), yaw])
        for _ in range(80):
            step()

    for br in bearings:
        for asp in aspects:
            put_slow(3.0, br, asp)
            scan_twice()
            rows_tail.append({"bearing_deg": br, "aspect_deg": asp, "tail_box_beams": count([tail_gid]),
                              "slow_car_beams": count(slow_gids)})
    by_dist = []
    for dist in ([1, 2, 3, 4, 5, 6, 7, 8] if not quick else [3, 5]):
        for asp in (0, 90, 180):
            put_slow(float(dist), 0, asp)
            scan_twice()
            by_dist.append({"distance_m": dist, "aspect_deg": asp, "tail_box_beams": count([tail_gid]),
                            "slow_car_beams": count(slow_gids)})
    min_box = min(r["beams"] for r in rows_box)
    min_tail = min(r["tail_box_beams"] for r in rows_tail)
    res = {"gate": "G1", "line": ">= 8 beams at 3 m, every configuration", "scan_origin": origin.tolist(),
           "boxes_min": min_box, "tail_box_min": min_tail,
           "boxes_pass": min_box >= 8, "tail_box_pass": min_tail >= 8, "pass": min_box >= 8 and min_tail >= 8,
           "worst_box": min(rows_box, key=lambda r: r["beams"]), "worst_tail": min(rows_tail, key=lambda r: r["tail_box_beams"]),
           "tail_by_aspect_min": {str(a): min(r["tail_box_beams"] for r in rows_tail if r["aspect_deg"] == a) for a in aspects},
           "slow_car_min": min(r["slow_car_beams"] for r in rows_tail), "by_distance": by_dist,
           "n_box_configs": len(rows_box), "n_tail_configs": len(rows_tail), "rows_box": rows_box, "rows_tail": rows_tail}
    _dump(os.path.join(out_dir, "g1_beams.json"), res)
    return res


# ============================================================================================== G2 90 s, no wall
def planned_distance(c, seconds, dt=K.PHYSICS_DT_S):
    """Distance along its path of a car that drives exactly speed_factor x reference speed for `seconds`."""
    s, dist = c.path.project(*c.path.at(c.s)[:2])[0], 0.0
    for _ in range(int(round(seconds / dt))):
        v = c.factor * c.path.speed(s)
        s += v * dt
        dist += v * dt
    return dist


def g2_world(seed, layout, tier):
    if seed is None:
        w = _min_world()
        return with_avoidance(w, np.random.default_rng([20261116, 5]), n_slow=2)
    return make_testbed(seed, tier, layout, n_slow=2)


class Clearance(object):
    """Smallest planar clearance of each slow car's footprint (tail box included) to the wall faces (the corridor
    boundary) and to the boxes' footprints, sampled every `every` ticks (reported only)."""

    def __init__(self, sim, every=4):
        from .path import transform
        from .placement import _poly_dist
        self.sim, self.every = sim, every
        self.tr = check.Track(sim.world)
        x0, x1, hw = car_extent(sim.params["tail_box_centre_m"])
        self.loc = np.array([[x0, -hw], [x0, hw], [x1, -hw], [x1, hw], [0.5 * (x0 + x1), -hw], [0.5 * (x0 + x1), hw]])
        self.transform, self.poly_dist = transform, _poly_dist
        self.wall = [math.inf] * len(sim.slow.cars)
        self.box = [math.inf] * len(sim.slow.cars)

    def __call__(self, sim, k, obs):
        if k % self.every:
            return
        polys = [np.asarray(check.box_corners(p, b["size_m"][0], b["size_m"][1]))
                 for p, b in zip(sim.boxes.poses().tolist(), sim.world["truth"]["avoidance"]["boxes"])]
        for c in sim.slow.cars:
            if not c.on:
                continue
            pts = self.transform(self.loc, c.pose)
            for qx, qy in pts:
                s, dd = self.tr.project(qx, qy)
                _, _, _, wl, wr = self.tr.at(s)
                self.wall[c.i] = min(self.wall[c.i], wl - dd, dd + wr)
            for poly in polys:
                if math.hypot(poly[:, 0].mean() - c.pose[0], poly[:, 1].mean() - c.pose[1]) < 1.5:
                    self.box[c.i] = min(self.box[c.i], self.poly_dist(pts, poly))

    def result(self):
        return {"wall_m": [round(v, 4) for v in self.wall], "box_m": [None if v == math.inf else round(v, 4) for v in self.box]}


def g2_one(args):
    seed, layout, tier, mode = args
    w = g2_world(seed, layout, tier)
    sim = AvoidSim(w, mode=mode)
    sim.park_ego()
    sim.slow.enter()
    plan = [planned_distance(c, 90.0) for c in sim.slow.cars]
    clr = Clearance(sim)
    t0 = time.thread_time()
    sim.run_ticks(3600, on_tick=clr)
    cpu = time.thread_time() - t0
    stats = sim.slow.stats()
    cover = [round(st["distance_m"] / p, 4) for st, p in zip(stats, plan)]
    walls = sum(c["wall_steps"] + c["box_steps"] for c in sim.contacts)
    return {"world_id": w["world_id"], "seed": seed, "layout": layout, "tier": tier, "mode": mode,
            "venue": sim.venue_source, "contacts": sim.contacts, "stats": stats, "planned_m": [round(p, 3) for p in plan],
            "coverage": cover, "resets": sum(st["resets"] for st in stats), "wall_or_box_steps": walls,
            "warnings": sim.new_warnings(), "cpu_s_incl_clearance": round(cpu, 2), "clearance": clr.result(),
            "pass": walls == 0, "sub_pass": all(cv >= 0.85 for cv in cover) and all(st["resets"] == 0 for st in stats)}


def g2_wall90(out_dir, quick=False, workers=6):
    jobs = [(None, "example", "mid", m) for m in MODES]
    lay = gate_layouts()[:2] if quick else gate_layouts()
    tiers = ("low",) if quick else TIERS
    for seed, layout in lay:
        for tier in tiers:
            for m in MODES:
                jobs.append((seed, layout, tier, m))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(g2_one, jobs))
    res = {"gate": "G2", "line": "no slow-car contact with a duct col or box col geom in 90 s; sub-line coverage >= 0.85, no reset",
           "n_runs": len(rows), "pass": all(r["pass"] for r in rows), "sub_pass": all(r["sub_pass"] for r in rows),
           "failures": [r["world_id"] + "/" + r["mode"] for r in rows if not r["pass"]],
           "sub_failures": [r["world_id"] + "/" + r["mode"] for r in rows if not r["sub_pass"]],
           "min_coverage": min(min(r["coverage"]) for r in rows),
           "max_lateral_error_m": {m: max(st["max_lateral_error_m"] for r in rows if r["mode"] == m for st in r["stats"]) for m in MODES},
           "min_wall_clearance_m": {m: min(min(r["clearance"]["wall_m"]) for r in rows if r["mode"] == m) for m in MODES},
           "min_box_clearance_m": {m: min((v for r in rows if r["mode"] == m for v in r["clearance"]["box_m"] if v is not None), default=None) for m in MODES},
           "rows": rows}
    _dump(os.path.join(out_dir, "g2_wall90.json"), res)
    return res


def g2_pool_one(path):
    """G2 on one generated venue file: pushable ducts (the duct model's active_window with its DuctTracker), dynamic
    slow cars, the avoidance set the generator placed; the procedure and the lines of G2 above, unchanged."""
    w = check.load_json(path)
    sim = AvoidSim(w, mode="dynamic", pushable=True)
    sim.park_ego()
    sim.slow.enter()
    plan = [planned_distance(c, 90.0) for c in sim.slow.cars]
    clr = Clearance(sim)
    t0 = time.thread_time()
    sim.run_ticks(3600, on_tick=clr)
    cpu = time.thread_time() - t0
    stats = sim.slow.stats()
    cover = [round(st["distance_m"] / p, 4) for st, p in zip(stats, plan)]
    walls = sum(c["wall_steps"] + c["box_steps"] for c in sim.contacts)
    return {"world_id": w["world_id"], "layout_digest": w["layout_digest"], "mode": "dynamic", "ducts": sim.ducts.mode,
            "venue": sim.venue_source, "contacts": sim.contacts, "stats": stats, "planned_m": [round(p, 3) for p in plan],
            "coverage": cover, "resets": sum(st["resets"] for st in stats), "wall_or_box_steps": walls,
            "max_duct_displacement_m": round(sim.max_duct_disp, 6), "max_box_displacement_m": round(sim.max_box_disp, 6),
            "warnings": sim.new_warnings(), "cpu_s_incl_clearance": round(cpu, 2), "clearance": clr.result(),
            "pass": walls == 0, "sub_pass": all(cv >= 0.85 for cv in cover) and all(st["resets"] == 0 for st in stats)}


def g2_pool(worlds_dir, out_dir, workers=6):
    """G2 on every venue file of a generated pool (s3c_*.json, s3f_*.json; not the sheets, truths or digests)."""
    names = sorted(f for f in os.listdir(worlds_dir) if f.startswith(("s3c_", "s3f_")) and f.endswith(".json")
                   and not f.endswith((".venue_sheet.json", ".truth.json")))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(g2_pool_one, [os.path.join(worlds_dir, f) for f in names]))
    res = {"gate": "G2 on the generated pool (X26)", "worlds_dir": worlds_dir,
           "line": "no slow-car contact with a duct col or box col geom in 90 s; sub-line coverage >= 0.85, no reset",
           "n_runs": len(rows), "pass": all(r["pass"] for r in rows), "sub_pass": all(r["sub_pass"] for r in rows),
           "failures": [r["world_id"] for r in rows if not r["pass"]],
           "sub_failures": [r["world_id"] for r in rows if not r["sub_pass"]],
           "min_coverage": min(min(r["coverage"]) for r in rows),
           "max_lateral_error_m": max(st["max_lateral_error_m"] for r in rows for st in r["stats"]),
           "min_wall_clearance_m": min(min(r["clearance"]["wall_m"]) for r in rows),
           "min_box_clearance_m": min((v for r in rows for v in r["clearance"]["box_m"] if v is not None), default=None),
           "rows": rows}
    _dump(os.path.join(out_dir, "g2_wall90_pool.json"), res)
    return res


# ============================================================================================== G3 determinism
def centre_path(world, factor=1.0):
    """The centreline as a path [x, y, v] with this car's reference speed on it (for the ego driver of G3)."""
    p = load_params()
    tr = check.Track(world)
    n = int(math.ceil(tr.length / 0.2))
    pts = np.array([tr.at(tr.length * i / n)[:2] for i in range(n)])
    fm = check.FloorMap(world)
    g = np.array([fm.grip_at(float(x), float(y)) for x, y in pts])
    from sketch.car import load_params as ldp
    v = reference_speed(pts, g, p, float(ldp()["top_speed_mps"]))
    return [[float(x), float(y), float(vv) * factor] for (x, y), vv in zip(pts, v)]


def g3_worlds():
    return [("example", None, "mid"), ("tb2", 2, "mid"), ("tb3", 3, "low")]


def _g3_world(tag):
    for name, seed, tier in g3_worlds():
        if name == tag:
            return g2_world(seed, "classic" if (seed or 1) % 2 else "featured", tier)
    raise KeyError(tag)


def determinism_run(tag, mode, seconds=90.0, ego_factor=0.75):
    """sha256 over the state after every tick of a run with an ego car that drives the centreline at ego_factor x
    its reference speed while the slow cars drive (they enter at t = 0, placed away from the ego car)."""
    w = _g3_world(tag)
    sim = AvoidSim(w, mode=mode)
    sim.slow.enter()
    drv = PathDriver(centre_path(w), factor=ego_factor)
    h = hashlib.sha256()

    def on_tick(s, k, obs):
        for part in s.state_digest_parts():
            h.update(part)

    sim.run_ticks(int(round(seconds / K.TICK_S)), ego_cmd=drv, on_tick=on_tick)
    return {"tag": tag, "mode": mode, "sha256": h.hexdigest(), "ego_object_steps": sim.ego_object_steps,
            "slow_contacts": sim.contacts, "slow_stats": sim.slow.stats(), "warnings": sim.new_warnings()}


def g3_determinism(out_dir, seconds=90.0):
    rows = []
    for tag, _, _ in g3_worlds():
        for mode in MODES:
            a = determinism_run(tag, mode, seconds)
            b = determinism_run(tag, mode, seconds)
            cmd = [sys.executable, "-B", "-m", "sketch.others.gates", "detrun", "--tag", tag, "--mode", mode,
                   "--seconds", str(seconds)]
            env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_CORETYPE"}     # never pinned
            env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR")
            out = subprocess.run(cmd, cwd=DEV, env=env, capture_output=True, text=True, check=True).stdout
            c = json.loads(out.strip().splitlines()[-1])
            rows.append({"tag": tag, "mode": mode, "sha_in_process": [a["sha256"], b["sha256"]], "sha_fresh_process": c["sha256"],
                         "identical": a["sha256"] == b["sha256"] == c["sha256"], "ego_object_steps": a["ego_object_steps"],
                         "slow_contacts": a["slow_contacts"], "slow_stats": a["slow_stats"], "warnings": a["warnings"]})
    res = {"gate": "G3 determinism", "pass": {m: all(r["identical"] for r in rows if r["mode"] == m) for m in MODES},
           "rows": rows, "host": os.uname().nodename}
    _dump(os.path.join(out_dir, "g3_determinism.json"), res)
    return res


# ============================================================================================== G3 cost
def cost_run(tag, mode, steps=2000):
    """Thread CPU time per physics step of the avoid model: slow cars staged, then on track (no monitor, no sensors)."""
    w = _g3_world(tag)
    out = {"tag": tag, "mode": mode}
    sim = AvoidSim(w, mode=mode, monitor=False)
    drv = PathDriver(centre_path(w), factor=0.6)
    for phase in ("staged", "on_track"):
        if phase == "on_track":
            sim.slow.enter()
        sim.run_ticks(40, ego_cmd=drv)                         # warm up
        n = steps // sim.N
        t0 = time.thread_time()
        sim.run_ticks(n, ego_cmd=drv)
        out[phase + "_us_per_step"] = round(1e6 * (time.thread_time() - t0) / (n * sim.N), 2)
    w0 = copy.deepcopy(w)
    w0["truth"]["avoidance"]["slow_cars"] = []
    nos = AvoidSim(w0, mode=mode, monitor=False)              # reference: the same venue, ego and boxes, no slow car
    d0 = PathDriver(centre_path(w), factor=0.6)
    nos.run_ticks(40, ego_cmd=d0)
    t0 = time.thread_time()
    nos.run_ticks(steps // nos.N, ego_cmd=d0)
    out["no_slow_car_us_per_step"] = round(1e6 * (time.thread_time() - t0) / ((steps // nos.N) * nos.N), 2)
    return out


def g3_cost(out_dir, repeats=3):
    rows = []
    for tag, _, _ in g3_worlds():
        for mode in MODES:
            reps = [cost_run(tag, mode) for _ in range(repeats)]
            rows.append({"tag": tag, "mode": mode,
                         "staged_us_per_step": float(np.median([r["staged_us_per_step"] for r in reps])),
                         "on_track_us_per_step": float(np.median([r["on_track_us_per_step"] for r in reps])),
                         "no_slow_car_us_per_step": float(np.median([r["no_slow_car_us_per_step"] for r in reps])),
                         "repeats": reps})
    diff = {}
    for tag, _, _ in g3_worlds():
        dyn = next(r for r in rows if r["tag"] == tag and r["mode"] == "dynamic")
        moc = next(r for r in rows if r["tag"] == tag and r["mode"] == "mocap")
        diff[tag] = {"on_track_extra_us_per_step": round(dyn["on_track_us_per_step"] - moc["on_track_us_per_step"], 2),
                     "on_track_extra_s_per_sim_s": round((dyn["on_track_us_per_step"] - moc["on_track_us_per_step"]) * 400e-6, 4),
                     "staged_extra_s_per_sim_s": round((dyn["staged_us_per_step"] - moc["staged_us_per_step"]) * 400e-6, 4)}
    res = {"gate": "G3 cost", "line_iii": "dynamic - mocap <= 0.05 s CPU per simulated second (two slow cars)",
           "host": os.uname().nodename, "load": os.getloadavg(), "rows": rows, "extra": diff,
           "worst_extra_s_per_sim_s": max(v["on_track_extra_s_per_sim_s"] for v in diff.values())}
    _dump(os.path.join(out_dir, "g3_cost.json"), res)
    return res


# ============================================================================================== G3 contact
def _lane(samples, offset):
    """A path moved `offset` metres to its left (same speed column)."""
    pa = Path(samples)
    out = []
    for i in range(pa.n):
        x, y, h, v = pa.at(float(pa.s[i]))
        out.append([x - offset * math.sin(h), y + offset * math.cos(h), v])
    return out


def scenario_world(tier, v_ref):
    w = make_stadium(grip_tier=tier)
    tr = check.Track(w)
    n = int(math.ceil(tr.length / 0.2))
    samples = [[*tr.at(tr.length * i / n)[:2], float(v_ref)] for i in range(n)]
    b = w["truth"]["bounds"]
    w["truth"]["avoidance"] = {"boxes": [], "slow_cars": [{"id": "S0", "speed_factor": 0.5, "gap_keep_m": 0.8,
                                                           "entry_s_m": 0.0, "staging_pose": [b["xmin"] - 2.0, b["ymin"] - 2.0, 0.0],
                                                           "path": {"closed": True, "samples": samples}}]}
    w = check.round_floats(w)
    w["layout_digest"] = check.world_digest(w)
    return w


class _Ego(object):
    """The scripted ego driver of a contact scenario: follows a lane at a speed; `phase` switches are done by the
    scenario through the attributes."""

    def __init__(self, lane, speed):
        self.drv = PathDriver(lane, speed=lambda sim, s: self.speed)
        self.speed = speed
        self.steer_override = None

    def __call__(self, sim):
        st, v = self.drv(sim)
        if self.steer_override is not None:
            return self.steer_override, self.speed
        return st, v


def contact_scenario(kind, mode, dv=1.5, tier="mid", horizon_s=3.0):
    """C1 rear_end (closing speed dv), C2 side_swipe (10 deg, 1.5 m/s), C3 standing (ego still in the lane),
    C4 cut_in (ego merges 0.5 m ahead and brakes). Returns the measures of the pre-registered text.
    Contacts are tested after every physics step; the scripted ego driver brakes 50 ms after its first contact;
    speeds are centre-of-mass speeds along the path; the momentum ratio m_s dv_s / (m_e dv_e) is taken over the first
    5 physics steps of the first contact (1 = momentum passed on in full; a mocap car takes none)."""
    v_ref = 5.0 if kind == "standing" else 3.0
    w = scenario_world(tier, v_ref)
    sim = AvoidSim(w, mode=mode)
    c = sim.slow.cars[0]
    path = c.path
    s0 = path.project(-10.0, -3.0)[0]
    sim.park_ego()
    sim.slow._place_on_path(c, s0)
    sim.slow.force_plan = True
    m_e = K.MASS_KG
    m_s = K.MASS_KG + float(sim.params["tail_box"]["mass_kg"])
    lane0 = [[r[0], r[1], r[2]] for r in w["truth"]["avoidance"]["slow_cars"][0]["path"]["samples"]]
    run_up = 2.0 if kind != "standing" else 0.0
    sim.run_ticks(int(run_up / K.TICK_S))
    sc = c.s
    behind = 0.2235 + 0.2055                                   # tail-box rear to ref + ego front to ref (approx.)

    def ego_pose_at(lane_samples, s):
        pa = Path(lane_samples)
        x, y, h, _ = pa.at(s)
        return [x - REF_X * math.cos(h), y - REF_X * math.sin(h), h]

    if kind == "rear_end":
        ego = _Ego(lane0, c.factor * v_ref + dv)
        sim.ego.place(ego_pose_at(lane0, sc - behind - 3.0))
    elif kind == "side_swipe":
        lane = _lane(lane0, 0.30)
        ego = _Ego(lane, c.factor * v_ref + 1.5)
        sim.ego.place(ego_pose_at(lane, sc - behind - 3.0))
    elif kind == "cut_in":
        lane = _lane(lane0, 0.35)
        ego = _Ego(lane, c.factor * v_ref + 1.5)
        sim.ego.place(ego_pose_at(lane, sc - behind - 2.0))
    elif kind == "standing":
        ego = _Ego(lane0, 0.0)
        sim.ego.place(ego_pose_at(lane0, sc + 6.0))
    else:
        raise ValueError(kind)
    sim.slow.force_plan = True
    rest_z = {"ego": float(sim.data.qpos[sim.ego.q_root + 2])}
    if mode == "dynamic":
        rest_z["slow"] = float(sim.data.qpos[c.qa + 2])
    rec = {"first_contact_t": None, "closing_mps": None, "contact_steps": 0, "max_pen_m": 0.0, "min_gap_m": math.inf,
           "max_vz": 0.0, "max_lift_m": 0.0, "nan": False}
    state = {"phase": 0, "t_contact": None, "turn_left": 0, "pre": None, "post": None, "in_episode": False,
             "episode_done": False}
    lat_max = [0.0]
    yaw_max = [0.0]
    t = [0.0]

    ego_body = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, K.EGO_PREFIX + "chassis")

    def speeds():
        """Centre-of-mass speeds of the two cars along the path direction at the slow car."""
        mujoco.mj_subtreeVel(sim.model, sim.data)
        h = path.at(c.s)[2]
        u = (math.cos(h), math.sin(h))
        ve = sim.data.subtree_linvel[ego_body]
        if mode == "mocap":
            vs = c.v
        else:
            v2 = sim.data.subtree_linvel[c.body]
            vs = float(v2[0] * u[0] + v2[1] * u[1])
        return float(ve[0] * u[0] + ve[1] * u[1]), vs

    def ego_contact_depth():
        d = sim.data
        n = d.ncon
        if not n:
            return None
        g = d.contact.geom[:n]
        ca, cb = sim.cat[g[:, 0]], sim.cat[g[:, 1]]
        sel = ((ca == 3) & (cb == 10)) | ((ca == 10) & (cb == 3))
        sel &= d.contact.dist[:n] <= 0.0
        if not sel.any():
            return None
        return float(-d.contact.dist[:n][sel].min())

    max_ticks = int(20.0 / K.TICK_S)
    end_tick = None
    state["last_speeds"] = speeds()
    for k in range(max_ticks):
        # the scenario's triggers, once per tick
        es = sim.ego.state()
        ex, ey, eyaw = es["pose"]
        s_e, lat_e, _ = path.project(ex + REF_X * math.cos(eyaw), ey + REF_X * math.sin(eyaw), s_hint=c.s, back_m=6.0, ahead_m=8.0)
        gap_rel = path.gap(c.s, s_e)
        if gap_rel > 0.5 * path.length:
            gap_rel -= path.length
        if kind == "side_swipe" and state["phase"] == 0 and gap_rel >= -0.45:
            state["phase"] = 1                                 # turn 10 deg toward the slow car's lane and hold it
        if kind == "side_swipe" and state["phase"] == 1 and state.get("brake_tick") is None:
            ph = path.at(s_e)[2] - math.radians(10.0)
            ego.steer_override = min(max(2.0 * math.atan2(math.sin(ph - eyaw), math.cos(ph - eyaw)), -0.4189), 0.4189)
        if kind == "cut_in" and state["phase"] == 0:
            ego_rear_rel = gap_rel - REF_X + K.X_REAR_M        # ego rear relative to the slow car's reference point
            if ego_rear_rel - (K.X_FRONT_M - REF_X) >= 0.5:
                ego.drv = PathDriver(lane0, speed=lambda sim_, s_: ego.speed)   # merge into the slow car's lane
                ego.drv.s = None
                state["phase"] = 1
        if kind == "cut_in" and state["phase"] == 1 and abs(lat_e) < 0.05:
            ego.speed, state["phase"] = 0.0, 2                                   # in the lane: brake to a stop
        if kind in ("standing", "cut_in") and abs(lat_e) < 2.0 * 0.5 * K.ENVELOPE_M[1]:
            front = c.s + (K.X_FRONT_M - REF_X)
            ego_rear = s_e - REF_X + K.X_REAR_M
            g_ = path.gap(front, ego_rear)
            if g_ < 0.5 * path.length:
                rec["min_gap_m"] = min(rec["min_gap_m"], g_)
        if state.get("brake_tick") is not None and k >= state["brake_tick"]:
            ego.speed = 0.0
            if kind == "side_swipe":
                ego.steer_override = 0.0
        sim.ego.set_targets(*ego(sim))
        for i in range(sim.N):
            sim.physics_step()
            t[0] = (k * sim.N + i + 1) * sim.dt
            depth = ego_contact_depth()
            if depth is not None:
                rec["contact_steps"] += 1
                rec["max_pen_m"] = max(rec["max_pen_m"], depth)
                if rec["first_contact_t"] is None:
                    rec["first_contact_t"] = round(t[0], 4)
                    state["pre"] = state["last_speeds"]
                    rec["closing_mps"] = round(state["pre"][0] - state["pre"][1], 4)
                    state["in_episode"] = True
                    state["brake_tick"] = k + 2                # the driver brakes after 50 ms (two ticks)
                    state["contact_step"] = 0
                    end_tick = k + int(horizon_s / K.TICK_S)
            if state["in_episode"] and not state["episode_done"]:
                state["contact_step"] += 1
                if depth is None or state["contact_step"] >= 5:  # momentum over the first 5 steps (12.5 ms)
                    state["post"] = speeds()
                    state["episode_done"] = True
            state["last_speeds"] = speeds()
            for name, qa, va in (("ego", sim.ego.q_root, sim.ego.v_root),) + ((("slow", c.qa, c.va),) if mode == "dynamic" else ()):
                q, v = sim.data.qpos, sim.data.qvel
                rec["max_vz"] = max(rec["max_vz"], abs(float(v[va + 2])))
                rec["max_lift_m"] = max(rec["max_lift_m"], float(q[qa + 2]) - rest_z[name])
        if not np.isfinite(sim.data.qpos).all():
            rec["nan"] = True
            break
        if rec["first_contact_t"] is not None:
            pose = c.pose if mode == "mocap" else sim.slow._pose_of(c)
            rx, ry = pose[0] + REF_X * math.cos(pose[2]), pose[1] + REF_X * math.sin(pose[2])
            ss, lat, _ = path.project(rx, ry)
            lat_max[0] = max(lat_max[0], abs(lat))
            yaw_max[0] = max(yaw_max[0], abs(math.atan2(math.sin(pose[2] - path.at(ss)[2]), math.cos(pose[2] - path.at(ss)[2]))))
        if end_tick is not None and k >= end_tick:
            break
        if kind == "standing" and k * K.TICK_S > 12.0:
            break
        if kind == "cut_in" and state["phase"] == 2 and rec["first_contact_t"] is None and c.speed < 0.02:
            break
    pre, post = state["pre"], state["post"]
    out = {"kind": kind, "mode": mode, "tier": tier, "dv": dv, **{k_: (None if v_ == math.inf else v_) for k_, v_ in rec.items()},
           "slow_lat_max_after_m": round(lat_max[0], 4), "slow_yaw_max_after_rad": round(yaw_max[0], 4),
           "slow_wall_steps": sim.contacts[0]["wall_steps"], "slow_box_steps": sim.contacts[0]["box_steps"],
           "warnings": sim.new_warnings(), "slow_resets": c.resets}
    if pre is not None and post is not None:
        dve, dvs = pre[0] - post[0], post[1] - pre[1]
        out.update({"ego_speed_before": round(pre[0], 4), "ego_speed_after": round(post[0], 4),
                    "slow_speed_before": round(pre[1], 4), "slow_speed_after": round(post[1], 4),
                    "momentum_ratio": None if abs(dve) < 1e-6 else round((m_s * dvs) / (m_e * dve), 4)})
    nonphys = out["nan"] or bool(out["warnings"]) or out["max_vz"] > 1.0 or out["max_lift_m"] > 0.05 or out["max_pen_m"] > 0.03
    out["non_physical"] = bool(nonphys)
    return out


def _contact_job(args):
    return contact_scenario(*args)


def g3_contact(out_dir, workers=6):
    jobs = []
    for mode in MODES:
        for dv in (1.0, 2.0, 3.0):
            jobs.append(("rear_end", mode, dv, "mid"))
        jobs.append(("side_swipe", mode, 1.5, "mid"))
        jobs.append(("standing", mode, 0.0, "mid"))
        jobs.append(("standing", mode, 0.0, "low"))
        jobs.append(("cut_in", mode, 1.5, "mid"))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_contact_job, jobs))
    res = {"gate": "G3 contact", "rows": rows,
           "non_physical": {m: [r["kind"] + "/" + str(r["dv"]) for r in rows if r["mode"] == m and r["non_physical"]] for m in MODES}}
    _dump(os.path.join(out_dir, "g3_contact.json"), res)
    return res


# ============================================================================================== the choice
def choose(det, cost, contact, g2):
    """The pre-registered choice rule (module text)."""
    reasons = {}
    for m in MODES:
        r = []
        if not det["pass"][m]:
            r.append("(i) same-machine determinism")
        g2_fail = [x for x in g2["rows"] if x["mode"] == m and not x["pass"]]
        other = "mocap" if m == "dynamic" else "dynamic"
        g2_other = [x for x in g2["rows"] if x["mode"] == other and not x["pass"]]
        if g2_fail and not g2_other:
            r.append("(ii) G2 fails where %s passes" % other)
        if m == "dynamic" and cost["worst_extra_s_per_sim_s"] > 0.05:
            r.append("(iii) %.4f s CPU per simulated second over mocap" % cost["worst_extra_s_per_sim_s"])
        if contact["non_physical"][m]:
            r.append("(iv) non-physical contact: %s" % contact["non_physical"][m])
        reasons[m] = r
    if not reasons["dynamic"]:
        choice = "dynamic"
    elif not reasons["mocap"]:
        choice = "mocap"
    else:
        choice = None
    return {"choice": choice, "vetoes": reasons, "blocked": choice is None}


def main(argv=None):
    ap = argparse.ArgumentParser(description="S5 gates")
    ap.add_argument("what", choices=["beams", "wall90", "wall90pool", "determinism", "cost", "contact", "compare", "choose", "all",
                                     "detrun"])
    ap.add_argument("--worlds", help="wall90pool: the folder of a generated pool's world files")
    ap.add_argument("--out", default=SCRATCH)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag")
    ap.add_argument("--mode")
    ap.add_argument("--seconds", type=float, default=90.0)
    a = ap.parse_args(argv)
    if a.what == "detrun":
        print(json.dumps(determinism_run(a.tag, a.mode, a.seconds)))
        return 0
    out = {}
    if a.what == "wall90pool":
        r = g2_pool(a.worlds, a.out, a.workers)
        print(json.dumps({k: r[k] for k in ("pass", "sub_pass", "n_runs", "failures", "sub_failures", "min_coverage",
                                            "max_lateral_error_m", "min_wall_clearance_m", "min_box_clearance_m")}, indent=1))
        return 0
    if a.what in ("beams", "all"):
        r = g1_beams(a.out, a.quick)
        out["G1"] = {k: r[k] for k in ("pass", "boxes_min", "tail_box_min", "slow_car_min", "worst_box", "worst_tail")}
    if a.what in ("wall90", "all", "compare"):
        r = g2_wall90(a.out, a.quick, a.workers)
        out["G2"] = {k: r[k] for k in ("pass", "sub_pass", "n_runs", "failures", "sub_failures", "min_coverage",
                                       "max_lateral_error_m", "min_wall_clearance_m", "min_box_clearance_m")}
    if a.what in ("determinism", "all", "compare"):
        r = g3_determinism(a.out, 30.0 if a.quick else 90.0)
        out["G3_determinism"] = r["pass"]
    if a.what in ("cost", "all", "compare"):
        r = g3_cost(a.out, 1 if a.quick else 3)
        out["G3_cost"] = {"extra": r["extra"], "worst": r["worst_extra_s_per_sim_s"]}
    if a.what in ("contact", "all", "compare"):
        r = g3_contact(a.out, a.workers)
        out["G3_contact"] = r["non_physical"]
    if a.what in ("compare", "all", "choose"):
        load = lambda n: json.load(open(os.path.join(a.out, n)))
        ch = choose(load("g3_determinism.json"), load("g3_cost.json"), load("g3_contact.json"), load("g2_wall90.json"))
        _dump(os.path.join(a.out, "g3_choice.json"), ch)
        out["G3_choice"] = ch
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
