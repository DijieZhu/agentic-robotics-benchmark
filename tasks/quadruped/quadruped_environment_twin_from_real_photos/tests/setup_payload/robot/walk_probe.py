#!/usr/bin/env python3
"""Walk probe: drive the frozen 12-DoF quadruped probe robot along routes inside a MuJoCo site twin (CPU only,
deterministic for a fixed binary, model and variant).

API
    load_twin(scene_xml_path) -> mujoco.MjSpec
    run_route(twin_spec_or_path, route, variant=None) -> dict
    make_variants(k) -> list[dict]
    route   = {"id": str, "waypoints": [[x, y], ...] (twin world frame, z up, metres),
               "modes": ["walk" | "high_step" | "crouch", one per segment], "start_yaw": rad (optional),
               "start_z": metres (optional hint for the start surface, default 0.0)}
    variant = {"start_dxy": [dx, dy] (world frame, m), "start_dyaw": rad, "phase": 0..1 (gait phase),
               "seed": int (sensor-noise seed), "floor_friction_scale": f (scales the sliding friction of every
               twin geom)}
CLI
    python walk_probe.py --twin scene.xml --routes routes.json [--variants K] [--procs N] --out results.json
                         [--video out.mp4 [--video-route ID]]

What a run does: the twin's MJCF is used as given (static geoms, free bodies, joints, ...) except that its <option>
is replaced by the standard simulation below; the probe robot is attached with the name prefix PREFIX; the robot is
placed standing at the first waypoint on the surface found there by vertical ray casts; a pure-pursuit follower on
the TRUE base pose drives it along the waypoints, switching the controller mode per segment by arc length. Outcome:
"reached" (base within REACH_RADIUS of the last waypoint before the time limit), "fell" (trunk tilt above 50 deg,
the trunk resting on a surface below it for 0.25 s, or the simulation going unstable), otherwise "stopped" (time
limit, or under 2.5 cm/s of route progress over 6 s).
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one core per run; small linear algebra only
if "--video" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from locomotion import QuadrupedController  # noqa: E402

ROBOT_XML = os.path.join(HERE, "robot", "model", "robot.xml")
PREFIX = "probe_robot_"
SPEED = {"walk": 0.25, "high_step": 0.15, "crouch": 0.12}          # m/s commanded on straight segments
YAW_MAX = {"walk": 1.0, "high_step": 0.5, "crouch": 0.5}           # rad/s (controller limits 1.2 / 0.6 / 0.6)
CTL_MODE = {"walk": "walk", "high_step": "high_step", "crouch": "crouch_walk"}
LOOKAHEAD = 0.30            # m, pure pursuit
CMD_PERIOD = 20             # physics steps (1 ms) between follower commands
MON_PERIOD = 5              # physics steps between fall / contact checks
REACH_RADIUS = 0.15         # m
TILT_FALL_DEG = 50.0
REST_FALL_S = 0.25          # s of the trunk resting on a surface below it -> fell
STALL_WINDOW_S = 6.0
STALL_MIN_RATE = 0.025      # m/s of route progress, averaged over the window
DISPLACED_M = 0.02
START_CLEARANCE = 0.40      # m of free height a start surface needs above it
STANDARD_OPTION = dict(timestep=0.001, integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                       cone=mujoco.mjtCone.mjCONE_ELLIPTIC, solver=mujoco.mjtSolver.mjSOL_NEWTON,
                       iterations=50, ls_iterations=50)
DEFAULT_VARIANT = {"start_dxy": [0.0, 0.0], "start_dyaw": 0.0, "phase": 0.0, "seed": 0, "floor_friction_scale": 1.0}

_ROBOT_MASS: list = []


# ------------------------------------------------------------------ twin + robot model
def load_twin(scene_xml_path: str) -> mujoco.MjSpec:
    """The twin's MJCF as given (parsing only; nothing is changed)."""
    return mujoco.MjSpec.from_file(os.fspath(scene_xml_path))


def _robot_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(ROBOT_XML)
    for k in list(spec.keys):          # the robot's own keyframes are not needed (and would pad the twin's)
        spec.delete(k)
    return spec


def _robot_mass() -> float:
    if not _ROBOT_MASS:
        m = mujoco.MjSpec.from_file(ROBOT_XML).compile()
        _ROBOT_MASS.append(float(m.body_subtreemass[m.body("base").id]))
    return _ROBOT_MASS[0]


def _standardise(spec: mujoco.MjSpec) -> None:
    """Replace the twin's <option> by MuJoCo's defaults plus the standard probe simulation."""
    spec.option = mujoco.MjSpec().option
    for k, v in STANDARD_OPTION.items():
        setattr(spec.option, k, v)
    if spec.compiler.settotalmass > 0:
        # settotalmass would rescale the attached robot too: freeze the twin's own compiled masses instead
        m = spec.compile()
        for i, b in enumerate(spec.bodies):        # spec order = compiled body order (world first)
            if i > 0 and m.body_mass[i] > 0:
                b.explicitinertial = True
                b.mass, b.ipos, b.iquat = float(m.body_mass[i]), m.body_ipos[i].copy(), m.body_iquat[i].copy()
                b.inertia = m.body_inertia[i].copy()
        spec.compiler.settotalmass = -1


def build_models(twin) -> tuple:
    """-> (model with the probe robot attached, ray model of the twin alone with colliding geoms in group 0,
    the robot's name prefix)."""
    spec = twin.copy() if isinstance(twin, mujoco.MjSpec) else load_twin(twin)
    _standardise(spec)
    ray_model = spec.compile()
    for prefix in [PREFIX] + [f"{PREFIX}{i}_" for i in range(1, 10)]:
        combined = spec.copy()
        try:
            combined.attach(_robot_spec(), prefix=prefix, frame=combined.worldbody.add_frame())
            model = combined.compile()
            break
        except ValueError as e:              # a twin element already uses this prefix: try the next one
            if "repeated name" not in str(e):
                raise
    else:
        raise ValueError("no free name prefix for the probe robot")
    base = model.body(prefix + "base").id
    if abs(float(model.body_subtreemass[base]) - _robot_mass()) > 1e-9:
        raise ValueError("the twin's compiler settings change the probe robot's mass")
    # ray model: only geoms the robot can collide with (robot contype 1, conaffinity 0), planes handled analytically
    hit = (ray_model.geom_conaffinity & 1).astype(bool)
    ray_model.geom_group[:] = np.where(hit & (ray_model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE), 0, 5)
    return model, ray_model, prefix


# ------------------------------------------------------------------ start surface by vertical ray casts
_MASK = np.array([1, 1, 1, 1, 1, 0], np.uint8)


def _solid_intervals(rm, rd, x: float, y: float) -> list:
    """Solid [bottom, top] intervals of the twin's colliding geoms along the vertical line through (x, y)."""
    out = []
    for g in np.flatnonzero((rm.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) & (rm.geom_conaffinity & 1)):
        n, p = rd.geom_xmat[g].reshape(3, 3)[:, 2], rd.geom_xpos[g]
        if abs(n[2]) > 1e-6:                      # a plane is a half-space below (or above) its surface
            z = p[2] - (n[0] * (x - p[0]) + n[1] * (y - p[1])) / n[2]
            out.append((-math.inf, z) if n[2] > 0 else (z, math.inf))
    sel = rm.geom_group == 0
    if not sel.any():
        return sorted(out)
    reach = rd.geom_xpos[sel, 2], rm.geom_rbound[sel]
    z_hi, z_lo = float(np.max(reach[0] + reach[1])) + 1.0, float(np.min(reach[0] - reach[1])) - 1.0
    gid = np.zeros(1, np.int32)
    saved = rm.geom_group.copy()
    found = {}
    for z0, vz in ((z_hi, -1.0), (z_lo, 1.0)):
        while True:
            dist = mujoco.mj_ray(rm, rd, np.array([x, y, z0]), np.array([0.0, 0.0, vz]), _MASK, 1, -1, gid)
            if dist < 0 or gid[0] < 0:
                break
            g = int(gid[0])
            found.setdefault(g, [-math.inf, math.inf])[0 if vz > 0 else 1] = z0 + vz * dist
            rm.geom_group[g] = 5
        rm.geom_group[:] = saved
    out += [tuple(v) for _, v in sorted(found.items())]
    return sorted(out)


def _surfaces(intervals: list, clearance: float) -> list:
    """Tops of the merged solids with at least `clearance` of free height above them."""
    merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    tops = []
    for i, (_, hi) in enumerate(merged):
        nxt = merged[i + 1][0] if i + 1 < len(merged) else math.inf
        if math.isfinite(hi) and nxt - hi >= clearance:
            tops.append(hi)
    return tops


class StartError(ValueError):
    """The start pose is unusable: reason 'no_start_surface' or 'start_blocked'."""


def start_ground_z(rm, x: float, y: float, yaw: float, z_hint: float = 0.0) -> float:
    """Height of the surface the robot stands on at (x, y): at the base centre the walkable surface (START_CLEARANCE
    of free height above it) closest to z_hint; under the four feet the highest surface within -5 / +10 cm of it.
    Raises StartError if there is no such surface or if geometry occupies the standing robot's envelope (10 to
    33 cm above the stand surface at the centre or a foot)."""
    rd = mujoco.MjData(rm)
    mujoco.mj_kinematics(rm, rd)
    centre = _solid_intervals(rm, rd, x, y)
    tops = _surfaces(centre, START_CLEARANCE)
    if not tops:
        raise StartError("no_start_surface")
    z0 = min(tops, key=lambda z: (abs(z - z_hint), z))
    c, s = math.cos(yaw), math.sin(yaw)
    lines = [centre] + [_solid_intervals(rm, rd, x + c * fx - s * fy, y + s * fx + c * fy)
                        for fx, fy in ((0.19, 0.15), (0.19, -0.15), (-0.19, 0.15), (-0.19, -0.15))]
    z = max([z0] + [t for iv in lines[1:] for t in _surfaces(iv, 0.0) if z0 - 0.05 <= t <= z0 + 0.10])
    if any(lo < z + 0.33 and hi > z + 0.10 for iv in lines for lo, hi in iv):
        raise StartError("start_blocked")
    return z


def surface_below(rm, rd, x: float, y: float, z: float) -> float:
    """Height of the first colliding twin surface below (x, y, z) (static geometry of the ray model)."""
    zs = [hi for lo, hi in _solid_intervals(rm, rd, x, y) if hi <= z + 1e-9]
    return max(zs) if zs else -math.inf


# ------------------------------------------------------------------ route geometry
class Path:
    def __init__(self, waypoints):
        self.p = np.asarray(waypoints, float).reshape(-1, 2)
        seg = np.diff(self.p, axis=0)
        self.len = np.linalg.norm(seg, axis=1)
        if len(self.p) < 2 or not np.all(self.len > 1e-6) or not np.all(np.isfinite(self.p)):
            raise ValueError("a route needs >= 2 distinct, finite waypoints")
        self.dir = seg / self.len[:, None]
        self.s0 = np.concatenate([[0.0], np.cumsum(self.len)])
        self.total = float(self.s0[-1])

    def project(self, xy, s_prev: float) -> float:
        """Arc length of the point of the path nearest to xy, searched near s_prev (no jumps across the route)."""
        best = (math.inf, s_prev)
        for i in range(len(self.len)):
            if self.s0[i + 1] < s_prev - 0.5 or self.s0[i] > s_prev + 1.0:
                continue
            t = float(np.clip((xy - self.p[i]) @ self.dir[i], 0.0, self.len[i]))
            dist = float(np.linalg.norm(self.p[i] + t * self.dir[i] - xy))
            if dist < best[0] - 1e-12:
                best = (dist, float(self.s0[i] + t))
        return best[1]

    def point(self, s: float):
        s = min(max(s, 0.0), self.total)
        i = int(min(np.searchsorted(self.s0, s, side="right") - 1, len(self.len) - 1))
        return self.p[i] + (s - self.s0[i]) * self.dir[i]

    def segment(self, s: float) -> int:
        return int(min(max(np.searchsorted(self.s0, s, side="right") - 1, 0), len(self.len) - 1))


# ------------------------------------------------------------------ one run
def _yaw(R) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def _name(m, g: int) -> str:
    if m.geom(g).name:
        return m.geom(g).name
    b = int(m.geom_bodyid[g])
    return m.body(b).name if b > 0 and m.body(b).name else f"geom{g}"


def run_route(twin, route: dict, variant: dict | None = None, recorder=None) -> dict:
    """Walk the probe robot along one route in the twin. recorder: optional callable(model, data) every 40 ms."""
    t_wall = time.perf_counter()
    v = dict(DEFAULT_VARIANT, **(variant or {}))
    path = Path(route["waypoints"])
    modes = list(route["modes"])
    if len(modes) != len(path.len) or any(md not in SPEED for md in modes):
        raise ValueError(f"route {route.get('id')}: one mode per segment from {sorted(SPEED)}")
    model, rm, prefix = build_models(twin)
    data = mujoco.MjData(model)
    robot = np.zeros(model.nbody, bool)
    base = model.body(prefix + "base").id
    for b in range(1, model.nbody):
        robot[b] = b == base or robot[model.body_parentid[b]]
    g_robot = robot[model.geom_bodyid]
    leg_names = tuple(prefix + f"{l}_{p}" for l in ("FL", "FR", "HL", "HR") for p in ("hip", "thigh", "shank"))
    legs = np.array([robot[b] and model.body(b).name in leg_names for b in range(model.nbody)])
    g_trunk = g_robot & ~legs[model.geom_bodyid] & (model.geom_contype > 0)        # body, head and lidar colliders
    model.geom_friction[~g_robot, 0] *= float(v["floor_friction_scale"])
    free = [b for b in range(1, model.nbody) if not robot[b] and model.body_jntnum[b] > 0
            and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE]

    # placement
    seg0 = path.dir[0]
    yaw0 = float(route["start_yaw"]) if route.get("start_yaw") is not None else math.atan2(seg0[1], seg0[0])
    yaw0 += float(v["start_dyaw"])
    x0, y0 = path.p[0] + np.asarray(v["start_dxy"], float)
    try:
        z_ground = start_ground_z(rm, float(x0), float(y0), yaw0, float(route.get("start_z", 0.0)))
    except StartError as e:              # the twin has no floor there (fell) or something stands there (stopped)
        return {"outcome": "fell" if str(e) == "no_start_surface" else "stopped", "reason": str(e),
                "progress": 0.0, "sim_time_s": 0.0, "end_xy": [float(x0), float(y0)], "min_base_height": 0.0,
                "max_tilt_deg": 0.0, "contacts_with": [], "displaced": [], "state_hash": "",
                "wall_time_s": round(time.perf_counter() - t_wall, 3)}
    ctl = QuadrupedController(model, data, prefix=prefix, seed=int(v["seed"]))
    ctl.reset(pos=(float(x0), float(y0)), yaw=yaw0, phase=float(v["phase"]) % 1.0, ground_z=z_ground)
    rd = mujoco.MjData(rm)
    mujoco.mj_kinematics(rm, rd)
    free0 = {b: data.xpos[b].copy() for b in free}
    t0 = float(data.time)
    t_limit = 2.0 * float(sum(path.len[i] / SPEED[modes[i]] for i in range(len(modes)))) + 5.0

    s_cur = s_max = 0.0
    hist = []                      # (t, s_max) per command tick
    j = 0                          # first entry of hist inside the stall window
    touched = set()
    rest_t, max_tilt, min_h = 0.0, 0.0, math.inf
    outcome, reason = None, ""
    goal = path.p[-1]
    n_steps = int(round(t_limit / model.opt.timestep))
    for k in range(n_steps):
        if k % CMD_PERIOD == 0:
            xy = data.xpos[base, :2].copy()
            yaw = _yaw(data.xmat[base].reshape(3, 3))
            s_cur = path.project(xy, s_cur)
            s_max = max(s_max, s_cur)
            t = float(data.time) - t0
            hist.append((t, s_max))
            if np.linalg.norm(xy - goal) <= REACH_RADIUS and s_max >= path.total - 0.5:
                outcome, reason = "reached", "goal"
                break
            while hist[j][0] < t - STALL_WINDOW_S - 1e-9:
                j += 1
            if t >= STALL_WINDOW_S and s_max - hist[j][1] < STALL_MIN_RATE * STALL_WINDOW_S:
                outcome, reason = "stopped", "stalled"
                break
            mode = modes[path.segment(s_cur)]
            tgt = path.point(s_cur + LOOKAHEAD)
            alpha = (math.atan2(tgt[1] - xy[1], tgt[0] - xy[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
            vx = SPEED[mode] * max(0.0, math.cos(alpha))
            wz = float(np.clip(2.0 * SPEED[mode] * math.sin(alpha) / LOOKAHEAD, -YAW_MAX[mode], YAW_MAX[mode]))
            ctl.set_command(vx, 0.0, wz, mode=CTL_MODE[mode])
            if k % (2 * CMD_PERIOD) == 0:
                if recorder is not None:
                    recorder(model, data)
                h = data.xpos[base, 2] - surface_below(rm, rd, xy[0], xy[1], data.xpos[base, 2])
                min_h = min(min_h, float(h))
        ctl.step()
        if k % MON_PERIOD == 0:
            if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0 or not np.isfinite(data.qpos).all():
                outcome, reason = "fell", "unstable"
                break
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[base, 8])))))
            max_tilt = max(max_tilt, tilt)
            resting = False
            if data.ncon:
                gg = data.contact.geom
                a, b = g_trunk[gg[:, 0]], g_trunk[gg[:, 1]]
                hit = (a & ~g_robot[gg[:, 1]]) | (b & ~g_robot[gg[:, 0]])
                for i in np.flatnonzero(hit):
                    other = int(gg[i, 1] if a[i] else gg[i, 0])
                    touched.add(_name(model, other))
                    nz = float(data.contact.frame[i, 2]) * (1.0 if b[i] else -1.0)   # normal into the trunk
                    resting |= nz > 0.7
            rest_t = rest_t + MON_PERIOD * model.opt.timestep if resting else 0.0
            if tilt > TILT_FALL_DEG:
                outcome, reason = "fell", "tilt"
                break
            if rest_t >= REST_FALL_S:
                outcome, reason = "fell", "trunk_resting"
                break
    if outcome is None:
        outcome, reason = "stopped", "time_limit"
    xy = data.xpos[base, :2]
    s_max = max(s_max, path.project(xy, s_cur))
    state = np.concatenate([[data.time], data.qpos, data.qvel]).astype(np.float64)
    return {
        "outcome": outcome,
        "reason": reason,
        "progress": 1.0 if outcome == "reached" else float(min(1.0, max(0.0, s_max / path.total))),
        "sim_time_s": round(float(data.time) - t0, 3),
        "end_xy": [float(xy[0]), float(xy[1])],
        "min_base_height": float(min_h),
        "max_tilt_deg": float(max_tilt),
        "contacts_with": sorted(touched),
        "displaced": sorted(_name_body(model, b) for b in free if np.linalg.norm(data.xpos[b] - free0[b]) > DISPLACED_M),
        "state_hash": hashlib.sha256(state.tobytes()).hexdigest()[:16],
        "wall_time_s": round(time.perf_counter() - t_wall, 3),
    }


def _name_body(m, b: int) -> str:
    return m.body(b).name or f"body{b}"


# ------------------------------------------------------------------ variants, batch runs, CLI
def make_variants(k: int, seed: int = 0) -> list:
    """k robustness variants: number 0 is the nominal start; the others draw start shifts of +-5 cm per axis,
    +-3 deg of heading, a gait phase, a sensor-noise seed and a friction scale in [0.92, 1.08]."""
    out = [dict(DEFAULT_VARIANT)]
    rng = np.random.default_rng(seed)
    for i in range(1, k):
        dx, dy = rng.uniform(-0.05, 0.05, 2)
        out.append({"start_dxy": [round(float(dx), 4), round(float(dy), 4)],
                    "start_dyaw": round(float(rng.uniform(-math.radians(3), math.radians(3))), 5),
                    "phase": round(float(rng.uniform(0, 1)), 4), "seed": i,
                    "floor_friction_scale": round(float(rng.uniform(0.92, 1.08)), 4)})
    return out[:k]


def instrument_hash() -> str:
    """sha256 over the frozen instrument files (robot model + meshes, controller, this harness)."""
    files = [os.path.join(HERE, "walk_probe.py")]
    for sub in ("locomotion", os.path.join("robot", "model"), os.path.join("robot", "model", "meshes")):
        d = os.path.join(HERE, sub)
        files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith((".py", ".xml", ".stl"))]
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.relpath(f, HERE).encode())
        with open(f, "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def _job(args):
    twin, route, variant = args
    return run_route(twin, route, variant)


def run_batch(twin: str, routes: list, variants: list, procs: int = 1) -> list:
    jobs = [(twin, r, v) for r in routes for v in variants]
    if procs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(procs) as pool:
            res = pool.map(_job, jobs, chunksize=1)
    else:
        res = [_job(j) for j in jobs]
    return [dict(route=j[1]["id"], variant_index=i % len(variants), variant=j[2], **r)
            for i, (j, r) in enumerate(zip(jobs, res))]


def _video(twin: str, route: dict, path: str, fps: int = 25) -> None:
    import subprocess
    frames = []
    renderer = {}
    cam = mujoco.MjvCamera()

    def rec(model, data):
        if "r" not in renderer:
            renderer["r"] = mujoco.Renderer(model, 480, 640)
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = max(i for i in range(model.nbody)
                                  if model.body(i).name.startswith(PREFIX) and model.body(i).name.endswith("base"))
            cam.distance, cam.elevation, cam.azimuth = 1.6, -25.0, 135.0
        renderer["r"].update_scene(data, camera=cam)
        frames.append(renderer["r"].render().copy())

    res = run_route(twin, route, None, recorder=rec)
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", path]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
    print(f"video {path}: {len(frames)} frames, outcome {res['outcome']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twin", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video")
    ap.add_argument("--video-route")
    a = ap.parse_args(argv)
    with open(a.routes) as fh:
        routes = json.load(fh)
    routes = routes["routes"] if isinstance(routes, dict) else routes
    variants = make_variants(a.variants)
    runs = run_batch(os.path.abspath(a.twin), routes, variants, a.procs)
    summary = {}
    for r in runs:
        s = summary.setdefault(r["route"], {"reached": 0, "stopped": 0, "fell": 0, "progress": []})
        s[r["outcome"]] += 1
        s["progress"].append(r["progress"])
    for s in summary.values():
        s["mean_progress"] = round(float(np.mean(s.pop("progress"))), 4)
    out = {"twin": os.path.abspath(a.twin), "mujoco": mujoco.__version__, "instrument_sha256": instrument_hash(),
           "variants": variants, "summary": summary, "runs": runs}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for rid, s in summary.items():
        print(f"{rid}: reached {s['reached']} stopped {s['stopped']} fell {s['fell']} mean progress {s['mean_progress']}")
    if a.video:
        route = next(r for r in routes if a.video_route in (None, r["id"]))
        try:
            _video(os.path.abspath(a.twin), route, a.video)
        except Exception as e:  # rendering needs an OpenGL context; the probe itself does not
            print(f"video skipped: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
