"""Walks of the probe robot in a twin: everything between the C6 grader and the frozen walk harness
(robot/walk_probe.py, the instrument).

    WP = instrument(robot_dir)                          # the frozen harness, imported once from its folder
    spec = prepare(twin, sec)                           # units held where submitted, gates fixed, contacts standard
    rm = ray_model(spec, WP)                            # the twin alone, as the harness casts its start rays
    m = map_route(route, simmap, rm, sec, WP)           # site route -> the agent's frame; the start rule
    vs = make_variants(sec)                             # the K agent-side variants (variant 0 = nominal)
    jobs = plan(mapped, vs, sec, WP, rm)                # one job per (route, variant); pre-decided ones included
    out = run({"twin": spec}, jobs, sec, robot_dir, deadline)   # a small fork pool: {key: result}, stats

Only stdlib, numpy and mujoco are imported here (the instrument adds nothing else), so this file can serve the agent's
own checker unchanged. Every quantity that reaches a score is deterministic: fixed variant seeds, results keyed by
(route, variant) whatever process ran them, no wall clock in a result (wall times are reported apart, under "timing").
No outcome comes from the wall clock either: a run has the harness's SIMULATED time limit;
its wall limit (run_wall: a generous multiple of the measured cost) and the batch's budget (the stage deadline) are
guards only, and run() raises WalkTimeError when one is reached (the grader turns that into a task error: the episode
is void and re-run), never "not reached".

What a walk run sees:
- the twin's grader copy (the gate's MjSpec) with the units the gate excluded deleted (as twin_gate.physics_model);
- every unit held: its free joint removed, so it is welded to the world at the pose of its MJCF (qpos0, as
  submitted); every gate fixed: its slide joint removed (the photographed position, qpos0);
- contact parameters of every collision geom replaced by the standard ones of the C6 config (the envelope's: friction
  0.8, MuJoCo's default solref / solimp, no margin or gap) and its collision bits set so that the robot touches every
  collision geom; visual geoms stay inert. Photos determine none of these parameters;
- the harness then replaces <option> by its own standard simulation and attaches the robot.
"""
import gc
import hashlib
import importlib.util
import math
import os
import sys
import time

import numpy as np

FORMAT = "site-walks/1"
NOT_RUN = ("start_rule", "start_window", "error")
_WP = {}


class WalkError(ValueError):
    """The twin cannot be prepared for walking (a structure the gate should have rejected)."""


class WalkTimeError(RuntimeError):
    """A walk run exceeded its wall guard, or the batch's wall budget ran out before every run finished, or a walk
    worker was killed from outside (SIGKILL: the machine's out-of-memory killer): the machine could not run the
    walks, not the twin's outcome. The grader makes it a task error."""


# ---------------------------------------------------------------------------------------------------------------
# the instrument

def instrument(robot_dir):
    """The frozen harness module walk_probe.py of `robot_dir` (imported once per process and folder)."""
    robot_dir = os.path.realpath(robot_dir)
    if robot_dir in _WP:
        return _WP[robot_dir]
    path = os.path.join(robot_dir, "walk_probe.py")
    if not os.path.isfile(path):
        raise FileNotFoundError("no walk_probe.py in %s" % robot_dir)
    loaded = sys.modules.get("walk_probe")
    if loaded is not None and os.path.realpath(os.path.dirname(getattr(loaded, "__file__", "") or "")) != robot_dir:
        raise RuntimeError("another walk_probe is already imported from %s" % loaded.__file__)
    if loaded is None:
        if robot_dir not in sys.path:
            sys.path.insert(0, robot_dir)
        spec = importlib.util.spec_from_file_location("walk_probe", path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules["walk_probe"] = loaded
        spec.loader.exec_module(loaded)
    _WP[robot_dir] = loaded
    return loaded


def instrument_hash(robot_dir):
    return instrument(robot_dir).instrument_hash()


# ---------------------------------------------------------------------------------------------------------------
# the twin for walking

def _gate_parts(twin):
    """(MjSpec, report) of a load.Twin or a twin_gate.GateResult."""
    gate = getattr(twin, "gate", twin)
    spec = getattr(gate, "spec", None)
    report = getattr(gate, "report", None) or {}
    if spec is None:
        raise WalkError("the twin has no compiled grader copy (it failed the gate)")
    return spec, report


def prepare(twin, sec=None, unit_offsets=None):
    """A copy of the twin's grader copy (MjSpec) ready for the harness: excluded units deleted, every unit held at
    its submitted pose (free joint removed), every gate fixed at its photographed position (slide joint removed),
    standard contact parameters on every collision geom (sec["contact"], when given). unit_offsets {unit body:
    (dx, dy, dz)} moves held units first (world frame; the truth's robustness variants only). Nothing else changes."""
    import mujoco
    src, report = _gate_parts(twin)
    sp = src.copy()
    excluded = set(report.get("excluded") or [])
    units = [u["body"] for u in report.get("units", []) if u.get("status") != "missing"]
    gates = [(g["body"], g["joint"]) for g in report.get("gates", [])]
    for nm in sorted(excluded):
        b = sp.body(nm)
        if b is not None:
            sp.delete(b)
    free = int(mujoco.mjtJoint.mjJNT_FREE)
    for nm in units:
        if nm in excluded:
            continue
        b = sp.body(nm)
        if b is None:
            raise WalkError("unit body %s is missing from the grader copy" % nm)
        for j in list(b.joints):
            if int(j.type) != free:
                raise WalkError("unit %s carries a joint that is not its free joint: %s" % (nm, j.name))
            sp.delete(j)
    for nm, (dx, dy, dz) in sorted((unit_offsets or {}).items()):
        b = sp.body(nm)
        if b is None:
            raise WalkError("unit_offsets: no unit body %s" % nm)
        b.pos = np.asarray(b.pos, float) + np.array([dx, dy, dz], float)
    for nm, jn in gates:
        j = sp.joint(jn)
        if j is None:
            raise WalkError("gate %s: no joint %s" % (nm, jn))
        sp.delete(j)
    left = [j.name or "(unnamed)" for j in sp.joints]
    if left:
        raise WalkError("joints outside the units and gates: %s" % ", ".join(left[:5]))
    con = (sec or {}).get("contact")
    if con and con.get("standardise", True):
        for g in sp.geoms:
            if int(g.contype) == 0 and int(g.conaffinity) == 0:
                continue                                      # visual: inert
            g.contype, g.conaffinity = int(con["contype"]), int(con["conaffinity"])
            g.condim = int(con["condim"])
            g.friction = [float(x) for x in con["friction"]]
            g.solref = [float(x) for x in con["solref"]]
            g.solimp = [float(x) for x in con["solimp"]]
            g.margin, g.gap = float(con["margin_m"]), float(con["gap_m"])
            g.priority, g.solmix = int(con["priority"]), float(con["solmix"])
    return sp


def ray_model(spec, WP):
    """The twin alone as the harness sees it (its standard options; colliding geoms in ray group 0)."""
    return WP.build_models(spec)[1]


def spec_digest(spec):
    """sha256 (16 hex) of the compiled twin's geometry and contact arrays: identical twins, identical digests."""
    m = spec.compile()
    h = hashlib.sha256()
    for a in (m.geom_type, m.geom_size, m.geom_pos, m.geom_quat, m.geom_contype, m.geom_conaffinity,
              m.geom_friction, m.geom_solref, m.geom_solimp, m.geom_margin, m.geom_gap, m.body_pos, m.body_quat,
              m.mesh_vert):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------
# routes into the agent's frame and the start rule

def _merged(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def start_support(rm, WP, x, y, z_ref, window):
    """The start rule: a vertical ray along the twin's -z from z_ref + window down to z_ref - window; the support
    is the top of the first solid below the ray's origin. Returns (z, None) or (None, reason): 'inside_solid' when
    the origin lies inside a solid, 'no_support' when nothing is hit within the window."""
    import mujoco
    rd = mujoco.MjData(rm)
    mujoco.mj_kinematics(rm, rd)
    top = z_ref + window
    for lo, hi in reversed(_merged(WP._solid_intervals(rm, rd, float(x), float(y)))):
        if lo < top < hi:
            return None, "inside_solid"
        if hi <= top:
            return (float(hi), None) if hi >= z_ref - window else (None, "no_support")
    return None, "no_support"


def map_route(route, simmap, rm, sec, WP):
    """A kept route (site frame) in the agent's frame: waypoints by the inverse alignment (route lengths scale with
    the map; the robot does not), the start heading turned with it, and the start rule applied at the first
    waypoint. Returns {"id", "harness": route dict for the harness or None, "start": {...}, "z_ref", ...}."""
    z0 = float(route.get("start_z", 0.0))
    W = np.asarray(route["waypoints"], float).reshape(-1, 2)
    P = np.c_[W, np.full(len(W), z0)]
    A = np.array([simmap.to_agent(p) for p in P])
    out = {"id": route["id"], "group": route.get("group"), "z_ref": float(A[0, 2]),
           "waypoints": [[float(a), float(b)] for a, b in A[:, :2]], "modes": list(route["modes"])}
    yaw = None
    if route.get("start_yaw_rad") is not None:
        d = simmap.dir_to_agent(np.array([math.cos(route["start_yaw_rad"]), math.sin(route["start_yaw_rad"]), 0.0]))
        yaw = float(math.atan2(d[1], d[0]))
    out["start_yaw"] = yaw
    win = float(sec["start_ray_m"])
    zs, why = start_support(rm, WP, A[0, 0], A[0, 1], out["z_ref"], win)
    out["start"] = {"support_z": zs, "reason": why, "window_m": win}
    if zs is None:
        out["harness"] = None
        return out
    h = {"id": route["id"], "waypoints": out["waypoints"], "modes": out["modes"], "start_z": zs}
    if yaw is not None:
        h["start_yaw"] = yaw
    out["harness"] = h
    return out


# ---------------------------------------------------------------------------------------------------------------
# variants

def make_variants(sec, k=None, wide=False, seed=None):
    """Seeded start variants in the ROUTE frame (along / left of the start heading, metres of the twin), the same for
    every submission and every route. Number 0 is the nominal start. wide=False: the agent-side set (start
    +-start_dxy_m, yaw +-start_dyaw_deg, any gait phase, sensor seed = index, friction scale 1); wide=True: the
    truth's wider set (sec["truth"]: +-5 cm, +-3 deg, phase, friction 0.92-1.08, unit offsets drawn by the caller)."""
    if wide:
        t = sec["truth"]
        k = int(t["n_wide"]) if k is None else int(k)
        dxy, dyaw, fr = float(t["start_dxy_m"]), float(t["start_dyaw_deg"]), t["friction_scale"]
        rng = np.random.default_rng(int(t["seed"]) if seed is None else int(seed))
        first, base = 0, 1000
    else:
        k = int(sec["variants"]) if k is None else int(k)
        dxy, dyaw, fr = float(sec["start_dxy_m"]), float(sec["start_dyaw_deg"]), None
        rng = np.random.default_rng(int(sec["variant_seed"]) if seed is None else int(seed))
        first, base = 1, 0
    out = [{"along": 0.0, "left": 0.0, "dyaw": 0.0, "phase": 0.0, "seed": 0, "friction": 1.0}] if first else []
    for i in range(first, k):
        a, l_ = rng.uniform(-dxy, dxy, 2)
        out.append({"along": round(float(a), 4), "left": round(float(l_), 4),
                    "dyaw": round(float(rng.uniform(-math.radians(dyaw), math.radians(dyaw))), 5),
                    "phase": round(float(rng.uniform(0.0, 1.0)), 4), "seed": base + i,
                    "friction": round(float(rng.uniform(fr[0], fr[1])), 4) if fr else 1.0})
    return out[:k]


def harness_variant(h, v):
    """The harness's variant dict for mapped route h and route-frame variant v."""
    p = np.asarray(h["waypoints"], float)
    if h.get("start_yaw") is not None:
        yaw = float(h["start_yaw"])
    else:
        d = p[1] - p[0]
        yaw = math.atan2(d[1], d[0])
    c, s = math.cos(yaw), math.sin(yaw)
    dx = c * v["along"] - s * v["left"]
    dy = s * v["along"] + c * v["left"]
    return {"start_dxy": [dx, dy], "start_dyaw": float(v["dyaw"]), "phase": float(v["phase"]),
            "seed": int(v["seed"]), "floor_friction_scale": float(v["friction"])}


def _sim_limit(h, WP):
    """The harness's own simulated-time limit of a route."""
    p = np.asarray(h["waypoints"], float)
    L = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return 2.0 * float(sum(L[i] / WP.SPEED[m] for i, m in enumerate(h["modes"]))) + 5.0


def _start_window_check(rm, WP, h, hv, z_ref, window):
    """The start height the harness will use must lie within the start window; (ok, z or reason)."""
    p0 = np.asarray(h["waypoints"][0], float) + np.asarray(hv["start_dxy"], float)
    d = np.asarray(h["waypoints"][1], float) - np.asarray(h["waypoints"][0], float)
    yaw = (float(h["start_yaw"]) if h.get("start_yaw") is not None else math.atan2(d[1], d[0])) + hv["start_dyaw"]
    try:
        z = WP.start_ground_z(rm, float(p0[0]), float(p0[1]), yaw, float(h.get("start_z", 0.0)))
    except WP.StartError as exc:
        return True, str(exc)                # the harness reports it itself (stopped / fell, progress 0)
    if not (z_ref - window - 1e-9 <= z <= z_ref + window + 1e-9):
        return False, "the harness's start height %.3f lies outside the start window %.3f +- %.2f" % (z, z_ref, window)
    return True, z


def not_run(how, why=""):
    """The result of a run that did not happen: it counts as not reached with progress 0."""
    return {"outcome": "not_reached", "reason": how, "how": how, "why": str(why)[:300], "progress": 0.0,
            "sim_time_s": 0.0, "contacts_with": [], "state_hash": ""}


def plan(mapped, variants, sec, WP, rm, spec_key="twin"):
    """Jobs for every (route, variant) in a fixed order. A route that fails the start rule, or a variant whose start
    the harness would take outside the window, gets a pre-decided result instead of a job. A variant may name its
    own twin ("spec": key of the specs given to run()); rm is a ray model or {spec key: ray model}."""
    jobs, pre = [], {}
    win = float(sec["start_ray_m"])
    rw = sec["run_wall"]
    for ri, m in enumerate(mapped):
        for vi, v in enumerate(variants):
            key = (ri, vi)
            h = m["harness"]
            if h is None:
                pre[key] = not_run("start_rule", "no start support: %s" % m["start"]["reason"])
                continue
            hv = harness_variant(h, v)
            rmv = rm.get(v.get("spec", spec_key)) if isinstance(rm, dict) else rm
            ok, z = _start_window_check(rmv, WP, h, hv, m["z_ref"], win)
            if not ok:
                pre[key] = not_run("start_window", z)
                continue
            tsim = _sim_limit(h, WP)
            jobs.append({"key": key, "spec": v.get("spec", spec_key), "route": h, "variant": hv, "sim_limit_s": tsim,
                         "wall_limit_s": max(float(rw["min_s"]), float(rw["factor"]) * float(rw["per_sim_s"]) * tsim),
                         "predicted_s": float(rw["per_sim_s"]) * 0.5 * tsim})
    jobs.sort(key=lambda j: (j["key"][1], -j["predicted_s"], j["key"][0]))
    return jobs, pre


# ---------------------------------------------------------------------------------------------------------------
# the pool

def cpu_count():
    """CPUs this process may use: the affinity mask, bounded by a cgroup CPU quota when one is set."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    quota = None
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, p = f.read().split()[:2]
            if q != "max":
                quota = float(q) / float(p)
    except (OSError, ValueError):
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                q = float(f.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                p = float(f.read())
            if q > 0 and p > 0:
                quota = q / p
        except (OSError, ValueError):
            pass
    if quota is not None:
        n = min(n, max(1, int(math.floor(quota + 1e-9))))
    return max(1, n)


def pool_size(sec, n_jobs, procs=None):
    """Worker processes: `procs` if given, else $ARB_TWIN_WALK_PROCS (maintainers), else the CPUs this process may
    use minus sec["procs"]["reserve"], capped at sec["procs"]["max"]; never more than the jobs."""
    if procs is None and os.environ.get("ARB_TWIN_WALK_PROCS", "").strip():
        procs = int(os.environ["ARB_TWIN_WALK_PROCS"])
    if procs is None:
        p = sec["procs"]
        procs = min(int(p["max"]), cpu_count() - int(p.get("reserve", 0)))
    return max(1, min(int(procs), max(1, n_jobs)))


def _die_with_parent(parent):
    """A forked worker holds copies of the parent's pipe ends (its own and its siblings'), so the parent's death
    never shows as EOF: ask the kernel for SIGKILL when the parent dies (Linux), and check the parent while idle."""
    import signal
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGKILL))       # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass
    if os.getppid() != parent:
        os._exit(0)


def _worker_main(conn, specs, robot_dir, parent=None):
    import signal
    import warnings
    _die_with_parent(parent if parent is not None else os.getppid())
    parent = os.getppid()
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    warnings.filterwarnings("ignore", message="Attaching a child without asset dict")
    WP = instrument(robot_dir)
    while True:
        try:
            while not conn.poll(2.0):
                if os.getppid() != parent:
                    os._exit(0)                        # orphaned: the grader is gone
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        key, spec_key, route, variant = msg
        t = time.time()
        try:
            res = WP.run_route(specs[spec_key], route, variant)
        except Exception as exc:  # noqa: BLE001 - the twin breaks the harness: this run is not reached
            res = dict(not_run("error", "%s: %s" % (type(exc).__name__, str(exc)[:300])))
        res["worker_wall_s"] = time.time() - t
        try:
            conn.send((key, res))
        except (BrokenPipeError, OSError):
            break


class _Slot:
    def __init__(self, mpctx, specs, robot_dir):
        a, b = mpctx.Pipe(duplex=True)
        self.proc = mpctx.Process(target=_worker_main, args=(b, specs, robot_dir, os.getpid()), daemon=True)
        self.proc.start()
        b.close()
        self.conn = a
        self.job = None
        self.t0 = 0.0

    def kill(self):
        try:
            self.proc.kill()
        except (OSError, AttributeError):
            pass
        self.proc.join(5)
        try:
            self.conn.close()
        except OSError:
            pass


def run(specs, jobs, sec, robot_dir, deadline=None, procs=None, log=None):
    """Run the jobs on a fork pool. specs {key: MjSpec} (the children inherit them), jobs from plan(). Each run has
    the harness's own simulated-time limit (its outcome) and a wall guard (job["wall_limit_s"]); the whole batch has
    the budget `deadline` - sec["budget_margin_s"] (wall clock). Reaching a guard or the budget raises WalkTimeError
    after the pool is shut down (a slow machine voids the episode instead of changing a score). Returns
    ({key: result}, stats); results do not depend on the pool size."""
    import multiprocessing as mp
    from multiprocessing.connection import wait as mp_wait
    t_start = time.time()
    n = pool_size(sec, len(jobs), procs)
    margin = float(sec.get("budget_margin_s", 0.0))
    stop_at = math.inf if deadline is None else float(deadline) - margin
    stats = {"procs": n, "jobs": len(jobs), "errors": 0, "predicted_cpu_s": float(sum(j["predicted_s"] for j in jobs))}
    results = {}
    if not jobs:
        stats["wall_s"] = 0.0
        return results, stats
    if stop_at - time.time() <= 0:
        raise WalkTimeError("no wall time was left for the %d walk runs before the verify deadline" % len(jobs))
    mpctx = mp.get_context("fork")
    pending = list(jobs)
    slots = []
    frozen = False
    try:
        gc.collect()
        gc.freeze()                    # the children share the parent's objects without copying their pages
        frozen = True
        slots = [_Slot(mpctx, specs, robot_dir) for _ in range(n)]
        while pending or any(s.job is not None for s in slots):
            now = time.time()
            for i, s in enumerate(slots):
                if isinstance(s, _Slot) and s.job is None and pending and now < stop_at:
                    j = pending.pop(0)
                    try:
                        s.conn.send((j["key"], j["spec"], j["route"], j["variant"]))
                    except (BrokenPipeError, OSError):        # the worker is gone: a fresh one takes the job
                        s.kill()
                        s = slots[i] = _Slot(mpctx, specs, robot_dir)
                        s.conn.send((j["key"], j["spec"], j["route"], j["variant"]))
                    s.job, s.t0 = j, time.time()
            if now >= stop_at:
                raise WalkTimeError("the walk budget ran out with %d of %d runs not finished (%d not started)" % (
                    len(pending) + sum(1 for s in slots if s.job is not None), len(jobs), len(pending)))
            busy = [s for s in slots if s.job is not None]
            if not busy:
                continue
            ends = [s.t0 + float(s.job["wall_limit_s"]) for s in busy] + [stop_at]
            timeout = max(0.0, min(ends) - time.time())
            ready = mp_wait([s.conn for s in busy], timeout=min(timeout, 5.0))
            for s in busy:
                if s.conn in ready:
                    dead = False
                    try:
                        key, res = s.conn.recv()
                    except (EOFError, OSError):
                        s.proc.join(5)
                        if s.proc.exitcode == -9:
                            # SIGKILL comes from outside the run (the out-of-memory killer, an operator): a property
                            # of the machine, so the episode is void, never "not reached"
                            raise WalkTimeError("a walk worker was killed (SIGKILL: out of memory?) during run %s "
                                                "(route %d, variant %d)" % (s.job["route"].get("id"), s.job["key"][0],
                                                                            s.job["key"][1]))
                        # any other death (a crash inside the harness) is the twin's: that run is not reached
                        key, res = s.job["key"], not_run("error", "the walk worker died during this run")
                        dead = True
                    if res.get("how") == "error":
                        stats["errors"] += 1
                    results[key] = res
                    s.job = None
                    if dead:                                  # replace the dead worker (or retire its slot)
                        s.kill()
                        i = slots.index(s)
                        slots[i] = _Slot(mpctx, specs, robot_dir) if (pending and time.time() < stop_at) else _Dead()
            now = time.time()
            for s in slots:
                if s.job is None:
                    continue
                if now - s.t0 > float(s.job["wall_limit_s"]):
                    raise WalkTimeError("walk run %s (route %d, variant %d) passed its wall guard of %.0f s (the "
                                        "harness's simulated limit is %.0f s)" % (
                                            s.job["route"].get("id"), s.job["key"][0], s.job["key"][1],
                                            float(s.job["wall_limit_s"]), float(s.job["sim_limit_s"])))
                if now >= stop_at:
                    raise WalkTimeError("the walk budget ran out with %d of %d runs not finished" % (
                        len(pending) + sum(1 for x in slots if x.job is not None), len(jobs)))
    finally:
        for s in slots:
            if isinstance(s, _Slot):
                try:
                    s.conn.send(None)
                except (BrokenPipeError, OSError):
                    pass
        for s in slots:
            if isinstance(s, _Slot):
                s.proc.join(2)
                if s.proc.is_alive():
                    s.kill()
        if frozen:
            gc.unfreeze()
    stats["wall_s"] = time.time() - t_start
    stats["max_run_wall_s"] = max((float(r.get("worker_wall_s", 0.0)) for r in results.values()), default=0.0)
    stats["max_run_wall_over_sim"] = max((float(r.get("worker_wall_s", 0.0)) / max(float(r.get("sim_time_s", 0.0)),
                                                                                   1.0) for r in results.values()),
                                         default=0.0)
    if log:
        log("walks: %d runs on %d processes in %.1f s (slowest run %.1f s, at most %.2f wall s per simulated s)" % (
            len(jobs), n, stats["wall_s"], stats["max_run_wall_s"], stats["max_run_wall_over_sim"]))
    return results, stats


class _Dead:
    """A slot whose worker was killed and not replaced (no work left)."""
    job = None
    conn = None


# ---------------------------------------------------------------------------------------------------------------
# scoring helpers (shared by score_c6 and build_routes)

def reached(res, sec):
    """A run reaches when the harness says so or its progress is at least sec["reach_progress"]."""
    return res.get("outcome") == "reached" or float(res.get("progress", 0.0)) >= float(sec["reach_progress"])


def variant_score(res, truth, sec):
    """1 if the truth reaches and the run reaches; band(|progress - progress_true|, *sec["progress_band"]) if
    neither reaches; 0 otherwise."""
    rr = reached(res, sec)
    if truth["outcome"] == "reached":
        return 1.0 if rr else 0.0
    if rr:
        return 0.0
    full, zero = sec["progress_band"]
    e = abs(float(res.get("progress", 0.0)) - float(truth["progress"]))
    if not math.isfinite(e) or e >= zero:
        return 0.0
    return 1.0 if e <= full else float((zero - e) / (zero - full))


def public(res):
    """The deterministic part of a run's result (what a score or a table may use)."""
    keep = ("outcome", "reason", "how", "progress", "sim_time_s", "contacts_with", "state_hash", "why",
            "max_tilt_deg", "end_xy")
    out = {k: res[k] for k in keep if k in res}
    if "progress" in out:
        out["progress"] = float(out["progress"])
    return out
