"""The grader: scores the submission's USD stage against the real bay, the brief and the requirements.

Runs as root after the agent's session (verify stage, `/opt/ale/tests/`). Nothing of the agent's runs: every process
of its account is killed, the submission is snapshotted (regular files only) under a fresh root-owned folder, the
shared Kit folders are emptied, and the measurements run as a separate unprivileged account with a clean environment
(`python3 -I -s`), each probe group in its own Isaac Sim process (one stage per process: a clean clock), with Kit's
cache in the grader's own folder. The static scan (`scan.py`, pxr only) decides whether the scene is measurable at
all; `probes.py` then measures it (layout, settle, the valve and the plume, the bucket on its ropes, the fleet's
design) and runs the bay mission on each day of the fixed hidden set (`hidden/fixed_set.json`: two missions, the same
for every submission): the agent's controller drives the agent's four robots in the grader's own stage of the real bay
(`mission_stage.py` on `hidden/bay_true.usda`), on the mission's day, while the grader keeps the bay's rules and
measures the outcomes. The controller runs as the agent's account behind the kit's solver wire (`GuardedSolver`:
started through `nonet_exec.py`, so that it can open no internet socket), started only once the probe has the mission
set up (and every process of the agent's account stopped before each run and after it); this process relays the wire
to the control probe through two pipes and applies the match lock. The wire carries no ground truth: each robot's
own state as its onboard systems estimate it, and a reference to the latest frame set its declared sensors delivered;
the frames themselves (the grader's cameras and lidar heads rendered in the real bay, `sensor_rig.py`) are slot
files in a folder root makes fresh for each mission (`work/frames`: the probe account's, 0755, readable by the agent's
program and writable by nobody else; removed after the run), named to the probe in its environment
(`ARB_FRAMES_DIR`). Two canaries go through the same launcher first: a trivial physics scene, and the real bay
dressed for the sensors rendered with one camera of each kind, a lidar head and the frame clock (the renderer's
pipelines compile there); when either fails, the grading environment is broken and the run ends as a task error, not
as a zero. So does a mission that cannot be set up twice in a row, before any of the agent's code ran, for a scene
that passed the scan and simulates on its own (unless both set-ups stopped at the sensors or their warm-up, after the
render canary rendered: then the fleet's own meshes or materials did not render, and that run is aborted, every check
0), and so does a missing or malformed set. Kit's shared folders (cache, logs, data) are emptied and made the probe
account's (0700): the compiled pipelines persist from one probe process to the next, and the agent's program cannot
start Kit while it is graded.

Nothing in a verify draws from OS entropy or the clock: the scene probes run at the set's seeds (0: truth.json's own),
each mission's sensor noise at its noise seed, and each mission's match lock from a stream keyed by its lock key (a
retry meets the same lock). What the probes read (their code, `grader_config.json` with the real bay's facts and the
set's days, the real bay's layer) sits in a folder only the probe account can read (`work/probe`, 0700); a mission's
seed and index travel in the probe's own environment, never on a command line (the agent's controller runs in the same
sandbox during the mission; only the probe's output file is named by the mission's place in the set's fixed order,
which tells nothing), and nothing of the set is written to the log.

A submission that carries the reference's token (`.oracle_token`, sha256 in `hidden/oracle_token.sha256`) gets the
true state in each observation (`privileged`): the reference program's channel; nothing else changes.

Scores: each check is 0..1 (most by closeness, some yes/no, the behaviour checks gated on what they need).
`acceptance_score` combines the sections as `grader_config.json` says (`combine`): "weighted", the weighted mean of the
sections (`sections`); "mission_gate", the weighted mean of the five scene sections times the mission section; or the
name of another weights block (`sections_<name>`, e.g. "mission_0.8"). With `progress_gate` the mission's checks that a
run earns without doing its jobs count only as far as it did them: the bucket's three (lift-off level, swing, water
kept) only in a run that put the fire out; the five safety checks (parts not dropped, no losses, no collisions,
separation; consistency by the two missions' mean) in proportion to the run's progress, (steam released + fire out +
station cleared) / 3. The metrics record the acceptance under every combination, with and without the gate.
"""

import hashlib
import hmac
import json
import math
import os
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
HIDDEN = os.path.join(HERE, "hidden")
PROBE_FILES = ("probes.py", "scan.py", "control_sim.py", "mission_stage.py", "bay_rig.py", "sensor_rig.py", "wire_obs.py")
FIXED_SET = os.path.join(HIDDEN, "fixed_set.json")  # (root's alone: never staged for the probe, never on a command line)
SET_FORMAT = "hazard_bay_build fixed set 1"
NONET = os.path.join(HERE, "nonet_exec.py")  # (run as root from this root-only folder)
sys.path.insert(0, HERE)
from robotics_grader import agent_user, fail, ratio_rewards, read_anchor, seed_rng, stage_config, write_verdict, zero_verdict  # noqa: E402
from robotics_grader.solver import ProbeLock, Solver, SolverAborted, SolverMissing  # noqa: E402
from robotics_grader.stage import secure_exec_path  # noqa: E402

PROBE_USER = "arbprobe"
KIT_SHARED = "/opt/venv/lib/python3.12/site-packages/isaacsim/kit"  # cache / logs / data: shared with the agent's runs
GROUPS = ("layout", "settle", "steam", "hoist", "fleet", "control")
FAC, FLEET = "/World/Facility", "/World/Fleet"
UAVS = ("UAV_1", "UAV_2", "UAV_3")


def log(*a):
    print("[grader]", *a, file=sys.stderr, flush=True)


class Reject(Exception):
    """The submission cannot be measured: it scores 0 (the agent's doing, not the grader's)."""


def num(x):
    """A finite float, or None (anything non-finite scores as missing)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def lin(err, full, zero):
    """1 up to `full`, 0 from `zero` on, linear between (None or non-finite -> 0)."""
    err = num(err)
    if err is None:
        return 0.0
    if err <= full:
        return 1.0
    if err >= zero:
        return 0.0
    return (zero - err) / (zero - full)


def dist(a, b):
    if a is None or b is None:
        return None
    try:
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
    except (TypeError, ValueError):
        return None


def yes(cond):
    return 1.0 if cond else 0.0


def nz(x, default):
    """A finite float, or `default` when missing (zero is a value, not a gap)."""
    v = num(x)
    return default if v is None else v


# --------------------------------------------------------------------------------------------- the fixed hidden set
def _int63(v):
    return not isinstance(v, bool) and isinstance(v, int) and 0 <= v < (1 << 63)


def read_missions(missions, part_names=None):
    """A list of missions as the fixed set holds them (each its `id`, `noise_seed` (63 bits: the sensors' noise),
    `lock_key` (64 hex digits: the match lock's stream) and `day`), checked and returned in order with each day
    normalised (`control_sim.normalise_day`; with `part_names`, a day must place exactly the real bay's parts). Raises
    ValueError saying what is wrong, never a value of the set."""
    import control_sim
    if not isinstance(missions, list) or not missions:
        raise ValueError("no missions")
    out, ids, seeds, keys = [], set(), set(), set()
    for k, m in enumerate(missions):
        if not isinstance(m, dict) or not isinstance(m.get("id"), str) or not m["id"]:
            raise ValueError(f"mission {k}: not an object with an id")
        if not _int63(m.get("noise_seed")):
            raise ValueError(f"mission {k}: noise_seed is not a 63-bit integer")
        key = m.get("lock_key")
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", key):
            raise ValueError(f"mission {k}: lock_key is not 64 hex digits")
        try:
            day = control_sim.normalise_day(m.get("day"))
        except ValueError as exc:
            raise ValueError(f"mission {k}: {exc}")
        if part_names is not None and set(day["parts"]) != set(part_names):
            raise ValueError(f"mission {k}: the day does not place exactly the real bay's parts")
        if m["id"] in ids or m["noise_seed"] in seeds or key.lower() in keys:
            raise ValueError(f"mission {k}: an id, noise seed or lock key of an earlier mission")
        ids.add(m["id"])
        seeds.add(m["noise_seed"])
        keys.add(key.lower())
        out.append(dict(id=m["id"], noise_seed=m["noise_seed"], lock_key=key.lower(), day=day))
    return out


def load_fixed_set(path=None, part_names=None):
    """The fixed hidden set (`hidden/fixed_set.json`, format SET_FORMAT): the scene probes' seeds and the missions,
    in order. Every submission meets all of it; a missing or malformed set is a task error. Nothing of it is logged."""
    path = path or FIXED_SET
    try:
        with open(path) as f:
            spec = json.load(f)
    except (OSError, ValueError) as exc:
        fail(f"cannot read the fixed hidden set ({os.path.basename(path)}): {type(exc).__name__}")
    try:
        if not isinstance(spec, dict) or spec.get("format") != SET_FORMAT:
            raise ValueError(f"not a set of the format '{SET_FORMAT}'")
        probes = spec.get("scene_probes")
        if not isinstance(probes, dict) or not all(_int63(probes.get(g)) for g in GROUPS if g != "control"):
            raise ValueError("scene_probes does not give each scene probe group a 63-bit seed")
        missions = read_missions(spec.get("missions"), part_names)
    except ValueError as exc:
        fail(f"the fixed hidden set is malformed: {exc}")
    return dict(scene_probes={g: probes[g] for g in GROUPS if g != "control"}, missions=missions)


# --------------------------------------------------------------------------------------------- lockdown
def inspect_submission(sub, max_bytes):
    """Every entry a regular file or directory, nothing else, within the size limit (no symlinks: the snapshot runs
    as root and must never follow a link out of the submission)."""
    total = 0
    for d, dirs, files in os.walk(sub, followlinks=False):
        for name in dirs + files:
            p = os.path.join(d, name)
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                raise Reject(f"symbolic link in the submission: {p}")
            if not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)):
                raise Reject(f"special file in the submission: {p}")
            if stat.S_ISREG(st.st_mode):
                total += st.st_size
                if total > max_bytes:
                    raise Reject(f"the submission exceeds {max_bytes} bytes")
    return total


def snapshot(sub, snap):
    """Copy regular files and directories only (checked again entry by entry), root-owned, world-readable."""
    os.makedirs(snap, mode=0o755)
    for d, dirs, files in os.walk(sub, followlinks=False):
        rel = os.path.relpath(d, sub)
        dst = snap if rel == "." else os.path.join(snap, rel)
        for name in dirs:
            p = os.path.join(d, name)
            if stat.S_ISDIR(os.lstat(p).st_mode):
                os.makedirs(os.path.join(dst, name), mode=0o755, exist_ok=True)
        for name in files:
            p = os.path.join(d, name)
            if stat.S_ISREG(os.lstat(p).st_mode):
                with open(p, "rb") as src, open(os.path.join(dst, name), "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                os.chmod(os.path.join(dst, name), 0o644)
    for d, dirs, files in os.walk(snap):
        os.chown(d, 0, 0)
        os.chmod(d, 0o755)
        for f in files:
            os.chown(os.path.join(d, f), 0, 0)


def lockdown(sub, work, max_bytes):
    kill_agent()
    clean_shm()
    for d in ("cache", "logs", "data"):  # nothing the agent's runs left there reaches the grader's Isaac Sim
        path = os.path.join(KIT_SHARED, d)
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, mode=0o1777, exist_ok=True)
        os.chmod(path, 0o1777)
    inspect_submission(sub, max_bytes)
    snap = os.path.join(work, "submission")
    snapshot(sub, snap)
    return snap


def kit_folders_to_probe():
    """Kit's shared folders (emptied by the lockdown) made the probe account's alone (0700): the renderer's compiled
    pipelines stay there from one probe process to the next (the probes' own folder arguments do not take in this
    Kit), and nothing the agent's account runs at grading can start Kit or plant anything there."""
    pw = pwd.getpwnam(PROBE_USER)
    for d in ("cache", "logs", "data"):
        path = os.path.join(KIT_SHARED, d)
        os.makedirs(path, exist_ok=True)
        os.chown(path, pw.pw_uid, pw.pw_gid)
        os.chmod(path, 0o700)


def clean_shm():
    """Remove what the agent's account left in /dev/shm. Kit's processes open shared memory there under predictable
    names (`carb-RStringInternals-<pid>` and its semaphore): one planted in advance would be opened by, or refuse,
    the grader's probe."""
    try:
        uid = pwd.getpwnam(agent_user()).pw_uid
    except KeyError:
        return
    for e in os.scandir("/dev/shm") if os.path.isdir("/dev/shm") else []:
        try:
            if e.stat(follow_symlinks=False).st_uid != uid:
                continue
            if e.is_dir(follow_symlinks=False):
                shutil.rmtree(e.path, ignore_errors=True)
            else:
                os.remove(e.path)
        except OSError:
            pass


def seal_probe_shm():
    """Close the probe's shared memory in /dev/shm to every account but its own. Kit creates it readable and writable
    by all (`carb-RStringInternals-<pid>` and its semaphore); done after the set-up and before the agent's program
    starts, so the program never reaches the probe's memory (the probe keeps the objects it has open)."""
    try:
        uid = pwd.getpwnam(PROBE_USER).pw_uid
    except KeyError:
        return
    for e in os.scandir("/dev/shm") if os.path.isdir("/dev/shm") else []:
        try:
            st = e.stat(follow_symlinks=False)
            if st.st_uid == uid and not stat.S_ISLNK(st.st_mode) and stat.S_IMODE(st.st_mode) & 0o077:
                os.chmod(e.path, stat.S_IMODE(st.st_mode) & 0o700)
        except OSError:
            pass


def bay_from_truth(truth):
    """The real bay's facts the mission is measured against (boxes as [lo, hi]): the plume, the crate, the gauge's
    dial, the two tables' tops."""
    lay = truth["groups"]["layout"]

    def box3(b):
        c, e = b["center"], b["extent"]
        return [[c[k] - e[k] / 2 for k in range(3)], [c[k] + e[k] / 2 for k in range(3)]]

    def box_xy(b):
        c, e = b["center"], b["extent"]
        return [[c[0] - e[0] / 2, c[1] - e[1] / 2, b["bottom"]], [c[0] + e[0] / 2, c[1] + e[1] / 2, b["top"]]]

    return dict(plume=box3(lay["plume"]), crate=box3(lay["crate"]), dial=box3(lay["gauge_dial"]),
                table_a=box_xy(lay["Table_A"]), table_b=box_xy(lay["Table_B"]))


def probe_account(work):
    home = os.path.join(work, "probe_home")  # (also its passwd home: some Kit extensions look it up there, not in $HOME)
    os.makedirs(home, exist_ok=True)
    try:
        pwd.getpwnam(PROBE_USER)
    except KeyError:
        subprocess.run(["/usr/sbin/useradd", "--system", "--no-create-home", "--home-dir", home, "--shell", "/usr/sbin/nologin",
                        PROBE_USER], check=True)
    pw = pwd.getpwnam(PROBE_USER)
    for d in (home, os.path.join(home, "tmp"), os.path.join(work, "kit"), os.path.join(work, "kit", "optix"), os.path.join(work, "out"),
              probe_dir(work), os.path.join(probe_dir(work), "hidden")):
        os.makedirs(d, exist_ok=True)
        os.chown(d, pw.pw_uid, pw.pw_gid)
        os.chmod(d, 0o700)
    return home


def probe_dir(work):
    """The probe account's own folder: the probes' code, the grader's config and the real bay (0700; the agent's
    controller runs as another account in this sandbox while the mission runs)."""
    return os.path.join(work, "probe")


def probe_env(work, seed=None):
    home = os.path.join(work, "probe_home")
    env = {"HOME": home, "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
           "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "ARB_KIT_DIR": os.path.join(work, "kit"),
           "ARB_GRADER_CFG": os.path.join(probe_dir(work), "grader_config.json"),
           "NVIDIA_DRIVER_CAPABILITIES": os.environ.get("NVIDIA_DRIVER_CAPABILITIES", ""),
           # the probe's temporary files and OptiX's disk cache in its own folders (OptiX's default, /var/tmp/OptixCache_<user>,
           # is a name the agent's session could create first)
           "TMPDIR": os.path.join(home, "tmp"), "OPTIX_CACHE_PATH": os.path.join(work, "kit", "optix")}
    if seed is not None:  # (a process's environment is readable by its own account and root only; its command line by all)
        env["ARB_SEED"] = str(int(seed))
    return env


def probe_cmd(args):
    return ["/usr/sbin/runuser", "-u", PROBE_USER, "--", "/opt/venv/bin/python3", "-I", "-s"] + list(args)


def as_probe(args, work, timeout, seed=None):
    return subprocess.run(probe_cmd(args), env=probe_env(work, seed), cwd=os.path.join(work, "probe_home"), capture_output=True, text=True,
                          timeout=timeout)


def stage_file(work, name, src=None):
    """A copy of one of this folder's files in the probe account's own folder, readable by that account only."""
    dst = os.path.join(probe_dir(work), name)
    shutil.copy(src or os.path.join(HERE, name), dst)
    pw = pwd.getpwnam(PROBE_USER)
    os.chown(dst, pw.pw_uid, pw.pw_gid)
    os.chmod(dst, 0o600)
    return dst


def write_probe_config(work, cfg, name="grader_config.json"):
    path = os.path.join(probe_dir(work), name)
    with open(path, "w") as f:
        json.dump(cfg, f)
    pw = pwd.getpwnam(PROBE_USER)
    os.chown(path, pw.pw_uid, pw.pw_gid)
    os.chmod(path, 0o600)
    return path


def token_state(snap):
    """(privileged, mismatch): the snapshot's .oracle_token against hidden/oracle_token.sha256."""
    p, want = os.path.join(snap, ".oracle_token"), os.path.join(HIDDEN, "oracle_token.sha256")
    if not os.path.isfile(p) or os.path.islink(p) or not os.path.isfile(want):
        return False, False
    try:
        with open(p, "rb") as f:
            got = hashlib.sha256(f.read().strip()).hexdigest()
        with open(want) as f:
            ok = hmac.compare_digest(got, f.read().strip())
    except OSError:
        return False, False
    return ok, not ok


def match_items(got_items, truth_items, tol):
    """The agent's parts matched to the real bay's by shape (the library model's size, within `item_size_zero`), then
    by position, one to one: {real name: (agent name, size error)}."""
    used, out = set(), {}
    for tname, tv in truth_items.items():
        cands = []
        for n, gv in (got_items or {}).items():
            if n in used or not gv.get("size") or not gv.get("center"):
                continue
            dsize = max(abs(float(x) - float(y)) for x, y in zip(gv["size"], tv["size"]))
            if dsize <= tol["item_size_zero"]:
                d = dist(gv["center"], tv["center"])
                cands.append((9.0 if d is None else d, dsize, n))
        if cands:
            _, dsize, best = min(cands)
            used.add(best)
            out[tname] = (best, dsize)
    return out


def canary(work, timeout, bay_layer=None):
    """A trivial scene through the probe launcher: a cube falling on a plane, the sensor extension loaded, a GPU
    present (physics itself runs on the CPU, as it did for truth.json); then (`bay_layer`) the render canary in a
    process of its own: the real bay dressed for the sensors, one camera of each kind, a lidar head, the frame clock
    (probes.py canary_render). If either does not work, the environment is at fault, not the submission."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    path = os.path.join(work, "canary.usda")
    st = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(st, 1.0)
    world = UsdGeom.Xform.Define(st, "/World")
    st.SetDefaultPrim(world.GetPrim())
    sc = UsdPhysics.Scene.Define(st, "/World/PhysicsScene")
    sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr(9.81)
    g = UsdGeom.Cube.Define(st, "/World/Ground")
    g.CreateSizeAttr(1.0)
    g.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
    g.AddScaleOp().Set(Gf.Vec3f(10, 10, 1))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    c = UsdGeom.Cube.Define(st, "/World/Cube")
    c.CreateSizeAttr(0.1)
    c.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.0))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(c.GetPrim())
    UsdPhysics.MassAPI.Apply(c.GetPrim()).CreateMassAttr(0.5)
    st.GetRootLayer().Save()
    os.chmod(path, 0o644)
    out = os.path.join(work, "out", "canary.json")
    t0 = time.time()
    try:
        r = as_probe([os.path.join(probe_dir(work), "probes.py"), path, out, "canary"], work, timeout, seed=0)
    except subprocess.TimeoutExpired:
        fail(f"the canary scene did not finish in {timeout} s: the grading environment is not usable")
    data = {}
    if os.path.isfile(out):
        try:
            data = json.load(open(out))["groups"].get("canary", {})
        except Exception:  # noqa: BLE001
            data = {}
    log(f"canary: {round(time.time() - t0)} s", json.dumps(data)[:300])
    if not data or "error" in data:
        fail(f"the canary scene failed (exit {r.returncode}): {data.get('error', '')[-400:]} | stderr: {(r.stderr or '')[-800:]}")
    if not data.get("fell"):
        fail(f"physics did not run in the canary scene: {data}")
    if not data.get("cuda_available"):
        fail("no usable GPU in the sandbox (torch.cuda.is_available() is false): the grading environment is not usable")
    if not data.get("sensors_ext"):
        fail("the physics sensors extension did not load: the grading environment is not usable")
    if bay_layer is not None:
        data["render"] = canary_render(work, timeout, bay_layer)
    return data


def canary_render(work, timeout, bay_layer):
    out = os.path.join(work, "out", "canary_render.json")
    t0 = time.time()
    try:
        r = as_probe([os.path.join(probe_dir(work), "probes.py"), bay_layer, out, "canary_render"], work, timeout, seed=0)
    except subprocess.TimeoutExpired:
        fail(f"the render canary did not finish in {timeout} s: the grading environment cannot render")
    data = {}
    if os.path.isfile(out):
        try:
            data = json.load(open(out))["groups"].get("canary_render", {})
        except Exception:  # noqa: BLE001
            data = {}
    keep = {k: data.get(k) for k in ("ok", "clock_ok", "checks", "first_render_s", "open_s", "wall_s", "gpu")}
    log(f"render canary: {round(time.time() - t0)} s", json.dumps(keep)[:600])
    if not data or "error" in data:
        fail(f"the render canary failed (exit {r.returncode}): {data.get('error', '')[-400:]} | stderr: {(r.stderr or '')[-800:]}")
    if not data.get("ok"):
        fail(f"the render canary rendered nothing usable: {json.dumps(keep)[:600]}")
    return keep


def flat_action(action):
    """The numbers of an action in a fixed order (the kit's lock compares flat lists of numbers): the nine thrusts,
    then the vehicle's targets by joint name and its grip."""
    thrust = (action.get("thrust") if isinstance(action, dict) else None) or {}
    out = []
    for n in UAVS:
        v = thrust.get(n) or [0.0, 0.0, 0.0]
        out += [float(x) for x in (list(v) + [0.0, 0.0, 0.0])[:3]]
    ugv = (action.get("ugv") if isinstance(action, dict) else None) or {}
    for key in ("vel", "pos"):
        d = ugv.get(key) or {}
        out += [float(d[k]) for k in sorted(d) if isinstance(d[k], (int, float))]
    out.append(1.0 if ugv.get("grip") else 0.0)
    return out


def probe_pair(lock, solver, t, obs):
    """The lock's probed step on a structured action: the two queries in the lock's hidden order, the real one
    applied, the pair recorded as flat numbers (`ProbeLock.probe` itself only flattens lists)."""
    perturbed = perturbed_obs(obs, lock)
    if lock.rng.random() < 0.5:
        action_probe = solver.act(t, perturbed)
        action_real = solver.act(t, obs)
    else:
        action_real = solver.act(t, obs)
        action_probe = solver.act(t, perturbed)
    lock.record(flat_action(action_probe), flat_action(action_real))
    return action_real


def perturbed_obs(obs, lock):
    """The match lock's perturbed copy of a structured observation: every robot's odometry shifted (and, in a
    privileged run, its true position by the same amount, so the probe asks the same question of every program)."""
    o = json.loads(json.dumps(obs))
    priv = o.get("privileged") or {}

    def shift(v, dv):  # (rounded as the observation rounds positions: the copy is shaped like any other request)
        return round(float(v) + dv, 4)
    for n, u in (o.get("uav") or {}).items():
        if isinstance(u.get("pos"), list):
            d = [lock.rng.uniform(-lock.magnitude, lock.magnitude) for _ in u["pos"]]
            u["pos"] = [shift(v, dv) for v, dv in zip(u["pos"], d)]
            pu = (priv.get("uav") or {}).get(n) or {}
            if isinstance(pu.get("pos"), list):
                pu["pos"] = [shift(v, dv) for v, dv in zip(pu["pos"], d)]
    g = o.get("ugv") or {}
    if isinstance(g.get("pose"), list) and len(g["pose"]) == 3:
        d = [lock.rng.uniform(-lock.magnitude, lock.magnitude) for _ in range(2)]
        g["pose"] = [shift(g["pose"][0], d[0]), shift(g["pose"][1], d[1]), g["pose"][2]]
        pg = priv.get("ugv") or {}
        if isinstance(pg.get("pos"), list) and len(pg["pos"]) == 3:
            pg["pos"] = [shift(pg["pos"][0], d[0]), shift(pg["pos"][1], d[1]), pg["pos"][2]]
    return o


def kill_agent():
    """Every process of the agent's account: what a controller left running (a helper it started with `setsid` leaves
    the solver's process group) must not reach the next mission run's set-up. Repeated until none is left (a process
    that keeps forking can put a child out of reach of one sweep)."""
    for _ in range(20):
        subprocess.run(["/usr/bin/pkill", "-KILL", "-u", agent_user()], check=False)
        time.sleep(0.25)
        ps = subprocess.run(["/usr/bin/ps", "-o", "stat=", "-u", agent_user()], capture_output=True, text=True)
        if not [st for st in ps.stdout.split() if not st.startswith("Z")]:  # (a zombie is already gone)
            return


# --------------------------------------------------------------------------------------------- the program's network
def agent_ids():
    pw = pwd.getpwnam(agent_user())
    return pw.pw_uid, pw.pw_gid


class GuardedSolver(Solver):
    """The kit's Solver (the kit stays as it is), its program started through `nonet_exec.py` (no_new_privs; a seccomp
    filter under which socket() and socketpair() work for AF_UNIX only, io_uring and foreign syscall ABIs refused; then
    the agent's account and a fixed environment: PATH /opt/venv/bin:/usr/local/bin:/usr/bin:/bin, HOME /home/user,
    LANG C.UTF-8). Where the filter cannot be installed (`guard` False: the egress self-test said "detect") the program
    starts as the agent's account with the same environment, and the relay watches its sockets instead. `env_prefix`
    (NAME=VALUE words) is for a maintainer's run only (`review/run_mission.py --program-env`); the grader passes none."""

    def __init__(self, *args, guard=True, env_prefix=(), **kwargs):
        self.guard = bool(guard)
        self.env_prefix = [str(e) for e in env_prefix]
        super().__init__(*args, **kwargs)

    def _spawn(self):
        secure_exec_path()  # (as the kit's deprivileged does: nothing agent-writable left on root's PATH)
        try:
            uid, gid = agent_ids()
        except KeyError:
            raise SolverAborted("no agent account to run the controller as")
        prog = (["/usr/bin/env"] + self.env_prefix if self.env_prefix else []) + ["bash", self.entry]
        if self.guard:
            cmd = [sys.executable, "-I", NONET, "--uid", str(uid), "--gid", str(gid), "--cwd", self.submission, "--"] + prog
        else:
            from nonet_exec import WHITELIST_ENV
            runuser = shutil.which("runuser", path="/usr/sbin:/usr/bin:/sbin:/bin") or "/usr/sbin/runuser"
            cmd = ([runuser, "-u", agent_user(), "--", "/usr/bin/env", "-i"] + [f"{k}={v}" for k, v in sorted(WHITELIST_ENV.items())]
                   + prog)
        try:
            return subprocess.Popen(cmd, cwd=self.submission, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    universal_newlines=True, bufsize=1, start_new_session=True, env={"PATH": "/usr/bin:/bin"})
        except OSError as exc:
            raise SolverAborted(f"the controller could not be started: {exc}")


def egress_self_test(uid, gid):
    """True if a program started through the launcher cannot open an internet socket (AF_INET, AF_INET6, AF_VSOCK)
    and can open a Unix one."""
    probe = ("import socket,sys\n"
             "bad=0\n"
             "for fam in (socket.AF_INET, socket.AF_INET6, 40):\n"
             "    try:\n"
             "        socket.socket(fam, socket.SOCK_STREAM).close(); bad=1\n"
             "    except OSError:\n"
             "        pass\n"
             "try:\n"
             "    socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).close()\n"
             "except OSError:\n"
             "    bad=1\n"
             "sys.exit(3 if bad else 0)\n")
    try:
        r = subprocess.run([sys.executable, "-I", NONET, "--uid", str(uid), "--gid", str(gid), "--cwd", "/", "--", sys.executable, "-I", "-c", probe],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, env={"PATH": "/usr/bin:/bin"})
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def egress_mode():
    """Decided once, before the missions: "seccomp" (the launcher's filter holds) or "detect" (it cannot be installed
    here: a run in which the agent's account holds an internet socket with a peer off this machine aborts)."""
    secure_exec_path()
    try:
        uid, gid = agent_ids()
    except KeyError:
        return "detect"
    return "seccomp" if egress_self_test(uid, gid) else "detect"


def agent_inet_sockets(uid):
    """How many internet sockets the agent's account holds with a peer that is not this machine's loopback (the
    detect mode's check)."""
    n = 0
    for name in ("tcp", "tcp6", "udp", "udp6", "raw", "raw6"):
        try:
            with open("/proc/net/" + name) as f:
                next(f, None)
                for line in f:
                    parts = line.split()
                    if len(parts) < 8 or int(parts[7]) != uid:
                        continue
                    rem = parts[2].split(":")[0]
                    if rem.strip("0") == "" or rem in ("0100007F", "00000000000000000000000001000000"):
                        continue
                    n += 1
        except (OSError, ValueError, StopIteration):
            continue
    return n


def run_control(scene, work, cfg, mission, index, snap, egress="seccomp", env_prefix=()):
    """The control probe (Isaac Sim, the probe account) on one side; the agent's controller (`GuardedSolver`: `bash
    run.sh` in the snapshot, the agent's account, no internet sockets) on the other. This process relays the wire
    through two pipes and applies the match lock. The controller sees observations and answers actions; nothing else
    crosses. `mission` is the fixed set's entry `index` (its day is the probe config's `missions[index]`): its noise
    seed and the index go to the probe in its environment, and its lock key keys the lock's stream, made here, so that
    a retry meets the same lock.

    Order: nothing of the agent's runs while the probe sets the mission up (it builds the mission stage, starts Isaac
    Sim, opens the stage, sets the day up, pre-rolls); the controller starts once the probe opens the wire. So a failure
    before that point (`setup_failed`) is the grader's or the environment's, never the program's doing. The set-up has
    its own bound (`setup_timeout_s`). The program's budget (`reply_budget_s`) is its own time, the time from each
    request to its reply summed over the run: the simulator's own pace, which depends on the machine and its load,
    never counts against it. A run that has not ended `episode_wall_cap_s` after the wire opened is stopped all the
    same (the bound on a verify; with the sensors rendered every 6 control steps the reference fleet ran at about
    0.13 s a control step in P2, the physics alone at 19-55 ms). Either ends the run as aborted. With `egress` "detect" the relay looks at the agent's account's sockets every 60 steps and aborts
    the run at the first internet socket with a peer off this machine."""
    ctl = cfg.get("control", {})
    budget = float(ctl.get("reply_budget_s", 1800))
    wall_cap = float(ctl.get("episode_wall_cap_s", 5400))
    if not os.path.isfile(os.path.join(snap, "run.sh")):
        log("control: no controller (no run.sh in the submission)")
        return dict(abort="no controller: no run.sh in the submission", end="aborted", egress_mode=egress, egress_hits=0)
    kill_agent()
    clean_shm()
    wire = os.path.join(work, "wire")
    os.makedirs(wire, exist_ok=True)
    pw = pwd.getpwnam(PROBE_USER)
    os.chown(wire, pw.pw_uid, pw.pw_gid)
    os.chmod(wire, 0o700)
    fifos = {}
    for n in ("obs.fifo", "act.fifo"):
        p = os.path.join(wire, n)
        if os.path.exists(p):
            os.remove(p)
        os.mkfifo(p, 0o600)
        os.chown(p, pw.pw_uid, pw.pw_gid)
        fifos[n] = p
    lock = ProbeLock(seed_rng(entropy=int(mission["lock_key"], 16)), rate=float(ctl.get("lock_rate", 0.05)),
                     magnitude=float(ctl.get("lock_magnitude", 0.4)))
    path = os.path.join(work, "out", f"control_{index}.json")
    if os.path.exists(path):
        os.remove(path)
    frames = os.path.join(work, "frames")  # the sensors' slot files: fresh for each mission, the probe's, readable by all
    shutil.rmtree(frames, ignore_errors=True)
    os.makedirs(frames, mode=0o755)
    os.chown(frames, pw.pw_uid, pw.pw_gid)
    os.chmod(frames, 0o755)
    env = probe_env(work, mission["noise_seed"])
    env["ARB_WIRE_DIR"] = wire
    env["ARB_FRAMES_DIR"] = frames
    env["ARB_MISSION"] = str(int(index))
    proc = subprocess.Popen(probe_cmd([os.path.join(probe_dir(work), "probes.py"), scene, path, "control"]), env=env,
                            cwd=os.path.join(work, "probe_home"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sink = {"out": [], "err": []}

    def pump(stream, key):
        for line in stream:
            sink[key].append(line)

    threading.Thread(target=pump, args=(proc.stdout, "out"), daemon=True).start()
    threading.Thread(target=pump, args=(proc.stderr, "err"), daemon=True).start()
    state = {"steps": 0, "abort": None, "wire": False, "solver": None, "egress_hits": 0, "reply_s": 0.0}
    try:
        agent_uid = agent_ids()[0]
    except KeyError:
        agent_uid = None

    def relay():
        try:
            with open(fifos["obs.fifo"]) as fi, open(fifos["act.fifo"], "w") as fo:
                if proc.poll() is not None:  # (opened by the unblocking below, not by the probe: it died during set-up)
                    return
                state["wire"] = time.time()  # (the probe has the mission ready: only now does the agent's program start)
                seal_probe_shm()
                try:
                    state["solver"] = GuardedSolver(step_timeout=float(ctl.get("step_timeout_s", 10.0)), submission=snap, entry="run.sh",
                                                    max_restarts=int(ctl.get("max_restarts", 1)), guard=egress == "seccomp",
                                                    env_prefix=env_prefix)
                    state["solver"].reset()
                except (SolverAborted, SolverMissing) as exc:
                    state["abort"] = f"the controller did not come up: {exc}"[:300]
                    log("control:", state["abort"])
                    fo.write(json.dumps({"abort": state["abort"]}) + "\n")
                    fo.flush()
                    return
                solver = state["solver"]
                for line in fi:
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    if msg.get("end"):
                        break
                    t, obs = int(msg["t"]), msg["obs"]
                    if egress != "seccomp" and state["steps"] % 60 == 0 and agent_uid is not None and agent_inet_sockets(agent_uid):
                        state["egress_hits"] += 1
                        state["abort"] = "the controller held an internet socket (no network while it is graded)"
                        fo.write(json.dumps({"abort": state["abort"]}) + "\n")
                        fo.flush()
                        break
                    asked = time.monotonic()
                    try:
                        if lock.due():
                            action = probe_pair(lock, solver, t, obs)
                        else:
                            action = solver.act(t, obs)
                    except SolverAborted as exc:
                        state["abort"] = str(exc)[:300]
                        fo.write(json.dumps({"abort": state["abort"]}) + "\n")
                        fo.flush()
                        break
                    state["reply_s"] += time.monotonic() - asked  # (the program's own time: from each request to its reply)
                    if state["reply_s"] > budget:
                        state["abort"] = f"the controller's replies took more than {budget:.0f} s in all"
                        fo.write(json.dumps({"abort": state["abort"]}) + "\n")
                        fo.flush()
                        break
                    fo.write(json.dumps({"action": action}) + "\n")
                    fo.flush()
                    state["steps"] += 1
        except (OSError, ValueError) as exc:
            state["abort"] = state["abort"] or f"wire: {exc}"[:300]

    th = threading.Thread(target=relay, daemon=True)
    th.start()
    t0 = time.time()
    setup_s = float(ctl.get("setup_timeout_s", 1200))
    while proc.poll() is None:  # (the set-up has its own bound; the run's wall-clock cap counts from the wire's opening)
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
        if proc.poll() is not None:
            break
        if state["wire"]:
            seal_probe_shm()  # (and whatever the probe opens there later)
        if not state["wire"] and time.time() - t0 > setup_s:
            proc.kill()
            state["abort"] = state["abort"] or f"the mission was not set up within {setup_s:.0f} s"
        elif state["wire"] and time.time() - state["wire"] > wall_cap:
            proc.kill()
            state["abort"] = state["abort"] or f"the run did not end within {wall_cap:.0f} s of wall clock"
    if th.is_alive():  # (the probe died before opening its ends, or mid-run: unblock the relay)
        for n, mode in (("obs.fifo", os.O_WRONLY), ("act.fifo", os.O_RDONLY)):
            try:
                os.close(os.open(fifos[n], mode | os.O_NONBLOCK))
            except OSError:
                pass
        th.join(timeout=60.0)  # (a controller still coming up has 40 s to print READY)
    if state["solver"] is not None:
        state["solver"].close()
    kill_agent()
    shutil.rmtree(frames, ignore_errors=True)
    data = {}
    if os.path.isfile(path):
        try:
            data = json.load(open(path))["groups"].get("control", {})
        except Exception:  # noqa: BLE001
            data = {}
    if not data:
        data = {"error": f"no result (exit {proc.returncode}); stderr: {''.join(sink['err'])[-800:]}"}
    if state["abort"] and "abort" not in data:
        data["abort"] = state["abort"]
    data["lock"] = lock.rewards()
    data["relay_steps"] = state["steps"]
    data["reply_s"] = round(state["reply_s"], 1)
    data["setup_failed"] = not state["wire"]
    data["egress_mode"], data["egress_hits"] = egress, state["egress_hits"]
    data["probe_out"] = "".join(sink["out"])[-3000:]  # (the probe's progress lines, for the record; never logged)
    log(f"control probe: {round(time.time() - t0)} s, {state['steps']} steps relayed, the controller's replies {state['reply_s']:.0f} s, "
        f"abort={state['abort']}, lock={lock.rewards()}")
    return data


def run_probes(scene, work, timeout, retries, fixed, cfg, snap, cfg_probe=None, truth=None):
    """The scene probe groups at the fixed set's seeds, then the bay mission on each of the set's days."""
    groups = {}
    for g in GROUPS:
        if g == "control":  # the bay mission on each day of the fixed set: the outcomes and how consistent they are
            if cfg_probe is not None:  # the real bay's parts by the agent's names (matched by shape after the settle probe)
                alias = match_items(((groups.get("settle") or {}).get("items")) or {}, truth["groups"]["settle"]["items"], cfg["tol"])
                cfg_probe["item_alias"] = {t: a for t, (a, _) in alias.items()}
                write_probe_config(work, cfg_probe)
                log("control: parts by the agent's names:", json.dumps(cfg_probe["item_alias"]))
            egress = egress_mode()  # (once, before the missions)
            log("control: egress guard:", egress)
            if egress != "seccomp":  # (fail closed: the fixed set is public, so a program with the network could fetch its
                # day; sampling sockets cannot stop a fetch made and closed between two looks. Grading without the filter
                # would not be the grading the instruction describes: a task error, not a score)
                fail("the grading environment cannot block the controller's network (the seccomp filter cannot be "
                     "installed here); the bay mission is not graded with the network open")
            runs = []
            for e, mission in enumerate(fixed["missions"]):
                data = run_control(scene, work, cfg, mission, e, snap, egress)
                if "error" in data and not data.get("relay_steps"):  # (a fault before the first step: once more, the same day and lock)
                    log(f"control run {e} failed before its first step:", str(data["error"])[-600:])
                    first_stage = data.get("setup_stage")
                    data = run_control(scene, work, cfg, mission, e, snap, egress)
                    data["retried"] = True
                    if ("error" in data and data.get("setup_failed") and first_stage in ("sensors", "warmup")
                            and data.get("setup_stage") in ("sensors", "warmup")):
                        # both set-ups stopped at the fleet's sensors after the render canary rendered the real bay: the
                        # fleet's own meshes or materials do not render (the agent's doing): the run is aborted
                        data["abort"] = (f"the fleet's sensors could not be rendered in the real bay (twice, at "
                                         f"{data.get('setup_stage')}): {str(data['error'])[-300:]}")
                        data["end"] = "aborted"
                    elif "error" in data and data.get("setup_failed") and "error" not in (groups.get("settle") or {"error": True}):
                        # the mission could not be set up twice, before any of the agent's code ran, for a scene that
                        # passed the scan and simulates on its own (the settle probe): the grader's fault, not a zero
                        fail(f"control run {e}: the bay mission could not be set up in the real bay (twice) for a scene that "
                             f"passed the scan and runs on its own: {str(data['error'])[-900:]}")
                runs.append(data)
            groups[g] = runs
            continue
        data, tail, rc = {}, "", None
        for attempt in range(1 + retries):
            path = os.path.join(work, "out", f"{g}_{attempt}.json")
            t0 = time.time()
            try:
                r = as_probe([os.path.join(probe_dir(work), "probes.py"), scene, path, g], work, timeout, seed=fixed["scene_probes"][g])
                rc, tail = r.returncode, (r.stderr or "")[-1500:]
            except subprocess.TimeoutExpired:
                rc, tail = "timeout", ""
            data = {}
            if os.path.isfile(path):
                try:
                    data = json.load(open(path))["groups"].get(g, {})
                except Exception:  # noqa: BLE001
                    data = {}
            if not data:
                data = {"error": f"no result (exit {rc})"}
            log(f"probe {g} (try {attempt}): {round(time.time() - t0)} s", "error" if "error" in data else "ok")
            if "error" not in data:
                break
            log(f"probe {g} failed:", data["error"][-600:], "| stderr:", tail[-900:])
        groups[g] = data
    return groups


# --------------------------------------------------------------------------------------------- scoring
def score(scan, got, truth, cfg):
    tol, spec, cat = cfg["tol"], cfg["spec"], cfg["catalog"]
    sections = {}

    def add(section, name, weight, value, detail=None):
        v = num(value)
        v = 0.0 if v is None else max(0.0, min(1.0, v))
        sections.setdefault(section, []).append(dict(name=f"{section}.{name}", weight=weight, score=round(v, 4), detail=detail))

    def mean(vals):
        vals = [num(v) for v in vals]
        vals = [0.0 if v is None else v for v in vals]
        return sum(vals) / len(vals) if vals else 0.0

    def box_score(a, b, pos=("pos_full", "pos_zero"), ext=("ext_full", "ext_zero"), top=True):
        if not a or not b:
            return 0.0
        parts = [lin(dist(a["center"], b["center"]), tol[pos[0]], tol[pos[1]])]
        if ext and a.get("extent") and b.get("extent"):
            parts.append(lin(max(abs(float(x) - float(y)) for x, y in zip(a["extent"], b["extent"])), tol[ext[0]], tol[ext[1]]))
        if top and a.get("top") is not None and b.get("top") is not None:
            parts.append(lin(abs(float(a["top"]) - float(b["top"])), tol["top_full"], tol["top_zero"]))
        return sum(parts) / len(parts)

    # ---- layout: the bay as the photographs show it
    lay, tlay = got.get("layout", {}) or {}, truth["groups"]["layout"]
    for name, w in (("Table_A", 1.0), ("Table_B", 1.0), ("crate", 0.8), ("tank", 0.6), ("gauge_dial", 0.6), ("valve_handle", 0.8)):
        add("layout", name, w, box_score(lay.get(name), tlay.get(name)), lay.get(name))
    add("layout", "plume", 1.0, box_score(lay.get("plume"), tlay["plume"], pos=("plume_pos_full", "plume_pos_zero"),
                                           ext=("plume_ext_full", "plume_ext_zero"), top=False), lay.get("plume"))
    fp, tfp = lay.get("fire_pan"), tlay["fire_pan"]
    add("layout", "fire_pan", 0.8, (lin(dist(fp["center"][:2], tfp["center"][:2]), tol["pos_full"], tol["pos_zero"])
                                    + lin(abs(max(fp["extent"][:2]) - max(tfp["extent"][:2])), tol["ext_full"], tol["ext_zero"])) / 2 if fp else 0.0, fp)
    bk, tbk = lay.get("bucket"), tlay["bucket"]
    add("layout", "bucket", 0.6, (lin(dist(bk["center"][:2], tbk["center"][:2]), tol["pos_full"], tol["pos_zero"])
                                  + lin(abs(bk["top"] - tbk["top"]), tol["top_full"], tol["top_zero"])) / 2 if bk else 0.0, bk)
    for name in ("Pad_1", "Pad_2", "Pad_3", "UGV_Bay", "Valve_Square"):
        a, b = (lay.get("pads") or {}).get(name), tlay["pads"][name]
        add("layout", f"pad.{name}", 0.3, lin(dist(a["center"], b["center"]), tol["pad_full"], tol["pad_zero"]) if a else 0.0, a and a["center"])
    # items: matched by shape (the library model's size), then by position; an extra body costs an item's credit
    set_items = (got.get("settle", {}) or {}).get("items", {}) or {}
    t_items = truth["groups"]["settle"]["items"]
    matched = match_items(set_items, t_items, tol)
    for tname, tv in t_items.items():
        if tname not in matched:
            add("layout", f"item.{tname}", 0.5, 0.0, "not found")
            add("physics", f"mass.{tname}", 0.5, 0.0, "not found")
            continue
        best, dsize = matched[tname]
        gv = set_items[best]
        add("layout", f"item.{tname}", 0.5, (lin(dist(gv["center"], tv["center"]), tol["pos_full"], tol["pos_zero"])
                                             + lin(dsize, tol["item_size_full"], tol["item_size_zero"])) / 2, dict(as_=best, center=gv["center"]))
        add("physics", f"mass.{tname}", 0.5, lin(abs(num(gv.get("mass")) - tv["mass"]) / tv["mass"] if num(gv.get("mass")) is not None else None,
                                                 tol["mass_full"], tol["mass_zero"]), gv.get("mass"))
    extra_items = max(0, len(set_items) - len(t_items))
    add("layout", "items.no_extras", 0.5 * max(1, extra_items), yes(extra_items == 0), len(set_items))
    robots = lay.get("robots") or {}
    for name, pad in (("UAV_1", "Pad_1"), ("UAV_2", "Pad_2"), ("UAV_3", "Pad_3"), ("UGV", "UGV_Bay")):
        r = robots.get(name)
        c = (r or {}).get("box", {}) and r["box"]["center"]
        add("layout", f"rest.{name}", 0.4, lin(dist(c[:2], tlay["pads"][pad]["center"]), tol["pad_full"], tol["pad_zero"]) if c else 0.0, c)

    # ---- physics: masses and rest
    settle = got.get("settle", {}) or {}
    bm = num(settle.get("bucket_mass"))
    add("physics", "bucket_mass", 1.0, lin(abs(bm - spec["bucket_kg"]) / spec["bucket_kg"] if bm is not None else None, tol["mass_full"], tol["mass_zero"]), bm)
    moved = settle.get("moved", {}) or {}
    fac_moved = [num(v) for k, v in moved.items() if k.startswith(FAC) and num(v) is not None]
    fleet_moved = [num(v) for k, v in moved.items() if k.startswith(FLEET) and num(v) is not None]
    rope_moved = [num(v) for v in (settle.get("rope_moved") or {}).values() if num(v) is not None]
    add("physics", "settle.facility", 1.5, lin(max(fac_moved), tol["settle_full"], tol["settle_zero"]) if fac_moved else 0.0, max(fac_moved, default=None))
    add("physics", "settle.fleet", 1.5, lin(max(fleet_moved), tol["settle_full"], tol["settle_zero"]) if fleet_moved else 0.0, max(fleet_moved, default=None))
    add("physics", "settle.ropes", 0.5, lin(max(rope_moved), 0.25, 0.60) if rope_moved else 0.0, max(rope_moved, default=None))

    # ---- the steam line and the hazards
    stm = got.get("steam", {}) or {}
    ini = stm.get("initial") or {}
    add("steam", "initial_state", 0.5, yes(ini.get("steam") is True and ini.get("fire") is True and ini.get("ugv_disabled") is False
                                           and all(v is False for v in (ini.get("disabled") or {}).values()) and len(ini.get("disabled") or {}) == 3), ini)
    add("steam", "plume_disables", 1.5, yes(stm.get("uav1_disabled_1s") is True), stm.get("uav1_pos"))
    add("steam", "outside_not_disabled", 0.8, yes(stm.get("uav2_disabled_outside") is False), stm.get("p_out"))
    add("steam", "untouched_not_disabled", 0.4, yes(stm.get("uav3_disabled_untouched") is False))
    add("steam", "fire_disables_ugv", 1.0, yes(stm.get("ugv_disabled_near_fire") is True))
    delay = num(stm.get("steam_off_delay_s"))
    add("steam", "valve_stops_steam", 1.5, lin(abs(delay - spec["steam_stop_s"]) if delay is not None else None, spec["steam_stop_tol_full"], spec["steam_stop_tol_zero"]),
        dict(turned=stm.get("handle_turned_deg"), t_closed=stm.get("t_closed"), delay=delay))
    closed = stm.get("steam_after_close") is False
    add("steam", "plume_harmless_when_off", 0.8, yes(closed and stm.get("uav3_disabled_after_close") is False))
    add("steam", "disabled_stays", 0.4, yes(stm.get("uav1_still_disabled") is True))
    add("steam", "reopen_restores", 0.6, yes(closed and stm.get("steam_back_on") is True and nz(stm.get("handle_reopened_deg"), 99) < 15))
    add("steam", "plume_dangerous_again", 0.4, yes(closed and stm.get("steam_back_on") is True and stm.get("uav3_disabled_after_reopen") is True))

    # ---- the hoist: the bucket on its ropes under the grader's own moves
    ho, th = got.get("hoist", {}) or {}, truth["groups"].get("hoist", {}) or {}
    ropes = ho.get("ropes") or {}
    ok_struct = all(((ropes.get(n) or {}).get("links") or 0) >= spec["rope_links_min"] and (ropes.get(n) or {}).get("hook_body")
                    and (ropes.get(n) or {}).get("ring_local") for n in UAVS)
    add("hoist", "ropes_are_chains", 0.5, yes(ok_struct), {n: (ropes.get(n) or {}).get("links") for n in UAVS})
    rm = ho.get("rope_mass") or {}
    exp_m = spec["rope_kg_per_m"] * spec["rope_length_m"]
    add("hoist", "rope_mass", 0.6, mean(lin(abs(num(rm.get(n)) - exp_m) / exp_m if num(rm.get(n)) is not None else None, spec["rope_mass_tol_full"], spec["rope_mass_tol_zero"]) for n in UAVS), rm)
    add("hoist", "rings_spread", 0.3, yes(nz(ho.get("ring_min_gap_deg"), 0) >= spec["bail_min_gap_deg"]), ho.get("ring_min_gap_deg"))

    def rope_vals(block, key):
        return [num((((block or {}).get("ropes") or {}).get(n) or {}).get(key)) for n in UAVS]

    sl, tsl = ho.get("slack") or {}, th.get("slack") or {}
    devs, tdevs = rope_vals(sl, "dev"), rope_vals(tsl, "dev")
    sag = [lin(abs(d - t) / t if (d is not None and t) else None, spec["sag_tol_full"], spec["sag_tol_zero"]) for d, t in zip(devs, tdevs)]
    add("hoist", "slack_sags", 1.2, mean(sag) * (1.0 if nz(sl.get("bucket_moved"), 9) < 0.08 else 0.5), dict(dev=devs, truth=tdevs, bucket_moved=sl.get("bucket_moved")))
    lf = (ho.get("lifted") or {}).get("bucket") or {}
    add("hoist", "lifted_level", 1.0, (lin(spec["lift_min_bottom_m"] - nz(lf.get("bottom_z"), 0), 0.0, spec["lift_min_bottom_m"])
                                       + lin(lf.get("tilt_deg"), spec["lift_max_tilt_deg"], 4 * spec["lift_max_tilt_deg"])) / 2 if lf else 0.0, lf)
    chords = rope_vals(ho.get("lifted"), "chord")
    add("hoist", "taut_length", 1.0, mean(lin(abs(c - spec["rope_length_m"]) if c is not None else None, spec["rope_len_tol_full"], spec["rope_len_tol_zero"]) for c in chords), chords)
    devl = rope_vals(ho.get("lifted"), "dev")
    add("hoist", "taut_straight", 0.4, mean(lin(d, 0.06, 0.15) for d in devl), devl)  # (a taut wire rope still sags a little under its own weight)
    hg, thg = ho.get("hang") or {}, th.get("hang") or {}
    add("hoist", "hang_geometry", 1.0, (lin(abs(nz(hg.get("shift_m"), 9) - nz(thg.get("shift_m"), 0)), spec["hang_tol_full"], spec["hang_tol_zero"])
                                        + lin(abs(nz((hg.get("bucket") or {}).get("tilt_deg"), 99) - nz((thg.get("bucket") or {}).get("tilt_deg"), 0)), 4.0, 12.0)) / 2,
        dict(shift=hg.get("shift_m"), truth=thg.get("shift_m"), tilt=(hg.get("bucket") or {}).get("tilt_deg"), truth_tilt=(thg.get("bucket") or {}).get("tilt_deg")))
    sw, tsw = ho.get("swing") or {}, th.get("swing") or {}
    T, tT = num(sw.get("period_s")), num(tsw.get("period_s"))
    add("hoist", "swing_period", 1.2, lin(abs(T - tT) / tT if (T is not None and tT) else None, spec["period_tol_full"], spec["period_tol_zero"]), dict(period=T, truth=tT))
    dc = num(sw.get("decay"))
    add("hoist", "swing_decay", 0.6, yes(dc is not None and spec["decay_min"] <= dc <= spec["decay_max"]), dict(decay=dc, truth=tsw.get("decay")))
    tp, ttp = (ho.get("tip") or {}).get("bucket") or {}, (th.get("tip") or {}).get("bucket") or {}
    add("hoist", "tip_angle", 0.8, lin(abs(nz(tp.get("tilt_deg"), 99) - nz(ttp.get("tilt_deg"), 0)), spec["tip_tol_full_deg"], spec["tip_tol_zero_deg"]),
        dict(tilt=tp.get("tilt_deg"), truth=ttp.get("tilt_deg")))
    op = ho.get("over_pan") or {}
    add("hoist", "burns_until_poured", 0.4, yes((op.get("fire") is True and op.get("pouring") is not True) or (op.get("fire") is False and op.get("tipped_in_zone") is True)), op)
    pr = ho.get("pour") or {}
    fod = num(pr.get("fire_out_delay_s"))
    add("hoist", "pour_puts_fire_out", 1.2, lin(fod, spec["fire_out_s"], 2 * spec["fire_out_s"]) if fod is not None else 0.0, dict(t_tilt=pr.get("t_tilt"), t_out=pr.get("t_fire_out")))
    add("hoist", "pan_harmless_when_out", 0.6, yes(ho.get("fire_end") is False and ho.get("ugv_disabled_after_fire_out") is False))
    add("hoist", "stays_out", 0.3, yes(ho.get("fire_end") is False))
    ab, stt = ho.get("abrupt") or {}, ho.get("stats") or {}
    add("hoist", "dead_stop", 0.8, (lin(ab.get("max_speed"), spec["stability_speed_max"], 2 * spec["stability_speed_max"])
                                    + lin(stt.get("stretch"), spec["stretch_max"], 3 * spec["stretch_max"])) / 2 * yes(stt.get("nan") is False),
        dict(max_speed=ab.get("max_speed"), stretch=stt.get("stretch")))
    sd = ho.get("set_down") or {}
    sdb = sd.get("bucket") or {}
    slack = rope_vals(sd, "chord")
    add("hoist", "set_down", 0.6, (lin(sdb.get("bottom_z"), 0.02, 0.15) + lin(sdb.get("tilt_deg"), 5.0, 20.0)
                                   + yes(all(c is not None and c <= spec["rope_length_m"] - 0.1 for c in slack))) / 3 if sdb else 0.0, dict(bucket=sdb, chords=slack))

    # ---- the bay mission in the real bay, driven by the agent's controller, on each day of the fixed set: outcomes only
    eps = got.get("control") or []
    eps = eps if isinstance(eps, list) else [eps]
    cs = cfg["control"]["score"]
    # a robot that does not keep its library models' colliders (the scan's check) scores 0 on no collisions
    honest = {n: ((scan["facts"].get("robots") or {}).get(n) or {}).get("colliders") or {} for n in ("UAV_1", "UAV_2", "UAV_3", "UGV")}
    dishonest = sorted(n for n, c in honest.items() if c and not c.get("ok", True))

    def episode_checks(co):
        """One episode's checks: (name, weight, score, detail). The details are logged: none carries a value of the day
        (the water, or the lever's angle, which is the day's while nothing turns it)."""
        co = co or {}
        aborted = bool(co.get("abort")) or "error" in co or not co
        ok = 0.0 if aborted else 1.0
        lifted = co.get("liftoff_t") is not None
        n_parts = max(1, int(nz(co.get("n_parts"), 6)))
        lost = co.get("lost") or {}
        done_t = num(co.get("done_t"))
        out = [
            ("controller_ran", 0.5, yes(not aborted), co.get("abort") or (co.get("error") or "")[-300:] or None),
            ("steam_released", 1.0, ok * yes(co.get("steam_end") is False), dict(closed_t=co.get("lever_closed_t"), off_t=co.get("steam_off_t"))),
            ("fire_out", 1.5, ok * yes(co.get("fire_end") is False), dict(t=co.get("fire_out_t"), poured=co.get("poured_l"))),
            ("station_cleared", 1.5, ok * min(1.0, nz(co.get("parts_in_crate"), 0) / n_parts), co.get("parts")),
            ("in_time", 1.0, ok * (lin(done_t, cs["time_full_s"], cs["time_zero_s"]) if done_t is not None else 0.0), dict(done_t=done_t, end=co.get("end"))),
            ("liftoff_level", 0.6, ok * lin(co.get("liftoff_tilt_max"), cs["liftoff_tilt_full_deg"], cs["liftoff_tilt_zero_deg"]) if lifted else 0.0,
             dict(t=co.get("liftoff_t"), tilt_max=co.get("liftoff_tilt_max"))),
            ("swing", 0.6, ok * lin(co.get("swing_max"), cs["swing_full_m"], cs["swing_zero_m"]) if lifted else 0.0, co.get("swing_max")),
            ("water_kept", 1.0, ok * lin(co.get("spilt_l"), cs["spill_full_l"], cs["spill_zero_l"]) if lifted else 0.0,
             dict(spilt=co.get("spilt_l"), poured=co.get("poured_l"))),
            ("parts_not_dropped", 0.4, ok * lin(nz(co.get("parts_dropped"), n_parts), 0.0, 2.0), co.get("parts_dropped")),
            ("no_losses", 1.0, ok * yes(not lost and not co.get("crash") and not co.get("dropped")),
             dict(lost=lost, crash=co.get("crash"), dropped=co.get("dropped"))),
            ("no_collisions", 1.0, ok * yes(not dishonest) * lin(nz(co.get("n_collisions"), 99), 0.0, cs["collisions_zero"]),
             dict(contacts=(co.get("collisions") or [])[:4], colliders_not_kept=dishonest) if dishonest else (co.get("collisions") or [])[:4]),
            ("separation", 0.6, ok * lin(cs["sep_full_m"] - nz(co.get("min_sep"), 0.0), 0.0, cs["sep_full_m"] - cs["sep_zero_m"]), co.get("min_sep")),
            ("steady_end", 0.4, ok * yes(co.get("end") == "done"), dict(end=co.get("end"), t=co.get("t_end"))),
        ]
        return out

    per_ep = [episode_checks(co) for co in eps] or [episode_checks({})]

    def mission_section(gated):
        """The mission's checks over the missions' runs: (checks as (name, weight, score, detail), section score, progress).
        With the progress gate the checks a run earns without doing its jobs count only as far as it did them: the
        bucket's (BUCKET_CHECKS) in a run that put the fire out, the safety checks (SAFETY_CHECKS) in proportion to the
        run's progress, consistency in proportion to the runs' mean progress."""
        progress = []
        runs = []
        for lst in per_ep:
            v = {c[0]: c[2] for c in lst}
            prog = (v["steam_released"] + v["fire_out"] + v["station_cleared"]) / 3.0
            progress.append(prog)
            factor = dict({n: v["fire_out"] for n in BUCKET_CHECKS}, **{n: prog for n in SAFETY_CHECKS}) if gated else {}
            runs.append([(c[0], c[1], c[2] * factor.get(c[0], 1.0), c[3]) for c in lst])
        totals = []
        for lst in runs:
            w = sum(c[1] for c in lst)
            totals.append(sum(c[1] * c[2] for c in lst) / w if w else 0.0)
        checks_ = []
        for i, (name, w, _, _) in enumerate(runs[0]):
            vals = [lst[i][2] for lst in runs]
            checks_.append((name, w, sum(vals) / len(vals), [lst[i][3] for lst in runs]))
        spread = (max(totals) - min(totals)) if len(totals) > 1 else 1.0
        mean_prog = sum(progress) / len(progress)
        checks_.append(("consistency", 1.0, lin(spread, cs["consistency_full"], cs["consistency_zero"]) * yes(len(totals) > 1)
                        * (mean_prog if gated else 1.0), dict(episodes=[round(t, 4) for t in totals], progress=[round(p, 4) for p in progress])))
        wsum = sum(c[1] for c in checks_)
        return checks_, (sum(c[1] * max(0.0, min(1.0, c[2])) for c in checks_) / wsum if wsum else 0.0), mean_prog

    gate = bool(cfg.get("progress_gate", False))
    ctl_checks, _, _ = mission_section(gate)
    for name, w, v, detail in ctl_checks:
        add("control", name, w, v, detail)
    control_alt = {g: mission_section(g)[1] for g in (False, True)}
    progress_mean = mission_section(False)[2]

    # ---- the fleet's design against the catalog and the requirements
    fl = (got.get("fleet", {}) or {}).get("robots", {}) or {}
    reach = (got.get("fleet", {}) or {}).get("reach") or {}
    scan_robots = scan["facts"].get("robots", {})
    sensors_cat, air_cat, base_cat, arm_cat = cat["sensors"], cat["airframes"], cat["bases"], cat["arms"]

    def honest_cameras(r):
        """The robot's cameras whose intrinsics match their declared catalog model (others do not count)."""
        ok_, bad = [], 0
        for c in r.get("sensors", []):
            m = sensors_cat.get(c.get("model"))
            if not c.get("camera") or not m:
                continue
            if m["type"] == "lidar" or m["focal"] is None:
                bad += 1
                continue
            f, h, v = num(c.get("focal")), num(c.get("hap")), num(c.get("vap"))
            if f and h and v and abs(f - m["focal"]) <= 0.02 * m["focal"] and abs(h - m["hap"]) <= 0.02 * m["hap"] and abs(v - m["vap"]) <= 0.02 * m["vap"]:
                ok_.append((c, m))
            else:
                bad += 1
        return ok_, bad

    def sensor_mass(r):
        return sum(sensors_cat[c["model"]]["mass"] for c in r.get("sensors", []) if c.get("model") in sensors_cat)

    def has_lidar(r):
        return any(c.get("model") in sensors_cat and sensors_cat[c["model"]]["type"] == "lidar" for c in r.get("sensors", []))

    def identity(name, model, table):
        entry = table.get(model or "")
        if not entry:
            return 0.0
        refs = scan_robots.get(name, {}).get("assets", [])
        return yes(any(entry["asset"] in ref for ref in refs))

    rope_total = sum(nz(v, 0.0) for v in rm.values()) if rm else 3 * exp_m
    share = (spec["bucket_kg"] + rope_total) / 3.0 * spec["tether_share_factor"]
    for name in UAVS:
        r = fl.get(name) or {}
        air = air_cat.get(r.get("airframe") or "")
        arm = arm_cat.get(r.get("arm") or "")
        add("fleet", f"{name}.airframe_declared", 0.3, yes(bool(air)) * (0.5 + 0.5 * identity(name, r.get("airframe"), air_cat)), r.get("airframe"))
        add("fleet", f"{name}.arm", 0.4, yes(bool(arm)) * (0.5 + 0.5 * identity(name, r.get("arm"), arm_cat)) * yes((r.get("dofs") or 0) >= 4),
            dict(arm=r.get("arm"), dofs=r.get("dofs")))
        total = num(r.get("mass"))
        ok_mass = False
        if air and arm and total is not None:
            floor_ = (air["empty"] + arm["mass"] + sensor_mass(r)) * 0.97
            ok_mass = total >= floor_ and total + share <= air["mtow"]
        add("fleet", f"{name}.mass_budget", 0.8, yes(ok_mass), dict(total=total, share=round(share, 3), mtow=air and air["mtow"]))
        cams, bad = honest_cameras(r)
        add("fleet", f"{name}.sensors_honest", 0.3, yes(bool(cams) and bad == 0), dict(ok=len(cams), bad=bad))
        best = 0.0
        for c, m in cams:
            v = c.get("pan_from_approach") or {}
            if m["type"] == "thermal" and v.get("in_view") and c.get("hfov_deg"):
                px = m["res"][0] * (num(v.get("ang_size_deg")) or 0.0) / c["hfov_deg"]
                best = max(best, lin(spec["pan_px_min"] - px, 0.0, spec["pan_px_min"] * 0.5))
        add("fleet", f"{name}.thermal_sees_pan", 0.8, best)
        add("fleet", f"{name}.sees_others", 0.5, yes(has_lidar(r)), [c.get("model") for c in r.get("sensors", []) if not c.get("camera")])
    r = fl.get("UGV") or {}
    base = base_cat.get(r.get("base") or "")
    arm = arm_cat.get(r.get("arm") or "")
    add("fleet", "UGV.base_declared", 0.3, yes(bool(base)) * (0.5 + 0.5 * identity("UGV", r.get("base"), base_cat)), r.get("base"))
    add("fleet", "UGV.arm_declared", 0.3, yes(bool(arm)) * (0.5 + 0.5 * identity("UGV", r.get("arm"), arm_cat)), r.get("arm"))
    add("fleet", "UGV.payload_budget", 0.5, yes(bool(base) and bool(arm) and arm["mass"] + sensor_mass(r) <= base["payload"]),
        dict(arm=arm and arm["mass"], sensors=round(sensor_mass(r), 3), payload=base and base["payload"]))
    add("fleet", "UGV.valve_torque", 0.5, yes(bool(arm) and arm["payload"] * spec["g"] * spec["lever_m"] >= spec["valve_torque_Nm"]),
        dict(payload=arm and arm["payload"]))
    cams, bad = honest_cameras(r)
    best = 0.0
    for c, m in cams:
        v = c.get("dial_from_bay") or {}
        if v.get("in_view") and v.get("in_front") and c.get("hfov_deg"):
            px = m["res"][0] * (num(v.get("ang_size_deg")) or 0.0) / c["hfov_deg"]
            best = max(best, lin(spec["dial_px_min"] - px, 0.0, spec["dial_px_min"] * 0.5))
    add("fleet", "UGV.reads_gauge", 1.0, best * yes(bad == 0), dict(ok=len(cams), bad=bad))
    add("fleet", "UGV.reaches_valve", 0.8, lin(reach.get("min_dist_to_grip"), spec["reach_tol_m"], 3 * spec["reach_tol_m"]), reach.get("min_dist_to_grip"))

    checks, section_scores, exact = [], {}, {}
    for sec in cfg["sections"]:
        lst = sections.get(sec, [])
        sw_ = sum(c["weight"] for c in lst)
        exact[sec] = sum(c["weight"] * c["score"] for c in lst) / sw_ if sw_ else 0.0
        section_scores[sec] = round(exact[sec], 4)
        checks += lst
    combos = {}
    for gated, ctl in control_alt.items():
        secs = dict(exact, control=ctl)
        for name in combinations(cfg):
            combos[(name, gated)] = combine(secs, cfg, name)
    total = combos[(cfg.get("combine", "weighted"), gate)]
    extra = {f"acceptance.{name}{'.progress_gate' if gated else ''}": round(v, 6) for (name, gated), v in combos.items()}
    extra.update({"section.control.progress_gate": round(control_alt[True], 4), "section.control.no_progress_gate": round(control_alt[False], 4),
                  "control.progress_mean": round(progress_mean, 4), "colliders_not_kept": float(len(dishonest))})
    return total, checks, section_scores, extra


# With the progress gate, the mission's checks a run earns without doing its jobs count only as far as it did them: the
# bucket's three (lifting the bucket and hovering earns them) only in a run that put the fire out, the four safety
# checks (doing nothing earns them) in proportion to the run's progress, consistency to the missions' mean progress. A
# run that does every job loses nothing by the gate.
BUCKET_CHECKS = ("liftoff_level", "swing", "water_kept")
SAFETY_CHECKS = ("parts_not_dropped", "no_losses", "no_collisions", "separation")
SCENE_SECTIONS = ("layout", "physics", "steam", "hoist", "fleet")


def combinations(cfg):
    """The ways the sections can be combined: weighted (`sections`), the mission as a gate on the scene's sections, and
    every other weights block `sections_<name>` by its name."""
    return ["weighted", "mission_gate"] + sorted(k[len("sections_"):] for k in cfg if k.startswith("sections_") and isinstance(cfg[k], dict))


def combine(secs, cfg, name):
    """One acceptance score from the section scores: "weighted" (the `sections` weights), "mission_gate" (the weighted
    mean of the five scene sections, by their `sections` weights, times the mission section) or another block's name."""
    if name == "mission_gate":
        w = cfg["sections"]
        wsum = sum(w[k] for k in SCENE_SECTIONS)
        return sum(w[k] * secs.get(k, 0.0) for k in SCENE_SECTIONS) / wsum * secs.get("control", 0.0) if wsum else 0.0
    w = cfg["sections"] if name == "weighted" else cfg[f"sections_{name}"]
    return sum(float(v) * secs.get(k, 0.0) for k, v in w.items())


def anchor_for(config, anchor, meta):
    """The anchor and full_at of the combination in use: anchor.json records the reference under each (`by_combine`,
    keyed `<combine>[.progress_gate]` as the metrics are; the top-level pair is the default's), so that switching
    `combine` or `progress_gate` in grader_config.json needs no new anchor."""
    full_at = meta.get("full_at", 1.0)
    key = f"{config.get('combine', 'weighted')}{'.progress_gate' if config.get('progress_gate') else ''}"
    alt = (meta.get("by_combine") or {}).get(key)
    if anchor is not None and alt:
        return float(alt["value"]), float(alt["full_at"])
    return anchor, full_at


# --------------------------------------------------------------------------------------------- main
def main():
    config, (anchor, meta) = stage_config(), read_anchor()
    truth = json.load(open(os.path.join(HERE, "truth.json")))
    fixed = load_fixed_set(part_names=truth["groups"]["settle"]["items"])  # (before anything of the submission is read)
    sub = os.path.join(os.environ.get("ALE_HOME", "/home/user"), "submission")
    scene = next((os.path.join(sub, n) for n in ("scene.usda", "scene.usd", "scene.usdc")
                  if os.path.isfile(os.path.join(sub, n)) and not os.path.islink(os.path.join(sub, n))), None)
    if scene is None:
        write_verdict(zero_verdict(reason="no scene in /home/user/submission", anchored=anchor is not None))
        return
    work = tempfile.mkdtemp(prefix="arb_verify_", dir="/var/tmp")
    os.chmod(work, 0o755)
    try:
        snap = lockdown(sub, work, int(config.get("max_submission_bytes", 2 << 30)))
    except Reject as exc:
        write_verdict(zero_verdict(reason=str(exc), anchored=anchor is not None))
        return
    scene = os.path.join(snap, os.path.basename(scene))
    os.environ["PATH"] = "/usr/sbin:/usr/bin:/bin:" + os.environ.get("PATH", "")  # (system tools first; the kit runs `runuser` by its absolute path)
    probe_account(work)
    kit_folders_to_probe()
    for name in PROBE_FILES:
        stage_file(work, name)
    bay_layer = stage_file(work, os.path.join("hidden", "bay_true.usda"))
    privileged, mismatch = token_state(snap)
    cfg_probe = json.load(open(os.path.join(HERE, "grader_config.json")))
    cfg_probe.update(bay=bay_from_truth(truth), bay_facts=json.load(open(os.path.join(HIDDEN, "bay_facts.json"))), bay_layer=bay_layer,
                     missions=[m["day"] for m in fixed["missions"]])  # (the days, for the probe alone: its own 0600 copy)
    if privileged:  # (only then: the key is absent otherwise, never false)
        cfg_probe["privileged"] = True
    write_probe_config(work, cfg_probe)
    log("privileged mode" if privileged else ("oracle token mismatch" if mismatch else "no oracle token"))
    canary(work, int(config.get("canary_timeout_s", 900)), bay_layer=bay_layer)
    catalog = write_probe_config(work, config["catalog"], "catalog.json")  # (the scan's colliders rule: the declared models' assets)
    try:
        r = as_probe([os.path.join(probe_dir(work), "scan.py"), scene, "--json", "--catalog", catalog], work, int(config.get("scan_timeout_s", 300)))
        scan = json.loads(r.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        scan = dict(ok=False, errors=["the scan did not finish in time"], facts={})
    except Exception:  # noqa: BLE001
        scan = dict(ok=False, errors=[f"scan crashed: {(r.stderr or '')[-500:]}"], facts={})
    log("scan:", "ok" if scan["ok"] else scan["errors"][:5])
    extra = {"scan_ok": float(scan["ok"]), "privileged_mode": float(privileged), "oracle_token_mismatch": float(mismatch)}
    cap = config.get("scoring_cap", 1.5)
    anchor, full_at = anchor_for(config, anchor, meta)
    extra.update({"anchor_value": float(anchor or 0.0), "anchor_full_at": float(full_at)})
    if not scan["ok"]:
        write_verdict(ratio_rewards(0.0, anchor, cap=cap, metric_name=meta["metric"], higher_is_better=True, full_at=full_at, extra=extra))
        return
    got = run_probes(scene, work, int(config.get("probe_timeout_s", 900)), int(config.get("probe_retries", 1)), fixed, config, snap,
                     cfg_probe=cfg_probe, truth=truth)
    measured, checks, section_scores, combos = score(scan, got, truth, config)
    for c in checks:
        log(f"{c['name']:34s} w={c['weight']:.2f} s={c['score']:.3f}  {json.dumps(c['detail'])[:110] if c['detail'] is not None else ''}")
    log("sections", section_scores)
    extra.update({f"check.{c['name']}": c["score"] for c in checks})
    extra.update({f"section.{k}": v for k, v in section_scores.items()})
    extra.update(combos)
    log("acceptance under each combination:", json.dumps({k: v for k, v in combos.items() if k.startswith("acceptance.")}))
    extra["probe_errors"] = float(sum(1 for g in got.values() for d in (g if isinstance(g, list) else [g]) if "error" in (d or {})))
    runs = [ep or {} for ep in (got.get("control") or [])]
    locks = [ep.get("lock") or {} for ep in runs]
    if locks:
        extra["probes_total"] = float(sum(nz(lk.get("probes_total"), 0) for lk in locks))
        extra["probes_passed"] = float(sum(nz(lk.get("probes_passed"), 0) for lk in locks))
        extra["match_lock_ok"] = float(all(nz(lk.get("match_lock_ok"), 0) >= 1.0 for lk in locks))
    if runs:  # the program's network: the guard that held (1: the seccomp filter, 0: detection) and the runs it aborted
        extra["egress_guard_seccomp"] = float(all(ep.get("egress_mode") == "seccomp" for ep in runs))
        extra["egress_hits"] = float(sum(nz(ep.get("egress_hits"), 0) for ep in runs))
    log("acceptance_score", round(measured, 4))
    write_verdict(ratio_rewards(measured, anchor, cap=cap, metric_name=meta["metric"], higher_is_better=True, full_at=full_at, extra=extra))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        fail("the grader crashed: " + traceback.format_exc()[-1200:])
