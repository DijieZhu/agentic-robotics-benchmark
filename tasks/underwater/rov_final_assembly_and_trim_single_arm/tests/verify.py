"""Grader of rov_final_assembly_and_trim_single_arm (an underwater vehicle finished and trimmed on a turnover positioner by
one arm with a wide gripper, from photos, drawings, a trim sheet, a work order and a weighing station): runs as ROOT in
the verify stage.

Sources (the structure copied, then changed for this cell; recorded in the dev folder's records/P2_P15.md):
  docs/plan/driving/tasks/racecar_full_assembly_tenth_scale_ur10e/tests/verify.py
      sha256 b8dc39d68ef62fd8dab3fac935522e26b1023d931194536dd0dc4ecc3d82ac52 (the lockdown, the givens check, the kit's
      fetch, netguard, the contract, the token, the episode runner, the verdict)
  tasks/quadruped/quadruped_full_assembly_12dof_ur10e/tests/verify.py (arb-main-20261113)
      sha256 4ec1fc64e7292ff5f32369e3ec157c116f8e7f8e11c915ff3ee81ec763493225

The agent left /home/user/submission/assemble.py (solve(cell): a program that reads the run's work order, weighs the
vehicle on the positioner's load cells, works out the ballast from the trim sheet, fits it at keel_up, turns the vehicle
and fits the electronics enclosure, the four vertical thrusters and the two guards with the cell's arm). Verification is
separate (task.toml): this runs in a fresh verifier sandbox from the task image, into which the engine restored only
the submission (root-owned). This grader:
  0. FIRST, stdlib only (`_early_lockdown`): records the upload time, freezes and kills every process of the agent
     account and keeps it stopped until the verdict is written (`verify_freeze`), locks the stage, snapshots the
     submission root-owned, gates late writes, quarantines late entries elsewhere;
  1. checks the root-owned copy of the givens at /opt/arb against given_hashes.json (tests/test.sh builds it from the
     image's task assets and tests/setup_payload/; a mismatch is a task error), reads the fixed hidden set
     (hidden/episodes.json, format rov-asm-set/1: four pre-registered episodes E1..E4, each with its layout seed, its
     dome end, its work order, its vehicle's per-unit deviation and its layout digest; a malformed set is a task error
     whose message names no value of it), reads the trim table (trim_table.json: the truth of the trim; the grader
     configuration's score block must equal it), then unpacks the cell's hidden kit (gt/: kit.json + meshes/, never in
     the repository) into this root-only stage folder (`fetch_hidden_gt`): from the local development copy
     assets/gt.tar.xz when it is present (an engine asset root), else downloaded (grader_config.json "hidden_gt");
     either way its sha256 and its file count must be the pinned ones;
  2. checks that netguard.py holds in this sandbox (selftest; a task error otherwise) and puts a root-owned copy where
     the runner account can read it: every agent process (the import check and the episodes) starts through it, so
     the program and everything it starts cannot create an AF_INET / AF_INET6 socket; the wire to the cell is a pipe pair;
  3. gates: no assemble.py -> the honest zero (`zero_verdict`); contract_ok = assemble.py is a regular file that parses
     and defines solve, the route gate (no weight files, no training calls, size cap), the network audit (static: no
     networking import, no computed import, no started program; runtime: no outbound network activity while the
     program runs), the module imports under the run regime, >= 1 step call, no late write, no surviving process;
     per episode `hard_ok` (H1: a robot link other than the gripper's four pads > 30 N on one body; H2: a unit pressed
     on the environment or another unit with a contact force > 150 N for 0.2 s) zeroes that episode; the dome rule
     and the scrap rules only scrap (the unit counts nothing);
  4. the reference program's token: submission/.oracle_token whose sha256 equals hidden/oracle_token.sha256 switches
     the cells to the privileged mode (`reference_assemble`; stat privileged_mode); a wrong token is ignored (stat
     oracle_token_mismatch); the privileged mode plays the same four hidden episodes (oracle_nominal_layout false);
  5. runs the program once per hidden episode on the cell (`arm.assembly_rov.server.RovInfoServer`) as `nobody` from a
     read-only copy, launched by `arm.assembly_rov.cell_client_ext` (the station's proxy: the camera cell's calls +
     fixture_index + command's force); at every reset the layout the cell built must have the hidden set's layout
     digest (a mismatch is a task error);
  6. criterion (DESIGN v1.2 section 3.1): assembly_score = contract_ok x (1/4) sum over the episodes of stage x hard_ok,
     stage = (the counted structural mates' weight + 4.5 x trim) / 15.5; trim = 0.4 [net buoyancy in window] + 0.3 [and
     pitch in window] + 0.3 [and roll in window] of the delivered vehicle, from the trim table, this vehicle's
     deviation, the work order's water (TEOS-10) and the ballast that is fastened at the end; every statistic goes to the
     metrics, finite numbers only;
  7. (a record only) every agent episode's replay trace and summary under /opt/arb-verify/out (root-only).
Untouched sandbox: no assemble.py -> zero_verdict -> 0. Reference: the token + `reference_assemble` -> 7 / 7 mates and
trim 1.0 on every episode -> 1.
"""

import ast
import json
import math
import os
import pwd
import re
import shutil
import stat as statmod
import subprocess
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True

STAGE_DIR = os.environ.get("ALE_STAGE_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STAGE_DIR)

HOME = os.environ.get("ALE_HOME", "/home/user")
AGENT_USER = os.path.basename(HOME.rstrip("/")) or "user"
SUBMISSION = os.path.join(HOME, "submission")
ROOT_PAYLOAD = os.environ.get("ARB_ROOT_PAYLOAD", "/opt/arb")
ROOT_GIVEN = os.path.join(ROOT_PAYLOAD, "given")
RUNS_ROOT = os.path.join(ROOT_PAYLOAD, "runs")
# the scenery of the agent cameras' renderer (arm/assembly_rov/agent_render.py), from the same root-owned copy
os.environ.setdefault("ARB_AGENT_RENDER_DIR", os.path.join(ROOT_PAYLOAD, "render"))
SNAPSHOT = os.path.join(RUNS_ROOT, "_submission_snapshot")
QUARANTINE = os.path.join(RUNS_ROOT, "_quarantine")
STICKY_DIRS = [d for d in ("/tmp", "/dev/shm", "/var/tmp", "/run/lock") if os.path.isdir(d)]
WATCH_ROOTS = STICKY_DIRS + [HOME]
LATE_GATE_MAX_GAP_S = 600.0

from arm.assembly_cell import runner  # noqa: E402  (stdlib only)
from arm.assembly_six import lockdown  # noqa: E402  (stdlib only)
import verify_freeze  # noqa: E402  (stdlib only)

FREEZER = [None]        # the agent account's verify_freeze.AgentFreezer: started by _early_lockdown, stopped after the verdict


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def _proc_state(pid):
    """The /proc state letter of `pid` ("X" once it is gone)."""
    try:
        with open("/proc/%d/status" % pid) as fh:
            for line in fh:
                if line.startswith("State:"):
                    return line.split()[1]
    except (OSError, IndexError):
        pass
    return "X"


def _still_running(uid, bound_s=3.0):
    """PIDs of the account's live processes that are not stopped, after the freezer has had up to `bound_s` to stop
    them (a respawn that appears meanwhile is frozen by the next pass); empty when every one is stopped."""
    deadline = time.time() + float(bound_s)
    while True:
        FREEZER[0].freeze_now()
        time.sleep(0.05)
        running = [pid for pid in lockdown.live_uid_processes(uid) if _proc_state(pid) not in ("T", "t", "X")]
        if not running or time.time() > deadline:
            return running


def _early_lockdown():
    out = {"agent_processes_killed": 0.0, "agent_processes_left": 0.0, "stage_locked": 0.0, "stage_parent_locked": 0.0,
           "stage_lock_latency_ms": -1.0, "submission_late_writes": 0.0, "late_writes_quarantined": 0.0, "late_write_gate_active": 0.0,
           "submission_snapshot": 0.0, "saved": None, "parent": None, "t_ref": None, "late": [], "reasons": []}
    if os.getuid() != 0:
        return out
    t_ref = lockdown.stage_upload_time(STAGE_DIR)
    out["t_ref"] = t_ref
    try:
        uid = pwd.getpwnam(AGENT_USER).pw_uid
    except KeyError:
        uid = None
    if uid is not None and uid != 0:
        killed, left = lockdown.stop_and_kill_uid(uid, grace_s=8.0)
        out["agent_processes_killed"], out["agent_processes_left"] = float(killed), float(left)
    # the image's supervisord (PID 1) respawns the agent account's desktop within seconds of the kill, and the respawned
    # session runs the account's autostart entries: every process of the account stays SIGSTOPped (polled every 0.1 s; a
    # stopped process does not exit, so it is not respawned) until the verdict is written
    FREEZER[0] = verify_freeze.AgentFreezer.for_user(AGENT_USER).start()
    if out["agent_processes_left"]:
        # in the separate verifier a process that outlives the kill is the image's own (its respawned desktop): it is frozen
        # with the rest of the account; only a process that is still running once the freezer had its turn voids the contract
        running = _still_running(uid)
        if running:
            out["reasons"].append("%d process(es) of the agent account survived the kill at verify start and could not be "
                                  "stopped" % len(running))
        else:
            log("[grader] %d process(es) of the agent account (the image's respawned desktop) outlived the kill at verify "
                "start: frozen until the verdict" % int(out["agent_processes_left"]))
    out["saved"] = runner.lock_stage(STAGE_DIR)
    out["stage_locked"] = 1.0 if out["saved"] is not None else 0.0
    parent = os.path.dirname(os.path.abspath(STAGE_DIR))
    if parent and parent not in ("/", os.path.abspath(HOME)) and os.path.isdir(parent):
        try:
            mode = statmod.S_IMODE(os.lstat(parent).st_mode)
            os.chmod(parent, 0o700)
            saved_mode = os.environ.get("ARB_STAGE_PARENT_MODE")
            out["parent"] = (parent, int(saved_mode, 8) if saved_mode else mode)
            out["stage_parent_locked"] = 1.0
        except (OSError, ValueError):
            pass
    gate_active = t_ref is not None
    cfg_json = os.path.join(STAGE_DIR, "config.json")
    if gate_active and os.path.isfile(cfg_json):
        try:
            gate_active = (os.lstat(cfg_json).st_ctime - t_ref) <= LATE_GATE_MAX_GAP_S
        except OSError:
            pass
    out["late_write_gate_active"] = 1.0 if gate_active else 0.0
    if os.path.isdir(SUBMISSION):
        if gate_active:
            out["late"] = lockdown.late_entries(SUBMISSION, t_ref)
            out["submission_late_writes"] = float(len(out["late"]))
            if out["late"]:
                out["reasons"].append("%d entr%s of the submission written after the agent phase ended (%s)" % (
                    len(out["late"]), "y" if len(out["late"]) == 1 else "ies", ", ".join(e[0] for e in out["late"][:5])))
        os.makedirs(RUNS_ROOT, exist_ok=True)
        os.chmod(RUNS_ROOT, 0o755)
        lockdown.snapshot_dir(SUBMISSION, SNAPSHOT)
        out["submission_snapshot"] = 1.0
    if gate_active:
        q = 0
        for root in WATCH_ROOTS:
            late = lockdown.late_entries(root, t_ref, exclude=(SUBMISSION, RUNS_ROOT, STAGE_DIR))
            if late:
                log("[grader] %d late entr%s under %s quarantined" % (len(late), "y" if len(late) == 1 else "ies", root))
                q += lockdown.quarantine(root, late, QUARANTINE)
        out["late_writes_quarantined"] = float(q)
    out["stage_lock_latency_ms"] = lockdown.process_age_ms()
    return out


EARLY = _early_lockdown() if __name__ == "__main__" else {"saved": None, "parent": None, "reasons": [], "late": [], "t_ref": None}

import hashlib  # noqa: E402

import numpy as np  # noqa: E402

from arm.assembly_cell import wire  # noqa: E402
from arm.assembly_info import netaudit  # noqa: E402
from arm.assembly_rov import fixtures as FX  # noqa: E402
from arm.assembly_rov import trim_eval as TE  # noqa: E402
from arm.assembly_rov.cell import load_kit  # noqa: E402
from arm.assembly_rov.cell_client_ext import launcher_argv  # noqa: E402
from arm.assembly_rov.server import RovInfoServer, check_rov_kit, check_trim_config, merge_config  # noqa: E402
import episode_trace  # noqa: E402  (the maintainers' replay record; never a scoring input)

GT = os.path.join(STAGE_DIR, "gt")                               # the cell's hidden kit: kit.json + meshes/ (fetch_hidden_gt)
LOCAL_GT = os.path.join(STAGE_DIR, "assets", "gt.tar.xz")        # the development copy of the same tarball (engine asset root)
ROBOT_DIR = os.path.join(ROOT_GIVEN, "robot")                    # the given robot models (universal_robots_ur10e/, wide_gripper/)
HIDDEN = os.path.join(STAGE_DIR, "hidden")
TRIM_TABLE = os.path.join(STAGE_DIR, "trim_table.json")
# The maintainers' record (the agent program's stderr, the replay traces, the episode summaries) lives outside /opt/ale: the
# engine deletes its stage folders before it keeps a sandbox, and the record is read from a kept one (the verifier
# sandbox: run with `[sandbox_retention] verifier = "keep"`). Root-only (0700).
RECORD_ROOT = os.environ.get("ARB_RECORD_ROOT", "/opt/arb-verify")
LOGS = os.path.join(RECORD_ROOT, "logs")
OUT = os.path.join(RECORD_ROOT, "out")
CRITERIA = ["assembly_score"]
GATES = ["contract_ok"]
LIB_PACKAGES = ("arm", "common", "render")
SUSPICIOUS = ("/opt/ale", "ALE_STAGE_DIR", "ALE_VERIFICATION", "hidden/episodes", "hidden/selection", "oracle_token", "verify/gt",
              "ALE_VERIFY", "tests/gt", "/opt/ale/tests", ".ale-solution", "verifier/gt", "/opt/arb-verify", "trim_table.json")
# the fixed hidden set (DESIGN v1.2 section 3.5; -dev/review/selection_rule.json "set"): its format, the four strata in
# their order, the two dome ends (two episodes each), the work order's ranges (TEOS-10's) and the seed's range
SET_FORMAT = "rov-asm-set/1"
STRATA = ("E1", "E2", "E3", "E4")
DOME_ENDS = (0, 180)
SEED_LIMIT = 1 << 63
WATER_T_RANGE_C = (-2.0, 40.0)
WATER_SP_RANGE = (0.0, 42.0)
UNIT_RE = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}"


def record_dir(path):
    """`path` under the root-only record root, created on demand; never a scoring input."""
    os.makedirs(RECORD_ROOT, exist_ok=True)
    os.chmod(RECORD_ROOT, 0o700)
    os.makedirs(path, exist_ok=True)
    return path


def program_source():
    return SNAPSHOT if os.getuid() == 0 and os.path.isdir(SNAPSHOT) else SUBMISSION


def fail(reason):
    log("[grader] TASK ERROR: " + reason)
    raise SystemExit(1)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_json(p):
    with open(p) as f:
        return json.load(f)


def root_payload_ok(expected):
    problems = []
    owner = 0 if os.getuid() == 0 else os.getuid()
    if not os.path.isdir(ROOT_PAYLOAD):
        return False, ["%s missing (tests/test.sh builds it)" % ROOT_PAYLOAD]
    for rel, digest in expected.items():
        p = os.path.join(ROOT_PAYLOAD, rel)
        try:
            st = os.lstat(p)
        except OSError:
            problems.append("missing " + rel)
            continue
        if os.path.islink(p) or st.st_uid != owner or (st.st_mode & 0o022):
            problems.append("not a root-owned, group/world-unwritable regular file: " + rel)
            continue
        if sha256_file(p) != digest:
            problems.append("differs " + rel)
    return not problems, problems


# ------------------------------------------------------------------------------------------------ the fixed hidden set
def _finite(v):
    return (not isinstance(v, bool)) and isinstance(v, (int, float)) and math.isfinite(float(v))


def load_hidden_set(cfg, hidden_dir, deviation_range=None):
    """The fixed hidden set (`hidden_dir`/episodes.json, SET_FORMAT): [{seed, stratum, dome_end_deg, work_order,
    deviation, layout_digest}] in the file's order. Exactly cfg["episodes"] episodes; with four, one per stratum in the
    order E1..E4 and each dome end twice; every seed an int in [0, 2**63) and distinct; every dome end 0 or 180; every
    work order {unit, delivery_site {water_temperature_C, salinity_psu}} with finite values in TEOS-10's range; every
    deviation {dm_kg, dx_mm, dy_mm} finite and inside `deviation_range` (the trim table's published range) when given;
    every layout digest 64 lower-case hex digits. Anything else is a task error whose message names no value of the set.
    Stdlib only."""
    path = os.path.join(hidden_dir, "episodes.json")
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        fail("the fixed hidden set (hidden/episodes.json) is missing or does not parse")
    if not isinstance(doc, dict) or doc.get("format") != SET_FORMAT:
        fail("the fixed hidden set is not in the %s format" % SET_FORMAT)
    eps = doc.get("episodes")
    try:
        n = int(cfg.get("episodes", len(STRATA)))
    except (TypeError, ValueError):
        fail("grader_config.json: episodes is not a number")
    if not isinstance(eps, list):
        fail("the fixed hidden set has no list of episodes")
    if len(eps) != n:
        fail("the fixed hidden set holds %d episode(s); grader_config.json expects %d" % (len(eps), n))
    out = []
    for k, e in enumerate(eps, 1):
        if not isinstance(e, dict):
            fail("episode %d of the fixed hidden set is not an object" % k)
        seed, stratum, digest, dome = e.get("seed"), e.get("stratum"), e.get("layout_digest"), e.get("dome_end_deg")
        if type(seed) is not int or not 0 <= seed < SEED_LIMIT:
            fail("episode %d of the fixed hidden set: its seed is not an integer in [0, 2**63)" % k)
        if not isinstance(stratum, str) or stratum not in STRATA:
            fail("episode %d of the fixed hidden set: its stratum is not one of the %d strata" % (k, len(STRATA)))
        if type(dome) is not int or dome not in DOME_ENDS:
            fail("episode %d of the fixed hidden set: its dome end is not 0 or 180" % k)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            fail("episode %d of the fixed hidden set: its layout digest is not 64 lower-case hex digits" % k)
        wo = e.get("work_order")
        site = wo.get("delivery_site") if isinstance(wo, dict) else None
        if not isinstance(site, dict) or not isinstance(wo.get("unit"), str) or not re.fullmatch(UNIT_RE, wo["unit"]):
            fail("episode %d of the fixed hidden set: its work order is not {unit, delivery_site}" % k)
        T, SP = site.get("water_temperature_C"), site.get("salinity_psu")
        if not (_finite(T) and _finite(SP) and WATER_T_RANGE_C[0] <= T <= WATER_T_RANGE_C[1] and WATER_SP_RANGE[0] <= SP <= WATER_SP_RANGE[1]):
            fail("episode %d of the fixed hidden set: its water is not a finite temperature and salinity in range" % k)
        dev = e.get("deviation")
        if not isinstance(dev, dict) or not all(_finite(dev.get(x)) for x in ("dm_kg", "dx_mm", "dy_mm")):
            fail("episode %d of the fixed hidden set: its deviation is not {dm_kg, dx_mm, dy_mm}" % k)
        if deviation_range is not None:
            for key, rk in (("dm_kg", "mass_kg"), ("dx_mm", "cg_x_mm"), ("dy_mm", "cg_y_mm")):
                lo, hi = (float(v) for v in deviation_range[rk])
                if not lo - 1e-9 <= float(dev[key]) <= hi + 1e-9:
                    fail("episode %d of the fixed hidden set: its deviation lies outside the trim table's published range" % k)
        out.append({"seed": seed, "stratum": stratum, "dome_end_deg": dome, "layout_digest": digest,
                    "work_order": {"unit": wo["unit"], "delivery_site": {"water_temperature_C": float(T), "salinity_psu": float(SP)}},
                    "deviation": {"dm_kg": float(dev["dm_kg"]), "dx_mm": float(dev["dx_mm"]), "dy_mm": float(dev["dy_mm"])}})
    if len(set(e["seed"] for e in out)) != len(out):
        fail("the fixed hidden set repeats a seed")
    if len(out) == len(STRATA):
        if [e["stratum"] for e in out] != list(STRATA):
            fail("the fixed hidden set does not hold one episode per stratum in the order %s" % ", ".join(STRATA))
        if sorted(e["dome_end_deg"] for e in out) != sorted(list(DOME_ENDS) * (len(out) // len(DOME_ENDS))):
            fail("the fixed hidden set does not hold each dome end twice")
    return out


def check_layout_digest(k, entry, got, nominal=False):
    """True when episode k's cell has the hidden set's layout (a task error otherwise); False when it was not compared (the
    privileged mode's nominal layout, oracle_nominal_layout true)"""
    if nominal:
        return False
    if got != entry["layout_digest"]:
        fail("episode %d: the cell built a layout that is not the fixed hidden set's (its layout digest differs): the kit, the "
             "layout code, the fixtures or grader_config.json's layout block changed after the set was selected" % k)
    return True


# ------------------------------------------------------------------------------------------------ the cell's hidden kit
def fetch_hidden_gt(cfg, stage_dir=None, local=None):
    """The cell's hidden kit (gt/: kit.json with the bodies, slots and carve keys, meshes/ with the units' visual meshes and
    collision pieces) is not in the repository: one tarball, pinned in grader_config.json ("hidden_gt": its URL at an
    asset-dataset commit, its sha256 and its file count). When the development copy assets/gt.tar.xz is in this stage
    folder (an engine asset root; `local`), that file is used; ARB_HIDDEN_GT_TARBALL (a maintainer's copy) likewise;
    otherwise the tarball is downloaded here at verify start (verification has network; the agent's sandbox never held
    it and the image has no copy). Either way it must have the pinned sha256, every member must be under gt/ and be a
    file or a folder, and gt/ must hold the pinned number of files and kit.json: anything else is a task error (nothing
    is scored without the exact kit). Unpacked into gt/ of the root-only stage folder. Returns (files, seconds, source)."""
    import tarfile
    import urllib.request
    stage_dir = stage_dir or STAGE_DIR
    gt = os.path.join(stage_dir, "gt")
    spec = cfg["hidden_gt"]
    url, want = spec["url"], spec["sha256"]
    tries, backoff = max(1, int(spec.get("tries", 4))), float(spec.get("backoff_s", 5.0))
    t0 = time.time()
    tarball = os.path.join(stage_dir, ".gt.tar.xz")
    part = os.path.join(stage_dir, ".gt.part")
    for p in (part, gt):
        shutil.rmtree(p, ignore_errors=True)
    if os.path.exists(tarball):
        os.remove(tarball)
    local = local if local is not None else os.path.join(stage_dir, "assets", "gt.tar.xz")
    env = os.environ.get("ARB_HIDDEN_GT_TARBALL")
    if env:
        src, source = env, "maintainer copy"
    elif os.path.isfile(local) and not os.path.islink(local):
        src, source = local, "local copy"
    else:
        src, source = tarball, "download"
        for k in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=120) as r, open(tarball, "wb") as fh:
                    for block in iter(lambda: r.read(1 << 20), b""):
                        fh.write(block)
                break
            except Exception as e:  # noqa: BLE001 - retried, then a task error
                log("[grader] the cell's kit: download %d/%d failed: %r" % (k + 1, tries, e))
                if os.path.exists(tarball):
                    os.remove(tarball)
                if k == tries - 1:
                    fail("could not download the cell's kit from %s: %r" % (url, e))
                time.sleep(backoff * (k + 1))
    got = sha256_file(src)
    if got != want:
        fail("the cell's kit (%s) has sha256 %s; grader_config.json pins %s" % (source, got, want))
    os.makedirs(part, mode=0o700)
    with tarfile.open(src, "r:xz") as tf:
        bad = [m.name for m in tf.getmembers() if not (m.isfile() or m.isdir()) or not (m.name == "gt" or m.name.startswith("gt/"))]
        if bad:
            fail("the cell's kit tarball holds unexpected members: %s" % bad[:5])
        tf.extractall(part, filter="data")
    os.rename(os.path.join(part, "gt"), gt)
    os.chmod(gt, 0o700)
    shutil.rmtree(part, ignore_errors=True)
    if os.path.exists(tarball):
        os.remove(tarball)
    n = sum(len(fns) for _dp, _dn, fns in os.walk(gt))
    if n != int(spec["files"]) or not os.path.isfile(os.path.join(gt, "kit.json")):
        fail("the cell's kit unpacked to %d files (kit.json %s); grader_config.json expects %d"
             % (n, os.path.isfile(os.path.join(gt, "kit.json")), int(spec["files"])))
    dt = time.time() - t0
    log("[grader] the cell's kit (%s): %d files, sha256 %s..., in %.1f s" % (source, n, got[:12], dt))
    return n, dt, source


# ------------------------------------------------------------------------------------------------ netguard
def netguard_folder(stage_dir=None):
    """(folder, guard path): netguard.py's selftest as this process (a task error when the filter does not hold here),
    then a root-owned 0755 folder (under /run when writable) with a 0644 copy of it, which the runner account can read
    (the stage folder is 0700)."""
    src = os.path.join(stage_dir or STAGE_DIR, "netguard.py")
    try:
        chk = subprocess.run([sys.executable, "-I", "-S", src, "selftest"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, timeout=60)
        ok, text = chk.returncode == 0, chk.stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        ok, text = False, repr(e)
    if not ok:
        fail("netguard does not hold in this sandbox: %s" % text[-300:])
    base = "/run" if os.path.isdir("/run") and os.access("/run", os.W_OK) else None
    folder = tempfile.mkdtemp(prefix="arb-netguard-", dir=base)
    os.chmod(folder, 0o755)
    guard = os.path.join(folder, "netguard.py")
    shutil.copyfile(src, guard)
    os.chmod(guard, 0o644)
    return folder, guard


def guarded(guard, program_dir, argv):
    """argv started through netguard (no IP sockets for it and everything it starts), cwd = program_dir"""
    if not guard:
        return list(argv)
    return [sys.executable, "-I", "-S", guard, "exec", program_dir] + list(argv)


# ------------------------------------------------------------------------------------------------ the contract
def route_gate(cfg, src):
    rg = cfg.get("route_gate", {})
    forb_ext = tuple(rg.get("forbidden_extensions", []))
    calls = rg.get("forbidden_calls", [])
    total, problems, suspicious = 0, [], 0
    for dp, _, fns in os.walk(src):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                total += os.lstat(p).st_size
            except OSError:
                continue
            if fn.lower().endswith(forb_ext):
                problems.append("weight file %s" % os.path.relpath(p, src))
            if fn.endswith((".py", ".sh")) and not os.path.islink(p):
                try:
                    text = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                for c in calls:
                    if c in text:
                        problems.append("%s uses %s" % (os.path.relpath(p, src), c))
                suspicious += sum(1 for line in text.splitlines() if any(s in line for s in SUSPICIOUS))
    if total > rg.get("max_submission_mb", 60) * 1024 * 1024:
        problems.append("submission larger than %d MB" % rg.get("max_submission_mb", 60))
    return problems, suspicious


def library_digests():
    """sha256 of every .py file of the shipped library (a submission may carry copies of it)."""
    out = set()
    for pkg in LIB_PACKAGES:
        d = os.path.join(ROOT_GIVEN, "lib", pkg)
        for dp, _dn, fns in os.walk(d):
            for fn in fns:
                if fn.endswith(".py"):
                    out.add(sha256_file(os.path.join(dp, fn)))
    return out


class Runner(object):
    def __init__(self, cfg):
        self.acct = runner.runner_account(cfg.get("runner_account", "nobody"), AGENT_USER)
        self.name, self.uid, self.gid = self.acct if self.acct else (None, None, None)
        self.lib_dirs = {p: os.path.join(ROOT_GIVEN, "lib", p) for p in LIB_PACKAGES}
        self.cpu_s = float(cfg.get("cpu_cap_factor", 4.0)) * float(cfg["wall_cap_s"])
        self.threads = int(cfg.get("program_threads", 2))
        self.killed = self.left = self.unclean_runs = self.stray_files = self.symlinks_skipped = 0
        os.makedirs(RUNS_ROOT, exist_ok=True)
        os.chmod(RUNS_ROOT, 0o755)

    def prepare(self, tag, program_dir, private=False):
        root = os.path.join(RUNS_ROOT, tag)
        if os.path.isdir(root):
            shutil.rmtree(root, ignore_errors=True)
        owner = (self.uid, self.gid) if (self.acct and not private) else None
        dirs = runner.prepare_run_dir(root, program_dir, self.lib_dirs, owner=owner, private=private)
        self.symlinks_skipped = max(self.symlinks_skipped, dirs["skipped"])
        return dirs

    def env(self, dirs, extra=None):
        e = runner.child_env(dirs["home"], dirs["tmp"], dirs["lib"], ROOT_GIVEN, threads=self.threads, extra=extra)
        e["LP_NUM_THREADS"] = str(self.threads)
        return e

    def snapshot(self, exclude):
        return {r: runner.snapshot_tree(r, exclude) for r in WATCH_ROOTS}

    def kill_runner_processes(self, grace_s=8.0):
        if not self.acct:
            return 0
        killed, left = lockdown.stop_and_kill_uid(self.uid, grace_s=grace_s)
        self.killed += killed
        self.left += left
        if left:
            self.unclean_runs += 1
        return left

    def cleanup(self, dirs, before):
        self.kill_runner_processes()
        if before is not None and self.acct:
            for r in WATCH_ROOTS:
                after = runner.snapshot_tree(r, (dirs["root"],))
                new, modified = runner.tree_changes(before[r], after, self.uid)
                self.stray_files += len(new) + len(modified)
                runner.sweep_new_entries(r, before[r], after, self.uid, (dirs["root"],))
        shutil.rmtree(dirs["root"], ignore_errors=True)


def submission_missing():
    """the reason when there is no regular file assemble.py in the submission (the honest zero), else None"""
    path = os.path.join(program_source(), "assemble.py")
    if not os.path.isfile(path) or os.path.islink(path):
        return "no assemble.py (regular file) in the submission"
    return None


def contract_check(cfg, rn, guard):
    """(ok, why, suspicious lines, network imports) of the submission's static and import contract"""
    src = program_source()
    path = os.path.join(src, "assemble.py")
    if not os.path.isfile(path) or os.path.islink(path):
        return False, "no assemble.py (regular file) in the submission", 0, []
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError as e:
        return False, "assemble.py does not parse: %s" % e, 0, []
    has_solve = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "solve" for n in ast.walk(tree)) or \
        any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "solve" for t in n.targets) for n in ast.walk(tree)) or \
        any(isinstance(n, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name) == "solve" for a in n.names) for n in ast.walk(tree))
    if not has_solve:
        return False, "assemble.py defines no solve", 0, []
    problems, suspicious = route_gate(cfg, src)
    if problems:
        return False, "route gate: " + "; ".join(problems[:5]), suspicious, []
    net = netaudit.scan_imports(src, allow_digests=library_digests())
    if net:
        return False, "network audit: networking import(s) %s" % ", ".join("%s: %s" % x for x in net[:5]), suspicious, net
    dirs = rn.prepare("import_check", src)
    before = rn.snapshot((dirs["root"],)) if rn.acct else None
    code = ("import sys, importlib; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); "
            "m = importlib.import_module('assemble'); assert callable(m.solve)")
    argv = guarded(guard, dirs["program"], [sys.executable] + runner.PYTHON_FLAGS + ["-c", code, dirs["lib"], dirs["program"]])
    argv = runner.wrap_limits(argv, cpu_s=120, user=rn.name)
    try:
        r = subprocess.run(argv, cwd=dirs["program"], env=rn.env(dirs), capture_output=True, text=True, timeout=120, start_new_session=True)
        ok, why = r.returncode == 0, ("ok" if r.returncode == 0 else "importing assemble.py failed: " + (r.stderr.strip().splitlines() or ["?"])[-1][:300])
    except subprocess.TimeoutExpired:
        ok, why = False, "importing assemble.py took more than 120 s"
    finally:
        rn.cleanup(dirs, before)
    return ok, why, suspicious, []


def token_state():
    """(privileged, mismatch): the submission's .oracle_token against hidden/oracle_token.sha256."""
    p = os.path.join(program_source(), ".oracle_token")
    want_p = os.path.join(HIDDEN, "oracle_token.sha256")
    if not os.path.isfile(p) or os.path.islink(p) or not os.path.isfile(want_p):
        return False, False
    try:
        tok = open(p, "rb").read().strip()
        want = open(want_p).read().strip()
    except OSError:
        return False, False
    got = hashlib.sha256(tok).hexdigest()
    return (got == want), (got != want)


# ------------------------------------------------------------------------------------------------ the formula (pure)
def contract_gate(step_calls, unclean_runs, remote_sockets_seen, net_outbound_events):
    """the run-time part of contract_ok (DESIGN v1.2 section 3.1: >= 1 step call x a clean run x no network); the static
    part (assemble.py, the route gate, the network audit, the import, no late write, no survivor) is decided before"""
    steps_gate = 1.0 if step_calls > 0 else 0.0
    clean_gate = 1.0 if unclean_runs == 0 else 0.0
    net_gate = 1.0 if (remote_sockets_seen <= 0 and net_outbound_events <= 0) else 0.0
    return steps_gate * clean_gate * net_gate, {"steps": steps_gate, "clean": clean_gate, "network": net_gate}


def assembly_score(contract_ok, stages, hard_oks):
    """contract_ok x (1 / n) sum over the episodes of stage x hard_ok, on [0, 1] (DESIGN v1.2 section 3.1)"""
    vals = [float(s) * (1.0 if h else 0.0) for s, h in zip(stages, hard_oks)]
    if len(vals) != len(list(stages)) or not vals:
        return 0.0
    return float(max(0.0, min(1.0, float(contract_ok) * float(np.mean(vals)))))


# ------------------------------------------------------------------------------------------------ one episode
def run_episode(cfg, k, ep, rn, privileged, kit, fx, table, guard, trace_tag=None):
    server = RovInfoServer(GT, ROBOT_DIR, config=cfg, mode="hidden", privileged=privileged, render=True, log=log, kit=kit, fx=fx,
                           table=table)
    r = server.reset(ep)
    checked = check_layout_digest(k, ep, r["layout_digest"], nominal=r["nominal"])
    trace = None
    tcfg = cfg.get("episode_trace") or {}
    if trace_tag and tcfg.get("agent"):
        try:
            trace = episode_trace.EpisodeTrace(server, every_ticks=int(tcfg.get("every_ticks", 4)), log=log)
        except Exception as e:  # noqa: BLE001 - a record only
            log("[grader] episode trace not attached: %r" % (e,))
    record_dir(LOGS)
    wire_error, t0 = None, time.time()
    dirs = rn.prepare("agent_ep%02d" % k, program_source())
    before = rn.snapshot((dirs["root"],)) if rn.acct else None
    argv = guarded(guard, dirs["program"], launcher_argv(sys.executable, dirs["lib"], dirs["program"]))
    # the privileged reference program runs inside one call (reference_assemble): its own wall-clock cap
    wall_cap = float(cfg["privileged_wall_cap_s"] if privileged else cfg["wall_cap_s"])
    proc = None
    sampler = netaudit.SocketSampler(uid=rn.uid if rn.acct else None, pids_fn=None if rn.acct else (lambda: []),
                                     period_s=float(cfg.get("socket_sample_s", 0.5)))
    counters = netaudit.NetCounters()
    try:
        with counters, sampler:
            try:
                proc = wire.AgentProcess(argv, cwd=dirs["program"], user=rn.name, env=rn.env(dirs, None), cpu_s=rn.cpu_s,
                                         stderr_path=os.path.join(LOGS, "agent_ep%02d.log" % k))
                reason = wire.serve_episode(server, proc, wall_cap, float(cfg["silence_s"]), log=log)
            finally:
                if proc is not None:
                    proc.terminate()
                rn.cleanup(dirs, before)             # every process of the runner account is gone before the counters are read
    except Exception:  # noqa: BLE001
        wire_error = traceback.format_exc()
        reason = "wire_error"
        if not server.done:
            server._end(reason)
    if wire_error:
        log("[grader] episode %d: wire error\n%s" % (k, wire_error))
    summary = server.finalize(None)
    summary["episode_wall_s"] = time.time() - t0
    summary["end_reason"] = reason
    summary["layout_digest_checked"] = bool(checked)
    if trace is not None:
        try:
            trace.detach()
            summary["trace_dir"] = trace.save(os.path.join(OUT, "trace"), trace_tag, summary, extra={"episode": int(k), "privileged": bool(privileged)})
        except Exception as e:  # noqa: BLE001 - a record only
            log("[grader] episode trace not written: %r" % (e,))
    summary["remote_sockets_seen"] = int(sampler.max_seen)
    summary["socket_samples"] = int(sampler.samples)
    summary["net_outbound"] = int(counters.total)
    summary["net_delta"] = dict(counters.delta)
    return summary


# ------------------------------------------------------------------------------------------------ the verdict
def finite_metrics(metrics):
    """the finite numbers of `metrics` ('.' and '/' in a name replaced by '_'); everything else is dropped"""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            v = float(v)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out[str(k).replace(".", "_").replace("/", "_")] = v
    return out


def write_verdict(rewards, metrics):
    """One gated `reward` in the benchmark kit's envelope (robotics_grader.ratio_rewards against anchor.json: value 1.0 =
    the reference program, full_at 0.97, so reward = min(assembly_score / 0.97, 1) x the gates); the gate, the criterion
    and every statistic go to `metrics` (finite numbers only). In the engine (`ALE_VERIFY_PYTHON` set) the kit writes
    reward.json and verification.json, the criteria text as the `diagnostic`; elsewhere the envelope goes to
    ROBOTICS_GRADER_VERDICT_PATH, else verdict.local.json in the stage folder."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write_verdict
    overall = float(rewards["assembly_score"])
    anchor, meta = read_anchor()
    extra = finite_metrics(metrics)
    extra.update({k: float(rewards[k]) for k in GATES + CRITERIA})
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "assembly_score"), extra=extra, full_at=meta.get("full_at", 1.0))
    text = ("contract_ok %.0f (the program imports, exposes solve, passes the route gate and the network audit and steps the "
            "cell; no late write, no surviving process); assembly_score %.4f (episode mean of (the counted mates' weight + 4.5 x "
            "trim) / 15.5 on the %d fixed hidden episodes, an episode with a hard safety violation scoring 0)") % (
                rewards["contract_ok"], rewards["assembly_score"], int(metrics.get("episodes", 0)))
    recorded = finite_metrics(verdict["metrics"])
    _kit_write(kit_write_verdict, {"rewards": verdict["rewards"], "metrics": recorded}, text)
    return overall


def write_zero_verdict(reason, metrics):
    """the honest zero (robotics_grader.zero_verdict) when there is nothing to run: no assemble.py"""
    from robotics_grader import read_anchor, zero_verdict
    from robotics_grader import write_verdict as kit_write_verdict
    anchor, _meta = read_anchor()
    extra = finite_metrics(metrics)
    extra.update({"contract_ok": 0.0, "assembly_score": 0.0})
    v = zero_verdict(reason=reason, anchored=anchor is not None, extra=extra)
    _kit_write(kit_write_verdict, v, v.get("diagnostic"))
    return 0.0


def _kit_write(kit_write_verdict, verdict, text):
    if os.environ.get("ALE_VERIFY_PYTHON"):
        kit_write_verdict(verdict, diagnostic=text)
        return
    kit_write_verdict(verdict, path=os.environ.get("ROBOTICS_GRADER_VERDICT_PATH") or os.path.join(STAGE_DIR, "verdict.local.json"),
                      diagnostic=text)


def _restore_modes():
    runner.unlock_stage(EARLY.get("saved"))
    mode = os.environ.get("ARB_STAGE_MODE")
    if mode:
        try:
            os.chmod(STAGE_DIR, int(mode, 8))
        except (OSError, ValueError):
            pass
    parent = EARLY.get("parent")
    if parent:
        try:
            os.chmod(parent[0], parent[1])
        except OSError:
            pass


def main():
    cfg = merge_config(load_json(os.path.join(STAGE_DIR, "grader_config.json")))
    rewards = dict((k, 0.0) for k in GATES + CRITERIA)
    metrics = {"grader_exception": 0.0, "episodes": float(cfg.get("episodes", 4)),
               "root_payload_ok": 0.0, "agent_deprivileged": 0.0, "runner_uid": -1.0, "privileged_mode": 0.0, "oracle_token_mismatch": 0.0,
               "runner_processes_killed": 0.0, "runner_processes_left": 0.0, "runner_unclean_runs": 0.0, "netguard_active": 0.0,
               "stray_files_removed": 0.0, "suspicious_lines": 0.0, "agent_wire_errors": 0.0, "remote_sockets_seen": 0.0, "net_outbound_events": 0.0,
               "network_imports": 0.0, "illegal_call_count": 0.0, "submission_present": 0.0}
    for k in ("stage_locked", "stage_parent_locked", "stage_lock_latency_ms", "agent_processes_killed", "agent_processes_left",
              "submission_late_writes", "late_writes_quarantined", "late_write_gate_active", "submission_snapshot"):
        metrics[k] = float(EARLY.get(k, -1.0 if k == "stage_lock_latency_ms" else 0.0))
    guard_dir = [None]
    try:
        _main_locked(cfg, rewards, metrics, guard_dir)
    finally:
        if guard_dir[0]:
            shutil.rmtree(guard_dir[0], ignore_errors=True)
        _restore_modes()
        if FREEZER[0] is not None:
            FREEZER[0].stop()                       # after the verdict (written by _main_locked); the frozen stay stopped


def _main_locked(cfg, rewards, metrics, guard_dir):
    t_start = time.time()
    expected = load_json(os.path.join(STAGE_DIR, "given_hashes.json"))
    ok_root, problems = root_payload_ok(expected)
    metrics["root_payload_ok"] = 1.0 if ok_root else 0.0
    if not ok_root:
        fail("the root-owned copy of the givens at %s is unusable: %s" % (ROOT_PAYLOAD, "; ".join(problems[:5])))
    try:
        table = TE.TrimTable(TRIM_TABLE)
        check_trim_config(cfg, table)
    except (OSError, ValueError, KeyError) as e:
        fail("the trim table or the grader configuration's score block is unusable: %s" % str(e)[:200])
    episodes = load_hidden_set(cfg, HIDDEN, deviation_range=table.raw["per_unit_deviation"])
    metrics["hidden_set_episodes"] = float(len(episodes))
    missing = submission_missing()
    if missing:
        # the honest zero: nothing to run (the kit is not even fetched)
        _finish_stats(None, metrics, t_start)
        write_zero_verdict(missing, metrics)
        log("[grader] %s: zero verdict" % missing)
        return
    metrics["submission_present"] = 1.0
    gt_files, gt_s, gt_source = fetch_hidden_gt(cfg)
    metrics["hidden_gt_files"] = float(gt_files)
    metrics["hidden_gt_fetch_s"] = float(gt_s)
    metrics["hidden_gt_local"] = 1.0 if gt_source != "download" else 0.0
    guard_dir[0], guard = netguard_folder()
    metrics["netguard_active"] = 1.0
    rn = Runner(cfg)
    if FREEZER[0] is not None and rn.acct and rn.uid == FREEZER[0].uid:
        FREEZER[0].stop()        # no `nobody` in the image: the runner fell back to the agent account, whose programs must run
        log("[grader] runner account = the agent account: the verify-stage freezer is stopped")
    metrics["runner_uid"] = float(rn.uid) if rn.acct else -1.0
    metrics["agent_deprivileged"] = 1.0 if rn.acct else 0.0
    privileged, mismatch = token_state()
    metrics["privileged_mode"] = 1.0 if privileged else 0.0
    metrics["oracle_token_mismatch"] = 1.0 if mismatch else 0.0
    ok, why, suspicious, net = contract_check(cfg, rn, guard)
    if ok and EARLY.get("reasons"):
        ok, why = False, "; ".join(EARLY["reasons"])
    metrics["suspicious_lines"] = float(suspicious)
    metrics["network_imports"] = float(len(net))
    log("[grader] contract_ok=%s (%s); privileged %s; runner %s under netguard; lock latency %.0f ms" % (
        ok, why, privileged, rn.name or "self", metrics["stage_lock_latency_ms"]))
    if not ok:
        rewards["contract_ok"] = 0.0
        _finish_stats(rn, metrics, t_start)
        write_verdict(rewards, metrics)
        return
    kit = load_kit(GT)
    check_rov_kit(kit)
    fx = FX.Fixtures(FX.load_cell_layout(), FX.load_facts(), lmag=bool(cfg["layout"].get("lmag", False)))
    n = len(episodes)
    agent = []
    step_calls = illegal = 0
    for k, ep in enumerate(episodes, 1):
        log("[grader] episode %d/%d (stratum %s, dome %d)" % (k, n, ep["stratum"], ep["dome_end_deg"]))
        s_a = run_episode(cfg, k, ep, rn, privileged, kit, fx, table, guard, trace_tag="agent_ep%02d" % k)
        metrics["episode_traces_written"] = metrics.get("episode_traces_written", 0.0) + (1.0 if s_a.get("trace_dir") else 0.0)
        log("[grader]   stage %.4f (%d / 7 mates, trim %.1f: dB %+.2f N pitch %+.2f roll %+.2f, %d blocks) hard_ok %s sim %.1f s "
            "wall %.1f s frames %d turns %d end %s scrapped %s sustained %.0f N violations %s" % (
                s_a["stage_score"], s_a["n_achieved"], s_a["trim"], s_a["trim_eval"]["dB_N"], s_a["trim_eval"]["theta_deg"],
                s_a["trim_eval"]["phi_deg"], s_a["trim_eval"]["n_ballast"], s_a["hard_ok"], s_a["sim_time_s"], s_a["episode_wall_s"],
                s_a["camera_frames_total"], s_a["turns_used"], s_a["end_reason"], sorted(s_a["scrapped"]),
                s_a["watch"].get("max_part_force_sustained_N", 0.0), [v["what"] for v in s_a["violations"]][:3]))
        agent.append(s_a)
        # the privileged reference program runs inside its one call: an ANSWERED reference_assemble stands for the steps
        # (a forbidden one, made without the token, is only an illegal call)
        step_calls += s_a["step_calls"] + s_a.get("reference_calls", 0)
        illegal += s_a["illegal_call_count"]
        metrics["agent_wire_errors"] += 1.0 if s_a["end_reason"] == "wire_error" else 0.0
        metrics["remote_sockets_seen"] = max(metrics["remote_sockets_seen"], float(s_a["remote_sockets_seen"]))
        metrics["net_outbound_events"] += float(s_a.get("net_outbound", 0))
        for ck, cv in (s_a.get("net_delta") or {}).items():
            metrics["net_" + ck] = metrics.get("net_" + ck, 0.0) + float(max(0, cv))
    gate, parts = contract_gate(step_calls, rn.unclean_runs, metrics["remote_sockets_seen"], metrics["net_outbound_events"])
    for gk, gv in parts.items():
        metrics["gate_" + gk] = gv
    rewards["contract_ok"] = gate
    if not parts["network"]:
        log("[grader] contract_ok 0: network activity during an episode (%d outbound events, %d remote sockets seen)" % (
            metrics["net_outbound_events"], metrics["remote_sockets_seen"]))
    rewards["assembly_score"] = assembly_score(gate, [s["stage_score"] for s in agent], [s["hard_ok"] for s in agent])
    episode_metrics(agent, metrics)
    metrics["illegal_call_count"] = float(illegal)
    metrics["step_calls"] = float(step_calls)
    record_dir(OUT)
    with open(os.path.join(OUT, "episodes.json"), "w") as f:
        json.dump([{k: v for k, v in s.items() if k not in ("events",)} for s in agent], f, indent=1, default=float)
    _finish_stats(rn, metrics, t_start)
    overall = write_verdict(rewards, metrics)
    log("[grader] rewards %s overall %.4f (verify wall %.0f s)" % ({k: round(v, 4) for k, v in rewards.items()}, overall, metrics["verify_wall_s"]))


def episode_metrics(agent, metrics):
    """the statistics of the episodes (DESIGN v1.2 section 3.4; metrics only, never the score): the ungated mean, every
    place's mate rate, LOCKED_OFF, scrapped units, every episode's net buoyancy / pitch / roll / trim / fastened blocks,
    the hard-violation rate, the largest sustained part contact force and, apart from it, the welds' largest sustained
    constraint force, turns, frames, sim time"""
    def mean(xs):
        xs = [float(x) for x in xs]
        return float(np.mean(xs)) if xs else 0.0
    metrics["stage_score_mean_ungated"] = mean(s["stage_score"] for s in agent)
    metrics["mates_achieved_mean"] = mean(s["n_achieved"] for s in agent)
    metrics["mates_fastened_mean"] = mean(s["n_fastened"] for s in agent)
    metrics["all_mates_rate"] = mean(1.0 if s["all_mates"] else 0.0 for s in agent)
    metrics["locked_off_mean"] = mean(len(s["locked_off"]) for s in agent)
    metrics["scrapped_units_mean"] = mean(len(s["scrapped"]) for s in agent)
    metrics["trim_mean"] = mean(s["trim"] for s in agent)
    metrics["hard_violation_rate"] = mean(0.0 if s["hard_ok"] else 1.0 for s in agent)
    metrics["robot_contact_episodes"] = float(sum(1 for s in agent if s["robot_contact_violations"]))
    metrics["part_force_episodes"] = float(sum(1 for s in agent if s["part_force_violations"]))
    metrics["mean_sim_s"] = mean(s["sim_time_s"] for s in agent)
    metrics["mean_episode_wall_s"] = mean(s["episode_wall_s"] for s in agent)
    metrics["max_part_force_sustained_N"] = max([float(s["watch"].get("max_part_force_sustained_N", 0.0)) for s in agent] or [0.0])
    metrics["max_robot_link_force_N"] = max([float(s["watch"].get("max_robot_link_force_N", 0.0)) for s in agent] or [0.0])
    metrics["max_flange_force_sustained_N"] = max([float(s["watch"].get("max_flange_force_sustained_N", 0.0)) for s in agent] or [0.0])
    metrics["max_dome_force_sustained_N"] = max([float(s["watch"].get("max_dome_force_sustained_N", 0.0)) for s in agent] or [0.0])
    metrics["weld_F_sustained_max_N"] = max([float(((s.get("cell_metrics") or {}).get("weld") or {}).get("weld_F_sustained_max_N", 0.0))
                                             for s in agent] or [0.0])
    metrics["turns_mean"] = mean(s["turns_used"] for s in agent)
    metrics["camera_frames_mean"] = mean(s["camera_frames_total"] for s in agent)
    for cam in ("top", "wrist"):
        metrics["camera_frames_mean." + cam] = mean(s["camera_frames"].get(cam, 0) for s in agent)
    metrics["render_wall_s_mean"] = mean(s["render_wall_s"] for s in agent)
    metrics["render_failures"] = float(sum(int(s.get("render_failures", 0)) for s in agent))
    metrics["steps_clamped"] = float(sum(int(s.get("steps_clamped", 0)) for s in agent))
    metrics["fixture_refused"] = float(sum(int((s.get("fixture_calls") or {}).get("refused", 0)) for s in agent))
    metrics["layout_digest_checked"] = float(sum(1 for s in agent if s.get("layout_digest_checked")))
    for k, s in enumerate(agent, 1):
        te = s["trim_eval"]
        p = "ep%d." % k
        metrics[p + "stage"] = float(s["stage_score"])
        metrics[p + "hard_ok"] = 1.0 if s["hard_ok"] else 0.0
        metrics[p + "trim"] = float(s["trim"])
        metrics[p + "net_buoyancy_N"] = float(te["dB_N"])
        metrics[p + "pitch_deg"] = float(te["theta_deg"])
        metrics[p + "roll_deg"] = float(te["phi_deg"])
        metrics[p + "ballast_fastened"] = float(te["n_ballast"])
        metrics[p + "mates"] = float(s["n_achieved"])
        metrics[p + "locked_off"] = float(len(s["locked_off"]))
        metrics[p + "scrapped"] = float(len(s["scrapped"]))
        metrics[p + "illegal_calls"] = float(s["illegal_call_count"])
    if agent:
        for mid in agent[0]["weights"]:
            metrics["mate_rate." + mid] = mean(1.0 if any(r["id"] == mid and r["achieved"] for r in s["mates"]) else 0.0 for s in agent)
    return metrics


def _finish_stats(rn, metrics, t_start):
    if rn is not None:
        metrics["runner_processes_killed"] = float(rn.killed)
        metrics["runner_processes_left"] = float(rn.left)
        metrics["runner_unclean_runs"] = float(rn.unclean_runs)
        metrics["stray_files_removed"] = float(rn.stray_files)
        metrics["submission_symlinks_skipped"] = float(rn.symlinks_skipped)
    metrics["agent_processes_frozen"] = float(len(FREEZER[0].frozen)) if FREEZER[0] is not None else 0.0
    metrics["verify_wall_s"] = time.time() - t_start
    shutil.rmtree(RUNS_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
