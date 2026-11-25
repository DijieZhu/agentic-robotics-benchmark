"""Grader of tensegrity_collapsed_warehouse_void_search (a three-strut cable-driven tensegrity rolling robot searches
the void under a collapsed slab of a warehouse, steered off-board from a fixed overhead RGB-D camera): runs as ROOT
in the verify stage. Verification is separate (task.toml): this runs in a fresh verifier sandbox from the task image,
with only the agent's submission restored into it; tests/test.sh built the root-owned copy of the givens (/opt/arb)
before this starts. Built on the precedent task's grader (quadruped_hazardous_plant_inspection/tests/verify.py, same
stages, same kit) with this task's cell, program and score.

The agent left /home/user/submission/operate.py (solve(station): the operator's program on the off-board PC: it reads
the search camera's raw RGB-D frames and the robot's telemetry, commands primitives and cable targets, reports the
three search points after a 20 s dwell and brings the robot back to the retrieval area). This grader:
  0. FIRST, stdlib only (`_early_lockdown`; tests/test.sh closed the stage folder already): records the upload time,
     freezes and kills every process of the agent account and keeps it stopped until the verdict is written
     (cellkit.verify_freeze), locks the stage, snapshots the submission root-owned, gates late writes, quarantines late
     entries elsewhere;
  1. checks the root-owned copy of the givens at /opt/arb against given_hashes.json (the client library handed to the
     programs comes from it; a mismatch is a task error), and that the cell's announced constants (mission(): dwell,
     credit radii, still tolerance, retrieval radius; the per-point report budget) equal the acceptance file's
     (voidscore/acceptance.yaml, the single source of every threshold; a difference is a task error); with
     "hidden_plates" in grader_config.json (the search camera's plate backend) it then fetches the hidden sites' plate
     bundles, checks them and installs them under plate_root (fetch_hidden_plates; any failure is a task error), and
     before the first episode it checks that every pinned site has its bundle;
  2. gates: contract_ok = operate.py is a regular file that parses and defines solve, the route gate (no weight files,
     no training calls, size cap), every .py file of the submission readable by the parser, the network audit (static:
     no networking import, no computed import, no computed code, no program start; runtime: every process of the
     program -- the import check and the episodes -- runs under the no-network seccomp filter (cellkit.nonet), and the
     runner account's remote inet sockets are sampled), the module imports under the run regime (120 s, 120 CPU-s),
     >= 1 step call, no late write, no surviving process;
  3. the oracle token: submission/.oracle_token whose sha256 equals hidden/oracle_token.sha256 switches the cells to the
     privileged mode (the call reference_run: the privileged reference program, tests/reference/reference.py, inside the
     cell; stat privileged_mode); a wrong token is ignored (stat oracle_token_mismatch);
  4. takes the `episodes` hidden sites pinned in hidden/seeds.json (a fixed hidden set selected by the pre-registered
     rule of hidden/selection.json): each a stored site package hidden/sites/<name>.tar.xz whose sha256 the file pins
     (a missing, short or mismatched set is a task error); the simulated-time limit comes from hidden/limits.json
     (`time_limit_s`; written with the selection), every other budget from grader_config.json `limits` (the cell's
     defaults otherwise). Per site the agent's program runs on the cell (voidcell.server.VoidCellServer, mode hidden:
     the site from the package, the robot set down on the pad with the pinned seed; the station's calls only) as
     `nobody` from a read-only copy under an opaque run tag, with oom_score_adj 1000; an episode in which the cell
     failed (an internal error while live, a frame it could not render) is run once more from scratch and the second
     run counts (at most one per verify; a second one is a task error);
  5. criterion: void_search_score = the mean over the sites of the episode's score S = F x (w1 c1 + w2 c2 + w3 c3)
     (voidscore.scoring from the cell's own truth record; the weights w, the credit radii, the dwell and the retrieval
     rules from voidscore/acceptance.yaml, which every site's search points must match), times the gates. A failure
     on the grader's side (the cell's build, the program's launch, the wire, the cleanup, the scoring, a score that
     is not a finite number in [0, 1]) is a TASK ERROR; a program that crashes, hangs or misbehaves ends its own
     episode, which is scored as it stands. Each episode caps the program's own time (limits.program_time_s): the
     wall clock minus the cell's simulation and rendering (cellkit.wire).
A scored episode has no debrief: nothing here builds or writes one (the practice ground's voidscore.debrief is never
imported), and the cell writes none.
Untouched sandbox: no operate.py -> contract_ok 0 -> 0. Oracle: the token + reference_run -> the recorded reference
value (tests/anchor.json) -> 1.
Records: out/episodes.json (every episode's record), out/traj_<tag>.npz (10 Hz truth samples) and logs/<tag>.log
(the program's stderr) under the root-only /opt/arb-verify.
"""

import ast
import json
import os
import pwd
import shutil
import stat as statmod
import subprocess
import sys
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
SNAPSHOT = os.path.join(RUNS_ROOT, "_submission_snapshot")
QUARANTINE = os.path.join(RUNS_ROOT, "_quarantine")
STICKY_DIRS = [d for d in ("/tmp", "/dev/shm", "/var/tmp", "/run/lock") if os.path.isdir(d)]
WATCH_ROOTS = STICKY_DIRS + [HOME]
LATE_GATE_MAX_GAP_S = 600.0

from cellkit import lockdown, runner, verify_freeze  # noqa: E402  (stdlib only)

FREEZER = [None]        # the agent account's verify_freeze.AgentFreezer: started by _early_lockdown, stopped after the verdict


def log(msg):
    print(msg, file=sys.stderr, flush=True)


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
        if left:
            log("[grader] %d process(es) of the agent account (the image's respawned desktop) outlived the kill at verify start: "
                "frozen until the verdict" % left)
    FREEZER[0] = verify_freeze.AgentFreezer.for_user(AGENT_USER).start()
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
        out["submission_snapshot_skipped"] = float(lockdown.snapshot_dir(SUBMISSION, SNAPSHOT) or 0)
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
import math  # noqa: E402

import numpy as np  # noqa: E402

from cellkit import netaudit, nonet, wire  # noqa: E402
from cellkit.seeds import load_pinned  # noqa: E402
from cellkit.seeds import seed_rng  # noqa: E402
import voidcell  # noqa: E402,F401  (puts the cell's vendored library on sys.path)
from voidcell._lib import budgets as cell_budgets  # noqa: E402
from voidcell._lib import mission as cell_mission  # noqa: E402
from voidcell.client import launcher_argv  # noqa: E402
from voidcell.server import VoidCellServer  # noqa: E402
from voidcell.wire_guard import GuardedProcess  # noqa: E402  (the request-line guard)
from voidscore import scoring  # noqa: E402

HIDDEN = os.path.join(STAGE_DIR, "hidden")
RECORD_ROOT = os.environ.get("ARB_RECORD_ROOT") or ("/opt/arb-verify" if os.getuid() == 0 else STAGE_DIR)
LOGS = os.path.join(RECORD_ROOT, "logs")
OUT = os.path.join(RECORD_ROOT, "out")
PROGRAM = "operate"
LIBRARY = "voidcell"
CRITERIA = ["void_search_score"]
GATES = ["contract_ok"]
IMPORT_WALL_S = 120          # the import check's limits (instruction.md, station_api.md): wall seconds and CPU seconds
IMPORT_CPU_S = 120
# the kernel's first choice when the sandbox's memory runs out is the program, never the grader (as in the precedent)
OOM_FIRST = ["choom", "-n", "1000", "--"]
# what ast.parse raises for a source it cannot read (as in the precedent): the contract fails, the grader goes on
PARSE_ERRORS = (SyntaxError, RecursionError, MemoryError, ValueError)
# the numerics of the program's process, the same on every x86-64 CPU (tests/test.sh exports the same values for this
# process; ~/training/run_mission.py gives the program the same three in a practice run): OpenBLAS's Haswell kernel
# (numpy's OpenBLAS otherwise picks SkylakeX on an AVX-512 host) with its built-in dgemm blocking (the L2 size 256:
# numpy 2.5's OpenBLAS otherwise raises the blocking on Zen 4 / Zen 5 CPUs, where a product with an inner dimension
# above about 300 then crashes or returns wrong numbers) and numpy's SIMD paths up to AVX2 (X86_V3); read when numpy is
# first imported, so they are in the environment the process starts with (Runner.env)
BLAS_PIN = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256", "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR"}
SUSPICIOUS = ("/opt/ale", "ALE_STAGE_DIR", "ALE_VERIFICATION", "hidden/seeds", "hidden/sites", "oracle_token", "reference_run",
              "ALE_VERIFY", "/opt/ale/tests", ".ale-solution", "voidcell-practice", "practice_ground")
# what an episode's record (VoidCellServer.finalize) must carry for the grader; every other key is a statistic
SUMMARY_REQUIRED = ("episode_score", "calls", "illegal_call_count", "internal_errors", "score")
ACC = [None]            # the acceptance file (voidscore.scoring.load_acceptance), read once in main


def record_dir(path):
    """`path` under the record root, created on demand; never a scoring input."""
    os.makedirs(RECORD_ROOT, exist_ok=True)
    if RECORD_ROOT != STAGE_DIR:
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


def _num(v, default=0.0):
    """A finite float of `v`, or `default` (statistics only; never a scoring input)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float(default)
    return f if math.isfinite(f) else float(default)


def _dict(v):
    return v if isinstance(v, dict) else {}


def _record_value(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


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
    lib = os.path.join(ROOT_GIVEN, "lib", LIBRARY, "client.py")
    if not os.path.isfile(lib):
        problems.append("the client library given/lib/%s/client.py is missing" % LIBRARY)
    return not problems, problems


def hidden_sites(n):
    """The fixed hidden set: [(seed, archive path, name)] for the first `n` sites pinned in hidden/seeds.json, each
    package checked against the sha256 the file records. Anything short, missing or different is a task error."""
    pinned_path = os.path.join(HIDDEN, "seeds.json")
    try:
        pinned = load_pinned(pinned_path)
        doc = load_json(pinned_path)
    except (OSError, ValueError) as e:
        fail("hidden/seeds.json is unreadable: %s" % e)
    if not pinned or len(pinned.get("seeds", [])) < n or len(doc.get("sites") or []) < n:
        fail("hidden/seeds.json must pin the %d hidden sites (the fixed hidden set: seeds and site packages)" % n)
    seeds = [int(x) for x in pinned["seeds"][:n]]
    out = []
    for i, (seed, site) in enumerate(zip(seeds, doc["sites"][:n])):
        if not isinstance(site, dict):
            fail("hidden/seeds.json: site %d is not an object" % (i + 1))
        rel = str(site.get("archive") or "")
        path = os.path.normpath(os.path.join(HIDDEN, rel))
        if not rel or not path.startswith(HIDDEN + os.sep) or os.path.islink(path) or not os.path.isfile(path):
            fail("hidden/seeds.json: site %d's package %r is not a regular file under hidden/" % (i + 1, rel))
        if site.get("seed") is not None and int(site["seed"]) != seed:
            fail("hidden/seeds.json: site %d's seed differs from the pinned seed list" % (i + 1))
        want = str(site.get("sha256") or "").strip().lower()
        if len(want) != 64 or sha256_file(path) != want:
            fail("hidden/seeds.json: the package of site %d (%s) does not match its recorded sha256" % (i + 1, rel))
        out.append((seed, path, os.path.basename(rel)))
    return seeds, out


def cell_config(cfg):
    """The cell's run settings: grader_config's limits, pose_source, render_backend, msaa_rgb; the simulated-time
    limit from hidden/limits.json (the selection's); the per-point report budget from the acceptance file; with a
    plate_root (the plate backend's bundles of the hidden sites), that folder resolved under this stage."""
    try:
        lim = load_json(os.path.join(HIDDEN, "limits.json"))
        t_lim = float(lim["time_limit_s"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        fail("hidden/limits.json (the simulated-time limit of the hidden set) is unusable: %s" % e)
    if not (math.isfinite(t_lim) and t_lim > 0):
        fail("hidden/limits.json: time_limit_s %r is not a positive number" % (t_lim,))
    limits = dict(cfg.get("limits") or {})
    limits["time_limit_s"] = t_lim
    limits["reports_per_point"] = int(ACC[0]["reports_per_point"])
    out = {"limits": limits}
    for k in ("pose_source", "render_backend", "msaa_rgb"):
        if k in cfg:
            out[k] = cfg[k]
    if cfg.get("plate_root"):                    # the plate bundles of the hidden set (plate_filament backend)
        root = os.path.normpath(os.path.join(STAGE_DIR, str(cfg["plate_root"])))
        if not root.startswith(STAGE_DIR + os.sep) or os.path.islink(root) or not os.path.isdir(root):
            fail("grader_config.json plate_root %r is not a folder under the staged tests/" % cfg["plate_root"])
        out["plate_root"] = root
    return out


def fetch_hidden_plates(cfg):
    """The plate bundles of the hidden sites (the plate backend's offline environment of each site) are not in the
    repository: one tarball in the benchmark's asset dataset, pinned in grader_config.json ("hidden_plates": the URL at a
    dataset commit, its sha256 and its file count). It is downloaded here, before the cell is configured and outside
    every episode's network count, checked, unpacked into a temporary folder of this root-only stage and installed into
    plate_root (voidcell/_lib/plate_install.py, the same code as the practice ground's image step: each bundle folder
    h01 ... h10 renamed to the site package sha256 its meta.json records, which must be the sha256 hidden/seeds.json
    pins for the site of the same name). A failed download, a hash mismatch, a member that is not a regular file of a
    bundle folder, another file count or a bundle of another site is a task error, and so is a plate_root that exists and
    holds anything but installed bundles (it is replaced, never merged). ARB_HIDDEN_PLATES_TARBALL (the local
    copy tests/assets/plates.tar.xz that tests/test.sh points at before the bundles are published; checked the same way)
    replaces the download. Returns (files, seconds)."""
    from voidcell._lib import plate_install
    spec = cfg.get("hidden_plates")
    if not isinstance(spec, dict) or not cfg.get("plate_root"):
        fail("grader_config.json hidden_plates must be an object (url, sha256, files) next to a plate_root")
    root = os.path.normpath(os.path.join(STAGE_DIR, str(cfg["plate_root"])))
    if not root.startswith(STAGE_DIR + os.sep) or os.path.islink(root):
        fail("grader_config.json plate_root %r is not a folder under the staged tests/" % cfg["plate_root"])
    n_want = spec.get("files")
    if not isinstance(n_want, int) or isinstance(n_want, bool) or n_want <= 0:
        fail("grader_config.json hidden_plates.files is not a number of files")
    if os.path.lexists(root) and not plate_install.holds_only_bundles(root):
        fail("grader_config.json plate_root %r exists and holds something other than plate bundles" % cfg["plate_root"])
    try:
        rows = load_json(os.path.join(HIDDEN, "seeds.json"))["sites"]
        expect = {}
        for row in rows:
            name = os.path.basename(str(row["archive"]))
            expect[name[:-len(".tar.xz")] if name.endswith(".tar.xz") else name] = str(row["sha256"]).lower()
    except (OSError, ValueError, KeyError, TypeError) as e:
        fail("hidden/seeds.json is unusable for the plate bundles: %s" % e)
    t0 = time.time()
    tarball = os.path.join(STAGE_DIR, ".plates.tar.xz")
    part = os.path.join(STAGE_DIR, ".plates.part")
    shutil.rmtree(part, ignore_errors=True)
    shutil.rmtree(root, ignore_errors=True)
    local = os.environ.get("ARB_HIDDEN_PLATES_TARBALL")
    try:
        plate_install.fetch_tarball(spec, tarball, local=local, log=lambda s: log("[grader] " + s))
        folders, n = plate_install.unpack_tarball(tarball, part)
        if n != n_want:
            fail("the hidden sites' plate bundles: the tarball holds %d files; grader_config.json expects %d" % (n, n_want))
        plate_install.install(folders, root, expect)
    except (plate_install.PlateInstallError, OSError) as e:
        fail("the hidden sites' plate bundles: %s" % e)
    finally:
        shutil.rmtree(part, ignore_errors=True)
        if os.path.lexists(tarball):
            os.remove(tarball)
    os.chmod(root, 0o700)
    dt = time.time() - t0
    log("[grader] the hidden sites' plate bundles: %d files, %d sites, in %.1f s" % (n, len(expect), dt))
    return n, dt


def plates_for_sites(root, sites):
    """Every pinned hidden site has its bundle <plate_root>/<site package sha256>/meta.json before the first episode
    (the camera checks the whole bundle again when it opens)."""
    missing = [name for _seed, path, name in sites
               if not os.path.isfile(os.path.join(root, sha256_file(path), "meta.json"))]
    if missing:
        fail("grader_config.json plate_root holds no plate bundle for the hidden site(s) %s" % missing)


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
    """sha256 of every .py file of the shipped client library (a submission may carry copies of it)."""
    out = set()
    d = os.path.join(ROOT_GIVEN, "lib", LIBRARY)
    for dp, _dn, fns in os.walk(d):
        for fn in fns:
            if fn.endswith(".py"):
                out.add(sha256_file(os.path.join(dp, fn)))
    return out


def blas_pin_stats(child):
    """Statistics only, never a scoring input: is the numerics' pin in effect. blas_pin_env: this process's
    environment holds BLAS_PIN (tests/test.sh sets it before the interpreter starts); blas_pin_child_env: the
    program's environment `child` (Runner.env) does; blas_l2_size_256: both of them hold OPENBLAS_L2_SIZE=256 1, else
    0; blas_kernel_haswell: every OpenBLAS loaded in this process reports the Haswell kernel 1, another kernel 0, not
    readable -1; blas_numpy_simd_capped: numpy dispatches none of the SIMD targets the pin disables 1, it does 0, not
    readable -1."""
    out = {"blas_pin_env": 1.0 if all(os.environ.get(k) == v for k, v in BLAS_PIN.items()) else 0.0,
           "blas_pin_child_env": 1.0 if all(child.get(k) == v for k, v in BLAS_PIN.items()) else 0.0,
           "blas_l2_size_256": 1.0 if os.environ.get("OPENBLAS_L2_SIZE") == "256" and child.get("OPENBLAS_L2_SIZE") == "256" else 0.0,
           "blas_kernel_haswell": -1.0, "blas_numpy_simd_capped": -1.0}
    try:
        import ctypes
        import re
        with open("/proc/self/maps") as fh:
            libs = sorted(set(re.findall(r"(/\S*openblas\S*\.so\S*)", fh.read())))
        names = []
        for lib in libs:
            h = ctypes.CDLL(lib)
            for sym in ("scipy_openblas_get_corename64_", "scipy_openblas_get_corename", "openblas_get_corename64_",
                        "openblas_get_corename"):
                f = getattr(h, sym, None)
                if f is not None:
                    f.restype = ctypes.c_char_p
                    names.append((f() or b"").decode("ascii", "replace").strip().lower())
                    break
        if names:
            out["blas_kernel_haswell"] = 1.0 if all(n == "haswell" for n in names) else 0.0
    except Exception:  # noqa: BLE001 - a record only
        pass
    try:
        try:
            from numpy._core._multiarray_umath import __cpu_features__ as feats
        except ImportError:
            from numpy.core._multiarray_umath import __cpu_features__ as feats
        out["blas_numpy_simd_capped"] = 0.0 if any(feats.get(k) for k in BLAS_PIN["NPY_DISABLE_CPU_FEATURES"].split()) else 1.0
    except Exception:  # noqa: BLE001 - a record only
        pass
    return out


class Runner(object):
    def __init__(self, cfg, cpu_s):
        self.acct = runner.runner_account(cfg.get("runner_account", "nobody"), AGENT_USER)
        self.name, self.uid, self.gid = self.acct if self.acct else (None, None, None)
        self.lib_dirs = {LIBRARY: os.path.join(ROOT_GIVEN, "lib", LIBRARY)}
        self.cpu_s = float(cpu_s)
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
        """The program's environment: cellkit.runner.child_env plus the numerics' pin (BLAS_PIN) and the rasteriser's
        threads; `extra` last."""
        pinned = dict(BLAS_PIN, **(extra or {}))
        e = runner.child_env(dirs["home"], dirs["tmp"], dirs["lib"], ROOT_GIVEN, threads=self.threads, extra=pinned)
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


def _parse_error(e):
    return str(e) if isinstance(e, SyntaxError) else "%s: %s" % (type(e).__name__, e)


def unparsable_files(src):
    """[(relpath, error text)] of the submission's .py files the parser cannot read at all (see precedent)."""
    out = []
    for dp, dns, fns in os.walk(src):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for fn in sorted(fns):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dp, fn)
            if os.path.islink(p):
                continue
            try:
                data = open(p, "rb").read()
            except OSError:
                continue
            try:
                ast.parse(data.decode("utf-8", "replace"))
            except SyntaxError:
                continue
            except PARSE_ERRORS as e:
                out.append((os.path.relpath(p, src), _parse_error(e)[:200]))
    return out


def contract_check(cfg, rn):
    src = program_source()
    path = os.path.join(src, PROGRAM + ".py")
    if not os.path.isfile(path) or os.path.islink(path):
        return False, "no %s.py (regular file) in the submission" % PROGRAM, 0, []
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except PARSE_ERRORS as e:
        return False, "%s.py does not parse: %s" % (PROGRAM, _parse_error(e)), 0, []
    has_solve = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "solve" for n in ast.walk(tree)) or \
        any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "solve" for t in n.targets) for n in ast.walk(tree)) or \
        any(isinstance(n, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name) == "solve" for a in n.names) for n in ast.walk(tree))
    if not has_solve:
        return False, "%s.py defines no solve" % PROGRAM, 0, []
    problems, suspicious = route_gate(cfg, src)
    if problems:
        return False, "route gate: " + "; ".join(problems[:5]), suspicious, []
    bad = unparsable_files(src)
    if bad:
        why = "a .py file the parser cannot read: %s" % "; ".join("%s (%s)" % x for x in bad[:5])
        return False, why, suspicious, []
    try:
        net = netaudit.scan_imports(src, allow_digests=library_digests())
    except PARSE_ERRORS as e:
        return False, "a .py file the network audit's parser cannot read (%s)" % _parse_error(e)[:200], suspicious, []
    if net:
        return False, "network audit: networking import(s) %s" % ", ".join("%s: %s" % x for x in net[:5]), suspicious, net
    dirs = rn.prepare("import_check", src)
    before = rn.snapshot((dirs["root"],)) if rn.acct else None
    code = ("import sys, importlib; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); "
            "m = importlib.import_module('%s'); assert callable(m.solve)" % PROGRAM)
    argv = runner.wrap_limits(OOM_FIRST + [sys.executable] + runner.PYTHON_FLAGS + ["-c", code, dirs["lib"], dirs["program"]],
                              cpu_s=IMPORT_CPU_S, user=rn.name, nonet=True)
    try:
        r = subprocess.run(argv, cwd=dirs["program"], env=rn.env(dirs), capture_output=True, text=True, timeout=IMPORT_WALL_S,
                           start_new_session=True)
        ok, why = r.returncode == 0, ("ok" if r.returncode == 0 else "importing %s.py failed: " % PROGRAM + (r.stderr.strip().splitlines() or ["?"])[-1][:300])
    except subprocess.TimeoutExpired:
        ok, why = False, "importing %s.py took more than %d s" % (PROGRAM, IMPORT_WALL_S)
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


def _settle_desktop(max_s=60.0, quiet_s=8.0, poll_s=0.25):
    """Wait, before the first episode's network baseline, until the frozen desktop is quiet (precedent)."""
    fr = FREEZER[0]
    if os.getuid() != 0 or fr is None or fr.uid is None:
        return 0.0, 0
    t0 = time.time()
    first = last = (netaudit.net_counters(), sorted(verify_freeze.uid_processes(fr.uid)))
    quiet_since = t0
    while time.time() - t0 < max_s:
        time.sleep(poll_s)
        now = (netaudit.net_counters(), sorted(verify_freeze.uid_processes(fr.uid)))
        if now != last:
            last, quiet_since = now, time.time()
        elif time.time() - quiet_since >= quiet_s:
            break
    seen = sum(max(0, int(last[0].get(k, 0)) - int(first[0].get(k, 0))) for k in first[0] if k != "unreadable")
    return time.time() - t0, seen


def _new_cell(ccfg, seed, package, privileged):
    """A fresh cell on one hidden site package, its scorer bound to the site (the one place this is spelled)."""
    server = VoidCellServer(config=ccfg, mode="hidden", privileged=privileged, log=log,
                            reference_dir=os.path.join(STAGE_DIR, "reference") if privileged else None)
    server.reset(package, seed=seed, kind="hidden")
    server.scorer = scoring.scorer_for(server.site, ACC[0])
    return server


def _close_cell(server):
    try:
        server.close()
    except Exception:  # noqa: BLE001 - a record of the cell's teardown is never a scoring input
        pass


def run_episode(cfg, ccfg, site, rn, privileged, tag):
    """One graded episode: a fresh cell, the program on the wire as the runner account, the cell's record. A failure on
    the grader's side is a task error (`fail`), never the agent's zero; whatever the program does ends at most its own
    episode."""
    seed, package, name = site
    try:
        server = _new_cell(ccfg, seed, package, privileged)
    except scoring.AcceptanceError as e:
        fail("%s: the site and the acceptance file disagree: %s" % (tag, e))
    except Exception:  # noqa: BLE001 - the cell depends on the stored package only: a build failure is the task's
        fail("%s: the cell could not be built\n%s" % (tag, traceback.format_exc()))
    record_dir(LOGS)
    t0 = time.time()
    dirs = rn.prepare(tag, program_source())          # an opaque tag: the hidden site never reaches a path the program sees
    before = rn.snapshot((dirs["root"],)) if rn.acct else None
    argv, user = OOM_FIRST + launcher_argv(sys.executable, dirs["lib"], dirs["program"], PROGRAM), rn.name
    program_cap = float(cfg["privileged_program_time_cap_s"] if privileged else server.limits["program_time_s"])
    silence = float(server.limits["silence_s"])
    proc = None
    sampler = netaudit.SocketSampler(uid=rn.uid if rn.acct else None, pids_fn=None if rn.acct else (lambda: []),
                                     period_s=float(cfg.get("socket_sample_s", 0.5)))
    counters = netaudit.NetCounters()
    reason = None
    with counters, sampler:
        try:
            try:
                proc = GuardedProcess(argv, cwd=dirs["program"], user=user, env=rn.env(dirs, None), cpu_s=rn.cpu_s,
                                      stderr_path=os.path.join(LOGS, "%s.log" % tag), nonet=True)
            except Exception:  # noqa: BLE001
                fail("%s: the program could not be launched\n%s" % (tag, traceback.format_exc()))
            try:
                reason = wire.serve_episode(server, proc, program_cap, silence, log=log)
            except Exception:  # noqa: BLE001 - the wire turns everything the program does into an end reason
                fail("%s: the wire failed\n%s" % (tag, traceback.format_exc()))
        finally:
            if proc is not None:
                proc.terminate()
            rn.cleanup(dirs, before)             # every process of the runner account is gone before the counters are read
    try:
        if not server.done:
            server._end(reason)
        summary = server.finalize()
    except Exception:  # noqa: BLE001 - finalize() and the scorer do not raise by design
        fail("%s: the episode could not be scored\n%s" % (tag, traceback.format_exc()))
    missing = [k for k in SUMMARY_REQUIRED if not isinstance(summary, dict) or k not in summary]
    if missing:
        fail("%s: the episode's record lacks %s" % (tag, ", ".join(missing)))
    try:
        sc = float(summary["episode_score"])
    except (TypeError, ValueError):
        sc = float("nan")
    if not (math.isfinite(sc) and 0.0 <= sc <= 1.0):
        fail("%s: the episode's score %r is not a finite number in [0, 1]" % (tag, summary.get("episode_score")))
    summary["episode_score"] = sc
    summary["episode_wall_s"] = time.time() - t0
    summary["end_reason"] = reason
    summary["site"] = name
    summary["step_calls"] = int(_num(_dict(summary.get("calls")).get("step", 0)))
    try:                                            # the truth samples, for the maintainers' replay tool
        record_dir(OUT)
        np.savez_compressed(os.path.join(OUT, "traj_%s.npz" % tag), traj=server.trajectory(),
                            columns=np.array(server.trajectory_columns()), seed=seed, site=name, kind="hidden")
    except Exception:  # noqa: BLE001 - a record is never a scoring input
        log("[grader] %s: trajectory record not written" % tag)
    _close_cell(server)
    summary["remote_sockets_seen"] = int(sampler.max_seen)
    summary["socket_samples"] = int(sampler.samples)
    summary["net_outbound"] = int(counters.total)
    summary["net_delta"] = dict(counters.delta)
    return summary


def _cell_failed(s):
    """What went wrong on the cell's side of an episode (not the program's doing): [] when nothing.
      internal_error  an exception in the cell's own work while the episode was live (every argument is checked at
                      the server's boundary first, voidcell/server.py check_args, and refused with bad_argument);
      render_failure  a frame that could not be rendered (what is rendered depends on the scene, not on the request)."""
    out = []
    live = s.get("internal_errors_live")
    if int(_num(s.get("internal_errors", 0) if live is None else live)) > 0:
        out.append("internal_error")
    if int(_num(s.get("render_failures", 0))) > 0:
        out.append("render_failure")
    return out


MAX_RERUNS_PER_VERIFY = 1      # task.toml's verifier timeout is derived for the hidden set and one re-run


def _episode_with_retry(cfg, ccfg, site, rn, privileged, tag, metrics):
    """One episode; when the cell failed in it, it is run once more from scratch and the second run counts. At most
    one episode of a verify is run again; a second one in which the cell failed is a task error; a render failure
    that repeats is a task error; an internal error that repeats with the same program is scored as it stands."""
    s_a = run_episode(cfg, ccfg, site, rn, privileged, tag=tag)
    why = _cell_failed(s_a)
    if not why:
        return s_a
    if metrics.get("episode_retries", 0.0) >= MAX_RERUNS_PER_VERIFY:
        fail("%s: the cell failed (%s) in a second episode of this verify; at most %d episode is run again per verify"
             % (tag, ", ".join(why), MAX_RERUNS_PER_VERIFY))
    log("[grader]   %s: the cell failed (%s): the episode is run once more" % (tag, ", ".join(why)))
    metrics["episode_retries"] = metrics.get("episode_retries", 0.0) + 1.0
    s_b = run_episode(cfg, ccfg, site, rn, privileged, tag=tag + "r")
    why_b = _cell_failed(s_b)
    if "render_failure" in why_b:
        fail("%s: the cell could not render frames in two runs of the episode" % tag)
    s_b["retried_after"] = why
    return s_b


def write_verdict(rewards, metrics, contract_why=None):
    """One gated `reward` in the benchmark kit's envelope (robotics_grader.ratio_rewards against the recorded reference
    value); the gate, the criterion and every statistic go to `metrics` (precedent)."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write_verdict
    overall = float(rewards["void_search_score"])
    anchor, meta = read_anchor()
    extra = {}
    for k, v in metrics.items():
        try:
            extra[k] = float(v)
        except (TypeError, ValueError):
            continue
    extra.update({k: float(rewards[k]) for k in GATES + CRITERIA})
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "void_search_score"), extra=extra, full_at=meta.get("full_at", 1.0))
    w = ACC[0]["weights"]
    text = ("contract_ok %.0f (the program imports, exposes solve, passes the route gate and the network audit and steps the "
            "station; no late write, no surviving process); void_search_score %.4f (the mean over the hidden sites of the "
            "episode's score S = F x (%g c1 + %g c2 + %g c3): c = a valid %g s dwell before the last accepted report of the "
            "point x the distance credit; F %g when the robot ends in the retrieval area with no command running, else "
            "%g)") % (rewards["contract_ok"], rewards["void_search_score"], w[1], w[2], w[3],
                      ACC[0]["dwell"]["window_s"], ACC[0]["retrieval"]["factor_retrieved"],
                      ACC[0]["retrieval"]["factor_not_retrieved"])
    if contract_why and not rewards["contract_ok"]:
        text += "; the contract failed: %s" % str(contract_why)[:300]
    if os.environ.get("ALE_VERIFY_PYTHON"):
        recorded = {}
        for k, v in sorted(verdict["metrics"].items()):
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v == v and abs(v) != float("inf"):
                recorded[k.replace(".", "_").replace("/", "_")] = v
        kit_write_verdict({"rewards": verdict["rewards"], "metrics": recorded}, diagnostic=text)
        return overall
    kit_write_verdict(verdict, path=os.path.join(STAGE_DIR, "verdict.local.json"), diagnostic=text)
    return overall


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
    cfg = load_json(os.path.join(STAGE_DIR, "grader_config.json"))
    rewards = dict((k, 0.0) for k in GATES + CRITERIA)
    metrics = {"grader_exception": 0.0, "episodes": float(cfg.get("episodes", 10)),
               "root_payload_ok": 0.0, "agent_deprivileged": 0.0, "runner_uid": -1.0, "privileged_mode": 0.0, "oracle_token_mismatch": 0.0,
               "runner_processes_killed": 0.0, "runner_processes_left": 0.0, "runner_unclean_runs": 0.0,
               "stray_files_removed": 0.0, "suspicious_lines": 0.0, "remote_sockets_seen": 0.0, "net_outbound_events": 0.0,
               "network_imports": 0.0, "illegal_call_count": 0.0}
    for k in ("stage_locked", "stage_parent_locked", "stage_lock_latency_ms", "agent_processes_killed", "agent_processes_left",
              "submission_late_writes", "late_writes_quarantined", "late_write_gate_active", "submission_snapshot"):
        metrics[k] = float(EARLY.get(k, -1.0 if k == "stage_lock_latency_ms" else 0.0))
    try:
        try:
            _main_locked(cfg, rewards, metrics)
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001 - a grader defect must still leave a verdict and say so
            log("[grader] EXCEPTION\n" + traceback.format_exc())
            metrics["grader_exception"] = 1.0
            raise
    finally:
        _restore_modes()
        if FREEZER[0] is not None:
            FREEZER[0].stop()


def _main_locked(cfg, rewards, metrics):
    t_start = time.time()
    try:
        ACC[0] = scoring.load_acceptance()
        scoring.check_cell_constants(cell_mission, cell_budgets, ACC[0])
    except Exception as e:  # noqa: BLE001 - the acceptance file or the cell's copies of it are the task's
        fail("the acceptance file is unusable or the cell disagrees with it: %s" % e)
    metrics["acceptance_version"] = float(ACC[0]["version"])
    expected = load_json(os.path.join(STAGE_DIR, "given_hashes.json"))
    ok_root, problems = root_payload_ok(expected)
    metrics["root_payload_ok"] = 1.0 if ok_root else 0.0
    if not ok_root:
        fail("the root-owned copy of the givens at %s is unusable: %s" % (ROOT_PAYLOAD, "; ".join(problems[:5])))
    if "hidden_plates" in cfg:
        pl_files, pl_s = fetch_hidden_plates(cfg)
        metrics["hidden_plates_files"] = float(pl_files)
        metrics["hidden_plates_fetch_s"] = float(pl_s)
        metrics["hidden_plates_local_copy"] = 1.0 if os.environ.get("ARB_HIDDEN_PLATES_TARBALL") else 0.0
    ccfg = cell_config(cfg)
    lim = cell_budgets.merged(ccfg["limits"])
    rn = Runner(cfg, cpu_s=lim["cpu_s"])
    if FREEZER[0] is not None and rn.acct and rn.uid == FREEZER[0].uid:
        FREEZER[0].stop()        # no `nobody` in the image: the runner fell back to the agent account, whose programs must run
        log("[grader] runner account = the agent account: the verify-stage freezer is stopped")
    metrics["runner_uid"] = float(rn.uid) if rn.acct else -1.0
    metrics["agent_deprivileged"] = 1.0 if rn.acct else 0.0
    metrics["time_limit_s"] = float(lim["time_limit_s"])
    metrics["program_time_cap_s"] = float(lim["program_time_s"])
    privileged, mismatch = token_state()
    metrics["privileged_mode"] = 1.0 if privileged else 0.0
    metrics["oracle_token_mismatch"] = 1.0 if mismatch else 0.0
    ok, why, suspicious, net = contract_check(cfg, rn)
    if ok and EARLY.get("reasons"):
        ok, why = False, "; ".join(EARLY["reasons"])
    metrics["suspicious_lines"] = float(suspicious)
    metrics["network_imports"] = float(len(net))
    log("[grader] contract_ok=%s (%s); privileged %s; runner %s; lock latency %.0f ms" % (ok, why, privileged, rn.name or "self", metrics["stage_lock_latency_ms"]))
    n = int(cfg.get("episodes", 10))
    ep_seeds, sites = hidden_sites(n)
    if ccfg.get("plate_root"):
        plates_for_sites(ccfg["plate_root"], sites)
    stream = seed_rng(pinned=ep_seeds)
    metrics.update(stream.stats())
    if not ok:
        rewards["contract_ok"] = 0.0
        _finish_stats(rn, metrics, t_start)
        write_verdict(rewards, metrics, contract_why=why)
        return
    settle_s, settle_seen = _settle_desktop()
    metrics["desktop_settle_s"] = round(settle_s, 1)
    metrics["net_settle_events"] = float(settle_seen)
    nn_ok, nn_why = nonet.selftest(wrap=lambda a: runner.wrap_limits(a, cpu_s=60, user=rn.name),
                                   env=runner.child_env("/tmp", "/tmp", "/nonexistent", ROOT_GIVEN))
    metrics["nonet_selftest_ok"] = 1.0 if nn_ok else 0.0
    if not nn_ok:
        fail("the no-network filter for the program does not work here: " + nn_why)
    agent = []
    step_calls = illegal = 0
    metrics["episode_retries"] = 0.0
    for i, site in enumerate(sites):
        tag = "ep%02d" % (i + 1)
        log("[grader] episode %d/%d" % (i + 1, n))
        s_a = _episode_with_retry(cfg, ccfg, site, rn, privileged, tag, metrics)
        sc = _dict(s_a.get("score"))
        log("[grader]   score %.3f c %s F %s sim %.1f s wall %.1f s (program %.1f s, cell %.1f s) frames %s end %s/%s" % (
            s_a["episode_score"], sc.get("c"), sc.get("F"), _num(s_a.get("sim_time_s")), s_a["episode_wall_s"],
            _num(s_a.get("program_time_s")), _num(s_a.get("cell_work_s")), _dict(s_a.get("budgets")).get("frames_used"),
            s_a.get("done_reason"), s_a["end_reason"]))
        agent.append(s_a)
        step_calls += int(_num(s_a["step_calls"])) + int(_num(_dict(s_a.get("calls")).get("reference_run", 0)))
        illegal += int(_num(s_a["illegal_call_count"]))
        metrics["remote_sockets_seen"] = max(metrics["remote_sockets_seen"], float(s_a["remote_sockets_seen"]))
        metrics["net_outbound_events"] += float(s_a.get("net_outbound", 0))
        for ck, cv in (s_a.get("net_delta") or {}).items():
            metrics["net_" + ck] = metrics.get("net_" + ck, 0.0) + float(max(0, cv))
    vals = [float(s["episode_score"]) for s in agent]          # each checked finite and in [0, 1] (run_episode)
    steps_gate = 1.0 if step_calls > 0 else 0.0
    clean_gate = 1.0 if rn.unclean_runs == 0 else 0.0
    net_gate = 1.0 if metrics["remote_sockets_seen"] <= 0 else 0.0
    gate = steps_gate * clean_gate * net_gate
    rewards["contract_ok"] = gate
    if not net_gate:
        log("[grader] contract_ok 0: the program held a remote inet socket during an episode (%d seen)" % metrics["remote_sockets_seen"])
    elif not steps_gate:
        log("[grader] contract_ok 0: the program never stepped the station")
    mean = float(np.mean(vals)) if vals else 0.0
    rewards["void_search_score"] = float(max(0.0, min(1.0, gate * mean)))
    for i, s in enumerate(agent):
        metrics["episode_score.ep%02d" % (i + 1)] = float(s["episode_score"])
    for part in ("1", "2", "3"):
        metrics["c_mean." + part] = float(np.mean([_num(_dict(_dict(s.get("score")).get("c")).get(part, 0.0)) for s in agent]))
    metrics["F_mean"] = float(np.mean([_num(_dict(s.get("score")).get("F", 0.0)) for s in agent]))
    reasons = [str(s.get("done_reason") or s.get("end_reason")) for s in agent]
    for r in sorted(set(reasons)):
        metrics["end_rate." + r] = float(sum(1 for x in reasons if x == r)) / max(1, len(reasons))
    metrics["mean_sim_time_s"] = float(np.mean([_num(s.get("sim_time_s")) for s in agent]))
    metrics["mean_episode_wall_s"] = float(np.mean([_num(s.get("episode_wall_s")) for s in agent]))
    metrics["frames_used_mean"] = float(np.mean([_num(_dict(s.get("budgets")).get("frames_used")) for s in agent]))
    metrics["illegal_call_count"] = float(illegal)
    metrics["step_calls"] = float(step_calls)
    for key in ("program_time_s", "cell_work_s", "build_wall_s", "physics_wall_s"):
        metrics[key + "_mean"] = float(np.mean([_num(s.get(key)) for s in agent]))
    metrics["cell_internal_errors"] = float(sum(int(_num(s["internal_errors"])) for s in agent))
    try:                                            # the maintainers' record; never a scoring input
        record_dir(OUT)
        with open(os.path.join(OUT, "episodes.json"), "w") as f:
            json.dump([{k: v for k, v in s.items() if k not in ("telemetry",)} for s in agent], f, indent=1,
                      default=_record_value)
    except Exception:  # noqa: BLE001
        log("[grader] episodes.json not written\n" + traceback.format_exc())
    _finish_stats(rn, metrics, t_start)
    overall = write_verdict(rewards, metrics)
    log("[grader] rewards %s overall %.4f (verify wall %.0f s)" % ({k: round(v, 4) for k, v in rewards.items()}, overall, metrics["verify_wall_s"]))


def _finish_stats(rn, metrics, t_start):
    metrics["runner_processes_killed"] = float(rn.killed)
    metrics["runner_processes_left"] = float(rn.left)
    metrics["runner_unclean_runs"] = float(rn.unclean_runs)
    metrics["stray_files_removed"] = float(rn.stray_files)
    metrics["submission_symlinks_skipped"] = float(max(rn.symlinks_skipped, int(EARLY.get("submission_snapshot_skipped", 0.0))))
    metrics["agent_processes_frozen"] = float(len(FREEZER[0].frozen)) if FREEZER[0] is not None else 0.0
    metrics.update(blas_pin_stats(rn.env({"home": "/", "tmp": "/", "lib": "/"})))   # after the episodes: every OpenBLAS the cell loaded is in by now
    log("[grader] numerics pin: env %.0f, programs' env %.0f, L2 size 256 %.0f, OpenBLAS kernel Haswell %.0f, numpy SIMD capped %.0f" % (
        metrics["blas_pin_env"], metrics["blas_pin_child_env"], metrics["blas_l2_size_256"], metrics["blas_kernel_haswell"],
        metrics["blas_numpy_simd_capped"]))
    metrics["verify_wall_s"] = time.time() - t_start
    shutil.rmtree(RUNS_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
