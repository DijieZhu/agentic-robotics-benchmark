#!/usr/bin/env python3
"""Pre-registered selection of the fixed hidden set of racecar_full_assembly_tenth_scale_ur10e (a maintainer's tool; the
grader never runs it). It writes tests/hidden/selection.json (every candidate, its runs and the decision) and
tests/hidden/episodes.json (format racecar-asm-set/1: the four episodes verify.py plays, one per stratum).

The rule, fixed before any candidate ran (written 2026-11-14 into this docstring; `draw` copies it, with the constants
below and the sha256 of every frozen file, into selection.json before the first candidate runs):

1. Freeze (as amended on 2026-11-15, amendment A1; the selection drawn at 2026-11-15T04:57:01Z keeps the rule text
   as it stood then in its rule_text, and its record holds the dated amendment, written before any file changed).
   The frozen table covers what determines an episode's outcome: the kit, the cell and layout code, the reference
   program, the server, verify and the grader's values. FROZEN below lists the files frozen by their bytes (verify.py,
   test.sh, netguard.py, episode_trace.py, the hidden bench plan, the oracle token's sha256, the solution's solve.sh /
   assemble.py / token); FROZEN_VALUES the grader's values: tests/grader_config.json frozen as
   'tests/grader_config.json#values', the sha256 of its canonical JSON (keys sorted, separators ',' and ':', ASCII)
   without PUBLICATION_KEYS of its hidden_gt (url, sha256, note: the hidden kit archive's location, its hash and the
   sentence that describes the location); FROZEN_MEMBERS the hidden kit by its content: tests/assets/gt.tar.xz frozen
   as 'tests/assets/gt.tar.xz#members', the sha256 over the sorted lines '<member name> <sha256 of the member's bytes>'
   of the archive's file members; and FROZEN_TREES the code and models underneath (tests/arm, tests/common,
   tests/render, tests/render_filament, tests/room, the given robot models). `draw` records their hashes; every later
   command refuses to go on when one of them differs (the robot models are compared through the folder a run is
   given). The publication metadata (the archive's sha256 and hidden_gt's url / sha256 / note) and the documents the
   agent reads (RECORDED_FILES, RECORDED_TREES: instruction.md, the setup payload, the asset list, the given parts,
   drawings, photo manual and render assets, tests/setup_payload and tests/given_hashes.json) are recorded by hash, but
   their change does not invalidate the set: the check names a changed one as a note, and `record` enters the new
   hashes in the selection's recorded_changes with the date and the reason. Files outside both tables are outside the
   rule. The task's files were not committed when the set was selected, so the draw's UTC time and these hashes are the
   record of the freeze.
2. Draw. 12 candidates c00..c11 from os.urandom: seed = int.from_bytes(8 bytes, "little") % 2**63, drawn again while
   below 2**32 or equal to an earlier seed. Candidate i is in stratum STRATA[i % 4] (layout.STRATA: the stand's yaw sign
   x the stick pack's lead end: yaw+lead0, yaw+lead180, yaw-lead0, yaw-lead180) and round i // 4, so every stratum has
   three candidates in drawn order. selection.json is written before any candidate runs; `digests` then adds each
   candidate's layout digest (layout.layout_digest(layout.episode_layout(seed, plan, kit, cfg = grader_config["layout"],
   stratum)): a pure function of the inputs, no physics) and its layout draws (for the hollowing check).
3. Run. Every candidate runs ONCE in each of three environments, the privileged reference program exactly as the
   oracle plays it (one frame of each camera, then `reference_assemble`, then `finish`):
   - host: the development venv on the development machine, in-process: CarInfoServer(the kit unpacked from
     tests/assets/gt.tar.xz, the given robot models, the hidden bench plan, grader_config.json with the MuJoCo agent
     renderer, privileged) -> reset(seed, stratum) (its digest must be the recorded one) -> the episode trace of
     verify.py attached -> handle("get_camera" top), handle("get_camera" wrist), handle("reference_assemble"),
     handle("finish") -> finalize();
   - pod: the same on the second machine (another CPU: the shared cloud machine), its own interpreter and libraries;
   - image: the task image, through tests/test.sh -> verify.py with the solution/solve.sh submission (the oracle
     token) and the local kit tarball, one verify run per round (its four candidates are the set of that run); and,
     because verify keeps no frame, the top frame 0 of every candidate rendered once more in the same image by the
     agent renderer of grader_config.json (Filament) to measure what the top camera sees (`view`).
   All 12 candidates run in every environment, also those of a stratum whose earlier candidate passes (evidence for
   the hollowing check). A run that ends without a complete result because of the infrastructure (the process was
   killed, the machine or the container went away) is run again; a completed run is final.
4. Keep. A candidate passes one environment's run when: the layout digest is the recorded one; the reference program
   ran (no exception, its report reached finalize); all eight mates are LOCKED and the stage score is 1.0; its
   unplanned contacts are 0; the episode is hard_ok (no robot link above 30 N, no sustained part force above 150 N)
   and the largest sustained part force is below 150 N; at every capture of every mate the pose error is at most 50 %
   of that mate's capture window (reference.capture_ratios: the largest of lateral / window, tilt / window (a seat's
   total rotation) and, keyed, clock / window; the distance short of home is not an error: a unit is captured on its
   way in); every loose unit is at least 30 % seen in the top camera's frame 0 (4000 area-weighted
   samples on the unit's visual meshes; a sample is seen when it projects into the frame and the frame's sensor
   depth there is valid and within 15 mm of the sample's depth). A candidate passes when it passes in all three
   environments. A stratum's first passing candidate in drawn order is its episode; the episodes go into
   episodes.json in the order of STRATA.
5. Fallback. If a stratum has no passing candidate, no set is written and grader_config.json is not changed: the
   pre-written fallback (the oracle on the nominal layout, oracle_nominal_layout true, the agent still graded
   on a fixed hidden set) is written up as a recommendation for the maintainers.
6. Hollowing check (`report`). Flags: (a) more than 1/3 of the candidates dropped; (b) two or more drops share a cause
   (the same clause on the same mate or unit); (c) capture margin: per mate class (battery, deck, compute, lidar,
   wheel), the kept episodes' capture ratio (the largest of the three environments) against the dropped candidates':
   flagged when, on a class that dropped no candidate through its capture clause, the kept mean is below the dropped
   mean by more than 0.10 of the window (the dropped layouts were harder for the reference on mates that did not
   reject them); on a class that did drop candidates through its capture clause the gap is by construction and is
   reported, with each kept episode's place in the population of the earlier 100-seed campaign when one is given
   (`--population`); (d) the layout draws of the kept, the dropped and all candidates are reported (stand / clamp /
   shelf / tray offsets, the plays, the bench units' poses, the hubs' angles off the nearest hex orientation). A flag
   is reported with the set; the remedy is the precedent's (fix the reference program, freeze again, draw a new
   selection that supersedes this one), never knowingly keeping an easy subset.
7. Determinism record: per candidate and environment the mates' states, capture ratios and capture times, sim time,
   ticks, the largest forces, an outcome fingerprint (sha256 over finalize's physics fields without wall-clock
   values) and the sha256 of the episode trace's qpos (float32 every 4 ticks, verify's own recorder); host and pod also
   a sha256 of every control tick's qpos + qvel (float64). `determinism` writes the tables.

Acceptance of the set (`accept`): two more verify runs of tests/hidden/episodes.json in the task image; the oracle must
score 1.0 in both with the same per-episode scores.

    python3 -I -B select_set.py draw        --selection SEL.json [--supersedes OLD.json --why TEXT]
    python3 -I -B select_set.py check       --selection SEL.json [--json OUT]     (the freeze; recorded changes as notes)
    python3 -I -B select_set.py record      --selection SEL.json --why TEXT        (enter changed recorded hashes, dated)
    python3 -I -B select_set.py digests     --selection SEL.json [--kit DIR]
    python3 -I -B select_set.py run         --selection SEL.json --env host|pod --out DIR [-P 6] [--robot DIR] [--kit DIR]
    python3 -I -B select_set.py image       --selection SEL.json --round R --out DIR [--image TAG]     (docker, on the host)
    python3 -I -B select_set.py image-view  --selection SEL.json --out DIR [--image TAG]                (docker, on the host)
    python3 -I -B select_set.py ingest      --selection SEL.json --env host|pod|image --root ROOT --runs DIR [--runs DIR ...]
    python3 -I -B select_set.py make        --selection SEL.json --out EPISODES.json
    python3 -I -B select_set.py report      --selection SEL.json [--population SUMMARY.json] [--json OUT]
    python3 -I -B select_set.py accept      --selection SEL.json --episodes EPISODES.json --root ROOT --out DIR [--image TAG] [--repeat 2]
    python3 -I -B select_set.py determinism --selection SEL.json --out determinism.md [--root ROOT] [--acceptance DIR]
    python3 -I -B select_set.py selftest    [--physics --kit DIR --robot DIR]

`episode` (one candidate in-process) and `view` (the top frame only) are what `run` starts per candidate. Paths are
found from this file, so the tool runs from any copy of the task (the pod's, the image's stage). The paths of runs that
`ingest` and `accept` write into the selection are relative to ROOT, the folder the selection's runs are kept under
(a path outside it is refused); `determinism` reads the run files back under --root.
"""

import argparse
import ast
import base64
import datetime
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
TASK = os.path.dirname(TESTS)
sys.dont_write_bytecode = True
if TESTS not in sys.path:
    sys.path.insert(0, TESTS)

SEL_FORMAT = "racecar-asm-selection/1"
SET_FORMAT = "racecar-asm-set/1"                       # = verify.SET_FORMAT (selftest)
STRATA = ("yaw+lead0", "yaw+lead180", "yaw-lead0", "yaw-lead180")    # = layout.STRATA = verify.STRATA (selftest)
N_CANDIDATES = 12
SEED_MOD = 1 << 63
SEED_MIN = 1 << 32
ENVS = ("host", "pod", "image")
SLOTS = ("battery", "deck", "compute", "lidar", "whl_FL", "whl_FR", "whl_RL", "whl_RR")
MATE_CLASS = {"battery": "battery", "deck": "deck", "compute": "compute", "lidar": "lidar",
              "whl_FL": "wheel", "whl_FR": "wheel", "whl_RL": "wheel", "whl_RR": "wheel"}
MAX_SUSTAINED_N = 150.0
CAPTURE_SHARE = 0.5
SEEN_MIN = 0.30
SEEN_TOL_M = 0.015
SAMPLES = 4000
DROP_SHARE = 1.0 / 3.0
MARGIN_GAP = 0.10
FROZEN = ("tests/verify.py", "tests/test.sh", "tests/netguard.py", "tests/episode_trace.py",
          "tests/hidden/bench_plan.json", "tests/hidden/oracle_token.sha256",
          "solution/solve.sh", "solution/reference/assemble.py", "solution/reference/oracle_token")
CONFIG_FILE = "tests/grader_config.json"
KIT_ARCHIVE = "tests/assets/gt.tar.xz"
PUBLICATION_KEYS = ("url", "sha256", "note")          # of grader_config["hidden_gt"]: recorded, never frozen (amendment A1)
FROZEN_VALUES = (CONFIG_FILE,)                        # frozen as "<path>#values" (PUBLICATION_KEYS removed)
FROZEN_MEMBERS = (KIT_ARCHIVE,)                       # frozen as "<path>#members" (the archive's file members)
FROZEN_TREES = ("tests/arm", "tests/common", "tests/render", "tests/render_filament", "tests/room")
ROBOT_TREE = "environment/assets/payload/given/robot"
RECORDED_FILES = ("instruction.md", "environment/assets.sha256", "tests/given_hashes.json")
RECORDED_TREES = ("environment/setup/payload", "environment/assets/payload/given/parts", "environment/assets/payload/given/drawings",
                  "environment/assets/payload/given/manual", "environment/assets/payload/render", "tests/setup_payload")
TOOLS = ("tests/review/select_set.py",)
WALL_KEYS = ("wall_s", "check_wall_s", "build_s", "max_rss_gb", "t_wall", "wall_time_s", "episode_wall_s", "render_wall_s",
             "recorded_wall_s")
FINGERPRINT_FIELDS = ("states", "sim_time_s", "ticks", "mates", "coupling_events", "hub_deg", "watch", "scrapped", "violations",
                      "stage_score", "reference")
IMAGE_DEFAULT = "racecar-asm-task:current"          # the runtime image the latest ale validate of this tree built (re-tagged after every validate)
RULE_START = "The rule, fixed before any candidate ran"


# ================================================================================================ small helpers
def log(msg):
    print(msg, file=sys.stderr, flush=True)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_digest(root):
    """sha256 over the sorted (relative path, sha256) lines of every file under root (no __pycache__ / *.pyc); None when
    root is not a folder"""
    if not os.path.isdir(root):
        return None
    lines = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for fn in sorted(fns):
            if fn.endswith((".pyc", ".pyo")):
                continue
            p = os.path.join(dp, fn)
            lines.append("%s %s" % (os.path.relpath(p, root).replace(os.sep, "/"), sha256_file(p)))
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def load(path):
    with open(path) as f:
        return json.load(f)


def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
        f.write("\n")
    os.replace(tmp, path)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def roundtrip(x):
    """x as JSON gives it back (numpy scalars to Python, tuples to lists, floats exact)"""
    return json.loads(json.dumps(x, default=_json_default))


def run_path(p, root):
    """p relative to the run root (the folder the selection's runs are kept under), '/'-separated; a path outside
    the root is refused, so the selection records no machine's absolute path"""
    ap, rr = os.path.abspath(p), os.path.abspath(root)
    if os.path.commonpath([ap, rr]) != rr:
        raise SystemExit("%s is not under the run root %s (--root)" % (p, root))
    return os.path.relpath(ap, rr).replace(os.sep, "/")


def from_root(p, root):
    """a recorded run path read back: relative paths under root (default: the working folder), absolute ones as they are"""
    return p if not p or os.path.isabs(p) else os.path.join(root or os.getcwd(), p)


def strip_wall(x):
    if isinstance(x, dict):
        return {k: strip_wall(v) for k, v in x.items() if k not in WALL_KEYS}
    if isinstance(x, list):
        return [strip_wall(v) for v in x]
    return x


def fingerprint(fin):
    """sha256 of finalize's physics fields (FINGERPRINT_FIELDS, wall-clock values removed, canonical JSON): the same
    physics gives the same fingerprint in every environment"""
    doc = strip_wall(roundtrip({k: fin.get(k) for k in FINGERPRINT_FIELDS}))
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def trace_digest(t, q):
    """sha256 of the episode trace's arrays (verify's recorder: t float64, qpos float32 every 4 ticks)"""
    import numpy as np
    T = np.asarray(t, np.float64)
    Q = np.stack([np.asarray(x, np.float32) for x in q]) if len(q) else np.zeros((0, 0), np.float32)
    return {"frames": int(len(T)), "t_sha256": hashlib.sha256(T.tobytes()).hexdigest(),
            "qpos_sha256": hashlib.sha256(np.ascontiguousarray(Q).tobytes()).hexdigest()}


def env_info():
    out = {"python": sys.version.split()[0], "machine": platform.machine(), "kernel": platform.release()}
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"):
                out["cpu"] = ln.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    for mod in ("mujoco", "numpy", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    out["cpu_count"] = os.cpu_count()
    return out


def rule_text():
    doc = __doc__
    i = doc.index(RULE_START)
    j = doc.index("Acceptance of the set")
    return doc[i:j].strip()


def constants():
    return {"N_CANDIDATES": N_CANDIDATES, "SEED_MOD": SEED_MOD, "SEED_MIN": SEED_MIN, "STRATA": list(STRATA), "ENVS": list(ENVS),
            "MAX_SUSTAINED_N": MAX_SUSTAINED_N, "CAPTURE_SHARE": CAPTURE_SHARE, "SEEN_MIN": SEEN_MIN, "SEEN_TOL_M": SEEN_TOL_M,
            "SAMPLES": SAMPLES, "DROP_SHARE": DROP_SHARE, "MARGIN_GAP": MARGIN_GAP}


# ================================================================================================ the freeze
def canonical_digest(obj):
    """sha256 of obj's canonical JSON (keys sorted, separators ',' and ':', ASCII)"""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def grader_values(cfg):
    """grader_config without the publication metadata of the hidden kit (hidden_gt's PUBLICATION_KEYS)"""
    out = dict(cfg)
    out["hidden_gt"] = {k: v for k, v in (cfg.get("hidden_gt") or {}).items() if k not in PUBLICATION_KEYS}
    return out


def grader_values_digest(path):
    """the grader's values of a grader_config.json file (None when missing or not JSON)"""
    try:
        return canonical_digest(grader_values(load(path)))
    except (OSError, ValueError):
        return None


def kit_members_digest(path):
    """(digest, n): the hidden kit by its content: sha256 over the sorted lines '<member name> <sha256 of its bytes>' of
    the archive's file members (folders, member order, owners, times and the compression do not count); (None, 0) when
    the archive is missing or unreadable"""
    if not os.path.isfile(path):
        return None, 0
    lines = []
    try:
        with tarfile.open(path, "r:*") as tf:
            for m in tf:
                if m.isfile():
                    h = hashlib.sha256()
                    f = tf.extractfile(m)
                    for b in iter(lambda: f.read(1 << 20), b""):
                        h.update(b)
                    lines.append("%s %s" % (m.name, h.hexdigest()))
    except (tarfile.TarError, OSError, EOFError):
        return None, 0
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest(), len(lines)


def hash_key(task, key, robot_dir=None):
    """the hash of one entry of a frozen table: '<path>#values' (grader_values_digest), '<path>#members'
    (kit_members_digest), a tree (FROZEN_TREES, ROBOT_TREE: tree_digest) or a file's sha256. The draw's own table of
    2026-11-15 uses plain paths for grader_config.json and the kit archive: they are then their files' sha256 (the rule
    as drawn), so a table is always checked by the rule that wrote it"""
    path, _, kind = key.partition("#")
    full = os.path.join(task, path)
    if kind == "values":
        return grader_values_digest(full)
    if kind == "members":
        return kit_members_digest(full)[0]
    if kind:
        raise ValueError("unknown frozen entry %r" % key)
    if path == ROBOT_TREE:
        return tree_digest(robot_dir or full)
    if path in FROZEN_TREES:
        return tree_digest(full)
    return sha256_file(full) if os.path.isfile(full) else None


def frozen_hashes(task=TASK, robot_dir=None):
    """(files, trees, counts) of the frozen table as amendment A1 defines it"""
    files = {p: hash_key(task, p) for p in FROZEN}
    for p in FROZEN_VALUES:
        files[p + "#values"] = hash_key(task, p + "#values")
    counts = {}
    for p in FROZEN_MEMBERS:
        files[p + "#members"], counts[p + "#members"] = kit_members_digest(os.path.join(task, p))
    trees = {p: tree_digest(os.path.join(task, p)) for p in FROZEN_TREES}
    trees[ROBOT_TREE] = tree_digest(robot_dir or os.path.join(task, ROBOT_TREE))
    return files, trees, counts


def recorded_hashes(task=TASK):
    """the table recorded by hash that never invalidates the set: the publication metadata of the hidden kit (the
    archive's sha256, canonical_digest of hidden_gt's url / sha256 / note) and the documents the agent reads"""
    pub = {KIT_ARCHIVE: sha256_file(os.path.join(task, KIT_ARCHIVE)) if os.path.isfile(os.path.join(task, KIT_ARCHIVE)) else None}
    try:
        hg = load(os.path.join(task, CONFIG_FILE)).get("hidden_gt") or {}
    except (OSError, ValueError):
        hg = {}
    for k in PUBLICATION_KEYS:
        pub["%s#hidden_gt.%s" % (CONFIG_FILE, k)] = canonical_digest(hg.get(k))
    docs = {p: (sha256_file(os.path.join(task, p)) if os.path.isfile(os.path.join(task, p)) else None) for p in RECORDED_FILES}
    for p in RECORDED_TREES:
        docs[p + "/"] = tree_digest(os.path.join(task, p))
    return {"publication": pub, "documents": docs}


def effective_frozen(sel):
    """(files, trees, source): the frozen table in force: the latest amendment's, else the draw's"""
    am = sel.get("amendments") or []
    if am:
        return am[-1]["frozen"], am[-1]["frozen_trees"], "amendment %s (%s)" % (am[-1].get("id"), am[-1].get("date"))
    return sel["frozen"], sel["frozen_trees"], "the draw (%s)" % sel.get("drawn_utc")


def latest_recorded(sel):
    """the latest record of the recorded table: recorded_changes[-1], else the latest amendment's, else the draw's"""
    ch = sel.get("recorded_changes") or []
    if ch:
        return ch[-1]["recorded"]
    am = sel.get("amendments") or []
    if am:
        return am[-1].get("recorded")
    return sel.get("recorded")


def recorded_diff(old, new):
    """{group/key: [old, new]} of the recorded entries that differ"""
    out = {}
    for grp in ("publication", "documents"):
        a, b = (old or {}).get(grp) or {}, (new or {}).get(grp) or {}
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                out["%s/%s" % (grp, k)] = [a.get(k), b.get(k)]
    return out


def check_frozen(sel, task=TASK, robot_dir=None, need_robot=True, quiet=False):
    """SystemExit when an entry of the frozen table in force (effective_frozen) differs (the robot models through
    robot_dir); a change of the recorded table is named on stderr and never fails"""
    files, trees, source = effective_frozen(sel)
    bad = [k for k in files if hash_key(task, k) != files[k]]
    bad += [p + "/" for p in trees if p != ROBOT_TREE and hash_key(task, p) != trees[p]]
    if need_robot and trees.get(ROBOT_TREE) != hash_key(task, ROBOT_TREE, robot_dir):
        bad.append(ROBOT_TREE + "/ (%s)" % (robot_dir or "default"))
    if bad:
        raise SystemExit("the frozen inputs changed since %s: %s" % (source, bad))
    rec = latest_recorded(sel)
    if rec is not None and not quiet:
        d = recorded_diff(rec, recorded_hashes(task))
        if d:
            log("note: recorded (not frozen) entries differ from their latest record, the set stays valid: %s" % sorted(d))
    return True


def grader_config(task=TASK):
    return load(os.path.join(task, "tests", "grader_config.json"))


# ================================================================================================ draw
def draw_seed(seen, urandom=os.urandom):
    """(seed, redraws): 8 bytes little-endian mod 2**63, drawn again below 2**32 or on a repeat"""
    redrawn = 0
    while True:
        s = int.from_bytes(urandom(8), "little") % SEED_MOD
        if s >= SEED_MIN and s not in seen:
            return s, redrawn
        redrawn += 1


def make_candidates(n=N_CANDIDATES, urandom=os.urandom, exclude=()):
    seen, cands, redrawn = set(exclude), [], 0
    for i in range(n):
        s, again = draw_seed(seen, urandom)
        redrawn += again
        seen.add(s)
        cands.append({"id": "c%02d" % i, "draw": i, "round": i // len(STRATA), "stratum": STRATA[i % len(STRATA)], "seed": s,
                      "status": "drawn"})
    return cands, redrawn


def cmd_draw(a):
    if os.path.exists(a.selection):
        raise SystemExit("%s exists: a selection is drawn once (write a new one that supersedes it)" % a.selection)
    files, trees, counts = frozen_hashes(robot_dir=a.robot)
    missing = [p for p, h in list(files.items()) + list(trees.items()) if h is None]
    if missing:
        raise SystemExit("frozen inputs missing: %s" % missing)
    cfg = grader_config()
    recorded = recorded_hashes()
    if cfg["hidden_gt"]["sha256"] != recorded["publication"][KIT_ARCHIVE]:
        raise SystemExit("tests/assets/gt.tar.xz is not the kit grader_config.json pins")
    if counts[KIT_ARCHIVE + "#members"] != int(cfg["hidden_gt"]["files"]):
        raise SystemExit("tests/assets/gt.tar.xz holds %d files, grader_config.json pins %d" % (counts[KIT_ARCHIVE + "#members"], cfg["hidden_gt"]["files"]))
    supersedes, old_seeds = None, ()
    if a.supersedes:                       # a new selection replacing an earlier one (move the old file aside first)
        old = load(a.supersedes)
        if not a.why:
            raise SystemExit("--supersedes needs --why")
        old_seeds = [c["seed"] for c in old.get("candidates", [])]
        supersedes = {"file": os.path.relpath(os.path.abspath(a.supersedes), TASK), "sha256": sha256_file(a.supersedes),
                      "drawn_utc": old.get("drawn_utc"), "episodes_sha256": (old.get("set") or {}).get("episodes_sha256"), "why": a.why}
    cands, redrawn = make_candidates(exclude=old_seeds)
    sel = {"format": SEL_FORMAT, "task": "racecar_full_assembly_tenth_scale_ur10e", "drawn_utc": utc_now(),
           "rule": "tests/review/select_set.py (module docstring; copied into rule_text)", "rule_text": rule_text(),
           "constants": constants(), "frozen": files, "frozen_counts": counts, "frozen_trees": trees, "recorded": recorded,
           "tools": {p: sha256_file(os.path.join(TASK, p)) for p in TOOLS},
           "kit_sha256": cfg["hidden_gt"]["sha256"], "kit_files": cfg["hidden_gt"]["files"],
           "source": "os.urandom: 8 bytes little-endian per seed mod 2**63, drawn again below 2**32 or on a repeat; candidate i -> "
                     "stratum STRATA[i %% 4], round i // 4 (%d seeds redrawn)" % redrawn,
           "expectation": a.expectation or None, "candidates": cands, "runs": [], "set": None}
    if supersedes:
        sel["supersedes"] = supersedes       # none of its seeds is drawn again
    os.makedirs(os.path.dirname(os.path.abspath(a.selection)), exist_ok=True)
    save(a.selection, sel)
    print("drew %d candidates (%d seeds redrawn) -> %s" % (len(cands), redrawn, a.selection))
    return 0


# ================================================================================================ check, record
def freeze_status(sel, task=TASK, robot_dir=None):
    """the freeze in one dict: the table in force and its source, whether every entry holds, episodes.json against the
    set, and the recorded entries that differ from their latest record (notes: never a failure)"""
    files, trees, source = effective_frozen(sel)
    bad = [k for k in files if hash_key(task, k) != files[k]]
    bad += [p + "/" for p in trees if hash_key(task, p, robot_dir) != trees[p]]
    eps = os.path.join(task, "tests", "hidden", "episodes.json")
    want = (sel.get("set") or {}).get("episodes_sha256")
    rec = latest_recorded(sel)
    return {"table": source, "files": len(files), "trees": len(trees), "frozen_ok": not bad, "frozen_changed": bad,
            "episodes_is_the_set": bool(want) and os.path.isfile(eps) and sha256_file(eps) == want,
            "amendments": [x.get("id") for x in sel.get("amendments") or []],
            "recorded_changed": sorted(recorded_diff(rec, recorded_hashes(task))) if rec is not None else None,
            "recorded_changes_entered": len(sel.get("recorded_changes") or [])}


def cmd_check(a):
    sel = load(a.selection)
    st = freeze_status(sel, robot_dir=a.robot)
    if a.json:
        save(a.json, st)
    print("frozen table (%s): %d entries, %d trees: %s; episodes.json %s; recorded entries changed since their latest record: %s"
          % (st["table"], st["files"], st["trees"], "unchanged" if st["frozen_ok"] else "CHANGED %s" % st["frozen_changed"],
             "is the selection's set" if st["episodes_is_the_set"] else "IS NOT the selection's set",
             st["recorded_changed"] or "none"))
    return 0 if st["frozen_ok"] and st["episodes_is_the_set"] else 1


def cmd_record(a):
    """enter the recorded table's current hashes, with the date and the reason, after the frozen table is checked"""
    sel = load(a.selection)
    check_frozen(sel, robot_dir=a.robot, quiet=True)
    new = recorded_hashes()
    changed = recorded_diff(latest_recorded(sel), new)
    tool = {p: sha256_file(os.path.join(TASK, p)) for p in TOOLS}
    last_tool = ((sel.get("recorded_changes") or [{}])[-1].get("tools") if sel.get("recorded_changes") else
                 ((sel.get("amendments") or [{}])[-1].get("tool_at_writing") if sel.get("amendments") else sel.get("tools")))
    if not changed and tool == last_tool:
        print("nothing to record: the recorded table and the tool are as last recorded")
        return 0
    entry = {"utc": utc_now(), "why": a.why, "changed": changed, "recorded": new, "tools": tool,
             "frozen_checked": "%s: unchanged" % effective_frozen(sel)[2]}
    if tool != last_tool:
        entry["tools_changed"] = {p: [(last_tool or {}).get(p), tool[p]] for p in tool if (last_tool or {}).get(p) != tool[p]}
    sel.setdefault("recorded_changes", []).append(entry)
    save(a.selection, sel)
    print("recorded %d changed entr%s%s -> %s" % (len(changed), "y" if len(changed) == 1 else "ies",
                                                  " and the tool's new sha256" if tool != last_tool else "", a.selection))
    return 0


# ================================================================================================ kit, digests, layout draws
def ensure_kit(kit, out_dir, task=TASK):
    """the hidden kit folder: `kit` when given, else tests/assets/gt.tar.xz unpacked under out_dir/_kit after the same
    checks as verify.fetch_hidden_gt (sha256 and file count of grader_config.json, members under gt/)"""
    if kit:
        return os.path.abspath(kit)
    spec = grader_config(task)["hidden_gt"]
    gt = os.path.join(os.path.abspath(out_dir), "_kit", "gt")
    stamp = os.path.join(os.path.abspath(out_dir), "_kit", "sha256")
    if os.path.isfile(os.path.join(gt, "kit.json")) and os.path.isfile(stamp) and open(stamp).read().strip() == spec["sha256"]:
        return gt
    src = os.path.join(task, "tests", "assets", "gt.tar.xz")
    got = sha256_file(src)
    if got != spec["sha256"]:
        raise SystemExit("tests/assets/gt.tar.xz has sha256 %s; grader_config.json pins %s" % (got, spec["sha256"]))
    part = os.path.join(os.path.abspath(out_dir), "_kit.part")
    shutil.rmtree(part, ignore_errors=True)
    os.makedirs(part)
    with tarfile.open(src, "r:xz") as tf:
        bad = [m.name for m in tf.getmembers() if not (m.isfile() or m.isdir()) or not (m.name == "gt" or m.name.startswith("gt/"))]
        if bad:
            raise SystemExit("the kit tarball holds unexpected members %s" % bad[:5])
        tf.extractall(part, filter="data")
    n = sum(len(fns) for _dp, _dn, fns in os.walk(os.path.join(part, "gt")))
    if n != int(spec["files"]):
        raise SystemExit("the kit unpacked to %d files; grader_config.json expects %d" % (n, int(spec["files"])))
    shutil.rmtree(os.path.join(os.path.abspath(out_dir), "_kit"), ignore_errors=True)
    os.makedirs(os.path.join(os.path.abspath(out_dir), "_kit"))
    os.rename(os.path.join(part, "gt"), gt)
    shutil.rmtree(part, ignore_errors=True)
    open(stamp, "w").write(spec["sha256"] + "\n")
    return gt


def _yaw_deg(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _wrap180(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def layout_draws(lay, cfg_layout):
    """the drawn quantities of a layout, as offsets from the nominal (for the hollowing check)"""
    L = cfg_layout
    out = {}
    st = lay["stand"]
    nx, ny = L["stand"]["nominal_xy"]
    out["stand_dx_mm"], out["stand_dy_mm"] = 1e3 * (st["xy"][0] - nx), 1e3 * (st["xy"][1] - ny)
    out["stand_dyaw_deg"] = _wrap180(st["yaw_deg"] - 180.0)
    cl = lay.get("clamp") or {}
    out["clamp_dx_mm"], out["clamp_dy_mm"], out["clamp_dyaw_deg"] = 1e3 * cl.get("dx_m", 0.0), 1e3 * cl.get("dy_m", 0.0), cl.get("dyaw_deg", 0.0)
    for name in ("shelf", "tray"):
        f = lay["fixtures"][name]
        fx, fy = L[name]["nominal_xy"]
        out[name + "_dx_mm"], out[name + "_dy_mm"] = 1e3 * (f["xy"][0] - fx), 1e3 * (f["xy"][1] - fy)
        out[name + "_yaw_deg"] = _wrap180(f["yaw_deg"])
    for u, p in (lay.get("plays") or {}).items():
        out["play_%s_mm" % u] = math.hypot(p.get("dx_mm", 0.0), p.get("dy_mm", 0.0))
        out["play_%s_dyaw_deg" % u] = p.get("dyaw_deg", 0.0)
    for u, (pos, q) in (lay.get("parts") or {}).items():
        if u.startswith("WHL") or u == "BAT":
            out["%s_x_m" % u], out["%s_y_m" % u] = pos[0], pos[1]
            out["%s_r_m" % u] = math.hypot(pos[0], pos[1])
            out["%s_yaw_deg" % u] = _yaw_deg(q)
    for hb, a in (lay.get("hub_deg") or {}).items():
        out["%s_off_hex_deg" % hb] = abs(_wrap180(float(a) * 6.0) / 6.0)       # distance to the nearest multiple of 60 deg
    out["lead_end_deg"] = lay.get("lead_end_deg")
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def cmd_digests(a):
    sel = load(a.selection)
    check_frozen(sel, robot_dir=a.robot)
    from arm.assembly_car import layout as LY
    from arm.assembly_car.cell import load_kit
    out = a.work or tempfile.mkdtemp(prefix="p213-digests-")
    kit_dir = ensure_kit(a.kit, out)
    kit = load_kit(kit_dir)
    uv = LY.unit_geometry(kit)
    cfg = grader_config()
    plan = load(os.path.join(TESTS, "hidden", "bench_plan.json"))
    for c in sel["candidates"]:
        lay = LY.episode_layout(c["seed"], plan, kit, uv=uv, cfg=cfg["layout"], stratum=c["stratum"])
        dg = LY.layout_digest(lay)
        if c.get("layout_digest") not in (None, dg):
            raise SystemExit("%s: the digest changed (%s -> %s)" % (c["id"], c["layout_digest"], dg))
        c["layout_digest"] = dg
        c["layout"] = layout_draws(lay, cfg["layout"])
        c["layout"]["layouts_tried"] = (lay.get("jitter") or {}).get("layouts_tried")
        print("%s %-12s seed %d... digest %s... tries %s" % (c["id"], c["stratum"], c["seed"] % 1000, dg[:12], c["layout"]["layouts_tried"]))
    sel["digests"] = {"utc": utc_now(), "env": env_info(), "kit_json_sha256": sha256_file(os.path.join(kit_dir, "kit.json"))}
    save(a.selection, sel)
    return 0


# ================================================================================================ one episode (host / pod / image view)
def decode_depth(b64, unit_m=0.001):
    """the 16-bit PNG sensor depth of a get_camera reply -> float metres (0 = invalid)"""
    import numpy as np
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    return np.asarray(im, dtype=np.float64) * float(unit_m)


def mesh_world(m, d, g):
    import numpy as np
    k = int(m.geom_dataid[g])
    va, vn = int(m.mesh_vertadr[k]), int(m.mesh_vertnum[k])
    fa, fn = int(m.mesh_faceadr[k]), int(m.mesh_facenum[k])
    V = np.asarray(m.mesh_vert[va:va + vn], float) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    F = np.asarray(m.mesh_face[fa:fa + fn], int)
    return V, F


def unit_visual_geoms(srv, unit):
    """the visual mesh geoms (type mesh, group 2, no contacts) of every body of a unit (the geoms the camera sees)"""
    m = srv.model
    bodies = set(b for b, u in srv.cell.unit_of_body.items() if u == unit)
    return [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies and int(m.geom_type[g]) == 7 and int(m.geom_group[g]) == 2
            and not (m.geom_contype[g] or m.geom_conaffinity[g])]


def unit_samples(srv, unit, n, rng):
    """area-weighted samples on the unit's visual meshes: (P (n, 3), area m^2)"""
    import numpy as np
    parts = [mesh_world(srv.model, srv.data, g) for g in unit_visual_geoms(srv, unit)]
    areas = []
    for V, F in parts:
        tri = V[F]
        areas.append(0.5 * float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()))
    tot = float(sum(areas))
    if tot <= 0:
        return np.zeros((0, 3)), 0.0
    Ps = []
    for (V, F), k in zip(parts, rng.multinomial(n, np.asarray(areas) / tot)):
        tri = V[F]
        cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = 0.5 * np.linalg.norm(cr, axis=1)
        if int(k) <= 0 or area.sum() <= 0:
            continue
        idx = rng.choice(len(F), size=int(k), p=area / area.sum())
        r1, r2 = rng.random(int(k)), rng.random(int(k))
        s = np.sqrt(r1)
        w0, w1, w2 = 1 - s, s * (1 - r2), s * r2
        Ps.append(w0[:, None] * tri[idx, 0] + w1[:, None] * tri[idx, 1] + w2[:, None] * tri[idx, 2])
    return (np.concatenate(Ps) if Ps else np.zeros((0, 3))), tot


def seen_fraction(cam, P, sensor_depth, tol=SEEN_TOL_M):
    """share of samples P that project into the frame where the sensor depth is valid and within tol of their depth"""
    import numpy as np
    if not len(P):
        return 0.0
    H, W = sensor_depth.shape
    uv, z = cam.project(P, distort=True)
    with np.errstate(invalid="ignore"):
        ui, vi = np.round(uv[:, 0]), np.round(uv[:, 1])
    ok = np.isfinite(ui) & np.isfinite(vi) & (z > 0)
    ui = np.where(ok, ui, -1).astype(int)
    vi = np.where(ok, vi, -1).astype(int)
    inside = ok & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    seen = np.zeros(len(P), bool)
    idx = np.where(inside)[0]
    sd = sensor_depth[vi[idx], ui[idx]]
    seen[idx] = (sd > 0) & (np.abs(sd - z[idx]) < tol)
    return float(seen.mean())


def top_view(srv, reply, seed, n=SAMPLES):
    """what the top camera's frame (a get_camera reply: the agent's own frame) shows of every loose unit"""
    import numpy as np
    from arm.assembly_car import render_twin as RT
    if not (isinstance(reply, dict) and reply.get("ok")):
        return {"error": "no frame: %s" % (reply,)}
    depth = decode_depth(reply["sensor_depth_u16_mm_b64"], reply.get("depth_unit_m", 0.001))
    cam = RT.camera_spec(srv, "top")
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, 0xC0E])
    units = {}
    for u in srv.cell.loose:
        P, _area = unit_samples(srv, u, n, rng)
        units[u] = round(seen_fraction(cam, P, depth), 4)
    least = min(units, key=lambda u: units[u])
    return {"seen": units, "least": [least, units[least]], "frame_index": int(reply.get("frame_index", -1)), "renderer": srv.renderer,
            "valid_depth_share": round(float((depth > 0).mean()), 4), "size": [int(reply.get("width", 0)), int(reply.get("height", 0))]}


def outcome(fin, ref_return=None):
    """the compact record of one episode from a finalize() summary (in-process or verify's episodes.json)"""
    ref = fin.get("reference") or None
    steps = (ref or {}).get("steps") or []
    caps, timeline = {}, []
    for s in steps:
        c = s.get("capture")
        timeline.append({"slot": s.get("slot"), "unit": s.get("unit"), "state": s.get("state"), "t0_s": s.get("t0_s"), "t1_s": s.get("t1_s"),
                         "sim_s": s.get("sim_s"), "attempts": len(s.get("attempts") or []), "retry": bool(s.get("retry")),
                         "grasp": ((s.get("attempts") or [{}])[-1] or {}).get("grasp"), "capture_t": None if not c else c.get("t")})
        if c:
            e = {"ratio_max": c["ratio_max"], "ratio": c.get("ratio"), "lateral_mm": c.get("lateral_mm"), "tilt_deg": c.get("tilt_deg"),
                 "clock_deg": c.get("clock_deg"), "axial_mm": c.get("axial_mm"), "t": c.get("t")}
            caps.setdefault(s["slot"], []).append(e)
    capture = {sid: max(v, key=lambda e: e["ratio_max"]) for sid, v in caps.items()}
    for sid, v in caps.items():
        capture[sid] = dict(capture[sid], n=len(v))
    w = fin.get("watch") or {}
    unpl = (ref or {}).get("unplanned_contacts") or []
    states = fin.get("states") or {}
    rec = {"reference_ran": ref is not None, "states": states, "n_locked": sum(1 for s in SLOTS if states.get(s) == "LOCKED"),
           "stage_score": fin.get("stage_score"), "n_achieved": fin.get("n_achieved"), "hard_ok": bool(fin.get("hard_ok")),
           "max_sustained_N": w.get("max_part_force_sustained_N"), "max_sustained_pair": w.get("max_part_force_sustained_pair"),
           "max_robot_link_N": w.get("max_robot_link_force_N"), "unplanned_contacts": len(unpl),
           "unplanned": [{k: c.get(k) for k in ("phase", "slot", "bodies", "force_N", "t")} for c in unpl][:8],
           "violations": [v.get("what") for v in (fin.get("violations") or [])][:8], "scrapped": sorted((fin.get("scrapped") or {}).keys()),
           "capture": capture, "steps": timeline, "sim_time_s": fin.get("sim_time_s"), "ticks": fin.get("ticks"),
           "hub_deg_end": fin.get("hub_deg"), "done_reason": fin.get("done_reason"), "fingerprint": fingerprint(fin)}
    if ref_return is not None:
        rec["reference_return"] = {k: ref_return.get(k) for k in ("ok", "score", "steps", "unplanned_contacts", "sim_s", "error") if k in ref_return}
    return rec


def run_episode(sel, cid, env, kit_dir, robot_dir, out, renderer="mujoco", view_only=False, sim_cap=None, candidate=None):
    """one candidate in-process (module docstring, rule 3); writes out/<cid>/result.json (complete: true) and keeps the
    finalize summary and the trace arrays next to it"""
    import numpy as np
    from arm.assembly_car import layout as LY
    from arm.assembly_car.cell import load_kit
    from arm.assembly_car.server import CarInfoServer
    import episode_trace
    c = candidate or [x for x in sel["candidates"] if x["id"] == cid][0]
    d = os.path.join(os.path.abspath(out), cid)
    os.makedirs(d, exist_ok=True)
    lf = open(os.path.join(d, "log.txt"), "w")

    def say(s):
        lf.write(str(s) + "\n")
        lf.flush()
    t0 = time.time()
    cfg = grader_config()
    cfg["agent_renderer"] = renderer
    if sim_cap:
        cfg["sim_cap_s"] = float(sim_cap)
    plan = load(os.path.join(TESTS, "hidden", "bench_plan.json"))
    kit = load_kit(kit_dir)
    uv = LY.unit_geometry(kit)
    rec = {"format": "racecar-asm-selection-run/1", "cand": cid, "env": env, "seed": c["seed"], "stratum": c["stratum"],
           "view_only": bool(view_only), "sim_cap_s": cfg["sim_cap_s"], "renderer": renderer, "env_info": env_info(),
           "robot_tree": tree_digest(robot_dir), "kit_json_sha256": sha256_file(os.path.join(kit_dir, "kit.json")),
           "tool_sha256": sha256_file(os.path.abspath(__file__)), "utc_start": utc_now(), "complete": False}
    try:
        srv = CarInfoServer(kit_dir, robot_dir, plan, config=cfg, mode="hidden", privileged=not view_only, render=True, log=say, kit=kit, uv=uv)
        r = srv.reset(c["seed"], stratum=c["stratum"])
        rec["layout_digest"] = r["layout_digest"]
        rec["digest_ok"] = r["layout_digest"] == c.get("layout_digest")
        rec["build_s"] = round(time.time() - t0, 1)
        if not rec["digest_ok"]:
            say("layout digest %s != recorded %s" % (r["layout_digest"], c.get("layout_digest")))
        else:
            trace = episode_trace.EpisodeTrace(srv, every_ticks=int((cfg.get("episode_trace") or {}).get("every_ticks", 4)), log=say)
            th, nt = hashlib.sha256(), [0]

            def tick_hash():
                th.update(srv.data.qpos.tobytes())
                th.update(srv.data.qvel.tobytes())
                nt[0] += 1
            srv.cell.hooks.append(tick_hash)
            srv.wall_start = time.time()
            top = srv.handle("get_camera", {"name": "top", "depth": True})
            rec["view"] = top_view(srv, top, c["seed"])
            say("top frame: least seen %s" % (rec["view"].get("least"),))
            if not view_only:
                wrist = srv.handle("get_camera", {"name": "wrist", "depth": True})
                rec["wrist_frame_ok"] = bool(isinstance(wrist, dict) and wrist.get("ok"))
                t1 = time.time()
                ra = srv.handle("reference_assemble", {})
                rec["reference_wall_s"] = round(time.time() - t1, 1)
                srv.handle("finish", {})
                fin = srv.finalize(None)
                trace.detach()
                rec.update(outcome(fin, ra))
                rec["tick_sha256"] = th.hexdigest()
                rec["ticks_hashed"] = nt[0]
                rec["trace"] = trace_digest(trace.t, trace.q)
                np.savez_compressed(os.path.join(d, "trace.npz"), t=np.asarray(trace.t, np.float64), qpos=np.stack(trace.q))
                save(os.path.join(d, "finalize.json"), roundtrip({k: v for k, v in fin.items() if k != "events"}))
                say("states %s score %.4f unplanned %d sustained %.1f N sim %.1f s" % (
                    rec["n_locked"], rec["stage_score"], rec["unplanned_contacts"], rec["max_sustained_N"] or 0.0, rec["sim_time_s"]))
            else:
                trace.detach()
    except Exception:  # noqa: BLE001 - recorded: the run completed with an error (a failure of this environment's run)
        rec["error"] = traceback.format_exc()[-3000:]
        say(rec["error"])
    rec["wall_s"] = round(time.time() - t0, 1)
    try:
        import resource
        rec["max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    except Exception:  # noqa: BLE001
        pass
    rec["utc_end"] = utc_now()
    rec["complete"] = True
    save(os.path.join(d, "result.json"), rec)
    lf.close()
    return rec


def cmd_episode(a):
    sel = load(a.selection)
    check_frozen(sel, robot_dir=a.robot)
    rec = run_episode(sel, a.cand, a.env, a.kit, a.robot, a.out, renderer=a.renderer, view_only=a.view_only)
    return 0 if not rec.get("error") else 1


def is_done(out, cid):
    p = os.path.join(out, cid, "result.json")
    try:
        return bool(load(p).get("complete"))
    except Exception:  # noqa: BLE001
        return False


def cmd_run(a):
    """every candidate not done yet, P at a time, each in its own interpreter (`episode` / `view`); resumable"""
    sel = load(a.selection)
    robot = os.path.abspath(a.robot or os.path.join(TASK, ROBOT_TREE))
    check_frozen(sel, robot_dir=robot)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    kit = ensure_kit(a.kit, out)
    if any(not c.get("layout_digest") for c in sel["candidates"]):
        raise SystemExit("run `digests` first")
    ids = [c["id"] for c in sel["candidates"] if (not a.only or c["id"] in a.only.split(","))]
    todo = [i for i in ids if not is_done(out, i)]
    log("[run] %s: %d candidates, %d done, %d to run, %d at a time" % (a.env, len(ids), len(ids) - len(todo), len(todo), a.P))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    if a.view_only:
        env.setdefault("LP_NUM_THREADS", "2")
    running, t0 = {}, time.time()
    while todo or running:
        while todo and len(running) < a.P:
            cid = todo.pop(0)
            cmd = (["nice", "-n", str(a.nice)] if a.nice else []) + [sys.executable, "-I", "-B", os.path.abspath(__file__), "episode",
                                                                     "--selection", os.path.abspath(a.selection), "--cand", cid, "--env", a.env,
                                                                     "--kit", kit, "--robot", robot, "--out", out, "--renderer", a.renderer]
            if a.view_only:
                cmd.append("--view-only")
            os.makedirs(os.path.join(out, cid), exist_ok=True)
            fo = open(os.path.join(out, cid, "stdout.txt"), "w")
            running[cid] = (subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, env=env, cwd="/tmp"), fo, time.time())
        time.sleep(2.0)
        for cid in list(running):
            p, fo, ts = running[cid]
            if p.poll() is None:
                continue
            fo.close()
            with open(os.path.join(out, "done.txt"), "a") as f:
                f.write("%s exit %d wall %.0f s %s\n" % (cid, p.returncode, time.time() - ts, utc_now()))
            log("[run] %s exit %d (%.0f s); %d running, %d to go, %.0f s" % (cid, p.returncode, time.time() - ts, len(running) - 1, len(todo),
                                                                            time.time() - t0))
            del running[cid]
    return 0


# ================================================================================================ the task image
def setup_payload_standin(dst):
    """a stand-in of tests/setup_payload/ while the task has none (the givens generator writes the real one): given/lib =
    byte copies of the client library's vendored modules (the earlier task's layout), given/cell/README.md"""
    lib = os.path.join(dst, "given", "lib")
    arm = os.path.join(lib, "arm")
    os.makedirs(arm)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    for f in ("__init__.py", "chain_model.py", "frames.py"):
        shutil.copyfile(os.path.join(TESTS, "arm", f), os.path.join(arm, f))
    for pkg in ("assembly_cell", "assembly_info", "assembly_six"):
        shutil.copytree(os.path.join(TESTS, "arm", pkg), os.path.join(arm, pkg), ignore=ign)
    os.makedirs(os.path.join(arm, "assembly_quad"))
    for f in ("__init__.py", "gripper_2f85.py", "ur_controller.py", "ur_robot.py", "ur_ik.py"):
        shutil.copyfile(os.path.join(TESTS, "arm", "assembly_quad", f), os.path.join(arm, "assembly_quad", f))
    for pkg in ("common", "render"):
        shutil.copytree(os.path.join(TESTS, pkg), os.path.join(lib, pkg), ignore=ign)
    os.makedirs(os.path.join(dst, "given", "cell"))
    with open(os.path.join(dst, "given", "cell", "README.md"), "w") as f:
        f.write("stand-in of the cell documents (hidden-set selection stage)\n")


def given_hashes(roots):
    out = {}
    for root in roots:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d != "__pycache__"]
            for fn in fns:
                p = os.path.join(dp, fn)
                out[os.path.relpath(p, root)] = sha256_file(p)
    return dict(sorted(out.items()))


def build_stage(root, set_doc=None):
    """root/tests = a copy of the task's tests/ (what the engine uploads to the verifier) + hidden/episodes.json = set_doc
    (when given) + the givens' stand-ins when the task has none yet; returns (tests dir, notes)"""
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    tests = os.path.join(root, "tests")
    shutil.copytree(TESTS, tests, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "gt", ".gt.part", ".gt.tar.xz", "logs", "out",
                                                                  "verdict.local.json"))
    notes = []
    if set_doc is not None:
        with open(os.path.join(tests, "hidden", "episodes.json"), "w") as f:
            json.dump(set_doc, f, indent=1)
    if not os.path.isdir(os.path.join(tests, "setup_payload")):
        setup_payload_standin(os.path.join(tests, "setup_payload"))
        notes.append("setup_payload: stand-in (given/lib byte copies of the vendored client library)")
    if not os.path.isfile(os.path.join(tests, "given_hashes.json")):
        gh = given_hashes([os.path.join(tests, "setup_payload"), os.path.join(TASK, "environment", "assets", "payload")])
        with open(os.path.join(tests, "given_hashes.json"), "w") as f:
            json.dump(gh, f, indent=1)
        notes.append("given_hashes.json: computed over setup_payload/ + environment/assets/payload/ (%d files)" % len(gh))
    return tests, notes


IMAGE_VERIFY = r'''set -u
export ALE_HOME=/home/user
bash /opt/ale/solution/solve.sh 2> /out/solve.stderr
chown -R root:root /home/user/submission
sleep 1
mkdir -p /opt/ale/stage && cp -r /opt/ale/tests/. /opt/ale/stage/
cd /opt/ale/stage
export ROBOTICS_GRADER_VERDICT_PATH=/out/verdict.json
T0=$(date +%s.%N)
bash test.sh > /out/verify.stdout 2> /out/verify.stderr
echo $? > /out/exit_code
T1=$(date +%s.%N)
echo "$T0 $T1" > /out/wall
if [ -d /opt/arb-verify ]; then cp -r /opt/arb-verify /out/arb-verify; fi
python3 -c "import json, platform; print(json.dumps({'cpu': [l.split(':',1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')][0], 'nproc': __import__('os').cpu_count(), 'python': platform.python_version(), 'mujoco': __import__('mujoco').__version__, 'numpy': __import__('numpy').__version__, 'scipy': __import__('scipy').__version__}))" > /out/env.json 2>/dev/null || true
chmod -R a+rwX /out
'''

IMAGE_VIEW = r'''set -eu
R=/opt/arb; rm -rf "$R"; mkdir -p "$R"
cp -r /opt/ale/tests/setup_payload/. "$R/"; cp -r /opt/task-assets/payload/. "$R/"; chown -R root:root "$R"; chmod -R u+rwX,go+rX,go-w "$R"
export ARB_AGENT_RENDER_DIR=$R/render ARB_ROOT_PAYLOAD=$R
mkdir -p /tmp/sel && cp /opt/ale/tests/hidden/selection.json /tmp/sel/selection.json
python3 -I -B /opt/ale/tests/review/select_set.py run --selection /tmp/sel/selection.json --env image --view-only --renderer filament \
  --out /out/view --robot $R/given/robot -P "${VIEW_P:-3}" --nice 0 > /out/view_run.log 2>&1 || echo "view run exit $?" >> /out/view_run.log
chmod -R a+rwX /out
'''


def docker_image_id(image):
    try:
        return subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, text=True, timeout=60,
                              env=dict(os.environ, DOCKER_HOST="unix:///var/run/docker.sock")).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def docker_run(image, tests_dir, out_dir, script, tag, extra_env=()):
    """the task sandbox's resources (8 CPUs, 24 GB, no network), tests/ read-only at /opt/ale/tests, solution/ read-only at
    /opt/ale/solution, out_dir at /out"""
    name = "p213-%s-%d-%d" % (tag, os.getpid(), int(time.time()))
    cmd = ["docker", "run", "--rm", "--init", "--network", "none", "--cpus", "8", "--memory", "24576m", "--memory-swap", "24576m",
           "--name", name, "--label", "racecar-asm=p2-13", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "MUJOCO_GL=osmesa"]
    for kv in extra_env:
        cmd += ["-e", kv]
    cmd += ["-v", "%s:/opt/ale/tests:ro" % tests_dir, "-v", "%s:/opt/ale/solution:ro" % os.path.join(TASK, "solution"),
            "-v", "%s:/out:rw" % out_dir, "-u", "root", "-w", "/opt/ale/tests", image, "bash", "-c", script]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, DOCKER_HOST="unix:///var/run/docker.sock"))
    return {"container": name, "docker_rc": p.returncode, "docker_stderr_tail": p.stderr[-800:], "wall_s": round(time.time() - t0, 1)}


def _oracle_lines(text):
    """the oracle program's '[oracle] reference: {...}' reply in its log"""
    for ln in reversed((text or "").splitlines()):
        if "[oracle] reference: " in ln:
            try:
                return ast.literal_eval(ln.split("[oracle] reference: ", 1)[1].strip())
            except (ValueError, SyntaxError):
                return {"unparsed": ln[-300:]}
    return None


def read_image_run(res_dir, set_doc):
    """the per-episode records of one verify run in the image (its out folder): verify's episodes.json, the traces, the
    oracle's log and the verdict"""
    import numpy as np
    rd = lambda n: open(os.path.join(res_dir, n), errors="replace").read() if os.path.isfile(os.path.join(res_dir, n)) else ""  # noqa: E731
    verdict = json.loads(rd("verdict.json")) if rd("verdict.json").strip() else None
    stderr = rd("verify.stderr")
    eps_path = os.path.join(res_dir, "arb-verify", "out", "episodes.json")
    eps = load(eps_path) if os.path.isfile(eps_path) else []
    envj = json.loads(rd("env.json")) if rd("env.json").strip() else {}
    out = {"exit_code": rd("exit_code").strip(), "verdict_reward": ((verdict or {}).get("rewards") or {}).get("reward"),
           "metrics": {k: v for k, v in ((verdict or {}).get("metrics") or {}).items()
                       if k in ("assembly_score", "contract_ok", "privileged_mode", "all_mates_rate", "hard_violation_rate", "layout_digest_checked",
                                "hidden_gt_local", "netguard_active", "max_part_force_sustained_N", "mean_sim_time_s", "mean_episode_wall_s",
                                "verify_wall_s", "illegal_call_count", "runner_uid")},
           "task_errors": [l for l in stderr.splitlines() if "TASK ERROR" in l], "env_info": envj, "episodes": []}
    for k, e in enumerate(set_doc["episodes"], 1):
        s = [x for x in eps if x.get("seed") == e["seed"]]
        if not s:
            out["episodes"].append({"seed": e["seed"], "missing": True})
            continue
        s = s[0]
        logtxt = rd(os.path.join("arb-verify", "logs", "agent_ep%02d.log" % k))
        rec = {"seed": e["seed"], "stratum": e["stratum"], "episode": k, "layout_digest": s.get("layout_digest"),
               "digest_ok": s.get("layout_digest") == e["layout_digest"] and bool(s.get("layout_digest_checked")),
               "episode_wall_s": s.get("episode_wall_s"), "end_reason": s.get("end_reason"), "camera_frames": s.get("camera_frames"),
               "renderer": s.get("renderer"), "privileged_mode": s.get("privileged_mode")}
        rec.update(outcome(s, _oracle_lines(logtxt)))
        tp = os.path.join(res_dir, "arb-verify", "out", "trace", "agent_ep%02d" % k, "trace.npz")
        if os.path.isfile(tp):
            z = np.load(tp)
            rec["trace"] = trace_digest(z["t"], list(z["qpos"]))
            rec["trace_file"] = tp
        out["episodes"].append(rec)
    return out


def set_of(cands):
    return {"format": SET_FORMAT, "episodes": [{"seed": c["seed"], "stratum": c["stratum"], "layout_digest": c["layout_digest"]} for c in cands]}


def cmd_image(a):
    """one verify run in the task image of a round's four candidates (rule 3)"""
    sel = load(a.selection)
    check_frozen(sel, need_robot=False)
    cands = sorted([c for c in sel["candidates"] if c["round"] == a.round], key=lambda c: STRATA.index(c["stratum"]))
    if len(cands) != len(STRATA) or any(not c.get("layout_digest") for c in cands):
        raise SystemExit("round %d: not four candidates with digests" % a.round)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    doc = set_of(cands)
    doc["note"] = "candidate round %d of the hidden-set selection (tests/review/select_set.py image)" % a.round
    tests, notes = build_stage(os.path.join(out, "stage"), doc)
    res = os.path.join(out, "res")
    shutil.rmtree(res, ignore_errors=True)
    os.makedirs(res)
    log("[image] round %d (%s) in %s ..." % (a.round, ", ".join(c["id"] for c in cands), a.image))
    run = docker_run(a.image, tests, res, IMAGE_VERIFY, "r%d" % a.round)
    got = read_image_run(res, doc)
    got.update({"round": a.round, "ids": [c["id"] for c in cands], "image": a.image, "image_id": docker_image_id(a.image),
                "stage_notes": notes, "docker": run, "tool_sha256": sha256_file(os.path.abspath(__file__)), "utc": utc_now(),
                "kind": "candidates", "complete": not run["docker_rc"] and bool(got["episodes"])})
    save(os.path.join(out, "image_run.json"), got)
    log("[image] round %d: exit %s reward %s, %s in %.0f s" % (a.round, got["exit_code"], got["verdict_reward"],
                                                             [(e.get("seed", 0) % 1000, e.get("n_locked"), e.get("stage_score")) for e in got["episodes"]],
                                                             run["wall_s"]))
    if not a.keep_stage:
        shutil.rmtree(os.path.join(out, "stage", "tests", "assets"), ignore_errors=True)
    return 0 if got["complete"] else 1


def cmd_image_smoke(a):
    """the image path on a non-candidate episode (seed / stratum given, one episode, the stage's sim cap lowered): proves
    the stage, the oracle submission, verify, the trace and the parsing before the candidates run; never a candidate"""
    from arm.assembly_car import layout as LY
    from arm.assembly_car.cell import load_kit
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    kit = load_kit(ensure_kit(a.kit, out))
    plan = load(os.path.join(TESTS, "hidden", "bench_plan.json"))
    dg = LY.layout_digest(LY.episode_layout(a.seed, plan, kit, uv=LY.unit_geometry(kit), cfg=grader_config()["layout"], stratum=a.stratum))
    doc = {"format": SET_FORMAT, "episodes": [{"seed": a.seed, "stratum": a.stratum, "layout_digest": dg}],
           "note": "SMOKE set of tests/review/select_set.py image-smoke (not a candidate, not the hidden set)"}
    tests, notes = build_stage(os.path.join(out, "stage"), doc)
    cfg = load(os.path.join(tests, "grader_config.json"))
    cfg["episodes"], cfg["sim_cap_s"] = 1, float(a.sim_cap)
    with open(os.path.join(tests, "grader_config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    notes.append("smoke: grader_config episodes 1, sim_cap_s %s" % a.sim_cap)
    res = os.path.join(out, "res")
    shutil.rmtree(res, ignore_errors=True)
    os.makedirs(res)
    run = docker_run(a.image, tests, res, IMAGE_VERIFY, "smoke")
    got = read_image_run(res, doc)
    got.update({"kind": "smoke", "docker": run, "stage_notes": notes, "image": a.image, "image_id": docker_image_id(a.image), "utc": utc_now()})
    save(os.path.join(out, "image_run.json"), got)
    e = got["episodes"][0] if got["episodes"] else {}
    log("[image-smoke] exit %s reward %s; locked %s sim %s fingerprint %s trace %s (%.0f s)" % (
        got["exit_code"], got["verdict_reward"], e.get("n_locked"), e.get("sim_time_s"), str(e.get("fingerprint"))[:12],
        str((e.get("trace") or {}).get("qpos_sha256"))[:12], run["wall_s"]))
    shutil.rmtree(os.path.join(out, "stage", "tests", "assets"), ignore_errors=True)
    return 0 if got["exit_code"] == "0" else 1


def cmd_image_view(a):
    """the top frame 0 of every candidate in the task image with the agent renderer (Filament): rule 3's `view`"""
    sel = load(a.selection)
    check_frozen(sel, need_robot=False)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    tests, notes = build_stage(os.path.join(out, "stage"))
    res = os.path.join(out, "res")
    os.makedirs(res, exist_ok=True)
    run = docker_run(a.image, tests, res, IMAGE_VIEW, "view", extra_env=("VIEW_P=%d" % a.P,))
    rows = {}
    vd = os.path.join(res, "view")
    for c in sel["candidates"]:
        p = os.path.join(vd, c["id"], "result.json")
        if os.path.isfile(p):
            rows[c["id"]] = load(p)
    got = {"kind": "view", "image": a.image, "image_id": docker_image_id(a.image), "docker": run, "stage_notes": notes, "n": len(rows),
           "utc": utc_now(), "complete": len(rows) == len(sel["candidates"]) and all(r.get("complete") for r in rows.values())}
    save(os.path.join(out, "image_view.json"), got)
    log("[image-view] %d / %d candidates, docker rc %s, %.0f s" % (len(rows), len(sel["candidates"]), run["docker_rc"], run["wall_s"]))
    return 0 if got["complete"] else 1


# ================================================================================================ ingest, keep, make
def clauses(rec, need_view=True, need_episode=True):
    """the clauses of rule 4 one environment's record fails ([]: passes); each (cause, text)"""
    if rec is None:
        return [("missing", "no run")]
    out = []
    if rec.get("error"):
        out.append(("error", "the run raised: %s" % rec["error"].strip().splitlines()[-1][:200]))
    if rec.get("digest_ok") is False:
        out.append(("digest", "the layout digest differs from the recorded one"))
    if need_episode:
        if not rec.get("reference_ran"):
            out.append(("reference", "the reference program's report did not reach finalize"))
        else:
            n = rec.get("n_locked") or 0
            if n < len(SLOTS) or rec.get("stage_score") != 1.0:
                out.append(("mates", "%d / 8 LOCKED, stage score %s (%s)" % (n, rec.get("stage_score"),
                                                                            {k: v for k, v in (rec.get("states") or {}).items() if v != "LOCKED"})))
            if rec.get("unplanned_contacts"):
                out.append(("unplanned", "%d unplanned contact(s): %s" % (rec["unplanned_contacts"], rec.get("unplanned"))))
            if not rec.get("hard_ok"):
                out.append(("hard", "hard violation: %s" % rec.get("violations")))
            if (rec.get("max_sustained_N") or 0.0) >= MAX_SUSTAINED_N:
                out.append(("sustained", "largest sustained part force %.1f N" % rec["max_sustained_N"]))
            cap = rec.get("capture") or {}
            for sid in SLOTS:
                c = cap.get(sid)
                if c is None:
                    if (rec.get("states") or {}).get(sid) == "LOCKED":
                        out.append(("capture:" + sid, "%s: no capture recorded" % sid))
                    continue
                if c["ratio_max"] > CAPTURE_SHARE:
                    which = max((c.get("ratio") or {}).items(), key=lambda kv: kv[1])[0] if c.get("ratio") else "?"
                    out.append(("capture:" + sid, "%s captured at %.3f of its window (%s)" % (sid, c["ratio_max"], which)))
    if need_view:
        v = rec.get("view")
        if not v or v.get("error"):
            out.append(("view", "no top frame measured: %s" % ((v or {}).get("error") or "missing")))
        else:
            for u, s in sorted(v["seen"].items()):
                if s < SEEN_MIN:
                    out.append(("seen:" + u, "%s %.1f %% seen in the top frame" % (u, 100 * s)))
    return out


def env_record(c, env):
    """the record of candidate c in env as rule 4 judges it (image: the verify episode with the image view's frame)"""
    r = (c.get("runs") or {}).get(env)
    if env == "image" and r is not None:
        v = (c.get("runs") or {}).get("image_view")
        r = dict(r, view=(v or {}).get("view"), view_error=(v or {}).get("error"), view_digest_ok=(v or {}).get("digest_ok"))
        if v is not None and v.get("digest_ok") is False:
            r["digest_ok"] = False
    return r


def judge(c):
    """(decided, passed, {env: [(cause, text)]})"""
    failed, decided = {}, True
    for env in ENVS:
        r = env_record(c, env)
        if r is None or (env == "image" and (c.get("runs") or {}).get("image_view") is None):
            decided = False
            failed[env] = [("missing", "not run yet")]
            continue
        failed[env] = clauses(r)
    passed = decided and not any(failed[e] for e in ENVS)
    return decided, passed, failed


def compact_run(rec, env):
    """what selection.json keeps of a run (the full record stays with the run's files): the outcome, every mate's
    capture ratio (the largest of its captures, per axis) and time, the hashes"""
    keep = ("env", "seed", "stratum", "layout_digest", "digest_ok", "error", "reference_ran", "states", "n_locked", "stage_score", "hard_ok",
            "max_sustained_N", "max_robot_link_N", "unplanned_contacts", "unplanned", "violations", "scrapped",
            "sim_time_s", "ticks", "fingerprint", "tick_sha256", "trace", "view", "wall_s", "max_rss_gb", "env_info", "robot_tree",
            "episode_wall_s", "end_reason", "renderer", "done_reason")
    out = {k: rec[k] for k in keep if k in rec}
    out["capture"] = {sid: {"ratio_max": e["ratio_max"], "ratio": e.get("ratio"), "t": e.get("t"), "n": e.get("n")}
                      for sid, e in (rec.get("capture") or {}).items()}
    if isinstance(out.get("trace"), dict):
        out["trace"] = {k: out["trace"][k] for k in ("frames", "qpos_sha256") if k in out["trace"]}
    if isinstance(out.get("view"), dict):
        out["view"] = {k: out["view"][k] for k in ("seen", "least", "renderer") if k in out["view"]}
    if out.get("error"):
        out["error"] = out["error"][-1500:]
    return out


def _log_run(sel, entry):
    """selection.json's list of ingested runs: one entry per (env, kind, dir), the latest ingest replacing an earlier one"""
    key = (entry.get("env"), entry.get("kind"), entry.get("dir"))
    runs = [r for r in sel.get("runs", []) if (r.get("env"), r.get("kind"), r.get("dir")) != key]
    runs.append(entry)
    sel["runs"] = runs


def cmd_ingest(a):
    sel = load(a.selection)
    check_frozen(sel, need_robot=False)
    by_id = {c["id"]: c for c in sel["candidates"]}
    n = 0
    for runs in a.runs:
        runs = os.path.abspath(runs)
        if a.env in ("host", "pod"):
            for cid, c in by_id.items():
                p = os.path.join(runs, cid, "result.json")
                if not os.path.isfile(p):
                    continue
                rec = load(p)
                if not rec.get("complete") or rec.get("view_only"):
                    continue
                if rec.get("robot_tree") != sel["frozen_trees"].get(ROBOT_TREE):
                    raise SystemExit("%s: the robot models of the %s run differ from the frozen ones" % (cid, a.env))
                c.setdefault("runs", {})[a.env] = dict(compact_run(rec, a.env), source=run_path(p, a.root))
                n += 1
        elif a.env == "image":
            ir = os.path.join(runs, "image_run.json")
            iv = os.path.join(runs, "image_view.json")
            if os.path.isfile(ir):
                got = load(ir)
                for e in got["episodes"]:
                    cid = [k for k, c in by_id.items() if c["seed"] == e.get("seed")][0]
                    if e.get("missing"):
                        continue
                    r = dict(compact_run(e, "image"), env="image", source=run_path(ir, a.root), round=got["round"], env_info=got.get("env_info"),
                             verdict_reward=got["verdict_reward"], exit_code=got["exit_code"])
                    if e.get("trace_file"):
                        r["trace_file"] = run_path(e["trace_file"], a.root)
                    by_id[cid].setdefault("runs", {})["image"] = r
                    n += 1
                _log_run(sel, {"env": "image", "kind": "candidates", "round": got["round"], "dir": run_path(runs, a.root), "image": got["image"],
                               "image_id": got["image_id"], "exit_code": got["exit_code"], "reward": got["verdict_reward"],
                               "wall_s": got["docker"]["wall_s"], "utc": got["utc"], "task_errors": got["task_errors"],
                               "stage_notes": got.get("stage_notes")})
            if os.path.isfile(iv):
                got = load(iv)
                for cid, c in by_id.items():
                    p = os.path.join(runs, "res", "view", cid, "result.json")
                    if os.path.isfile(p):
                        rec = load(p)
                        if rec.get("complete"):
                            c.setdefault("runs", {})["image_view"] = dict({k: rec.get(k) for k in ("layout_digest", "digest_ok", "error", "env_info",
                                                                                                    "renderer", "wall_s", "robot_tree")},
                                                                          view=compact_run(rec, "image")["view"])
                            c["runs"]["image_view"]["source"] = run_path(p, a.root)
                            if rec.get("robot_tree") != sel["frozen_trees"].get(ROBOT_TREE):
                                c["runs"]["image_view"]["robot_tree_differs"] = True
                            n += 1
                _log_run(sel, {"env": "image", "kind": "view", "dir": run_path(runs, a.root), "image": got["image"], "image_id": got["image_id"],
                               "wall_s": got["docker"]["wall_s"], "utc": got["utc"], "n": got["n"]})
        if a.env in ("host", "pod"):
            _log_run(sel, {"env": a.env, "kind": "candidates", "dir": run_path(runs, a.root), "utc": utc_now()})
    for c in sel["candidates"]:
        decided, passed, failed = judge(c)
        c["failed"] = {e: [t for _k, t in v] for e, v in failed.items()}
        c["causes"] = sorted(set(k for v in failed.values() for k, _t in v if k != "missing"))
        c["status"] = "ran" if decided else ("running" if c.get("runs") else "drawn")
        c["pass"] = passed if decided else None
    save(a.selection, sel)
    print("ingested %d %s record(s) -> %s" % (n, a.env, a.selection))
    for c in sel["candidates"]:
        print("  %s %-12s %-8s pass %-5s %s" % (c["id"], c["stratum"], c["status"], c["pass"], "; ".join(c["causes"])))
    return 0


def classify(sel):
    """(kept {stratum: cand}, dropped, passed_not_needed, open strata)"""
    kept, dropped, spare, open_ = {}, [], [], []
    for st in STRATA:
        found = None
        for c in sorted((c for c in sel["candidates"] if c["stratum"] == st), key=lambda c: c["draw"]):
            if c.get("pass") is None:
                if found is None:            # an earlier candidate is not decided: the stratum's episode is not known yet
                    open_.append(st)
                    break
                continue
            if c["pass"]:
                if found is None:
                    found = c
                else:
                    spare.append(c)
            else:
                dropped.append(c)
        if found is not None:
            kept[st] = found
        elif st not in open_:
            open_.append(st)
    return kept, dropped, spare, open_


def cmd_make(a):
    sel = load(a.selection)
    check_frozen(sel, need_robot=False)
    kept, dropped, spare, open_ = classify(sel)
    undecided = [c["id"] for c in sel["candidates"] if c.get("pass") is None]
    if undecided:
        raise SystemExit("candidates not decided in all three environments: %s" % undecided)
    for c in sel["candidates"]:
        c["selection"] = "kept" if c in kept.values() else ("dropped" if c in dropped else "passed, not needed")
    if open_:
        sel["set"] = {"made_utc": utc_now(), "written": False, "strata_without_a_pass": open_,
                      "fallback": "the pre-written fallback: oracle on the nominal layout (oracle_nominal_layout true), the agent still on a fixed set; "
                                  "grader_config.json is not changed by this tool (a recommendation for the maintainers)"}
        save(a.selection, sel)
        print("NO SET: strata without a passing candidate: %s (the pre-written fallback recommended; nothing written to %s)" % (open_, a.out))
        return 2
    doc = set_of([kept[st] for st in STRATA])
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    sel["set"] = {"made_utc": utc_now(), "written": True, "kept": [kept[st]["id"] for st in STRATA], "dropped": [c["id"] for c in dropped],
                  "passed_not_needed": [c["id"] for c in spare], "episodes_sha256": sha256_file(a.out)}
    save(a.selection, sel)
    print("set: kept %s; dropped %s; passed, not needed %s -> %s" % (sel["set"]["kept"], sel["set"]["dropped"], sel["set"]["passed_not_needed"], a.out))
    return 0


# ================================================================================================ the hollowing check
def worst_capture(c, sid):
    vals = []
    for env in ENVS:
        r = (c.get("runs") or {}).get(env) or {}
        e = (r.get("capture") or {}).get(sid)
        if e is not None:
            vals.append(float(e["ratio_max"]))
    return max(vals) if vals else None


def class_ratio(c, cls):
    vals = [worst_capture(c, sid) for sid in SLOTS if MATE_CLASS[sid] == cls]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def population(path):
    """{stratum: {slot: [capture ratio]}} of an earlier campaign summary (its summary_acceptance.json)"""
    if not path:
        return None
    d = load(path)
    out = {}
    for r in d.get("runs", []):
        for sid, ents in (r.get("per_slot") or {}).items():
            for e in ents:
                if e.get("capture_ratio_max") is not None:
                    out.setdefault(r["stratum"], {}).setdefault(sid, []).append(float(e["capture_ratio_max"]))
    return out


def hollowing(sel, pop=None):
    kept, dropped, spare, open_ = classify(sel)
    ran = [c for c in sel["candidates"] if c.get("pass") is not None]
    flags, notes = [], []
    share = len(dropped) / float(len(ran)) if ran else 0.0
    if share > DROP_SHARE:
        flags.append("(a) %d of %d candidates dropped (%.0f %% > 1/3)" % (len(dropped), len(ran), 100 * share))
    causes = {}
    for c in dropped:
        for k in c.get("causes") or []:
            causes.setdefault(k, []).append(c["id"])
    for k, ids in sorted(causes.items()):
        if len(ids) >= 2:
            flags.append("(b) %d drops share a cause: %s (%s)" % (len(ids), k, ", ".join(ids)))
    cap_causes = set(k.split(":", 1)[1] for k in causes if k.startswith("capture:"))
    cap_classes = set(MATE_CLASS[s] for s in cap_causes)
    classes = ("battery", "deck", "compute", "lidar", "wheel")
    margin = {}
    K = list(kept.values())
    for cls in classes:
        km = _mean([class_ratio(c, cls) for c in K])
        dm = _mean([class_ratio(c, cls) for c in dropped])
        am = _mean([class_ratio(c, cls) for c in ran])
        row = {"kept_mean": km, "dropped_mean": dm, "all_mean": am, "kept": {c["id"]: class_ratio(c, cls) for c in K},
               "dropped": {c["id"]: class_ratio(c, cls) for c in dropped}, "rejected_through_capture": cls in cap_classes}
        if km is not None and dm is not None and dm - km > MARGIN_GAP:
            if cls in cap_classes:
                notes.append("(c) %s: kept mean %.3f vs dropped %.3f: by construction (candidates dropped through this class's capture clause)"
                             % (cls, km, dm))
            else:
                flags.append("(c) %s: kept mean capture ratio %.3f is %.3f below the dropped mean %.3f (no drop through this class)"
                             % (cls, km, dm - km, dm))
        if pop:
            pl = {}
            for c in K:
                for sid in SLOTS:
                    if MATE_CLASS[sid] != cls:
                        continue
                    xs = sorted((pop.get(c["stratum"]) or {}).get(sid) or [])
                    v = worst_capture(c, sid)
                    if xs and v is not None:
                        pl.setdefault(c["id"], {})[sid] = round(sum(1 for x in xs if x <= v) / float(len(xs)), 3)
            row["kept_percentile_in_population"] = pl
        margin[cls] = row
    keys = sorted(set(k for c in ran for k in (c.get("layout") or {}) if isinstance((c.get("layout") or {}).get(k), (int, float))))
    draws = {}
    for k in keys:
        def rng_(cs):
            xs = [c["layout"][k] for c in cs if isinstance((c.get("layout") or {}).get(k), (int, float))]
            return [round(min(xs), 4), round(max(xs), 4), round(sum(xs) / len(xs), 4)] if xs else None
        draws[k] = {"kept": rng_(K), "dropped": rng_(dropped), "all": rng_(ran)}
    by_stratum = {st: {"ran": sum(1 for c in ran if c["stratum"] == st), "dropped": sum(1 for c in dropped if c["stratum"] == st),
                       "kept": kept[st]["id"] if st in kept else None} for st in STRATA}
    return {"flags": flags, "notes": notes, "drop_share": share, "causes": causes, "by_stratum": by_stratum, "capture_margin": margin,
            "layout_draws": draws, "open_strata": open_, "kept": [c["id"] for c in K], "dropped": [c["id"] for c in dropped],
            "passed_not_needed": [c["id"] for c in spare]}


def cmd_report(a):
    sel = load(a.selection)
    pop = population(a.population)
    h = hollowing(sel, pop)
    print("candidates %d; kept %s; dropped %s; passed, not needed %s; open strata %s" % (len(sel["candidates"]), h["kept"], h["dropped"],
                                                                                       h["passed_not_needed"], h["open_strata"]))
    for st, v in h["by_stratum"].items():
        print("  %-12s ran %d dropped %d kept %s" % (st, v["ran"], v["dropped"], v["kept"]))
    for cls, v in h["capture_margin"].items():
        print("  capture %-8s kept %s dropped %s all %s%s" % (cls, _f3(v["kept_mean"]), _f3(v["dropped_mean"]), _f3(v["all_mean"]),
                                                              " (drops through this class)" if v["rejected_through_capture"] else ""))
    for n in h["notes"]:
        print("  note: " + n)
    print("hollowing check: %s" % ("FLAGGED: " + "; ".join(h["flags"]) if h["flags"] else "no flag"))
    sel["hollowing"] = {"utc": utc_now(), "flags": h["flags"], "notes": h["notes"], "drop_share": h["drop_share"],
                        "population": os.path.basename(a.population) if a.population else None}
    save(a.selection, sel)
    if a.json:
        save(a.json, h)
    return 0


def _f3(x):
    return "-" if x is None else "%.3f" % x


# ================================================================================================ acceptance
def cmd_accept(a):
    """`repeat` verify runs in the task image of the frozen set (tests/hidden/episodes.json as the task holds it)"""
    sel = load(a.selection)
    check_frozen(sel, need_robot=False)
    doc = load(a.episodes)
    if (sel.get("set") or {}).get("episodes_sha256") != sha256_file(a.episodes):
        raise SystemExit("%s is not the set this selection made" % a.episodes)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    rows = []
    for k in range(a.repeat):
        d = os.path.join(out, "run%d" % (k + 1))
        tests, notes = build_stage(os.path.join(d, "stage"))          # the task's own tests/ (episodes.json included)
        if load(os.path.join(tests, "hidden", "episodes.json")) != doc:
            raise SystemExit("the stage's episodes.json is not %s" % a.episodes)
        res = os.path.join(d, "res")
        shutil.rmtree(res, ignore_errors=True)
        os.makedirs(res)
        rows.append((d, tests, res, notes))
    if a.parallel:
        import threading
        results, threads = [None] * len(rows), []

        def go(i, tests, res):
            results[i] = docker_run(a.image, tests, res, IMAGE_VERIFY, "acc%d" % (i + 1))
        for i, (d, tests, res, _n) in enumerate(rows):
            th = threading.Thread(target=go, args=(i, tests, res))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
    else:
        results = [docker_run(a.image, tests, res, IMAGE_VERIFY, "acc%d" % (i + 1)) for i, (d, tests, res, _n) in enumerate(rows)]
    runs = []
    for (d, tests, res, notes), run in zip(rows, results):
        got = read_image_run(res, doc)
        got.update({"docker": run, "stage_notes": notes, "image": a.image, "image_id": docker_image_id(a.image), "utc": utc_now()})
        save(os.path.join(d, "image_run.json"), got)
        runs.append(got)
        shutil.rmtree(os.path.join(d, "stage", "tests", "assets"), ignore_errors=True)
    per = [[(e.get("stage_score"), e.get("hard_ok")) for e in r["episodes"]] for r in runs]
    same_scores = all(p == per[0] for p in per)
    same_fp = all([e.get("fingerprint") for e in r["episodes"]] == [e.get("fingerprint") for e in runs[0]["episodes"]] for r in runs)
    same_trace = all([(e.get("trace") or {}).get("qpos_sha256") for e in r["episodes"]] ==
                     [(e.get("trace") or {}).get("qpos_sha256") for e in runs[0]["episodes"]] for r in runs)
    ok = all(r["verdict_reward"] == 1.0 for r in runs) and same_scores and all(len(r["episodes"]) == 4 for r in runs)
    summary = {"format": "racecar-asm-set-acceptance/1", "episodes_sha256": sha256_file(a.episodes), "repeat": a.repeat, "parallel": bool(a.parallel),
               "rewards": [r["verdict_reward"] for r in runs], "per_episode": per, "same_per_episode_scores": same_scores,
               "same_fingerprints": same_fp, "same_traces": same_trace, "passed": ok,
               "runs": [{"dir": run_path(rows[i][0], a.root), "exit_code": r["exit_code"], "reward": r["verdict_reward"], "metrics": r["metrics"],
                         "task_errors": r["task_errors"], "wall_s": r["docker"]["wall_s"], "env_info": r.get("env_info"),
                         "episodes": [{k: e.get(k) for k in ("seed", "stratum", "digest_ok", "stage_score", "n_locked", "hard_ok", "unplanned_contacts",
                                                             "max_sustained_N", "sim_time_s", "ticks", "fingerprint", "trace", "episode_wall_s",
                                                             "capture", "reference_return")} for e in r["episodes"]]}
                        for i, r in enumerate(runs)], "image": a.image, "image_id": docker_image_id(a.image), "utc": utc_now()}
    save(os.path.join(out, "acceptance.json"), summary)
    sel["acceptance"] = {k: summary[k] for k in ("utc", "rewards", "per_episode", "same_per_episode_scores", "same_fingerprints", "same_traces",
                                                 "passed", "image", "image_id")}
    sel["acceptance"]["dir"] = run_path(out, a.root)
    save(a.selection, sel)
    print("acceptance: rewards %s; per-episode %s; same scores %s, same fingerprints %s, same traces %s -> %s" % (
        summary["rewards"], per, same_scores, same_fp, same_trace, "PASSED" if ok else "FAILED"))
    return 0 if ok else 1


# ================================================================================================ the determinism record
def _trace_compare(fa, fb):
    """(first differing frame or None, frames a / b, largest |dq|) of two trace.npz files"""
    import numpy as np
    if not (fa and fb and os.path.isfile(fa) and os.path.isfile(fb)):
        return None
    A, B = np.load(fa), np.load(fb)
    qa, qb = A["qpos"], B["qpos"]
    n = min(len(qa), len(qb))
    diff = np.any(qa[:n] != qb[:n], axis=1) if n else np.zeros(0, bool)
    first = int(np.argmax(diff)) if diff.any() else (None if len(qa) == len(qb) else n)
    dq = float(np.max(np.abs(qa[:n].astype(np.float64) - qb[:n].astype(np.float64)))) if n else 0.0
    t_first = None if first is None or first >= len(A["t"]) else float(A["t"][first])
    return {"first_diff_frame": first, "first_diff_t_s": t_first, "frames": [int(len(qa)), int(len(qb))], "max_abs_dq": dq}


def cmd_determinism(a):
    sel = load(a.selection)
    L = []
    w = L.append
    w("# Determinism of the reference program on the hidden-set candidates (generated by tests/review/select_set.py determinism)")
    w("")
    w("Selection drawn %s; %d candidates; rule: tests/review/select_set.py (module docstring, copied into selection.json)." % (
        sel["drawn_utc"], len(sel["candidates"])))
    if sel.get("set"):
        w("Set: %s" % json.dumps({k: v for k, v in sel["set"].items() if k != "episodes_sha256"}))
    w("")
    w("## Environments")
    w("")
    w("| env | cpu | python | mujoco | numpy | scipy | top-frame renderer |")
    w("|---|---|---|---|---|---|---|")
    for env in ENVS:
        infos = [((c.get("runs") or {}).get(env) or {}).get("env_info") or {} for c in sel["candidates"]]
        infos = [i for i in infos if i]
        rend = sorted(set(str(((c.get("runs") or {}).get("image_view" if env == "image" else env) or {}).get("view", {}).get("renderer"))
                          for c in sel["candidates"]))
        if infos:
            i = infos[0]
            w("| %s | %s | %s | %s | %s | %s | %s |" % (env, i.get("cpu"), i.get("python"), i.get("mujoco"), i.get("numpy"), i.get("scipy"), ", ".join(rend)))
    w("")
    w("## Outcome per candidate and environment")
    w("")
    w("`locked` = mates LOCKED of 8; `unpl` = unplanned contacts; `sust` = largest sustained part force (N); `cap` = the largest capture ratio "
      "(pose error / window) over the eight mates; `seen` = the least-seen loose unit in the top frame 0; `sim` = sim time (s).")
    w("")
    w("| cand | stratum | env | digest | locked | score | unpl | sust N | cap max (mate) | seen min | sim s | ticks | clauses failed |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in sel["candidates"]:
        for env in ENVS:
            r = env_record(c, env) or {}
            cap = r.get("capture") or {}
            cm = max(cap.items(), key=lambda kv: kv[1]["ratio_max"]) if cap else None
            v = r.get("view") or {}
            least = v.get("least")
            w("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                c["id"], c["stratum"], env, {True: "ok", False: "DIFF"}.get(r.get("digest_ok"), "-"), r.get("n_locked", "-"),
                r.get("stage_score", "-"), r.get("unplanned_contacts", "-"), _f1(r.get("max_sustained_N")),
                "-" if not cm else "%.3f (%s)" % (cm[1]["ratio_max"], cm[0]), "-" if not least else "%s %.1f %%" % (least[0], 100 * least[1]),
                _f2(r.get("sim_time_s")), r.get("ticks", "-"), "; ".join((c.get("failed") or {}).get(env) or []) or "-"))
    w("")
    w("## Capture ratio and capture time per mate (host / pod / image)")
    w("")
    w("| cand | mate | ratio host | ratio pod | ratio image | t host s | t pod s | t image s | state host / pod / image |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c in sel["candidates"]:
        for sid in SLOTS:
            vals, ts, sts = [], [], []
            for env in ENVS:
                r = (c.get("runs") or {}).get(env) or {}
                e = (r.get("capture") or {}).get(sid)
                vals.append("-" if e is None else "%.4f" % e["ratio_max"])
                ts.append("-" if e is None else "%.2f" % e["t"])
                sts.append(str((r.get("states") or {}).get(sid, "-")))
            w("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (c["id"], sid, vals[0], vals[1], vals[2], ts[0], ts[1], ts[2], " / ".join(sts)))
    w("")
    w("## Bitwise comparison")
    w("")
    w("`fingerprint` = sha256 over finalize's physics fields (states, sim time, ticks, mates, coupling events, end hub angles, watch, "
      "scrapped, violations, stage score, the reference report without wall-clock values); `trace` = sha256 of verify's episode-trace qpos "
      "(float32 every 4 ticks); `tick` = sha256 of every control tick's qpos + qvel (float64; host and pod only). First difference from the "
      "trace files where both are present.")
    w("")
    w("| cand | host = image fingerprint | host = image trace | host = pod fingerprint | host = pod trace | host / pod first trace difference (frame, t s, max abs dq) | "
      "sim s host / pod / image | ticks host / pod / image |")
    w("|---|---|---|---|---|---|---|---|")
    stats = {"host_image_fp": 0, "host_image_trace": 0, "host_pod_fp": 0, "host_pod_trace": 0, "n": 0}
    notes_hi = []
    for c in sel["candidates"]:
        R = {e: (c.get("runs") or {}).get(e) or {} for e in ENVS}
        if not all(R[e] for e in ENVS):
            continue
        stats["n"] += 1
        hi_fp = R["host"].get("fingerprint") == R["image"].get("fingerprint")
        hi_tr = (R["host"].get("trace") or {}).get("qpos_sha256") == (R["image"].get("trace") or {}).get("qpos_sha256")
        hp_fp = R["host"].get("fingerprint") == R["pod"].get("fingerprint")
        hp_tr = (R["host"].get("trace") or {}).get("qpos_sha256") == (R["pod"].get("trace") or {}).get("qpos_sha256")
        for k, v in (("host_image_fp", hi_fp), ("host_image_trace", hi_tr), ("host_pod_fp", hp_fp), ("host_pod_trace", hp_tr)):
            stats[k] += int(v)
        cmpr = None
        fa = os.path.join(os.path.dirname(from_root(R["host"].get("source", ""), a.root)), "trace.npz")
        if not hp_tr:
            fb = _remap(os.path.join(os.path.dirname(from_root(R["pod"].get("source", ""), a.root)), "trace.npz"), a.remap)
            cmpr = _trace_compare(fa, fb)
        if not hi_tr:
            ci = _trace_compare(fa, from_root(R["image"].get("trace_file"), a.root))
            if ci is not None:
                notes_hi.append("%s host / image first trace difference: frame %s, t %s s, max |dq| %.3g" % (
                    c["id"], ci["first_diff_frame"], _f2(ci["first_diff_t_s"]), ci["max_abs_dq"]))
        w("| %s | %s | %s | %s | %s | %s | %s / %s / %s | %s / %s / %s |" % (
            c["id"], _yn(hi_fp), _yn(hi_tr), _yn(hp_fp), _yn(hp_tr),
            "-" if cmpr is None else "%s, %s, %.3g" % (cmpr["first_diff_frame"], _f2(cmpr["first_diff_t_s"]), cmpr["max_abs_dq"]),
            _f2(R["host"].get("sim_time_s")), _f2(R["pod"].get("sim_time_s")), _f2(R["image"].get("sim_time_s")),
            R["host"].get("ticks"), R["pod"].get("ticks"), R["image"].get("ticks")))
    w("")
    w("Equal: host = image fingerprint %d / %d, trace %d / %d; host = pod fingerprint %d / %d, trace %d / %d." % (
        stats["host_image_fp"], stats["n"], stats["host_image_trace"], stats["n"], stats["host_pod_fp"], stats["n"], stats["host_pod_trace"], stats["n"]))
    for n_ in notes_hi:
        w("- " + n_)
    if sel.get("acceptance"):
        acc = sel["acceptance"]
        w("")
        w("## Acceptance: the frozen set in the task image, verify run twice")
        w("")
        w("Rewards %s; per-episode (stage score, hard_ok) %s; same per-episode scores %s; same fingerprints %s; same traces %s; %s." % (
            acc["rewards"], acc["per_episode"], acc["same_per_episode_scores"], acc["same_fingerprints"], acc["same_traces"],
            "PASSED" if acc["passed"] else "FAILED"))
    if sel.get("hollowing"):
        w("")
        w("## Hollowing check")
        w("")
        hh = sel["hollowing"]
        w("Flags: %s." % ("; ".join(hh["flags"]) if hh["flags"] else "none"))
        for n in hh.get("notes") or []:
            w("- note: " + n)
    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("determinism record -> %s (%s)" % (a.out, json.dumps(stats)))
    return 0


def _remap(p, pairs):
    for pr in pairs or []:
        src, dst = pr.split("=", 1)
        if p.startswith(src):
            return dst + p[len(src):]
    return p


def _yn(b):
    return "yes" if b else "NO"


def _f1(x):
    return "-" if x is None else "%.1f" % x


def _f2(x):
    return "-" if x is None else "%.2f" % x


# ================================================================================================ selftest
def selftest(a):
    results = []

    def check(name, fn):
        try:
            fn()
            results.append((name, True, ""))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, "%s: %s" % (type(e).__name__, e)))

    def t_constants():
        tree = ast.parse(open(os.path.join(TESTS, "arm", "assembly_car", "layout.py")).read())
        ly = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "STRATA" for t in n.targets)][0]
        vt = ast.parse(open(os.path.join(TESTS, "verify.py")).read())
        vals = {t.id: ast.literal_eval(n.value) for n in vt.body if isinstance(n, ast.Assign) for t in n.targets
                if isinstance(t, ast.Name) and t.id in ("STRATA", "SET_FORMAT")}
        assert tuple(ly) == STRATA == tuple(vals["STRATA"]), (ly, vals)
        assert vals["SET_FORMAT"] == SET_FORMAT

    def t_seeds():
        seq = [b"\x00" * 8, (5).to_bytes(8, "little"), ((1 << 63) + 7 + (1 << 40)).to_bytes(8, "little"),
               ((1 << 40) + 7).to_bytes(8, "little"), ((1 << 41)).to_bytes(8, "little")]
        it = iter(seq)
        s, again = draw_seed(set(), lambda n: next(it))
        assert s == (1 << 40) + 7 and again == 2, (s, again)      # 0 and 5 are below 2**32; the third is taken mod 2**63
        s2, again2 = draw_seed({s}, lambda n: next(it))
        assert s2 == (1 << 41) and again2 == 1, (s2, again2)      # the repeat (2**40 + 7) is drawn again
        cands, _r = make_candidates()
        assert len(cands) == 12 and len({c["seed"] for c in cands}) == 12
        assert all(SEED_MIN <= c["seed"] < SEED_MOD for c in cands)
        assert [c["stratum"] for c in cands] == list(STRATA) * 3 and [c["round"] for c in cands] == [0] * 4 + [1] * 4 + [2] * 4
        assert [c["id"] for c in cands] == ["c%02d" % i for i in range(12)]
        it2 = iter([((1 << 40) + 7).to_bytes(8, "little"), ((1 << 42)).to_bytes(8, "little")] + [os.urandom(8) for _ in range(40)])
        c2, again3 = make_candidates(n=2, urandom=lambda n: next(it2), exclude=[(1 << 40) + 7])
        assert c2[0]["seed"] == 1 << 42 and again3 >= 1          # a superseded selection's seed is never drawn again

    def good(**kw):
        r = {"digest_ok": True, "reference_ran": True, "states": {s: "LOCKED" for s in SLOTS}, "n_locked": 8, "stage_score": 1.0, "hard_ok": True,
             "max_sustained_N": 13.0, "unplanned_contacts": 0, "capture": {s: {"ratio_max": 0.3, "ratio": {"lateral": 0.3, "tilt": 0.1}} for s in SLOTS},
             "view": {"seen": {u: 0.5 for u in ("BAT", "DCK", "CMP", "LDR", "WHL1", "WHL2", "WHL3", "WHL4")}}}
        r.update(kw)
        return r

    def t_clauses():
        assert clauses(good()) == []
        bad = {"digest": good(digest_ok=False), "reference": good(reference_ran=False), "mates": good(n_locked=7, stage_score=10.0 / 11.5),
               "unplanned": good(unplanned_contacts=1), "hard": good(hard_ok=False), "sustained": good(max_sustained_N=150.0),
               "capture:battery": good(capture=dict(good()["capture"], battery={"ratio_max": 0.51, "ratio": {"lateral": 0.2, "tilt": 0.51}})),
               "seen:LDR": good(view={"seen": dict(good()["view"]["seen"], LDR=0.299)}), "view": good(view=None),
               "error": good(error="Traceback\nValueError: x")}
        for cause, rec in bad.items():
            got = [k for k, _t in clauses(rec)]
            assert cause in got, (cause, got)
        assert clauses(good(capture=dict(good()["capture"], battery={"ratio_max": 0.5, "ratio": {"tilt": 0.5}}))) == []    # at most 50 %
        assert clauses(good(view={"seen": dict(good()["view"]["seen"], LDR=0.30)})) == []                                # at least 30 %
        missing = good(capture={s: v for s, v in good()["capture"].items() if s != "deck"})
        assert [k for k, _t in clauses(missing)] == ["capture:deck"]

    def t_judge_and_make():
        tmp = tempfile.mkdtemp(prefix="p213-st-")
        try:
            cands, _r = make_candidates()
            for c in cands:
                c["layout_digest"] = hashlib.sha256(str(c["seed"]).encode()).hexdigest()
                c["runs"] = {"host": good(), "pod": good(), "image": good(), "image_view": {"view": good()["view"], "digest_ok": True}}
                c["layout"] = {"stand_dx_mm": float(c["draw"])}
            # c01 (yaw+lead180, round 0) fails on the pod; c05 (same stratum, round 1) passes -> kept; c09 passes, not needed
            cands[1]["runs"]["pod"] = good(capture=dict(good()["capture"], battery={"ratio_max": 0.8, "ratio": {"tilt": 0.8}}))
            cands[2]["runs"]["image_view"] = {"view": {"seen": dict(good()["view"]["seen"], LDR=0.1)}, "digest_ok": True}
            del cands[3]["runs"]["pod"]
            sel = {"candidates": cands}
            for c in cands:
                decided, passed, failed = judge(c)
                c["failed"] = {e: [t for _k, t in v] for e, v in failed.items()}
                c["causes"] = sorted(set(k for v in failed.values() for k, _t in v if k != "missing"))
                c["pass"] = passed if decided else None
            assert cands[1]["pass"] is False and cands[1]["causes"] == ["capture:battery"]
            assert cands[2]["pass"] is False and cands[2]["causes"] == ["seen:LDR"]
            assert cands[3]["pass"] is None
            kept, dropped, spare, open_ = classify(sel)
            assert open_ == ["yaw-lead180"] and kept["yaw+lead0"]["id"] == "c00" and kept["yaw+lead180"]["id"] == "c05", (open_, kept)
            assert {c["id"] for c in dropped} == {"c01", "c02"} and "c09" in {c["id"] for c in spare}
            cands[3]["runs"]["pod"] = good()
            for c in cands:
                d_, p_, f_ = judge(c)
                c["pass"] = p_ if d_ else None
                c["causes"] = sorted(set(k for v in f_.values() for k, _t in v if k != "missing"))
            kept, dropped, spare, open_ = classify(sel)
            assert not open_ and [kept[s]["id"] for s in STRATA] == ["c00", "c05", "c06", "c03"]
            doc = set_of([kept[s] for s in STRATA])
            os.makedirs(os.path.join(tmp, "hidden"))
            with open(os.path.join(tmp, "hidden", "episodes.json"), "w") as f:
                json.dump(doc, f)
            G = _verify_fns(["load_hidden_set"])
            eps = G["load_hidden_set"]({"episodes": 4}, os.path.join(tmp, "hidden"))
            assert [e["stratum"] for e in eps] == list(STRATA) and len({e["seed"] for e in eps}) == 4
            h = hollowing(sel)
            assert h["kept"] == ["c00", "c05", "c06", "c03"] and h["dropped"] == ["c01", "c02"] and not h["flags"], h["flags"]
            cands[7]["runs"]["host"] = good(capture=dict(good()["capture"], battery={"ratio_max": 0.9, "ratio": {"tilt": 0.9}}))
            for c in cands:
                d_, p_, f_ = judge(c)
                c["pass"] = p_ if d_ else None
                c["causes"] = sorted(set(k for v in f_.values() for k, _t in v if k != "missing"))
            h = hollowing(sel)
            assert any(f.startswith("(b) 2 drops share a cause: capture:battery") for f in h["flags"]), h["flags"]
            assert any("by construction" in n for n in h["notes"]), h["notes"]
            # (c): a class that dropped nobody through its capture clause but whose dropped candidates were harder there
            for c in (cands[1], cands[2], cands[7]):
                for env in ("host",):
                    c["runs"][env] = dict(c["runs"][env], capture=dict(c["runs"][env]["capture"], deck={"ratio_max": 0.45, "ratio": {"lateral": 0.45}}))
            h = hollowing(sel)
            assert any(f.startswith("(c) deck") for f in h["flags"]), h["flags"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def t_fingerprint():
        fin = {"states": {"a": "LOCKED"}, "sim_time_s": 1.25, "ticks": 125, "watch": {"max_part_force_sustained_N": 3.0},
               "reference": {"steps": [{"slot": "battery", "wall_s": 3.0, "capture": {"ratio_max": 0.2, "t": 1.0}}], "unplanned_contacts": []},
               "episode_wall_s": 99.0, "camera_frames": {"top": 1}}
        f2 = json.loads(json.dumps(fin))
        f2["reference"]["steps"][0]["wall_s"] = 7.0
        f2["episode_wall_s"] = 1.0
        f2["camera_frames"] = {"top": 2}
        assert fingerprint(fin) == fingerprint(f2)
        f2["sim_time_s"] = 1.2500000000000002
        assert fingerprint(fin) != fingerprint(f2)

    def t_visibility():
        import numpy as np
        from render.spec import CameraSpec
        R = np.diag([1.0, -1.0, -1.0])                       # looking down -z
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [0.0, 0.0, 2.0]
        cam = CameraSpec(name="t", width=200, height=100, fx=100.0, fy=100.0, cx=100.0, cy=50.0, world_T_cam=T.tolist())
        g = np.linspace(-0.3, 0.3, 11)
        P = np.array([[x, y, 0.0] for x in g for y in g])
        dep = np.full((100, 200), 2.0)
        assert seen_fraction(cam, P, dep) == 1.0
        assert seen_fraction(cam, P, dep - 0.05) == 0.0
        assert seen_fraction(cam, P, np.zeros_like(dep)) == 0.0
        assert seen_fraction(cam, P + [0.0, 0.0, 0.010], dep) == 1.0          # within 15 mm
        assert seen_fraction(cam, np.array([[5.0, 0.0, 0.0]]), dep) == 0.0       # outside the frame
        from PIL import Image
        arr = (np.arange(12, dtype=np.uint16).reshape(3, 4) * 1000)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        got = decode_depth(base64.b64encode(buf.getvalue()).decode())
        assert np.allclose(got, arr * 0.001), got

    def t_frozen():
        """amendment A1: outcome inputs frozen; publication metadata and the agent's documents recorded, never invalidating"""
        tmp = tempfile.mkdtemp(prefix="p213-fz-")

        def put(rel, text):
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
            with open(os.path.join(tmp, rel), "w") as f:
                f.write(text)

        def pack(members, mtime=0, preset=None):
            path = os.path.join(tmp, KIT_ARCHIVE)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with tarfile.open(path, "w:xz", **({} if preset is None else {"preset": preset})) as tf:
                ti = tarfile.TarInfo("gt")
                ti.type, ti.mtime = tarfile.DIRTYPE, mtime
                tf.addfile(ti)
                for name, data in members:
                    ti = tarfile.TarInfo(name)
                    ti.size, ti.mtime = len(data), mtime
                    tf.addfile(ti, io.BytesIO(data))

        def fails(sel, needle):
            try:
                check_frozen(sel, task=tmp, quiet=True)
            except SystemExit as e:
                assert needle in str(e), (needle, str(e))
                return
            raise AssertionError("a change of %s passed the freeze" % needle)
        try:
            for p in FROZEN:
                put(p, p)
            for p in FROZEN_TREES + (ROBOT_TREE,):
                put(p + "/sub/x.py", p)
            for p in RECORDED_FILES:
                put(p, p)
            for p in RECORDED_TREES:
                put(p + "/a.md", p)
            cfg = {"safety": {"robot_link_N": 30.0}, "budgets": [650.0, 3100.0],
                   "hidden_gt": {"url": "https://h/r/0000/gt.tar.xz", "sha256": "0" * 64, "note": "a placeholder revision", "files": 2,
                                 "tries": 4, "backoff_s": 5.0}}
            put(CONFIG_FILE, json.dumps(cfg))
            members = [("gt/kit.json", b"{}"), ("gt/meshes/a.stl", b"abc")]
            pack(members)
            files, trees, counts = frozen_hashes(tmp)
            assert set(files) == set(FROZEN) | {CONFIG_FILE + "#values", KIT_ARCHIVE + "#members"} and all(files.values()), files
            assert counts == {KIT_ARCHIVE + "#members": 2}, counts
            rec0 = recorded_hashes(tmp)
            sel = {"frozen": files, "frozen_trees": trees, "recorded": rec0, "drawn_utc": "t0"}
            assert check_frozen(sel, task=tmp, quiet=True)
            os.makedirs(os.path.join(tmp, "tests", "arm", "__pycache__"))
            put("tests/arm/__pycache__/x.pyc", "bytecode")
            assert check_frozen(sel, task=tmp, quiet=True)                       # __pycache__ is not an input
            # publication: the archive's location, hash and note change, the same members are re-packed (other times,
            # another compression), the file is laid out differently: the set stays valid, the record names all four
            cfg_pub = json.loads(json.dumps(cfg))
            cfg_pub["hidden_gt"].update(url="https://h/r/abcd/gt.tar.xz", sha256="1" * 64, note="published")
            put(CONFIG_FILE, json.dumps(cfg_pub, indent=2))
            pack(members, mtime=1234567, preset=1)
            assert check_frozen(sel, task=tmp, quiet=True)
            d = recorded_diff(rec0, recorded_hashes(tmp))
            want = {"publication/" + KIT_ARCHIVE} | {"publication/%s#hidden_gt.%s" % (CONFIG_FILE, k) for k in PUBLICATION_KEYS}
            assert set(d) == want, sorted(d)
            # the documents the agent reads change: valid, named
            put("instruction.md", "a corrected text")
            put(RECORDED_TREES[0] + "/a.md", "a corrected manual")
            assert check_frozen(sel, task=tmp, quiet=True)
            d = recorded_diff(rec0, recorded_hashes(tmp))
            assert {"documents/instruction.md", "documents/%s/" % RECORDED_TREES[0]} <= set(d), sorted(d)
            # a grader value (also hidden_gt's file count or retries) changes: invalid
            for path, val in ((("safety", "robot_link_N"), 31.0), (("hidden_gt", "files"), 3), (("hidden_gt", "tries"), 5)):
                c = json.loads(json.dumps(cfg_pub))
                c[path[0]][path[1]] = val
                put(CONFIG_FILE, json.dumps(c))
                fails(sel, CONFIG_FILE + "#values")
            put(CONFIG_FILE, json.dumps(cfg_pub))
            assert check_frozen(sel, task=tmp, quiet=True)
            # the kit's content changes (a member's bytes, a member more, a member renamed): invalid
            for mem in ([("gt/kit.json", b"{}"), ("gt/meshes/a.stl", b"abd")], members + [("gt/meshes/b.stl", b"")],
                        [("gt/kit.json", b"{}"), ("gt/meshes/A.stl", b"abc")]):
                pack(mem)
                fails(sel, KIT_ARCHIVE + "#members")
            os.remove(os.path.join(tmp, KIT_ARCHIVE))
            fails(sel, KIT_ARCHIVE + "#members")                                  # a missing kit is not the kit
            pack(members)
            # a frozen file or tree changes: invalid
            put("tests/verify.py", "changed")
            fails(sel, "tests/verify.py")
            put("tests/verify.py", "tests/verify.py")
            with open(os.path.join(tmp, "tests", "arm", "sub", "x.py"), "a") as f:
                f.write("#")
            fails(sel, "tests/arm/")
            put("tests/arm/sub/x.py", "tests/arm")
            assert check_frozen(sel, task=tmp, quiet=True)
            # a table as drawn before A1 (plain paths) is checked by the rule that wrote it: bytes
            legacy = {"frozen": {p: sha256_file(os.path.join(tmp, p)) for p in FROZEN + (CONFIG_FILE, KIT_ARCHIVE)},
                      "frozen_trees": dict(trees), "drawn_utc": "t1"}
            assert check_frozen(legacy, task=tmp, quiet=True)
            put(CONFIG_FILE, json.dumps(cfg_pub, indent=1))                       # the same values, other bytes
            fails(legacy, CONFIG_FILE)
            # ... and an amendment's table is the one in force over the draw's
            legacy["amendments"] = [{"id": "A1", "date": "d", "frozen": files, "frozen_trees": dict(trees), "recorded": rec0}]
            assert check_frozen(legacy, task=tmp, quiet=True)
            st = freeze_status(legacy, task=tmp)
            assert st["frozen_ok"] and st["table"].startswith("amendment A1") and "documents/instruction.md" in st["recorded_changed"], st
            # recorded_changes are the latest record
            legacy["recorded_changes"] = [{"recorded": recorded_hashes(tmp)}]
            assert freeze_status(legacy, task=tmp)["recorded_changed"] == []
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def t_amendment_record():
        """the record's amendment holds the table this tool computes on this task (skips without the record or the kit)"""
        path = os.path.join(TESTS, "hidden", "selection.json")
        if not os.path.isfile(path) or not os.path.isfile(os.path.join(TASK, KIT_ARCHIVE)):
            print("  (skipped: no selection record or no local kit archive)")
            return
        sel = load(path)
        am = sel.get("amendments") or []
        assert am and am[0]["id"] == "A1" and am[0]["date"] == "2026-11-15", [x.get("id") for x in am]
        assert am[0]["checked_at_writing"]["all_as_drawn"] is True
        files, trees, counts = frozen_hashes()
        assert files == am[-1]["frozen"], sorted(k for k in files if files[k] != am[-1]["frozen"].get(k))
        assert trees == am[-1]["frozen_trees"] and counts == am[-1]["frozen_counts"]
        # what A1 kept frozen as drawn is the draw's own entry
        for p in FROZEN:
            assert am[0]["frozen"][p] == sel["frozen"][p], p
        assert am[0]["frozen_trees"] == sel["frozen_trees"]

    def t_rule_in_doc():
        txt = rule_text()
        for needle in ("os.urandom", "2**63", "below 2**32", "STRATA[i % 4]", "50 %", "30 %", "150 N", "first passing candidate", "1/3",
                       "0.10 of the window", "oracle_nominal_layout true", "15 mm", "4000", "amendment A1",
                       "'tests/grader_config.json#values'", "'tests/assets/gt.tar.xz#members'", "recorded by hash",
                       "does not invalidate the set", "url, sha256, note"):
            assert needle in txt, needle

    def t_layout_draws():
        lay = {"stand": {"xy": [0.63, -0.01], "yaw_deg": 181.5}, "clamp": {"dx_m": 0.001, "dy_m": 0.0, "dyaw_deg": 0.1},
               "fixtures": {"shelf": {"xy": [0.31, -0.45], "yaw_deg": 183.0}, "tray": {"xy": [0.30, 0.47], "yaw_deg": -2.0}},
               "plays": {"DCK": {"dx_mm": 0.3, "dy_mm": 0.4, "dyaw_deg": 0.1}}, "parts": {"BAT": [[0.95, -0.4, 0.0], [1.0, 0.0, 0.0, 0.0]]},
               "hub_deg": {"hub_FL": 59.0, "hub_FR": 31.0}, "lead_end_deg": 180}
        L = {"stand": {"nominal_xy": [0.62, 0.0]}, "shelf": {"nominal_xy": [0.30, -0.45]}, "tray": {"nominal_xy": [0.30, 0.45]}}
        d = layout_draws(lay, L)
        assert abs(d["stand_dx_mm"] - 10.0) < 1e-6 and abs(d["stand_dyaw_deg"] - 1.5) < 1e-6 and abs(d["play_DCK_mm"] - 0.5) < 1e-6
        assert abs(d["hub_FL_off_hex_deg"] - 1.0) < 1e-6 and abs(d["hub_FR_off_hex_deg"] - 29.0) < 1e-6 and abs(d["BAT_r_m"] - math.hypot(0.95, 0.4)) < 1e-4

    for name, fn in (("constants = layout / verify", t_constants), ("seed draw and strata", t_seeds), ("rule 4 clauses", t_clauses),
                     ("judge, keep, make, hollowing", t_judge_and_make), ("fingerprint", t_fingerprint), ("visibility and depth decode", t_visibility),
                     ("freeze (amendment A1)", t_frozen), ("the record's amendment = this tool's table", t_amendment_record),
                     ("the rule in the docstring", t_rule_in_doc), ("layout draws", t_layout_draws)):
        check(name, fn)
    if a.physics:
        check("physics: two runs of a non-candidate episode are identical", lambda: _t_physics(a))
    for name, ok, why in results:
        print("%-55s %s %s" % (name, "PASS" if ok else "FAIL", why))
    n_ok = sum(1 for r in results if r[1])
    print("selftest: %d / %d passed (tool sha256 %s)" % (n_ok, len(results), sha256_file(os.path.abspath(__file__))[:16]))
    return 0 if n_ok == len(results) else 1


def _verify_fns(names):
    """functions of tests/verify.py by name in a stdlib namespace (verify imports numpy and MuJoCo at module level)"""
    import re
    tree = ast.parse(open(os.path.join(TESTS, "verify.py")).read())
    body = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("SET_FORMAT", "STRATA", "SEED_LIMIT")
                                                                       for t in n.targets)]
    body += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]

    def fail(reason):
        raise ValueError(reason)
    ns = {"json": json, "os": os, "re": re, "fail": fail}
    exec(compile(ast.Module(body=body, type_ignores=[]), "verify.py", "exec"), ns)
    return ns


def _t_physics(a):
    """a non-candidate episode (seed 1009, yaw+lead0, a development seed outside the selection) twice with the sim cap at 25 s: the whole path
    of rule 3 (reset, digest, trace, top / wrist frames, visibility, reference_assemble stopped by the cap, finish,
    finalize) and two identical results on one machine"""
    from arm.assembly_car import layout as LY
    from arm.assembly_car.cell import load_kit
    tmp = a.work or tempfile.mkdtemp(prefix="p213-phys-")
    kit_dir = ensure_kit(a.kit, tmp)
    kit = load_kit(kit_dir)
    plan = load(os.path.join(TESTS, "hidden", "bench_plan.json"))
    cand = {"id": "smoke", "seed": 1009, "stratum": "yaw+lead0"}
    cand["layout_digest"] = LY.layout_digest(LY.episode_layout(1009, plan, kit, uv=LY.unit_geometry(kit), cfg=grader_config()["layout"],
                                                                stratum="yaw+lead0"))
    robot = a.robot or os.path.join(TASK, ROBOT_TREE)
    recs = []
    for k in range(2):
        recs.append(run_episode(None, "smoke%d" % k, "host", kit_dir, robot, tmp, renderer="mujoco", sim_cap=25.0, candidate=cand))
    for r in recs:
        assert not r.get("error"), r.get("error")
        assert r["digest_ok"] and r["view"]["least"][1] > 0.0 and r["reference_ran"], r
        assert abs(r["sim_time_s"] - 25.0) < 0.02 and r["done_reason"] == "sim_cap", (r["sim_time_s"], r["done_reason"])
        assert r["trace"]["frames"] > 600 and r["ticks_hashed"] > 2000
    for key in ("tick_sha256", "fingerprint", "trace", "view", "capture", "states"):
        assert recs[0][key] == recs[1][key], key
    print("physics: tick sha %s..., trace %s..., least seen %s, states %s" % (recs[0]["tick_sha256"][:12], recs[0]["trace"]["qpos_sha256"][:12],
                                                                             recs[0]["view"]["least"], recs[0]["n_locked"]))


# ================================================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("draw")
    p.add_argument("--selection", required=True)
    p.add_argument("--robot")
    p.add_argument("--expectation", help="an a priori expectation written into the selection before the draw")
    p.add_argument("--supersedes", help="the earlier selection this one replaces (moved aside, e.g. to tests/hidden/superseded/)")
    p.add_argument("--why", help="why the earlier selection is replaced")
    p.set_defaults(fn=cmd_draw)
    p = sp.add_parser("check")
    p.add_argument("--selection", required=True)
    p.add_argument("--robot")
    p.add_argument("--json")
    p.set_defaults(fn=cmd_check)
    p = sp.add_parser("record")
    p.add_argument("--selection", required=True)
    p.add_argument("--why", required=True)
    p.add_argument("--robot")
    p.set_defaults(fn=cmd_record)
    p = sp.add_parser("digests")
    p.add_argument("--selection", required=True)
    p.add_argument("--kit")
    p.add_argument("--robot")
    p.add_argument("--work")
    p.set_defaults(fn=cmd_digests)
    for name, fn in (("episode", cmd_episode), ("run", cmd_run)):
        p = sp.add_parser(name)
        p.add_argument("--selection", required=True)
        p.add_argument("--env", required=True, choices=ENVS)
        p.add_argument("--out", required=True)
        p.add_argument("--kit")
        p.add_argument("--robot")
        p.add_argument("--renderer", default="mujoco", choices=("mujoco", "filament"))
        p.add_argument("--view-only", action="store_true")
        if name == "episode":
            p.add_argument("--cand", required=True)
        else:
            p.add_argument("-P", type=int, default=4)
            p.add_argument("--nice", type=int, default=10)
            p.add_argument("--only")
        p.set_defaults(fn=fn)
    p = sp.add_parser("image")
    p.add_argument("--selection", required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--image", default=IMAGE_DEFAULT)
    p.add_argument("--keep-stage", action="store_true")
    p.set_defaults(fn=cmd_image)
    p = sp.add_parser("image-smoke")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1009)
    p.add_argument("--stratum", default="yaw+lead0", choices=STRATA)
    p.add_argument("--sim-cap", type=float, default=25.0)
    p.add_argument("--kit")
    p.add_argument("--image", default=IMAGE_DEFAULT)
    p.set_defaults(fn=cmd_image_smoke)
    p = sp.add_parser("image-view")
    p.add_argument("--selection", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--image", default=IMAGE_DEFAULT)
    p.add_argument("-P", type=int, default=3)
    p.set_defaults(fn=cmd_image_view)
    p = sp.add_parser("ingest")
    p.add_argument("--selection", required=True)
    p.add_argument("--env", required=True, choices=ENVS)
    p.add_argument("--root", required=True, help="the folder the runs are kept under; recorded paths are relative to it")
    p.add_argument("--runs", required=True, action="append")
    p.set_defaults(fn=cmd_ingest)
    p = sp.add_parser("make")
    p.add_argument("--selection", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_make)
    p = sp.add_parser("report")
    p.add_argument("--selection", required=True)
    p.add_argument("--population")
    p.add_argument("--json")
    p.set_defaults(fn=cmd_report)
    p = sp.add_parser("accept")
    p.add_argument("--selection", required=True)
    p.add_argument("--episodes", required=True)
    p.add_argument("--root", required=True, help="the folder --out is under; recorded paths are relative to it")
    p.add_argument("--out", required=True)
    p.add_argument("--image", default=IMAGE_DEFAULT)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--parallel", action="store_true")
    p.set_defaults(fn=cmd_accept)
    p = sp.add_parser("determinism")
    p.add_argument("--selection", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--remap", action="append", help="SRC=DST: where the pod's trace files are on this machine")
    p.add_argument("--root", help="the folder the recorded run paths are relative to (default: the working folder)")
    p.set_defaults(fn=cmd_determinism)
    p = sp.add_parser("selftest")
    p.add_argument("--physics", action="store_true")
    p.add_argument("--kit")
    p.add_argument("--robot")
    p.add_argument("--work")
    p.set_defaults(fn=selftest)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
