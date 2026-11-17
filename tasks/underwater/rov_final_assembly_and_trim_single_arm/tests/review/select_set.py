#!/usr/bin/env python3
"""Pre-registered selection of the fixed hidden set of rov_final_assembly_and_trim_single_arm (a maintainer's tool; the
grader never runs it). It carries out rule_version 3 of the selection rule (-dev/review/selection_rule.json and
-dev/review/PREREGISTRATION.md, sha256 below) step by step, records every candidate and decision in
tests/hidden/selection.json and, once the set is selected, writes tests/hidden/episodes.json (format rov-asm-set/1: the
four episodes verify.py plays, one per stratum E1..E4, in that order).

The rule decides, this tool executes it. Where the rule does not decide a case the tool stops, writes the case into
selection.json and leaves it to the main session (the rule's 'no_discretion'). Steps (PREREGISTRATION.md section 4):

  S0      the sha256 of selection_rule.json and PREREGISTRATION.md are the two values of rule_version 3 that the main
          session wrote into shared/DECISIONS_P2_ASM.md, section 7.7 (and that this tool carries)
  S1      freeze. Precondition: CELL_FREEZE.json exists, lists at least the seven classes of the rule, every sha256 it
          lists equals that file now, its two numerics variables are the task's (Dockerfile = test.sh) and take
          effect in the verification image (probes as root, as the agent account and through test.sh). Then every
          frozen file, tree, the hidden kit's members and the image id are hashed into selection.json (state frozen);
          every later command re-hashes first and refuses on any difference.
  S2      self-tests before any seed is read: the stream code's known-answer test (appendix A, copied verbatim below
          and compared byte for byte with the document at run time), the reference values of
          trim_evaluation.expected_under_digest_v3, this tool's own tests, and the rehearsal on the PUBLIC test
          vectors (the trim stage until every stratum has its first qualified test candidate; (d) and (e) for both
          dome ends on the 12 test layout seeds; one test pair played by the reference program in the image).
          Go line: every stratum qualified, >= 6 test layouts pass both dome ends, the pipeline reached finalize.
  S3      precondition: P19's slow-release probe evidence (-dev/evidence/entry/slow_release_probe.json, format
          rov-asm-slow-release/2: the unit let go above its window; per unit kind -- vertical thruster, guard -- >= 20
          trials and 0 false locks, a false lock being a capture at whose moment no finger pad touches the unit; the
          captures with the pads still on the unit are placements, recorded, not limited; on the frozen cell). Then
          the lowest unspent generation's two sealed seeds are read,
          checked against the rule's commitments and published (state opened).
  S4      the dome assignment and the 12 layout seeds (appendix A).
  S5-S7   rounds. Each stratum without an episode (E1..E4 in order) takes its next qualified trim candidate (b1 -> c
          -> b2 on the readings library of P11; lazily, in index order, at most 20 000 draws) and the next layout
          candidate of c00..c11 that passes (d) and then (e) for the stratum's dome end (a candidate failing on the
          way is consumed with its reason). Every pair of a round is played ONCE in the verification image (one
          verify run per round: tests/test.sh -> verify.py with solution/solve.sh, the round's pairs as the set) and
          judged by (a); a passing pair is the stratum's episode, a failing pair consumes both members.
          FB-L (layout candidates run out) and FB-W (no qualified trim candidate within 20 000 draws): no set, stop.
  S8      episodes.json (E1..E4) and selection.json, state selected.
  S9      two more verify runs of episodes.json in the image: both reward 1.0, identical per-episode scores ->
          accepted (otherwise rejected, no set).
  S10     report: the trim stage's counts, every drop and its cause, the hollowing flags, the diagnostics.
  After the acceptance (P32b): a change the main session classifies as class N or V is written into selection.json's
          `amendments` by `amend` (the rule's entry format -- utc, class, what, why, authorised_by,
          superseded_selection_sha256, generation_opened, rule_version -- plus every frozen value it changes, with its
          frozen / before / after sha256, and the maintainer's detail); `check` then reports a difference whose value now
          is the `after` of the last entry that recorded it as explained, any other as unexplained (exit 1). Class S is a
          new selection with the next generation and is never an entry here.

    python3 -I -B select_set.py selftest
    python3 -I -B select_set.py freeze   [--image TAG] [--why TEXT]     S0, S1 (a new freeze only before any opening)
    python3 -I -B select_set.py check                                   the freeze (exit 1 on any difference that no
                                                                        amendment entry explains)
    python3 -I -B select_set.py amend --class N|V --what T --why T --authorised-by T [--detail F.json]
                                                                        after the acceptance: one amendment entry (below)
    python3 -I -B select_set.py rehearse [-P 4]                         S2
    python3 -I -B select_set.py open                                    S3, S4
    python3 -I -B select_set.py dryopen                                 a DRY run of S4-S9 on the PUBLIC test vectors (with
                                                                        ROV_ASM_SEL_PATH / _SET_PATH / _EVIDENCE elsewhere)
    python3 -I -B select_set.py select   [-P 4]                         S5-S8 (resumes from selection.json)
    python3 -I -B select_set.py accept                                  S9
    python3 -I -B select_set.py report                                  S10
    python3 -I -B select_set.py view --jobs JOBS.json --out OUT.json    (internal: condition (e) for (seed, dome) jobs)

Environment: the two pinned numerics variables of the task (-dev/tools/pinned_env.py --sh; refused without them);
MUJOCO_GL=osmesa with the OSMesa library path for (e); Docker (DOCKER_HOST) for the image runs; ARB_AGENT_RENDER_DIR,
the render payload of the views. TASK = this file's ../.., DEV = TASK + '-dev' (ROV_ASM_DEV overrides), the scratch of
stages and run outputs = ROV_ASM_P20_WORK (default DEV/evidence/hidden_set/work).
"""
import argparse
import datetime
import fnmatch
import hashlib
import hmac
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
TASK = os.path.dirname(TESTS)
SLUG = os.path.basename(TASK)
DEV = os.environ.get("ROV_ASM_DEV") or TASK + "-dev"
UW = os.path.dirname(os.path.dirname(TASK))
HIDDEN = os.path.join(TESTS, "hidden")
OFFICIAL_SEL_PATH = os.path.join(HIDDEN, "selection.json")
OFFICIAL_SET_PATH = os.path.join(HIDDEN, "episodes.json")
# a dry run of the selection loop on the PUBLIC test vectors writes elsewhere (ROV_ASM_SEL_PATH / ROV_ASM_SET_PATH /
# ROV_ASM_EVIDENCE) and can never read a sealed seed or write the task's files (`dryopen`)
SEL_PATH = os.environ.get("ROV_ASM_SEL_PATH") or OFFICIAL_SEL_PATH
SET_PATH = os.environ.get("ROV_ASM_SET_PATH") or OFFICIAL_SET_PATH
DRY = os.path.abspath(SEL_PATH) != os.path.abspath(OFFICIAL_SEL_PATH)
SUPERSEDED = os.path.join(HIDDEN, "superseded")
RULE_PATH = os.path.join(DEV, "review", "selection_rule.json")
PREREG_PATH = os.path.join(DEV, "review", "PREREGISTRATION.md")
DECISIONS_PATH = os.path.join(UW, "shared", "DECISIONS.md")
DECISIONS_ASM_PATH = os.path.join(UW, "shared", "DECISIONS_P2_ASM.md")
DECISIONS_ASM_SECTION = "### 7.7 "     # the heading of the section that holds rule_version 3's two sha256
CELL_FREEZE_PATH = os.path.join(DEV, "evidence", "cell", "CELL_FREEZE.json")
SLOW_RELEASE_PATH = os.path.join(DEV, "evidence", "entry", "slow_release_probe.json")
SEALED_DIR = os.environ.get("ROV_ASM_SEALED") or "/data/xiatao/tmp/underwater_20261114/work/p2/asm/P12/sealed"
EVIDENCE = os.environ.get("ROV_ASM_EVIDENCE") or os.path.join(DEV, "evidence", "hidden_set")
WORK = os.environ.get("ROV_ASM_P20_WORK") or os.path.join(EVIDENCE, "work")
V12 = os.path.join(DEV, "trim", "v12")
TOOLS = os.path.join(DEV, "tools")
RUN_IN_IMAGE = os.path.join(DEV, "evidence", "skeleton", "smoke_kit", "run_in_image.sh")
ROBOT_TREE = os.path.join(DEV, "build", "robot", "tree")
GT_TARBALL = os.path.join(TESTS, "assets", "gt.tar.xz")
IMAGE_DEFAULT = os.environ.get("ROV_ASM_IMAGE", "rov-asm-task:p20")    # built from the task's environment/

TOOL_FORMAT = "rov-asm-selection/1"
SET_FORMAT = "rov-asm-set/1"
# rule_version 3 (P18e, 2026-11-16T12:39:24Z): the two values the main session wrote into shared/DECISIONS_P2_ASM.md
# section 7.7 (2026-11-16 07:48 EST). They supersede rule_version 2 (7f715dbb...ddd5, e5e4646e...7670; section 6.1)
# and rule_version 1 (record P2_P12.md), whose values do not pass.
RULE_VERSION = 3
RULE_SHA256 = "2a29d03489bf751070f14b91b22d3b85d6aae436c0868172bd9a258d706fc5cf"
PREREG_SHA256 = "1319da80df3c781d1bb23596a7fea1d4b54719136cff41553acf72a1aab9717c"
SLOW_RELEASE_FORMAT = "rov-asm-slow-release/2"     # P19b's re-summary under ruling 7.1 (format /1 counted strictly)
STRATA = ("E1", "E2", "E3", "E4")
N_LAYOUT = 12
WALL_LIMIT_S = 4 * 3600            # one verify run in the image (four episodes); a run cut by it is run again
RUN_ATTEMPTS = 3                   # infrastructure interruptions (no verdict) are run again, at most this often

# ------------------------------------------------------------------------------------------------ the frozen table
# "<root>:<path>": task = the task folder, dev = its -dev folder, uw = docs/plan/underwater, env:<VAR> = a folder named
# by an environment variable. Files are frozen by their bytes, trees by the sorted (relative path, sha256) list of their
# files (no __pycache__ / *.pyc), the hidden kit additionally by its members' bytes.
FROZEN_FILES = [
    # the rule
    "dev:review/selection_rule.json", "dev:review/PREREGISTRATION.md",
    # trim truth and the public route (the readings library of P11 and what it reads)
    "task:tests/trim_table.json", "task:tests/trim_sheet_printed.json", "task:tests/cell_public.json",
    "dev:trim/v12/readings.py", "dev:trim/v12/public_route.py", "dev:trim/v12/sampler.py",
    "dev:trim/v12/trim_sheet_printed.json", "dev:trim/v12/cell_public.json", "dev:trim/v12/work_order_example.json",
    "uw:shared/spec_values.json",
    # layout, cell and grading
    "task:tests/cell_layout.json", "task:tests/fixture_facts.json", "task:tests/grader_config.json",
    "task:tests/verify.py", "task:tests/test.sh", "task:tests/verify_freeze.py", "task:tests/netguard.py",
    "task:tests/episode_trace.py", "task:tests/anchor.json",
    # kit
    "task:tests/assets/gt.tar.xz",
    # reference program and oracle
    "task:solution/solve.sh", "task:solution/assemble.py", "task:tests/hidden/oracle_token.sha256",
    # the verification image's build inputs (since the resume of 2026-11-16 the image is built from the task's own
    # environment/, its local asset folder pinned file by file by assets.sha256) and the givens the verifier checks
    # /opt/arb against (P24 / P24b)
    "task:environment/Dockerfile", "task:environment/requirements.txt", "task:environment/requirements-photo.txt",
    "task:environment/fetch_assets.py", "task:environment/setup/setup.sh", "task:environment/assets.sha256",
    "task:tests/given_hashes.json",
    # view check, the cell freeze and the tools of this selection
    "dev:tools/coverage_check.py", "dev:tools/pinned_env.py", "dev:tools/cell_freeze.py",
    "dev:evidence/skeleton/smoke_kit/run_in_image.sh", "dev:evidence/cell/CELL_FREEZE.json",
    "task:tests/review/select_set.py",
]
# the whole grader folder (less what this selection writes, the maintainers' notes and the agent-visible copies P24
# places there) and the code / models underneath
TESTS_TREE_EXCLUDE = ("hidden/selection.json", "hidden/episodes.json", "hidden/superseded/", "README.md",
                      "setup_payload/", "given_hashes.json")
FROZEN_TREES = ["task:tests", "task:tests/arm", "task:tests/common", "task:tests/render", "task:tests/render_filament",
                "task:tests/room", "task:tests/robotics_grader", "task:solution", "dev:build/robot/tree",
                "env:ARB_AGENT_RENDER_DIR", "task:tests/setup_payload"]
RECORDED_FILES = ["task:tests/README.md"]       # recorded by hash; a change is a note, not a violation


# ---- BEGIN NORMATIVE STREAM CODE (rov-asm-hidden/1) ----
import hashlib, hmac

DOMAIN = "rov-asm-hidden/1"
BALANCED_DOME = [(0, 0, 180, 180), (0, 180, 0, 180), (0, 180, 180, 0),
                 (180, 0, 0, 180), (180, 0, 180, 0), (180, 180, 0, 0)]      # dome end of (E1, E2, E3, E4), deg


def commitment(raw):
    """what is published before the seed: sha256 over the raw seed bytes"""
    return hashlib.sha256(bytes(raw)).hexdigest()


def u64(key, *parts):
    """HMAC-SHA256(key, 'rov-asm-hidden/1|part|part|...') -> first 8 bytes, big-endian"""
    msg = "|".join([DOMAIN] + [str(p) for p in parts]).encode("ascii")
    return int.from_bytes(hmac.new(bytes(key), msg, hashlib.sha256).digest()[:8], "big")


def grid(key, lo, hi, *parts):
    """uniform integer on [lo, hi] (both ends included)"""
    return lo + u64(key, *parts) % (hi - lo + 1)


def trim_candidate(key_w, stratum, deviation, c):
    """candidate c (0, 1, 2, ...) of one stratum of stream W.
    stratum = an element of selection_rule.json 'strata'; deviation = its 'per_unit_deviation'."""
    sid = stratum["id"]
    t10 = grid(key_w, stratum["T_C_tenths"][0], stratum["T_C_tenths"][1], "W", sid, c, "T")
    s10 = grid(key_w, stratum["SP_tenths"][0], stratum["SP_tenths"][1], "W", sid, c, "SP")
    dm = stratum["mass_sign"] * grid(key_w, deviation["dm_abs_g"][0], deviation["dm_abs_g"][1], "W", sid, c, "dm")
    dx = grid(key_w, deviation["dx_centimm"][0], deviation["dx_centimm"][1], "W", sid, c, "dx")
    dy = grid(key_w, deviation["dy_centimm"][0], deviation["dy_centimm"][1], "W", sid, c, "dy")
    serial = grid(key_w, 1000, 9999, "W", sid, c, "unit")
    return {"stratum": sid, "index": c, "T_C_tenths": t10, "SP_tenths": s10, "dm_g": dm, "dx_centimm": dx,
            "dy_centimm": dy, "unit_serial": serial,
            "T_C": t10 / 10.0, "SP": s10 / 10.0, "dm_kg": dm / 1000.0, "dx_mm": dx / 100.0, "dy_mm": dy / 100.0}


def noise_seed(key_w, sid, c):
    """128-bit seed of the numpy PCG64 generator of candidate c's noise trials (condition b2)"""
    msg = "|".join([DOMAIN, "W", sid, str(c), "noise"]).encode("ascii")
    return int.from_bytes(hmac.new(bytes(key_w), msg, hashlib.sha256).digest()[:16], "big")


def dome_assignment(key_w):
    """dome end (deg) of E1..E4: one of the six balanced assignments"""
    return BALANCED_DOME[u64(key_w, "W", "dome") % 6]


def layout_seeds(blob_l):
    """stream L: 96 raw bytes -> the 12 layout candidate seeds c00..c11"""
    blob_l = bytes(blob_l)
    if len(blob_l) != 96:
        raise ValueError("stream L is 96 bytes")
    seeds = [int.from_bytes(blob_l[8 * i:8 * i + 8], "little") % (1 << 62) for i in range(12)]
    if min(seeds) < (1 << 32) or len(set(seeds)) != 12:
        raise ValueError("stream L breaks its sealing constraints (seed < 2**32 or repeated)")
    return seeds


def prior_lattice(deviation_range):
    """the 200 points of the prior-robust no-weigh reading: 8 x 5 x 5 cell midpoints of the published range"""
    (m0, m1), (x0, x1), (y0, y1) = deviation_range["mass_kg"], deviation_range["cg_x_mm"], deviation_range["cg_y_mm"]
    return [(m0 + (m1 - m0) * (i + 0.5) / 8, x0 + (x1 - x0) * (j + 0.5) / 5, y0 + (y1 - y0) * (k + 0.5) / 5)
            for i in range(8) for j in range(5) for k in range(5)]
# ---- END NORMATIVE STREAM CODE (rov-asm-hidden/1) ----


# ------------------------------------------------------------------------------------------------ small helpers
def log(msg):
    sys.stdout.write("[select_set %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stdout.flush()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
    except ImportError:
        pass
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError("not JSON serialisable: %r" % (type(o),))


def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
        f.write("\n")
    os.replace(tmp, path)


class Stop(Exception):
    """the rule does not let the selection go on: the case goes to the main session"""


def resolve(key):
    root, rel = key.split(":", 1)
    if root == "task":
        return os.path.join(TASK, rel)
    if root == "dev":
        return os.path.join(DEV, rel)
    if root == "uw":
        return os.path.join(UW, rel)
    if root == "env":
        v = os.environ.get(rel)
        if not v:
            raise Stop("the environment variable %s (a frozen tree) is not set" % rel)
        return v
    raise ValueError(key)


def tree_digest(root, exclude=()):
    """(sha256 over the sorted lines '<sha256>  <relative path>', file count) of the files under root"""
    lines = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for fn in fns:
            if fn.endswith(".pyc"):
                continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            if any(rel == e or (e.endswith("/") and rel.startswith(e)) for e in exclude):
                continue
            lines.append("%s  %s" % (sha256_file(p), rel))
    lines.sort(key=lambda s: s[66:])
    return sha256_bytes(("\n".join(lines) + "\n").encode()), len(lines)


def kit_members_digest(path=GT_TARBALL):
    """sha256 over the sorted lines '<member> <sha256 of its bytes>' of the archive's file members, and their number"""
    lines = []
    with tarfile.open(path, "r:xz") as tf:
        for m in tf.getmembers():
            if m.isfile():
                lines.append("%s %s" % (m.name, sha256_bytes(tf.extractfile(m).read())))
    lines.sort()
    return sha256_bytes(("\n".join(lines) + "\n").encode()), len(lines)


def ensure_kit():
    """the hidden kit unpacked from tests/assets/gt.tar.xz into WORK/kit/gt (the folder (e) renders), checked against
    the archive member by member"""
    dst = os.path.join(WORK, "kit")
    gt = os.path.join(dst, "gt")
    want = {}
    with tarfile.open(GT_TARBALL, "r:xz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        for m in members:
            want[m.name] = sha256_bytes(tf.extractfile(m).read())
        ok = os.path.isdir(gt) and all(os.path.isfile(os.path.join(dst, n)) and sha256_file(os.path.join(dst, n)) == h
                                       for n, h in want.items())
        if not ok:
            shutil.rmtree(dst, ignore_errors=True)
            os.makedirs(dst)
            for m in tf.getmembers():
                if not (m.isfile() or m.isdir()) or m.name.startswith("/") or ".." in m.name.split("/"):
                    raise Stop("the hidden kit archive holds an unexpected member %r" % m.name)
            tf.extractall(dst, filter="data")
    have = sum(len(fns) for _dp, _dn, fns in os.walk(gt))
    if have != len(want):
        raise Stop("the unpacked kit holds %d files, the archive %d" % (have, len(want)))
    return gt


def git_free_tree_check():
    """no sealed seed file anywhere in the development documents (they stay outside the repository until opened)"""
    found = []
    for dp, dns, fns in os.walk(os.path.join(UW)):
        dns[:] = [d for d in dns if d not in ("__pycache__", "assets")]
        for fn in fns:
            if fnmatch.fnmatch(fn, "gen*_W.hex") or fnmatch.fnmatch(fn, "gen*_L.hex"):
                found.append(os.path.relpath(os.path.join(dp, fn), UW))
    return found


# ------------------------------------------------------------------------------------------------ the rule (S0)
def rule():
    return load(RULE_PATH)


def appendix_a_block(text):
    """the lines of PREREGISTRATION.md's normative block (inside the fenced code of appendix A), markers included"""
    lines = text.split("\n")
    b = [i for i, ln in enumerate(lines) if ln.startswith("# ---- BEGIN NORMATIVE STREAM CODE (rov-asm-hidden/1) ----")]
    e = [i for i, ln in enumerate(lines) if ln.startswith("# ---- END NORMATIVE STREAM CODE (rov-asm-hidden/1) ----")]
    if len(b) != 1 or len(e) != 1 or e[0] <= b[0]:
        raise Stop("PREREGISTRATION.md: the normative block's markers are not there exactly once")
    return "\n".join(lines[b[0] + 1:e[0]])


def own_block():
    return appendix_a_block(open(os.path.abspath(__file__)).read())


def decisions_section(text, heading):
    """the lines of a markdown file from the heading line that starts with `heading` to the next heading of level 2 or
    3 (or the end); None when the heading is not there exactly once"""
    lines = text.split("\n")
    at = [i for i, ln in enumerate(lines) if ln.startswith(heading)]
    if len(at) != 1:
        return None
    end = len(lines)
    for j in range(at[0] + 1, len(lines)):
        if lines[j].startswith("## ") or lines[j].startswith("### "):
            end = j
            break
    return "\n".join(lines[at[0]:end])


def git_head_record(paths):
    """information, not a condition of S0: the bytes of the two rule files in the dev repository's HEAD (read-only git
    commands; None where git or the repository is not there) -- the main session's time evidence (DECISIONS_P2_ASM 7.7)"""
    out = {}
    try:
        top = subprocess.run(["git", "-C", UW, "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=60)
        if top.returncode != 0:
            return None
        top = top.stdout.strip()
        head = subprocess.run(["git", "-C", top, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60).stdout.strip()
        out["head"] = head
        for p in paths:
            rel = os.path.relpath(p, top)
            show = subprocess.run(["git", "-C", top, "show", "HEAD:" + rel], capture_output=True, timeout=60)
            last = subprocess.run(["git", "-C", top, "log", "-1", "--format=%H %cI", "--", rel], capture_output=True, text=True, timeout=60)
            out[os.path.basename(p)] = {"sha256_in_head": hashlib.sha256(show.stdout).hexdigest() if show.returncode == 0 else None,
                                        "last_commit": last.stdout.strip() or None}
        return out
    except (OSError, subprocess.SubprocessError):
        return None


def check_s0():
    """S0: the two files' sha256 against the values this tool carries and the ones the main session wrote into
    shared/DECISIONS_P2_ASM.md section 7.7 (rule_version 3)"""
    r = {"rule": {"path": os.path.relpath(RULE_PATH, UW), "sha256": sha256_file(RULE_PATH)},
         "preregistration": {"path": os.path.relpath(PREREG_PATH, UW), "sha256": sha256_file(PREREG_PATH)}}
    sec = decisions_section(open(DECISIONS_ASM_PATH, encoding="utf-8").read(), DECISIONS_ASM_SECTION)
    r["decisions"] = {"path": os.path.relpath(DECISIONS_ASM_PATH, UW), "sha256": sha256_file(DECISIONS_ASM_PATH),
                      "section": DECISIONS_ASM_SECTION.strip(), "section_found": sec is not None,
                      "section_sha256": hashlib.sha256(sec.encode("utf-8")).hexdigest() if sec is not None else None,
                      "holds_rule_sha256": sec is not None and RULE_SHA256 in sec,
                      "holds_preregistration_sha256": sec is not None and PREREG_SHA256 in sec}
    dec = open(DECISIONS_PATH, encoding="utf-8").read()
    r["decisions_md"] = {"path": os.path.relpath(DECISIONS_PATH, UW), "sha256": sha256_file(DECISIONS_PATH),
                         "holds_rule_sha256": RULE_SHA256 in dec, "holds_preregistration_sha256": PREREG_SHA256 in dec,
                         "role": "information (its last 03 section points to DECISIONS_P2_ASM 7.7 for the two values)"}
    r["rule"]["rule_version"] = rule()["rule_version"]
    r["git_head"] = git_head_record([RULE_PATH, PREREG_PATH])
    ok = (r["rule"]["sha256"] == RULE_SHA256 and r["preregistration"]["sha256"] == PREREG_SHA256
          and r["decisions"]["holds_rule_sha256"] and r["decisions"]["holds_preregistration_sha256"]
          and r["rule"]["rule_version"] == RULE_VERSION)
    r["pass"] = bool(ok)
    return r


# ------------------------------------------------------------------------------------------------ numerics
def pinned_env_module():
    sys.path.insert(0, TOOLS)
    import pinned_env
    sys.path.remove(TOOLS)
    return pinned_env


def need_pinned():
    pe = pinned_env_module()
    r = pe.read(TASK)
    if not (r["agree"] and r["test_sh"]["before_exec"] and r["equal_the_ruling"]):
        raise Stop("environment/Dockerfile and tests/test.sh do not pin the same numerics: %r" % (r,))
    bad = {k: os.environ.get(k) for k, v in r["values"].items() if os.environ.get(k) != v}
    if bad:
        raise Stop("this process runs without the task's pinned numerics (eval \"$(python -I -B %s --sh)\"): %r" % (
            os.path.join(TOOLS, "pinned_env.py"), bad))
    return r


def numerics_ok(p, values):
    """a probe of pinned_env.py: the variables are the pinned values, every OpenBLAS is Haswell, NumPy's dispatch has no
    512-bit target"""
    if not isinstance(p, dict):
        return False
    env_ok = all(p.get("env", {}).get(k) == v for k, v in values.items())
    cores = p.get("openblas_core") or {}
    core_ok = bool(cores) and all(v == "Haswell" for v in cores.values())
    disp = set(p.get("numpy_dispatch_on") or [])
    return bool(env_ok and core_ok and not (disp & {"X86_V4", "AVX512_ICL", "AVX512_SPR", "AVX512F", "AVX512_SKX"}))


# ------------------------------------------------------------------------------------------------ docker
def docker_env():
    e = dict(os.environ)
    e.setdefault("DOCKER_HOST", "unix:///var/run/docker.sock")
    return e


def docker_image_id(image):
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, text=True, env=docker_env())
    if r.returncode != 0:
        raise Stop("no image %s (build it: make_smoke.sh --image --tag %s)" % (image, image))
    return r.stdout.strip()


def run_in_image(image, tests_dir, out_dir, script, mounts=(), envs=(), timeout=WALL_LIMIT_S, log_path=None):
    """one command in the verification image with the task sandbox's resources (P01's run_in_image.sh: 8 CPUs, 24 GB,
    no network, tests/ mounted read-only at /opt/ale/tests, out_dir at /out); returns (exit code, wall s, timed out)"""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [RUN_IN_IMAGE, "--image", image, "--tests", tests_dir, "--out", out_dir]
    for m in mounts:
        cmd += ["--mount", m]
    for e in envs:
        cmd += ["--env", e]
    cmd += ["--", "bash", "-c", script]
    t0 = time.time()
    lf = open(log_path, "a") if log_path else subprocess.DEVNULL
    try:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=docker_env(), start_new_session=False)
        try:
            rc = p.wait(timeout=timeout)
            to = False
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                rc = p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
                rc = p.wait()
            to = True
    finally:
        if log_path:
            lf.close()
    return rc, time.time() - t0, to


# ------------------------------------------------------------------------------------------------ S1: the cell freeze
def check_cell_freeze(rl):
    """precondition S1_cell_freeze: the file, every listed sha256 against the file now, the classes it must list, the
    numerics section"""
    pre = rl["procedure"]["preconditions"]["S1_cell_freeze"]
    if os.path.abspath(pre["file"]) != os.path.abspath(CELL_FREEZE_PATH):
        raise Stop("the rule names another cell freeze file: %s" % pre["file"])
    out = {"file": os.path.relpath(CELL_FREEZE_PATH, UW), "exists": os.path.isfile(CELL_FREEZE_PATH)}
    if not out["exists"]:
        out["pass"] = False
        return out
    out["sha256"] = sha256_file(CELL_FREEZE_PATH)
    cf = load(CELL_FREEZE_PATH)
    out["format"], out["package"], out["date"] = cf.get("format"), cf.get("package"), cf.get("date")
    roots = {"frozen": TESTS, "also_pinned_tests": TESTS, "also_pinned_environment": TASK, "also_pinned_dev": DEV}
    rows, problems = [], []
    for group, files in (cf.get("sha256") or {}).items():
        root = roots.get(group)
        if root is None:
            problems.append("unknown group %s" % group)
            continue
        for rel, want in files.items():
            p = os.path.join(root, rel)
            now = sha256_file(p) if os.path.isfile(p) else None
            rows.append({"group": group, "file": rel, "listed": want, "now": now, "equal": now == want})
            if now != want:
                problems.append("%s:%s listed %s now %s" % (group, rel, want, now))
    out["files"] = rows
    listed = [r["file"] for r in rows]
    base = [os.path.basename(f) for f in listed]
    must = []
    for pat in pre["must_list_at_least"]:
        hit = sorted(set(b for b in base if fnmatch.fnmatch(b, pat)))
        ok = bool(hit)
        if "*" in pat:      # every module of that name in the cell's package is listed
            pkg = os.path.join(TESTS, "arm", "assembly_rov")
            have = sorted(fn for fn in os.listdir(pkg) if fnmatch.fnmatch(fn, pat))
            ok = ok and set(have) <= set(hit)
            hit = {"listed": hit, "in_the_package": have}
        must.append({"pattern": pat, "found": hit, "pass": ok})
        if not ok:
            problems.append("CELL_FREEZE.json does not list %s" % pat)
    out["must_list_at_least"] = must
    pe = pinned_env_module().read(TASK)
    env = cf.get("environment") or {}
    out["environment"] = {"listed": env.get("variables"), "dockerfile_and_test_sh": pe["values"], "agree": pe["agree"],
                          "test_sh_before_exec": pe["test_sh"]["before_exec"],
                          "pass": env.get("variables") == pe["values"] and pe["agree"] and pe["test_sh"]["before_exec"]}
    if not out["environment"]["pass"]:
        problems.append("the numerics section of CELL_FREEZE.json is not the task's")
    out["problems"] = problems
    out["pass"] = not problems
    return out


PROBE_SH = """#!/bin/bash
# P20: stands in for the verifier's interpreter at tests/test.sh's exec (ALE_VERIFY_PYTHON): records the numerics the
# verifier starts with, then stops (nothing is verified)
exec python3 -I -B /opt/p20tools/pinned_env.py --probe > /out/probe_${PROBE_TAG:-test_sh}.json
"""


def image_probe(image, values):
    """the two variables in the verification image: its own environment as root and as the agent account, and through
    tests/test.sh (the verifier's interpreter replaced by the probe), once with the image's two ENV removed first"""
    d = os.path.join(WORK, "image_probe")
    shutil.rmtree(d, ignore_errors=True)
    out = os.path.join(d, "out")
    tools = os.path.join(d, "tools")
    os.makedirs(tools)
    shutil.copy2(os.path.join(TOOLS, "pinned_env.py"), os.path.join(tools, "pinned_env.py"))
    with open(os.path.join(tools, "probe_python.sh"), "w") as f:
        f.write(PROBE_SH)
    os.chmod(os.path.join(tools, "probe_python.sh"), 0o755)
    stage = os.path.join(d, "tests")
    build_stage(stage, None)
    m = ["%s:/opt/p20tools:ro" % tools]
    lg = os.path.join(d, "probe.log")
    scripts = {
        "image_root": "python3 -I -B /opt/p20tools/pinned_env.py --probe > /out/probe_image_root.json",
        "image_user": "runuser -u user -- python3 -I -B /opt/p20tools/pinned_env.py --probe > /out/probe_image_user.json",
        "test_sh": "mkdir -p /opt/ale/stage && cp -r /opt/ale/tests/. /opt/ale/stage/ && cd /opt/ale/stage && "
                   "PROBE_TAG=test_sh ALE_VERIFY_PYTHON=/opt/p20tools/probe_python.sh bash test.sh",
        "test_sh_without_image_env": "mkdir -p /opt/ale/stage && cp -r /opt/ale/tests/. /opt/ale/stage/ && cd /opt/ale/stage && "
                                     "env -u OPENBLAS_CORETYPE -u NPY_DISABLE_CPU_FEATURES PROBE_TAG=test_sh_without_image_env "
                                     "ALE_VERIFY_PYTHON=/opt/p20tools/probe_python.sh bash test.sh",
    }
    res = {}
    for tag, sc in scripts.items():
        rc, wall, _to = run_in_image(image, stage, out, sc, mounts=m, timeout=1800, log_path=lg)
        p = os.path.join(out, "probe_%s.json" % tag)
        res[tag] = {"exit": rc, "wall_s": round(wall, 1), "probe": load(p) if os.path.isfile(p) else None}
        res[tag]["pass"] = rc == 0 and numerics_ok(res[tag]["probe"], values)
    return {"image": image, "probes": res, "pass": all(v["pass"] for v in res.values())}


# ------------------------------------------------------------------------------------------------ S1: the freeze
def versions():
    v = {"python": platform.python_version(), "host": platform.node(), "machine": platform.machine()}
    try:
        cpu = [ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")]
        v["cpu"] = cpu[0] if cpu else None
    except OSError:
        v["cpu"] = None
    for mod in ("numpy", "scipy", "gsw", "mujoco", "cv2"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            v[mod] = None
    return v


def frozen_hashes():
    files = {}
    for k in FROZEN_FILES:
        p = resolve(k)
        files[k] = sha256_file(p) if os.path.isfile(p) else None
    trees = {}
    for k in FROZEN_TREES:
        p = resolve(k)
        ex = TESTS_TREE_EXCLUDE if k == "task:tests" else ()
        if not os.path.isdir(p):
            trees[k] = None
            continue
        dg, n = tree_digest(p, ex)
        trees[k] = {"sha256": dg, "files": n}
    km, kn = kit_members_digest()
    rec = {}
    for k in RECORDED_FILES:
        p = resolve(k)
        rec[k] = sha256_file(p) if os.path.isfile(p) else None
    return {"files": files, "trees": trees, "kit_members": {"sha256": km, "files": kn}, "recorded": rec}


def check_frozen(sel, quiet=False, image=True, now=None):
    """every frozen hash against the files now (and the image id); [] when nothing changed (`now`: frozen_hashes()
    already computed by the caller)"""
    fr = sel.get("frozen")
    if not fr:
        raise Stop("selection.json has no freeze (run `freeze`)")
    now = now if now is not None else frozen_hashes()
    diff = []
    for grp in ("files", "trees", "kit_members"):
        a, b = fr["hashes"][grp], now[grp]
        if grp == "kit_members":
            if a != b:
                diff.append("kit_members")
            continue
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                diff.append("%s %s" % (grp, k))
    if image:
        try:
            iid = docker_image_id(fr["image"]["tag"])
        except Stop as e:
            iid = str(e)
        if iid != fr["image"]["id"]:
            diff.append("image %s: %s, frozen %s" % (fr["image"]["tag"], iid, fr["image"]["id"]))
    notes = [k for k, v in now["recorded"].items() if fr["hashes"]["recorded"].get(k) != v]
    if diff and not quiet:
        for d_ in diff:
            log("FROZEN FILE CHANGED: %s" % d_)
    if notes and not quiet:
        log("note: recorded (not frozen) files changed: %s" % notes)
    return diff


def require_frozen(sel, image=True):
    d = check_frozen(sel, image=image)
    if d:
        raise Stop("a frozen file changed after the freeze: %s" % d)


def cmd_freeze(a):
    if DRY:
        raise Stop("freeze is never run in a dry run")
    if os.path.isfile(SEL_PATH):
        old = load(SEL_PATH)
        if old.get("reveal"):
            raise Stop("selection.json already holds an opened generation: a new freeze is a new selection (amendment)")
        prev = {"sha256": sha256_file(SEL_PATH), "state": old.get("state"), "frozen_utc": (old.get("frozen") or {}).get("utc"),
                "why_frozen_again": a.why or "a new freeze before any generation was opened"}
        log("an earlier freeze (state %s) is replaced before any opening: %s" % (old.get("state"), prev["why_frozen_again"]))
    else:
        prev, old = None, None
    s0 = check_s0()
    log("S0 rule %s (%s) / preregistration %s: %s" % (s0["rule"]["sha256"][:12], s0["rule"]["rule_version"],
                                                     s0["preregistration"]["sha256"][:12], "pass" if s0["pass"] else "FAIL"))
    if not s0["pass"]:
        raise Stop("S0: the rule files are not the ones of shared/DECISIONS_P2_ASM.md 7.7: %r" % (s0,))
    rl = rule()
    need_pinned()
    cfz = check_cell_freeze(rl)
    log("S1 precondition CELL_FREEZE.json %s: %d files, %s" % ((cfz.get("sha256") or "-")[:12], len(cfz.get("files", [])),
                                                             "pass" if cfz["pass"] else "FAIL %s" % cfz.get("problems")))
    pe = pinned_env_module().read(TASK)
    iid = docker_image_id(a.image)
    probe = image_probe(a.image, pe["values"])
    cfz["environment"]["in_the_image"] = probe
    cfz["environment"]["pass_in_the_image"] = probe["pass"]
    log("S1 precondition: the numerics in the image %s: %s" % (a.image, "pass" if probe["pass"] else "FAIL"))
    sel = {"format": TOOL_FORMAT, "task": SLUG, "state": None, "states": [],
           "rule": {"path": s0["rule"]["path"], "sha256": s0["rule"]["sha256"], "rule_version": s0["rule"]["rule_version"]},
           "preregistration": s0["preregistration"], "s0": s0,
           "rulings": {"decisions": s0["decisions_md"], "group_q": s0["decisions"],
                       "read": ["DECISIONS.md: 开工闸门, 停自己的作业, 03 题 rule_version 2 的两个 sha256 与 P18d, W5, "
                                "03 题：P19 之后的裁决与 rule_version 3 (P20 的开工闸门)",
                                "DECISIONS_P2_ASM.md: 0.2, 1 (P12-1 .. P12-6, P13-1), 2 (the row for P20), 6.1, 6.3, 6.4, "
                                "7 (7.1 the S3 precondition, 7.4, 7.5 room.json, 7.6, 7.7 the two sha256 and the gate)"]},
           "preconditions": {"S1_cell_freeze": cfz}, "amendments": [], "supersedes": None, "earlier_freeze": prev,
           "tool": {"path": os.path.relpath(os.path.abspath(__file__), TASK), "sha256": sha256_file(os.path.abspath(__file__))}}
    if not (cfz["pass"] and probe["pass"]):
        sel["state"] = "stopped_before_freeze"
        sel["states"].append({"state": sel["state"], "utc": utc_now()})
        save(SEL_PATH, sel)
        raise Stop("S1 precondition (the cell freeze) does not hold: %s" % (cfz.get("problems") or "the numerics in the image"))
    sealed = []
    for g in range(1, 5):
        for s in ("W", "L"):
            p = os.path.join(SEALED_DIR, "gen%d_%s.hex" % (g, s))
            st = os.stat(p) if os.path.exists(p) else None
            sealed.append({"file": os.path.basename(p), "exists": st is not None,
                           "mode": oct(st.st_mode & 0o777) if st else None, "size": st.st_size if st else None,
                           "mtime_utc": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if st else None})
    copies = git_free_tree_check()
    log("freeze: hashing the frozen files and trees ...")
    t0 = time.time()
    hashes = frozen_hashes()
    missing = [k for k, v in hashes["files"].items() if v is None] + [k for k, v in hashes["trees"].items() if v is None]
    if missing:
        raise Stop("frozen files missing: %s" % missing)
    sel["frozen"] = {"utc": utc_now(), "hashes": hashes, "image": {"tag": a.image, "id": iid}, "versions": versions(),
                     "numerics": pe["values"], "cell_freeze_sha256": cfz["sha256"],
                     "sealed_files_metadata_only": sealed, "sealed_copies_in_the_documents": copies,
                     "hash_seconds": round(time.time() - t0, 1),
                     "frozen_table": {"files": FROZEN_FILES, "trees": FROZEN_TREES, "tests_tree_excludes": list(TESTS_TREE_EXCLUDE),
                                      "recorded": RECORDED_FILES}}
    if any(not s["exists"] or s["mode"] != "0o600" for s in sealed) or copies:
        raise Stop("the sealed seed files are not as sealed (8 files, mode 0600, no copy in the documents): %r %r" % (sealed, copies))
    sel["state"] = "frozen"
    sel["states"].append({"state": "frozen", "utc": sel["frozen"]["utc"]})
    save(SEL_PATH, sel)
    log("S1 frozen at %s: %d files, %d trees, kit %d members, image %s %s" % (
        sel["frozen"]["utc"], len(hashes["files"]), len(hashes["trees"]), hashes["kit_members"]["files"], a.image, iid[7:19]))
    return 0


def frozen_value(now, key):
    """the current value of one key of check_frozen's list ("files K", "trees K", "kit_members"); None for an image"""
    grp, _, k = key.partition(" ")
    if grp == "kit_members":
        return now["kit_members"]
    if grp in ("files", "trees"):
        return now[grp].get(k)
    return None


def amendment_coverage(sel, changed, now):
    """({key: index of the amendment that explains it}, [keys no amendment explains]) for check_frozen's list `changed`.
    A key is explained when the LAST amendment that lists it under `changes` recorded as `after` exactly its value now
    (a later change of the same file, or an image id, is never explained). Class S and rule entries explain nothing: a
    class S amendment is a new selection, not an entry of this one."""
    last = {}
    for i, am in enumerate(sel.get("amendments") or []):
        if am.get("class") not in ("N", "V"):
            continue
        for ch in am.get("changes") or []:
            last[ch.get("key")] = (i, ch.get("after"))
    explained, unexplained = {}, []
    for key in changed:
        hit = last.get(key)
        v = frozen_value(now, key)
        if hit is not None and v is not None and hit[1] == v:
            explained[key] = hit[0]
        else:
            unexplained.append(key)
    return explained, unexplained


def cmd_check(a):
    """the freeze against the files now; a difference that a class N / V amendment entry of selection.json recorded
    (its value now = the entry's `after`) is reported as explained. Exit 1 iff a difference is unexplained."""
    sel = load(SEL_PATH)
    now = frozen_hashes()
    d = check_frozen(sel, now=now)
    explained, unexplained = amendment_coverage(sel, d, now)
    for k in unexplained:
        log("UNEXPLAINED (no amendment entry records this value): %s" % k)
    print(json.dumps({"state": sel.get("state"), "changed": d, "amendments": len(sel.get("amendments") or []),
                      "explained_by_amendment": explained, "unexplained": unexplained}, indent=1))
    return 1 if unexplained else 0


def cmd_amend(a):
    """append one class N or V amendment entry to selection.json after the set is accepted (the rule's
    amendments.entry_format, plus what this tool can check later): every frozen key whose value now differs from the
    freeze or from the last entry that recorded it, with its frozen / before / after values; `--detail` (a JSON file
    prepared by the maintainer: the changed files, the evidence) is kept verbatim. The main session classifies
    (`--authorised-by` names its ruling); the tool refuses class S (a new selection), an unaccepted set and an entry
    that records nothing."""
    if a.klass not in ("N", "V"):
        raise Stop("class %s is not an entry of this selection (class S: select again with the next generation)" % a.klass)
    sel = load(SEL_PATH)
    if sel.get("state") != "accepted":
        raise Stop("selection.json is in state %s: amendments follow the acceptance" % sel.get("state"))
    now = frozen_hashes()
    changed = check_frozen(sel, quiet=True, now=now)
    fr = sel["frozen"]["hashes"]
    prev = {}
    for am in sel.get("amendments") or []:
        for ch in am.get("changes") or []:
            prev[ch.get("key")] = ch.get("after")
    changes = []
    for key in changed:
        if key.startswith("image "):
            raise Stop("the verification image changed (%s): not an amendment this tool records" % key)
        grp, _, k = key.partition(" ")
        frozen = fr["kit_members"] if grp == "kit_members" else fr[grp].get(k)
        before = prev.get(key, frozen)
        after = frozen_value(now, key)
        if after != before:
            changes.append({"key": key, "frozen": frozen, "before": before, "after": after})
    if not changes:
        raise Stop("nothing to record: every frozen value equals the freeze or the last amendment entry")
    detail = load(a.detail) if a.detail else None
    gen = (sel.get("reveal") or {}).get("generation")
    entry = {"utc": utc_now(), "class": a.klass, "what": a.what, "why": a.why, "authorised_by": a.authorised_by,
             "superseded_selection_sha256": sha256_file(SEL_PATH), "generation_opened": int(gen) if gen is not None else 0,
             "rule_version": int(sel["rule"]["rule_version"]), "changes": changes,
             "recorded_notes": {k: v for k, v in now["recorded"].items() if fr["recorded"].get(k) != v},
             "hashes_after": now, "detail": detail,
             "tool": {"path": os.path.relpath(os.path.abspath(__file__), TASK), "sha256": sha256_file(os.path.abspath(__file__))}}
    mode = os.stat(SEL_PATH).st_mode & 0o777
    sel.setdefault("amendments", []).append(entry)
    save(SEL_PATH, sel)
    os.chmod(SEL_PATH, mode)
    log("amendment %d (class %s) recorded: %d frozen value(s): %s" % (len(sel["amendments"]), a.klass, len(changes),
                                                                     ", ".join(c["key"] for c in changes)))
    return 0


# ------------------------------------------------------------------------------------------------ the trim stage (P11)
_STUDY = {}


def study():
    if "S" not in _STUDY:
        if V12 not in sys.path:
            sys.path.insert(0, V12)
        import readings as R
        _STUDY["R"] = R
        _STUDY["S"] = R.Study(rule_path=RULE_PATH)
    return _STUDY["S"]


def stratum_spec(rl, sid):
    return [s for s in rl["strata"] if s["id"] == sid][0]


def eval_trim(S, key_w, st, dev_spec, c):
    """candidate c of stratum st on stream W through (b1) -> (c) -> (b2) (readings.Study.evaluate_candidate, the
    rule's order); the six integers, the outcome, the failed conditions; for a qualified one the details"""
    import numpy as np
    cand = trim_candidate(key_w, st, dev_spec, c)
    sid = st["id"]
    dev = (cand["dm_kg"], cand["dx_mm"], cand["dy_mm"])
    fac = (lambda: np.random.Generator(np.random.PCG64(noise_seed(key_w, sid, c))))
    r = S.evaluate_candidate(cand["T_C"], cand["SP"], dev, st, rng_factory=fac)
    rec = {"index": c, "T_C_tenths": cand["T_C_tenths"], "SP_tenths": cand["SP_tenths"], "dm_g": cand["dm_g"],
           "dx_centimm": cand["dx_centimm"], "dy_centimm": cand["dy_centimm"], "unit_serial": cand["unit_serial"],
           "qualified": not r["fail"], "failed": list(r["fail"])}
    if not r["fail"]:
        rec["detail"] = {"rho_kg_m3": r["rho"], "reference": r["reference"], "b2": r["b2"], "robust_noweigh": r["robust_noweigh"],
                         "lenient": r["lenient"], "strict": r["strict"], "layouts": r["layouts"],
                         "loads_keel_up_N": r["loads_keel_up_N"], "net_without_ballast_N": r["net_without_ballast_N"],
                         "layouts_full": r["layouts_full"], "layouts_dB_ok": r["layouts_dB_ok"],
                         "fail_if_both_rules_everywhere": r["fail_if_both_rules_everywhere"]}
    return rec


def _trim_worker(args):
    key_hex, sid, start, need, max_draws = args
    t0 = time.process_time()
    S = study()
    rl = S.rule
    st = stratum_spec(rl, sid)
    key = bytes.fromhex(key_hex)
    recs, q, c = [], 0, start
    while c < max_draws and q < need:
        rec = eval_trim(S, key, st, rl["per_unit_deviation"], c)
        recs.append(rec)
        q += 1 if rec["qualified"] else 0
        c += 1
    return sid, recs, time.process_time() - t0


def trim_extend(stage, key_hex, sids, need, procs, max_draws):
    """evaluate, per stratum in sids, the next candidates (in index order, from where the stage stopped) until `need`
    more are qualified or max_draws is reached; `stage` = {sid: {records, next_index, cpu_s}}"""
    jobs = [(key_hex, s, stage[s]["next_index"], need, max_draws) for s in sids if stage[s]["next_index"] < max_draws]
    if not jobs:
        return
    if procs > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(procs, len(jobs))) as p:
            res = p.map(_trim_worker, jobs, chunksize=1)
    else:
        res = [_trim_worker(j) for j in jobs]
    for sid, recs, cpu in res:
        st = stage[sid]
        if recs and recs[0]["index"] != st["next_index"]:
            raise RuntimeError("trim stage out of order")
        st["records"] += recs
        st["next_index"] += len(recs)
        st["cpu_s"] = round(st.get("cpu_s", 0.0) + cpu, 2)


def new_trim_stage():
    return {s: {"records": [], "next_index": 0, "consumed": [], "cpu_s": 0.0} for s in STRATA}


def trim_counts(st):
    rs = st["records"]
    hist = {}
    for r in rs:
        for f in r["failed"]:
            hist[f] = hist.get(f, 0) + 1
    passed_b1 = sum(1 for r in rs if r["failed"] != ["b1"])
    passed_c = sum(1 for r in rs if r["qualified"] or r["failed"] == ["b2"])
    return {"drawn": len(rs), "passed_b1": passed_b1, "passed_c": passed_c, "qualified": sum(1 for r in rs if r["qualified"]),
            "qualified_indices": [r["index"] for r in rs if r["qualified"]], "failed_histogram": hist}


# ------------------------------------------------------------------------------------------------ (d): the tray arrangement
_CELL = {}


def cell_mods():
    if not _CELL:
        if TESTS not in sys.path:
            sys.path.insert(0, TESTS)
        from arm.assembly_rov import fixtures as FX
        from arm.assembly_rov import layout as LY
        from arm.assembly_rov import server as SV
        cfg = SV.merge_config(load(os.path.join(TESTS, "grader_config.json")))
        fx = FX.Fixtures(FX.load_cell_layout(), FX.load_facts(), lmag=bool(cfg["layout"].get("lmag", False)))
        _CELL.update(FX=FX, LY=LY, SV=SV, cfg=cfg, fx=fx)
    return _CELL


def kit_slots(kit_json):
    return {s["id"]: s for s in kit_json["slots"]}


def condition_d(products, required):
    """(d) fails when the thruster in pocket k has the handedness slot k of the mate order needs, for every k"""
    if len(products) != len(required):
        raise ValueError("4 pockets and 4 slots expected")
    in_order_right = sum(1 for p, r in zip(products, required) if p == r)
    return {"products_by_pocket": list(products), "required_by_slot_order": list(required),
            "in_order_correct": in_order_right, "pass": in_order_right != len(required)}


def layout_record(seed, dome, rl, kit_json):
    """the layout of (seed, dome end): its digest and condition (d)"""
    C = cell_mods()
    LY = C["LY"]
    lay = LY.episode_layout(int(seed), fx=C["fx"], cfg=C["cfg"]["layout"], stratum="dome%d" % int(dome), nominal=False)
    dig = LY.layout_digest(lay)
    sl = kit_slots(kit_json)
    order = rl["conditions"]["d"]["slot_order"]
    req = [sl[s]["design_part"] for s in order]
    d = condition_d(LY.nest_products(lay), req)
    d["nests"] = sorted(lay["nests"], key=lambda n: n["nest"])
    return {"seed": int(seed), "dome": int(dome), "layout_digest": dig, "d": d}


# ------------------------------------------------------------------------------------------------ (e): the views
def view_jobs(jobs, kit_dir, tag, procs=1):
    """condition (e) for [(id, seed, dome, digest)] in child processes (the renderer's state stays out of this one;
    `procs` of them at once, the jobs dealt round robin; the result does not depend on how they are dealt)"""
    d = os.path.join(WORK, "views")
    os.makedirs(d, exist_ok=True)
    k = max(1, min(int(procs), len(jobs)))
    chunks = [jobs[i::k] for i in range(k)]
    ps = []
    t0 = time.time()
    for i, ch in enumerate(chunks):
        jp = os.path.join(d, "jobs_%s_%d.json" % (tag, i))
        op = os.path.join(d, "views_%s_%d.json" % (tag, i))
        save(jp, {"kit": kit_dir, "robot": ROBOT_TREE, "jobs": [{"id": a, "seed": s, "dome": dm, "digest": dg} for a, s, dm, dg in ch]})
        if os.path.isfile(op):
            os.remove(op)
        lf = open(os.path.join(d, "views_%s_%d.log" % (tag, i)), "a")
        cmd = [sys.executable, "-I", "-B", os.path.abspath(__file__), "view", "--jobs", jp, "--out", op]
        ps.append((subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT), lf, op))
    res = None
    for p, lf, op in ps:
        rc = p.wait()
        lf.close()
        if rc != 0 or not os.path.isfile(op):
            raise Stop("the view check %s failed (exit %d; log %s)" % (tag, rc, lf.name))
        r = load(op)
        if res is None:
            res = r
        else:
            if r["rule"] != res["rule"] or r["coverage_check_sha256"] != res["coverage_check_sha256"]:
                raise Stop("the view processes of %s disagree on the rule or the tool" % tag)
            res["jobs"].update(r["jobs"])
    res["jobs"] = {a: res["jobs"][a] for a, _s, _dm, _dg in jobs}
    res["wall_s"] = round(time.time() - t0, 1)
    res["processes"] = k
    return res


def cmd_view(a):
    """internal: condition (e) per job, written to --out"""
    if os.environ.get("MUJOCO_GL") not in ("osmesa", "egl"):
        raise SystemExit("set MUJOCO_GL=osmesa (and the OSMesa library path) for the view check")
    need_pinned()
    for k in ("DISPLAY", "WAYLAND_DISPLAY"):
        os.environ.pop(k, None)
    J = load(a.jobs)
    rl = rule()
    e = rl["conditions"]["e"]
    thr = float(e["e1_top_camera"]["min_visible_share"])
    n_units = int(e["e1_top_camera"]["n_units"])
    n_samples = int(e["e1_top_camera"]["samples_per_unit"])
    px_min = float(e["e2_wrist_camera"]["min_propeller_diameter_px"])
    standoff = float(e["e2_wrist_camera"]["standoff_m"])
    if TESTS not in sys.path:
        sys.path.insert(0, TESTS)
    sys.path.insert(0, TOOLS)
    import numpy as np
    from arm.assembly_rov import render_twin as RT
    from arm.assembly_rov import server as SV
    from arm.assembly_rov import layout as LY
    from render.spec import CameraSpec
    import coverage_check as CC
    cfg = SV.merge_config(load(os.path.join(TESTS, "grader_config.json")))
    kit = RT.load_kit(J["kit"])
    out = {"format": "rov-asm-view-check/1", "rule": {"e1_min_visible_share": thr, "n_units": n_units, "samples_per_unit": n_samples,
                                                       "e2_min_propeller_diameter_px": px_min, "standoff_m": standoff},
           "renderer": "mujoco (coverage_check.py coverage's default)", "coverage_check_sha256": sha256_file(os.path.join(TOOLS, "coverage_check.py")),
           "jobs": {}}
    us = None
    srv = None
    for j in J["jobs"]:
        t0, c0 = time.time(), time.process_time()
        host = RT.ViewHost(J["kit"], J["robot"], int(j["seed"]), stratum="dome%d" % int(j["dome"]), kit=kit, renderer="mujoco",
                           config={"layout": cfg["layout"], "info_layout": cfg["info_layout"]})
        dig = LY.layout_digest(host.layout)
        if dig != j["digest"]:
            raise SystemExit("job %s: the view host's layout digest %s is not the cell's %s" % (j["id"], dig, j["digest"]))
        # the server's own draws of this episode: the same layout and cameras as the view host
        if srv is None:
            srv = SV.RovInfoServer(J["kit"], J["robot"], config=cfg, mode="hidden", privileged=False, render=False,
                                   log=lambda *x: None, kit=kit, fx=host.fx)
        rr = srv.reset({"seed": int(j["seed"]), "stratum": "E1", "dome_end_deg": int(j["dome"]),
                        "work_order": {"unit": "ROV-B07-0000", "delivery_site": {"water_temperature_C": 10.0, "salinity_psu": 35.0}},
                        "deviation": {"dm_kg": 0.0, "dx_mm": 0.0, "dy_mm": 0.0}})
        same = {"layout_digest": rr["layout_digest"] == dig,
                "top": json.dumps(srv.episode["cameras"]["top"], sort_keys=True, default=_json_default) ==
                json.dumps(host.episode["cameras"]["top"], sort_keys=True, default=_json_default),
                "wrist_intrinsics": json.dumps(srv.episode["cameras"]["wrist"]["intrinsics"], sort_keys=True, default=_json_default) ==
                json.dumps(host.episode["cameras"]["wrist"]["intrinsics"], sort_keys=True, default=_json_default)}
        if not all(same.values()):
            raise SystemExit("job %s: the view host and the cell server draw different episodes: %r" % (j["id"], same))
        if us is None:
            us = CC.UnitSamples(J["kit"], J["robot"], kit, host.fx, n=n_samples, log=lambda *x: None)
        _r, _cam, units = CC.top_coverage(host, us)
        e1 = {}
        for u in sorted(units):
            v = units[u]
            vis = float(v["in_frame"]) * float(v["footprint"])
            e1[u] = {"in_frame": v["in_frame"], "footprint": v["footprint"], "visible": vis, "pass": vis >= thr,
                     "seen_diagnostic": v["seen"], "footprint_px": v["footprint_px"]}
        e1_pass = len(e1) == n_units and all(v["pass"] for v in e1.values())
        cw = host.camera_spec("wrist")
        props = []
        for nest in sorted(host.layout["nests"], key=lambda n_: n_["nest"]):
            u = nest["unit"]
            T = RT._body_T(host.model, host.data, u)
            r_tip, (pz0, pz1), z_rim = CC.prop_geometry(host, u)
            axis = T[:3, 2]
            if float(axis[2]) < 0.9:
                raise SystemExit("job %s: thruster %s does not stand nose down in its pocket (axis %r)" % (j["id"], u, axis))
            rim_c = (T @ np.array([0.0, 0.0, z_rim, 1.0]))[:3]
            W = CC.look_pose(rim_c, -axis, standoff)
            cam = CameraSpec(name="wrist", width=cw.width, height=cw.height, fx=cw.fx, fy=cw.fy, cx=cw.cx, cy=cw.cy,
                             world_T_cam=W.tolist(), k1=cw.k1, k2=cw.k2, p1=cw.p1, p2=cw.p2)
            mn1, mx1, _uv1 = CC.projected_circle_axes(cam, T, r_tip, pz1)
            mn0, mx0, _uv0 = CC.projected_circle_axes(cam, T, r_tip, pz0)
            dia = min(mx1, mx0)
            props.append({"nest": nest["nest"], "unit": u, "product": nest["product"], "tip_r_mm": round(r_tip * 1e3, 3),
                          "diam_px": round(dia, 3), "diam_px_minor": round(min(mn1, mn0), 3), "fx": round(cw.fx, 3),
                          "pass": dia >= px_min})
        e2_pass = len(props) == 4 and all(p["pass"] for p in props)
        host.close()
        out["jobs"][j["id"]] = {"seed": int(j["seed"]), "dome": int(j["dome"]), "layout_digest": dig, "server_draws_same": same,
                                "e1": e1, "e1_pass": e1_pass,
                                "e1_least_visible": min(([v["visible"], u] for u, v in e1.items()), default=None),
                                "e2": props, "e2_pass": e2_pass,
                                "e2_least_diam_px": min((p["diam_px"] for p in props), default=None),
                                "pass": bool(e1_pass and e2_pass),
                                "cpu_s": round(time.process_time() - c0, 2), "wall_s": round(time.time() - t0, 2)}
        log("view %s seed %d dome %d: e1 %s (least %.4f %s), e2 %s (least %.1f px)" % (
            j["id"], int(j["seed"]), int(j["dome"]), e1_pass, out["jobs"][j["id"]]["e1_least_visible"][0],
            out["jobs"][j["id"]]["e1_least_visible"][1], e2_pass, out["jobs"][j["id"]]["e2_least_diam_px"]))
    save(a.out, out)
    sys.stdout.flush()
    os._exit(0)          # skip the renderer's teardown at interpreter exit


# ------------------------------------------------------------------------------------------------ (a): the image runs
def unit_name(serial):
    return "ROV-B07-%04d" % int(serial)


def episode_entry(sid, trim_rec, lay_rec, extra=None):
    """one episode of a set (rov-asm-set/1) from a trim candidate and a layout candidate"""
    e = {"seed": int(lay_rec["seed"]), "stratum": sid, "dome_end_deg": int(lay_rec["dome"]),
         "work_order": {"unit": unit_name(trim_rec["unit_serial"]),
                        "delivery_site": {"water_temperature_C": trim_rec["T_C_tenths"] / 10.0, "salinity_psu": trim_rec["SP_tenths"] / 10.0}},
         "deviation": {"dm_kg": trim_rec["dm_g"] / 1000.0, "dx_mm": trim_rec["dx_centimm"] / 100.0, "dy_mm": trim_rec["dy_centimm"] / 100.0},
         "layout_digest": lay_rec["layout_digest"]}
    if extra:
        e.update(extra)
    return e


def given_hashes(root):
    out = {}
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            if rel.startswith("given/robot/universal_robots_ur10e/"):
                continue        # the image's task assets carry their own copy of the arm (test.sh copies them over this one)
            out[rel] = sha256_file(p)
    return out


def build_stage(dst, episodes, keep_set=False):
    """a tests/ folder for one verify run in the image: the task's tests/ (less this selection's own outputs), the
    round's set as hidden/episodes.json (grader_config.json's episode count set to the set's size when it is not 4;
    keep_set: tests/hidden/episodes.json itself, byte for byte, the acceptance runs),
    and -- until the givens are assembled (P24) -- the givens stand-in of P15 (given/robot = the robot models, given/lib =
    the grader's arm / common / render packages) with its given_hashes.json"""
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(TESTS, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "selection.json", "superseded"))
    hp = os.path.join(dst, "hidden", "episodes.json")
    meta = {"grader_config_changed": False}
    if keep_set:
        shutil.copy2(SET_PATH, hp)
        if sha256_file(hp) != sha256_file(SET_PATH):
            raise Stop("the stage's episodes.json is not the written set")
        meta["set"] = {"episodes.json": sha256_file(hp), "from": SET_PATH}
        episodes = None
    elif os.path.isfile(hp):
        os.remove(hp)
    if episodes is not None:
        save(hp, {"format": SET_FORMAT, "episodes": episodes})
        if len(episodes) != 4:
            cp = os.path.join(dst, "grader_config.json")
            cfg = load(cp)
            cfg["episodes"] = len(episodes)
            with open(cp, "w") as f:
                json.dump(cfg, f, indent=1, sort_keys=True)
            meta["grader_config_changed"] = {"episodes": len(episodes)}
    if not os.path.isdir(os.path.join(TESTS, "setup_payload")):
        g = os.path.join(dst, "setup_payload", "given")
        os.makedirs(os.path.join(g, "lib"))
        shutil.copytree(ROBOT_TREE, os.path.join(g, "robot"))
        for pkg in ("arm", "common", "render"):
            shutil.copytree(os.path.join(TESTS, pkg), os.path.join(g, "lib", pkg), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        gh = given_hashes(os.path.join(dst, "setup_payload"))
        with open(os.path.join(dst, "given_hashes.json"), "w") as f:
            json.dump(gh, f, indent=1, sort_keys=True)
        meta["givens_standin"] = {"files_hashed": len(gh), "robot": os.path.relpath(ROBOT_TREE, DEV)}
    else:
        meta["givens"] = {"from": "tests/setup_payload + tests/given_hashes.json (assembled by P24 / P24b)",
                          "given_hashes_sha256": sha256_file(os.path.join(dst, "given_hashes.json")),
                          "given_hashes_entries": len(load(os.path.join(dst, "given_hashes.json")))}
    return meta


SOLUTION_SCRIPT = ("runuser -u user -- bash /opt/sol/solve.sh && chown -R root:root /home/user/submission && mkdir -p /opt/ale/stage && "
                   "cp -r /opt/ale/tests/. /opt/ale/stage/ && cd /opt/ale/stage && (time bash test.sh) > /out/verify.log 2>&1; rc=$?; "
                   "chown -R $HOST_ID /out; echo rc=$rc; exit $rc")


def play_in_image(image, tag, episodes, keep_set=False):
    """the set `episodes` played once by the oracle (solution/solve.sh) through tests/test.sh -> verify.py in the image;
    a run that ends without a verdict (the infrastructure) is run again. -> (result dict, out dir)"""
    base = os.path.join(WORK, "runs", tag)
    stage = os.path.join(base, "tests")
    meta = build_stage(stage, episodes, keep_set=keep_set)
    attempts = []
    for k in range(1, RUN_ATTEMPTS + 1):
        out = os.path.join(base, "out%d" % k)
        shutil.rmtree(out, ignore_errors=True)
        log("image run %s attempt %d: %d episode(s) in %s (load %s)" % (tag, k, len(episodes), image, " ".join(open("/proc/loadavg").read().split()[:3])))
        u0 = utc_now()
        rc, wall, to = run_in_image(image, stage, out, SOLUTION_SCRIPT, mounts=["%s:/opt/sol:ro" % os.path.join(TASK, "solution")],
                                    envs=["ROBOTICS_GRADER_VERDICT_PATH=/out/verdict.json", "ARB_RECORD_ROOT=/out/record",
                                          "HOST_ID=%d:%d" % (os.getuid(), os.getgid())],
                                    log_path=os.path.join(base, "docker_%d.log" % k))
        vp = os.path.join(out, "verdict.json")
        ep = os.path.join(out, "record", "out", "episodes.json")
        a = {"attempt": k, "utc_start": u0, "utc_end": utc_now(), "exit": rc, "wall_s": round(wall, 1), "timed_out": to,
             "verdict": os.path.isfile(vp), "episode_summaries": os.path.isfile(ep), "loadavg_end": open("/proc/loadavg").read().split()[:3]}
        attempts.append(a)
        if a["verdict"] and a["episode_summaries"]:
            break
        log("image run %s attempt %d ended without a verdict (exit %d, %.0f s, timed out %s): run again" % (tag, k, rc, wall, to))
    else:
        raise Stop("image run %s: no verdict after %d attempts (infrastructure?)" % (tag, RUN_ATTEMPTS))
    verdict = load(vp)
    summaries = load(ep)
    # the evidence: verdict, verify log, the per-episode summaries (the replay traces stay in the scratch)
    ev = os.path.join(EVIDENCE, "runs", tag)
    os.makedirs(ev, exist_ok=True)
    for src, nm in ((vp, "verdict.json"), (ep, "episodes_summaries.json"), (os.path.join(out, "verify.log"), "verify.log")):
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(ev, nm))
    save(os.path.join(ev, "set_played.json"), {"format": SET_FORMAT, "episodes": episodes, "stage": meta})
    return {"tag": tag, "image": image, "attempts": attempts, "stage": meta, "verdict_rewards": verdict.get("rewards"),
            "evidence": os.path.relpath(ev, DEV)}, verdict, summaries


def judge_a(entry, s, rl):
    """condition (a) on one episode's finalize summary (verify's record/out/episodes.json)"""
    need = rl["conditions"]["a"]["pass_requires"]
    ref = s.get("reference") or {}
    mates = [m for m in s.get("mates", []) if m.get("scored", True)]
    counted = sum(1 for m in mates if m.get("achieved"))
    watch = s.get("watch") or {}
    f = watch.get("max_part_force_sustained_N")
    uc = ref.get("unplanned_contacts")
    fields = {"layout_digest": s.get("layout_digest"), "layout_digest_checked": s.get("layout_digest_checked"),
              "reference_ok": ref.get("ok"), "reference_calls": s.get("reference_calls"), "done_reason": s.get("done_reason"),
              "end_reason": s.get("end_reason"), "mates_counted": counted, "mates_total": len(mates), "n_achieved": s.get("n_achieved"),
              "trim": s.get("trim"), "stage_score": s.get("stage_score"),
              "unplanned_contacts": len(uc) if isinstance(uc, list) else uc, "hard_ok": s.get("hard_ok"),
              "violations": [v.get("what") for v in s.get("violations", [])],
              "max_part_force_sustained_N": f, "max_part_force_sustained_pair": watch.get("max_part_force_sustained_pair"),
              "scrapped": sorted((s.get("scrapped") or {}).keys()) if isinstance(s.get("scrapped"), dict) else s.get("scrapped"),
              "sim_time_s": s.get("sim_time_s"), "episode_wall_s": s.get("episode_wall_s"), "turns_used": s.get("turns_used"),
              "camera_frames_total": s.get("camera_frames_total"), "trim_eval": s.get("trim_eval"), "ballast_holes": s.get("ballast_holes"),
              "max_robot_link_force_N": watch.get("max_robot_link_force_N"), "max_dome_force_sustained_N": watch.get("max_dome_force_sustained_N"),
              "max_flange_force_sustained_N": watch.get("max_flange_force_sustained_N"),
              "weld_F_sustained_max_N": ((s.get("cell_metrics") or {}).get("weld") or {}).get("weld_F_sustained_max_N"),
              "seed": s.get("seed"), "stratum": s.get("stratum"), "dome_end_deg": s.get("dome_end_deg"),
              "work_order": s.get("work_order"), "deviation": s.get("deviation")}
    clauses = {
        "layout_digest_equals_recorded": s.get("layout_digest") == entry["layout_digest"] and bool(s.get("layout_digest_checked")),
        "reference_program_completed": bool(ref.get("ok")) and int(s.get("reference_calls") or 0) == 1 and s.get("done_reason") == "finish",
        "structural_mates_counted": counted == int(need["structural_mates_counted"]) and len(mates) == int(need["structural_mates_total"]),
        "trim": s.get("trim") is not None and float(s["trim"]) >= float(need["trim"]) - 1e-12,
        "stage_score": s.get("stage_score") is not None and float(s["stage_score"]) >= float(need["stage_score"]) - 1e-12,
        "unplanned_contacts": isinstance(uc, list) and len(uc) == int(need["unplanned_contacts"]),
        "hard_ok": s.get("hard_ok") is True and not s.get("violations"),
        "max_sustained_part_contact_force": f is not None and float(f) < float(need["max_sustained_part_contact_force_N_strictly_below"]),
        "scrapped_units": len(s.get("scrapped") or {}) == int(need["scrapped_units"]),
    }
    same = (s.get("seed") == entry["seed"] and s.get("dome_end_deg") == entry["dome_end_deg"] and s.get("work_order") == entry["work_order"]
            and s.get("deviation") == entry["deviation"])
    clauses["episode_is_the_pair"] = bool(same)
    failed = [k for k, v in clauses.items() if not v]
    detail = {"mates_not_counted": sorted(m.get("id") for m in mates if not m.get("achieved")),
              "scrapped": fields["scrapped"], "violations": fields["violations"],
              "unplanned_contacts": [str(c)[:200] for c in uc] if isinstance(uc, list) else uc,
              "force_pair": watch.get("max_part_force_sustained_pair")}
    return {"pass": not failed, "failed": failed, "clauses": clauses, "fields": fields, "failed_detail": detail}


def judge_run(entries, verdict, summaries, rl):
    if len(summaries) != len(entries):
        raise Stop("the verify run reports %d episode(s) for a set of %d" % (len(summaries), len(entries)))
    m = verdict.get("metrics") or {}
    if float(m.get("grader_exception", 0.0)) != 0.0:
        raise Stop("verify raised a grader exception: the pipeline, not the pair (see the verify log)")
    out = []
    for k, (e, s) in enumerate(zip(entries, summaries), 1):
        j = judge_a(e, s, rl)
        j["verdict_metrics"] = {x: m.get("ep%d_%s" % (k, x)) for x in ("stage", "hard_ok", "trim", "mates", "scrapped", "net_buoyancy_N",
                                                                      "pitch_deg", "roll_deg", "ballast_fastened")}
        out.append(j)
    return out


# ------------------------------------------------------------------------------------------------ the rounds (S6, S7)
class Selector(object):
    """rounds of S6 / S7 over injected checks (the real ones or the self-test's):
      next_trim(sid) -> the stratum's next qualified trim record not yet consumed (evaluated lazily), None = FB-W
      next_layout(sid, dome) -> (the next layout candidate id not yet consumed that passes (d) and (e) for `dome`, or
                                 None = FB-L; [the candidates dropped on the way, consumed by next_layout itself])
      consume(sid, trim_index, layout_id) -> both members of a pair consumed, when the pair enters its round
      play(round, pairs) -> [{'pass': bool, ...}] per pair (one run per pair)
    state = the selection's 'rounds' list, saved after every pair entered and every round played; an interrupted
    selection resumes where it stopped (a round being formed is formed on, a formed round is played, a played round
    is not played again)"""

    def __init__(self, dome, next_trim, next_layout, consume, play, state, save_fn=None):
        self.dome, self.next_trim, self.next_layout, self.consume, self.play = dome, next_trim, next_layout, consume, play
        self.state = state
        self.save = save_fn or (lambda: None)

    def episodes(self):
        got = {}
        for r in self.state:
            for p in r["pairs"]:
                if p.get("result") and p["result"]["pass"]:
                    got[p["stratum"]] = p
        return got

    def run(self):
        while True:
            got = self.episodes()
            todo = [s for s in STRATA if s not in got]
            if not todo:
                return {"outcome": "selected", "episodes": got}
            if self.state and not self.state[-1].get("played"):
                rnd = self.state[-1]                         # resumed: a round being formed or formed and not played
                if rnd.get("stopped"):
                    return {"outcome": rnd["stopped"]["fallback"], "stratum": rnd["stopped"]["stratum"], "episodes": got}
            else:
                rnd = {"round": len(self.state) + 1, "pairs": [], "formed": False, "played": False}
                self.state.append(rnd)
                self.save()
            if not rnd.get("formed"):
                have = set(p["stratum"] for p in rnd["pairs"])
                for sid in todo:
                    if sid in have:
                        continue
                    t = self.next_trim(sid)
                    if t is None:
                        rnd["stopped"] = {"fallback": "FB-W", "stratum": sid}
                        self.save()
                        return {"outcome": "FB-W", "stratum": sid, "episodes": got}
                    lid, drops = self.next_layout(sid, self.dome[sid])
                    rnd.setdefault("layout_drops", []).extend(drops)
                    if lid is None:
                        rnd["stopped"] = {"fallback": "FB-L", "stratum": sid, "trim_candidate": t["index"]}
                        rnd["unplayed_pairs"] = rnd["pairs"]
                        rnd["pairs"] = []
                        self.save()
                        return {"outcome": "FB-L", "stratum": sid, "episodes": got}
                    self.consume(sid, t["index"], lid)
                    rnd["pairs"].append({"stratum": sid, "trim_index": t["index"], "layout_id": lid, "dome": self.dome[sid]})
                    self.save()
                rnd["pairs"].sort(key=lambda p_: STRATA.index(p_["stratum"]))
                rnd["formed"] = True
                self.save()
            res = self.play(rnd["round"], rnd["pairs"])
            for p, r in zip(rnd["pairs"], res):
                p["result"] = r
            rnd["played"] = True
            self.save()


# ------------------------------------------------------------------------------------------------ S2: self-tests
def kat(rl):
    """the known-answer test of the stream code on the rule's public test vectors"""
    t = rl["streams"]["known_answer_test"]
    kw, bl = bytes.fromhex(t["key_W_hex"]), bytes.fromhex(t["blob_L_hex"])
    res = {}
    res["commitment_W"] = commitment(kw) == t["commitment_W"]
    res["commitment_L"] = commitment(bl) == t["commitment_L"]
    res["u64_dome"] = u64(kw, "W", "dome") == t["u64(key_W,'W','dome')"]
    res["dome_assignment"] = list(dome_assignment(kw)) == list(t["dome_assignment"])
    ok = True
    n = 0
    for sid, lst in t["trim_candidates"].items():
        st = stratum_spec(rl, sid)
        for want in lst:
            got = trim_candidate(kw, st, rl["per_unit_deviation"], want["index"])
            n += 1
            ok = ok and all(got[k] == v for k, v in want.items())
    res["trim_candidates"] = ok and n > 0
    res["trim_candidates_n"] = n
    ns = t["noise_seed"]
    res["noise_seed"] = all(str(noise_seed(kw, k.split("/")[0], int(k.split("/")[1]))) == v for k, v in ns.items())
    res["layout_seeds"] = [str(x) for x in layout_seeds(bl)] == [str(x) for x in t["layout_seeds"]]
    pl = prior_lattice(rl["per_unit_deviation"]["published_range"])
    w = t["prior_lattice"]
    res["prior_lattice"] = (len(pl) == w["points"] and list(pl[0]) == list(w["first"]) and list(pl[-1]) == list(w["last"])
                            and sha256_bytes(json.dumps(pl).encode()) == w["sha256_of_json"])
    flip = bytearray(kw)
    flip[0] ^= 1
    res["commitment_changes_with_one_bit"] = commitment(bytes(flip)) != t["commitment_W"]
    res["pass"] = all(v for k, v in res.items() if k != "trim_candidates_n")
    return res


def appendix_identity():
    doc = appendix_a_block(open(PREREG_PATH).read())
    mine = own_block()
    sm = open(os.path.join(V12, "sampler.py")).read()
    try:
        smb = appendix_a_block(sm)
    except Stop:
        smb = None
    return {"tool_equals_document": mine == doc, "document_block_sha256": sha256_bytes(doc.encode()),
            "readings_library_sampler_equals_document": smb == doc, "pass": mine == doc and smb == doc}


def reference_values(rl):
    """trim_evaluation.expected_under_digest_v3 computed from the frozen files (the standard map by the rule's
    definition: the batch's median vehicle at 20 degC fresh water, maximin)"""
    S = study()
    R = _STUDY["R"]
    E = R.E
    exp = rl["trim_evaluation"]["expected_under_digest_v3"]
    usable = [h for h in S.t.names if h not in S.t.blocked]
    pr = S.pr
    rng = rl["per_unit_deviation"]["published_range"]
    mid = [(rng[k][0] + rng[k][1]) / 2.0 for k in ("mass_kg", "cg_x_mm", "cg_y_mm")]
    std = S.choose("maximin", E.rho_teos10(20.0, 0.0), (pr.base_m + mid[0], pr.base_cg[0] + mid[1], pr.base_cg[1] + mid[2]))
    got = {"usable_holes": len(usable), "blocked_holes": sorted(S.t.blocked), "legal_layouts": len(S.sp.L), "most_weights": int(S.sp.NN.max()),
           "standard_map": sorted(std), "standard_map_in_the_table": sorted(S.t.raw["standard_pattern"]["holes"]),
           "vend5": sorted(S.maps["vend5"]), "digest_fresh": sorted(S.maps["digest_fresh"]), "digest_sea": sorted(S.maps["digest_sea"])}
    want = {"usable_holes": exp["usable_holes"], "blocked_holes": sorted(exp["blocked_holes"]), "legal_layouts": exp["legal_layouts"],
            "most_weights": exp["most_weights"], "standard_map": sorted(exp["standard_map"]), "standard_map_in_the_table": sorted(exp["standard_map"]),
            "vend5": sorted(exp["vend5"]), "digest_fresh": sorted(exp["digest_fresh"]), "digest_sea": sorted(exp["digest_sea"])}
    rows = {k: {"got": got[k], "expected": want[k], "pass": got[k] == want[k]} for k in want}
    return {"rows": rows, "pass": all(r["pass"] for r in rows.values())}


def _grader_set_loader():
    """verify.py's own loader of the set and its digest check, executed in a stdlib namespace (as the repository test)"""
    import ast
    import re
    vp = os.path.join(TESTS, "verify.py")
    tree = ast.parse(open(vp).read())
    consts = ("SET_FORMAT", "STRATA", "DOME_ENDS", "SEED_LIMIT", "WATER_T_RANGE_C", "WATER_SP_RANGE", "UNIT_RE")
    names = ("load_hidden_set", "check_layout_digest", "_finite")
    body = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in consts for t in n.targets)]
    body += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]

    class TaskError(Exception):
        pass

    def fail(reason):
        raise TaskError(reason)
    ns = {"json": json, "os": os, "re": re, "math": math, "fail": fail}
    exec(compile(ast.Module(body=body, type_ignores=[]), vp, "exec"), ns)
    ns["TaskError"] = TaskError
    return ns


def selftest(quick=False):
    """this tool's own tests (no seed, no physics, no renderer)"""
    rl = rule()
    res = {}
    res["appendix_a"] = appendix_identity()
    res["known_answer_test"] = kat(rl)
    # (d)
    sl = ["thruster_b", "thruster_a", "thruster_a", "thruster_b"]
    import itertools
    arr = sorted(set(itertools.permutations(["thruster_a", "thruster_a", "thruster_b", "thruster_b"])))
    fails = [a for a in arr if not condition_d(list(a), sl)["pass"]]
    res["d"] = {"arrangements": len(arr), "failing": [list(a) for a in fails],
                "pass": fails == [tuple(sl)] and condition_d(list(reversed(sl)), sl)["pass"] is False}
    # (a) on a synthetic summary: passes, and every clause broken alone is caught
    entry = {"seed": 5, "stratum": "E1", "dome_end_deg": 0, "work_order": {"unit": "ROV-B07-1234", "delivery_site": {"water_temperature_C": 10.0, "salinity_psu": 0.2}},
             "deviation": {"dm_kg": 0.12, "dx_mm": -1.0, "dy_mm": 2.0}, "layout_digest": "a" * 64}
    good = {"layout_digest": "a" * 64, "layout_digest_checked": True, "reference": {"ok": True, "unplanned_contacts": []}, "reference_calls": 1,
            "done_reason": "finish", "mates": [{"id": "m%d" % i, "scored": True, "achieved": True} for i in range(7)], "n_achieved": 7,
            "trim": 1.0, "stage_score": 1.0, "hard_ok": True, "violations": [], "watch": {"max_part_force_sustained_N": 26.3}, "scrapped": {},
            "seed": 5, "dome_end_deg": 0, "work_order": entry["work_order"], "deviation": entry["deviation"]}
    breaks = {"layout_digest_equals_recorded": ("layout_digest", "b" * 64), "reference_program_completed": ("reference", {"ok": False, "unplanned_contacts": []}),
              "trim": ("trim", 0.7), "stage_score": ("stage_score", 0.99), "unplanned_contacts": ("reference", {"ok": True, "unplanned_contacts": [{"x": 1}]}),
              "hard_ok": ("hard_ok", False), "max_sustained_part_contact_force": ("watch", {"max_part_force_sustained_N": 150.0}),
              "scrapped_units": ("scrapped", {"GRD1": "dropped"}), "episode_is_the_pair": ("seed", 6)}
    caught = {}
    for clause, (k, v) in breaks.items():
        s = json.loads(json.dumps(good))
        s[k] = v
        j = judge_a(entry, s, rl)
        caught[clause] = (not j["pass"]) and clause in j["failed"]
    s = json.loads(json.dumps(good))
    s["mates"][3]["achieved"] = False
    j = judge_a(entry, s, rl)
    caught["structural_mates_counted"] = (not j["pass"]) and "structural_mates_counted" in j["failed"]
    res["a"] = {"good_passes": judge_a(entry, good, rl)["pass"], "caught": caught,
                "pass": judge_a(entry, good, rl)["pass"] and all(caught.values())}
    # the rounds, on a synthetic world
    res["rounds"] = _selftest_rounds()
    # the set's format against verify.py's own loader and its digest check
    G = _grader_set_loader()
    import tempfile
    tmp = tempfile.mkdtemp(prefix="p20_selftest_", dir=WORK if os.path.isdir(WORK) else None)
    try:
        eps = []
        domes = [0, 180, 180, 0]
        for k, sid in enumerate(STRATA):
            st = stratum_spec(rl, sid)
            c = trim_candidate(bytes(range(32)), st, rl["per_unit_deviation"], 0)
            eps.append(episode_entry(sid, c, {"seed": (1 << 40) + k, "dome": domes[k], "layout_digest": "%064x" % (k + 1)}))
        save(os.path.join(tmp, "episodes.json"), {"format": SET_FORMAT, "episodes": eps})
        table = load(os.path.join(TESTS, "trim_table.json"))
        got = G["load_hidden_set"]({"episodes": 4}, tmp, table["per_unit_deviation"])
        ok_load = [g["stratum"] for g in got] == list(STRATA) and all(g["layout_digest"] == e["layout_digest"] for g, e in zip(got, eps))
        try:
            G["check_layout_digest"](1, got[0], "f" * 64)
            caught_digest = False
        except G["TaskError"]:
            caught_digest = True
        res["set_format"] = {"verify_loader_reads_it": ok_load, "digest_mismatch_is_a_task_error": caught_digest, "pass": ok_load and caught_digest}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # the slow-release precondition on synthetic evidence files
    tmp = tempfile.mkdtemp(prefix="p20_selftest_", dir=WORK if os.path.isdir(WORK) else None)
    try:
        now = sha256_file(CELL_FREEZE_PATH)

        def doc(vt=(20, 0, 0), gd=(20, 0, 0), runs=None, fmt=SLOW_RELEASE_FORMAT, drop=None):
            d = {"format": fmt, "cell_freeze_sha256": now, "cell_freeze_sha256_of_the_runs": runs if runs is not None else [now],
                 "kinds": {"vertical_thruster": {"trials": vt[0], "false_locks": vt[1], "captured_with_the_pads_on_the_unit": vt[2], "errors": 0},
                           "guard": {"trials": gd[0], "false_locks": gd[1], "captured_with_the_pads_on_the_unit": gd[2], "errors": 0}},
                 "pass": True}
            if drop:
                del d["kinds"][drop[0]][drop[1]]
            return d
        cases = {"good": (doc(), True),
                 "placements_do_not_hold_it_up": (doc(gd=(228, 0, 72)), True),
                 "thruster_19_trials": (doc(vt=(19, 0, 0)), False), "guard_one_false_lock": (doc(gd=(25, 1, 0)), False),
                 "guard_one_false_lock_besides_placements": (doc(gd=(228, 1, 72)), False),
                 "no_placement_count": (doc(drop=("guard", "captured_with_the_pads_on_the_unit")), False),
                 "no_false_lock_count": (doc(drop=("vertical_thruster", "false_locks")), False),
                 "placement_count_not_an_integer": (dict(doc(), kinds=dict(doc()["kinds"], guard={"trials": 20, "false_locks": 0,
                                                                                                  "captured_with_the_pads_on_the_unit": 0.0})), False),
                 "another_cell": (doc(runs=["0" * 64]), False), "format_1": (doc(fmt="rov-asm-slow-release/1"), False),
                 "another_format": (doc(fmt="x"), False)}
        got = {}
        for nm, (d_, want) in cases.items():
            fp = os.path.join(tmp, nm + ".json")
            save(fp, d_)
            got[nm] = read_slow_release(rl, fp)["pass"] == want
        got["no_file"] = read_slow_release(rl, os.path.join(tmp, "absent.json"))["pass"] is False
        res["slow_release_precondition"] = dict(got, **{"pass": all(got.values())})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # S0: the two values are read from section 7.7 only (values that stand elsewhere, e.g. rule_version 2's in 6.1, do not pass)
    sec_ok = "| x | `%s` |\n| y | `%s` |" % (RULE_SHA256, PREREG_SHA256)
    texts = {"in_7_7": ("# t\n### 6.1 a\nold\n%sb\n%s\n## 8 c\n" % (DECISIONS_ASM_SECTION, sec_ok), True),
             "only_in_6_1": ("# t\n### 6.1 a\n%s\n%sb\nnone yet\n" % (sec_ok, DECISIONS_ASM_SECTION), False),
             "after_7_7_ends": ("# t\n%sb\nnone yet\n### 7.8 c\n%s\n" % (DECISIONS_ASM_SECTION, sec_ok), False),
             "no_section": ("# t\n%s\n" % sec_ok, False),
             "two_sections": ("%sb\n%s\n%sc\n" % (DECISIONS_ASM_SECTION, sec_ok, DECISIONS_ASM_SECTION), False)}
    got = {}
    for nm, (txt, want) in texts.items():
        sec = decisions_section(txt, DECISIONS_ASM_SECTION)
        got[nm] = (sec is not None and RULE_SHA256 in sec and PREREG_SHA256 in sec) == want
    real = check_s0()
    got["the_real_files"] = bool(real["pass"]) and real["rule"]["rule_version"] == RULE_VERSION
    res["s0_reads_section_7_7"] = dict(got, **{"pass": all(got.values())})
    res["amendment_coverage"] = _selftest_amendment_coverage()
    res["pass"] = all(v.get("pass") for v in res.values() if isinstance(v, dict))
    return res


def _selftest_rounds():
    """S6 / S7 on a synthetic world: consumption, the first passing pair, FB-L, FB-W, resuming"""
    out = {}

    def world(trim_q, lay_ok, plays):
        cons = {s: 0 for s in STRATA}
        lay = {"next": 0}
        log_ = []

        def next_trim(sid):
            q = trim_q[sid]
            if cons[sid] >= len(q):
                return None
            return {"index": q[cons[sid]]}

        def next_layout(sid, dome):
            drops = []
            while lay["next"] < N_LAYOUT:
                i = lay["next"]
                if lay_ok(i, dome):
                    return "c%02d" % i, drops
                lay["next"] += 1
                drops.append({"layout_id": "c%02d" % i, "stratum": sid})
            return None, drops

        def consume(sid, idx, lid):
            assert trim_q[sid][cons[sid]] == idx and lid == "c%02d" % lay["next"]
            cons[sid] += 1
            lay["next"] += 1

        def play(rnd, pairs):
            log_.append((rnd, [(p["stratum"], p["trim_index"], p["layout_id"]) for p in pairs]))
            return [{"pass": plays(p)} for p in pairs]
        return next_trim, next_layout, consume, play, log_, cons, lay
    dome = {"E1": 0, "E2": 180, "E3": 180, "E4": 0}
    # 1: all pass in round 1; layouts 1 and 4 fail (e) for dome 180 only: E2 (180) drops c01, E4 (dome 0) takes c04
    nt, nl, cs, pl, lg, _c, _l = world({s: [3, 9] for s in STRATA}, lambda i, d: not (d == 180 and i in (1, 4)), lambda p: True)
    r = Selector(dome, nt, nl, cs, pl, []).run()
    out["all_pass"] = r["outcome"] == "selected" and lg == [(1, [("E1", 3, "c00"), ("E2", 3, "c02"), ("E3", 3, "c03"), ("E4", 3, "c04")])]
    # 1b: the same with layout 4 failing for dome 0 too: E4 drops it and takes c05
    nt, nl, cs, pl, lg, _c, _l = world({s: [3, 9] for s in STRATA}, lambda i, d: not ((d == 180 and i == 1) or i == 4), lambda p: True)
    r = Selector(dome, nt, nl, cs, pl, []).run()
    out["drop_on_the_way"] = r["outcome"] == "selected" and lg == [(1, [("E1", 3, "c00"), ("E2", 3, "c02"), ("E3", 3, "c03"), ("E4", 3, "c05")])]
    # 2: E2's first pair fails (a) -> round 2: E2 with its next trim candidate and the next layout
    nt, nl, cs, pl, lg, _c, _l = world({s: [3, 9] for s in STRATA}, lambda i, d: True, lambda p: not (p["stratum"] == "E2" and p["layout_id"] == "c01"))
    r = Selector(dome, nt, nl, cs, pl, []).run()
    out["fail_then_next"] = (r["outcome"] == "selected" and lg[1] == (2, [("E2", 9, "c04")]) and r["episodes"]["E2"]["layout_id"] == "c04")
    # 3: FB-L (every pair fails)
    nt, nl, cs, pl, lg, _c, _l = world({s: list(range(100)) for s in STRATA}, lambda i, d: True, lambda p: False)
    st = []
    r = Selector(dome, nt, nl, cs, pl, st).run()
    out["fb_l"] = r["outcome"] == "FB-L" and sum(len(x["pairs"]) for x in st) == N_LAYOUT
    # 4: FB-W (E3 has one qualified candidate and its pair fails)
    nt, nl, cs, pl, lg, _c, _l = world({"E1": [1], "E2": [1], "E3": [1], "E4": [1]}, lambda i, d: True, lambda p: p["stratum"] != "E3")
    r = Selector(dome, nt, nl, cs, pl, []).run()
    out["fb_w"] = r["outcome"] == "FB-W" and r["stratum"] == "E3"
    # 5: resume: a round formed but not played is played, not formed again
    nt, nl, cs, pl, lg, _c, _l = world({s: [3, 9] for s in STRATA}, lambda i, d: True, lambda p: True)
    st = [{"round": 1, "formed": True, "played": False,
           "pairs": [{"stratum": s, "trim_index": 3, "layout_id": "c%02d" % k, "dome": dome[s]} for k, s in enumerate(STRATA)]}]
    r = Selector(dome, nt, nl, cs, pl, st).run()
    out["resume"] = r["outcome"] == "selected" and len(lg) == 1 and lg[0][0] == 1
    # 6: resume in the middle of forming a round (E1's pair entered and consumed, then an interruption): E2..E4 are
    # formed on with the next candidates, and E1's members are not taken again
    nt, nl, cs, pl, lg, cons, lay = world({s: [3, 9] for s in STRATA}, lambda i, d: i != 1, lambda p: True)
    cons["E1"], lay["next"] = 1, 1
    st = [{"round": 1, "formed": False, "played": False, "pairs": [{"stratum": "E1", "trim_index": 3, "layout_id": "c00", "dome": 0}]}]
    r = Selector(dome, nt, nl, cs, pl, st).run()
    out["resume_forming"] = (r["outcome"] == "selected" and lg == [(1, [("E1", 3, "c00"), ("E2", 3, "c02"), ("E3", 3, "c03"), ("E4", 3, "c04")])])
    out["pass"] = all(out.values())
    return out


def _selftest_amendment_coverage():
    """amendment_coverage on synthetic records: what an entry explains and what it does not"""
    now = {"files": {"task:a": "a2", "task:b": "b1", "task:c": "c3"}, "trees": {"task:t": {"sha256": "t2", "files": 3}},
           "kit_members": {"sha256": "k1", "files": 1}, "recorded": {}}
    changed = ["files task:a", "files task:c", "trees task:t", "image x: y, frozen z"]
    am = lambda cls, *ch: {"class": cls, "changes": [{"key": k, "after": v} for k, v in ch]}   # noqa: E731
    cases = {
        "none": ({"amendments": []}, {}, changed),
        "all_but_the_image": ({"amendments": [am("N", ("files task:a", "a2"), ("files task:c", "c3"), ("trees task:t", {"sha256": "t2", "files": 3}))]},
                              {"files task:a": 0, "files task:c": 0, "trees task:t": 0}, ["image x: y, frozen z"]),
        "changed_again_after_the_entry": ({"amendments": [am("N", ("files task:a", "a2"), ("files task:c", "c2"))]},
                                          {"files task:a": 0}, ["files task:c", "trees task:t", "image x: y, frozen z"]),
        "the_last_entry_counts": ({"amendments": [am("N", ("files task:c", "c3")), am("V", ("files task:c", "c4"))]},
                                  {}, changed),
        "class_S_explains_nothing": ({"amendments": [am("S", ("files task:a", "a2"))]}, {}, changed),
        "a_tree_by_its_whole_value": ({"amendments": [am("N", ("trees task:t", {"sha256": "t2", "files": 4}))]}, {}, changed)}
    got = {}
    for nm, (sel, want_e, want_u) in cases.items():
        e, u = amendment_coverage(sel, changed, now)
        got[nm] = e == want_e and u == want_u
    return dict(got, **{"pass": all(got.values())})


def cmd_selftest(a):
    r = selftest()
    print(json.dumps(r, indent=1, default=_json_default))
    return 0 if r["pass"] else 1


# ------------------------------------------------------------------------------------------------ layout candidates
def layout_stage_check(lid, seed, dome, rl, kit_json, view):
    """(d) then (e) of one layout candidate for one dome end: {pass_d, pass_e, reason, ...}"""
    rec = layout_record(seed, dome, rl, kit_json)
    out = {"layout_id": lid, "seed": int(seed), "dome": int(dome), "layout_digest": rec["layout_digest"], "d": rec["d"]}
    if not rec["d"]["pass"]:
        out.update(pass_d=False, pass_e=None, reason="dropped_d: the tray holds the thrusters in the slots' order (%s)" % rec["d"]["products_by_pocket"])
        return out
    v = view[(lid, int(dome))]
    if v["layout_digest"] != rec["layout_digest"]:
        raise Stop("the view check of %s dome %d rendered another layout" % (lid, dome))
    out["e"] = v
    out["pass_d"] = True
    out["pass_e"] = bool(v["pass"])
    if not v["pass"]:
        bad1 = sorted(u for u, x in v["e1"].items() if not x["pass"])
        bad2 = sorted(p["unit"] for p in v["e2"] if not p["pass"])
        out["reason"] = "dropped_e: %s%s" % (("top camera below %.0f %% visible: %s" % (100 * v["e1"][bad1[0]]["visible"] if bad1 else 0, bad1)) if bad1 else "",
                                              (" wrist camera propeller below 80 px: %s" % bad2) if bad2 else "")
    return out


def all_views(ids_seeds, domes, rl, kit_dir, kit_json, tag, procs=1):
    jobs = []
    for lid, seed in ids_seeds:
        for dm in domes:
            rec = layout_record(seed, dm, rl, kit_json)
            jobs.append(("%s/dome%d" % (lid, dm), int(seed), int(dm), rec["layout_digest"]))
    res = view_jobs(jobs, kit_dir, tag, procs)
    out = {}
    for jid, v in res["jobs"].items():
        lid, d = jid.split("/dome")
        out[(lid, int(d))] = v
    return out, res


# ------------------------------------------------------------------------------------------------ S2: the rehearsal
def cmd_rehearse(a):
    if DRY:
        raise Stop("rehearse is never run in a dry run")
    sel = load(SEL_PATH)
    if sel.get("state") not in ("frozen", "rehearsed"):
        raise Stop("rehearse needs a frozen selection (state %s)" % sel.get("state"))
    require_frozen(sel)
    need_pinned()
    rl = rule()
    t_all, c_all = time.time(), time.process_time()
    st = {"utc_start": utc_now(), "seeds": "the PUBLIC test vectors of streams.known_answer_test (never a sealed seed)"}
    st["appendix_a"] = appendix_identity()
    st["known_answer_test"] = kat(rl)
    st["reference_values"] = reference_values(rl)
    st["selftest"] = selftest()
    log("S2 appendix A %s, known-answer test %s, reference values %s, self-tests %s" % (
        st["appendix_a"]["pass"], st["known_answer_test"]["pass"], st["reference_values"]["pass"], st["selftest"]["pass"]))
    if not (st["appendix_a"]["pass"] and st["known_answer_test"]["pass"] and st["reference_values"]["pass"] and st["selftest"]["pass"]):
        sel["self_tests"] = st
        save(SEL_PATH, sel)
        raise Stop("S2: a self-test failed (selection.json self_tests)")
    t = rl["streams"]["known_answer_test"]
    kw, bl = t["key_W_hex"], bytes.fromhex(t["blob_L_hex"])
    # (1) the trim stage on the test key until every stratum has its first qualified candidate
    ts = new_trim_stage()
    t0 = time.time()
    trim_extend(ts, kw, STRATA, 1, a.procs, int(rl["streams"]["W"]["max_draws_per_stratum"]))
    reh = {"trim_stage": {s: dict(trim_counts(ts[s]), cpu_s=ts[s]["cpu_s"]) for s in STRATA}, "trim_wall_s": round(time.time() - t0, 1)}
    first = {s: next((r for r in ts[s]["records"] if r["qualified"]), None) for s in STRATA}
    reh["first_qualified"] = {s: (dict({k: v for k, v in r.items() if k != "detail"}, reference=r["detail"]["reference"],
                                       b2_full=r["detail"]["b2"]["full"]) if r else None) for s, r in first.items()}
    log("S2 rehearsal trim stage: first qualified %s (wall %.0f s)" % ({s: (r["index"] if r else None) for s, r in first.items()}, reh["trim_wall_s"]))
    save(os.path.join(EVIDENCE, "rehearsal_trim_stage.json"), {"key": "public test key", "strata": ts})
    # (2) (d) and (e) for both dome ends on the 12 test layout seeds
    kit_dir = ensure_kit()
    kit_json = load(os.path.join(kit_dir, "kit.json"))
    seeds = layout_seeds(bl)
    ids = [("t%02d" % i, s) for i, s in enumerate(seeds)]
    views, vres = all_views(ids, (0, 180), rl, kit_dir, kit_json, "rehearsal", a.procs)
    save(os.path.join(EVIDENCE, "rehearsal_views.json"), vres)
    rows = []
    for lid, seed in ids:
        row = {"layout_id": lid, "seed": int(seed)}
        for dm in (0, 180):
            row["dome%d" % dm] = layout_stage_check(lid, seed, dm, rl, kit_json, views)
        row["pass_both"] = all(row["dome%d" % dm]["pass_d"] and row["dome%d" % dm]["pass_e"] for dm in (0, 180))
        rows.append(row)
    reh["layouts"] = [{"layout_id": r["layout_id"], "seed": r["seed"], "pass_both": r["pass_both"],
                       "dome0": {k: r["dome0"].get(k) for k in ("pass_d", "pass_e", "reason", "layout_digest")},
                       "dome180": {k: r["dome180"].get(k) for k in ("pass_d", "pass_e", "reason", "layout_digest")},
                       "in_order_correct": r["dome0"]["d"]["in_order_correct"],
                       "least_visible": {dm: (r["dome%d" % dm].get("e") or {}).get("e1_least_visible") for dm in (0, 180)},
                       "least_diam_px": {dm: (r["dome%d" % dm].get("e") or {}).get("e2_least_diam_px") for dm in (0, 180)}} for r in rows]
    n_both = sum(1 for r in rows if r["pass_both"])
    reh["layouts_passing_both"] = n_both
    log("S2 rehearsal layouts: %d / 12 pass (d) and (e) for both dome ends" % n_both)
    # (3) the pipeline: E1's first qualified test candidate on the first test layout passing (d), (e) for E1's dome end
    dome_t = dict(zip(STRATA, dome_assignment(bytes.fromhex(kw))))
    pair = None
    for r in rows:
        x = r["dome%d" % dome_t["E1"]]
        if x["pass_d"] and x["pass_e"]:
            pair = r
            break
    if first["E1"] is None or pair is None:
        reh["pipeline"] = {"run": False, "why": "no E1 test candidate or no test layout passing (d), (e)"}
    else:
        x = pair["dome%d" % dome_t["E1"]]
        entry = episode_entry("E1", first["E1"], {"seed": pair["seed"], "dome": dome_t["E1"], "layout_digest": x["layout_digest"]},
                              {"provenance": {"rehearsal": True, "trim_candidate": first["E1"]["index"], "layout_candidate": pair["layout_id"]}})
        run, verdict, summ = play_in_image(sel["frozen"]["image"]["tag"], "rehearsal", [entry])
        j = judge_run([entry], verdict, summ, rl)[0]
        reached = summ[0].get("done_reason") is not None and summ[0].get("reference") is not None
        reh["pipeline"] = {"run": True, "pair": {"stratum": "E1", "trim_candidate": first["E1"]["index"], "layout_candidate": pair["layout_id"],
                                                 "dome": dome_t["E1"]}, "entry": entry, "image_run": run, "reached_finalize": bool(reached),
                           "a": j, "reward": (verdict.get("rewards") or {}).get("reward")}
        log("S2 rehearsal pipeline: reached finalize %s, (a) %s %s" % (reached, j["pass"], j["failed"]))
    go_t = all(first[s] is not None for s in STRATA)
    go_l = n_both >= int(rl["procedure"]["rehearsal_before_opening"]["min_test_layouts_passing_d_and_e"])
    go_p = bool(reh["pipeline"].get("reached_finalize"))
    reh["go"] = {"every_stratum_qualified": go_t, "test_layouts_passing_both": n_both, "layouts_line": go_l, "pipeline_reached_finalize": go_p,
                 "pipeline_pair_passes_a": bool(reh["pipeline"].get("a", {}).get("pass")), "go": bool(go_t and go_l and go_p and reh["pipeline"].get("a", {}).get("pass"))}
    st["rehearsal"] = reh
    st["utc_end"] = utc_now()
    st["wall_s"] = round(time.time() - t_all, 1)
    st["cpu_s_this_process"] = round(time.process_time() - c_all, 1)
    sel["self_tests"] = st
    require_frozen(sel)
    sel["state"] = "rehearsed" if reh["go"]["go"] else "stopped_rehearsal"
    sel["states"].append({"state": sel["state"], "utc": st["utc_end"]})
    save(SEL_PATH, sel)
    log("S2 rehearsal go: %s" % reh["go"])
    if not reh["go"]["go"]:
        raise Stop("S2: the rehearsal does not reach the go line: %r" % reh["go"])
    return 0


# ------------------------------------------------------------------------------------------------ S3: the probe, the reveal
def read_slow_release(rl, path=None):
    """precondition S3_slow_release_probe of rule_version 3 on P19's evidence (DECISIONS_P2_ASM.md 7.1: -dev/evidence/
    entry/slow_release_probe.json, re-summarised by P19b in format rov-asm-slow-release/2): per unit kind -- its 'kinds'
    vertical_thruster and guard -- the three fields the rule names (evidence_fields): the counted trials of the probe
    class (the unit let go above its window), the false locks (captures at whose moment no finger pad touched the
    unit) and the captures with the pads still on the unit (placements: recorded, no limit, they do not hold up the
    opening), on the product windows of the cell frozen now (the runs' CELL_FREEZE.json sha256 = the file's now). The
    rule's numbers decide (>= min_trials_per_unit_kind trials and <= max_false_locks false locks for each kind); a
    missing field is 'no such evidence'. P19's own verdicts, the strict count of format /1 (captured_after_the_release),
    the trials that ended in an error and the sensitivity set are recorded, not judged"""
    pre = rl["procedure"]["preconditions"]["S3_slow_release_probe"]
    fields = pre["evidence_fields"]
    path = path or SLOW_RELEASE_PATH
    out = {"path": os.path.relpath(path, UW), "exists": os.path.isfile(path),
           "min_trials_per_unit_kind": pre["min_trials_per_unit_kind"], "max_false_locks": pre["max_false_locks"],
           "probe_class_of_the_rule": pre["probe_class"], "false_lock_of_the_rule": pre["false_lock"],
           "evidence_fields_of_the_rule": {k: fields[k] for k in ("trials", "false_locks", "captured_with_the_pads_on_the_unit")},
           "format_required": SLOW_RELEASE_FORMAT}
    if not out["exists"]:
        out["pass"] = False
        out["why"] = "no evidence file"
        return out
    out["sha256"] = sha256_file(path)
    ev = load(path)
    out["format"], out["package"], out["date"] = ev.get("format"), ev.get("package"), ev.get("date")
    out["trials_by_package"], out["ruling"] = ev.get("trials_by_package"), ev.get("ruling")
    out["supersedes"] = ev.get("supersedes")
    out["definition"] = ev.get("definition")
    out["evidence_pass"] = ev.get("pass")
    out["evidence_pass_by_the_strict_count"] = ev.get("pass_by_the_strict_count")
    out["evidence_errors"] = ev.get("errors")
    kinds = ev.get("kinds") if isinstance(ev.get("kinds"), dict) else {}
    names = {"vertical thruster": "vertical_thruster", "thruster guard": "guard"}
    # the rule's path templates kinds.<kind>.<field>: the three fields, read through the template
    tmpl = {k: fields[k].split(".") for k in ("trials", "false_locks", "captured_with_the_pads_on_the_unit")}
    found, missing = {}, []
    for kind, key in names.items():
        got = {}
        for fld, parts in tmpl.items():
            node = ev
            for p in parts:
                p = key if p == "<kind>" else p
                node = node.get(p) if isinstance(node, dict) else None
            if type(node) is int:
                got[fld] = node
            else:
                missing.append("%s: no integer %s" % (kind, ".".join(key if p == "<kind>" else p for p in parts)))
        if len(got) == len(tmpl):
            v = kinds.get(key) or {}
            found[kind] = dict(got, **{"key": key, "errors": v.get("errors"),
                                       "captured": v.get("captured"), "fastened": v.get("fastened"),
                                       "captured_with_no_pad_on_the_unit": v.get("captured_with_no_pad_on_the_unit"),
                                       "captured_after_the_release_strict_count_of_format_1": v.get("captured_after_the_release"),
                                       "fates": v.get("fates"), "points": v.get("points"),
                                       "pad_force_at_capture": v.get("pad_force_at_capture"),
                                       "sensitivity_slowed_opening": {k: (v.get("sensitivity_slowed_opening") or {}).get(k)
                                                                      for k in ("trials", "captured", "fastened", "captured_with_the_pads_on_the_unit",
                                                                                "captured_with_no_pad_on_the_unit")}})
    out["by_unit_kind"] = found
    now = sha256_file(CELL_FREEZE_PATH)
    out["cell_freeze_sha256_now"] = now
    out["cell_freeze_sha256_of_the_evidence"] = ev.get("cell_freeze_sha256")
    out["cell_freeze_sha256_of_the_runs"] = ev.get("cell_freeze_sha256_of_the_runs")
    probs = []
    if ev.get("format") != SLOW_RELEASE_FORMAT:
        probs.append("the evidence is not in the format %s (format %r)" % (SLOW_RELEASE_FORMAT, ev.get("format")))
    probs.extend(missing)
    for kind in pre["unit_kinds"]:
        f = found.get(kind)
        if f is None:
            probs.append("no count for %s" % kind)
            continue
        if f["trials"] < int(pre["min_trials_per_unit_kind"]):
            probs.append("%s: %d trials" % (kind, f["trials"]))
        if f["false_locks"] > int(pre["max_false_locks"]):
            probs.append("%s: %d false locks" % (kind, f["false_locks"]))
    if ev.get("cell_freeze_sha256") != now or ev.get("cell_freeze_sha256_of_the_runs") != [now]:
        probs.append("the probe's runs are not on the cell frozen now (%r, %r)" % (ev.get("cell_freeze_sha256"), ev.get("cell_freeze_sha256_of_the_runs")))
    out["problems"] = probs
    out["pass"] = not probs
    return out


def spent_generations():
    g = set()
    if os.path.isdir(SUPERSEDED):
        for fn in os.listdir(SUPERSEDED):
            if fn.endswith(".json"):
                try:
                    r = load(os.path.join(SUPERSEDED, fn)).get("reveal") or {}
                    if r.get("generation"):
                        g.add(int(r["generation"]))
                except (OSError, ValueError):
                    pass
    return g


def cmd_dryopen(a):
    """a DRY run of S4-S9 on the PUBLIC test vectors (never a sealed seed): a copy of the rehearsed selection with the
    test key and test layout blob in place of a reveal, written to ROV_ASM_SEL_PATH (never the task's selection.json)"""
    if not DRY or os.path.abspath(SET_PATH) == os.path.abspath(OFFICIAL_SET_PATH) or os.path.abspath(EVIDENCE).startswith(os.path.abspath(DEV)):
        raise Stop("dryopen needs ROV_ASM_SEL_PATH, ROV_ASM_SET_PATH and ROV_ASM_EVIDENCE outside the task and its -dev folder")
    sel = load(OFFICIAL_SEL_PATH)
    if sel.get("state") != "rehearsed":
        raise Stop("dryopen needs the official selection rehearsed (state %s)" % sel.get("state"))
    require_frozen(sel)
    rl = rule()
    t = rl["streams"]["known_answer_test"]
    W, L = bytes.fromhex(t["key_W_hex"]), bytes.fromhex(t["blob_L_hex"])
    sel["dry_run"] = {"what": "S4-S9 on the PUBLIC test vectors of streams.known_answer_test, to test the selection loop before a "
                              "generation is opened; never a set", "utc": utc_now(), "official_selection_sha256": sha256_file(OFFICIAL_SEL_PATH)}
    sel["reveal"] = {"generation": 0, "dry": True, "utc": utc_now(), "W_hex": W.hex(), "L_hex": L.hex(), "W_sha256": commitment(W),
                     "L_sha256": commitment(L), "W_equals_commitment": commitment(W) == t["commitment_W"],
                     "L_equals_commitment": commitment(L) == t["commitment_L"]}
    seeds = layout_seeds(L)
    sel["derived"] = {"dome_u64": u64(W, "W", "dome"), "dome_index": u64(W, "W", "dome") % 6,
                      "dome_assignment": dict(zip(STRATA, dome_assignment(W))), "layout_seeds": {"c%02d" % i: int(s) for i, s in enumerate(seeds)}}
    sel["state"] = "opened"
    sel["states"].append({"state": "opened (dry)", "utc": utc_now()})
    save(SEL_PATH, sel)
    log("dry run opened on the public test vectors: %s" % SEL_PATH)
    return 0


def cmd_open(a):
    if DRY:
        raise Stop("open is never run in a dry run")
    sel = load(SEL_PATH)
    if sel.get("state") != "rehearsed":
        raise Stop("open needs a rehearsed selection that reached the go line (state %s)" % sel.get("state"))
    require_frozen(sel)
    rl = rule()
    sr = read_slow_release(rl)
    sel["preconditions"]["S3_slow_release_probe"] = sr
    log("S3 precondition slow-release probe: %s %s" % ("pass" if sr["pass"] else "FAIL", sr.get("problems") or sr.get("why") or sr.get("by_unit_kind")))
    if not sr["pass"]:
        save(SEL_PATH, sel)
        raise Stop("S3: the slow-release probe evidence of P19 does not hold (%s): no generation is opened" % (sr.get("problems") or sr.get("why")))
    gens = rl["streams"]["commitment"]["generations"]
    spent = spent_generations()
    g = min(x["generation"] for x in gens if x["generation"] not in spent)
    com = [x for x in gens if x["generation"] == g][0]
    u0 = utc_now()
    W = bytes.fromhex(open(os.path.join(SEALED_DIR, "gen%d_W.hex" % g)).read().strip())
    L = bytes.fromhex(open(os.path.join(SEALED_DIR, "gen%d_L.hex" % g)).read().strip())
    rv = {"generation": g, "utc": u0, "W_hex": W.hex(), "L_hex": L.hex(), "W_bytes": len(W), "L_bytes": len(L),
          "W_sha256": commitment(W), "L_sha256": commitment(L), "spent_before": sorted(spent)}
    rv["W_equals_commitment"] = rv["W_sha256"] == com["W_sha256"]
    rv["L_equals_commitment"] = rv["L_sha256"] == com["L_sha256"]
    sel["reveal"] = rv
    sel["state"] = "opened"
    sel["states"].append({"state": "opened", "utc": u0})
    if not (rv["W_equals_commitment"] and rv["L_equals_commitment"] and len(W) == 32 and len(L) == 96):
        sel["state"] = "rejected"
        save(SEL_PATH, sel)
        raise Stop("S3: the sealed seeds of generation %d are not the committed ones" % g)
    seeds = layout_seeds(L)
    dome = list(dome_assignment(W))
    sel["derived"] = {"dome_u64": u64(W, "W", "dome"), "dome_index": u64(W, "W", "dome") % 6, "dome_assignment": dict(zip(STRATA, dome)),
                      "layout_seeds": {"c%02d" % i: int(s) for i, s in enumerate(seeds)}}
    save(SEL_PATH, sel)
    log("S3 opened generation %d (commitments equal); S4 dome assignment %s" % (g, sel["derived"]["dome_assignment"]))
    return 0


# ------------------------------------------------------------------------------------------------ S5-S8: the selection
def cmd_select(a):
    sel = load(SEL_PATH)
    if sel.get("state") not in ("opened", "selecting"):
        raise Stop("select needs an opened generation (state %s)" % sel.get("state"))
    require_frozen(sel)
    need_pinned()
    rl = rule()
    W = bytes.fromhex(sel["reveal"]["W_hex"])
    key_hex = W.hex()
    max_draws = int(rl["streams"]["W"]["max_draws_per_stratum"])
    dome = sel["derived"]["dome_assignment"]
    seeds = sel["derived"]["layout_seeds"]
    kit_dir = ensure_kit()
    kit_json = load(os.path.join(kit_dir, "kit.json"))
    if sel.get("state") == "opened":
        sel["state"] = "selecting"
        sel["states"].append({"state": "selecting", "utc": utc_now()})
    ts = sel.setdefault("trim_stage", new_trim_stage())
    lcs = sel.setdefault("layout_candidates", [{"id": "c%02d" % i, "seed": int(seeds["c%02d" % i]), "status": "not_run"} for i in range(N_LAYOUT)])
    rounds = sel.setdefault("rounds", [])
    runs = sel.setdefault("runs", [])
    # the view check of every layout candidate for both dome ends at once (a pure function of the frozen files; the rule
    # judges candidate k only for the stratum that consumes it, in order)
    vpath = os.path.join(EVIDENCE, "views_gen%d.json" % sel["reveal"]["generation"])
    if not os.path.isfile(vpath):
        views, vres = all_views([(x["id"], x["seed"]) for x in lcs], (0, 180), rl, kit_dir, kit_json, "gen%d" % sel["reveal"]["generation"], a.procs)
        save(vpath, vres)
    vres = load(vpath)
    views = {}
    for jid, v in vres["jobs"].items():
        lid, d = jid.split("/dome")
        views[(lid, int(d))] = v
    sel["views"] = {"file": os.path.relpath(vpath, DEV), "sha256": sha256_file(vpath)}

    def persist():
        require_frozen(sel, image=False)
        save(SEL_PATH, sel)

    def next_trim(sid):
        st = ts[sid]
        while True:
            avail = [r for r in st["records"] if r["qualified"] and r["index"] not in st["consumed"]]
            if avail:
                return avail[0]
            if st["next_index"] >= max_draws:
                return None
            trim_extend(ts, key_hex, [sid], 1, 1, max_draws)
            log("trim stage %s: evaluated through %d, qualified %s" % (sid, st["next_index"] - 1, trim_counts(st)["qualified_indices"][:6]))
            persist()

    def next_layout(sid, dm):
        drops = []
        for x in lcs:
            if x.get("consumed"):
                continue
            r = layout_stage_check(x["id"], x["seed"], dm, rl, kit_json, views)
            x.update({"stratum": sid, "dome_judged": int(dm), "layout_digest": r["layout_digest"], "d": r["d"],
                      "e": {k: r["e"][k] for k in ("e1", "e1_pass", "e1_least_visible", "e2", "e2_pass", "e2_least_diam_px", "pass")} if r.get("e") else None,
                      "round": rounds[-1]["round"]})
            if not (r["pass_d"] and r["pass_e"]):
                x["consumed"] = True
            if not r["pass_d"]:
                x["status"] = "dropped_d"
                x["reason"] = r["reason"]
                drops.append({"layout_id": x["id"], "stratum": sid, "reason": r["reason"]})
                continue
            if not r["pass_e"]:
                x["status"] = "dropped_e"
                x["reason"] = r["reason"]
                drops.append({"layout_id": x["id"], "stratum": sid, "reason": r["reason"]})
                continue
            persist()
            return x["id"], drops
        persist()
        return None, drops

    def consume(sid, idx, lid):
        st = ts[sid]
        if [r["index"] for r in st["records"] if r["qualified"] and r["index"] not in st["consumed"]][:1] != [idx]:
            raise RuntimeError("consume: %s %d is not the stratum's next qualified candidate" % (sid, idx))
        st["consumed"].append(idx)
        x = lc(lid)
        if x.get("consumed") or [y["id"] for y in lcs if not y.get("consumed")][:1] != [lid]:
            raise RuntimeError("consume: %s is not the next layout candidate" % lid)
        x["consumed"] = True
        x["status"] = "run"
        x["trim_candidate"] = idx

    def lc(lid):
        return [x for x in lcs if x["id"] == lid][0]

    def trec(sid, idx):
        return [r for r in ts[sid]["records"] if r["index"] == idx][0]

    def play(rnd, pairs):
        entries = [episode_entry(p["stratum"], trec(p["stratum"], p["trim_index"]), {"seed": lc(p["layout_id"])["seed"], "dome": p["dome"],
                                                                                     "layout_digest": lc(p["layout_id"])["layout_digest"]},
                                 {"provenance": {"generation": sel["reveal"]["generation"], "trim_candidate": p["trim_index"],
                                                 "layout_candidate": p["layout_id"], "round": rnd}}) for p in pairs]
        tag = "gen%d_round%d" % (sel["reveal"]["generation"], rnd)
        require_frozen(sel)
        u0 = utc_now()
        t0 = time.time()
        run, verdict, summ = play_in_image(sel["frozen"]["image"]["tag"], tag, entries)
        js = judge_run(entries, verdict, summ, rl)
        runs.append({"round": rnd, "tag": tag, "utc_start": u0, "utc_end": utc_now(), "wall_s": round(time.time() - t0, 1), "image_run": run,
                     "pairs": [{"stratum": p["stratum"], "trim_index": p["trim_index"], "layout_id": p["layout_id"], "dome": p["dome"],
                                "entry": e, "a": j} for p, e, j in zip(pairs, entries, js)]})
        for p, j in zip(pairs, js):
            log("round %d %s: trim %d + %s -> (a) %s %s" % (rnd, p["stratum"], p["trim_index"], p["layout_id"], "pass" if j["pass"] else "FAIL", j["failed"]))
            if not j["pass"]:
                lc(p["layout_id"])["a_failed"] = j["failed"]
        persist()
        return [{"pass": j["pass"], "failed": j["failed"]} for j in js]

    if all(not ts[s]["records"] for s in STRATA):
        # the four strata's streams to their first qualified candidate at once (the order of evaluation does not change
        # any record: candidate c of a stratum is a function of the key, the stratum and c)
        trim_extend(ts, key_hex, STRATA, 1, a.procs, max_draws)
        log("trim stage: first qualified %s" % {s: trim_counts(ts[s])["qualified_indices"][:1] for s in STRATA})
        persist()
    S = Selector(dome, next_trim, next_layout, consume, play, rounds, persist)
    t0 = time.time()
    r = S.run()
    sel["select_wall_s"] = round(sel.get("select_wall_s", 0.0) + time.time() - t0, 1)
    for s in STRATA:
        ts[s]["counts"] = trim_counts(ts[s])
    if r["outcome"] != "selected":
        sel["state"] = "rejected"
        sel["fallback"] = {"fallback": r["outcome"], "stratum": r.get("stratum"), "set_written": False, "episode_taken_automatically": False,
                           "nominal_layout_retreat": False, "oracle_nominal_layout": False,
                           "failed_runs": [p for x in runs for p in x["pairs"] if not p["a"]["pass"]],
                           "dropped_layout_candidates": [x for x in lcs if x["status"] in ("dropped_d", "dropped_e")],
                           "then": "stop; back to the main session with the cause of every failed run"}
        if r["outcome"] == "FB-W":
            sel["fallback"]["trim_counts"] = ts[r["stratum"]]["counts"]
        sel["states"].append({"state": "rejected", "utc": utc_now()})
        save(SEL_PATH, sel)
        raise Stop("fallback %s (stratum %s): no set is written" % (r["outcome"], r.get("stratum")))
    # S8
    eps = []
    for sid in STRATA:
        p = r["episodes"][sid]
        run = [x for x in runs if x["round"] == [q for q in rounds if p in q["pairs"]][0]["round"]][0]
        e = [q for q in run["pairs"] if q["stratum"] == sid][0]["entry"]
        eps.append(e)
    check_set(eps, rl, sel)
    doc = {"format": SET_FORMAT, "task": SLUG, "rule_version": rl["rule_version"], "generation": sel["reveal"]["generation"], "episodes": eps}
    save(SET_PATH, doc)
    sel["set"] = {"episodes": eps, "episodes_sha256": sha256_file(SET_PATH), "path": os.path.relpath(SET_PATH, TASK)}
    sel["fallback"] = None
    sel["state"] = "selected"
    sel["states"].append({"state": "selected", "utc": utc_now()})
    persist()
    log("S8 selected: %s" % [(e["stratum"], e["provenance"]["trim_candidate"], e["provenance"]["layout_candidate"]) for e in eps])
    return 0


def check_set(eps, rl, sel):
    """the set against verify.py's own loader and the requests of the main session (DECISIONS_P2_ASM 2, the row for
    P20): format rov-asm-set/1; the public example work order >= 1 degC and >= 1 psu from every episode; a stored
    density equals trim_eval.rho_teos10 bit for bit"""
    import tempfile
    G = _grader_set_loader()
    tmp = tempfile.mkdtemp(prefix="p20_set_", dir=WORK)
    try:
        save(os.path.join(tmp, "episodes.json"), {"format": SET_FORMAT, "episodes": eps})
        got = G["load_hidden_set"]({"episodes": 4}, tmp, load(os.path.join(TESTS, "trim_table.json"))["per_unit_deviation"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if [g["stratum"] for g in got] != list(STRATA):
        raise Stop("the set is not E1..E4 in order")
    ex = load(os.path.join(V12, "work_order_example.json"))["work_order"]["delivery_site"]
    for e in eps:
        s = e["work_order"]["delivery_site"]
        if abs(s["water_temperature_C"] - ex["water_temperature_C"]) < 1.0 or abs(s["salinity_psu"] - ex["salinity_psu"]) < 1.0:
            raise Stop("episode %s is within 1 degC / 1 psu of the public example work order" % e["stratum"])
    study()
    E = _STUDY["R"].E
    for sid in STRATA:
        st = sel["trim_stage"][sid]
        idx = [e for e in eps if e["stratum"] == sid][0]["provenance"]["trim_candidate"]
        rec = [r for r in st["records"] if r["index"] == idx][0]
        s = [e for e in eps if e["stratum"] == sid][0]["work_order"]["delivery_site"]
        rho = E.rho_teos10(s["water_temperature_C"], s["salinity_psu"])
        if rec["detail"]["rho_kg_m3"] != rho:
            raise Stop("the stored density of %s is not trim_eval.rho_teos10 bit for bit" % sid)
    return True


# ------------------------------------------------------------------------------------------------ S9: acceptance
def cmd_accept(a):
    sel = load(SEL_PATH)
    if sel.get("state") != "selected":
        raise Stop("accept needs a selected set (state %s)" % sel.get("state"))
    require_frozen(sel)
    rl = rule()
    doc = load(SET_PATH)
    if sha256_file(SET_PATH) != sel["set"]["episodes_sha256"]:
        raise Stop("episodes.json is not the one selection.json records")
    eps = doc["episodes"]
    runs = []
    for k in (1, 2):
        tag = "gen%d_accept%d" % (sel["reveal"]["generation"], k)
        run, verdict, summ = play_in_image(sel["frozen"]["image"]["tag"], tag, eps, keep_set=True)
        js = judge_run(eps, verdict, summ, rl)
        m = verdict.get("metrics") or {}
        runs.append({"tag": tag, "image_run": run, "reward": (verdict.get("rewards") or {}).get("reward"),
                     "assembly_score": m.get("assembly_score"),
                     "per_episode": [{x: m.get("ep%d_%s" % (i, x)) for x in ("stage", "hard_ok", "trim", "mates", "net_buoyancy_N", "pitch_deg", "roll_deg")}
                                     for i in range(1, len(eps) + 1)],
                     "a": js, "sim_time_s": [s.get("sim_time_s") for s in summ],
                     "max_part_force_sustained_N": [(s.get("watch") or {}).get("max_part_force_sustained_N") for s in summ]})
        log("S9 acceptance run %d: reward %s, per episode %s" % (k, runs[-1]["reward"], [p["stage"] for p in runs[-1]["per_episode"]]))
    ok = all(r["reward"] == 1.0 for r in runs) and runs[0]["per_episode"] == runs[1]["per_episode"]
    sel["acceptance"] = {"runs": runs, "both_reward_1": all(r["reward"] == 1.0 for r in runs),
                         "identical_per_episode_scores": runs[0]["per_episode"] == runs[1]["per_episode"],
                         "identical_sim_times": runs[0]["sim_time_s"] == runs[1]["sim_time_s"], "pass": ok}
    sel["state"] = "accepted" if ok else "rejected"
    sel["states"].append({"state": sel["state"], "utc": utc_now()})
    save(SEL_PATH, sel)
    if not ok:
        raise Stop("S9: the set is not accepted (see selection.json acceptance)")
    return 0


# ------------------------------------------------------------------------------------------------ S10: the report
def cmd_report(a):
    sel = load(SEL_PATH)
    need_pinned()
    rl = rule()
    rep = {"utc": utc_now(), "state": sel.get("state")}
    ts = sel.get("trim_stage") or {}
    rep["trim_stage"] = {s: trim_counts(ts[s]) for s in ts}
    lcs = sel.get("layout_candidates") or []
    rep["layout_candidates"] = {x["id"]: {k: x.get(k) for k in ("status", "stratum", "dome_judged", "round", "reason", "a_failed")} for x in lcs}
    ran = [x for x in lcs if x.get("status") == "run"]
    dropped_after_a = [x for x in ran if x.get("a_failed")]
    causes = {}
    for x in sel.get("runs", []):
        for p in x["pairs"]:
            if not p["a"]["pass"]:
                fd = p["a"].get("failed_detail") or {}
                for c in p["a"]["failed"]:
                    # the same clause on the same mate or unit (rule hollowing_report.flags.shared_cause)
                    on = {"structural_mates_counted": fd.get("mates_not_counted"), "scrapped_units": fd.get("scrapped"),
                          "max_sustained_part_contact_force": (fd.get("force_pair") or [None])[:1], "hard_ok": fd.get("violations"),
                          "unplanned_contacts": fd.get("unplanned_contacts")}.get(c) or ["(no mate or unit named)"]
                    for item in on:
                        key = (c, str(item))
                        causes[key] = causes.get(key, 0) + 1
    fl = rl["hollowing_report"]["flags"]
    share = (len(dropped_after_a) / len(ran)) if ran else 0.0
    rep["hollowing"] = {"run_for_a": len(ran), "dropped_after_a": len(dropped_after_a), "dropped_share": round(share, 4),
                        "flag_dropped_share": share > float(fl["dropped_share_above"]),
                        "shared_causes": {"%s @ %s" % k: v for k, v in causes.items() if v >= 2},
                        "flag_shared_cause": any(v >= 2 for v in causes.values())}
    # diagnostics of the kept episodes
    diag = {}
    for e in (sel.get("set") or {}).get("episodes", []):
        sid = e["stratum"]
        rec = [r for r in ts[sid]["records"] if r["index"] == e["provenance"]["trim_candidate"]][0]
        x = [y for y in lcs if y["id"] == e["provenance"]["layout_candidate"]][0]
        run = [p for r in sel.get("runs", []) for p in r["pairs"] if p["stratum"] == sid and p["a"]["pass"]][0]
        d = rec["detail"]
        four = {k + "@" + rr: (d["strict"].get(k + "@" + rr), d["lenient"].get(k + "@" + rr)) for k in rl["conditions"]["c"]["two_choice_rules_for"]
                for rr in ("maximin", "dB_first")}
        diag[sid] = {"work_order": e["work_order"], "deviation": e["deviation"], "rho_kg_m3": d["rho_kg_m3"], "reference": d["reference"],
                     "layouts_full": d["layouts_full"], "b2_full": d["b2"]["full"], "strict": d["strict"], "lenient": d["lenient"],
                     "partial_weighing_both_rules_strict_lenient": four, "in_order_correct_M13": x["d"]["in_order_correct"],
                     "views": x.get("e"), "run": {k: run["a"]["fields"].get(k) for k in ("sim_time_s", "episode_wall_s", "turns_used", "camera_frames_total",
                                                                                          "max_part_force_sustained_N", "max_part_force_sustained_pair",
                                                                                          "max_robot_link_force_N", "max_dome_force_sustained_N",
                                                                                          "max_flange_force_sustained_N", "weld_F_sustained_max_N", "trim_eval")}}
    rep["diagnostics"] = diag
    # (c) on the four episodes, every reading against its limit (lenient window; the four partial-weighing readings
    # under both choice rules), re-evaluated in this process and compared with the stored record
    if diag:
        S = study()
        cc = {}
        for e in sel["set"]["episodes"]:
            sid = e["stratum"]
            st = stratum_spec(rl, sid)
            idx = e["provenance"]["trim_candidate"]
            rec = [r for r in ts[sid]["records"] if r["index"] == idx][0]
            again = eval_trim(S, bytes.fromhex(sel["reveal"]["W_hex"]), st, rl["per_unit_deviation"], idx)
            lim = S.limits(st["c3_zero"])
            rows = {k: {"lenient": rec["detail"]["lenient"][k], "limit": v, "pass": rec["detail"]["lenient"][k] <= v + S.tol} for k, v in sorted(lim.items())}
            rows["robust_noweigh"] = {"lenient": rec["detail"]["robust_noweigh"]["lenient"], "limit": 0.0,
                                      "pass": rec["detail"]["robust_noweigh"]["lenient"] <= S.tol}
            same = json.loads(json.dumps(again, default=_json_default)) == json.loads(json.dumps(rec, default=_json_default))
            cc[sid] = {"readings": rows, "all_pass": all(r["pass"] for r in rows.values()), "re_evaluated_identical": same,
                       "two_choice_rules_for": rl["conditions"]["c"]["two_choice_rules_for"]}
        rep["c_on_the_set"] = {"by_stratum": cc, "pass": all(v["all_pass"] and v["re_evaluated_identical"] for v in cc.values())}
    # audit of the first-qualifying rule: every stratum consumed its qualified trim candidates in index order and the
    # layout candidates were consumed in c00..c11 order without a gap; the kept pair is the stratum's last consumed
    au = {}
    order = [x["id"] for x in lcs if x.get("consumed")]
    au["layouts_consumed_in_order"] = order == [x["id"] for x in lcs][:len(order)]
    for s in STRATA:
        st = ts.get(s) or {}
        q = [r["index"] for r in st.get("records", []) if r["qualified"]]
        cons = st.get("consumed", [])
        kept = [e for e in (sel.get("set") or {}).get("episodes", []) if e["stratum"] == s]
        pairs = [{"round": r["round"], "trim_index": p["trim_index"], "layout_id": p["layout_id"], "result": p.get("result")}
                 for r in sel.get("rounds", []) for p in r["pairs"] if p["stratum"] == s]
        au[s] = {"qualified_indices": q, "consumed": cons, "consumed_are_the_first_qualified": cons == q[:len(cons)],
                 "pairs": pairs, "kept": (kept[0]["provenance"] if kept else None),
                 "kept_is_the_last_pair_and_the_only_pass": bool(kept) and bool(pairs) and pairs[-1]["trim_index"] == kept[0]["provenance"]["trim_candidate"]
                 and pairs[-1]["layout_id"] == kept[0]["provenance"]["layout_candidate"]
                 and [bool((p["result"] or {}).get("pass")) for p in pairs] == [False] * (len(pairs) - 1) + [True]}
    au["pass"] = au["layouts_consumed_in_order"] and all(au[s]["consumed_are_the_first_qualified"] and (au[s]["kept_is_the_last_pair_and_the_only_pass"]
                                                                                                        or not (sel.get("set") or {}).get("episodes"))
                                                         for s in STRATA)
    rep["first_qualifying_audit"] = au
    rep["drops"] = {"layout": [{k: x.get(k) for k in ("id", "status", "stratum", "dome_judged", "reason")} for x in lcs if x.get("status") in ("dropped_d", "dropped_e")],
                    "failed_runs": [{"round": r["round"], "stratum": p["stratum"], "trim_index": p["trim_index"], "layout_id": p["layout_id"],
                                     "failed": p["a"]["failed"], "detail": p["a"].get("failed_detail")} for r in sel.get("runs", []) for p in r["pairs"] if not p["a"]["pass"]],
                    "trim_failed_histogram": {s: trim_counts(ts[s])["failed_histogram"] for s in ts}}
    sel["report"] = rep
    save(SEL_PATH, sel)
    print(json.dumps({k: v for k, v in rep.items() if k != "diagnostics"}, indent=1, default=_json_default))
    return 0


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("freeze")
    p.add_argument("--image", default=IMAGE_DEFAULT)
    p.add_argument("--why", default=None)
    sub.add_parser("check")
    p = sub.add_parser("amend")
    p.add_argument("--class", dest="klass", required=True, choices=("N", "V", "S"))
    p.add_argument("--what", required=True)
    p.add_argument("--why", required=True)
    p.add_argument("--authorised-by", dest="authorised_by", required=True)
    p.add_argument("--detail", default=None, help="a JSON file kept verbatim in the entry")
    p = sub.add_parser("rehearse")
    p.add_argument("-P", "--procs", type=int, default=4)
    sub.add_parser("open")
    sub.add_parser("dryopen")
    p = sub.add_parser("select")
    p.add_argument("-P", "--procs", type=int, default=4)
    sub.add_parser("accept")
    sub.add_parser("report")
    p = sub.add_parser("view")
    p.add_argument("--jobs", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    fn = {"selftest": cmd_selftest, "freeze": cmd_freeze, "check": cmd_check, "amend": cmd_amend, "rehearse": cmd_rehearse, "open": cmd_open, "dryopen": cmd_dryopen,
          "select": cmd_select, "accept": cmd_accept, "report": cmd_report, "view": cmd_view}[a.cmd]
    try:
        return fn(a)
    except Stop as e:
        log("STOP (back to the main session): %s" % e)
        return 3


if __name__ == "__main__":
    sys.exit(main())
