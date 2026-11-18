"""The hidden set's selection rule, re-runnable from the records. Version 2.1: candidates k = 1..48 of the label
.../hidden/v2/<k>, each put through the admission rows on its own (row 3's time part an absolute bound since
version 2.1: the reference finishes within 1200 s of simulated
time); then, going through the admitted candidates in k order, a candidate joins the set unless it would be the
seventh with its bund kerb class, the seventh with its B-step variant, the seventh with its B-low variant, or the fifth
with its sheet place; the set is complete at twelve (the quota rule with every quota doubled); the practice
buildings are chosen by the practice rule over practice candidates 1..12 (`practice`, unchanged). Version 2.0's rule (six; quotas
3 / 3 / 3 / 2; row 3's time part 50 % of the mission's 2400 s) and version 1's rule
(k = 1..24) are replaced and not kept in this folder.

    python3 -B -s tests/review/select_set.py rule    --out OUT [--label hidden] [--k-range 1-48] [--tables TABLES.json]
    python3 -B -s tests/review/select_set.py freeze  --candidates DIR --proofs RECORDS --out OUT [--label hidden]
                                                     [--k-range 1-48]
    python3 -B -s tests/review/select_set.py run     --candidates DIR --out OUT [--k 1-24] [--workers 12]
                                                     [--baselines all|admissible]
    python3 -B -s tests/review/select_set.py run-one --candidates DIR --out OUT --k K --prog oracle|<baseline>
    python3 -B -s tests/review/select_set.py decide  --out OUT
    python3 -B -s tests/review/select_set.py practice --out OUT [--write DIR --candidates DIR --hall HALL.tar.xz]
    python3 -B -s tests/review/select_set.py report  --out OUT [--md TABLE.md]
    python3 -B -s tests/review/select_set.py write-hidden --out OUT --hidden tests/hidden
    python3 -B -s tests/review/select_set.py compare A/result.json B/result.json
    python3 -B -s tests/review/select_set.py --freeze-only --candidates DIR --proofs CANDIDATES.json --out OUT

  DIR         the candidate packages (the generator's <label>_kNN.tar.xz, e.g. hidden_k01.tar.xz .. hidden_k48.tar.xz;
              cNN.tar.xz also read)
  --proofs    the generator's candidate records: version 2's JSON lines (site_gen.sets, one record per candidate: k,
              seed, generated, proofs_ok, failed_rows, the table's columns, sheet_count, c_sealed_way, deviations,
              package.sha256, ...) or version 1's candidates.json (rows) / {"c01": {...}}
  --label     the seed label: seed_k = int.from_bytes(sha256(b"quadruped_hazardous_plant_inspection/<label>/v2/<k>")
              .digest()[:8], "big") >> 1 (hidden, practice, or a development label); the mission's noise seed is the
              candidate's seed_k, as in tests/hidden/seeds.json
  OUT         the selection's folder: selection.json (the rule, the frozen files' sha256, the candidates' sha256, then
              every candidate's result and the set), runs/cKK/<prog>/ (each mission's records and result.json),
              jobs.jsonl (the job log of `run`)

`rule` writes the rule (with the quota rule and the note on admission row 3), the label, the k range, the generator's
tables (the type table by candidate number with the exclusion list and its measured rates: `python -m site_gen.sets --table 1-48`) and
the exclusion list's file with its sha256 (exclusion_v2.json, sha256 3413ac3b..., the tables' list must be the
file's) into OUT/selection.json BEFORE any seed of the label is generated; `freeze` then adds the sha256 of every frozen file and of every candidate package
BEFORE any candidate runs with the frozen cell (it refuses a rule, label or tables other than those `rule` wrote);
every later command refuses to run when a frozen file changed (record an amendment instead).
`run-one` runs one mission exactly as the grader runs it (tests/verify.py): a fresh PlantCellServer from
tests/grader_config.json on the package, reset(package, seed=seed_k, kind="hidden"), the submission as a child process
over cellkit.wire with verify.py's caps, the process under the image's ENV and tests/test.sh's exports (it re-executes
itself into that environment), and verify.py's one rerun of a mission in which the cell failed. The oracle is
solution/reference/patrol.py with the privileged calls; a baseline is tests/review/baselines.py's submission. `run`
runs the jobs (every candidate's oracle, then the row-4 baselines) in a pool of worker processes, one fresh process per
mission, resumable (a job with a result.json is not run again). `decide` applies the rows to every candidate from the records and
then the quota rule to the admitted ones in k order, writes every candidate's verdict, the set and the hollowing table
into selection.json (version 2: the table's columns,
the number of instruments on the missing sheet and the as-built way of C-sealed among the features). `practice`
applies the practice rule to the decided records of a practice label (qualifying: row 1, every item delivered with accepted evidence
and S >= 0.80; then the first triple i < j < l showing both bund classes, both B-step and B-low variants, three
sheet places and every deviation class: the generator's own `site_gen.variants.select_practice`, from the task's
generator copy), records t1-t3 and the candidates passed over in OUT/practice.json, and with --write lays the practice
ground's packages out (hall.tar.xz, t1..t3.tar.xz, index.json) for the maintainers' assembly tool.
`write-hidden` writes the set in the grader's layout: sites/h01..h12.tar.xz (byte copies of the twelve packages of the
set, in k order), seeds.json (the schema tests/verify.py reads) and selection.json.
"""

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time

REVIEW = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(REVIEW)
TASK = os.path.dirname(TESTS)
HERE = os.path.abspath(__file__)
LABEL_FMT = "quadruped_hazardous_plant_inspection/%s/v2/%d"
N_SET = 12                  # version 2.1 (2026-11-17): twelve scored buildings (version 2.0: six)
K_RANGE_DEFAULT = {"hidden": (1, 48), "practice": (1, 12)}      # two cycles of the table; 12 practice
N_INSTRUMENTS = 8
# row 3's time part: an absolute bound since version 2.1 (the reference finishes within 1200 s of simulated time:
# what "50 % of the 2400 s" of 2026-11-14 12:25 EST meant); the mission's limit itself is 1350 s
SIM_TIME_MAX_S = 1200.0
DOSE_SHARE_MAX = 0.50
FRAME_SHARE_MAX = 0.60
S_MIN = 0.80
CORRIDOR_MAX = 0.10
# every quota doubled in version 2.1: the quota rule over the admitted candidates in k
# order: at most six of each bund kerb class, of each B-step variant and of each B-low variant, at most four of each
# sheet place (so a complete set holds six and six of each and four of each place, by construction)
QUOTA = {"bund": 6, "b_step": 6, "b_low": 6, "sheet_place": 4}
QUOTA_WORDS = {"bund": "bund kerb class", "b_step": "B-step variant", "b_low": "B-low variant",
               "sheet_place": "sheet place"}
ORDINAL = {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh"}
QUOTA_RECORD_KEYS = {"bund": "bund", "b_step": "b_step_variant", "b_low": "b_low_variant", "sheet_place": "sheet_place"}
# the exclusion list stays as measured (two entries); it is written into selection.json with its sha256
EXCLUSION_SHA256 = "3413ac3bfd7de3aa8c9f6b916c78aed92e1ee11652e3dc0f8f5f99752bdc849d"
EXCLUSION_FILE = os.path.join(REVIEW, "sitegen", "site", "config", "exclusion_v2.json")
SUBSEED_CAP = 64
DEGENERATE = ("idle", "walk_out_and_back", "last_logged_values", "constant_values")
ROW4 = DEGENERATE + ("corridor_only",)
RULE = {
    "source": "version 2.1: notes/design/04_v2_spec.md sections 1 (Q14), 7 (R1), 8 (R12, R13) and 11 (R26, "
              "2026-11-17), adopting draft 04_v2_spec_draft.md section 7 (S-1 to S-4); version 1's rule (build spec "
              "B12 as amended) is replaced; R12's quota rule replaces \"the first six admitted\" (Q14 and S-3's last "
              "sentence); R26 doubles every quota of R12 (twelve scored buildings instead of six; the rule as fixed in "
              "build/v2/final/pool.md, rule 1, before the pool run) and keeps row 3's time part as the absolute bound "
              "it meant (1200 s); version 2.0's rule is in git history (commit 40c2da2c)",
    "candidates": "k = 1..48 (two cycles of the table by candidate number, S-2) with seed_k = int.from_bytes(sha256("
                  "b\"quadruped_hazardous_plant_inspection/hidden/v2/<k>\").digest()[:8], \"big\") >> 1; practice "
                  "candidates use the label .../practice/v2/<k>, the documents batch .../doctest/v2/<k>, so the sets "
                  "are disjoint by construction; version 1's seeds, sites and practice buildings are not reused",
    "order": "each candidate is put through the admission rows on its own; one is admitted when all of these hold",
    "rows": {
        "1": "it generates within 64 sub-seeds with every required row (version 1's rows as changed by L1-13, L2-9 and "
             "L3-6, with the new ones: plate_readable, tags, sheet_items, sheet_rules, sheet_nothing_drawn, "
             "sheet_edges, outline_walls, drawing_fails_missing, approach_flat, B-low_back)",
        "2": "the frozen reference, run once exactly as the grader runs the oracle, delivers a report with accepted "
             "evidence for all ten items, read each of the eight plates as its tag before reporting (in its log), ends "
             "upright on the pad, and scores S of at least 0.80",
        "3": "the reference uses at most 1200 s of simulated time (R26: an absolute bound, what 50 % of version "
             "2.0's 2400 s limit meant), 50 % of the dose rating and 60 % of each frame budget",
        "4": "on that site: idle, walk-out-and-back, last-logged-values and constant-values score 0.00, and "
             "corridor-only scores at most 0.10",
    },
    "set": "R12, the quota rule, with every quota doubled by R26: going through the admitted candidates in k order, a "
           "candidate joins the set unless it would be the seventh with its bund kerb class, the seventh with its "
           "B-step variant, the seventh with its B-low variant, or the fifth with its sheet place; the set is "
           "complete at twelve (their k order is the episode order h01..h12). Quotas give six of each kerb class and "
           "of each B variant and four of each sheet place by construction, without choosing by hand; the order is "
           "k, so no tie arises",
    "quota": {"bund": 6, "b_step": 6, "b_low": 6, "sheet_place": 4,
              "read_from": "the generator's record of the candidate as built (bund, b_step_variant, b_low_variant, "
                           "sheet_place); each is checked against the table's row for k (tables.table), a difference "
                           "is recorded with the candidate and the as-built value counts"},
    "exclusion": "R12: the exclusion list stays as measured (two entries, sketch/site/config/exclusion_v2.json, sha256 "
                 "3413ac3b...), with the pass rate taken over every layout that reached the slot's placement stage; "
                 "the sub-seed cap stays (64); the list and the table are written into selection.json before any seed "
                 "of the hidden label is generated",
    "notes": {"R13": "Admission row 3 stays, and the lean it gives is accepted: a site where the reference uses more "
                     "than half of the dose rating or of the time is not scored (a strong source on the corridor every "
                     "route passes: 2 of 13 and 1 of 5 development sites in wave 2). The reference stays time-optimal; "
                     "no dose-aware planner. Reason: the half is the headroom an unprivileged program needs; version 1 "
                     "made the same choice (spec 04_v2_spec.md section 8, R13)",
              "R26": "Version 2.1 (spec 04_v2_spec.md section 11, 2026-11-17): twelve scored buildings, by R12's rule "
                     "with every quota doubled, and a tighter simulated-time limit for every mission, scored and "
                     "practice alike (the training hall keeps its 900 s; the value is R27's). Row 3 keeps its absolute "
                     "bound for time, 1200 s of simulated time (what \"at most 50 % of 2400 s\" meant), with 50 % of "
                     "the dose rating and 60 % of each frame budget as before, so the admitted candidates are the same "
                     "by construction; the admission is run again under the changed configuration and must reproduce "
                     "every final state. The practice rule F-1, the label, the k range, the tables and the exclusion "
                     "list are unchanged"},
    "record": "every candidate's result is recorded, also the dropped ones, with a hollowing check (features of the "
              "twelve against all candidates: the table's columns, the number of instruments on the missing sheet and "
              "the as-built way of C-sealed among them)",
    "if_short": "if fewer than twelve join the set, stop and report: do not change the rule",
    "practice": "F-1 (S-4): practice candidates k = 1..12 of .../practice/v2/<k> with the same table; one qualifies "
                "when row 1 holds and the frozen reference (run as the grader runs the oracle) delivers all ten items "
                "with accepted evidence and scores 0.80 or more; t1, t2, t3 are the first triple i < j < l, in the "
                "order (i, j, l), of qualifying candidates whose buildings together show both bund classes, both "
                "B-step variants, both B-low variants, three different places of the missing sheet and every "
                "deviation class; the candidates passed over are recorded",
    "how_the_rows_are_read": {
        "1": "the generator's candidate record: generated, proofs_ok (every required row in memory), no failed_rows, "
             "and recheck_from_files.required_ok where the record has a recheck, for the package whose sha256 the "
             "record names (a record for other bytes fails the row)",
        "2": "one mission of the oracle submission (solution/reference/patrol.py: one frame of every sensor, then "
             "reference_run) on a privileged PlantCellServer(config=tests/grader_config.json, mode='hidden', "
             "reference_dir=tests/reference) reset(package, seed=seed_k, kind='hidden') over cellkit.wire with "
             "tests/verify.py's caps, in the image's ENV and tests/test.sh's exports; a mission in which the cell "
             "failed (internal error, physics fault, lost frame) is run once more and the second run counts "
             "(verify.py _episode_with_retry). Every item delivered; each instrument's report cites a frame that meets "
             "its evidence condition (E = 1); the mark's dwell condition and the source's approach condition hold "
             "(E = 1); R = 1.0 (upright within 1.0 m of the pad centre, no fall, no dose limit); S >= 0.80; the "
             "reference's own log (reference_log.json of the counted attempt) holds, for each of the eight "
             "instruments, a plate read whose text equals the tag no later than the item's report",
        "3": "from the same mission: sim_time_s <= 1200 s (R26: absolute, whatever the mission's limit; the record "
             "also gives the share of the mission's limit); dose_fraction <= 0.50; for each frame budget of "
             "tests/grader_config.json (nav, inspect, depth, thermal, lidar) frames used <= 0.60 x the budget",
        "4": "one mission of each of tests/review/baselines.py's submissions (unprivileged, the same seed and the "
             "same way as row 2); 'score 0.00' = an S that rounds to 0.00 at two decimals (S < 0.005); corridor_only "
             "S <= 0.10",
    },
}
# what determines a selection mission: the cell, the reference, the wire kit, the run settings, the grader's way of
# running it (verify.py, test.sh's exports, the image's ENV and packages), the programs and this tool
FROZEN = ["tests/plantcell", "tests/reference", "tests/cellkit", "tests/grader_config.json", "tests/verify.py",
          "tests/test.sh", "solution/reference/patrol.py", "tests/review/baselines.py", "tests/review/_lib",
          "tests/review/cell/run_wire.py", "tests/review/select_set.py", "environment/Dockerfile",
          "environment/requirements.txt",
          # version 2: the practice rule's code (`practice`) and the render bundle's pin (tests/plantcell holds it)
          "tests/review/sitegen/site/site_gen/variants.py", "environment/assets.sha256",
          # the exclusion list the table was made with
          "tests/review/sitegen/site/config/exclusion_v2.json"]
# image ENV keys that are paths of the build, not run settings
ENV_SKIP = ("PATH", "VIRTUAL_ENV", "UV_PYTHON_INSTALL_DIR", "UV_PROJECT_ENVIRONMENT", "HOME", "XDG_CONFIG_HOME",
            "XDG_CACHE_HOME", "MPLCONFIGDIR")
ENV_MARK = "QINSPECT_SELECT_ENV"


# ================================================================================================ helpers
def seed_k(k, label="hidden"):
    return int.from_bytes(hashlib.sha256((LABEL_FMT % (label, k)).encode()).digest()[:8], "big") >> 1


def k_range_of(sel):
    lo, hi = sel.get("k_range") or K_RANGE_DEFAULT["hidden"]
    return int(lo), int(hi)


def label_of(sel):
    return sel.get("label") or "hidden"


def key_of(k):
    return "c%02d" % k


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest_of(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def load_json(p):
    with open(p) as f:
        return json.load(f)


def dump_json(obj, p):
    tmp = p + ".tmp%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.write("\n")
    os.replace(tmp, p)


def frozen_hashes(task=TASK):
    out = {}
    for rel in FROZEN:
        p = os.path.join(task, rel)
        if os.path.isfile(p):
            out[rel] = sha256_file(p)
            continue
        if not os.path.isdir(p):
            out[rel] = None
            continue
        for dp, dns, fns in os.walk(p):
            dns[:] = sorted(d for d in dns if d != "__pycache__")
            for fn in sorted(fns):
                if fn.endswith((".pyc", ".pyo")):
                    continue
                q = os.path.join(dp, fn)
                out[os.path.relpath(q, task)] = sha256_file(q)
    return out


def check_frozen(sel):
    now = frozen_hashes()
    if sel.get("frozen") != now:
        old = sel.get("frozen") or {}
        diff = sorted(k for k in set(old) | set(now) if old.get(k) != now.get(k))
        raise SystemExit("a frozen file changed since the freeze (%s): record an amendment instead; differing: %s"
                         % (sel.get("frozen_at"), ", ".join(diff[:8]) + (" ..." if len(diff) > 8 else "")))


def candidate_path(cand_dir, k):
    """The candidate's package: hidden_kNN.tar.xz, cNN.tar.xz, or the one <anything>_kNN.tar.xz of the folder."""
    for name in ("hidden_k%02d.tar.xz" % k, "c%02d.tar.xz" % k):
        p = os.path.join(cand_dir, name)
        if os.path.isfile(p):
            return p
    hits = sorted(f for f in os.listdir(cand_dir) if f.endswith("_k%02d.tar.xz" % k)) if os.path.isdir(cand_dir) else []
    if len(hits) > 1:
        raise SystemExit("%s holds more than one package of k = %d: %s" % (cand_dir, k, hits))
    return os.path.join(cand_dir, hits[0]) if hits else None


def load_proofs(path):
    """{k: the generator's record of candidate k} from version 2's JSON lines (one record per line; the last record of
    a k counts), version 1's candidates.json (rows) or the older {"c01": {...}} form."""
    with open(path) as f:
        text = f.read()
    try:
        doc = json.loads(text)
    except ValueError:
        doc = None
    if doc is None or (isinstance(doc, dict) and "k" in doc and "rows" not in doc):
        out = {}
        for line in text.splitlines():
            if line.strip():
                r = json.loads(line)
                out[int(r["k"])] = r
        return out
    out = {}
    if isinstance(doc, dict) and isinstance(doc.get("rows"), list):
        for r in doc["rows"]:
            out[int(r["k"])] = r
    elif isinstance(doc, dict):
        for key, r in doc.items():
            if key.startswith("c") and key[1:].isdigit():
                out[int(key[1:])] = dict(r, k=int(key[1:]))
    return out


def _env_assignments(text):
    """K=V assignments with literal values from shell-style `export K=V ...` / Dockerfile `ENV K=V ...` lines."""
    out = {}
    lines, buf = [], ""
    for raw in text.splitlines():
        s = raw.rstrip()
        if s.endswith("\\"):
            buf += s[:-1] + " "
            continue
        lines.append(buf + s)
        buf = ""
    for line in lines:
        s = line.strip()
        if s.startswith("export "):
            rest = s[len("export "):]
        elif s.startswith("ENV "):
            rest = s[len("ENV "):]
        else:
            continue
        try:
            toks = shlex.split(rest, comments=True)
        except ValueError:
            continue
        for t in toks:
            if "=" not in t:
                continue
            k, v = t.split("=", 1)
            if "$" in v or "`" in v:
                continue
            out[k] = v
    return out


def grader_env(home):
    """The verify stage's environment for the cell's process: the image's ENV (environment/Dockerfile), then
    tests/test.sh's exports, without the build's and root's paths; HOME and the config folders under `home`."""
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "LANG": "C.UTF-8"}
    for src in (os.path.join(TASK, "environment", "Dockerfile"), os.path.join(TESTS, "test.sh")):
        with open(src) as f:
            for k, v in _env_assignments(f.read()).items():
                if k in ENV_SKIP or k.startswith(("ARB_", "ALE_")):
                    continue
                env[k] = v
    env.update({"HOME": home, "XDG_CONFIG_HOME": os.path.join(home, ".config"),
                "XDG_CACHE_HOME": os.path.join(home, ".cache"),
                "MPLCONFIGDIR": os.path.join(home, ".config", "matplotlib")})
    return env


def env_record(env):
    keep = {k: v for k, v in sorted(env.items()) if k not in ("PATH", "HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
                                                                "MPLCONFIGDIR", ENV_MARK)}
    return keep, digest_of(keep)


def machine():
    cpu = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    vers = {}
    for mod in ("mujoco", "numpy", "scipy", "cv2", "PIL", "yaml"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - a record only
            vers[mod] = None
    return {"host": platform.node(), "cpu": cpu, "nproc": os.cpu_count(), "python": platform.python_version(),
            "versions": vers, "in_container": os.path.exists("/.dockerenv")}


def proofs_path(out, sel):
    """The generator's record copied at the freeze (OUT/proofs.json), checked against its recorded sha256."""
    p = os.path.join(out, sel["proofs"]["file"])
    if not os.path.isfile(p) or sha256_file(p) != sel["proofs"]["sha256"]:
        raise SystemExit("the proof record %s is absent or changed since the freeze" % p)
    return p


def load_sel(out):
    p = os.path.join(out, "selection.json")
    if not os.path.isfile(p):
        raise SystemExit("%s does not exist: run `freeze` first" % p)
    return load_json(p), p


# ================================================================================================ rule, freeze
def _k_range_arg(a):
    if getattr(a, "k_range", None):
        lo, _, hi = a.k_range.partition("-")
        return [int(lo), int(hi or lo)]
    return list(K_RANGE_DEFAULT.get(a.label, K_RANGE_DEFAULT["hidden"]))


def exclusion_record(path, tables):
    """The exclusion list's file, its sha256 (it must be the measured list, EXCLUSION_SHA256) and its entries; the
    generator's tables block must carry the same entries (the table was made with this list)."""
    if not path or not os.path.isfile(path):
        raise SystemExit("the exclusion list %s does not exist (R12: it goes into selection.json with its sha256)" % path)
    sha = sha256_file(path)
    if sha != EXCLUSION_SHA256:
        raise SystemExit("%s has sha256 %s, not the measured list's %s (R12: the list stays as measured)"
                         % (path, sha, EXCLUSION_SHA256))
    doc = load_json(path)
    entries = doc.get("exclusion")
    if tables is not None and (tables or {}).get("exclusion") != entries:
        raise SystemExit("the tables block's exclusion list differs from %s: the table was not made with it" % path)
    rel = os.path.relpath(path, TASK) if os.path.abspath(path).startswith(TASK + os.sep) else os.path.basename(path)
    return {"file": rel, "sha256": sha, "entries": entries, "rule": doc.get("rule"), "measured": doc.get("measured"),
            "sub_seed_cap": SUBSEED_CAP}


def table_rows(sel):
    """{k: the table's row} from selection.json's tables block (python -m site_gen.sets --table: {"table": [...]})."""
    t = (sel or {}).get("tables") or {}
    rows = t.get("table") if isinstance(t, dict) else t
    return {int(r["k"]): r for r in (rows or []) if isinstance(r, dict) and "k" in r}


def cmd_rule(a):
    """The rule (with the quota rule and the note on row 3), the label, the k range, the generator's tables and the
    exclusion list with its sha256, written before any seed of the label exists."""
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "selection.json")
    tables = load_json(a.tables) if a.tables else None
    k_range = _k_range_arg(a)
    if a.label == "hidden" and tables is None:
        raise SystemExit("R12: the hidden label's rule needs the generator's tables (--tables)")
    if tables is not None:
        ks = sorted(table_rows({"tables": tables}))
        want = list(range(k_range[0], k_range[1] + 1))
        if ks != want:
            raise SystemExit("the tables block covers k %s..%s (%d rows), the rule's k range is %s"
                             % (ks[0] if ks else None, ks[-1] if ks else None, len(ks), k_range))
    excl = exclusion_record(a.exclusion or EXCLUSION_FILE, tables)
    doc = {"rule": RULE, "label": a.label, "k_range": k_range, "tables": tables,
           "tables_sha256": digest_of(tables) if tables is not None else None, "exclusion": excl,
           "rule_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "candidates": {}, "set": [], "amendments": []}
    if os.path.isfile(p):
        old = load_json(p)
        if any(old.get(k) != doc[k] for k in ("rule", "label", "k_range", "tables", "exclusion")):
            raise SystemExit("%s exists with another rule, label, k range, tables or exclusion list: record an "
                             "amendment instead" % p)
        print("rule already written at %s; nothing changed" % old.get("rule_at"))
        return 0
    dump_json(doc, p)
    print("rule, label %s, k %s, %s and the exclusion list (%s, sha256 %s...) written to %s" % (
        a.label, doc["k_range"], "the tables" if tables else "no tables", excl["file"], excl["sha256"][:8], p))
    return 0


def cmd_freeze(a):
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "selection.json")
    pre = load_json(p) if os.path.isfile(p) else None
    if pre is not None and "frozen" not in pre:
        # `rule` wrote the rule first: the freeze keeps its label, k range and tables and must not change them
        if pre.get("rule") != RULE or pre.get("label") != a.label or list(pre.get("k_range")) != _k_range_arg(a):
            raise SystemExit("%s: the rule, label or k range differ from what `rule` wrote (%s, %s)"
                             % (p, pre.get("label"), pre.get("k_range")))
    label, k_range = a.label, _k_range_arg(a)
    if label == "hidden" and (pre is None or pre.get("tables") is None or pre.get("exclusion") is None):
        raise SystemExit("%s: R12: the hidden label's rule, tables and exclusion list must be written by `rule` before "
                         "any seed of the label is generated; `freeze` found none" % p)
    proofs = load_proofs(a.proofs)
    cands = {}
    for k in range(k_range[0], k_range[1] + 1):
        pkg = candidate_path(a.candidates, k)
        pr = proofs.get(k) or {}
        cands[key_of(k)] = {"k": k, "seed": seed_k(k, label), "package": os.path.basename(pkg) if pkg else None,
                            "sha256": sha256_file(pkg) if pkg else None, "bytes": os.path.getsize(pkg) if pkg else None,
                            "proof_record_sha256": (pr.get("package") or {}).get("sha256"),
                            "record_seed": pr.get("seed")}
        if pr.get("seed") is not None and int(pr["seed"]) != cands[key_of(k)]["seed"]:
            raise SystemExit("%s: the generator's record has seed %s, the label %s gives %s"
                             % (key_of(k), pr["seed"], label, cands[key_of(k)]["seed"]))
    doc = {"rule": RULE, "label": label, "k_range": k_range,
           "tables": (pre or {}).get("tables"), "tables_sha256": (pre or {}).get("tables_sha256"),
           "exclusion": (pre or {}).get("exclusion"), "rule_at": (pre or {}).get("rule_at"),
           "frozen": frozen_hashes(), "candidates_frozen": cands,
           "proofs": {"file": "proofs.json", "source": os.path.basename(a.proofs), "sha256": sha256_file(a.proofs)},
           "environment": dict(zip(("vars", "digest"), env_record(grader_env("/nonexistent")))),
           "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "frozen_on": {k: v for k, v in machine().items() if k != "host"},     # no host name in the task's record
           "candidates": {}, "set": [], "amendments": []}
    if pre is not None and "frozen" in pre:
        old = pre
        if old.get("frozen") != doc["frozen"] or old.get("candidates_frozen") != cands or old.get("rule") != RULE:
            raise SystemExit("%s exists with other frozen files, candidates or rule: record an amendment instead" % p)
        print("already frozen at %s (%d files, %d candidates); nothing changed" % (old.get("frozen_at"),
                                                                                  len(old["frozen"]), len(cands)))
        return 0
    shutil.copyfile(a.proofs, os.path.join(a.out, "proofs.json"))      # the record travels with the selection
    dump_json(doc, p)
    print("rule, %d frozen files and %d candidates' sha256 written to %s" % (len(doc["frozen"]), len(cands), p))
    return 0


# ================================================================================================ one mission
def _cell_failed(s):
    """verify.py's _cell_failed: what went wrong on the cell's side of a mission ([] when nothing); an internal error
    counts only while the mission was live (internal_errors_live)."""
    out = []
    try:
        live = s.get("internal_errors_live")
        if int((s.get("internal_errors", 0) if live is None else live) or 0) > 0:
            out.append("internal_error")
    except (TypeError, ValueError):
        pass
    if s.get("done_reason") == "physics_fault" or s.get("end_reason") == "physics_fault":
        out.append("physics_fault")
    try:
        if int(s.get("frames_lost", 0) or 0) > 0:
            out.append("render_failure")
    except (TypeError, ValueError):
        pass
    return out


def _compact(s):
    keep = ("episode_score", "R", "parts", "done_reason", "end_reason", "sim_time_s", "sim_time_limit_s",
            "dose_fraction", "dose_gy", "frames", "reports", "reports_made", "reports_delivered", "end",
            "illegal_call_count", "internal_errors", "internal_errors_live", "frames_lost", "render_failures", "render_retries",
            "final_state_sha256", "model_hash", "site_sha256", "noise_seed", "episode_wall_s", "program_time_s",
            "cell_work_s", "physics_wall_s", "render_wall_s", "build_wall_s", "cell_build_s", "distance_m",
            "event_counts", "reference")
    out = {k: s.get(k) for k in keep if k in s}
    out["items"] = [{k: it.get(k) for k in ("tag", "type", "slot", "delivered", "E", "V", "A", "error", "tolerance",
                                            "credit", "evidence_ok", "evidence_why", "dwell_s", "max_3s_reading_gyh")
                     if k in it} for it in s.get("items", [])]
    return out


def cmd_run_one(a):
    if os.environ.get(ENV_MARK) != "1":                    # into the verify stage's environment before numpy loads
        home = tempfile.mkdtemp(prefix="select_home_")
        env = grader_env(home)
        env[ENV_MARK] = "1"
        argv = [sys.executable, "-B", "-s", HERE] + sys.argv[1:]
        os.execve(sys.executable, argv, env)
    env_vars, env_digest = env_record(dict(os.environ))     # before any import adds to it (cv2 sets LD_LIBRARY_PATH)
    sel, _ = load_sel(a.out)
    check_frozen(sel)
    k, prog = int(a.k), a.prog
    key = key_of(k)
    fz = (sel.get("candidates_frozen") or {}).get(key) or {}
    pkg = candidate_path(a.candidates, k)
    if pkg is None or sha256_file(pkg) != fz.get("sha256"):
        raise SystemExit("%s: the package in %s is absent or not the frozen bytes" % (key, a.candidates))
    sys.path.insert(0, os.path.join(REVIEW, "cell"))
    sys.path.insert(0, TESTS)
    sys.path.insert(0, REVIEW)
    import run_wire                        # noqa: E402  (tests/review/cell/run_wire.py)
    import baselines as B                  # noqa: E402  (tests/review/baselines.py)
    root = os.path.join(a.out, "runs", key, prog)
    os.makedirs(root, exist_ok=True)
    work = tempfile.mkdtemp(prefix="select_%s_%s_" % (key, prog))
    if prog == "oracle":
        sub = os.path.join(work, "submission")
        os.makedirs(sub)
        shutil.copyfile(os.path.join(TASK, "solution", "reference", "patrol.py"), os.path.join(sub, "patrol.py"))
        privileged = True
    elif prog in B.NAMES:
        sub = B.make(prog, os.path.join(work, "submission"))
        privileged = False
    else:
        raise SystemExit("unknown program %r (oracle | %s)" % (prog, ", ".join(B.NAMES)))
    t0 = time.time()
    attempts = []
    lines = []
    log = lambda m: lines.append(str(m))   # noqa: E731
    sk = seed_k(k, label_of(sel))
    for i in (1, 2):
        s = run_wire.run(pkg, sub, os.path.join(root, "a%d" % i), privileged=privileged, seed=sk, log=log)
        rec = _compact(s)
        rec["cell_failed"] = _cell_failed(s)
        if privileged:
            rec["plates"] = plates_before_reports(os.path.join(root, "a%d" % i), s)
        attempts.append(rec)
        if not rec["cell_failed"]:
            break
    task_error = len(attempts) == 2 and "render_failure" in attempts[1]["cell_failed"]
    counted = attempts[-1]
    with open(os.path.join(root, "cell.log"), "w") as f:
        f.write("\n".join(lines[-4000:]) + "\n")
    result = {"k": k, "candidate": key, "prog": prog, "privileged": privileged, "seed": sk, "label": label_of(sel),
              "package": os.path.basename(pkg), "package_sha256": fz["sha256"],
              "frozen_digest": digest_of(sel["frozen"]), "env": env_vars, "env_digest": env_digest,
              "machine": machine(), "attempts": attempts, "counted_attempt": len(attempts), "task_error": task_error,
              "score": float(counted.get("episode_score", 0.0)), "job_wall_s": round(time.time() - t0, 1),
              "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    dump_json(result, os.path.join(root, "result.json"))
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({"k": k, "prog": prog, "score": result["score"], "sim_time_s": counted.get("sim_time_s"),
                      "attempts": len(attempts), "final_state_sha256": counted.get("final_state_sha256"),
                      "job_wall_s": result["job_wall_s"]}), flush=True)
    return 0


# ================================================================================================ the pool
def _parse_k(text):
    """k values from "a-b", "k" or a comma list of both ("1-16,20,33-48"), in k order without repeats."""
    out = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


def plates_before_reports(run_dir, s):
    """Row 2's plate condition (version 2) from the reference's own log of one attempt: for each instrument of
    the work order, a `plate_read` whose text equals the tag (ok) no later than the item's report."""
    p = os.path.join(run_dir, "reference_log.json")
    out = {"log": os.path.isfile(p), "instruments": 0, "read_before_report": 0, "missing": [], "reported_unread": []}
    if not out["log"]:
        return out
    lg = load_json(p)
    log = lg.get("log") or []
    first_ok, reported = {}, {}
    for n in log:
        if n.get("kind") == "plate_read" and n.get("ok") and n.get("tag") not in first_ok:
            first_ok[n["tag"]] = float(n.get("t", 0.0))
        if n.get("kind") in ("report", "report_no_value") and n.get("tag") not in reported:
            reported[n["tag"]] = float(n.get("t", 0.0))
    inst = [it for it in s.get("items", []) if it.get("type") not in ("dose_point", "hotspot")]
    out["instruments"] = len(inst)
    for it in inst:
        tag = it.get("tag")
        if tag in reported and tag in first_ok and first_ok[tag] <= reported[tag] + 1e-6:
            out["read_before_report"] += 1
        elif tag in reported:
            out["reported_unread"].append(tag)
        else:
            out["missing"].append(tag)
    return out


def reference_rows(res, cfg):
    """Rows 2 and 3 from the oracle mission's record (the counted attempt)."""
    s = res["attempts"][res["counted_attempt"] - 1]
    plates = s.get("plates") or {}
    items = s.get("items", [])
    instruments = [it for it in items if it.get("type") not in ("dose_point", "hotspot")]
    row2 = {"all_delivered": bool(items) and all(it.get("delivered") for it in items),
            "n_items": len(items),
            "evidence_all_instruments": bool(instruments) and all(it.get("evidence_ok") for it in instruments),
            "mark_dwell_ok": all(bool(it.get("E")) for it in items if it.get("type") == "dose_point")
            and any(it.get("type") == "dose_point" for it in items),
            "source_approach_ok": all(bool(it.get("E")) for it in items if it.get("type") == "hotspot")
            and any(it.get("type") == "hotspot" for it in items),
            "R": float(s.get("R", 0.0)), "upright_on_pad": float(s.get("R", 0.0)) == 1.0,
            "S": float(s.get("episode_score", 0.0)), "task_error": bool(res.get("task_error")),
            "plates_read_before_report": int(plates.get("read_before_report", 0)),
            "plates_instruments": int(plates.get("instruments", 0)),
            "plates_ok": bool(plates.get("log")) and int(plates.get("instruments", 0)) == N_INSTRUMENTS
            and int(plates.get("read_before_report", 0)) == N_INSTRUMENTS}
    row2["ok"] = bool(row2["all_delivered"] and row2["n_items"] == 10 and row2["evidence_all_instruments"]
                      and row2["mark_dwell_ok"] and row2["source_approach_ok"] and row2["upright_on_pad"]
                      and row2["S"] >= S_MIN and not row2["task_error"] and row2["plates_ok"])
    lim = (cfg.get("budgets") or {}).get("frames") or {}
    sim_lim = float(s.get("sim_time_limit_s") or (cfg.get("time") or {}).get("sim_time_limit_s", 2400.0))
    frames = s.get("frames") or {}
    row3 = {"sim_time_s": float(s.get("sim_time_s", 0.0)), "sim_time_limit_s": sim_lim,
            "sim_time_max_s": SIM_TIME_MAX_S,
            "sim_share": round(float(s.get("sim_time_s", 0.0)) / sim_lim, 4),
            "dose_share": round(float(s.get("dose_fraction", 0.0)), 4),
            "frame_shares": {k: round(float(frames.get(k, 0)) / float(v), 4) for k, v in sorted(lim.items())
                             if float(v) > 0}}
    # the time part is the absolute bound (sim_share stays in the record as the share of the mission's limit)
    row3["ok"] = bool(row3["sim_time_s"] <= SIM_TIME_MAX_S and row3["dose_share"] <= DOSE_SHARE_MAX
                      and all(x <= FRAME_SHARE_MAX for x in row3["frame_shares"].values()))
    return row2, row3


def row4_of(results):
    out = {"scores": {}, "missing": []}
    ok = True
    for name in ROW4:
        r = results.get(name)
        if r is None:
            out["missing"].append(name)
            ok = False
            continue
        sc = float(r["score"])
        out["scores"][name] = sc
        if r.get("task_error"):
            ok = False
        elif name == "corridor_only":
            ok = ok and sc <= CORRIDOR_MAX + 1e-9
        else:
            ok = ok and sc < 0.005                    # rounds to 0.00 at two decimals
    out["ok"] = None if out["missing"] else bool(ok)          # None: not decided yet (a record is missing)
    return out


def _result(out, k, prog):
    p = os.path.join(out, "runs", key_of(k), prog, "result.json")
    return load_json(p) if os.path.isfile(p) else None


def cmd_run(a):
    sel, _ = load_sel(a.out)
    check_frozen(sel)
    cfg = load_json(os.path.join(TESTS, "grader_config.json"))
    proofs = load_proofs(proofs_path(a.out, sel))
    ks = _parse_k(a.k) if a.k else list(range(k_range_of(sel)[0], k_range_of(sel)[1] + 1))
    jobs = []
    for k in ks:
        r1 = row1_of(proofs.get(k), sel["candidates_frozen"].get(key_of(k)) or {})
        if not r1["ok"]:
            print("c%02d: row 1 fails (%s): not run" % (k, r1), flush=True)
            continue
        jobs.append((k, "oracle"))
    if a.baselines == "all":
        jobs += [(k, b) for k in ks for b in ROW4 if (k, "oracle") in jobs]
    log_p = os.path.join(a.out, "jobs.jsonl")
    lock = threading.Lock()
    pending = [j for j in jobs if _result(a.out, *j) is None]
    print("%d jobs, %d to run, %d workers" % (len(jobs), len(pending), a.workers), flush=True)
    queue = list(pending)
    t_start = time.time()

    def log(rec):
        with lock:
            with open(log_p, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def worker(wid):
        while True:
            with lock:
                if not queue:
                    return
                k, prog = queue.pop(0)
            jdir = os.path.join(a.out, "runs", key_of(k), prog)
            os.makedirs(jdir, exist_ok=True)
            t0 = time.time()
            log({"event": "start", "k": k, "prog": prog, "worker": wid, "t": round(t0 - t_start, 1)})
            with open(os.path.join(jdir, "job.log"), "w") as lf:
                rc = subprocess.call([sys.executable, "-B", "-s", HERE, "run-one", "--candidates", a.candidates,
                                      "--out", a.out, "--k", str(k), "--prog", prog],
                                     stdout=lf, stderr=subprocess.STDOUT, cwd=tempfile.gettempdir(),
                                     env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")})
            res = _result(a.out, k, prog)
            rec = {"event": "end", "k": k, "prog": prog, "worker": wid, "rc": rc, "wall_s": round(time.time() - t0, 1),
                   "t": round(time.time() - t_start, 1), "score": res["score"] if res else None}
            log(rec)
            print(json.dumps(rec), flush=True)
            if a.baselines == "admissible" and prog == "oracle" and res is not None:
                r2, r3 = reference_rows(res, cfg)
                if r2["ok"] and r3["ok"]:
                    with lock:
                        queue.extend((k, b) for b in ROW4 if _result(a.out, k, b) is None)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(max(1, int(a.workers)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = round(time.time() - t_start, 1)
    log({"event": "pool_done", "wall_s": wall, "jobs": len(jobs), "ran": len(pending)})
    print("pool done in %.0f s (%d jobs run)" % (wall, len(pending)), flush=True)
    if label_of(sel) == "practice":
        print("practice label: R12's quota rule is the hidden set's; the practice buildings are chosen by `practice` "
              "(F-1)", flush=True)
        return 0
    return cmd_decide(a)


# ================================================================================================ the rule
def row1_of(pr, fz):
    """Row 1 from the generator's record. Version 2's records carry generated, proofs_ok and failed_rows (every
    required row, the new ones included, in memory at generation); a recheck from the stored files counts where the
    record has one (version 1's records always had)."""
    pr = pr or {}
    rc = pr.get("recheck_from_files")
    rec = {"generated": bool(pr.get("generated")), "proofs_ok": bool(pr.get("proofs_ok")),
           "failed_rows": list(pr.get("failed_rows") or []),
           "recheck_from_files_ok": None if rc is None else bool((rc or {}).get("required_ok")),
           "record_for_these_bytes": bool(fz.get("sha256")) and (pr.get("package") or {}).get("sha256") == fz.get("sha256"),
           "attempts": pr.get("attempts"),
           "documents_failed": list(pr.get("documents_failed") or []),
           "access_and_poses_ok": pr.get("access_and_poses_ok"),
           "independent_check_ok": (rc or {}).get("indep_ok")}
    rec["ok"] = bool(rec["generated"] and rec["proofs_ok"] and not rec["failed_rows"]
                     and rec["recheck_from_files_ok"] is not False and rec["record_for_these_bytes"])
    return rec


def features_of(pr):
    """The generator's description of a candidate (sitegen's record), flattened for the hollowing table."""
    pr = pr or {}
    f = {}
    for k, v in (pr.get("features") or {}).items():
        if isinstance(v, list):
            for x in v:
                f["features.%s=%s" % (k, x)] = True
        else:
            f["features." + k] = v
    for k, v in (pr.get("route") or {}).items():
        if k == "modes_used":
            for x in v:
                f["route.mode=%s" % x] = True
        else:
            f["route." + k] = v
    for slot, typ in (pr.get("assignment") or {}).items():
        f["slot.%s" % slot] = typ
    for d in pr.get("deviations") or []:
        f["deviation=%s" % d] = True
    if pr.get("attempts") is not None:
        f["attempts"] = pr["attempts"]
    # version 2 (the hollowing check): the table's columns as built, the missing sheet and what it holds, C-sealed
    for k in ("bund", "b_step_variant", "b_low_variant", "sheet_place", "missing_sheet", "sheet_count", "missing_rooms",
              "c_sealed_way", "corridor_sill_kind"):
        if k in pr:
            f["v2.%s" % k] = pr[k]
    for slot in pr.get("sheet_slots") or []:
        f["v2.on_missing_sheet=%s" % slot] = True
    if isinstance(pr.get("table"), dict):
        for k in ("bund", "b_step", "b_low", "sheet_place"):
            if k in pr["table"]:
                f["table.%s" % k] = pr["table"][k]
    if isinstance(pr.get("obstacles"), list):
        f["v2.obstacles_off_drawing"] = len(pr["obstacles"])
    return f


def hollowing(feats, set_keys):
    """Per feature: numeric -> mean / min / max of the set and of all; categorical -> counts in the set and in all."""
    table = {}
    names = sorted({n for v in feats.values() for n in v})
    for n in names:
        allv = [v.get(n) for v in feats.values()]
        setv = [feats[k].get(n) for k in set_keys if k in feats]
        if all(isinstance(x, bool) or x is None for x in allv):
            table[n] = {"kind": "flag", "set": sum(1 for x in setv if x), "all": sum(1 for x in allv if x),
                        "n_set": len(setv), "n_all": len(allv)}
        elif all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in allv if x is not None):
            num = lambda xs: [float(x) for x in xs if x is not None]  # noqa: E731
            sa, ss = num(allv), num(setv)
            table[n] = {"kind": "number",
                        "set": {"mean": round(sum(ss) / len(ss), 4) if ss else None, "min": min(ss) if ss else None,
                                "max": max(ss) if ss else None},
                        "all": {"mean": round(sum(sa) / len(sa), 4) if sa else None, "min": min(sa) if sa else None,
                                "max": max(sa) if sa else None},
                        "n_set": len(ss), "n_all": len(sa)}
        else:
            cnt = lambda xs: {str(x): sum(1 for y in xs if str(y) == str(x)) for x in sorted(set(map(str, xs)))}  # noqa: E731
            table[n] = {"kind": "category", "set": cnt(setv), "all": cnt(allv), "n_set": len(setv), "n_all": len(allv)}
    return table


def quota_values(pr, row):
    """The candidate's bund kerb class, B-step variant, B-low variant and sheet place as built (the generator's
    record), each checked against the table's row for k; a value the record lacks is taken from the table (and said
    so)."""
    pr, row = pr or {}, row or {}
    vals, table, diff, from_table = {}, {}, [], []
    for q, rk in QUOTA_RECORD_KEYS.items():
        tv = row.get(q)
        table[q] = tv
        v = pr.get(rk)
        if v is None:
            v = (pr.get("table") or {}).get(q, tv)
            from_table.append(q)
        vals[q] = v
        if tv is not None and v is not None and str(v) != str(tv):
            diff.append(q)
    return {"values": vals, "table": table, "differs_from_table": diff, "taken_from_table": from_table}


def quota_select(cands):
    """The quota rule on candidates already put through the admission rows: `cands` = [{"k", "admitted": True | False | None
    (None: not decided, a record is missing), "quota": {"bund", "b_step", "b_low", "sheet_place"}}] in any order;
    they are taken in k order. Returns {"set": [k ...], "status": {k: text}, "counts": the set's counts per quota,
    "skipped": [k ...] admitted but over a quota, "complete": bool, "stopped_at": k of an undecided candidate that
    stopped the rule or None}."""
    counts = {q: {} for q in QUOTA}
    chosen, status, skipped, stopped = [], {}, [], None
    for c in sorted(cands, key=lambda c: int(c["k"])):
        k = int(c["k"])
        if stopped is not None:
            status[k] = "not decided (an earlier candidate is undecided)"
            continue
        full = len(chosen) >= N_SET
        adm = c.get("admitted")
        if adm is None:
            if full:
                status[k] = "not decided (the set was complete)"
                continue
            status[k] = "not decided (missing records)"
            stopped = k
            continue
        if not adm:
            status[k] = "not admitted"
            continue
        if full:
            status[k] = "admitted, not taken (the set was complete)"
            continue
        vals = c.get("quota") or {}
        missing = [q for q in QUOTA if vals.get(q) is None]
        if missing:
            raise SystemExit("c%02d: no %s in its records: the quota rule cannot be applied" % (k, ", ".join(missing)))
        over = [q for q in QUOTA if counts[q].get(str(vals[q]), 0) + 1 > QUOTA[q]]
        if over:
            status[k] = "admitted, skipped by the quota (it would be the %s)" % " and the ".join(
                "%s with its %s %s" % (ORDINAL[QUOTA[q] + 1], QUOTA_WORDS[q], vals[q]) for q in over)
            skipped.append(k)
            continue
        chosen.append(k)
        for q in QUOTA:
            counts[q][str(vals[q])] = counts[q].get(str(vals[q]), 0) + 1
        status[k] = "admitted, in the set (h%02d)" % len(chosen)
    return {"set": chosen, "status": status, "counts": counts, "skipped": skipped,
            "complete": len(chosen) >= N_SET, "stopped_at": stopped}


def cmd_decide(a):
    sel, sel_p = load_sel(a.out)
    check_frozen(sel)
    cfg = load_json(os.path.join(TESTS, "grader_config.json"))
    proofs = load_proofs(proofs_path(a.out, sel))
    trows = table_rows(sel)
    cands, feats, qin = {}, {}, []
    K_RANGE = k_range_of(sel)
    # 1. every candidate through the admission rows on its own (as before)
    for k in range(K_RANGE[0], K_RANGE[1] + 1):
        key = key_of(k)
        fz = sel["candidates_frozen"].get(key) or {}
        pr = proofs.get(k)
        rec = {"k": k, "seed": seed_k(k, label_of(sel)), "package": fz.get("package"), "sha256": fz.get("sha256"),
               "row1": row1_of(pr, fz), "quota": quota_values(pr, trows.get(k))}
        feats[key] = features_of(pr)
        res = _result(a.out, k, "oracle")
        if res is not None:
            if res.get("frozen_digest") != digest_of(sel["frozen"]) or res.get("package_sha256") != fz.get("sha256"):
                raise SystemExit("%s: the oracle record was not made with the frozen files and package" % key)
            c = res["attempts"][res["counted_attempt"] - 1]
            rec["reference"] = {"S": res["score"], "R": c.get("R"), "parts": c.get("parts"),
                                "sim_time_s": c.get("sim_time_s"), "dose_fraction": c.get("dose_fraction"),
                                "frames": c.get("frames"), "done_reason": c.get("done_reason"),
                                "end_reason": c.get("end_reason"), "final_state_sha256": c.get("final_state_sha256"),
                                "attempts": len(res["attempts"]), "cell_failed": [x["cell_failed"] for x in res["attempts"]],
                                "job_wall_s": res.get("job_wall_s"), "cpu": (res.get("machine") or {}).get("cpu"),
                                "env_digest": res.get("env_digest"), "plates": c.get("plates"),
                                "blocked_holds": (c.get("reference") or {}).get("blocked_holds")}
            rec["row2"], rec["row3"] = reference_rows(res, cfg)
            feats[key].update({"run.reference_S": res["score"], "run.sim_share": rec["row3"]["sim_share"],
                               "run.dose_share": rec["row3"]["dose_share"],
                               "run.blocked_holds": (c.get("reference") or {}).get("blocked_holds")})
        bl = {}
        for name in ROW4:
            r = _result(a.out, k, name)
            if r is not None:
                if r.get("frozen_digest") != digest_of(sel["frozen"]):
                    raise SystemExit("%s: the %s record was not made with the frozen files" % (key, name))
                bl[name] = r
        if bl:
            rec["row4"] = row4_of(bl)
        rows_ok = [rec["row1"]["ok"], (rec.get("row2") or {}).get("ok"), (rec.get("row3") or {}).get("ok"),
                   (rec.get("row4") or {}).get("ok")]
        failed = [str(i + 1) for i, v in enumerate(rows_ok) if v is False]
        if failed:
            rec["admitted"] = False
            rec["admission"] = "dropped (row %s)" % ", ".join(failed)
        elif all(v is True for v in rows_ok):
            rec["admitted"] = True
            rec["admission"] = "admitted"
        else:
            rec["admitted"] = None
            rec["admission"] = "not decided (missing records: %s)" % ", ".join(
                n for n, v in zip(("row1", "row2", "row3", "row4"), rows_ok) if v is None)
        rec["features"] = (pr or {}).get("features")
        cands[key] = rec
        qin.append({"k": k, "admitted": rec["admitted"], "quota": rec["quota"]["values"]})
    # 2. The quota rule over the admitted candidates in k order
    q = quota_select(qin)
    for k, st in q["status"].items():
        rec = cands[key_of(k)]
        rec["status"] = rec["admission"] if st == "not admitted" else st
        rec["in_set"] = k in q["set"]
    chosen = [key_of(k) for k in q["set"]]
    sel["candidates"] = cands
    sel["set"] = chosen
    sel["complete"] = bool(q["complete"])
    sel["quota_result"] = {"set_k": q["set"], "counts_in_set": q["counts"], "skipped_by_quota": q["skipped"],
                           "stopped_at_undecided": q["stopped_at"], "limits": dict(QUOTA),
                           "admitted": [c["k"] for c in qin if c["admitted"]],
                           "table_differences": {key_of(c["k"]): cands[key_of(c["k"])]["quota"]["differs_from_table"]
                                                 for c in qin if cands[key_of(c["k"])]["quota"]["differs_from_table"]}}
    sel["hollowing"] = {"note": "the twelve against all candidates with a generator record (and, for run.*, all "
                                "candidates whose reference ran); the table's columns are v2.bund, v2.b_step_variant, "
                                "v2.b_low_variant, v2.sheet_place (as built) and table.* (as planned)",
                        "set": chosen, "table": hollowing(feats, chosen)}
    jl = os.path.join(a.out, "jobs.jsonl")
    if os.path.isfile(jl):
        evs = [json.loads(x) for x in open(jl) if x.strip()]
        done = [e for e in evs if e.get("event") == "pool_done"]
        sel["selection_run"] = {"pool_runs": done, "jobs_ended": sum(1 for e in evs if e.get("event") == "end")}
    sel["decided_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dump_json(sel, sel_p)
    for key in sorted(cands):
        c = cands[key]
        ref = c.get("reference") or {}
        qv = (c.get("quota") or {}).get("values") or {}
        print("%s %-60s %-30s S %-6s sim %-7s dose %-6s row4 %s" % (
            key, c.get("status"), "/".join(str(qv.get(x)) for x in QUOTA), ref.get("S"),
            (c.get("row3") or {}).get("sim_share"), (c.get("row3") or {}).get("dose_share"),
            (c.get("row4") or {}).get("scores")))
    if not sel["complete"]:
        print("fewer than %d candidates in the set (%d: %s): stop and report; the rule is not changed"
              % (N_SET, len(chosen), chosen))
        return 1
    print("set:", chosen, "counts:", q["counts"])
    return 0


# ================================================================================================ practice
def cmd_practice(a):
    """The practice rule over the decided records of a practice label: qualifying candidates (row 1; every item delivered with
    accepted evidence; S >= 0.80), then the generator's own triple rule (site_gen.variants.select_practice from the
    task's generator copy tests/review/sitegen/site). Writes OUT/practice.json; with --write DIR the practice ground's
    folder (hall.tar.xz from --hall, t1..t3.tar.xz byte copies of the chosen packages, index.json)."""
    sel, _ = load_sel(a.out)
    check_frozen(sel)
    cfg = load_json(os.path.join(TESTS, "grader_config.json"))
    proofs = load_proofs(proofs_path(a.out, sel))
    sys.path.insert(0, os.path.join(REVIEW, "sitegen", "site"))
    from site_gen import variants as V          # noqa: E402  (the generator's practice rule, the task's copy)
    lo, hi = k_range_of(sel)
    records, rows = [], {}
    for k in range(lo, hi + 1):
        key = key_of(k)
        fz = sel["candidates_frozen"].get(key) or {}
        pr = dict(proofs.get(k) or {"k": k})
        r1 = row1_of(proofs.get(k), fz)
        res = _result(a.out, k, "oracle")
        row = {"k": k, "row1": r1["ok"]}
        if res is not None:
            if res.get("frozen_digest") != digest_of(sel["frozen"]) or res.get("package_sha256") != fz.get("sha256"):
                raise SystemExit("%s: the oracle record was not made with the frozen files and package" % key)
            r2, r3 = reference_rows(res, cfg)
            delivered = bool(r2["all_delivered"] and r2["n_items"] == 10 and r2["evidence_all_instruments"]
                             and r2["mark_dwell_ok"] and r2["source_approach_ok"])
            pr["reference"] = {"all_delivered": delivered, "score": float(res["score"])}
            row.update(reference_S=res["score"], delivered_with_evidence=delivered, plates_ok=r2["plates_ok"],
                       sim_share=r3["sim_share"], dose_share=r3["dose_share"])
        if not r1["ok"]:
            pr["generated"] = False          # the practice rule asks row 1 of the record as the selection reads it
        row["qualifies"] = bool(V.qualifies(pr))
        row["features"] = {k2: (sorted(v) if isinstance(v, set) else v) for k2, v in V.practice_features(pr).items()}
        rows[key] = row
        records.append(pr)
    pick = V.select_practice(records)
    doc = {"rule": RULE["practice"], "label": label_of(sel), "k_range": [lo, hi], "candidates": rows,
           "t": None if pick.get("k") is None else list(pick["k"]),
           "passed_over": pick.get("passed_over"), "not_qualified": pick.get("not_qualified"),
           "classes": pick.get("classes"), "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if pick.get("k") is not None:
        doc["sites"] = []
        for i, k in enumerate(pick["k"]):
            fz = sel["candidates_frozen"][key_of(k)]
            doc["sites"].append({"name": "t%d" % (i + 1), "k": k, "seed": fz["seed"], "package": fz["package"],
                                 "sha256": fz["sha256"]})
    dump_json(doc, os.path.join(a.out, "practice.json"))
    for key in sorted(rows):
        r = rows[key]
        print("%s qualifies %-5s S %-6s %s" % (key, r["qualifies"], r.get("reference_S"), r["features"]))
    if pick.get("k") is None:
        print("no triple of the qualifying candidates meets F-1: stop and report; the rule is not changed")
        return 1
    print("practice t1, t2, t3 = k %s; passed over %s" % (list(pick["k"]), pick.get("passed_over")))
    if a.write:
        if not (a.candidates and a.hall):
            raise SystemExit("--write needs --candidates and --hall")
        os.makedirs(a.write, exist_ok=True)
        idx = {"rule": RULE["practice"], "label": label_of(sel), "sites": []}
        shutil.copyfile(a.hall, os.path.join(a.write, "hall.tar.xz"))
        idx["hall"] = {"package": "hall.tar.xz", "sha256": sha256_file(os.path.join(a.write, "hall.tar.xz"))}
        for st in doc["sites"]:
            src = candidate_path(a.candidates, st["k"])
            dst = os.path.join(a.write, st["name"] + ".tar.xz")
            shutil.copyfile(src, dst)
            if sha256_file(dst) != st["sha256"]:
                raise SystemExit("%s: the copy's sha256 differs from the frozen package" % st["name"])
            idx["sites"].append(dict(st, package=st["name"] + ".tar.xz"))
        idx["passed_over"] = doc["passed_over"]
        dump_json(idx, os.path.join(a.write, "index.json"))
        print("practice ground packages written to %s" % a.write)
    return 0


# ================================================================================================ the hidden set
def cmd_write_hidden(a):
    sel, sel_p = load_sel(a.out)
    check_frozen(sel)
    if not sel.get("complete") or len(sel.get("set") or []) < N_SET:
        raise SystemExit("the selection is not complete (%s): nothing written" % sel.get("set"))
    os.makedirs(os.path.join(a.hidden, "sites"), exist_ok=True)
    for f in os.listdir(os.path.join(a.hidden, "sites")):
        if f.endswith(".tar.xz"):
            os.remove(os.path.join(a.hidden, "sites", f))
    sites, seeds = [], []
    for i, key in enumerate(sel["set"][:N_SET]):
        fz = sel["candidates_frozen"][key]
        src = candidate_path(a.candidates, fz["k"]) if a.candidates else None
        if src is None:
            raise SystemExit("%s: --candidates must hold the frozen package" % key)
        tag = "h%02d" % (i + 1)
        dst = os.path.join(a.hidden, "sites", tag + ".tar.xz")
        shutil.copyfile(src, dst)
        got = sha256_file(dst)
        if got != fz["sha256"]:
            raise SystemExit("%s: the copy's sha256 differs from the frozen package" % key)
        seeds.append(fz["seed"])
        sites.append({"tag": tag, "archive": "sites/%s.tar.xz" % tag, "sha256": got, "seed": fz["seed"],
                      "candidate": key, "k": fz["k"], "bytes": os.path.getsize(dst)})
    doc = {"seeds": seeds, "sites": sites,
           "note": "the fixed hidden set: the twelve candidates the pre-registered rule of selection.json chose (R26: "
                   "R12's quota rule over the admitted candidates in k order with every quota doubled), in k order "
                   "(episode order); each archive is a byte copy of the frozen candidate package; seed = the "
                   "candidate's seed_k, the mission's noise seed"}
    dump_json(doc, os.path.join(a.hidden, "seeds.json"))
    dump_json(sel, os.path.join(a.hidden, "selection.json"))
    print("hidden set written to %s: %s" % (a.hidden, ", ".join("%s=%s" % (s["tag"], s["candidate"]) for s in sites)))
    return 0


# ================================================================================================ report
def cmd_report(a):
    """The admission table of a decided selection (markdown): one row per candidate with every admission row."""
    sel, _ = load_sel(a.out)
    proofs = {}
    try:
        proofs = load_proofs(proofs_path(a.out, sel))
    except SystemExit:
        pass
    lines = ["| k | bund | B-step | B-low | sheet | on sheet | C-sealed | row 1 | S | items + evidence | plates | "
             "time | dose | max frame | holds | degenerate max | corridor | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(sel.get("candidates") or {}):
        c = sel["candidates"][key]
        pr = proofs.get(c.get("k")) or {}
        r1, r2, r3, r4 = (c.get("row1") or {}), (c.get("row2") or {}), (c.get("row3") or {}), (c.get("row4") or {})
        sc = (r4.get("scores") or {})
        deg = [v for n, v in sc.items() if n != "corridor_only"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            c.get("k"), pr.get("bund"), pr.get("b_step_variant"), pr.get("b_low_variant"), pr.get("sheet_place"),
            pr.get("sheet_count"), pr.get("c_sealed_way"),
            "ok" if r1.get("ok") else ("fail" if r1 else ""), r2.get("S"),
            "" if not r2 else ("yes" if (r2.get("all_delivered") and r2.get("evidence_all_instruments") and
                                         r2.get("mark_dwell_ok") and r2.get("source_approach_ok")) else "no"),
            "" if not r2 else "%s / %s" % (r2.get("plates_read_before_report"), r2.get("plates_instruments")),
            r3.get("sim_share"), r3.get("dose_share"),
            max((r3.get("frame_shares") or {}).values()) if r3.get("frame_shares") else None,
            ((c.get("reference") or {}).get("blocked_holds")), max(deg) if deg else None, sc.get("corridor_only"),
            c.get("status")))
    qr = sel.get("quota_result") or {}
    if qr:
        lines += ["", "R12 (the quota rule over the admitted candidates in k order): set k %s; counts in the set %s; "
                  "admitted %s; skipped by a quota %s%s" % (
                      qr.get("set_k"), json.dumps(qr.get("counts_in_set"), sort_keys=True), qr.get("admitted"),
                      qr.get("skipped_by_quota"), "; stopped at the undecided k %s" % qr["stopped_at_undecided"]
                      if qr.get("stopped_at_undecided") is not None else ""),
                  "", "R13: " + ((sel.get("rule") or {}).get("notes") or {}).get("R13", "")]
    text = "\n".join(lines) + "\n"
    print(text)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text)
    return 0


# ================================================================================================ compare
def cmd_compare(a):
    ra, rb = load_json(a.a), load_json(a.b)
    ca, cb = ra["attempts"][ra["counted_attempt"] - 1], rb["attempts"][rb["counted_attempt"] - 1]
    rows = []
    for k in ("episode_score", "R", "parts", "sim_time_s", "dose_fraction", "frames", "reports",
              "final_state_sha256", "done_reason"):
        rows.append({"key": k, "same": ca.get(k) == cb.get(k), "a": ca.get(k), "b": cb.get(k)})
    ia = [(it.get("tag"), it.get("credit"), it.get("error")) for it in ca.get("items", [])]
    ib = [(it.get("tag"), it.get("credit"), it.get("error")) for it in cb.get("items", [])]
    rows.append({"key": "items (tag, credit, error)", "same": ia == ib, "a": ia, "b": ib})
    rows.append({"key": "env_digest", "same": ra.get("env_digest") == rb.get("env_digest"), "a": ra.get("env_digest"),
                 "b": rb.get("env_digest")})
    out = {"a": a.a, "b": a.b, "machine_a": ra.get("machine"), "machine_b": rb.get("machine"),
           "identical": all(r["same"] for r in rows), "rows": rows}
    print(json.dumps(out, indent=1, default=str))
    return 0 if out["identical"] else 1


# ================================================================================================ main
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--freeze-only" in argv:                    # the older form: select_set.py --freeze-only --candidates ...
        argv = ["freeze"] + [x for x in argv if x != "--freeze-only"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ru = sub.add_parser("rule")
    ru.add_argument("--out", required=True)
    ru.add_argument("--label", default="hidden")
    ru.add_argument("--k-range", default=None)
    ru.add_argument("--tables", default=None, help="the generator's tables block (python -m site_gen.sets --table)")
    ru.add_argument("--exclusion", default=None, help="the exclusion list (default: the task's generator copy's "
                                                      "site/config/exclusion_v2.json)")
    f = sub.add_parser("freeze")
    f.add_argument("--candidates", required=True)
    f.add_argument("--proofs", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--label", default="hidden")
    f.add_argument("--k-range", default=None)
    r = sub.add_parser("run")
    r.add_argument("--candidates", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--k", default=None, help="k values: a-b, k or a comma list of both (default: the selection's)")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--baselines", choices=("all", "admissible"), default="all")
    o = sub.add_parser("run-one")
    o.add_argument("--candidates", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--k", required=True)
    o.add_argument("--prog", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--out", required=True)
    w = sub.add_parser("write-hidden")
    w.add_argument("--out", required=True)
    w.add_argument("--hidden", required=True)
    w.add_argument("--candidates", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--out", required=True)
    rp.add_argument("--md", default=None)
    pp = sub.add_parser("practice")
    pp.add_argument("--out", required=True)
    pp.add_argument("--write", default=None)
    pp.add_argument("--candidates", default=None)
    pp.add_argument("--hall", default=None)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args(argv)
    for name in ("out", "candidates", "hidden", "write", "hall", "tables", "md", "exclusion"):
        if getattr(a, name, None):
            setattr(a, name, os.path.abspath(getattr(a, name)))
    return {"rule": cmd_rule, "freeze": cmd_freeze, "run": cmd_run, "run-one": cmd_run_one, "decide": cmd_decide,
            "practice": cmd_practice, "report": cmd_report, "write-hidden": cmd_write_hidden,
            "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
