"""select_set.py -- the pre-registered selection of the fixed hidden set of freeflyer_uncrewed_station_caretaking
(the selection rules and the rule that every candidate runs once).
Maintainers' tool on the hidden side: never part of the grade, never named to the agent.

The rule (``RULE``; the rule file is ``rule`` 's output and must equal it, ``check-rule``):
  candidates   6 cells (module of the blocked inlet x tier, layout.cell.*) x k = 0 .. layout.candidates_per_cell - 1
               (6 per cell, 36 in all, no extra candidates); candidate (cell, k) is the first attempt
               j = 0, 1, ... (j < MAX_ATTEMPTS) of the seed stream sha256("<label>/<cell>/<k>/<j>") that passes every
               stage of the site generator (tests/review/sitegen/gen: the draw, the screen, the 40 s constant-force
               fixture, the 600 s settle, the telemetry, check_site, the 2400 s robot-free dry run, the package);
               label = gen.cells.LABELS['hidden'], kind 'hidden'
  physics      generated, the dry runs passed (both in the passing attempt), and ONE as-is run of the privileged
               reference program (direct mode, the package's own noise seed, no perturbation) with raw score 1.000,
               score 1.000 and every selection margin of grader.selection (tests/grader_config.json) met
  observability  tests/review/check_observability.py (CHECK_VERSION as frozen) passes on the package with the
               station look of the asset root in force
  robustness   ONE run with every free body's start moved by layout.robustness_perturb_um (1 um) and ONE as-is run on
               the second CPU generation: each with raw and score 1.000 and every binary criterion of grade() equal
               to the as-is run's (the energy line of both from the as-is run's energy); the second-CPU run must also
               end in the as-is run's state and qpos sha256 (bit-identical across CPUs)
  order        per cell, k ascending: the first candidate that passes physics, observability and robustness is the
               cell's site; a candidate that passes physics and observability and has no robustness run yet holds
               the cell (it is never skipped); every dropped candidate keeps its reasons
  empty cell   never loosened: the cell is reported with its counts and reasons (the maintainers decide: 8 candidates per
               cell, or the reference)
  contingency  the station look may be replaced by the finished look (physics unchanged): then only the
               observability step is computed again for all 36 candidates under the same rules, and ``reevaluate``
               applies the same order to the stored physics and robustness verdicts and the fresh observability
               verdicts; a cell whose new first candidate lacks a robustness run is held until that run is made (once)
Every candidate is generated once and run once per run kind; robustness runs are made for every
candidate that passes physics (those that also pass observability first), so the contingency needs no new run in
the common case.  Runs made for candidates after a cell's site change nothing (the order decides).

Commands (TASK = the folder above tests/; DIR after --dev = the folder of the maintainers' further frozen files):
  rule         --out FILE                                     the rule document
  check-rule   --rule FILE                                    exit 1 when the file differs from RULE
  freeze       --rule FILE --dev DIR --out FILE [--archives DIR]   sha256 of every frozen file and archive, UTC time
  check-freeze --freeze FILE [--dev DIR] [--archives DIR]    exit 1 when a frozen file differs
  generate     CELL K --out CAMP --freeze FILE [--dev DIR] [--schema S]     one candidate (sitegen/gen/)
  run          SITE --kind asis|um1|cpu2 --out CAMP --freeze FILE [--dev DIR] [--ref-energy-wh X]   one reference run
  observe      SITE --out CAMP --freeze FILE [--dev DIR] [--frames DIR] [--look-root DIR] [--obs-dir NAME]
  decide       --camp CAMP [--camp CAMP2 ...] --out FILE [--obs-dir NAME]    every verdict and the set
  plan-runs    --decided FILE                                 the robustness runs still to make, by priority
  reevaluate   --decided FILE --obs DIR --out FILE            the contingency: the same order on fresh observability
  write-hidden --decided FILE --camp CAMP ... --hidden DIR --frozen FILE [--commit HASH]   the grader's layout
CAMP layout: gen/<site_id>.candidate.json, gen/attempts/<site_id>_j<j>.json, gen/packages/<site_id>.site.tar.xz,
runs/<site_id>_<kind>.summary.json and .record.json, obs/<site_id>.obs.json and .obs.out, ledger.jsonl.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import sys
import time

REVIEW = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(REVIEW)
TASK = os.path.dirname(TESTS)
SITEGEN = os.path.join(REVIEW, "sitegen")
for _p in (TESTS, os.path.join(TESTS, "cabincell"), os.path.join(TESTS, "cabincell", "_lib"), SITEGEN, REVIEW):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True

TOOL_VERSION = "1.0.0"
SCHEMA_RULE = "caretaking.selection_rule/1"
SCHEMA_FREEZE = "caretaking.selection_freeze/1"
SCHEMA_CAND = "caretaking.selection_candidate/1"
SCHEMA_RUN = "caretaking.selection_run/1"
SCHEMA_DECIDED = "caretaking.selection_decided/1"
SCHEMA_HIDDEN = "caretaking.hidden_selection/1"
KIND = "hidden"
LABEL_KEY = "hidden"
MAX_ATTEMPTS = 80
RUN_KINDS = ("asis", "um1", "cpu2")
PINS = dict(NPY_DISABLE_CPU_FEATURES="X86_V4,AVX512_ICL", OPENBLAS_CORETYPE="Haswell", OPENBLAS_L2_SIZE="256")
EPS = 1e-9
# the frozen files: what can change a candidate or a verdict (relative to TASK; the --dev parts relative to that folder)
FROZEN_TASK = ("tests/cabincell", "tests/reference", "tests/review/sitegen", "tests/review/check_observability.py",
               "tests/review/select_set.py", "tests/grader_config.json")
FROZEN_DEV = ("selection/rules.json", "contracts/site.schema.json", "contracts/schemalite.py")
ASSET_COMPONENTS = ("robot", "station", "robot_author")
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SKIP_FILES = {"MUJOCO_LOG.TXT"}
SKIP_SUFFIX = (".pyc", ".pyo")

RULE = dict(
    schema=SCHEMA_RULE, rule_version="1.0.0", task="freeflyer_uncrewed_station_caretaking",
    basis=["the task's site, observability and hidden-set rules",
           "the privileged reference program, six candidates per cell and the station look in force", "every candidate runs once"],
    candidates=dict(label_key=LABEL_KEY, kind=KIND, cells="gen.cells.CELL_ORDER (L2xS, L2xP, H2xS, H2xP, L1xS, L1xP)",
                    per_cell_key="layout.candidates_per_cell", per_cell=6, extra_candidates=0,
                    seed_rule="layout.seed_rule: candidate (cell, k) = the first passing attempt j of sha256('<label>/<cell>/<k>/<j>')",
                    max_attempts=MAX_ATTEMPTS, generator="tests/review/sitegen/gen pipeline.run_attempt; "
                    "every stage A-H (draw, screen, fixture, settle, telemetry, check_site, dry run, pack)"),
    physics=dict(generation="a passing attempt with j < max_attempts (the generation conditions and check_site)",
                 dry_runs="the passing attempt's 600 s settle and 2400 s robot-free dry run (sheet cover change < 1 %, "
                          "displacement <= 3 mm; pouch <= 1 mm) passed",
                 reference_asis="one run of tests/reference (CabinCellServer(privileged=True).reference_run(), direct mode, "
                                "seed = the package's noise seed, perturb 0): grade() raw == 1.000 and score == 1.000 with "
                                "the run's own energy as the energy line, and every margin of grade.margins() ok "
                                "(grader.selection: time <= 2040 s, docking charge >= 20 %, frames <= 50 %, captures <= 15, "
                                "hard collisions 0, keep-out 0 s, fan stop <= 30 s)",
                 energy_reserve_skip="a candidate on which the reference skips the blocker by its own reserve rule fails "
                                     "by raw != 1.000"),
    observability=dict(check="tests/review/check_observability.check_site, CHECK_VERSION as frozen (1.1.0); every row passes",
                       look="the station look of the asset root in force (data/station/ASSETS.json; station_look_v04 at "
                            "the freeze)", independent_of_physics=True),
    robustness=dict(um1="one run as reference_asis with layout.robustness_perturb_um (1 um) on every free body's start",
                    cpu2="one as-is run on the second CPU generation (Zen 4; the as-is runs are made on Zen 5)",
                    pass_rule="raw == 1.000 and score == 1.000, every key of grade()['binary'] equal to the as-is run's "
                              "(the energy line from the as-is run's energy); cpu2 also state_sha256 and qpos_sha256 "
                              "equal to the as-is run's (bit-identical across CPUs)",
                    margins="recorded, not required (the pass rule reads the binary criteria and the raw score)",
                    which="every candidate that passes physics: first those that also pass observability (k ascending per "
                          "cell), then the others (stored for the contingency); each run once"),
    order="per cell, k ascending: the first candidate with physics, observability and robustness passed; a candidate "
          "with physics and observability passed but a robustness run missing holds the cell; never skipped",
    secondary_attributes="fixed per cell by layout.cell.* (the generator draws them; check_site checks them); the set "
                         "must give layout.counts (low battery 3, wrong advisory 2, failed light 2, moved stowage unit 2)",
    empty_cell="no loosening: report the cell's pass counts, every failure reason, and the options (8 candidates in that "
               "cell before a new freeze / a reference change) to the maintainers",
    contingency_look=dict(ruling="the finished look may replace the station look", what="the station look replaced by the finished look, physics unchanged "
                          "(collision hashes equal)", then="observe all 36 candidates again under the same check; "
                          "reevaluate: the same order on the stored physics and robustness verdicts with the fresh "
                          "observability verdicts; held cells get their missing robustness run (once)",
                          stored="per candidate: physics (generation, dry runs, reference as-is raw / score / margins) and "
                                 "robustness verdicts separately from the observability verdict"),
    runs_once="each candidate: one generation, one as-is run, at most one um1 and one cpu2 run; a second record of the "
              "same (candidate, kind) is an error",
    hollowing=dict(features=["hold_N", "chain_fold_deg", "offset_long_m", "offset_short_m", "cover_m2"],
                   compare="the six against all generated candidates, by tier; the dropped candidates' placements",
                   blind_mc="a blind-sweep Monte Carlo of the removal (capture gate 4 cm / 8 deg, "
                            "budget 20) on the selected placements; reported to the maintainers above 0.20 (a report, not a gate)"),
    outputs=dict(hidden="tests/hidden/selection.json (set: tag, package, sha256, seed null, ref_energy_wh = the as-is "
                        "run's energy), tests/hidden/candidates.json, tests/hidden/sites/<site_id>.site.tar.xz",
                 review="tests/review/observability/ (the six sites' observability numbers and thumbnails)"),
)


# ============================================================================================== small helpers
def utc(t: float | None = None) -> str:
    return datetime.datetime.fromtimestamp(time.time() if t is None else t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def jdump(obj, path: str) -> str:
    """Write JSON atomically (tmp + rename), sorted keys; return the sha256 of the bytes."""
    data = (json.dumps(_jsonable(obj), indent=1, sort_keys=True) + "\n").encode("utf-8")
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return sha_bytes(data)


def jload(path: str):
    with open(path) as fh:
        return json.load(fh)


def _jsonable(x):
    try:
        import numpy as np
    except ImportError:          # pragma: no cover
        np = None
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if np is not None:
        if isinstance(x, np.ndarray):
            return _jsonable(x.tolist())
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            x = float(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def ledger(camp: str, **kw) -> None:
    os.makedirs(camp, exist_ok=True)
    with open(os.path.join(camp, "ledger.jsonl"), "a") as fh:
        fh.write(json.dumps(dict(t=utc(), host=platform.node(), pid=os.getpid(), **kw), sort_keys=True) + "\n")


def need_pins() -> None:
    miss = {k: v for k, v in PINS.items() if os.environ.get(k) != v}
    if miss:
        raise SystemExit(f"the cell's numeric pins: set {miss} in the environment")


def site_id_of(cell: str, k: int) -> str:
    from gen import cells as CE
    return CE.site_id(KIND, cell, int(k))


def plan() -> list:
    """[(cell, k)] of the rule, cells in CELL_ORDER, k ascending; the per-cell count from the runtime site_spec."""
    from gen import cells as CE
    from cabincell.site import spec as SPEC
    n = int(SPEC.value("layout.candidates_per_cell"))
    if n != RULE["candidates"]["per_cell"]:
        raise SystemExit(f"layout.candidates_per_cell {n} != the rule's {RULE['candidates']['per_cell']}")
    return [(c, k) for c in CE.CELL_ORDER for k in range(n)]


# ============================================================================================== rule and freeze
def rule_doc() -> dict:
    from gen import cells as CE
    d = json.loads(json.dumps(RULE))
    d["candidates"]["label"] = CE.LABELS[LABEL_KEY]
    d["candidates"]["cell_order"] = list(CE.CELL_ORDER)
    return d


def _walk(root: str, part: str) -> list:
    p = os.path.join(root, part)
    if os.path.isfile(p):
        return [part]
    if not os.path.isdir(p):
        raise FileNotFoundError(p)
    out = []
    for r, dirs, fs in os.walk(p):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for f in sorted(fs):
            if f in SKIP_FILES or f.endswith(SKIP_SUFFIX) or f.endswith(".tmp"):
                continue
            out.append(os.path.relpath(os.path.join(r, f), root).replace(os.sep, "/"))
    return out


def frozen_table(task: str = TASK, dev: str | None = None) -> dict:
    """{'task:<rel>' | 'dev:<rel>': sha256} of every frozen file."""
    T = {}
    for part in FROZEN_TASK:
        for rel in _walk(task, part):
            T["task:" + rel] = sha_file(os.path.join(task, rel))
    if dev:
        for part in FROZEN_DEV:
            for rel in _walk(dev, part):
                T["dev:" + rel] = sha_file(os.path.join(dev, rel))
    return dict(sorted(T.items()))


def table_digest(T: dict) -> str:
    return sha_bytes("".join(f"{k} {v}\n" for k, v in sorted(T.items())).encode("utf-8"))


def archives(task: str = TASK, archive_dir: str | None = None) -> dict:
    """{component: dict(name, sha256 pinned in data/<c>/ASSETS.json, file sha256 when the archive is found)}."""
    out = {}
    for c in ASSET_COMPONENTS:
        man = jload(os.path.join(task, "tests", "cabincell", "data", c, "ASSETS.json"))
        a = man["archive"]
        e = dict(name=a["name"], sha256=a["sha256"], bytes=a.get("bytes"), unpacked_files=man["unpacked"]["files"])
        if archive_dir:
            for cand in (os.path.join(archive_dir, c, a["name"]), os.path.join(archive_dir, a["name"])):
                if os.path.isfile(cand):
                    e["file_sha256"] = sha_file(cand)
                    e["file"] = cand
                    break
        out[c] = e
    return out


def freeze_doc(rule_path: str, dev: str, archive_dir: str | None = None) -> dict:
    rd = jload(rule_path)
    if rd != rule_doc():
        raise SystemExit(f"{rule_path} is not this tool's rule (run check-rule)")
    from gen import freeze as FZ
    from cabincell.site import spec as SPEC
    T = frozen_table(TASK, dev)
    A = archives(TASK, archive_dir)
    bad = [c for c, e in A.items() if "file_sha256" in e and e["file_sha256"] != e["sha256"]]
    if bad:
        raise SystemExit(f"archives differ from their manifests: {bad}")
    gen = FZ.report()
    return dict(schema=SCHEMA_FREEZE, tool_version=TOOL_VERSION, rule_version=rd["rule_version"],
                rule_sha256=sha_file(rule_path), created_utc=utc(), host=platform.node(), cpu=cpu_name(),
                files=T, files_count=len(T), files_digest=table_digest(T), archives=A,
                generator_sha256=gen["generator_sha256"], generator_parts={k: v["sha256"] for k, v in gen["parts"].items()},
                runtime_site_spec=dict(sha256=SPEC.sha256(), contract_version=SPEC.document().get("contract_version")),
                check_observability_version=_check_version(), firmware_version=_firmware_version(),
                note="no candidate of the hidden label is generated or run before the maintainers commit this list "
                     "(the commit is recorded with the freeze)")


def _check_version() -> str:
    import re
    txt = open(os.path.join(REVIEW, "check_observability.py")).read()
    m = re.search(r'^CHECK_VERSION = "([^"]+)"', txt, re.M)
    return m.group(1) if m else "unknown"


def _firmware_version() -> str:
    import re
    txt = open(os.path.join(TESTS, "cabincell", "firmware", "__init__.py")).read()
    m = re.search(r'^VERSION = "([^"]+)"', txt, re.M)
    return m.group(1) if m else "unknown"


def check_freeze(freeze_path: str, dev: str | None = None, archive_dir: str | None = None, task: str = TASK) -> list:
    """Problems of the tree against a freeze document ([] = equal).  Without --dev only the task files are checked."""
    F = jload(freeze_path)
    if F.get("schema") != SCHEMA_FREEZE:
        return [f"{freeze_path}: not a {SCHEMA_FREEZE} document"]
    want = {k: v for k, v in F["files"].items() if dev or k.startswith("task:")}
    got = frozen_table(task, dev)
    got = {k: v for k, v in got.items() if dev or k.startswith("task:")}
    probs = [f"changed {k}" for k in sorted(want) if k in got and got[k] != want[k]]
    probs += [f"missing {k}" for k in sorted(want) if k not in got]
    probs += [f"new {k}" for k in sorted(got) if k not in want]
    A = archives(task, archive_dir)
    for c, e in F["archives"].items():
        if A[c]["sha256"] != e["sha256"]:
            probs.append(f"archive {c}: manifest pins {A[c]['sha256']} != frozen {e['sha256']}")
        if "file_sha256" in A[c] and A[c]["file_sha256"] != e["sha256"]:
            probs.append(f"archive {c}: file {A[c]['file_sha256']} != frozen {e['sha256']}")
    return probs


def _guard(a) -> dict:
    """Refuse a command that makes a candidate or a verdict when the tree is not the frozen one."""
    if not a.freeze or not os.path.isfile(a.freeze):
        raise SystemExit("--freeze FILE is required (the committed freeze document)")
    probs = check_freeze(a.freeze, getattr(a, "dev", None))
    if probs:
        raise SystemExit("the tree differs from the freeze: " + "; ".join(probs[:10]))
    need_pins()
    return jload(a.freeze)


# ============================================================================================== generation
def generate(cell: str, k: int, camp: str, schema_path: str | None, freeze: dict | None, max_attempts: int = MAX_ATTEMPTS,
             label: str | None = None, test_kw: dict | None = None) -> dict:
    """Candidate (cell, k): attempts j = 0.. until one passes (module doc); every attempt record and the candidate
    record are written under camp/gen.  ``label`` / ``test_kw`` never carry the hidden label."""
    from gen import cells as CE
    from gen import freeze as FZ
    from gen import pipeline as PI
    lab = label or CE.LABELS[LABEL_KEY]
    if test_kw and lab == CE.LABELS[LABEL_KEY]:
        raise SystemExit("test mode never draws a candidate of the hidden label")
    sid = CE.site_id(KIND, cell, k)
    gdir = os.path.join(camp, "gen")
    os.makedirs(os.path.join(gdir, "attempts"), exist_ok=True)
    os.makedirs(os.path.join(gdir, "packages"), exist_ok=True)
    cpath = os.path.join(gdir, f"{sid}.candidate.json")
    if os.path.exists(cpath):
        raise SystemExit(f"{cpath} exists: a candidate is generated once")
    schema = None
    if schema_path:
        d = os.path.dirname(os.path.abspath(schema_path))
        if d not in sys.path:
            sys.path.insert(0, d)
        schema = jload(schema_path)
    gen_sha = FZ.digest()
    if freeze and gen_sha != freeze["generator_sha256"]:
        raise SystemExit(f"generator digest {gen_sha} != the frozen {freeze['generator_sha256']}")
    t0, c0 = time.time(), time.process_time()
    ledger(camp, cmd="generate", site_id=sid, cell=cell, k=int(k), what="start")
    tries, chosen = [], None
    for j in range(max_attempts):
        r = PI.run_attempt(lab, KIND, cell, k, j, keep=True, schema=schema, gen_sha=gen_sha, **(test_kw or {}))
        R = r["record"]
        jdump(R, os.path.join(gdir, "attempts", f"{sid}_j{j}.json"))
        rej = R.get("rejected") or {}
        tries.append(dict(j=j, passed=bool(R["passed"]), stage=rej.get("stage"), cause=rej.get("cause"),
                          cpu_s=R.get("cost_cpu_s"), error=bool(R.get("error"))))
        if R["passed"]:
            data = r["package"]
            pk = os.path.join(gdir, "packages", f"{sid}.site.tar.xz")
            with open(pk + ".tmp", "wb") as fh:
                fh.write(data)
            os.replace(pk + ".tmp", pk)
            chosen = dict(j=j, record=R, package=os.path.relpath(pk, camp), sha=sha_bytes(data), bytes=len(data))
            break
    rec = dict(schema=SCHEMA_CAND, tool_version=TOOL_VERSION, site_id=sid, cell=cell, k=int(k), label=lab, kind=KIND,
               seed_text=CE.seed_text(lab, cell, k), candidate_seed=CE.candidate_seed(lab, cell, k),
               generated=chosen is not None, attempts=tries, n_attempts=len(tries), max_attempts=int(max_attempts),
               generator_sha256=gen_sha, freeze_digest=(freeze or {}).get("files_digest"),
               t_start=utc(t0), t_end=utc(), wall_s=round(time.time() - t0, 2), cpu_s=round(time.process_time() - c0, 2),
               host=platform.node(), cpu=cpu_name(), test_mode=bool(test_kw))
    if chosen:
        R = chosen["record"]
        rec.update(j=chosen["j"], package=chosen["package"], package_sha256=chosen["sha"], package_bytes=chosen["bytes"],
                   site_spec_sha256=R.get("site_spec_sha256"), dry_run=R.get("dry_run"),
                   settle=(R.get("stages") or {}).get("D_settle", {}).get("settle_disp"),
                   summary=R.get("summary"), features=features_of(R.get("summary") or {}))
    else:
        rec.update(j=None, why=f"no passing attempt in {max_attempts}")
    jdump(rec, cpath)
    ledger(camp, cmd="generate", site_id=sid, what="end", generated=rec["generated"], j=rec.get("j"))
    return rec


def features_of(summary: dict) -> dict:
    """The hollowing features of a candidate: hold force, chain angle, offsets, covered area."""
    off = summary.get("offset_m") or [None, None]
    return dict(tier=summary.get("tier"), blocked_inlet=summary.get("blocked_inlet"), family=summary.get("family"),
                hold_N=summary.get("hold_N"), chain_fold_deg=summary.get("chain_fold_deg"), theta_deg=summary.get("theta_deg"),
                offset_long_m=None if off[0] is None else abs(float(off[0])),
                offset_short_m=None if off[1] is None else abs(float(off[1])),
                offset_m=off, cover_m2=summary.get("cover_m2"), r_face=summary.get("r_face"),
                n_pouches=summary.get("n_pouches"), battery_pct=summary.get("battery_pct"))


# ============================================================================================== reference runs
def stable_digest(rec: dict) -> str:
    """sha256 of the record without what depends on the wall clock or the machine."""
    r = json.loads(json.dumps(rec))
    r.pop("metrics", None)
    r.get("budgets_used", {}).pop("program_compute_s", None)
    r.get("run", {}).pop("label", None)
    r.get("run", {}).pop("cpu", None)
    return sha_bytes(json.dumps(r, sort_keys=True, separators=(",", ":")).encode("ascii"))


def grade_record(rec: dict, site_d: dict, ref_e: float | None) -> dict:
    """grade(), margins(), chain_pass() of one record against its site (the published grader, its config)."""
    from cabincell import grade as GR
    g = GR.grade(rec, site_d, ref_e)
    mg = GR.margins(rec, g)
    cp = GR.chain_pass(g, mg)
    return dict(score=g["score"], raw=g["raw"], R=g["R"], items=g["items"], binary=g["binary"],
                margins={k: dict(value=v["value"], limit=v["limit"], ok=v["ok"]) for k, v in mg.items()},
                full=cp["full"], chain=cp["chain"], margins_ok=cp["margins_ok"], ref_energy_wh=ref_e)


def run_reference(site: str, kind: str, camp: str, ref_energy_wh: float | None = None, limits: dict | None = None,
                  tag_suffix: str = "") -> dict:
    """One run of the privileged reference program (module doc) -> camp/runs/<site_id>_<kind>.{record,summary}.json."""
    if kind not in RUN_KINDS:
        raise SystemExit(f"--kind {kind}: one of {RUN_KINDS}")
    from cabincell import server as SV
    from cabincell.site import spec as SPEC
    site_d, _m, site_sha = SV.open_site(site)
    sid = site_d["site_id"]
    rdir = os.path.join(camp, "runs")
    os.makedirs(rdir, exist_ok=True)
    tag = f"{sid}_{kind}{tag_suffix}"
    spath = os.path.join(rdir, f"{tag}.summary.json")
    if os.path.exists(spath):
        raise SystemExit(f"{spath} exists: every candidate runs once per kind")
    um = float(SPEC.value("layout.robustness_perturb_um")) if kind == "um1" else 0.0
    logs = []
    t0, c0 = time.time(), time.process_time()
    ledger(camp, cmd="run", site_id=sid, kind=kind, what="start")
    srv = SV.CabinCellServer(privileged=True, reference_dir=os.path.join(TESTS, "reference"), limits=limits, log=logs.append)
    srv.reset(site, seed=None, label=f"p14_{tag}", perturb_um=um)
    t_build = time.time() - t0
    reply = srv.reference_run()
    rec = srv.finalize()
    ref = srv.reference_result or {}
    rpath = os.path.join(rdir, f"{tag}.record.json")
    rec_sha = SV.write_record(rpath, rec)
    if kind == "asis":
        ref_energy_wh = rec.get("energy_wh")
    g = grade_record(rec, site_d, ref_energy_wh)
    summ = dict(schema=SCHEMA_RUN, tool_version=TOOL_VERSION, site_id=sid, cell=site_d.get("cell"), kind=kind, tag=tag,
                package_sha256=site_sha, perturb_um=um, seed=int(srv.cell.seed), limits=limits,
                pins={k: os.environ.get(k) for k in PINS}, record=os.path.basename(rpath), record_sha256=rec_sha,
                stable_digest=stable_digest(rec), state_sha256=rec["hashes"]["state_sha256"], qpos_sha256=rec["hashes"]["qpos_sha256"],
                exit=rec["times"]["exit"], program_end=rec["times"]["program_end"], end=rec["times"]["end"],
                energy_wh=rec.get("energy_wh"), dock=rec.get("dock"), keepout_s=rec.get("keepout_s"),
                budgets_used=rec.get("budgets_used"), fan_stops=rec.get("fan_stops"), violations=rec.get("violations"),
                loss_of_control=rec.get("loss_of_control"), grade=g, reference_reply=reply,
                reference_summary=ref.get("summary"), notes=ref.get("notes"), internal_errors=srv.internal_errors,
                logs=[str(x)[:400] for x in logs[-10:]], t_start=utc(t0), t_end=utc(), build_wall_s=round(t_build, 2),
                wall_s=round(time.time() - t0, 2), cpu_s=round(time.process_time() - c0, 2), host=platform.node(),
                cpu=rec.get("run", {}).get("cpu") or cpu_name())
    jdump(summ, spath)
    srv.close()
    ledger(camp, cmd="run", site_id=sid, kind=kind, what="end", score=g["score"], raw=g["raw"], full=g["full"])
    return summ


# ============================================================================================== observability
def look_digest(look_root: str) -> str:
    rows = []
    for r, dirs, fs in os.walk(look_root):
        dirs[:] = sorted(dirs)
        for f in sorted(fs):
            p = os.path.join(r, f)
            rows.append(f"{os.path.relpath(p, look_root)} {sha_file(p)}\n")
    return sha_bytes("".join(rows).encode("utf-8"))


def observe(site: str, camp: str, frames_dir: str | None = None, look_root: str | None = None, obs_name: str = "obs") -> dict:
    import check_observability as CO
    import assetroot
    odir = os.path.join(camp, obs_name)
    os.makedirs(odir, exist_ok=True)
    t0 = time.time()
    r = CO._jsonable(CO.check_site(site, frames_dir, look_root))
    sid = r["site_id"]
    man = assetroot.manifest("station")
    lr = look_root or str(assetroot.find("station"))
    # the observation records no machine or account name (it is published with the hidden set): the CPU model only
    r["selection"] = dict(tool_version=TOOL_VERSION, package_sha256=sha_file(site), t_start=utc(t0), t_end=utc(),
                          cpu=cpu_name(), check_observability_sha256=sha_file(CO.__file__),
                          station_manifest_archive_sha256=man["archive"]["sha256"], look_root_digest=look_digest(lr),
                          look_root_given=look_root is not None)
    jdump(r, os.path.join(odir, f"{sid}.obs.json"))
    with open(os.path.join(odir, f"{sid}.obs.out"), "w") as fh:
        fh.write(CO.summary_line(r) + "\n")
    ledger(camp, cmd="observe", site_id=sid, what="end", passed=r["passed"], obs_dir=obs_name)
    return r


# ============================================================================================== decision
def _close(x, v=1.0) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x)) and abs(float(x) - v) < EPS


def verdict_asis(s: dict | None) -> dict:
    if s is None:
        return dict(ok=None, why=["no as-is run"])
    g = s["grade"]
    why = []
    if not _close(g.get("raw")):
        why.append(f"raw {g.get('raw')}")
    if not _close(g.get("score")):
        why.append(f"score {g.get('score')}")
    for k, m in sorted((g.get("margins") or {}).items()):
        if not m.get("ok"):
            why.append(f"margin {k} {m.get('value')} (limit {m.get('limit')})")
    if s.get("exit") == "cell_error" or s.get("internal_errors"):
        why.append(f"cell exit {s.get('exit')} internal errors {s.get('internal_errors')}")
    if not why and not g.get("full"):
        why.append("chain_pass full false")
    lost = [k for k, v in (g.get("binary") or {}).items() if not v and not str(k).startswith("lost:")]
    return dict(ok=not why, why=why, failed_binary=sorted(lost))


def verdict_robust(s: dict | None, asis: dict | None, kind: str) -> dict:
    if s is None:
        return dict(ok=None, why=[f"no {kind} run"])
    if asis is None:
        return dict(ok=False, why=["no as-is run to compare"])
    g, ga = s["grade"], asis["grade"]
    why = []
    if not _close(g.get("raw")):
        why.append(f"raw {g.get('raw')}")
    if not _close(g.get("score")):
        why.append(f"score {g.get('score')}")
    if g.get("ref_energy_wh") is None or ga.get("ref_energy_wh") is None or abs(float(g["ref_energy_wh"]) - float(ga["ref_energy_wh"])) > 1e-9:
        why.append("energy line is not the as-is run's energy")
    b, ba = g.get("binary") or {}, ga.get("binary") or {}
    changed = sorted(k for k in set(b) | set(ba) if b.get(k) != ba.get(k))
    if changed:
        why.append("binary changed: " + ",".join(changed))
    same_hash = bool(s.get("state_sha256") == asis.get("state_sha256") and s.get("qpos_sha256") == asis.get("qpos_sha256"))
    if kind == "cpu2" and not same_hash:
        why.append("state / qpos sha256 differ from the as-is run")
    return dict(ok=not why, why=why, changed=changed, same_hashes=same_hash, stable_same=s.get("stable_digest") == asis.get("stable_digest"),
                cpu=s.get("cpu"), margins_ok=bool(g.get("margins_ok")))


def verdict_obs(o: dict | None) -> dict:
    if o is None:
        return dict(ok=None, why=["no observability result"])
    return dict(ok=bool(o.get("passed")), why=list(o.get("why") or []), fails=list(o.get("fails") or []),
                check_version=o.get("check_version"), rows=o.get("rows"),
                look=(o.get("selection") or {}).get("station_manifest_archive_sha256"),
                look_root_digest=(o.get("selection") or {}).get("look_root_digest"))


def candidate_verdicts(c: dict | None, runs: dict, obs: dict | None) -> dict:
    """Every verdict of one candidate: physics (generation, dry runs, as-is), observability, robustness (um1, cpu2)."""
    if c is None or not c.get("generated"):
        why = ["not generated" if c is None else (c.get("why") or "no passing attempt")]
        return dict(physics=dict(ok=False if c is not None else None, generation=False, why=why), observability=verdict_obs(obs),
                    um1=dict(ok=None, why=[]), cpu2=dict(ok=None, why=[]))
    dry = bool((c.get("dry_run") or {}).get("passed"))
    asis = runs.get("asis")
    va = verdict_asis(asis)
    why = ([] if dry else ["dry run failed"]) + list(va["why"])
    ok_phys = None if va["ok"] is None else bool(dry and va["ok"])
    return dict(physics=dict(ok=ok_phys, generation=True, dry_runs=dry, asis=va, why=why,
                             energy_wh=(asis or {}).get("energy_wh"), t_docked=((asis or {}).get("dock") or {}).get("t_docked"),
                             battery_at_docked=((asis or {}).get("dock") or {}).get("battery_pct_at_docked")),
                observability=verdict_obs(obs), um1=verdict_robust(runs.get("um1"), asis, "um1"),
                cpu2=verdict_robust(runs.get("cpu2"), asis, "cpu2"))


def select(per: dict, cells: list, per_cell: int) -> dict:
    """The rule's order (module doc) over per[site_id] = verdicts (+ cell, k)."""
    out = {}
    for cell in cells:
        rows = sorted((v for v in per.values() if v["cell"] == cell), key=lambda v: v["k"])
        chosen, held, dropped = None, None, []
        for v in rows:
            P, O, U, C = v["physics"]["ok"], v["observability"]["ok"], v["um1"]["ok"], v["cpu2"]["ok"]
            if P is None or (P and O is None):
                held = dict(site_id=v["site_id"], k=v["k"], waiting_for=("as-is run" if P is None else "observability"))
                break
            if not P or not O:
                dropped.append(dict(site_id=v["site_id"], k=v["k"], why=(v["physics"]["why"] if not P else []) +
                                    ([] if O or O is None else ["observability: " + ",".join(v["observability"]["fails"])])))
                continue
            if U is None or C is None:
                held = dict(site_id=v["site_id"], k=v["k"], waiting_for=",".join(x for x, y in (("um1", U), ("cpu2", C)) if y is None))
                break
            if U and C:
                chosen = v["site_id"]
                break
            dropped.append(dict(site_id=v["site_id"], k=v["k"], why=["um1: " + "; ".join(v["um1"]["why"])] * (not U) +
                                ["cpu2: " + "; ".join(v["cpu2"]["why"])] * (not C)))
        n = len(rows)
        state = "selected" if chosen else ("held" if held else "empty")
        counts = dict(candidates=n, generated=sum(1 for v in rows if v["physics"]["generation"]),
                      physics=sum(1 for v in rows if v["physics"]["ok"]), observability=sum(1 for v in rows if v["observability"]["ok"]),
                      physics_and_observability=sum(1 for v in rows if v["physics"]["ok"] and v["observability"]["ok"]),
                      robust=sum(1 for v in rows if v["um1"]["ok"] and v["cpu2"]["ok"]),
                      all=sum(1 for v in rows if v["physics"]["ok"] and v["observability"]["ok"] and v["um1"]["ok"] and v["cpu2"]["ok"]))
        out[cell] = dict(state=state, site_id=chosen, held=held, dropped_before=dropped, counts=counts, of=per_cell,
                         complete_count=(n == per_cell))
    return out


def load_camps(camps: list, obs_name: str = "obs") -> dict:
    """{site_id: dict(candidate, runs{kind: summary}, obs, files)}; a second record of the same thing is an error."""
    D = {}
    dup = []
    for camp in camps:
        g = os.path.join(camp, "gen")
        if os.path.isdir(g):
            for f in sorted(os.listdir(g)):
                if f.endswith(".candidate.json"):
                    c = jload(os.path.join(g, f))
                    e = D.setdefault(c["site_id"], dict(runs={}, files={}))
                    if "candidate" in e:
                        dup.append(f"candidate {c['site_id']}")
                    e["candidate"], e["files"]["candidate"] = c, os.path.join(g, f)
                    if c.get("generated"):
                        e["files"]["package"] = os.path.join(camp, c["package"])
        r = os.path.join(camp, "runs")
        if os.path.isdir(r):
            for f in sorted(os.listdir(r)):
                if f.endswith(".summary.json"):
                    s = jload(os.path.join(r, f))
                    e = D.setdefault(s["site_id"], dict(runs={}, files={}))
                    if s["kind"] in e["runs"]:
                        dup.append(f"run {s['site_id']} {s['kind']}")
                    e["runs"][s["kind"]] = s
                    e["files"][f"run_{s['kind']}"] = os.path.join(r, f)
                    e["files"][f"record_{s['kind']}"] = os.path.join(r, s["record"])
        o = os.path.join(camp, obs_name)
        if os.path.isdir(o):
            for f in sorted(os.listdir(o)):
                if f.endswith(".obs.json"):
                    x = jload(os.path.join(o, f))
                    e = D.setdefault(x["site_id"], dict(runs={}, files={}))
                    if "obs" in e:
                        dup.append(f"observability {x['site_id']}")
                    e["obs"], e["files"]["obs"] = x, os.path.join(o, f)
    if dup:
        raise SystemExit("records made twice: " + "; ".join(dup))
    return D


def decide(camps: list, obs_name: str = "obs") -> dict:
    from gen import cells as CE
    P = plan()
    D = load_camps(camps, obs_name)
    per = {}
    for cell, k in P:
        sid = site_id_of(cell, k)
        e = D.get(sid, dict(runs={}, files={}))
        o = e.get("obs")
        c = e.get("candidate")
        if c is not None and c.get("generated") and "package" in e["files"]:
            if not os.path.isfile(e["files"]["package"]) or sha_file(e["files"]["package"]) != c["package_sha256"]:
                raise SystemExit(f"{sid}: the package differs from its candidate record")
        for kind, s in e["runs"].items():
            if c is None or s.get("package_sha256") != c.get("package_sha256"):
                raise SystemExit(f"{sid} {kind}: the run is not on the candidate's package")
        if o is not None and c is not None and (o.get("selection") or {}).get("package_sha256") not in (None, c.get("package_sha256")):
            raise SystemExit(f"{sid}: the observability result is not on the candidate's package")
        v = candidate_verdicts(c, e["runs"], o)
        v.update(site_id=sid, cell=cell, k=k, tier=CE.split(cell)[1], features=(c or {}).get("features"),
                 package_sha256=(c or {}).get("package_sha256"), j=(c or {}).get("j"), n_attempts=(c or {}).get("n_attempts"),
                 t_generated=(c or {}).get("t_start"), files=e["files"])
        per[sid] = v
    extra = sorted(set(D) - set(per))
    if extra:
        raise SystemExit(f"records of sites outside the plan: {extra}")
    sel = select(per, list(CE.CELL_ORDER), RULE["candidates"]["per_cell"])
    the_set = [dict(cell=c, site_id=sel[c]["site_id"]) for c in CE.CELL_ORDER if sel[c]["state"] == "selected"]
    complete = len(the_set) == len(CE.CELL_ORDER)
    return dict(schema=SCHEMA_DECIDED, tool_version=TOOL_VERSION, rule=rule_doc(), decided_utc=utc(), candidates=per,
                cells=sel, set=the_set, complete=complete, attributes=attribute_check(per, the_set),
                firsts=first_times(per))


def first_times(per: dict) -> dict:
    ts = sorted(v["t_generated"] for v in per.values() if v.get("t_generated"))
    return dict(first_candidate_generation_utc=ts[0] if ts else None, last=ts[-1] if ts else None)


def attribute_check(per: dict, the_set: list) -> dict:
    """The set's secondary attributes, read from each package's site.json, against layout.cell.* and layout.counts."""
    from cabincell import server as SV
    from cabincell.site import spec as SPEC
    v = SPEC.values()
    rows, probs = {}, []
    cnt = dict(low_battery=0, wrong_advisory=0, failed_light=0, moved_stowage_unit=0)
    for e in the_set:
        c = per[e["site_id"]]
        site_d, _m, _sha = SV.open_site(c["files"]["package"])
        want = v[f"layout.cell.{e['cell']}"]
        got = attributes_of_site(site_d, v)
        rows[e["site_id"]] = dict(want=want, got=got)
        for key in ("module", "tier", "battery", "advisory", "light_circuit", "stowage_unit"):
            if got.get(key) != want.get(key):
                probs.append(f"{e['site_id']}: {key} {got.get(key)} != {want.get(key)}")
        if want.get("blocked_unit_limit") == "below" and got.get("blocked_unit_over_limit"):
            probs.append(f"{e['site_id']}: blocked unit over its advisory limit")
        cnt["low_battery"] += got.get("battery") == "low"
        cnt["wrong_advisory"] += got.get("advisory") == "wrong_unit"
        cnt["failed_light"] += got.get("light_circuit") == "failed"
        cnt["moved_stowage_unit"] += got.get("stowage_unit") == "moved"
    want_cnt = v["layout.counts"]
    if len(the_set) == 6:
        probs += [f"count {k} {cnt[k]} != {want_cnt[k]}" for k in want_cnt if cnt.get(k) != want_cnt[k]]
    return dict(rows=rows, counts=cnt, want_counts=want_cnt, ok=not probs and len(the_set) == 6, problems=probs)


def attributes_of_site(site_d: dict, values: dict) -> dict:
    """The secondary attributes of a site from its description (truth, robot, lighting)."""
    T = site_d.get("truth") or {}
    m = (site_d.get("cell") or {}).get("module")
    b = (site_d.get("robot") or {}).get("battery_pct")
    band = None
    for name, (lo, hi) in sorted(values["world.battery_start_pct"].items()):
        if b is not None and lo - 1e-9 <= float(b) <= hi + 1e-9:
            band = name
    failed = T.get("failed_light_module")
    moved = T.get("moved_stowage_unit")
    return dict(module=str(m).upper() if m else None, tier=(site_d.get("cell") or {}).get("tier"), battery=band, battery_pct=b,
                advisory="wrong_unit" if T.get("advisory_points_elsewhere") else "none",
                blocked_unit_over_limit=bool(T.get("blocked_unit_over_limit")),
                light_circuit="failed" if failed else "ok", failed_light_module=failed,
                lighting_blocked_module=(site_d.get("lighting") or {}).get(m),
                stowage_unit="moved" if moved else "registered", moved_stowage_unit=moved,
                blocked_inlet=T.get("blocked_inlet"))


def plan_runs(decided: dict) -> dict:
    """The robustness runs still to make, by the rule's priority (1: physics and observability passed, k ascending per
    cell; 2: physics passed, observability failed or missing)."""
    pri1, pri2 = [], []
    for sid, v in sorted(decided["candidates"].items(), key=lambda kv: (kv[1]["cell"], kv[1]["k"])):
        if not v["physics"]["ok"]:
            continue
        for kind in ("um1", "cpu2"):
            if v[kind]["ok"] is None:
                (pri1 if v["observability"]["ok"] else pri2).append(dict(site_id=sid, kind=kind, cell=v["cell"], k=v["k"]))
    return dict(priority1=pri1, priority2=pri2)


def reevaluate(decided: dict, obs_dir: str) -> dict:
    """The contingency: the same order with fresh observability verdicts for every candidate."""
    fresh = {}
    for f in sorted(os.listdir(obs_dir)):
        if f.endswith(".obs.json"):
            x = jload(os.path.join(obs_dir, f))
            fresh[x["site_id"]] = x
    per = json.loads(json.dumps(decided["candidates"]))
    missing = [sid for sid, v in per.items() if v["physics"]["generation"] and sid not in fresh]
    for sid, v in per.items():
        o = fresh.get(sid)
        if o is not None and (o.get("selection") or {}).get("package_sha256") not in (None, v.get("package_sha256")):
            raise SystemExit(f"{sid}: the fresh observability result is not on the candidate's package")
        v["observability_before"] = v["observability"]
        v["observability"] = verdict_obs(o)
    from gen import cells as CE
    sel = select(per, list(CE.CELL_ORDER), RULE["candidates"]["per_cell"])
    the_set = [dict(cell=c, site_id=sel[c]["site_id"]) for c in CE.CELL_ORDER if sel[c]["state"] == "selected"]
    before = {e["cell"]: e["site_id"] for e in decided["set"]}
    out = dict(decided, candidates=per, cells=sel, set=the_set, complete=len(the_set) == 6, reevaluated_utc=utc(),
               reevaluated_from=dict(obs_dir=obs_dir, n=len(fresh), missing=missing),
               changed_cells=sorted(c for c in CE.CELL_ORDER if before.get(c) != sel[c]["site_id"]),
               attributes=attribute_check(per, the_set))
    out["pending_runs"] = plan_runs(out)
    return out


def write_hidden(decided: dict, hidden: str, freeze_path: str, commit: str | None = None) -> dict:
    """tests/hidden/sites/<site_id>.site.tar.xz (byte copies), selection.json (the grader's `set`), candidates.json."""
    import shutil
    if not decided.get("complete"):
        raise SystemExit("the set is not complete: no hidden layout is written")
    F = jload(freeze_path)
    sdir = os.path.join(hidden, "sites")
    os.makedirs(sdir, exist_ok=True)
    entries = []
    for e in decided["set"]:
        v = decided["candidates"][e["site_id"]]
        src = v["files"]["package"]
        dst = os.path.join(sdir, f"{e['site_id']}.site.tar.xz")
        shutil.copyfile(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        sha = sha_file(dst)
        if sha != v["package_sha256"]:
            raise SystemExit(f"{dst}: sha256 {sha} != {v['package_sha256']}")
        entries.append(dict(tag=e["site_id"], cell=e["cell"], package=f"sites/{e['site_id']}.site.tar.xz", sha256=sha,
                            bytes=os.path.getsize(dst), seed=None, ref_energy_wh=v["physics"]["energy_wh"],
                            k=v["k"], attempt=v["j"]))
    for f in os.listdir(sdir):
        if f.endswith(".site.tar.xz") and f not in {os.path.basename(x["package"]) for x in entries}:
            raise SystemExit(f"{sdir}/{f}: not in the set")
    cand_rows = {}
    for sid, v in sorted(decided["candidates"].items()):
        cand_rows[sid] = dict(cell=v["cell"], k=v["k"], tier=v["tier"], attempt=v["j"], attempts=v.get("n_attempts"),
                              package_sha256=v.get("package_sha256"), features=v.get("features"),
                              physics=dict(ok=v["physics"]["ok"], why=v["physics"]["why"], energy_wh=v["physics"].get("energy_wh"),
                                           t_docked=v["physics"].get("t_docked"), battery_at_docked=v["physics"].get("battery_at_docked")),
                              observability=dict(ok=v["observability"]["ok"], fails=v["observability"].get("fails"),
                                                 why=v["observability"]["why"]),
                              um1=dict(ok=v["um1"]["ok"], why=v["um1"]["why"]), cpu2=dict(ok=v["cpu2"]["ok"], why=v["cpu2"]["why"],
                                                                                    same_hashes=v["cpu2"].get("same_hashes")),
                              selected=any(x["site_id"] == sid for x in decided["set"]))
    cand_doc = dict(schema="caretaking.hidden_candidates/1", rule_version=decided["rule"]["rule_version"], label=decided["rule"]["candidates"]["label"],
                    candidates=cand_rows)
    cand_sha = jdump(cand_doc, os.path.join(hidden, "candidates.json"))
    sel_doc = dict(schema=SCHEMA_HIDDEN, rule_version=decided["rule"]["rule_version"], tool_version=TOOL_VERSION,
                   set=entries, complete=True,
                   frozen=dict(created_utc=F["created_utc"], files_digest=F["files_digest"], files=F["files"],
                               archives={c: {k: x for k, x in e.items() if k != "file"} for c, e in F["archives"].items()},
                               generator_sha256=F["generator_sha256"], runtime_site_spec=F["runtime_site_spec"],
                               rule_sha256=F["rule_sha256"], commit=commit, check_observability_version=F["check_observability_version"],
                               firmware_version=F["firmware_version"]),
                   first_candidate_generation_utc=decided["firsts"]["first_candidate_generation_utc"],
                   cells={c: dict(state=x["state"], site_id=x["site_id"], counts=x["counts"],
                                  dropped_before=[dict(site_id=d["site_id"], why=d["why"]) for d in x["dropped_before"]])
                          for c, x in decided["cells"].items()},
                   attributes=dict(counts=decided["attributes"]["counts"], ok=decided["attributes"]["ok"]),
                   candidates_sha256=cand_sha, decided_utc=decided["decided_utc"])
    sha = jdump(sel_doc, os.path.join(hidden, "selection.json"))
    return dict(selection_sha256=sha, candidates_sha256=cand_sha, entries=entries)


# ============================================================================================== command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("rule")
    s.add_argument("--out", required=True)
    s = sub.add_parser("check-rule")
    s.add_argument("--rule", required=True)
    s = sub.add_parser("freeze")
    s.add_argument("--rule", required=True)
    s.add_argument("--dev", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--archives", default=None)
    s = sub.add_parser("check-freeze")
    s.add_argument("--freeze", required=True)
    s.add_argument("--dev", default=None)
    s.add_argument("--archives", default=None)
    s = sub.add_parser("generate")
    s.add_argument("cell")
    s.add_argument("k", type=int)
    s.add_argument("--out", required=True)
    s.add_argument("--freeze", required=True)
    s.add_argument("--dev", default=None)
    s.add_argument("--schema", default=None)
    s = sub.add_parser("run")
    s.add_argument("site")
    s.add_argument("--kind", required=True, choices=RUN_KINDS)
    s.add_argument("--out", required=True)
    s.add_argument("--freeze", required=True)
    s.add_argument("--dev", default=None)
    s.add_argument("--ref-energy-wh", type=float, default=None)
    s = sub.add_parser("observe")
    s.add_argument("site")
    s.add_argument("--out", required=True)
    s.add_argument("--freeze", required=True)
    s.add_argument("--dev", default=None)
    s.add_argument("--frames", default=None)
    s.add_argument("--look-root", default=None)
    s.add_argument("--obs-dir", default="obs")
    s = sub.add_parser("decide")
    s.add_argument("--camp", action="append", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--obs-dir", default="obs")
    s = sub.add_parser("plan-runs")
    s.add_argument("--decided", required=True)
    s = sub.add_parser("reevaluate")
    s.add_argument("--decided", required=True)
    s.add_argument("--obs", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("write-hidden")
    s.add_argument("--decided", required=True)
    s.add_argument("--hidden", required=True)
    s.add_argument("--frozen", required=True)
    s.add_argument("--commit", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "rule":
        jdump(rule_doc(), a.out)
        print(sha_file(a.out))
        return 0
    if a.cmd == "check-rule":
        ok = jload(a.rule) == rule_doc()
        print("rule file = RULE" if ok else "rule file DIFFERS from RULE")
        return 0 if ok else 1
    if a.cmd == "freeze":
        F = freeze_doc(a.rule, a.dev, a.archives)
        jdump(F, a.out)
        print(json.dumps(dict(files=F["files_count"], files_digest=F["files_digest"], generator=F["generator_sha256"],
                              created=F["created_utc"])))
        return 0
    if a.cmd == "check-freeze":
        probs = check_freeze(a.freeze, a.dev, a.archives)
        print("\n".join(probs) if probs else "tree = freeze")
        return 1 if probs else 0
    if a.cmd == "generate":
        F = _guard(a)
        r = generate(a.cell, a.k, a.out, a.schema, F)
        print(json.dumps(dict(site=r["site_id"], generated=r["generated"], j=r.get("j"), attempts=r["n_attempts"],
                              package_sha256=r.get("package_sha256"), wall=r["wall_s"])), flush=True)
        return 0
    if a.cmd == "run":
        _guard(a)
        r = run_reference(a.site, a.kind, a.out, a.ref_energy_wh)
        print(json.dumps(dict(site=r["site_id"], kind=a.kind, score=r["grade"]["score"], raw=r["grade"]["raw"],
                              full=r["grade"]["full"], energy=r["energy_wh"], end=r["end"], state=r["state_sha256"][:12],
                              wall=r["wall_s"], cpu=r["cpu"])), flush=True)
        return 0
    if a.cmd == "observe":
        _guard(a)
        r = observe(a.site, a.out, a.frames, a.look_root, a.obs_dir)
        print(json.dumps(dict(site=r["site_id"], passed=r["passed"], fails=r["fails"], wall=r["wall_s"])), flush=True)
        return 0
    if a.cmd == "decide":
        d = decide(a.camp, a.obs_dir)
        jdump(d, a.out)
        print(json.dumps(dict(complete=d["complete"], set=d["set"], cells={c: x["state"] for c, x in d["cells"].items()})))
        return 0
    if a.cmd == "plan-runs":
        print(json.dumps(plan_runs(jload(a.decided)), indent=1))
        return 0
    if a.cmd == "reevaluate":
        d = reevaluate(jload(a.decided), a.obs)
        jdump(d, a.out)
        print(json.dumps(dict(complete=d["complete"], set=d["set"], changed=d["changed_cells"], pending=d["pending_runs"])))
        return 0
    if a.cmd == "write-hidden":
        r = write_hidden(jload(a.decided), a.hidden, a.frozen, a.commit)
        print(json.dumps(r, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
