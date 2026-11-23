"""The hidden set's selection by its registered rule (a maintainer's tool; the grader never runs it). It runs inside
the task image (the grading environment), on the frozen tree's own cell, generator and configuration.

    python3 -B -s select_hidden.py rule
    python3 -B -s select_hidden.py run --tests <tree>/tests --reference <folder of controller.py> --salt draw|<32 hex>
                                       --label <set label> --out <folder> --rule-file <tree>/tests/hidden/selection_rule.txt
                                       [--jobs 4] [--max-k 400]
    python3 -B -s select_hidden.py admit-one --tests ... --reference ... --package <shift.json> --out <run.json>

THE RULE (``RULE`` below is the text that goes into the selection file). Eight shifts. Slot order and belt class:
1 slow, 2 medium, 3 fast, 4 medium, 5 fast, 6 slow, 7 medium, 8 fast (the classes of grader_config.json
shifts.belt_speed_classes). One salt of 16 bytes is drawn once from the system's entropy at selection time
(``--salt draw``) and recorded in the selection file; a candidate's seed is
``shiftgen.labelled_seed("<set label>-<slot>", k, mix=salt)`` (sha256 of the task id, the slot label, the version,
the counter k = 0, 1, 2, ... and the salt). The frozen generator draws each candidate (``shiftgen.draw_shift`` with the
slot's belt class: the belt speed inside the class). A slot takes the FIRST candidate that meets all of:
  (q1) at least MIN_W (6) in-duty parcels arriving label not up (class W), of which at least 3 with the label
       underneath and at least 2 with the label on the downstream face;
  (q2) at least 2 cartons arriving label-up;
  (q3) at least 2 mailers;
  (q4) at least 3 cartons outside the duty (class X);
  (q5) the shift's natural end at most 600 s: last arrival + 2.1 m / belt speed + 40 s (the last parcel's ride from
       the entry line past the scan line, and the tail), computed from the arrivals and the belt speed, never
       from the package's duration field (every full shift lasts the same fixed time; true of every generated shift);
  (q6) at least 2 in-duty cartons (class W) whose length along the belt is more than 0.26 m (the side that lies
       along the belt as the carton lands: the second side at a landing yaw near 0 or a half turn, the first side
       near a quarter turn);
  (a)  the frozen reference, run as the grader drives the oracle (a privileged cell built from grader_config.json,
       ``describe()``, one frame of every camera, ``reference_run()``, ``finish()``) with the grader's numerics
       settings, leaves every in-duty parcel and every label-up parcel readable at the scan line, loses or damages
       nothing, and has no protective stop and no limiter clip (and the cell did not fail: no internal error, no frame
       lost, no render failure, the reference program ended normally);
  (b)  a second run of the same candidate in a fresh process gives the identical record (the shift's record without
       its wall-clock and render-timing fields, the final physics state digest and the limiter count).
Every rejected candidate is recorded with its reason. Nothing else influences the choice; no candidate is looked at
with a camera-driven program before the set is frozen. The quotas (q1-q6) need only the generator; the reference runs
(a, b) only for candidates that meet them. Each slot is processed independently in k order (slots run side by side;
the result does not depend on the order in which runs finish).

``run`` writes <out>/selection.json (salt, rule, generator version, sha256 of every frozen file, every candidate with
its class counts, duration, quota result, admission result and reasons, the run digests), <out>/shifts/<label>_NN.json
(the eight chosen shifts in the tree's shift package format), <out>/seeds.json (the hidden/seeds.json form of the set)
and <out>/runs/ (the full record of every reference run).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

SLOTS = ("slow", "medium", "fast", "medium", "fast", "slow", "medium", "fast")
MIN_W = 6                 # (q1): in-duty parcels arriving label not up, at least (the one value: RULE, the quota and the
                          # documents' "at least N W parcels" follow from it)
MIN_W_UNDERNEATH = 3      # (q1): of them with the label underneath, at least
MIN_W_DOWNSTREAM = 2      # (q1): of them with the label on the downstream face, at least
MIN_CARTON_LABEL_UP = 2   # (q2)
MIN_MAILERS = 2           # (q3)
MIN_CARTONS_OUTSIDE = 3   # (q4)
LONG_ALONG_M = 0.26       # (q6): an in-duty carton longer than this along the belt counts
MIN_W_LONG = 2            # (q6)
MAX_NATURAL_END_S = 600.0
RIDE_M = 2.1              # entry line to scan line (m)
TAIL_S = 40.0
CAMERAS = ("infeed", "head", "wrist_left", "wrist_right")
RULE = ("Eight shifts. Slot order and belt class: 1 slow, 2 medium, 3 fast, 4 medium, 5 fast, 6 slow, 7 medium, "
        "8 fast. One salt of 16 bytes is drawn once from the system's entropy at selection time and recorded here; a "
        "candidate's seed is labelled_seed('<set label>-<slot>', k, mix=salt) for k = 0, 1, 2, ... The frozen "
        "generator draws each candidate (belt speed inside the slot's class). A slot takes the FIRST candidate that "
        "meets all of: (q1) at least %d in-duty parcels arriving label not up, of which at least %d with the label "
        "underneath and at least %d with the label on the downstream face; (q2) at least %d cartons arriving "
        "label-up; (q3) at least %d mailers; (q4) at least %d cartons outside the duty; (q5) the shift's natural end "
        "(last arrival + 2.1 m / belt speed + 40 s, from the arrivals and the belt speed) at most 600 s (true of every "
        "generated shift); (q6) at least %d in-duty cartons whose length along the belt is more than %s m; (a) the "
        "frozen reference, through the privileged path on the task's own cell with the task's numerics settings, "
        "leaves every in-duty parcel and every label-up parcel readable at the scan line, loses or damages nothing, "
        "and has no protective stop and no limiter clip; (b) a second run in a fresh process gives the identical "
        "record. Every rejected candidate is recorded with its reason. Nothing else may influence the choice; no "
        "candidate is run with a camera-driven program before the set is frozen."
        % (MIN_W, MIN_W_UNDERNEATH, MIN_W_DOWNSTREAM, MIN_CARTON_LABEL_UP, MIN_MAILERS, MIN_CARTONS_OUTSIDE,
           MIN_W_LONG, "%g" % LONG_ALONG_M))
# the grader's numerics settings (tests/test.sh exports them for the verify stage)
GRADER_ENV = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256",
              "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR", "OMP_NUM_THREADS": "2",
              "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "LP_NUM_THREADS": "8",
              "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "PYTHONNOUSERSITE": "1",
              "PYTHONDONTWRITEBYTECODE": "1"}
# record fields that measure wall-clock or render time (never compared between runs)
TIMING_KEYS = {"wall_s", "wall_time_s", "build_wall_s", "program_time_s", "cell_work_s", "tick_wall_s",
               "render_wall_s", "render_median_s", "renderer", "program_budget_s", "plan_s"}


def sha256_file(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def tree_hashes(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d != "__pycache__")
        for f in sorted(fn):
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root)] = sha256_file(p)
    return out


def frozen_hashes(tests, reference):
    h = {"parcelcell/" + k: v for k, v in tree_hashes(os.path.join(tests, "parcelcell")).items()}
    h.update({"reference/" + k: v for k, v in tree_hashes(reference).items()})
    h["grader_config.json"] = sha256_file(os.path.join(tests, "grader_config.json"))
    return h


def setup_path(tests):
    tests = os.path.abspath(tests)
    if tests not in sys.path:
        sys.path.insert(0, tests)
    sys.dont_write_bytecode = True
    return tests


def load_cfg(tests):
    from parcelcell.config import load_config
    return load_config(os.path.join(tests, "grader_config.json"))


# ------------------------------------------------------------------------------------------------ candidates
def slot_label(label, slot):
    return "%s-%d" % (label, slot)


def draw_candidate(cfg, label, slot, k, salt):
    from parcelcell import shiftgen
    sc = cfg["shifts"]
    lab = slot_label(label, slot)
    seed = shiftgen.labelled_seed(lab, k, task_id=sc["task_id"], version=sc["version"], mix=salt)
    sh, meta = shiftgen.draw_shift(seed, cfg, profile="full", belt_class=SLOTS[slot - 1], label=lab, k=k)
    return sh, meta


def along_belt_m(p):
    """The side of a landed parcel that lies along the belt (the belt runs along y; the landing yaw is a quarter
    turn k x 90 degrees within the yaw tolerance: the second side for even k, the first for odd k)."""
    import math
    k90 = int(round(float(p.yaw) / (math.pi / 2))) % 4
    return float(p.size[1] if k90 % 2 == 0 else p.size[0])


def counts(shift, cfg):
    from parcelcell.episode import label_position, parcel_class
    duty = (cfg.get("score") or {}).get("duty")
    c = {"n_parcels": 0, "W": 0, "U": 0, "X": 0, "W_underneath": 0, "W_downstream": 0, "carton_label_up": 0,
         "mailers": 0, "cartons_outside_duty": 0, "W_long": 0}
    for a in shift.arrivals:
        p = a.parcel
        cls, pos = parcel_class(p, duty), label_position(p)
        c["n_parcels"] += 1
        c[cls] += 1
        if cls == "W" and pos == "underneath":
            c["W_underneath"] += 1
        if cls == "W" and pos == "downstream":
            c["W_downstream"] += 1
        if cls == "W" and p.kind == "rigid" and along_belt_m(p) > LONG_ALONG_M + 1e-9:
            c["W_long"] += 1
        if p.kind == "rigid" and pos == "top":
            c["carton_label_up"] += 1
        if p.kind == "soft":
            c["mailers"] += 1
        if p.kind == "rigid" and cls == "X":
            c["cartons_outside_duty"] += 1
    c["duration_s"] = float(shift.duration)
    c["belt_speed"] = float(shift.belt_speed[0][1])
    c["last_arrival_s"] = float(max(a.t for a in shift.arrivals))
    c["natural_end_s"] = round(c["last_arrival_s"] + RIDE_M / c["belt_speed"] + TAIL_S, 3)
    return c


def quota_reasons(c):
    why = []
    if c["W"] < MIN_W or c["W_underneath"] < MIN_W_UNDERNEATH or c["W_downstream"] < MIN_W_DOWNSTREAM:
        why.append("q1: in-duty %d (underneath %d, downstream %d)" % (c["W"], c["W_underneath"], c["W_downstream"]))
    if c["carton_label_up"] < MIN_CARTON_LABEL_UP:
        why.append("q2: %d cartons arriving label-up" % c["carton_label_up"])
    if c["mailers"] < MIN_MAILERS:
        why.append("q3: %d mailers" % c["mailers"])
    if c["cartons_outside_duty"] < MIN_CARTONS_OUTSIDE:
        why.append("q4: %d cartons outside the duty" % c["cartons_outside_duty"])
    if c["natural_end_s"] > MAX_NATURAL_END_S:
        why.append("q5: natural end %.1f s" % c["natural_end_s"])
    if c["W_long"] < MIN_W_LONG:
        why.append("q6: %d in-duty cartons longer than %g m along the belt" % (c["W_long"], LONG_ALONG_M))
    return why


def ride_check(cfg):
    """The configuration's entry-to-scan ride and tail beside the rule's constants (recorded, not used)."""
    w, sc = cfg["world"], cfg["shifts"]
    ride = round(float(w["scan_y"]) - float(w["entry_y"]), 6)
    return {"ride_m": ride, "tail_s": sc.get("tail_s"), "equal_to_rule": ride == RIDE_M and sc.get("tail_s") == TAIL_S}


# ------------------------------------------------------------------------------------------------ one run
def canonical(obj):
    if isinstance(obj, dict):
        return {k: canonical(v) for k, v in sorted(obj.items()) if k not in TIMING_KEYS}
    if isinstance(obj, list):
        return [canonical(v) for v in obj]
    return obj


def cmd_admit_one(a):
    tests = setup_path(a.tests)
    from parcelcell.server import ParcelCellServer
    cfg = load_cfg(tests)
    t0 = time.time()
    srv = ParcelCellServer(config=cfg, mode="hidden", privileged=True, reference_dir=os.path.abspath(a.reference))
    srv.reset(a.package)
    srv.wall_start = time.time()
    srv.handle("describe", {})
    canary = {}
    for name in CAMERAS:                                   # as the oracle's sort.py asks
        r = srv.handle("get_camera", {"name": name, "depth": True})
        canary[name] = r.get("error") or "ok"
    ref = srv.handle("reference_run", {})
    srv.handle("finish", {})
    srv.wall_used = time.time() - srv.wall_start
    rec = srv.finalize()
    w = srv.world
    out = {"package": os.path.basename(a.package), "package_sha256": sha256_file(a.package),
           "record": rec, "canary": canary,
           "reference_reply": {k: ref.get(k) for k in ("ok", "error", "detail")} if isinstance(ref, dict) else None,
           "clip_ticks": int(w.n_slew_clip_ticks), "state_digest": w.state_digest(),
           "wall_s": round(time.time() - t0, 1), "pid": os.getpid(),
           "env": {k: os.environ.get(k) for k in GRADER_ENV}}
    canon = {"record": canonical(rec), "clip_ticks": out["clip_ticks"], "state_digest": out["state_digest"]}
    out["record_digest"] = hashlib.sha256(json.dumps(canon, sort_keys=True, default=float).encode()).hexdigest()
    srv.close()
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    return 0


def admission_reasons(run):
    rec, why = run["record"], []
    if rec.get("internal_errors") or rec["frames"].get("frames_lost") or rec["frames"].get("render_failures"):
        why.append("a: cell failure (internal errors %s, frames lost %s, render failures %s)"
                   % (rec.get("internal_errors"), rec["frames"].get("frames_lost"), rec["frames"].get("render_failures")))
    rr = run.get("reference_reply") or {}
    if not rr.get("ok"):
        why.append("a: the reference run did not end normally (%s)" % (rr.get("error") or "no reply"))
    bad_c = [n for n, v in run["canary"].items() if v != "ok"]
    if bad_c:
        why.append("a: camera canary failed (%s)" % ", ".join(bad_c))
    if rec["done_reason"] not in ("shift_end", "finished"):
        why.append("a: shift ended by %s" % rec["done_reason"])
    W_bad = [p["i"] for p in rec["parcels"] if p["cls"] == "W" and not p["readable"]]
    U_bad = [p["i"] for p in rec["parcels"] if p["cls"] == "U" and not p["readable"]]
    if W_bad:
        why.append("a: in-duty parcels not readable %s" % W_bad)
    if U_bad:
        why.append("a: label-up parcels not readable %s" % U_bad)
    lost = [(p["i"], p["lost_reason"]) for p in rec["parcels"] if p["lost"]]
    if lost:
        why.append("a: parcels lost %s" % lost)
    if rec.get("parcel_damage"):
        why.append("a: parcels damaged %s" % sorted(rec["parcel_damage"]))
    s = rec["safety"]
    if s.get("protective_stop_count") or s.get("locked_out"):
        why.append("a: %s protective stops%s" % (s.get("protective_stop_count"), ", locked out" if s.get("locked_out") else ""))
    if run["clip_ticks"]:
        why.append("a: %d limiter clip ticks" % run["clip_ticks"])
    return why


# ------------------------------------------------------------------------------------------------ orchestration
class Runner(object):
    def __init__(self, a):
        self.a = a
        self.sem = threading.Semaphore(max(1, int(a.jobs)))
        self.lock = threading.Lock()
        self.log_fh = open(os.path.join(a.out, "selection.log"), "a")

    def log(self, msg):
        with self.lock:
            line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
            print(line, flush=True)
            self.log_fh.write(line + "\n")
            self.log_fh.flush()

    def one_run(self, package, out):
        env = dict(os.environ)
        env.update(GRADER_ENV)
        argv = [sys.executable, "-B", "-s", os.path.abspath(__file__), "admit-one", "--tests", self.a.tests,
                "--reference", self.a.reference, "--package", package, "--out", out]
        with self.sem:
            t0 = time.time()
            p = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=7200)
        with open(out + ".log", "wb") as fh:
            fh.write(p.stdout)
        if p.returncode != 0 or not os.path.exists(out):
            return None, round(time.time() - t0, 1)
        with open(out) as fh:
            return json.load(fh), round(time.time() - t0, 1)

    def slot(self, slot, cfg, salt):
        from parcelcell.shiftgen import save_package
        a = self.a
        rows, chosen = [], None
        for k in range(a.max_k):
            t_c = time.time()
            sh, meta = draw_candidate(cfg, a.label, slot, k, salt)
            c = counts(sh, cfg)
            row = {"slot": slot, "belt_class": SLOTS[slot - 1], "k": k, "seed": int(meta["seed"]), "counts": c,
                   "quota_reasons": quota_reasons(c)}
            if row["quota_reasons"]:
                row.update(result="rejected", reasons=row["quota_reasons"])
                rows.append(row)
                continue
            pk = os.path.join(a.out, "candidates", "slot%d_k%03d.json" % (slot, k))
            row["package_sha256"] = save_package(pk, sh, meta)
            outs = [os.path.join(a.out, "runs", "slot%d_k%03d_run%d.json" % (slot, k, i)) for i in (1, 2)]
            with cf.ThreadPoolExecutor(2) as ex:                       # the two runs: separate fresh processes
                res = list(ex.map(lambda o: self.one_run(pk, o), outs))
            runs = [r for r, _ in res]
            row["run_wall_s"] = [w for _, w in res]
            if runs[0] is None or runs[1] is None:
                row.update(result="rejected", reasons=["a: a reference run process failed (see runs/*.log)"])
                rows.append(row)
                self.log("slot %d k %d: run process failed" % (slot, k))
                continue
            why = admission_reasons(runs[0])
            row["run_digests"] = [r["record_digest"] for r in runs]
            row["run_summary"] = {"shift_score": runs[0]["record"]["shift_score"],
                                  "credit": runs[0]["record"]["credit"], "t_end": runs[0]["record"]["t_end"],
                                  "ticks": runs[0]["record"]["ticks"], "done_reason": runs[0]["record"]["done_reason"],
                                  "W_readable": runs[0]["record"]["W_readable"], "n_W": runs[0]["record"]["n_W"],
                                  "U_not_readable": runs[0]["record"]["U_not_readable"],
                                  "lost": runs[0]["record"]["lost"],
                                  "stops": runs[0]["record"]["safety"].get("protective_stop_count"),
                                  "clip_ticks": runs[0]["clip_ticks"], "frames_used": runs[0]["record"]["frames"]["used"],
                                  "state_digest": runs[0]["state_digest"]}
            if runs[0]["record_digest"] != runs[1]["record_digest"]:
                why.append("b: the second run's record differs (%s)" % ", ".join(diff_paths(runs[0], runs[1])[:6]))
            row["candidate_wall_s"] = round(time.time() - t_c, 1)
            if why:
                row.update(result="rejected", reasons=why)
                rows.append(row)
                self.log("slot %d k %d rejected: %s" % (slot, k, "; ".join(why)))
                continue
            row.update(result="chosen", reasons=[])
            rows.append(row)
            chosen = (k, sh, meta, row)
            self.log("slot %d k %d CHOSEN (score %.4f, %d W, natural end %.0f s)" % (
                slot, k, row["run_summary"]["shift_score"], c["W"], c["natural_end_s"]))
            break
        return rows, chosen


def diff_paths(r1, r2):
    out = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k in TIMING_KEYS:
                    continue
                walk(x.get(k), y.get(k), path + "." + str(k))
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, "%s[%d]" % (path, i))
        elif x != y:
            out.append(path)
    walk({"record": r1["record"], "clip_ticks": r1["clip_ticks"], "state_digest": r1["state_digest"]},
         {"record": r2["record"], "clip_ticks": r2["clip_ticks"], "state_digest": r2["state_digest"]}, "")
    return out


def cmd_run(a):
    tests = setup_path(a.tests)
    a.tests, a.reference = tests, os.path.abspath(a.reference)
    for d in ("", "candidates", "runs", "shifts"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    if a.salt == "draw":
        salt = os.urandom(16)
        salt_source = "os.urandom(16) at selection time"
    else:
        salt = bytes.fromhex(a.salt)
        if len(salt) != 16:
            sys.exit("the salt must be 16 bytes (32 hex digits)")
        salt_source = "given on the command line"
    with open(a.rule_file, "rb") as fh:                       # the registered rule, written before the draw
        rule_raw = fh.read()
    registered = {"file": os.path.basename(a.rule_file), "sha256": hashlib.sha256(rule_raw).hexdigest(),
                  "text": rule_raw.decode("utf-8")}
    from parcelcell import shiftgen
    from parcelcell.shiftgen import save_package
    cfg = load_cfg(tests)
    frozen = frozen_hashes(tests, a.reference)
    R = Runner(a)
    R.log("selection: label %s, salt %s (%s), %d jobs" % (a.label, salt.hex(), salt_source, a.jobs))
    t0 = time.time()
    with cf.ThreadPoolExecutor(len(SLOTS)) as ex:
        futs = [ex.submit(R.slot, s, cfg, salt) for s in range(1, len(SLOTS) + 1)]
        results = [f.result() for f in futs]
    frozen_after = frozen_hashes(tests, a.reference)
    rows = [r for rs, _ in results for r in rs]
    chosen = [c for _, c in results]
    set_rows = []
    for slot, c in enumerate(chosen, start=1):
        if c is None:
            continue
        k, sh, meta, row = c
        name = "%s_%02d" % (a.label, slot)
        sha = save_package(os.path.join(a.out, "shifts", name + ".json"), sh, meta)
        f = meta["features"]
        set_rows.append({"tag": name, "package": "shifts/%s.json" % name, "sha256": sha, "seed": int(meta["seed"]),
                         "k": k, "label": meta["label"], "slot": slot, "belt_class": meta["belt_class"],
                         "mix_class": meta["mix_class"], "n_parcels": f["n_parcels"],
                         "n_label_not_up": f["n_label_not_up"], "classes": f["classes"],
                         "duration_s": f["duration_s"]})
    n_ref = [r for r in rows if "run_digests" in r or "run_wall_s" in r]
    per_slot = {}
    for s in range(1, len(SLOTS) + 1):
        rs = [r for r in rows if r["slot"] == s]
        per_slot[str(s)] = {"belt_class": SLOTS[s - 1], "candidates": len(rs),
                            "quota_passed": sum(1 for r in rs if not r["quota_reasons"]),
                            "admitted": sum(1 for r in rs if r["result"] == "chosen"),
                            "chosen_k": next((r["k"] for r in rs if r["result"] == "chosen"), None)}
    out = {"placeholder": False, "registered_rule": registered, "rule": RULE, "label": a.label, "salt_hex": salt.hex(), "salt_source": salt_source,
           "generator": shiftgen.GENERATOR_VERSION, "slots": list(SLOTS), "max_natural_end_s": MAX_NATURAL_END_S, "natural_end_rule": {"ride_m": RIDE_M, "tail_s": TAIL_S},
           "config_ride_and_tail": ride_check(cfg),
           "frozen": frozen, "frozen_unchanged_during_selection": frozen == frozen_after,
           "grader_env": GRADER_ENV, "candidates": rows, "per_slot": per_slot, "hidden_set": set_rows,
           "complete": len(set_rows) == len(SLOTS),
           "stats": {"candidates": len(rows), "quota_passed": len(n_ref),
                     "admitted_of_quota_passed": "%d of %d" % (sum(1 for r in rows if r["result"] == "chosen"), len(n_ref)),
                     "rejected_after_runs": [{"slot": r["slot"], "k": r["k"], "reasons": r["reasons"]}
                                             for r in n_ref if r["result"] != "chosen"],
                     "candidate_wall_s": [r.get("candidate_wall_s") for r in n_ref],
                     "selection_wall_s": round(time.time() - t0, 1)},
           "tool_sha256": sha256_file(os.path.abspath(__file__)),
           "selection_run": {"hidden_set": [r["seed"] for r in set_rows], "candidate_seeds": [r["seed"] for r in rows],
                             "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    with open(os.path.join(a.out, "selection.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
        fh.write("\n")
    seeds = {"placeholder": False,
             "note": "The fixed hidden set: eight shifts chosen by the registered rule (selection.json).",
             "seeds": [r["seed"] for r in set_rows], "shifts": set_rows}
    with open(os.path.join(a.out, "seeds.json"), "w") as fh:
        json.dump(seeds, fh, indent=1)
        fh.write("\n")
    R.log("done: %d of %d slots filled; %d candidates, %d ran the reference; %.0f s"
          % (len(set_rows), len(SLOTS), len(rows), len(n_ref), time.time() - t0))
    return 0 if out["complete"] and out["frozen_unchanged_during_selection"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rule")
    r = sub.add_parser("run")
    r.add_argument("--tests", required=True)
    r.add_argument("--reference", required=True)
    r.add_argument("--salt", required=True, help="draw (16 bytes of system entropy) or 32 hex digits")
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--rule-file", required=True, help="the registered rule's text, written before the draw")
    r.add_argument("--jobs", type=int, default=4, help="reference runs at a time")
    r.add_argument("--max-k", type=int, default=400)
    o = sub.add_parser("admit-one")
    o.add_argument("--tests", required=True)
    o.add_argument("--reference", required=True)
    o.add_argument("--package", required=True)
    o.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "rule":
        print(RULE)
        return 0
    if a.cmd == "admit-one":
        return cmd_admit_one(a)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
