"""The worker: a separate process started by verify.py (as the unprivileged account where one exists), under an
address-space limit, on private copies of the grading code, the configuration, the truth, the reference, the givens
the grader reads and the submission.

    worker_main(job_path)  ->  exit 0 (graded), 3 (a task error: the verification is void)

  1. the self-test, under the stage alarm "selftest": the configuration, the givens and the truth are read and checked
     (a defect is a task error); the reference twin is graded FIRST, through the same code as every submission, and
     must score exactly 1 in every term of every flight under every aggregation (grade.check_reference); its
     cameras.json (frozen with the truth) declares the scored moments only, so it is read with the first version's
     entry rule (grade_submission(..., entry_rule="key_frames")); every step after the gate is the submission's;
  2. the submission, every stage under its own alarm (guard.Guard): a stage that trips on the submission zeroes that
     flight with a named reason; the ray scene and its rays are built and cast in a child process
     (grade.scene_in_child) with the scene's and the rays' own limits and the address-space limit, and a stop there at
     a stage's own limit or by memory zeroes the flight under rule Z7 while this process, whose own memory the child
     never used, goes on to the verdict; a stage that trips for any other cause (another stage's alarm, the
     verification's deadline, this process's own address-space limit, the child ending by another signal, an
     internal error) voids the verification: the progress document records a task error and no score.
Every step is written to the progress document (out/progress.json), which verify.py reads.
"""
import json
import math
import os
import sys
import time

import numpy as np

from . import config as config_mod
from . import givens as givens_mod
from . import truth as truth_mod
from .grade import Context, check_reference, grade_submission
from .guard import Guard, GraderVoid

PROGRESS_FORMAT = "flight-twin-worker-progress/1"


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    return x


def _peak_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class Progress:
    def __init__(self, path):
        self.path = path
        self.doc = {"format": PROGRESS_FORMAT, "selftest": None, "submission": None, "task_error": None, "done": False}

    def write(self):
        self.doc["peak_rss_mb"] = _peak_mb()
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_clean(self.doc), f, allow_nan=False, sort_keys=True)
        os.replace(tmp, self.path)


def load_all(job):
    """(cfg, ctx) from the job's private copies; ConfigError, GivenError or TruthError on a defect."""
    cfg = config_mod.load(job["grader_config"])
    gv = givens_mod.load(job["stage"], cfg)
    from .twinprobe.hazard import directions
    n_dirs = len(directions(cfg["hazard.directions_azimuth"], cfg["hazard.directions_elevation"]))
    truth_dir = os.path.join(job["stage"], cfg.layout["truth_dir"])
    ref_dir = os.path.join(job["stage"], cfg.layout["reference_dir"])
    truths = {f: truth_mod.load_flight(truth_dir, ref_dir, f, cfg, gv.rays, n_dirs) for f in cfg["flights.names_neutral"]}
    return cfg, Context(cfg, gv, truths), ref_dir


def selftest(job):
    cfg, ctx, ref_dir = load_all(job)
    res = grade_submission(ref_dir, ctx, Guard(enabled=False), entry_rule="key_frames")
    check_reference(res, ctx)
    return cfg, ctx, {"ok": True, "reference": {f: dict(r["terms"]) for f, r in res["flights"].items()},
                      "reference_twin_score": res["twin_score"], "truth_counts": {f: t.counts() for f, t in ctx.truths.items()}}


def worker_main(job_path):
    import resource
    with open(job_path) as f:
        job = json.load(f)
    lim = int(job["address_space_mb"]) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    prog = Progress(job["progress"])
    prog.write()
    deadline = float(job["deadline_wall"])
    caps = dict(job["stage_caps_s"])
    t0 = time.time()
    try:
        cfg, ctx, st = Guard(deadline, caps).run("selftest", selftest, job)
    except BaseException as exc:  # noqa: BLE001 - a truth that cannot be read or does not reproduce is a task error
        if isinstance(exc, KeyboardInterrupt):
            raise
        import traceback
        prog.doc["selftest"] = {"ok": False, "reason": "%s: %s" % (type(exc).__name__, str(exc)[:1500]),
                                "trace": traceback.format_exc()[-1500:]}
        prog.doc["timing"] = {"selftest_s": time.time() - t0}
        prog.write()
        return 3
    prog.doc["selftest"] = st
    prog.doc["timing"] = {"selftest_s": time.time() - t0}
    prog.write()
    t1 = time.time()
    guard = Guard(deadline, caps)
    try:
        res = grade_submission(job["submission"], ctx, guard)
    except BaseException as exc:  # noqa: BLE001 - GraderVoid and any error outside a stage: the verification is void
        if isinstance(exc, KeyboardInterrupt):
            raise
        import traceback
        reason = str(exc) if isinstance(exc, GraderVoid) else "%s: %s" % (type(exc).__name__, str(exc)[:600])
        prog.doc["task_error"] = {"reason": reason[:1500], "trace": traceback.format_exc()[-1500:]}
        prog.doc["timing"]["submission_s"] = time.time() - t1
        prog.write()
        return 3
    prog.doc["submission"] = res
    prog.doc["timing"]["submission_s"] = time.time() - t1
    for f, st in sorted(guard.stats.items()):       # the ray scene's child process per flight (timing record only)
        for k in ("build_s", "work_s", "peak_rss_mb"):
            if st.get(k) is not None:
                prog.doc["timing"]["%s_scene_child_%s" % (f, k)] = float(st[k])
        prog.doc["timing"]["%s_scene_child_stop" % f] = " ".join(str(st.get(k)) for k in ("stage", "failed", "error")
                                                                  if st.get(k))
    prog.doc["done"] = True
    prog.write()
    return 0


if __name__ == "__main__":  # pragma: no cover - verify.py starts the worker through a bootstrap
    sys.exit(worker_main(sys.argv[1]))
