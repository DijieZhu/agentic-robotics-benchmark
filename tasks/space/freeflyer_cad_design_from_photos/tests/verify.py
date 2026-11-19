"""Grader of freeflyer_cad_design_from_photos (a cabin free-flyer re-modelled in FreeCAD from photographs; the standard is
the maker's published appearance meshes and robot description, as given). Root, verify stage, agent gone.

Where it runs: a separate verifier sandbox (task.toml `environment_mode = "separate"`), started fresh from the task
image; the engine restores only /home/user/submission into it, so nothing else the agent left is present. setup.sh does
not run there. The grader reads tests/ and the image only: the rule values (grader_config.json), the maker's files and
their pins (reference/maker/, read by makerstd/), the derived placement (reference/placement.json, re-derived from the
pinned texts at every run), the graded samples (reference/seen.json and reference/seen/) and the blocklist
(public_blocklist.json). A grader input that is missing or differs from its pin is a task error, not a zero.

  0. FIRST THING, stdlib only (lockdown_runtime._early_lockdown, the UAV pattern, file unchanged): record when the
     engine began uploading this folder, stop and kill every process of the agent account (a survivor that cannot be
     frozen voids the run: task error), freeze it, lock the stage, take a root-owned 0700 snapshot of the submission
     that every later read uses, and zero the score (stat submission_final = 0) when an entry of the submission was
     written at or after the stage upload began. A submission root that is a symbolic link, or that resolves outside
     the agent's home, is not read at all. The directories on root's PATH are reclaimed (robotics_grader.stage
     .secure_exec_path) first.
  1. Tool use from the engine's solver trajectory (ATIF, ALE_TRAJECTORY_PATH), stats only. When the engine marks the
     trajectory incomplete (extra.ale.incomplete, e.g. a conversion that failed), its steps are a fragment: the counts
     are then unknown, and the diagnostic says so instead of printing zeros.
  2. The submitted document (snapshot unit.FCStd, else the only *.FCStd, with a note) is read by FreeCAD's headless
     interpreter as `nobody`, through runuser and /usr/bin/env HOME=..., in a private work folder, with a time limit
     (fcstd_extract.py, the UAV reader unchanged): every shape leaf of the Assembly (else the visible top-level
     objects) in world coordinates, solid by solid, with labels, types and the joints. Before that the document's
     archive is checked (zipfile: central directory and every member's CRC): a document that is cut off or damaged
     cannot be read and scores 0, also when FreeCAD would recover its first members from what is left.
  3. grade.py (a child process with a time limit, single-threaded, CPU features and the BLAS kernel pinned, the kernel
     pin never without its block-size guard OPENBLAS_L2_SIZE, checked before the child starts, else a task error): the
     standard checked, the members and tags, the whole-unit alignment, surface, joint_axes, structure, the floor
     (cadgrade/run.py's docstring states the rules). A failure of the grader itself (its inputs, an exception outside
     a grading item, a crash or a time-out of the child) fails the run as a task error, never as a silent 0.
  4. public_model_reuse: a file of the submission whose sha256 is one of the maker's published files
     (public_blocklist.json) zeroes the design score.
design_score = 0.65 surface + floor x (0.25 joint_axes + 0.10 structure), times public_model_reuse; the verdict's single
gated `reward` comes from the benchmark kit (robotics_grader.ratio_rewards against anchor.json: value 1.0 = the
reference, full_at 1.0), so reward = design_score. `metrics` holds finite numbers only and no wall-clock durations
(every *_s statistic goes to stderr and to the root-only record /opt/arb-verify/out/verify_timing.json instead, so two
runs give the same verdict); the diagnostic is at most 2000 characters. No agent code runs here (the document is data
that FreeCAD reads).
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zipfile

import lockdown_runtime as runtime

runtime.DOCUMENT = "unit.FCStd"     # the runtime is the UAV file byte for byte; only the document name differs
CHECKS = ["cad_valid", "public_model_reuse"]
CRITERIA = ["surface", "joint_axes", "structure"]
DIAG_MAX = 2000
# The grade child's numerical pins. OPENBLAS_L2_SIZE=256 goes with the kernel pin: numpy 2.5.2-2.5.5's bundled OpenBLAS
# enlarges the block size of every kernel table, the pinned one included, on hosts that report a 1024 KB L2 cache with
# AVX-512 (recent AMD desktop and server CPUs), and dgemm_kernel_HASWELL then overruns its stack buffer (a crash, or a
# wrong number with exit 0). Any L2 size other than 1024 disables that change; 256 is the pinned core's own L2, so the
# block sizes, and with them every bit of the grade, are the same on every host.
CHILD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
             "NPY_DISABLE_CPU_FEATURES": "X86_V4,AVX512_ICL", "OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256",
             "PYTHONDONTWRITEBYTECODE": "1"}


def blas_pin_problem(env):
    """None when the environment keeps the numerical-pin rule, else the reason: OPENBLAS_CORETYPE is never set without
    OPENBLAS_L2_SIZE, a positive whole number other than 1024 (unset, empty, unparsable or 1024 leaves the block-size
    change of the comment above in force)."""
    core = str(env.get("OPENBLAS_CORETYPE") or "").strip()
    if not core:
        return None
    l2 = str(env.get("OPENBLAS_L2_SIZE") or "").strip()
    if not l2:
        return "OPENBLAS_CORETYPE=%s is set without OPENBLAS_L2_SIZE" % core
    if not l2.isdigit() or int(l2) <= 0:
        return "OPENBLAS_CORETYPE=%s is set with OPENBLAS_L2_SIZE=%r, not a positive whole number" % (core, l2[:40])
    if int(l2) == 1024:
        return ("OPENBLAS_CORETYPE=%s is set with OPENBLAS_L2_SIZE=1024, the value that leaves the block-size change "
                "in force" % core)
    return None


def tool_usage(stats):
    """runtime.tool_usage (the UAV file, unchanged) unless the engine marked the solver trajectory incomplete
    (extra.ale.incomplete, for example a conversion that failed): its steps are then a fragment, so the tool counts are
    unknown; none is recorded and the text says so. Statistics only, never a gate."""
    path = os.environ.get("ALE_TRAJECTORY_PATH")
    if path and os.path.isfile(path):
        try:
            with open(path) as fh:
                traj = json.load(fh)
        except (OSError, ValueError):
            traj = None
        extra = traj.get("extra") if isinstance(traj, dict) else None
        ale = extra.get("ale") if isinstance(extra, dict) else None
        if isinstance(ale, dict) and ale.get("incomplete"):
            reason = str(ale.get("incomplete_reason") or "no reason given")[:80]
            return "the solver trajectory is incomplete (%s): tool counts unknown" % reason
    return runtime.tool_usage(stats)


def archive_intact(path):
    """(ok, reason): a FreeCAD document is a zip archive; one that is cut off or damaged (no central directory, a member
    whose CRC fails) cannot be read, even when FreeCAD would recover its first members from what is left."""
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            names = z.namelist()
    except (zipfile.BadZipFile, OSError, EOFError, RuntimeError, NotImplementedError, ValueError) as exc:
        return False, "the document is cut off or damaged: not a complete FreeCAD archive (%s)" % str(exc)[:120]
    if bad is not None:
        return False, "the document is damaged: the archive member %s fails its check" % bad[:120]
    if "Document.xml" not in names:
        return False, "the document is not a FreeCAD archive (no Document.xml)"
    return True, None


def public_reuse(root, blocked):
    """Files under the submission whose sha256 is a published maker file (symbolic links are not followed)."""
    reused = []
    if not os.path.isdir(root):
        return reused
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in sorted(filenames):
            p = os.path.join(dirpath, f)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest() in blocked:
                reused.append(os.path.relpath(p, root))
    return reused


def split_timing(stats):
    """(stats without durations, durations): every key ending in _s is a wall-clock duration (kept out of the verdict)."""
    timing = {k: v for k, v in stats.items() if str(k).endswith("_s")}
    return {k: v for k, v in stats.items() if k not in timing}, timing


def _finite_metrics(metrics):
    """`metrics` in name order, finite numbers only (ale_verify.write refuses NaN and infinity; a string would crash)."""
    out = {}
    for k, v in sorted(metrics.items()):
        if isinstance(v, bool):
            v = float(v)
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out[str(k)] = x
    return out


def record_timing(timing):
    try:
        out = os.path.join(runtime.SNAP_ROOT, "out")
        if os.getuid() == 0:
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "verify_timing.json"), "w") as fh:
                json.dump(_finite_metrics(timing), fh, indent=1, sort_keys=True)
    except OSError:
        pass
    runtime.log("timing: " + ", ".join("%s %.1f" % (k, float(v)) for k, v in sorted(_finite_metrics(timing).items())))


def grade_child(extract_dir, stats, timeout_s):
    """grade.py in a child process with a time limit (an OCC call that hangs on a broken shape cannot be interrupted
    from Python): (its result or None, its last words)."""
    work = tempfile.mkdtemp(prefix="arb_grade_")
    out = os.path.join(work, "result.json")
    argv = [sys.executable, "-B", os.path.join(runtime.STAGE_DIR, "grade.py"), extract_dir, out,
            "--config", os.path.join(runtime.STAGE_DIR, "grader_config.json"), "--stage", runtime.STAGE_DIR]
    env = dict(os.environ, **CHILD_ENV)
    problem = blas_pin_problem(env)
    if problem:
        from robotics_grader import fail
        fail("the grade child's numerical pins break the platform rule (the kernel pin never without its block-size "
             "guard): %s; the grade was not started" % problem)
    t = time.time()
    try:
        r = subprocess.run(argv, cwd=runtime.STAGE_DIR, env=env, capture_output=True, text=True, timeout=timeout_s)
        stats["grade_child_exit"] = float(r.returncode)
        tail = (r.stderr or "")[-1500:]
    except subprocess.TimeoutExpired:
        stats["grade_child_exit"] = -1.0
        return None, "the grade did not finish within %d s" % timeout_s
    finally:
        stats["grade_child_s"] = time.time() - t
    if not os.path.isfile(out):
        return None, "the grader child wrote no result: " + tail
    with open(out) as fh:
        return json.load(fh), tail


def grade_metrics(res):
    """The finite per-item numbers of a grade: each tag's pooled p90 and score, each hinge's score, offsets and angles,
    the floor, the counts and the alignment's residual."""
    m = {}
    for k, v in (res.get("statistics") or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            m[k] = v
    m["shape_floor"] = res.get("floor", 0.0)
    for tag, r in sorted((res.get("tags") or {}).items()):
        m["surface_score_" + tag] = r.get("score", 0.0)
        for key, name in (("p90_mm", "surface_p90_mm_"), ("submitted_to_maker_p90_mm", "surface_sub_p90_mm_"),
                          ("maker_to_submitted_p90_mm", "surface_maker_p90_mm_")):
            if isinstance(r.get(key), (int, float)):
                m[name + tag] = r[key]
        if "p90_exact" in r:
            m["surface_p90_exact_" + tag] = 1.0 if r["p90_exact"] else 0.0
    for pair, h in sorted((res.get("hinges") or {}).items()):
        key = pair.replace("-", "_")
        m["joint_score_" + key] = h.get("score", 0.0)
        m["joint_candidates_" + key] = h.get("candidates", 0)
        if h.get("offset_mm"):
            m["joint_offset_mm_" + key] = max(h["offset_mm"])
            m["joint_angle_deg_" + key] = max(h["angle_deg"])
    st = res.get("structure") or {}
    m["structure_connected_share"] = st.get("connected_share", 0.0)
    m["structure_present_share"] = st.get("present_share", 0.0)
    return m


def diagnostic_text(rewards, details):
    text = "design_score %.4f; " % float(rewards.get("design_score", 0.0))
    text += ", ".join("%s %.4f" % (k, float(rewards.get(k, 0.0))) for k in CHECKS + CRITERIA)
    for key in ("submission", "cad", "public_model_reuse", "grade"):
        if details.get(key):
            text += "; " + " | ".join(str(d) for d in details[key])
    if len(text) > DIAG_MAX:
        text = text[:DIAG_MAX - 3] + "..."
    return text


def write_verdict(overall, rewards, stats, details):
    """One gated `reward` in the kit's envelope (ratio_rewards against anchor.json: value 1.0 = the reference, full_at
    1.0, so reward = design_score); the checks, criteria and statistics go to `metrics` (finite numbers only), the
    diagnostics into the verdict's `diagnostic` text (<= 2000 characters). In the engine (ALE_VERIFY_PYTHON set) the kit
    writes /logs/verifier/reward.json and verification.json through ale_verify; a host-side run writes the plain
    envelope to ROBOTICS_GRADER_VERDICT_PATH, else verdict.local.json in the stage folder."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write
    anchor, meta = read_anchor()
    extra = _finite_metrics(stats)
    extra.update(_finite_metrics({k: rewards[k] for k in CHECKS + CRITERIA}))
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "design_score"), extra=extra, full_at=meta.get("full_at", 1.0))
    recorded = {"rewards": verdict["rewards"], "metrics": _finite_metrics(verdict["metrics"])}
    text = diagnostic_text(rewards, details)
    if os.environ.get("ALE_VERIFY_PYTHON"):
        kit_write(recorded, diagnostic=text)
        return recorded
    kit_write(recorded, path=os.environ.get("ROBOTICS_GRADER_VERDICT_PATH") or os.path.join(runtime.STAGE_DIR,
                                                                                            "verdict.local.json"),
              diagnostic=text)
    return recorded


def main():
    t0 = time.time()
    freezer = runtime.verify_freeze.AgentFreezer.for_user(runtime.AGENT_USER)
    early = runtime._early_lockdown(freezer)
    stats = {k: v for k, v in early.items() if isinstance(v, float)}
    stats["lockdown_s"] = time.time() - t0
    try:
        if early["agent_processes_running"] > 0:
            raise RuntimeError("%d agent process(es) survived the verify-start kill and could not be stopped: run void"
                               % early["agent_processes_running"])
        from robotics_grader import fail
        from robotics_grader.stage import secure_exec_path
        stats["root_path_dirs_secured"] = float(len(secure_exec_path()))
        with open(os.path.join(runtime.STAGE_DIR, "grader_config.json")) as fh:
            cfg = json.load(fh)
        if cfg.get("document") != runtime.DOCUMENT:
            fail("grader_config.json names the document %r, verify.py %r" % (cfg.get("document"), runtime.DOCUMENT))
        rewards = {k: 0.0 for k in CHECKS + CRITERIA + ["design_score"]}
        details = {"submission": [tool_usage(stats)]}
        sub_dir = runtime.SNAPSHOT if early["submission_snapshot"] else runtime.SUBMISSION
        if early["submission_root_rejected"] > 0:
            sub_dir = tempfile.mkdtemp(prefix="arb_rejected_submission_")
        doc, note = runtime.find_document(sub_dir)
        stats["zero_no_document"] = 0.0
        stats["zero_unreadable"] = 0.0
        stats["zero_no_valid_solid"] = 0.0
        intact, why = archive_intact(doc) if doc is not None else (False, None)
        if doc is None:
            stats["zero_no_document"] = 1.0
            details["cad"] = ["zero: no document: " + (note or "no document")]
        elif not intact:
            stats["zero_unreadable"] = 1.0
            details["cad"] = ([note] if note else []) + ["zero: " + why]
        else:
            if note:
                details["cad"] = [note]
            out_dir, tail = runtime.extract(doc, int(cfg["extract_timeout_s"]), stats)
            res, ctail = grade_child(out_dir, stats, int(cfg["grade_timeout_s"]))
            if res is None:
                fail("the grade did not complete: " + str(ctail)[-1200:])
            if res.get("task_error") or not res.get("ok"):
                fail("the grader failed: %s | %s" % (res.get("task_error"), str(res.get("error", ""))[-1200:]))
            rewards["cad_valid"] = float(res["checks"].get("cad_valid", 0.0))
            for k in CRITERIA:
                rewards[k] = float(res["criteria"].get(k, 0.0))
            rewards["design_score"] = float(res["design_score"])
            stats.update(grade_metrics(res))
            for k, v in (res.get("timing") or {}).items():
                stats["grade_" + k] = v
            if res.get("zero_case") == "unreadable":
                stats["zero_unreadable"] = 1.0
            elif res.get("zero_case") == "no_valid_solid":
                stats["zero_no_valid_solid"] = 1.0
            msgs = [str(d)[:300] for d in (res.get("diagnostics") or [])[:12]]
            if res.get("zero_case") and stats.get("extract_exit", 0.0) != 0.0:
                msgs.append("FreeCAD reader: " + tail[-300:])
            details["grade"] = msgs
            with open(os.path.join(runtime.STAGE_DIR, "public_blocklist.json")) as fh:
                blocked = set(json.load(fh)["files"].values())
            reused = public_reuse(sub_dir, blocked)
            rewards["public_model_reuse"] = 0.0 if reused else 1.0
            if reused:
                details["public_model_reuse"] = ["zero: an unchanged published file of the maker in the submission: " +
                                                 ", ".join(reused)[:300]]
            rewards["design_score"] *= rewards["public_model_reuse"]
        final = 0.0 if (early["submission_late_writes"] > 0 or early["submission_root_rejected"] > 0) else 1.0
        stats["submission_final"] = final
        if final < 1.0:
            for k in rewards:
                rewards[k] = 0.0
            details["submission"] = ["submission_final: " + "; ".join(early["reasons"])] + list(details["submission"])
        stats["verify_s"] = time.time() - t0
        stats["agent_processes_frozen"] = float(len(freezer.frozen))
        for k in list(stats):
            if k in rewards:
                del stats[k]            # the checks, criteria and design_score are written from `rewards` only
        stats, timing = split_timing(stats)
        record_timing(timing)
        write_verdict(float(rewards["design_score"]), rewards, stats, details)
        runtime.log("design_score {:.4f} checks {} criteria {} ({:.0f} s)".format(
            rewards["design_score"], {g: rewards[g] for g in CHECKS}, {c: round(rewards[c], 4) for c in CRITERIA},
            timing["verify_s"]))
    finally:
        runtime._restore(early)
        freezer.stop()


if __name__ == "__main__":
    main()
