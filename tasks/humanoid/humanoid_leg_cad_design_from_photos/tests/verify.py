"""Grader of humanoid_leg_cad_design_from_photos (the leg as designed is the ground truth). Root, verify stage, agent gone.

forked from tasks/ground/racecar_cad_design_from_photos/tests/verify.py (benchmark main 850dee3, sha256
da34edc526dbcbb401358dd193936002d0e0b98630d0d9549ce543a3fc715d9c). Changes: the document is leg.FCStd (set on the vendored,
unchanged lockdown_runtime); six criteria (mechanism added); the neutral public_model_reuse diagnostic names the leg; the
summary line names the alignment mode and the three segment floors. As before: the statistics written to `metrics`
leave out the timings (keys ending in _s; those go to stderr only); the diagnostic is capped at 2000 characters in both
branches; the grade child runs on OpenBLAS's Haswell kernels with their own block sizes and on numpy's AVX2 loops
(OPENBLAS_CORETYPE, OPENBLAS_L2_SIZE and NPY_DISABLE_CPU_FEATURES, set below). FreeCAD's assembly solver is not run
anywhere in this stage.

Where it runs: a separate verifier sandbox (task.toml `environment_mode = "separate"`), started fresh from the task image.
The engine restores only the declared artifact /home/user/submission into it, so nothing the agent left anywhere else is
present. setup.sh does not run there and nothing it staged is read. The given parts the grader compares with are the
image's own payload files (/opt/task-assets/payload: given/<PN>/<PN>.step and the shell's BREP, fetched and checked at
image build), each pinned by sha256 in reference/purchased_pins.json: the verifier image is the agent's image
unchanged by the agent, and a payload that is not the one the grader was built for is a task error, not a zero. The
lockdown below works the same in a shared sandbox. In the verifier, the agent account's processes are the image's own
desktop.

  0. FIRST THING, stdlib only (lockdown_runtime._early_lockdown): record when the engine began uploading this folder,
     stop and kill every process of the agent account (a survivor that cannot be frozen voids the run: task error),
     freeze it, lock the stage, take a root-owned 0700 snapshot of the submission that every later read uses, and zero
     the score (stat submission_final = 0) when an entry of the submission was written at or after the stage upload
     began. A submission root that is a symbolic link, or that resolves outside the agent's home, is not read at all.
     The directories on root's PATH are reclaimed (robotics_grader.stage.secure_exec_path) before anything else runs.
  1. Tool use from the engine's solver trajectory (ATIF, ALE_TRAJECTORY_PATH), stats only.
  2. The submitted FreeCAD document (snapshot leg.FCStd, else the only *.FCStd of the submission root) is read by
     FreeCAD's headless interpreter as `nobody`, through runuser (absolute path) and /usr/bin/env HOME=..., in a private
     work folder, with a time limit (fcstd_extract.py): every shape leaf of the Assembly in world coordinates, solid by
     solid, with labels, types and the joints; no BRepCheck.
  3. grade.py, a child process with a time limit, grades the extraction against reference/ (every number in
     grader_config.json): the document-level conditions (cad_valid: readable, holds a solid, the shell or the two
     thigh actuators usable for the alignment), the local rules (a bad or duplicated tag gets no credit, bad untagged
     pieces are left out; each reported), alignment, given-part recognition, the six criteria and the segment floors.
     A grading item that fails
     scores 0 (grader_item_errors); a failure of the grader itself (reference, payload files, an exception outside every
     item, a crash or a time-out of the child) fails the run as a task error, never as a silent 0.
  4. public_model_reuse: a file of the submission whose sha256 is one of the robot model's public source files
     (public_blocklist.json) zeroes the design score.
design_score = grade's weighted sum x public_model_reuse; the verdict's single gated `reward` comes from the benchmark
kit (robotics_grader.ratio_rewards against anchor.json: value 1.0 = the oracle, full_at 1.0), so reward = design_score.
No agent code runs here (the document is data that FreeCAD reads).
"""
import hashlib
import json
import math
import os
import sys
import tempfile
import time

import lockdown_runtime as runtime

runtime.DOCUMENT = "leg.FCStd"   # find_document reads it at call time (the vendored runtime stays unchanged)
# The grade child inherits this environment (runtime.grade_child copies os.environ), and its numerics run on numpy's and
# scipy's own OpenBLAS copies. Left alone, OpenBLAS picks its kernels by the CPU it finds (AVX-512 kernels on one
# machine, AVX2 kernels on another), and their sums can differ in the last bits, enough to change how a solid is
# meshed. The kernels are fixed so that the loaded reference and the scores do not depend on which kernels the CPU of
# the grading machine would select (the Haswell kernels need AVX2 and FMA). The FreeCAD reader loads no BLAS.
os.environ["OPENBLAS_CORETYPE"] = "Haswell"
# The L2 size is set with the kernel. On some AMD CPUs with AVX-512 and 1024 KB of L2 cache per core, the OpenBLAS
# that numpy bundles enlarges its float64 matrix-product block for every kernel, the forced Haswell one included,
# beyond what the Haswell kernel's fixed stack buffer holds; a product with a long inner dimension then crashes or
# comes out wrong. The enlargement applies only when the L2 size reads 1024, so 256 (the L2 size the Haswell kernels
# are tuned for) keeps the blocks they were built with, on every CPU.
os.environ["OPENBLAS_L2_SIZE"] = "256"
# numpy's own SIMD loops are held at their AVX2 level (X86_V3) on every machine, as the BLAS kernels are above:
# numpy picks its loops by the CPU when it is imported, and its AVX-512 loops of arccos, arctan2, exp, log and power
# give other last bits than its AVX2 loops. This process imports no numpy; the grade child does, after this line.
os.environ["NPY_DISABLE_CPU_FEATURES"] = "X86_V4 AVX512_ICL AVX512_SPR"
CHECKS = ["cad_valid", "public_model_reuse"]
CRITERIA = ["shape", "features", "installation", "mechanism", "mass", "fit"]
DIAGNOSTIC_MAX = 2000


def public_reuse(root, blocked):
    """Files under the submission whose sha256 is a public export (symbolic links are not followed)."""
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


def _finite_metrics(metrics):
    """`metrics` in name order, finite numbers only (ale_verify.write refuses NaN and infinity)."""
    out = {}
    for k, v in sorted(metrics.items()):
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out[k] = x
    return out


def write_verdict(overall, rewards, stats, details):
    """One gated `reward` in the kit's envelope (ratio_rewards against anchor.json: value 1.0 = the oracle, full_at 1.0,
    so reward = design_score); checks, criteria and statistics go to `metrics`, the diagnostics into the verdict's
    `diagnostic` text. In the engine (ALE_VERIFY_PYTHON set) the kit writes /logs/verifier/reward.json and
    verification.json through ale_verify; a host-side run writes the plain envelope to ROBOTICS_GRADER_VERDICT_PATH,
    else verdict.local.json in the stage folder."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write
    anchor, meta = read_anchor()
    # timings (keys ending in _s) stay out of the metrics; main() writes them to stderr
    extra = {k: float(v) for k, v in stats.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool) and not k.endswith("_s")}
    extra.update({k: float(rewards[k]) for k in CHECKS + CRITERIA})
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "design_score"), extra=extra, full_at=meta.get("full_at", 1.0))
    text = "; ".join("%s %.4f%s" % (k, float(rewards[k]), (": " + " | ".join(str(d) for d in details.get(k, []))[:200])
                                    if details.get(k) else "") for k in CHECKS + CRITERIA)
    if details.get("grade"):
        text += "; " + " | ".join(str(d) for d in details["grade"])[:600]
    # the diagnostic cap of the earlier ale_verify API (2000 characters), kept in both branches
    text = text[:DIAGNOSTIC_MAX]
    if os.environ.get("ALE_VERIFY_PYTHON"):
        recorded = {"rewards": verdict["rewards"], "metrics": _finite_metrics(verdict["metrics"])}
        kit_write(recorded, diagnostic=text)
        return
    kit_write(dict(verdict, diagnostics=details),
              path=os.environ.get("ROBOTICS_GRADER_VERDICT_PATH") or os.path.join(runtime.STAGE_DIR, "verdict.local.json"),
              diagnostic=text)


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
        rewards = {k: 0.0 for k in CHECKS + CRITERIA + ["design_score"]}
        details = {"tool_usage": [runtime.tool_usage(stats)]}
        sub_dir = runtime.SNAPSHOT if early["submission_snapshot"] else runtime.SUBMISSION
        if early["submission_root_rejected"] > 0:
            sub_dir = tempfile.mkdtemp(prefix="arb_rejected_submission_")
        doc, note = runtime.find_document(sub_dir)
        if doc is None:
            details["cad_valid"] = [note or "no document"]
        else:
            if note:
                details["cad_valid"] = [note]
            out_dir, tail = runtime.extract(doc, int(cfg.get("extract_timeout_s", 600)), stats)
            purchased = os.environ.get("ARB_PURCHASED_ROOT") or cfg["purchased_root"]
            res, ctail = runtime.grade_child(out_dir, purchased, cfg, stats)
            # A failure of the grader itself is never a silent 0: the reference or the vendor files unusable, an
            # exception outside a grading item (grade.py turns those into task_error), a child that crashed, timed out
            # or wrote nothing. A grading item that fails on the document scores 0 inside grade.py (grader_item_errors).
            if res is None:
                fail("the grade did not complete: " + str(ctail)[-1200:])
            if res.get("task_error") or not res.get("ok"):
                fail("the grader failed: %s | %s" % (res.get("task_error"), str(res.get("error", ""))[-1200:]))
            rewards["cad_valid"] = float(res["checks"].get("cad_valid", 0.0))
            for k in CRITERIA:
                rewards[k] = float(res["criteria"].get(k, 0.0))
            rewards["design_score"] = float(res["design_score"])
            for k, v in res.get("statistics", {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    stats[k] = float(v)
            if res.get("diagnostics"):
                details["cad_valid"] = list(details.get("cad_valid", [])) + [str(d)[:300] for d in res["diagnostics"][:8]]
                if stats.get("extract_exit", 0.0) != 0.0:
                    details["cad_valid"].append("FreeCAD reader: " + tail[-300:])
            if rewards["cad_valid"] > 0:
                parts = res.get("parts") or {}
                missing = sorted(t for t, p in parts.items() if not p.get("present"))
                purchased_rec = res.get("purchased") or {}
                absent = sorted(p for p, r in purchased_rec.items() if not r.get("present"))
                details["grade"] = ["parts missing: %s" % (", ".join(missing) or "none"),
                                    "given instances not found: %s" % (", ".join(absent) or "none"),
                                    "alignment mode %d, residual %.4f mm; segment floors %s" % (
                                        int(stats.get("alignment_mode", 0.0)), stats.get("alignment_residual_mm", float("nan")),
                                        ", ".join("%s %.3f" % (k[6:], v) for k, v in sorted(stats.items())
                                                  if k.startswith("floor_") and k != "floor_leg"))]
            with open(os.path.join(runtime.STAGE_DIR, "public_blocklist.json")) as fh:
                blocked = set(json.load(fh)["files"].values())
            reused = public_reuse(sub_dir, blocked)
            rewards["public_model_reuse"] = 0.0 if reused else 1.0
            if reused:
                details["public_model_reuse"] = ["a public source file of the robot's model, unchanged, in the submission: " +
                                                 ", ".join(reused)[:300]]
            rewards["design_score"] *= rewards["public_model_reuse"]
        final = 0.0 if (early["submission_late_writes"] > 0 or early["submission_root_rejected"] > 0) else 1.0
        stats["submission_final"] = final
        if final < 1.0:
            for k in rewards:
                rewards[k] = 0.0
            details["cad_valid"] = ["submission_final: " + "; ".join(early["reasons"])] + list(details.get("cad_valid", []))
        stats["verify_s"] = time.time() - t0
        stats["agent_processes_frozen"] = float(len(freezer.frozen))
        write_verdict(float(rewards["design_score"]), rewards, stats, details)
        runtime.log("timings " + json.dumps({k: round(float(v), 1) for k, v in sorted(stats.items())
                                             if k.endswith("_s") and isinstance(v, (int, float))}))
        runtime.log("checks {} criteria {} design_score {:.4f} ({:.0f} s)".format(
            {g: rewards[g] for g in CHECKS}, {c: round(rewards[c], 4) for c in CRITERIA}, rewards["design_score"],
            stats["verify_s"]))
    finally:
        runtime._restore(early)
        freezer.stop()


if __name__ == "__main__":
    main()
