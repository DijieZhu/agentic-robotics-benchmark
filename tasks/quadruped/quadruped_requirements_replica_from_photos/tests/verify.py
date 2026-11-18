"""Grader of quadruped_requirements_replica_from_photos (information tier, requirements stage). Root, verify stage, the
agent gone. No agent code runs here: the submission is three static files (YAML / JSON).

Separate verification (task.toml: [verifier] environment_mode = "separate"). The engine releases the agent's sandbox,
starts a fresh verifier sandbox from the task image and restores into it only the declared artifact
/home/user/submission (regular files and directories, owned by root; links and special files are never copied), then
uploads this folder. Everything read here is in this folder (gt/, grader_config.json, anchor.json, the grading modules
quad/, stage_lock/, the kit), in the image (/opt/venv) or in that restored submission. environment/setup/setup.sh does
not run in the verifier, so the givens it stages into the agent home are not there; the grader reads its own truth
(gt/) and none of the agent's files besides the submission. given_hashes.json stays here as the maintainers' pin of the
givens; this grader does not read it.

  0. FIRST THING, stdlib only (`_early_lockdown`; test.sh has already chmod 0700 the stage and its parent): record when
     the engine began uploading this folder (the earliest ctime of its files), freeze and kill every process of the
     agent account (a survivor that cannot be frozen voids the run: task error), lock the stage (0700 / 0600, restored
     at the end), gate G1 (submission integrity): an entry of the submission written at or after the upload began, or a
     submission ROOT that is a symbolic link, not a directory, or a directory replaced or changed after the agent phase
     (lstat, never followed), zeroes every criterion, and nothing behind such a root is read. Then a root-owned 0700
     snapshot that every later read uses: the three deliverables only (the only files the grader reads), each a regular
     file opened without following a link and read up to MAX_FILE_BYTES + 1 bytes (a sparse or oversize file costs at
     most that, and an oversize deliverable still reads as oversize); notes/ and everything else stay behind. In the
     separate verifier nothing the agent started is running: the kill stops only the image's own desktop session, which
     the base image's supervisord runs as the agent account (Xvnc, noVNC, PulseAudio, GNOME: about 50 processes) and
     which this grader does not use. supervisord respawns it as fast as it is killed, so a process that outlives the
     kill is frozen (SIGSTOP: a stopped process cannot exit, so it is not respawned; stat agent_processes_running counts
     what is still running after up to 3 s of freezing), and only one still running then voids the run. The engine
     restores the submission before it uploads this folder, so no restored entry is late. The lockdown stays as defence
     in depth. A submission root that is not a directory is restored as such and still zeroes G1; a symbolic-link root
     is not restored at all (the engine copies no link), so G0 then finds the three files missing.
  1. G0 structure (grade.load_submission + grade.g0_rules with the truth layer and catalogue): the files parse and
     validate, ids unique, units convert, the six BOM lines on the quote with the quote's roles, the battery family,
     the driver count, the sketch. A failure zeroes every criterion.
  2. The truth of the hidden instance (gt/, CONTRACTS i.1) and the self-tests T1-T12 (CONTRACTS j), on every run, with
     the tolerances of grader_config.json's selftest block; any failure is a TASK ERROR (never a score):
     T1 the truth model validates and its dynamics MJCF is gt/quad_dynamics.xml byte for byte; T2 the compiled CoM of
     every added body at q = 0, the B-05 stance and the proof poses, and the compiled inertias; T3 the analytic rates
     against central differences; T4 the regressor against direct inverse passes; T5 the base-wrench residual; T6 the
     centroid form of the contact rule; T7 the box-wide minimum normal force; T8 the duty against gt/duty_set.json;
     T9 the live optimum J* against gt/optimum.json; T10 the derived items of J*'s BOM against
     gt/derived_for_optimal.json; T11 the knife edge (the search's, and every utilisation of J*'s evaluation but the
     hub's and the budget's); T12 the config's consistency (C5 band = the brief's ECO, the weights sum to 1, the repair
     order is a permutation of the BOM lines).
  3. The criteria (grade.grade with the whole config): kinematic_replication 0.30, requirements_sheet 0.10,
     derived_requirements 0.20, bom_acceptance 0.10, bom_economy 0.30. requirements_score = their weighted mean; the
     verdict's single gated `reward` comes from the benchmark kit (robotics_grader.ratio_rewards against anchor.json:
     value 1.0 = the oracle, full_at 1.0), so reward = requirements_score. The five criteria, the gates, the
     self-tests, the timings and every stat (finite: non-finite values are -1) ride along in `metrics`; the
     per-criterion diagnostics are cut to 380 characters each and the joined text to 2000.
Knobs: grader_config.json.
"""
import json
import math
import os
import pwd
import signal
import stat as statmod
import sys
import time

sys.dont_write_bytecode = True
STAGE_DIR = os.environ.get("ALE_STAGE_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STAGE_DIR)
HOME = os.environ.get("ALE_HOME", "/home/user")
AGENT_USER = os.path.basename(HOME.rstrip("/")) or "user"
SUBMISSION = os.path.join(HOME, "submission")
SNAP_ROOT = os.environ.get("ARB_SNAPSHOT_ROOT", "/opt/arb-verify")
SNAPSHOT = os.path.join(SNAP_ROOT, "submission")
LATE_GATE_MAX_GAP_S = 600.0
REGRESSOR_SEED = 20261028           # CONTRACTS j T4: the regressor self-test's mass draws
DIAG_EMPTY = ""

from stage_lock import lockdown  # noqa: E402  (stdlib only)
from stage_lock import runner  # noqa: E402  (stdlib only)

CRITERIA = ("kinematic_replication", "requirements_sheet", "derived_requirements", "bom_acceptance", "bom_economy")


def log(msg):
    print("[grader] " + msg, file=sys.stderr, flush=True)


# The verify-start snapshot (R2-D2): the deliverables grade.load_submission reads (grade.SUBMISSION_FILES), each read
# up to grade.MAX_FILE_BYTES + 1 bytes. Stdlib only, so the lockdown imports nothing of the grader; main() checks both
# against grade.py before any grading (a mismatch is a task error).
DELIVERABLES = ("requirements.yaml", "bom.json", "kinematic_sketch.json")
SNAPSHOT_MAX_BYTES = 2 * 10 ** 6 + 1


def _snapshot(src, dst):
    """Root-owned 0700 / 0600 copy of the three DELIVERABLES of `src`: each opened with O_NOFOLLOW | O_NONBLOCK (a
    symbolic link or a FIFO never blocks or redirects the read), kept only if it is a regular file, and read up to
    SNAPSHOT_MAX_BYTES (a sparse file is never written out at its apparent size; an oversize deliverable still reads
    as oversize). Everything else (notes/, links, special files) stays behind; the late-write gate (when active) has
    walked the whole submission before. Returns the number of deliverables copied."""
    import shutil
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    kept = 0
    for name in DELIVERABLES:
        try:
            fd = os.open(os.path.join(src, name), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            continue
        try:
            if not statmod.S_ISREG(os.fstat(fd).st_mode):
                continue
            with open(os.path.join(dst, name), "wb") as out:
                left = SNAPSHOT_MAX_BYTES
                while left > 0:
                    blk = os.read(fd, min(left, 1 << 20))
                    if not blk:
                        break
                    out.write(blk)
                    left -= len(blk)
            kept += 1
        finally:
            os.close(fd)
    runner.chmod_tree(dst, 0o700, 0o600)
    return kept


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
    """SIGSTOP every live process of `uid` until a pass finds none still running (a respawn that appears meanwhile is
    stopped by the next pass) or `bound_s` elapsed; returns the PIDs still running (empty: every one is stopped)."""
    deadline = time.time() + float(bound_s)
    while True:
        for pid in lockdown.live_uid_processes(uid):
            try:
                os.kill(pid, signal.SIGSTOP)
            except OSError:
                pass
        time.sleep(0.05)
        running = [pid for pid in lockdown.live_uid_processes(uid) if _proc_state(pid) not in ("T", "t", "X")]
        if not running or time.time() > deadline:
            return running


def _early_lockdown():
    out = {"agent_processes_killed": 0.0, "agent_processes_left": 0.0, "agent_processes_running": 0.0,
           "stage_locked": 0.0, "stage_parent_locked": 0.0, "submission_late_writes": 0.0,
           "submission_root_violation": 0.0, "late_write_gate_active": 0.0, "submission_snapshot": 0.0,
           "saved": None, "parent": None, "reasons": [], "t_ref": None}
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
            # The image's supervisord respawns the account's desktop as fast as it is killed, so in the separate
            # verifier a process that outlives the kill is the image's own (2026-11-07: one respawned desktop process
            # outlived this kill and voided a run of another task). It is frozen (a stopped process cannot exit, so it
            # is not respawned); only one still running once frozen voids the run.
            out["agent_processes_running"] = float(len(_still_running(uid)))
            log("%d process(es) of the agent account outlived the kill at verify start, %d still running after the "
                "freeze" % (left, out["agent_processes_running"]))
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
    try:
        rst = os.lstat(SUBMISSION)
    except OSError:
        rst = None
    root_problem = None
    if rst is not None:
        if statmod.S_ISLNK(rst.st_mode):
            root_problem = "the submission root is a symbolic link"
        elif not statmod.S_ISDIR(rst.st_mode):
            root_problem = "the submission root is not a directory"
        elif gate_active and max(rst.st_ctime, rst.st_mtime) >= t_ref:
            root_problem = "the submission directory itself was replaced or changed after the agent phase ended"
    if root_problem:
        out["submission_root_violation"] = 1.0
        out["submission_late_writes"] += 1.0
        out["reasons"].append(root_problem)
    if rst is not None and statmod.S_ISDIR(rst.st_mode) and not root_problem:
        if gate_active:
            late = lockdown.late_entries(SUBMISSION, t_ref)
            out["submission_late_writes"] += float(len(late))
            if late:
                out["reasons"].append("%d entr%s of the submission written after the agent phase ended (%s)" % (
                    len(late), "y" if len(late) == 1 else "ies", ", ".join(e[0] for e in late[:5])))
        os.makedirs(SNAP_ROOT, exist_ok=True)
        os.chmod(SNAP_ROOT, 0o700)
        _snapshot(SUBMISSION, SNAPSHOT)
        out["submission_snapshot"] = 1.0
    return out


def _restore(early):
    try:
        if early.get("saved") is not None:
            runner.unlock_stage(early["saved"])
        if early.get("parent"):
            os.chmod(early["parent"][0], early["parent"][1])
        mode = os.environ.get("ARB_STAGE_MODE")
        if mode:
            os.chmod(STAGE_DIR, int(mode, 8))
    except OSError:
        pass


def _fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return -1.0
    return v if math.isfinite(v) else -1.0


def _cut(text, n):
    text = str(text)
    return text if len(text) <= n else (text[:n - 3] + "..." if n >= 3 else text[:max(0, n)])


def write_verdict(rewards, weights, stats, diagnostics, per, total):
    """One gated `reward` (the kit's envelope); the criteria and stats go to `metrics`, the per-criterion diagnostics
    into the verdict's `diagnostic` text. In the engine (ALE_VERIFY_PYTHON set) the kit writes
    /logs/verifier/reward.json and verification.json through ale_verify, every metric through _fin as on the earlier
    engine API; a host-side run writes the plain envelope to ROBOTICS_GRADER_VERDICT_PATH, else verdict.local.json in
    the stage folder (and the text to
    ARB_DIAGNOSTICS_PATH when that is set)."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write
    overall = sum(float(rewards[c]) * float(weights[c]) for c in CRITERIA) / sum(float(weights[c]) for c in CRITERIA)
    anchor, meta = read_anchor()
    extra = {k: _fin(v) for k, v in stats.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    extra.update({c: _fin(rewards[c]) for c in CRITERIA})
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "requirements_score"), extra=extra,
                            full_at=meta.get("full_at", 1.0))
    text = "; ".join("%s %.4f: %s" % (c, float(rewards[c]), _cut(diagnostics.get(c, DIAG_EMPTY), per))
                     for c in CRITERIA)
    # R2-D1: agent strings reach the text (a G0 message quotes a sketch value with %s, the G1 reason lists late file
    # names decoded with surrogateescape); a lone surrogate would make every UTF-8 writer raise, so escape it
    text = _cut(text.encode("utf-8", "backslashreplace").decode("utf-8"), total)
    if os.environ.get("ALE_VERIFY_PYTHON"):
        recorded = {"rewards": verdict["rewards"],
                    "metrics": {k: _fin(v) for k, v in sorted(verdict["metrics"].items())}}
        kit_write(recorded, diagnostic=text)    # text: cut to total_chars (2000) above, as the earlier API required
        return overall
    kit_write(verdict, path=os.environ.get("ROBOTICS_GRADER_VERDICT_PATH") or os.path.join(STAGE_DIR, "verdict.local.json"),
              diagnostic=text)
    if os.environ.get("ARB_DIAGNOSTICS_PATH"):                 # outside the engine: the verdict's diagnostic text
        with open(os.environ["ARB_DIAGNOSTICS_PATH"], "w", encoding="utf-8") as fh:
            fh.write(text)
    return overall


# ------------------------------------------------------------------------------------------------ the self-tests
class SelfTestError(RuntimeError):
    pass


def _require(ok, test, msg):
    if not ok:
        raise SelfTestError("self-test %s failed: %s (a task error)" % (test, msg))


def selftests_static(gt, cfg, stats):
    """T1 (the truth model and its dynamics MJCF) and T12 (the config's consistency)."""
    from quad.req_info.q1 import quadmodel as Q
    from quad.req_info.q1 import schemas as S
    probs = Q.validate(gt["quad_model"])
    _require(not probs, "T1", "the truth model is invalid: %s" % probs[:3])
    xml = Q.build_xml(gt["quad_model"], float(gt["layer"]["brief"]["gravity_m_s2"]))
    _require(xml == gt["dynamics_xml"], "T1", "build_xml(truth) differs from gt/quad_dynamics.xml")
    stats["selftest_T1"] = 1.0
    g = cfg["grader"]
    eco = float(gt["layer"]["brief"]["eco_usd"])
    _require(float(g["c5"]["band_usd"]) == eco, "T12", "c5.band_usd %r != the brief's eco_usd %r" % (
        g["c5"]["band_usd"], eco))
    w = cfg["weights"]
    _require(sorted(w) == sorted(CRITERIA), "T12", "the weights are not the five criteria")
    _require(abs(sum(float(w[c]) for c in CRITERIA) - 1.0) <= 1e-12, "T12", "the weights do not sum to 1")
    _require(sorted(g["repair"]["order"]) == sorted(S.LINES) and len(g["repair"]["order"]) == len(S.LINES), "T12",
             "repair.order is not a permutation of %s" % (S.LINES,))
    stats["selftest_T12"] = 1.0


def selftests_procedure(proc, gt, cat, cfg, stats):
    """T2-T8 on the truth Procedure (T4-T7 at J*'s family box and masses, from gt/optimum.json)."""
    import numpy as np
    from quad.req_info.q1 import optimum as O
    from quad.req_info.q1.procedure import bom_from_json
    st = cfg["selftest"]
    x0 = float(proc.dyn.xipos_error_m)
    x1 = float(proc.selftest_xipos())
    stats["selftest_T2"] = max(x0, x1)
    stats["selftest_T2_inertia"] = max(float(proc.dyn.inertia_error_rel), float(proc.selftest_inertia()))
    _require(stats["selftest_T2"] <= float(st["xipos_m"]), "T2", "xipos error %g m" % stats["selftest_T2"])
    _require(stats["selftest_T2_inertia"] <= float(st["inertia_rel"]), "T2",
             "inertia error %g" % stats["selftest_T2_inertia"])
    stats["selftest_T3"] = float(proc.selftest_derivatives(float(st["deriv_h_s"])))
    _require(stats["selftest_T3"] <= float(st["deriv_rel"]), "T3", "derivative error %g" % stats["selftest_T3"])
    bom = bom_from_json(gt["optimum"]["bom"])
    box = O.mass_box(proc, cat, gt["optimum"]["family"])
    stats["selftest_T4"] = float(proc.selftest_regressor(np.random.default_rng(REGRESSOR_SEED),
                                                         int(st["regressor_draws"]), box))
    _require(stats["selftest_T4"] <= float(st["regressor_rel"]), "T4", "regressor error %g" % stats["selftest_T4"])
    con = proc.selftest_contact(proc.masses_of(bom, cat))
    stats["selftest_T5"] = float(con["base_residual"])
    stats["selftest_T6"] = float(con["centroid_form"])
    _require(stats["selftest_T5"] <= float(st["base_residual"]), "T5", "base residual %g" % stats["selftest_T5"])
    _require(stats["selftest_T6"] <= float(st["contact_n"]), "T6", "centroid form %g N" % stats["selftest_T6"])
    stats["selftest_T7"] = float(proc.normal_force_lower_bound(box))
    _require(stats["selftest_T7"] >= float(st["normal_min_n"]), "T7", "box-wide normal force %g N" %
             stats["selftest_T7"])
    ds = gt["duty_set"]
    _require(int(ds["every"]) == int(st["duty_every"]), "T8", "gt/duty_set.json probes every %s, the config says %s"
             % (ds["every"], st["duty_every"]))
    fp = proc.duty_fingerprint(int(ds["every"]))
    _require(fp["counts"] == ds["counts"], "T8", "sample counts %s != %s" % (fp["counts"], ds["counts"]))
    _require(fp["stance_sha256"] == ds["stance_sha256"] and fp["swing_sha256"] == ds["swing_sha256"], "T8",
             "stance / swing digests differ")
    _require(len(fp["probes"]) == len(ds["probes"]), "T8", "probe counts differ")
    worst = 0.0
    for a, b in zip(fp["probes"], ds["probes"]):
        _require(a["kind"] == b["kind"] and int(a["index"]) == int(b["index"]), "T8", "probe positions differ")
        for key in ("trunk_p", "q", "qd", "qdd"):
            worst = max(worst, max(abs(float(x) - float(y)) for x, y in zip(a[key], b[key])))
    stats["selftest_T8"] = worst
    _require(worst <= float(st["duty_abs"]), "T8", "duty probes differ by %g" % worst)


def selftests_optimum(best, proc, gt, cat, cfg, stats):
    """T9 (live J* = gt/optimum.json), T10 (derived items of J*'s BOM = gt/derived_for_optimal.json), T11 (the knife
    edge: the search's and every utilisation of J*'s evaluation except the hub's and the budget's)."""
    from quad.req_info.q1 import schemas as S
    from quad.req_info.q1.procedure import bom_from_json, bom_to_json
    st = cfg["selftest"]
    stored = gt["optimum"]
    _require(best is not None, "T9", "the live optimum search found no BOM")
    dcost = abs(float(best["cost"]) - float(stored["cost"]))
    stats["selftest_T9"] = dcost
    same = best["bom"] == bom_from_json(stored["bom"])
    _require(dcost <= float(st["optimum_usd"]) and same, "T9", "live J* %s / %r differs from gt/optimum.json %s / %r"
             % (best["cost"], best["bom"], stored["cost"], stored["bom"]))
    _require(bom_to_json(best["bom"], cat) == gt["optimal_bom"], "T10", "gt/optimal_bom.json is not J*'s BOM")
    _require(gt["derived_for_optimal"]["bom"] == gt["optimal_bom"], "T10", "gt/derived_for_optimal.json is of "
                                                                          "another BOM")
    der = proc.derived(best["bom"], cat, gt["lots"])
    worst = 0.0
    for q in S.DERIVED_QUANTITIES:
        g = float(gt["derived_for_optimal"]["values"][q])
        worst = max(worst, abs(float(der[q]) - g) / max(abs(g), 1e-9))
    stats["selftest_T10"] = worst
    _require(worst <= float(st["derived_rel"]), "T10", "derived items differ by %g (relative)" % worst)
    ev = best["evaluation"]["util"]
    us = []
    for j in S.JOINTS:
        us += [ev[j][k] for k in ("move", "proof", "rms", "worst")]
    us += [ev["battery"][k] for k in ("charge", "current", "worst")]
    us += [ev["drivers"][k] for k in ("peak", "rms", "worst")]
    edge = min([float(best["knife_edge_min"])] + [abs(float(u) - 1.0) for u in us])
    stats["selftest_T11"] = edge
    _require(edge >= float(st["knife_edge"]), "T11", "a utilisation lies %g from the pass threshold" % edge)


# ------------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    early = _early_lockdown()
    # every float of the lockdown is a stat, except t_ref (the upload epoch, a clock reading, not a measurement)
    stats = {k: v for k, v in early.items() if isinstance(v, float) and k != "t_ref"}
    stats["lockdown_s"] = time.time() - t0
    try:
        if early["agent_processes_running"] > 0:
            raise RuntimeError("%d agent process(es) survived the verify-start kill and could not be stopped: run void"
                               % early["agent_processes_running"])
        with open(os.path.join(STAGE_DIR, "grader_config.json")) as fh:
            cfg = json.load(fh)
        weights = cfg["weights"]
        per = int(cfg["grader"]["diagnostics"]["per_criterion_chars"])
        total = int(cfg["grader"]["diagnostics"]["total_chars"])
        sub = SNAPSHOT if early["submission_snapshot"] else SUBMISSION
        from quad.req_info.q1 import grade as GR
        from quad.req_info.q1 import optimum as O
        from quad.req_info.q1.catalog import Catalogue
        from quad.req_info.q1.procedure import Procedure
        if sorted(DELIVERABLES) != sorted(GR.SUBMISSION_FILES.values()) or \
                SNAPSHOT_MAX_BYTES != GR.MAX_FILE_BYTES + 1:
            raise RuntimeError("the verify-start snapshot (DELIVERABLES, SNAPSHOT_MAX_BYTES) disagrees with grade.py's "
                               "SUBMISSION_FILES / MAX_FILE_BYTES + 1")
        gt_dir = os.path.join(STAGE_DIR, "gt")
        cat = Catalogue(os.path.join(gt_dir, "catalogue.json"))
        roles = cat.quote_roles()
        with open(os.path.join(gt_dir, "layer.json")) as fh:
            layer = json.load(fh)
        rewards = {c: 0.0 for c in CRITERIA}
        diagnostics = {}
        # G1, then G0 (nothing behind a violated submission root is read)
        g1 = early["submission_late_writes"] == 0
        stats["gate_g1_submission_integrity"] = 1.0 if g1 else 0.0
        docs, problems = None, []
        if g1:
            docs, problems = GR.load_submission(sub, roles)
            if not problems:
                problems = GR.g0_rules(docs, roles, layer, cat)
            stats["gate_g0_structure"] = 0.0 if problems else 1.0
            stats["g0_problems"] = float(len(problems))
            if problems:
                diagnostics = {c: "G0 structure: " + "; ".join(problems[:6]) for c in CRITERIA}
        else:
            stats["gate_g0_structure"] = -1.0
            diagnostics = {c: "G1: " + "; ".join(early["reasons"]) for c in CRITERIA}
        # the truth and the self-tests (every run)
        t1 = time.time()
        gt = GR.load_gt(gt_dir)
        selftests_static(gt, cfg, stats)
        t2 = time.time()
        proc = Procedure(gt["quad_model"], gt["layer"])
        stats["procedure_build_s"] = time.time() - t2
        t3 = time.time()
        selftests_procedure(proc, gt, cat, cfg, stats)
        stats["selftest_s"] = time.time() - t3
        best = None
        if g1 and not problems:
            t4 = time.time()
            res = GR.grade(docs, gt, cat, cfg, proc)
            stats["grade_s"] = time.time() - t4
            best = res["optimum"]
            rewards = {c: float(res["criteria"][c]) for c in CRITERIA}
            stats.update({k: v for k, v in res["stats"].items() if isinstance(v, (int, float))})
            diagnostics.update(res["diagnostics"])
        if best is None:
            t5 = time.time()
            best = O.optimal_bom(proc, cat, gt["lots"], int(cfg["grader"]["optimum"]["node_cap"]))
            stats["optimum_s"] = time.time() - t5
        selftests_optimum(best, proc, gt, cat, cfg, stats)
        stats["gt_s"] = time.time() - t1
        stats["verify_wall_s"] = time.time() - t0
        return write_verdict(rewards, weights, stats, diagnostics, per, total)
    finally:
        _restore(early)


if __name__ == "__main__":
    main()
