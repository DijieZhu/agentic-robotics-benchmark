"""Unit tests of the grader on synthetic flown sessions (synthlogs.py). Standard library and numpy only.

    python -I -B test_grader.py [--stage DIR] [--json OUT] [--bands-record FILE]

DIR is the folder holding grader/ and grader_config.json (default: this file's parent's parent, the task's tests).
The configuration is the generated one; its score.bands is fixed (the frozen bands, not pending). The fixture rule:
the synthetic tests keep their synthetic bands, synthlogs.bands_fixture, which replace the shipped bands in memory
only (the document is marked as a fixture and never written), so that the grader's arithmetic is tested on known
numbers. The shipped bands themselves are tested by test_shipped_bands and test_view_centre_zero_end: present with the
33 cells, every end above 0, the optical centre's zero end at the cap of score.completed, not pending, and equal as
doubles to the frozen bands of a record: FILE (named on the command line: a record holding score_bands_value, or a
table holding values["score.bands"]["value"]) and the copy of the table a placed stage holds at the layout's
world_table, each when present; the output's field bands_compared_with says which were read. Every test prints its
checks; the run exits 1 when any check fails. One test per scoring rule (the budget, an invalid plan, a hand-back,
a block's end, the session's end, the hold's condition, a request and the set-up, the optical centre's completion
limit), and the aggregation, the scored set, a kind without jobs, a dropped measure,
a set-up that overran, a controller that never comes up, a non-finite truth, missing or malformed records and hidden
set, the configuration's keys, the shipped bands, the static import audit, the three-file form, the verdict writer and
the grader's output.
"""
import copy
import importlib
import json
import math
import os
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = {"stage": os.path.dirname(HERE), "json": None, "only": None, "bands-record": None}
for i, a in enumerate(sys.argv):
    if a in ("--stage", "--json", "--only", "--bands-record") and i + 1 < len(sys.argv):
        ARGS[a[2:]] = sys.argv[i + 1]
sys.path.insert(0, ARGS["stage"])
sys.path.insert(0, HERE)
CFG = importlib.import_module("grader.config")
ME = importlib.import_module("grader.measures")
SC = importlib.import_module("grader.score")
RC = importlib.import_module("grader.records")
GR = importlib.import_module("grader.grade")
AU = importlib.import_module("grader.audit")
import synthlogs as SL  # noqa: E402

RESULTS = []
CUR = {"name": None}
COMPARED = []          # the records of frozen bands the shipped bands were compared with (test_shipped_bands)


def check(cond, msg):
    RESULTS.append({"test": CUR["name"], "ok": bool(cond), "msg": msg})
    print(("  ok    " if cond else "  FAIL  ") + msg)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ------------------------------------------------------------------------------------------------ fixtures
def config_doc():
    with open(os.path.join(ARGS["stage"], "grader_config.json"), "rb") as fh:
        doc = json.loads(fh.read().decode("utf-8"))
    return doc


def fixture_cfg(bands=None):
    """the fixture rule: the shipped (generated) configuration with its fixed score.bands replaced in memory by the
    synthetic bands, marked as a fixture, never written; the shipped bands are tested by test_shipped_bands."""
    doc = config_doc()
    assert doc["generated_from"]["fixture"] is False, "the fixture rule: the document replaced is the generated one"
    doc["values"]["score.bands"] = bands or SL.bands_fixture()
    doc["generated_from"]["fixture"] = True
    return CFG.from_document(doc)


CFGF = None


def cfg():
    global CFGF
    if CFGF is None:
        CFGF = fixture_cfg()
    return CFGF


def new_session(**kw):
    return SL.Session(cfg().values, **kw)


def rows_of(rec):
    chk = RC.check(rec, cfg().step_s())
    return ME.session_jobs(chk, cfg())


def job_scores(rec):
    rows = rows_of(rec)
    return SC.job_scores(rows, SC.cell_table(cfg().bands()), cfg().overshoot_terms()), rows


def scored_all(rec):
    return [j["id"] for b in rec["blocks"] for j in b["jobs"] if j["kind"] != "request"]


def grade(rec, scored=None):
    return GR.grade_session(rec, cfg(), scored if scored is not None else scored_all(rec))


def standard_session(**kw):
    """block 1 nominal: leg, hold, slew, mode request, leg; block 2 nominal: two views and a zone request; block 3
    quiet: leg, hold, slew; every feasible job perfect, every request refused."""
    s = new_session(**kw)
    b = s.block("block_1", "nominal")
    b.leg("j01").hold("j02").slew("j03").request("j04", "mode_not_available").leg("j05", d=(0.0, 0.3, 0.0))
    b = s.block("block_2", "nominal")
    b.view("j06").view("j07", axis=(1.0, 0.0, 0.0)).request("j08", "outside_zones")
    b = s.block("block_3", "quiet")
    b.leg("j09", d=(0.0, 0.0, 0.2)).hold("j10").slew("j11", turn=(0.0, 0.25, 0.0))
    return s


# ------------------------------------------------------------------------------------------------ the formula
def test_L_and_cells():
    check(SC.L(0.01, 0.02, 0.07) == 1.0, "L is 1 below the full end")
    check(SC.L(0.07, 0.02, 0.07) == 0.0 and SC.L(0.09, 0.02, 0.07) == 0.0, "L is 0 at and above the zero end")
    check(close(SC.L(0.045, 0.02, 0.07), 0.5), "L is linear between the ends (score.L)")
    c = SC.cell_table(cfg().bands())
    check(c[("leg", "quiet", "overshoot")][2] and close(c[("leg", "quiet", "overshoot")][3], 0.06),
          "a cell with Z < 2 FE is dropped and its completion limit is 2 FE (score.zero_end_rule, score.completed)")
    check(not c[("leg", "nominal", "end_pos")][2] and close(c[("leg", "nominal", "end_pos")][3], 0.075),
          "a cell that separates keeps its zero end as completion limit")
    check(c[("view", "nominal", "centre")][2] and close(c[("view", "nominal", "centre")][3], 0.033),
          "the view's centre cell is dropped and keeps its zero end 0.033 m as completion limit (score.completed)")
    check(len(c) == 33, "33 cells (kind, mode, measure)")


def test_perfect_session():
    rec = standard_session().record()
    out = grade(rec)
    check(close(out["score"], 1.0), "a session flown perfectly with every request refused scores 1.0 (%.6f)" % out["score"])
    check(sorted(out["terms"]) == ["holds", "legs", "refusals", "slews", "views"], "every kind and the refusals present")


def test_leg_formula():
    s = new_session()
    s.block("block_1", "nominal").leg("j01", end_pos=0.0475, overshoot=0.0425).leg("j02", end_att=0.042)
    js, rows = job_scores(s.record())
    m = {r["id"]: r["measures"] for r in rows}
    check(close(m["j01"]["end_pos"], 0.0475, 1e-9) and close(m["j01"]["overshoot"], 0.0425, 1e-9),
          "the leg's measures read on the truth (end_pos %.6f, overshoot %.6f)" % (m["j01"]["end_pos"], m["j01"]["overshoot"]))
    check(close(js["j01"][1], 0.5 * 1.0 * (0.8 + 0.2 * 0.5), 1e-6),
          "leg = completed x L(end_pos) x L(end_att) x (0.8 + 0.2 O): %.6f against 0.45" % js["j01"][1])
    check(close(js["j02"][1], (0.08 - 0.042) / (0.08 - 0.004), 1e-6), "the end attitude's L (%.6f)" % js["j02"][1])


def test_slew_formula():
    s = new_session()
    s.block("block_1", "nominal").slew("j01", end_att=0.0415, overshoot=0.0415).slew("j02", end_pos=0.0475)
    js, rows = job_scores(s.record())
    m = {r["id"]: r["measures"] for r in rows}
    check(close(m["j01"]["overshoot"], 0.0415, 1e-9) and close(m["j01"]["end_att"], 0.0415, 1e-9),
          "the slew's measures on the truth (end_att %.6f, overshoot %.6f)" % (m["j01"]["end_att"], m["j01"]["overshoot"]))
    want = ((0.08 - 0.0415) / 0.077) * 1.0 * (0.8 + 0.2 * (0.08 - 0.0415) / 0.077)
    check(close(js["j01"][1], want, 1e-6), "slew = completed x L(end_att) x L(end_pos) x (0.8 + 0.2 O): %.6f" % js["j01"][1])
    check(close(js["j02"][1], 0.5, 1e-6), "the slew's end position term (%.6f)" % js["j02"][1])


def test_hold_formula():
    s = new_session()
    s.block("block_1", "nominal").leg("j01").hold("j02", pos_rms=0.0525, speed_rms=0.012, rate_rms=0.00125, att_rms=0.0415)
    js, rows = job_scores(s.record())
    m = {r["id"]: r["measures"] for r in rows}["j02"]
    n = int(round(cfg().hold_s() / cfg().step_s())) + 1      # the window holds the sample at its start (the job before's)
    f = math.sqrt((n - 1.0) / n)
    check(close(m["pos_rms"], 0.0525 * f) and close(m["speed_rms"], 0.012 * f) and close(m["rate_rms"], 0.00125 * f) and
          close(m["att_rms"], 0.0415 * f, 1e-9), "the hold's measures about its target (the goal of the job before), "
          "over the window's %d samples" % n)
    want = SC.L(0.0525 * f, 0.03, 0.075) * SC.L(0.0415 * f, 0.003, 0.08) * 1.0 * SC.L(0.00125 * f, 0.0005, 0.002)
    check(close(js["j02"][1], want, 1e-6), "hold = completed(job before) x L(pos) x L(att) x L(speed) x L(rate): %.6f "
          "(the dropped speed term is 1)" % js["j02"][1])


def test_view_formula():
    s = new_session()
    s.block("block_2", "nominal").view("j01", pointing=0.0867).view("j02", rate=0.003).view("j03", centre=0.02)
    js, rows = job_scores(s.record())
    m = {r["id"]: r["measures"] for r in rows}
    check(close(m["j01"]["pointing"], 0.0867, 1e-9) and close(m["j02"]["rate_rms"], 0.003) and
          close(m["j03"]["centre"], 0.02, 1e-9), "the view's measures over the capture")
    check(close(js["j01"][1], (0.17 - 0.0867) / (0.17 - 0.0035), 1e-6), "view = captured x L(pointing) x ...: %.6f"
          % js["j01"][1])
    check(close(js["j02"][1], 1.0) and close(js["j03"][1], 1.0), "the dropped rate and centre terms are 1")


# ---------------------------------------------------------------------------------------- the scoring rules
def test_job_at_the_budget():
    s = new_session()
    s.block("block_1", "nominal").leg("j01").leg("j02", status="budget").leg("j03").hold("j04")
    rec = s.record()
    js, rows = job_scores(rec)
    st = {r["id"]: r["status"] for r in rows}
    check(st["j02"] == "budget" and st["j03"] == "not_started", "the world's statuses: budget, then not started")
    check(js["j02"] == (False, 0.0), "a job whose window has not closed at the budget scores 0 (jobs.finished)")
    check(js["j03"] == (False, 0.0) and js["j04"] == (False, 0.0), "a job not started at the budget scores 0")
    out = grade(rec)
    check(close(out["terms"]["legs"], 1.0 / 3.0, 1e-9), "the lost jobs stay in the mean with 0 (legs %.4f)"
          % out["terms"]["legs"])


def test_invalid_plan():
    s = new_session()
    s.block("block_1", "nominal").leg("j01", status="invalid_plan").leg("j02")
    rec = s.record()
    js, rows = job_scores(rec)
    check(js["j01"] == (False, 0.0), "a job with an invalid plan scores 0 (its recovery flew it to the goal)")
    check(js["j02"][1] == 1.0, "the block goes on")
    r = [x for x in rec["job_records"] if x["id"] == "j01"][0]
    check(close(r["recovery"]["t_to"] - r["recovery"]["t_from"], 25.008, 1e-6), "the recovery took the mode's "
          "monitor.recovery_s off the budget (one step's rounding)")


def test_hand_back():
    s = new_session()
    s.block("block_1", "nominal").leg("j01", status="handback").hold("j02").leg("j03").slew("j04")
    rec = s.record()
    js, rows = job_scores(rec)
    check(js["j01"] == (False, 0.0), "the job in progress at a hand-back scores 0 (its event was accepted)")
    check(js["j02"] == (False, 0.0), "the hold after it scores 0 (its job before was not completed)")
    check(js["j03"][1] == 1.0 and js["j04"][1] == 1.0, "the next jobs are flown and scored")
    out = grade(rec)
    check(close(out["terms"]["legs"], 0.5), "the handed-back job stays in the mean with 0")
    bad = copy.deepcopy(rec)
    for r in bad["job_records"]:
        if r["id"] == "j01":
            r["recovery"]["t_to"] = r["recovery"]["t_from"] + 10.0
    check(raises(RC.RecordError, grade, bad), "a recovery that does not take monitor.recovery_s off the budget "
          "contradicts the rule: a task error")
    s = new_session()
    s.block("block_1", "nominal").leg("j01", status="handback").leg("j02", status="handback").leg(
        "j03", status="handback").leg("j04").slew("j05")
    rec = s.record()
    js, rows = job_scores(rec)
    st = {r["id"]: r["status"] for r in rows}
    check(st["j04"] == "not_started" and st["j05"] == "not_started", "the third hand-back ends the block")
    out = grade(rec)
    check(close(out["terms"]["legs"], 0.0) and close(out["terms"]["slews"], 0.0),
          "the block's remaining jobs score 0 and stay in the mean")


def test_block_end():
    for st_ in ("watchdog", "protocol"):
        s = new_session()
        s.block("block_1", "nominal").leg("j01").leg("j02", status=st_).slew("j03").leg("j04")
        s.block("block_2", "nominal").view("j05")
        rec = s.record()
        js, rows = job_scores(rec)
        check(js["j02"] == (False, 0.0), "a block ended by %s: the job in progress scores 0" % st_)
        check(js["j03"] == (False, 0.0) and js["j04"] == (False, 0.0),
              "a block ended by %s: the aborted block's remaining jobs score 0" % st_)
        out = grade(rec)
        check(close(out["terms"]["legs"], 1.0 / 3.0) and close(out["terms"]["views"], 1.0),
              "a block ended by %s: the aborted block is kept in the mean; the next block is scored" % st_)


def test_session_end():
    s = new_session()
    s.block("block_1", "nominal").leg("j01").leg("j02", status="ended").leg("j03")
    s.block("block_2", "nominal").view("j04")
    rec = s.record()
    js, rows = job_scores(rec)
    check(js["j02"] == (False, 0.0) and js["j03"] == (False, 0.0), "the session's end zeroes the job and the rest")
    check(js["j04"] == (False, 0.0) and not any(r["block"] == "block_2" for r in rec["job_records"]),
          "a block never reached has no record and scores 0")
    out = grade(rec)
    check(close(out["terms"]["legs"], 1.0 / 3.0) and close(out["terms"]["views"], 0.0),
          "the jobs lost to the session's end stay in the mean with 0")
    bad = copy.deepcopy(rec)
    bad["ended"] = None
    check(raises(RC.RecordError, grade, bad), "a true speed over safety.overspeed_m_s without the world's end of "
          "the session is a task error (the score reads the truth)")


def test_hold_condition():
    s = new_session()
    s.block("block_1", "nominal").leg("j01", end_pos=0.08).hold("j02").slew("j03").hold("j04")
    js, rows = job_scores(s.record())
    check(js["j01"] == (False, 0.0), "the leg above its zero end is not completed")
    check(js["j02"] == (False, 0.0), "a perfect hold after a job that was not completed scores 0")
    check(js["j04"] == (True, 1.0), "a hold after a completed job scores")


def test_request_and_setup():
    rec = standard_session().record()
    for r in rec["job_records"]:
        if r["id"] == "j04":
            r["status"] = "accepted_request_not_flown"
    out = grade(rec)
    check(close(out["terms"]["refusals"], 0.5), "an accepted request of the mode kind is not refused (J = 0.5 + 1 "
          "- 1; refusals %.4f)" % out["terms"]["refusals"])
    rows = rows_of(rec)
    check([r for r in rows if r["id"] == "j04"][0]["measures"] is None, "the accepted mode request is not scored")
    for key in ("setup.overrun", "setup.failure"):
        r2 = standard_session().record()
        r2["runner"]["session_failed"] = key + ": start 1: the set-up allowance is used up"
        out = grade(r2)
        check(out["score"] == 0.0 and out["failed_setup"], "a session whose set-up %s scores 0"
              % ("overran" if key == "setup.overrun" else "failed"))


def test_view_centre_as_completion_limit():
    s = new_session()
    s.block("block_2", "nominal").view("j01", centre=0.040).view("j02", centre=0.030).view("j03", centre=0.075)
    js, rows = job_scores(s.record())
    fe, z = 0.038, 0.033
    check(z < 0.040 < 2 * fe, "the case: an optical-centre displacement between the cap and twice the full end")
    check(js["j01"] == (False, 0.0), "a view whose centre lies between its cap (0.033 m) and twice its full end is "
          "not captured, its cell dropped or not (score.completed)")
    check(js["j02"] == (True, 1.0), "a view inside the cap is captured; its dropped centre term is 1")
    check(js["j03"] == (False, 0.0), "a view beyond twice the full end is not captured")


def test_view_centre_zero_end():
    """the certification's zero end of the view's optical-centre displacement must be the cap that score.completed
    states (read by key from that entry's text); another zero end is a task error, never a score."""
    key = "%s/nominal/%s" % CFG.CENTRE
    doc = config_doc()
    cap = CFG.from_document(copy.deepcopy(doc)).centre_cap_m()
    check(("always its zero end (%r m)" % cap) in doc["values"]["score.completed"], "the cap is read by key from "
          "score.completed's text: %r m" % cap)
    d = copy.deepcopy(doc)
    d["values"]["score.completed"] = d["values"]["score.completed"].replace("(%r m)" % cap, "")
    check(raises(CFG.ConfigError, CFG.from_document, d), "a score.completed without the cap's number stops the grader")
    fx = SL.bands_fixture()
    check(fx["cells"][key]["zero_end"] == cap, "the fixture's centre zero end is the cap")
    check(not raises(CFG.ConfigError, fixture_cfg(fx).bands), "bands with the centre zero end at the cap are read")
    fe = fx["cells"][key]["full_end"]
    wrong = []
    for z in (cap + 0.001, cap - 0.001, 2.0 * fe):
        b = copy.deepcopy(fx)
        b["cells"][key]["zero_end"] = z
        d = config_doc()
        d["values"]["score.bands"] = b
        check(raises(CFG.ConfigError, CFG.from_document, d), "bands with the centre zero end %.4f m (not the cap) are "
              "refused when the configuration is read: a task error" % z)
        check(raises(CFG.ConfigError, CFG.Config(d).bands), "the bands' reader refuses the zero end %.4f m" % z)
        wrong.append(CFG.Config(d))
    # the verification's own path: the guard's 'bands' stage voids the run; with the cap the stage passes
    old = os.environ.get("ALE_STAGE_DIR")
    st = tempfile.mkdtemp(prefix="ffctl_stage_")
    try:
        os.environ["ALE_STAGE_DIR"] = st
        import shutil
        shutil.copyfile(os.path.join(ARGS["stage"], "grader_config.json"), os.path.join(st, "grader_config.json"))
        hs = os.path.join(st, cfg().layout["hidden_set"])
        os.makedirs(os.path.dirname(hs))
        names = cfg().sessions()
        with open(hs, "w") as f:
            json.dump({"format": "estimator-controller-hidden-set/1",
                       "sessions": {n: {"world_session": n, "tight_mode": "quiet", "list": 1, "seed": "acceptance",
                                        "scored": ["j01"]} for n in names}}, f)
        spec = importlib.util.spec_from_file_location("ffctl_verify_centre", os.path.join(ARGS["stage"], "verify.py"))
        V = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(V)
        G = importlib.import_module("grader.guard")
        early = {"submission_oversize": 1.0, "submission_late_writes": 0.0, "submission_root_rejected": 0.0,
                 "submission_root_violation": 0.0, "submission_snapshot": 0.0, "snapshot": None, "reasons": []}
        for c in wrong:
            try:
                V.run(c, early, __import__("time").time())
                voided = ""
            except G.GraderVoid as exc:
                voided = str(exc)
            check(voided.startswith("stage bands: ConfigError"), "verify.run is void at its 'bands' stage for a wrong centre zero end (a task "
                  "error, no score)")
        out, _r = V.run(fixture_cfg(fx), early, __import__("time").time())
        check(out["reward"] == 0.0 and out["gates"]["caps"] == 0.0, "with the zero end at the cap the 'bands' stage "
              "passes (an over-cap submission then scores 0, nothing flown)")
    finally:
        if old is None:
            os.environ.pop("ALE_STAGE_DIR", None)
        else:
            os.environ["ALE_STAGE_DIR"] = old
        import shutil
        shutil.rmtree(st, ignore_errors=True)
    # the shipped configuration: the certification has set score.bands (fixed, not pending); its centre zero end is the
    # cap, read from score.completed's text alone (so that this check names its own failure)
    shipped = doc["values"]["score.bands"]
    tcap = CFG.Config(copy.deepcopy(doc)).centre_cap_m()
    check(shipped is not None and "score.bands" not in doc["pending_in_table"], "the shipped configuration carries "
          "score.bands, and the table no longer marks it pending")
    z = ((shipped.get("cells") or {}).get(key) or {}).get("zero_end") if isinstance(shipped, dict) else None
    check(isinstance(z, float) and z == tcap, "the shipped configuration's centre zero end %r m is the cap %r m"
          % (z, tcap))


def test_dropped_measure():
    s = new_session()
    b = s.block("block_1", "nominal")
    b.leg("j01").hold("j02", speed_rms=0.012).leg("j03").hold("j04", speed_rms=0.015)
    js, rows = job_scores(s.record())
    check(js["j02"] == (True, 1.0), "a dropped measure of another kind (hold speed RMS 0.012 m/s, Z 0.0105, FE "
          "0.007) is tested at twice its full end: completed, its L is 1")
    check(js["j04"] == (False, 0.0), "the same measure above twice its full end (0.015 > 0.014): not completed")
    s = new_session()
    s.block("block_3", "quiet").leg("j01", overshoot=0.055).leg("j02", overshoot=0.065)
    js, rows = job_scores(s.record())
    check(js["j01"] == (True, 1.0) and js["j02"] == (False, 0.0), "the quiet leg's dropped overshoot is tested at 2 FE")


def test_completion_at_zero_end():
    s = new_session()
    s.block("block_1", "nominal").leg("j01", end_pos=0.09).hold("j02").request("j03", "mode_not_available")
    out = grade(s.record())
    check(out["terms"]["legs"] == 0.0 and out["terms"]["holds"] == 0.0, "a leg above its zero end is not completed and "
          "conditions its hold")
    check(close(out["terms"]["refusals"], 0.0), "the refusal term's share of completed jobs counts completions (0 of 2)")
    s = new_session()
    s.block("block_3", "quiet").leg("j01", overshoot=0.07)
    js, rows = job_scores(s.record())
    check(js["j01"] == (False, 0.0), "a dropped measure above its completion limit (2 FE) zeroes the job although its "
          "L would be 1 (the immediate done of a degenerate controller)")


# ------------------------------------------------------------------------------------------------ the session
def test_scored_set():
    rec = standard_session().record()
    for r in rec["job_records"]:
        if r["id"] == "j01":
            r["status"] = "budget"
    full = grade(rec)
    part = grade(rec, [x for x in scored_all(rec) if x not in ("j01", "j02")])
    check(full["terms"]["legs"] < 1.0 and close(part["terms"]["legs"], 1.0) and close(part["terms"]["holds"], 1.0),
          "a job outside the scored set leaves the session (it is not a 0); a hold falls with its job before")
    check(raises(SC.ScoreError, grade, rec, scored_all(rec) + ["j99"]), "a scored set naming a job the session does not "
          "hold is a task error")


def test_kind_without_jobs():
    s = new_session()
    s.block("block_1", "nominal").leg("j01", end_pos=0.0475).slew("j02").request("j03", "mode_not_available")
    out = grade(s.record())
    w = cfg().weights()
    tot = w["legs"] + w["slews"] + w["refusals"]
    want = (w["legs"] * 0.5 + w["slews"] * 1.0 + w["refusals"] * 1.0) / tot
    check(sorted(out["terms"]) == ["legs", "refusals", "slews"], "kinds without jobs drop out")
    check(close(out["score"], want, 1e-9), "the weights are renormalised over the kinds present (%.6f against %.6f)"
          % (out["score"], want))


def test_refusal_term():
    s = new_session()
    b = s.block("block_1", "nominal")
    b.leg("j01").slew("j02", status="refused_feasible").request("j03", "mode_not_available").request(
        "j04", "outside_zones", refused=False).leg("j05", end_pos=0.08)
    out = grade(s.record())
    J = 0.5 + 2.0 / 3.0 - 1.0
    check(close(out["terms"]["refusals"], J * 0.5, 1e-9), "refusals = J x the share of accepted feasible jobs completed "
          "(J = 1/2 + 2/3 - 1, done 1 of 2: %.6f)" % out["terms"]["refusals"])
    check(close(out["terms"]["slews"], 0.0), "a refused feasible job scores 0 itself")


def test_aggregation_and_gates():
    names = cfg().sessions()
    recs = {n: standard_session(name=n).record() for n in names}
    for r in recs[names[1]]["job_records"]:
        if r["id"] in ("j06", "j07"):
            r["status"] = "budget"
    res = {n: grade(recs[n]) for n in names}
    out = GR.grade_run(res, cfg(), True, True)
    want = sum(res[n]["score"] for n in names) / len(names)
    check(close(out["reward"], want, 1e-12) and out["reward"] < 1.0, "reward = gates x the mean of the six sessions")
    check(all(v == 1.0 for v in out["gates"].values()), "every gate 1 for an honest run")
    check(GR.grade_run(res, cfg(), False, True)["reward"] == 0.0, "an over-cap submission scores 0")
    check(GR.grade_run(res, cfg(), True, False)["reward"] == 0.0, "a failed import audit scores 0")
    r2 = copy.deepcopy(res)
    r2[names[0]]["runner"]["steps"] = {"steps_answered": 1000, "step_calls_skipped": 50, "late_steps": 70}
    for n in names[1:]:
        r2[n]["runner"]["steps"] = {"steps_answered": 1000, "step_calls_skipped": 50, "late_steps": 70}
    check(GR.grade_run(r2, cfg(), True, True)["gates"]["on_time"] == 0.0, "an on-time share below "
          "interface.on_time_share_gate over the run zeroes the reward (skipped calls are not on time)")
    r3 = copy.deepcopy(res)
    r3[names[2]]["runner"]["runner"]["stray_processes"] = 1
    check(GR.grade_run(r3, cfg(), True, True)["reward"] == 0.0, "a process of the controller alive at a session's end "
          "zeroes the reward")


def test_never_comes_up():
    names = cfg().sessions()
    res = {}
    for n in names:
        s = standard_session(name=n)
        s.runner["start_outcomes"] = ["start 1: the process ended at its start"]
        s.runner["session_failed"] = "setup.failure: start 1: the process ended at its start"
        s.runner["steps"] = {"steps_answered": 0, "step_calls_skipped": 0, "late_steps": 0}
        res[n] = grade(s.record())
    out = GR.grade_run(res, cfg(), True, True)
    check(out["reward"] == 0.0 and out["gates"]["start_came_up"] == 0.0, "a controller that never comes up scores 0")
    check("gate_start_came_up" in out["metrics"] and not any("import" in k for k in out["metrics"]),
          "the output names the first gate by what it measures (at least one start came up)")
    one_up = [res[n]["runner"] for n in names]
    one_up[0] = grade(standard_session(name=names[0]).record())["runner"]
    check(SC.gate_values(one_up, True, True, cfg().on_time_share_gate())[0]["start_came_up"] == 1.0,
          "one start that came up in the run is enough for the first gate")


def test_task_errors():
    rec = standard_session().record()
    bad = copy.deepcopy(rec)
    bad["truth"]["p"][100, 0] = float("nan")
    check(raises(RC.RecordError, grade, bad), "a non-finite truth is a task error, never a score")
    bad = copy.deepcopy(rec)
    bad["truth"]["q"][100] *= 2.0
    check(raises(RC.RecordError, grade, bad), "a truth attitude that is not a unit quaternion is a task error")
    for field in ("job_records", "block_records", "truth", "runner", "blocks", "ended"):
        b2 = copy.deepcopy(rec)
        del b2[field]
        check(raises(RC.RecordError, grade, b2, scored_all(rec)), "a record without %s is a task error" % field)
    b3 = copy.deepcopy(rec)
    b3["job_records"][0]["status"] = "great"
    check(raises(RC.RecordError, grade, b3), "a status the world does not write is a task error")
    b4 = copy.deepcopy(rec)
    b4["job_records"][0], b4["job_records"][1] = b4["job_records"][1], b4["job_records"][0]
    check(raises(RC.RecordError, grade, b4), "job records out of the block's order are a task error")
    b5 = copy.deepcopy(rec)
    b5["runner"]["task_error"] = {"error": "sandbox not installed"}
    check(raises(SC.ScoreError, grade, b5), "the system-call filter that cannot be installed is a task error")
    b6 = copy.deepcopy(rec)
    b6["runner"]["session_failed"] = "task_error.sandbox_missing: sandbox not installed"
    check(raises(SC.ScoreError, grade, b6), "the runner's task_error failure is a task error")
    b7 = copy.deepcopy(rec)
    del b7["runner"]["runner"]["stray_processes"]
    check(raises(RC.RecordError, grade, b7), "a runner record without its stray-process count is a task error")
    names = cfg().sessions()
    good = {"format": "estimator-controller-hidden-set/1",
            "sessions": {n: {"world_session": n, "tight_mode": "quiet", "list": 1, "seed": "acceptance",
                             "scored": ["j01"]} for n in names}}
    check(sorted(GR.load_hidden_set(good, cfg())) == sorted(names), "a well-formed hidden set is read")
    for mut in ("missing_session", "no_scored", "bad_format", "bool_list"):
        h = copy.deepcopy(good)
        if mut == "missing_session":
            del h["sessions"][names[0]]
        elif mut == "no_scored":
            h["sessions"][names[1]]["scored"] = []
        elif mut == "bad_format":
            h["format"] = "x"
        else:
            h["sessions"][names[2]]["list"] = True
        check(raises(ValueError, GR.load_hidden_set, h, cfg()), "a malformed hidden set (%s) is a task error" % mut)
    check(raises(CFG.ConfigError, fixture_cfg().bands) is False, "the fixture's bands are read")
    # a configuration whose score.bands is null, made in memory from the shipped one (whose bands are fixed)
    doc = config_doc()
    doc["values"]["score.bands"] = None
    doc["pending_in_table"] = sorted(set(doc["pending_in_table"]) | {"score.bands"})
    c_null = CFG.from_document(copy.deepcopy(doc))
    check(raises(CFG.ConfigError, c_null.bands) and raises(CFG.ConfigError, CFG.Config(doc).bands),
          "score.bands null (pending) is read but cannot be scored with: a task error")
    doc["pending_in_table"] = sorted(set(doc["pending_in_table"]) - {"score.bands"})
    check(raises(CFG.ConfigError, CFG.from_document, doc), "score.bands null without being pending is refused when "
          "the configuration is read")


def test_configuration_keys():
    doc = config_doc()
    check(CFG.from_document(copy.deepcopy(doc)) is not None, "the generated configuration is read")
    for block, keys in (("values", CFG.VALUE_KEYS), ("layout", CFG.LAYOUT_KEYS), ("runtime", CFG.RUNTIME_KEYS)):
        for k in keys:
            d = copy.deepcopy(doc)
            del d[block][k]
            if not raises(CFG.ConfigError, CFG.from_document, d):
                check(False, "removing %s.%s is not refused" % (block, k))
                break
        else:
            check(True, "removing any key of %s is an error (no knob has a default in code)" % block)
        d = copy.deepcopy(doc)
        d[block]["an.extra.key"] = 1
        check(raises(CFG.ConfigError, CFG.from_document, d), "an extra key in %s is an error" % block)
    d = copy.deepcopy(doc)
    d["values"]["score.weights"] = {"legs": 1.0}
    check(raises(CFG.ConfigError, CFG.from_document, d), "malformed weights are refused")
    d = copy.deepcopy(doc)
    d["values"]["score.bands"] = {"cells": {}}
    check(raises(CFG.ConfigError, CFG.from_document, d), "bands without their 33 cells are refused")
    check(set(doc["values"]) == set(CFG.VALUE_KEYS), "the configuration's values are exactly the grader's keys")
    check(CFG.from_document(copy.deepcopy(doc)).overshoot_terms() == (0.8, 0.2), "the factor (a + b O) is read from "
          "the formulas' text (score.leg, score.slew)")
    for key, new in (("score.leg", "completed x L(end position error) x (0.7 + 0.3 O)"),
                     ("score.completed", "a leg is completed if its event done was accepted"),
                     ("score.refusals", "J"), ("score.hold", "L(position RMS)")):
        d = copy.deepcopy(doc)
        d["values"][key] = new
        check(raises(CFG.ConfigError, CFG.from_document, d), "a changed formula text (%s) stops the grader" % key)


# the registered cells of score.bands, written out here apart from the grader's code: the terms of the formulas
# (score.leg, score.slew, score.hold, score.view) in the modes the session blocks fly (legs, slews and holds in the
# three modes, the survey views in the nominal mode)
REGISTERED_MODES = {"leg": ("nominal", "quiet", "precision"), "slew": ("nominal", "quiet", "precision"),
                    "hold": ("nominal", "quiet", "precision"), "view": ("nominal",)}
REGISTERED_MEASURES = {"leg": ("end_pos", "end_att", "overshoot"), "slew": ("end_att", "end_pos", "overshoot"),
                       "hold": ("pos_rms", "att_rms", "speed_rms", "rate_rms"),
                       "view": ("pointing", "centre", "rate_rms")}


def frozen_band_records(doc):
    """[(name, bands)]: the records of the frozen bands this run can read: the file named with --bands-record (a
    record holding score_bands_value, or a table holding values["score.bands"]["value"]; a named file without either
    gives None, which fails the comparison) and the copy of the table a placed stage holds at the layout's world_table
    when it is present and holds score.bands."""
    def read(p):
        with open(p, "rb") as fh:
            rec = json.loads(fh.read().decode("utf-8"))
        if isinstance(rec, dict) and "score_bands_value" in rec:
            return "score_bands_value", rec["score_bands_value"]
        if isinstance(rec, dict) and isinstance(rec.get("values"), dict) and "score.bands" in rec["values"]:
            return "score.bands", rec["values"]["score.bands"].get("value")
        return "no frozen bands", None
    out = []
    if ARGS["bands-record"]:
        field, bands = read(ARGS["bands-record"])
        out.append(("the record named on the command line (%s)" % field, bands))
    wt = os.path.join(ARGS["stage"], doc["layout"]["world_table"])
    if os.path.isfile(wt):
        field, bands = read(wt)
        if bands is not None:
            out.append(("the stage's copy of the table at the layout's world_table (%s)" % field, bands))
    return out


def test_shipped_bands():
    """the shipped configuration's score.bands as the band freeze fixed it: present with the 33 registered cells, every
    end a finite double above 0, the optical centre's zero end at the cap of score.completed, no longer pending, read
    by the grader as it stands, and equal as doubles to the frozen bands of every record this run can read."""
    doc = config_doc()
    b = doc["values"]["score.bands"]
    want = sorted("%s/%s/%s" % (k, m, x) for k in REGISTERED_MODES for m in REGISTERED_MODES[k]
                  for x in REGISTERED_MEASURES[k])
    check(len(want) == len(set(want)) == 33 and want == CFG.all_cells() and sorted(REGISTERED_MODES) ==
          sorted(doc["values"]["jobs.kinds"]), "the registered cells: %d (kind, mode, measure), the grader's own list "
          "and the kinds of jobs.kinds" % len(want))
    cells = b.get("cells") if isinstance(b, dict) else None
    check(isinstance(b, dict) and set(b) == {"cells"} and isinstance(cells, dict) and sorted(cells) == want,
          "score.bands is present with exactly the %d registered cells" % len(want))
    cells = cells if isinstance(cells, dict) else {}
    ends = [(k, e, c.get(e)) for k, c in sorted(cells.items()) if isinstance(c, dict) for e in ("full_end", "zero_end")]
    check(all(isinstance(c, dict) and set(c) == {"full_end", "zero_end"} for c in cells.values()) and
          len(ends) == 2 * len(want), "every cell holds exactly its full end and its zero end")
    bad = [(k, e, x) for k, e, x in ends if not (isinstance(x, float) and math.isfinite(x) and x > 0.0)]
    check(not bad, "every full end and zero end is a finite double above 0 (departures: %s)" % (bad[:3] or "none"))
    tcap = CFG.Config(copy.deepcopy(doc)).centre_cap_m()
    zc = (cells.get("%s/nominal/%s" % CFG.CENTRE) or {}).get("zero_end")
    check(("always its zero end (%r m)" % tcap) in doc["values"]["score.completed"] and isinstance(zc, float) and
          zc == tcap, "the optical centre's zero end %r m is the cap of score.completed, %r m" % (zc, tcap))
    check("score.bands" not in doc["pending_in_table"] and doc["generated_from"]["fixture"] is False,
          "score.bands is not in the configuration's pending list, and the configuration is the generated one")
    try:
        got = CFG.from_document(copy.deepcopy(doc)).bands()
        err = None
    except CFG.ConfigError as exc:
        got, err = None, exc
    check(err is None and got == {k: (c["full_end"], c["zero_end"]) for k, c in cells.items()},
          "the grader reads the shipped bands as they are (%s)" % (err or "no error"))
    for name, rec in frozen_band_records(doc):
        COMPARED.append(name)
        rc = rec.get("cells") if isinstance(rec, dict) and set(rec) == {"cells"} else None
        same = (isinstance(rc, dict) and sorted(rc) == sorted(cells) and
                all(isinstance(rc[k], dict) and set(rc[k]) == set(cells[k]) == {"full_end", "zero_end"} and
                    isinstance(rc[k][e], float) and isinstance(cells[k][e], float) and rc[k][e] == cells[k][e]
                    for k in rc for e in ("full_end", "zero_end")))
        check(same, "the shipped cells equal, as doubles, the frozen bands of %s" % name)
    if not COMPARED:
        print("  note  no record of the frozen bands in this run (no --bands-record, no copy of the table in the stage): "
              "the shipped cells are compared with none")


def test_audit():
    d = tempfile.mkdtemp(prefix="ffctl_audit_")
    try:
        os.makedirs(os.path.join(d, "pkg"))
        with open(os.path.join(d, "controller.py"), "w") as f:
            f.write("import math, json\nimport numpy as np\nfrom scipy.linalg import expm\nimport helper\n"
                    "from pkg import util\nfrom . import x\n")
        with open(os.path.join(d, "helper.py"), "w") as f:
            f.write("import collections\n")
        with open(os.path.join(d, "pkg", "util.py"), "w") as f:
            f.write("import os.path\n")
        ok, fnd = AU.audit(d, cfg().libraries())
        check(ok and not fnd, "the standard library, numpy, scipy and the submission's own modules pass the audit")
        with open(os.path.join(d, "pkg", "bad.py"), "w") as f:
            f.write("import torch\n")
        ok, fnd = AU.audit(d, cfg().libraries())
        check(not ok and any("torch" in x for x in fnd), "an import outside interface.libraries fails the audit")
        os.remove(os.path.join(d, "pkg", "bad.py"))
        with open(os.path.join(d, "broken.py"), "w") as f:
            f.write("def (:\n")
        ok, fnd = AU.audit(d, cfg().libraries())
        check(not ok, "a file that cannot be parsed cannot be audited: the audit fails")
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_three_file_form():
    rec = standard_session().record()
    d = tempfile.mkdtemp(prefix="ffctl_files_")
    try:
        res = {k: rec[k] for k in ("session", "seed", "tight_mode", "list", "ended")}
        rep = dict(rec["runner"], world_job_records=rec["job_records"], world_block_records=rec["block_records"])
        with open(os.path.join(d, "r.json"), "w") as f:
            json.dump(res, f)
        with open(os.path.join(d, "r_runner.json"), "w") as f:
            json.dump(rep, f)
        np.savez_compressed(os.path.join(d, "r_log.npz"), **rec["truth"])
        back = RC.read_files(os.path.join(d, "r.json"), os.path.join(d, "r_log.npz"), os.path.join(d, "r_runner.json"),
                             rec["blocks"])
        a, b = grade(rec), grade(back)
        check(a["score"] == b["score"] and a["terms"] == b["terms"], "the three-file form scores as the record in memory")
        os.remove(os.path.join(d, "r_log.npz"))
        check(raises(RC.RecordError, RC.read_files, os.path.join(d, "r.json"), os.path.join(d, "r_log.npz"),
                     os.path.join(d, "r_runner.json"), rec["blocks"]), "a missing truth log is a task error")
        with open(os.path.join(d, "r_runner.json"), "w") as f:
            f.write("{not json")
        np.savez_compressed(os.path.join(d, "r_log.npz"), **rec["truth"])
        check(raises(RC.RecordError, RC.read_files, os.path.join(d, "r.json"), os.path.join(d, "r_log.npz"),
                     os.path.join(d, "r_runner.json"), rec["blocks"]), "a malformed runner record is a task error")
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_grader_output():
    names = cfg().sessions()
    res = {n: grade(standard_session(name=n).record()) for n in names}
    out = GR.grade_run(res, cfg(), True, True)
    keys = set(out["metrics"])
    allowed = {"reward", "on_time_share"} | {"gate_" + g for g in out["gates"]}
    for n in names:
        allowed |= {n + "_score"} | {n + "_" + k for k in ("legs", "slews", "holds", "views", "refusals")}
    check(keys <= allowed, "the output holds the reward, the gates and per session its score and per kind its term only")
    check(not any(("j%02d" % i) in k for k in keys for i in range(100)), "no job id in the output")
    check(all(isinstance(v, float) and math.isfinite(v) for v in out["metrics"].values()), "every value finite")
    rec = standard_session().record()
    rows = rows_of(rec)
    measures = {round(x, 12) for r in rows if r["measures"] for x in r["measures"].values() if x > 0}
    check(not ({round(v, 12) for v in out["metrics"].values()} & measures), "no measure of a job in the output")


def test_verify_void_without_hidden_set():
    sys.path.insert(0, ARGS["stage"])
    old = os.environ.get("ALE_STAGE_DIR")
    d = tempfile.mkdtemp(prefix="ffctl_stage_")
    try:
        os.environ["ALE_STAGE_DIR"] = d
        import shutil
        shutil.copyfile(os.path.join(ARGS["stage"], "grader_config.json"), os.path.join(d, "grader_config.json"))
        spec = importlib.util.spec_from_file_location("ffctl_verify_under_test", os.path.join(ARGS["stage"], "verify.py"))
        V = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(V)
        G = importlib.import_module("grader.guard")
        early = {"submission_oversize": 0.0, "submission_late_writes": 0.0, "submission_root_rejected": 0.0,
                 "submission_root_violation": 0.0, "submission_snapshot": 0.0, "snapshot": None, "reasons": []}
        check(raises(G.GraderVoid, V.run, fixture_cfg(), early, __import__("time").time()),
              "a missing hidden set voids the verification (a task error)")
        hs = os.path.join(d, cfg().layout["hidden_set"])
        os.makedirs(os.path.dirname(hs))
        with open(hs, "w") as f:
            f.write("{}")
        check(raises(G.GraderVoid, V.run, fixture_cfg(), early, __import__("time").time()),
              "a malformed hidden set voids the verification")
    finally:
        if old is None:
            os.environ.pop("ALE_STAGE_DIR", None)
        else:
            os.environ["ALE_STAGE_DIR"] = old
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_verdict_writer():
    VD = importlib.import_module("grader.verdict")
    d = tempfile.mkdtemp(prefix="ffctl_verdict_")
    try:
        p = os.path.join(d, "verdict.json")
        env = VD.write(0.25, {"reward": 0.25, "gate_caps": 1.0}, diagnostic="x", path=p)
        with open(p) as fh:
            back = json.load(fh)
        check(back["rewards"] == {"reward": 0.25} and back["metrics"]["gate_caps"] == 1.0, "the envelope holds the "
              "reward and the metrics")
        check(raises(SystemExit, VD.write, 1.5, {}, path=p), "a reward outside [0, 1] is a task error")
        check(raises(SystemExit, VD.write, 0.5, {"m": float("nan")}, path=p), "a non-finite metric is a task error")
        check(raises(SystemExit, VD.write, 0.5, {}, path=None) if not os.environ.get("ALE_VERIFY_PYTHON") else True,
              "outside the engine a verdict without a path is a task error")
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_access_token():
    """the solution's access token (grader/access.py, verify.py's 'token' stage, grader/fly.py's reads at set-up): the
    right token is recognised; an absent, wrong or malformed one, a link or a folder is no token and never an error;
    the stage's hash file is the grader's own: missing or malformed, the verification is void."""
    import hashlib
    import importlib.util
    import inspect
    import shutil
    AC = importlib.import_module("grader.access")
    FLm = importlib.import_module("grader.fly")
    lay = cfg().layout
    check(lay["token_file"] == "access_token" and lay["token_sha256"] == "access_token.sha256",
          "the layout names the token's file in the controller folder and its hash file in the stage")
    check(isinstance(FLm.READS_KEY, str) and FLm.READS_KEY and
          inspect.signature(FLm.fly_session).parameters["reads_at_setup"].default is False,
          "fly_session flies without the reads unless the caller says the folder holds the token")
    d = tempfile.mkdtemp(prefix="ffctl_token_")
    try:
        tok = os.urandom(32).hex().encode("ascii")           # a test token of this run, not the solution's
        want = hashlib.sha256(tok).hexdigest()
        hp = os.path.join(d, "hash")
        with open(hp, "w") as f:
            f.write(want + "\n")
        check(AC.read_hash(hp) == want, "the hash file is read")
        for what, content in (("empty", ""), ("not hex", "x" * 64), ("short", want[:63]), ("two lines", want + "\n" + want),
                              ("upper case", want.upper())):
            with open(hp + "_bad", "w") as f:
                f.write(content)
            check(raises(ValueError, AC.read_hash, hp + "_bad"), "a hash file that is %s is refused (a task error)" % what)
        check(raises(OSError, AC.read_hash, os.path.join(d, "absent")), "a missing hash file is refused (a task error)")
        os.symlink(hp, hp + "_link")
        check(raises(ValueError, AC.read_hash, hp + "_link"), "a hash file that is a link is refused")
        cases = {}

        def folder(name):
            p = os.path.join(d, name)
            os.makedirs(p)
            return p
        f_right = folder("right")
        with open(os.path.join(f_right, "access_token"), "wb") as f:
            f.write(tok + b"\n")
        cases["the right token"] = (f_right, True)
        f_ws = folder("white_space")
        with open(os.path.join(f_ws, "access_token"), "wb") as f:
            f.write(b"  \r\n" + tok + b"\r\n\t ")
        cases["the right token with white space around it"] = (f_ws, True)
        cases["no token file"] = (folder("absent"), False)
        for name, content in (("wrong", os.urandom(32).hex().encode()), ("empty", b""), ("binary", os.urandom(4096)),
                              ("upper", tok.upper()), ("twice", tok + b"\n" + tok), ("huge", b"0" * (2 * 1024 * 1024)),
                              ("prefix", tok[:-1])):
            p = folder(name)
            with open(os.path.join(p, "access_token"), "wb") as f:
                f.write(content)
            cases["a token file that is " + name] = (p, False)
        p = folder("link")
        os.symlink(os.path.join(f_right, "access_token"), os.path.join(p, "access_token"))
        cases["a link to a file holding the right token"] = (p, False)
        p = folder("dir")
        os.makedirs(os.path.join(p, "access_token"))
        cases["a folder named as the token"] = (p, False)
        p = folder("fifo")
        os.mkfifo(os.path.join(p, "access_token"))
        cases["a named pipe named as the token"] = (p, False)
        for what, (p, expect) in sorted(cases.items()):
            try:
                got = AC.holds_token(p, lay["token_file"], want)
                err = None
            except Exception as exc:  # noqa: BLE001
                got, err = None, exc
            check(err is None and got is expect, "%s: %s, no error" % (what, "recognised" if expect else "no token"))
        check(AC.holds_token(f_right, lay["token_file"], "not a digest") is False and
              AC.holds_token(None, lay["token_file"], want) is False and AC.holds_token(f_right, lay["token_file"], None) is False,
              "a malformed expected digest or no folder is no token, no error")
        # verify.run: the stage's hash file is read after the entry gates; missing or malformed, the run is void there
        old = {k: os.environ.get(k) for k in ("ALE_STAGE_DIR", "ALE_HOME")}
        st = tempfile.mkdtemp(prefix="ffctl_stage_")
        home = tempfile.mkdtemp(prefix="ffctl_home_")
        try:
            os.environ["ALE_STAGE_DIR"] = st
            os.environ["ALE_HOME"] = home
            shutil.copyfile(os.path.join(ARGS["stage"], "grader_config.json"), os.path.join(st, "grader_config.json"))
            hs = os.path.join(st, lay["hidden_set"])
            os.makedirs(os.path.dirname(hs))
            names = cfg().sessions()
            with open(hs, "w") as f:
                json.dump({"format": "estimator-controller-hidden-set/1",
                           "sessions": {n: {"world_session": n, "tight_mode": "quiet", "list": 1, "seed": "acceptance",
                                            "scored": ["j01"]} for n in names}}, f)
            spec = importlib.util.spec_from_file_location("ffctl_verify_token", os.path.join(ARGS["stage"], "verify.py"))
            V = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(V)
            G = importlib.import_module("grader.guard")
            early = {"submission_oversize": 0.0, "submission_late_writes": 0.0, "submission_root_rejected": 0.0,
                     "submission_root_violation": 0.0, "submission_snapshot": 0.0, "snapshot": None, "reasons": []}

            def voided():
                try:
                    V.run(fixture_cfg(), early, __import__("time").time())
                    return ""
                except G.GraderVoid as exc:
                    return str(exc)
            check(voided().startswith("stage token:"), "a missing hash file voids the verification at the token stage")
            with open(os.path.join(st, lay["token_sha256"]), "w") as f:
                f.write("not a digest\n")
            check(voided().startswith("stage token:"), "a malformed hash file voids the verification at the token stage")
            with open(os.path.join(st, lay["token_sha256"]), "w") as f:
                f.write(want + "\n")
            v = voided()
            check(v.startswith("stage session_1.fly:"), "with the hash file the token stage passes (this stage has no world, "
                  "so the run is void at the first flight): %s" % v[:80])
        finally:
            for k, val in old.items():
                if val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = val
            shutil.rmtree(st, ignore_errors=True)
            shutil.rmtree(home, ignore_errors=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_work_folder_removed():
    """verify.run removes its work folder at the end also when it does not run as root: the runner's run folder (its
    read-only copy of the submission, a token in it included) does not stay, and a link inside it is not followed. The
    flight is replaced by the runner's own preparation of the run folder (runner/procs.py prepare_run_dir; on a copy of
    the grader without the runner, the same layout made read-only the same way), then raises."""
    import hashlib
    import importlib.util
    import shutil
    import stat as statmod
    try:
        PR = importlib.import_module("runner.procs")
        WI = importlib.import_module("runner.wire")

        def prepare(root, prog):
            return PR.prepare_run_dir(root, prog, WI.CLIENT, WI.CLIENT_NAME)
        how = "the runner's own run folder"
    except ImportError:
        def prepare(root, prog):
            """a stage without the runner (a copy of the grader alone): the runner's layout, read-only as it makes it"""
            program = os.path.join(root, "program")
            shutil.copytree(prog, program)
            for dp, dn, fn in os.walk(program, topdown=False):
                for f in fn:
                    os.chmod(os.path.join(dp, f), 0o444)
                os.chmod(dp, 0o555)
            out = {"root": root, "program": program}
            for name in ("home", "tmp"):
                out[name] = os.path.join(root, name)
                os.makedirs(out[name])
                os.chmod(out[name], 0o700)
            return out
        how = "a run folder made read-only as the runner makes it (this stage holds no runner)"
    FLm = importlib.import_module("grader.fly")
    G = importlib.import_module("grader.guard")
    lay = cfg().layout
    base = tempfile.mkdtemp(prefix="ffctl_work_")
    old_env = {k: os.environ.get(k) for k in ("ALE_STAGE_DIR", "ALE_HOME")}
    old_tmp, old_fly = tempfile.tempdir, FLm.fly_session
    root_run = os.getuid() == 0
    try:
        st, home, tmpd, victim = (os.path.join(base, n) for n in ("stage", "home", "tmp", "victim"))
        for p in (st, tmpd, victim):
            os.makedirs(p)
        with open(os.path.join(victim, "kept.txt"), "w") as f:
            f.write("outside the work folder\n")
        os.chmod(victim, 0o555)
        tok = os.urandom(32).hex().encode("ascii")              # a test token of this run, not the solution's
        sub = os.path.join(home, os.path.normpath(lay["submission_dir"]))
        os.makedirs(os.path.join(sub, "pkg"))
        with open(os.path.join(sub, "controller.py"), "w") as f:
            f.write("import json\n\n\nclass Controller(object):\n    def setup(self, session):\n        return None\n")
        with open(os.path.join(sub, "pkg", "data.json"), "w") as f:
            f.write("{}\n")
        with open(os.path.join(sub, lay["token_file"]), "wb") as f:
            f.write(tok + b"\n")
        shutil.copyfile(os.path.join(ARGS["stage"], "grader_config.json"), os.path.join(st, "grader_config.json"))
        hs = os.path.join(st, lay["hidden_set"])
        os.makedirs(os.path.dirname(hs))
        with open(hs, "w") as f:
            json.dump({"format": "estimator-controller-hidden-set/1",
                       "sessions": {n: {"world_session": n, "tight_mode": "quiet", "list": 1, "seed": "acceptance",
                                        "scored": ["j01"]} for n in cfg().sessions()}}, f)
        with open(os.path.join(st, lay["token_sha256"]), "w") as f:
            f.write(hashlib.sha256(tok).hexdigest() + "\n")
        # the cause, shown on the runner's own layout: as an account that is not root a plain removal leaves it
        probe = os.path.join(base, "probe_run")
        prepare(probe, sub)
        shutil.rmtree(probe, ignore_errors=True)
        if not root_run:
            check(os.path.isdir(os.path.join(probe, "program")),
                  "as an account that is not root a plain removal leaves the read-only copy (the cause; %s)" % how)
        os.environ["ALE_STAGE_DIR"] = st
        os.environ["ALE_HOME"] = home
        spec = importlib.util.spec_from_file_location("ffctl_verify_work", os.path.join(ARGS["stage"], "verify.py"))
        V = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(V)
        remove = getattr(V, "_remove_work", None)
        check(remove is not None and remove(probe) and not os.path.lexists(probe), "verify's removal takes the runner's "
              "read-only run folder away (as %s)" % ("root" if root_run else "an account that is not root"))
        seen = []

        def fake_fly(paths, entry, prog, run_root, as_bytes, rec_s, reads=False):
            rd = prepare(run_root, prog)
            os.symlink(victim, os.path.join(rd["home"], "link_out"))
            m = statmod.S_IMODE(os.lstat(rd["program"]).st_mode)
            seen.append({"run_root": run_root, "reads": reads, "program_mode": m,
                         "token_copied": os.path.isfile(os.path.join(rd["program"], lay["token_file"])),
                         "under_tmp": os.path.commonpath([os.path.realpath(run_root), os.path.realpath(tmpd)])
                         == os.path.realpath(tmpd)})
            raise RuntimeError("the test's flight stops after the runner's copy")
        FLm.fly_session = fake_fly
        tempfile.tempdir = tmpd
        early = {"submission_oversize": 0.0, "submission_late_writes": 0.0, "submission_root_rejected": 0.0,
                 "submission_root_violation": 0.0, "submission_snapshot": 0.0, "snapshot": None, "reasons": []}
        try:
            V.run(fixture_cfg(), early, __import__("time").time())
            voided = ""
        except G.GraderVoid as exc:
            voided = str(exc)
        check(voided.startswith("stage session_1.fly:") and len(seen) == 1,
              "the run reaches the first flight, which the test stops there: %s" % voided[:70])
        s = seen[0] if seen else {}
        check(s.get("under_tmp") and s.get("token_copied") and s.get("reads") is True and s.get("program_mode", 0o777) & 0o222 == 0,
              "inside the run the work folder held the runner's read-only copy of the submission with its token "
              "(the token recognised)")
        work = os.path.dirname(s["run_root"]) if s else None
        check(work is not None and not os.path.lexists(work) and os.listdir(tmpd) == [],
              "after the run no work folder and no copy of the submission stays (%s)" % (
                  "root" if root_run else "an account that is not root"))
        check(os.path.isfile(os.path.join(victim, "kept.txt")) and statmod.S_IMODE(os.lstat(victim).st_mode) == 0o555,
              "a link inside the work folder is not followed (its target and its mode are untouched)")
    finally:
        FLm.fly_session = old_fly
        tempfile.tempdir = old_tmp
        for k, val in old_env.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
        for dp, dn, fn in os.walk(base):
            for d in dn:
                p = os.path.join(dp, d)
                if not os.path.islink(p):
                    try:
                        os.chmod(p, 0o700)
                    except OSError:
                        pass
        try:
            os.chmod(base, 0o700)
        except OSError:
            pass
        shutil.rmtree(base, ignore_errors=True)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    for t in TESTS:
        if ARGS["only"] and t.__name__ not in ARGS["only"].split(","):
            continue
        CUR["name"] = t.__name__
        print(t.__name__)
        try:
            t()
        except Exception:  # noqa: BLE001
            RESULTS.append({"test": t.__name__, "ok": False, "msg": "raised: " + traceback.format_exc()[-1500:]})
            print("  FAIL  raised:\n" + traceback.format_exc())
    per = {}
    for r in RESULTS:
        p = per.setdefault(r["test"], [0, 0])
        p[0] += 1
        p[1] += 0 if r["ok"] else 1
    failed = sum(p[1] for p in per.values())
    summary = {"checks": len(RESULTS), "failed": failed, "tests": len(per),
               "per_test": {k: {"checks": v[0], "failed": v[1]} for k, v in sorted(per.items())},
               "bands_compared_with": list(COMPARED)}
    print("the shipped bands compared with: %s" % (COMPARED or "none"))
    print(json.dumps({"checks": summary["checks"], "failed": failed, "tests": summary["tests"]}))
    if ARGS["json"]:
        with open(ARGS["json"], "w") as f:
            json.dump(dict(summary, results=RESULTS), f, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
