"""QC Q4: the station's interface on the FROZEN hidden sites does not leak (task version 2.0).

    q4_interface.py OUT_DIR [--sites h01,...,h10] [--detail-sites h01,h04] [--plate-root DIR]

With the plate backend (grader_config.json render_backend plate_filament) the servers that render need the installed
bundles of the hidden sites: --plate-root is the absolute path of such a plate root (verify.py installs one only
inside its own stage; grader_config.json's plate_root is a path under the stage).

In-process cells (tests/voidcell/server.py VoidCellServer, mode hidden, the task's grader_config.json and the hidden
limits.json; the station through voidcell.client.connect_handler, i.e. the wire's JSON round trip) on frozen hidden
site packages. Checks, each a row of OUT_DIR/q4_interface.json:
  B  bad arguments for every public call (unknown, missing, wrong JSON type, non-finite, huge, wrong length, params
     not an object): the reply is {"error": "bad_argument", "detail"} and nothing changes (state hash, clock, budgets)
  W  every reply of every call made here (valid, refused, bad) has keys inside protocol.REPLY_KEYS / NESTED_KEYS and
     an error code inside protocol.ERROR_CODES; the whitelist itself has no layout word
  M  mission() on every site of --sites (default: the ten hidden sites h01-h10; --detail-sites does not change it):
     exactly the whitelisted keys, no layout key at any depth, no number of the site's hidden layout (hidden_numbers:
     boxes with the rubble under the slab, openings, slab, the void's runs, pockets and chamber, plug, patches, chips,
     frictions, the heaps of the open floor (rectangle, centre, top, line, grid, placement); it does not walk
     site["camera"], and its 3rd-decimal filter drops integers such as seeds) among its numbers; the given pose
     differs from the true one in translation and in rotation; the announced constants are the
     acceptance file's (weights 0.05 / 0.15 / 0.80, credit radii 0.20 / 0.40 m, dwell 20 s, still 0.02 m, 3 reports
     per point) and the hidden time limit; the site rectangle is the family's [4.4, 5.8] m
     Given by design in the row "no number of the hidden layout": the pad and search-point coordinates, by value
     wherever they stand in mission() (the rule of version 1.1), and the twelve numbers of the upper 3 x 4 of
     mission()["camera"]["T_world_camera"], by location, these only when the given pose lies within the generator's
     bounds of the true pose (translation error <= 5 mm + 2e-6 m, rotation angle <= 0.2 + 0.005 deg; outside them the
     row fails). A hidden number at any other place of mission(), the rest of the camera block included, fails, also
     when the same value stands among the twelve, unless it equals a pad or search-point coordinate. Every hidden
     number equal to a given number is listed, whether it also fails or not.
  R  the station's command refusals (invalid_name, invalid_bottom, busy, out_of_range, budget_exhausted for
     frames and the fourth report, dt off the grid) change nothing; state calls after the end answer episode_over
  P  privileged and practice calls without the token: reference_run -> forbidden, reset -> forbidden, unknown method
     -> forbidden, and nothing runs; the oracle token is only a file of the submission (no call takes it)
  O  pose(): pose_source "estimated" -> not_available and the pose source never updates; "cell" -> whitelisted reply
     with 6 nodes within 45 mm of the truth when >= 4 caps are in sight, null under the slab; the switch does not
     change the physics (same commands, same state hash)
  D  a hidden episode builds no debrief: voidscore.debrief.build refuses kind "hidden"; verify.py never imports the
     debrief module; the server class has no debrief method
Every reply (frames without their pixel blobs) is appended to OUT_DIR/replies.jsonl for the leak scan (Q5); no reply
holds the hidden box names (rubble__void_debris, plug__, support__void_fill) or the word void_debris.
The refusal, privileged, bad-argument checks run on the first of --detail-sites, the pose checks on the second.
Tool version 2.1 on task 2.0: the given pose is exempt by location within the calibration bounds.
"""
import argparse
import ast
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.normpath(os.path.join(HERE, "..", ".."))
TASK = os.path.join(os.environ["ARB_SRC_ROOT"], os.path.basename(DEV)[:-len("-dev")]) if os.environ.get("ARB_SRC_ROOT") else DEV[:-len("-dev")]
TESTS = os.path.join(TASK, "tests")
sys.path.insert(0, TESTS)

import numpy as np  # noqa: E402

from voidcell.server import VoidCellServer  # noqa: E402
from voidcell import client as vclient  # noqa: E402
from voidcell._lib import protocol as proto  # noqa: E402

LAYOUT_WORDS = ("box", "boxes", "opening", "openings", "slab", "void", "gallery", "plug", "mouth", "r1", "r2", "patch",
                "patches", "chip", "chips", "friction", "true", "seed", "cls", "class", "decoy", "blind", "tunnel",
                "gap", "slot", "rubble", "crate", "support", "roof", "wall", "envelope", "lighting", "light", "debris",
                "chamber", "pocket", "pockets", "runs", "corridor", "cavity", "dead_end", "partition", "fill",
                "heap", "heaps")
PLATE_ROOT = None
HIDDEN_NAMES = ("rubble__void_debris", "void_debris", "plug__", "support__void_fill")
# the site generator's calibration error (sitegen draw_camera: |dt| <= 5 mm, angle <= 0.2 deg) + the 6-decimal rounding
POSE_T_BOUND_M = 0.005 + 2e-6
POSE_R_BOUND_DEG = 0.2 + 0.005


class Rec(object):
    def __init__(self, out_dir):
        self.rows = []
        self.fh = open(os.path.join(out_dir, "replies.jsonl"), "w")
        self.n_replies = 0
        self.bad_replies = []

    def reply(self, site, method, params, res):
        why = proto.check_reply(method, res) if method in proto.PUBLIC_METHODS else None
        if isinstance(res, dict) and "error" in res and res.get("error") is not None and \
                res["error"] not in proto.ERROR_CODES:
            why = "code %r" % res["error"]
        if why:
            self.bad_replies.append([site, method, params, why])
        slim = {k: v for k, v in res.items() if not k.endswith("_b64")} if isinstance(res, dict) else res
        self.fh.write(json.dumps({"site": site, "method": method, "params": params, "reply": slim}) + "\n")
        self.n_replies += 1

    def check(self, cid, ok, **kw):
        kw.update(id=cid, ok=bool(ok))
        self.rows.append(kw)
        return ok


def load_acceptance():
    from voidscore import scoring
    return scoring.load_acceptance()


def load_cfg():
    cfg = json.load(open(os.path.join(TESTS, "grader_config.json")))
    lim = json.load(open(os.path.join(TESTS, "hidden", "limits.json")))
    c = {k: cfg[k] for k in ("limits", "pose_source", "render_backend", "msaa_rgb") if k in cfg}
    c["limits"] = dict(c.get("limits") or {}, time_limit_s=float(lim["time_limit_s"]))
    if PLATE_ROOT:
        c["plate_root"] = PLATE_ROOT
    return c


def seeds():
    doc = json.load(open(os.path.join(TESTS, "hidden", "seeds.json")))
    return {r["tag"]: (int(r["seed"]), os.path.join(TESTS, "hidden", r["archive"])) for r in doc["sites"]}


def server(tag, cfg=None, render=True, **over):
    c = dict(cfg or load_cfg())
    for k, v in over.items():
        if k == "limits":
            c["limits"] = dict(c["limits"], **v)
        else:
            c[k] = v
    seed, pkg = seeds()[tag]
    srv = VoidCellServer(config=c, mode="hidden", privileged=False, render=render)
    srv.reset(pkg, seed=seed, kind="hidden")
    return srv


def raw(srv, R, tag, method, params):
    """One request through the wire's JSON encoding, the raw reply (error replies included)."""
    msg = json.loads(json.dumps({"method": method, "params": params}))
    res = srv.handle(msg["method"], msg["params"])
    res = json.loads(json.dumps(res))
    R.reply(tag, method, params, res)
    return res


def state(srv):
    b = srv.budgets.record()
    # the physics, the clock, the budgets, the executor's command history and its busy flag (the cell's call record
    # `commands` lists refused calls too: a record of the call, not a change of the episode)
    return (srv.S.state_hash(), round(srv.X.t(), 9), json.dumps(b, sort_keys=True), len(srv.X.history),
            bool(srv.X.busy), sum(1 for c in srv.commands if c.get("accepted")))


BAD = {
    "mission": [{"x": 1}],
    "get_frame": [{"x": 1}],
    "telemetry": [{"since": 0}],
    "status": [{"verbose": True}],
    "stop": [{"now": 1}],
    "finish": [{"x": None}],
    "pose": [{"x": 1}],
    "run_primitive": [{}, {"name": "ccw"}, {"bottom_nodes": [0, 2, 5]}, {"name": 5, "bottom_nodes": [0, 2, 5]},
                      {"name": "ccw", "bottom_nodes": "025"}, {"name": "ccw", "bottom_nodes": [0, 2]},
                      {"name": "ccw", "bottom_nodes": [0, 2, 5, 1]}, {"name": "ccw", "bottom_nodes": [0, "2", 5]},
                      {"name": "ccw", "bottom_nodes": [0, 2, None]}, {"name": "ccw", "bottom_nodes": [0, 2, True]},
                      {"name": "ccw", "bottom_nodes": [0, 2, 1e300]}, {"name": "x" * 65, "bottom_nodes": [0, 2, 5]},
                      {"name": "ccw", "bottom_nodes": [0, 2, 5], "speed": 2}],
    "set_targets": [{}, {"lengths_mm": 200}, {"lengths_mm": [200] * 5}, {"lengths_mm": [200] * 7},
                    {"lengths_mm": [200, 200, 200, 200, 200, "200"]}, {"lengths_mm": [200, 200, 200, 200, 200, None]},
                    {"lengths_mm": [200, 200, 200, 200, 200, 1e9]}, {"lengths_mm": [200] * 6, "x": 1}],
    "report_point": [{}, {"i": 0}, {"i": 4}, {"i": 1.0}, {"i": "1"}, {"i": True}, {"i": None}, {"i": 1, "x": 2}],
    "step": [{}, {"dt": "1"}, {"dt": None}, {"dt": True}, {"dt": 1e300}, {"dt": 0.5, "x": 1}],
}
NONFINITE = [("step", {"dt": float("nan")}), ("step", {"dt": float("inf")}),
             ("set_targets", {"lengths_mm": [200, 200, 200, 200, 200, float("nan")]})]


def check_bad_arguments(R, tag):
    srv = server(tag, render=True)
    raw(srv, R, tag, "step", {"dt": 1.0})
    for method, cases in BAD.items():
        for p in cases:
            s0 = state(srv)
            res = raw(srv, R, tag, method, p)
            ok = isinstance(res, dict) and res.get("error") == "bad_argument" and set(res) <= {"error", "detail"} \
                and state(srv) == s0
            R.check("B", ok, site=tag, method=method, params=p, reply=res)
    for method, p in NONFINITE:          # what a JSON client cannot send but a hand-written line can (NaN / Infinity)
        s0 = state(srv)
        res = srv.handle(method, p)
        R.reply(tag, method, repr(p), res)
        ok = isinstance(res, dict) and res.get("error") == "bad_argument" and state(srv) == s0
        R.check("B", ok, site=tag, method=method, params=repr(p), reply=res)
    for p in ([1, 2], "x", 3):
        s0 = state(srv)
        res = srv.handle("step", p)
        R.reply(tag, "step", repr(p), res)
        R.check("B", res.get("error") == "bad_argument" and state(srv) == s0, site=tag, method="step",
                params=repr(p), reply=res, note="params not an object")
    srv.close()


def hidden_numbers(site):
    """The numbers of the site's hidden layout (rounded to 4 decimals, each with its path) under the keys params,
    bands, both rows' openings, the void, patches, boxes, lighting, physics, chips, free blocks and the heaps of the
    open floor (every heap's rect, cx, top_m, y_line, grid and the placement's numbers). Only numbers with
    abs(v) >= 0.05 and round(v, 3) != round(v, 2) are kept: integers such as seeds, numbers with two decimals or fewer
    and numbers within 0.0005 of a two-decimal number (e.g. 0.4501) are dropped. site["camera"] is not walked: the true
    extrinsic is not among these numbers (check_mission compares it with the given pose instead); check_mission says
    which numbers of mission() are given by design; the family constants (envelope, classes), the draw statistics and
    the generator's bookkeeping are not layout."""
    out = set()

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + (str(k),))
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, path)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            v = float(o)
            if math.isfinite(v) and abs(v) >= 0.05 and round(v, 3) != round(v, 2):
                out.add((round(v, 4), "/".join(path)))
    for k in ("params", "bands", "r1", "r2", "void", "patches", "boxes", "lighting", "physics", "chips",
              "free_blocks", "heaps"):
        if k in site:
            walk(site[k], (k,))
    return out


def numbers_of(o):
    out = []
    if isinstance(o, dict):
        for v in o.values():
            out += numbers_of(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            out += numbers_of(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        out.append(float(o))
    return out


def keys_of(o, path=()):
    out = []
    if isinstance(o, dict):
        for k, v in o.items():
            out.append(path + (k,))
            out += keys_of(v, path + (k,))
    elif isinstance(o, list):
        for v in o:
            out += keys_of(v, path)
    return out


def pose_error(T_true, T_given):
    """(|t_given - t_true| in m, the rotation angle of R_given R_true^T in degrees) of two 4 x 4 poses; (inf, inf) when
    either is not a finite 4 x 4. The angle is 2 asin(|R_given - R_true|_F / (2 sqrt 2)): for two rotations that norm is
    |R_given R_true^T - I|_F = 2 sqrt(2) sin(angle / 2), so it is accurate near zero (no arccos of a number near 1), it
    is zero for equal matrices (neither is exactly orthonormal after the 6-decimal rounding), and a change of any entry
    counts (a matrix that is not a rotation is not read as a small rotation)."""
    try:
        A, B = np.asarray(T_given, dtype=float), np.asarray(T_true, dtype=float)
    except (TypeError, ValueError):
        return math.inf, math.inf
    if A.shape != (4, 4) or B.shape != (4, 4) or not (np.isfinite(A).all() and np.isfinite(B).all()):
        return math.inf, math.inf
    t = float(np.linalg.norm(A[:3, 3] - B[:3, 3]))
    s = float(np.linalg.norm(A[:3, :3] - B[:3, :3])) / (2.0 * math.sqrt(2.0))
    return t, math.degrees(2.0 * math.asin(min(1.0, s)))


def check_mission(R, tag):
    srv = server(tag, render=False)
    m = raw(srv, R, tag, "mission", {})
    R.check("M", set(m) == set(proto.REPLY_KEYS["mission"]), site=tag, what="top-level keys", keys=sorted(m))
    words = [k for k in keys_of(m) if any(w == str(x).lower() or str(x).lower().startswith(w + "_")
                                           for x in k for w in LAYOUT_WORDS)]
    R.check("M", not words, site=tag, what="no layout key at any depth", hits=words)
    hid = hidden_numbers(srv.site)
    hv = {}
    for v, p in hid:
        hv.setdefault(v, []).append(p)
    Tt = srv.site["camera"]["true"]["T_world_camera"]
    Tg = m["camera"]["T_world_camera"]
    t_err, r_err = pose_error(Tt, Tg)
    in_bounds = t_err <= POSE_T_BOUND_M and r_err <= POSE_R_BOUND_DEG
    shown = sorted(set(round(v, 4) for v in numbers_of(m)))
    given = set(round(v, 4) for v in numbers_of({"pad": m["pad"]["xy"], "sp": [p["xy"] for p in m["search_points"]]}))
    rest = json.loads(json.dumps(m))      # mission() without the exempt places (the pose's bottom row stays)
    pose12 = set()
    if in_bounds:
        pose12 = set(round(v, 4) for v in numbers_of(Tg[:3]))
        rest["camera"]["T_world_camera"] = rest["camera"]["T_world_camera"][3:]
    others = sorted(set(round(v, 4) for v in numbers_of(rest)))
    lay = {v: hv[v] for v in others if v in hv and v not in given}
    coinc = {v: hv[v] for v in shown if v in hv and (v in given or v in pose12)}
    R.check("M", not lay and in_bounds, site=tag,
            what="no number of the hidden layout (>= 3 significant decimals) in mission()",
            n_numbers=len(shown), layout_hits={str(k): v[:4] for k, v in lay.items()},
            given_coordinates_equal_to_a_hidden_number={str(k): v[:4] for k, v in coinc.items()},
            pose_within_calibration_bounds=bool(in_bounds),
            **({} if in_bounds else {"why": "the given pose is outside the generator's bounds of the true pose: its "
                                            "twelve numbers are not exempt"}),
            note="pad / search-point coordinates (%d numbers, by value wherever they stand in mission()) and, within "
                 "the calibration bounds, the twelve numbers of the given pose's upper 3 x 4 (by location) are given "
                 "by design; a hidden number equal to one of them (point 3's clearance = its distance to a site wall; "
                 "a chance equality at the 4th decimal with the calibrated pose) is listed; it fails when the same "
                 "value also stands at a place of mission() that is not exempt and it is not a pad or search-point "
                 "coordinate" % len(given))
    acc = load_acceptance()
    lim = json.load(open(os.path.join(TESTS, "hidden", "limits.json")))
    want_w = {int(k): float(v) for k, v in acc["weights"].items()}
    got_w = {int(p["id"]): float(p["weight"]) for p in m["search_points"]}
    R.check("M", m["site"]["size_m"] == [4.4, 5.8], site=tag, what="the site rectangle is the family's [4.4, 5.8] m",
            size_m=m["site"]["size_m"])
    R.check("M", got_w == want_w and [float(v) for v in m["credit_radius_m"]] ==
            [float(acc["credit_radius_m"]["full"]), float(acc["credit_radius_m"]["zero"])] and
            float(m["dwell_s"]) == float(acc["dwell"]["window_s"]) and
            float(m["still_tolerance_m"]) == float(acc["dwell"]["still_tolerance_m"]) and
            int(m["limits"]["reports_per_point"]) == int(acc["reports_per_point"]) and
            float(m["limits"]["time_limit_s"]) == float(lim["time_limit_s"]) and
            float(m["pad"]["retrieval_radius_m"]) == float(acc["retrieval"]["radius_m"]), site=tag,
            what="mission() announces the acceptance file's constants and the hidden time limit",
            weights=got_w, credit_radius_m=m["credit_radius_m"], dwell_s=m["dwell_s"],
            time_limit_s=m["limits"]["time_limit_s"], reports_per_point=m["limits"]["reports_per_point"])
    dt = max(abs(float(Tt[i][3]) - float(Tg[i][3])) for i in range(3))
    dR = max(abs(float(Tt[i][j]) - float(Tg[i][j])) for i in range(3) for j in range(3))
    R.check("M", t_err > 1e-6 and r_err > 1e-4, site=tag,
            what="mission() gives the calibrated extrinsic, not the true one",
            max_translation_diff_m=round(dt, 6), max_rotation_entry_diff=round(dR, 6),
            translation_error_m=round(t_err, 6), rotation_angle_deg=round(r_err, 6))
    srv.close()
    return m


def check_refusals(R, tag):
    srv = server(tag, render=True, limits={"frames": 3})
    st = vclient.connect_handler(srv.handle, decode=False)
    true_b = list(srv.robot_truth.bottom()["nodes"])
    raw(srv, R, tag, "step", {"dt": 0.5})
    cases = []
    s0 = state(srv)
    cases.append(("invalid_name", raw(srv, R, tag, "run_primitive", {"name": "roll_backward", "bottom_nodes": true_b}),
                  s0))
    s0 = state(srv)
    cases.append(("invalid_bottom", raw(srv, R, tag, "run_primitive", {"name": "ccw", "bottom_nodes": [0, 1, 2]}), s0))
    s0 = state(srv)
    cases.append(("out_of_range", raw(srv, R, tag, "set_targets", {"lengths_mm": [99.9, 200, 200, 200, 200, 200]}),
                  s0))
    s0 = state(srv)
    cases.append(("out_of_range", raw(srv, R, tag, "set_targets", {"lengths_mm": [200, 200, 200, 200, 200, 240.1]}),
                  s0))
    for code, res, before in cases:
        R.check("R", res.get("accepted") is False and res.get("error") == code and state(srv) == before, site=tag,
                expect=code, reply=res)
    r = raw(srv, R, tag, "run_primitive", {"name": "ccw", "bottom_nodes": true_b})
    R.check("R", r == {"accepted": True, "error": None}, site=tag, expect="accepted", reply=r)
    s0 = state(srv)[0]
    r2 = raw(srv, R, tag, "run_primitive", {"name": "cw", "bottom_nodes": true_b})
    r3 = raw(srv, R, tag, "set_targets", {"lengths_mm": [200] * 6})
    R.check("R", r2.get("error") == "busy" and r3.get("error") == "busy" and len(srv.X.history) == 0, site=tag,
            expect="busy", reply=[r2, r3])
    for dt in (0.005, 0.015, 10.01, 0.0, -1.0, 0.333):
        s1 = state(srv)
        res = raw(srv, R, tag, "step", {"dt": dt})
        R.check("R", res.get("error") == "bad_argument" and state(srv) == s1, site=tag, expect="dt off the grid",
                dt=dt, reply=res)
    while raw(srv, R, tag, "status", {})["busy"]:
        raw(srv, R, tag, "step", {"dt": 0.5})
    frames = [raw(srv, R, tag, "get_frame", {}) for _ in range(4)]
    R.check("R", all("index" in f for f in frames[:3]) and frames[3].get("error") == "budget_exhausted"
            and srv.budgets.frames_used == 3, site=tag, expect="frame budget (3 in this check)",
            replies=[{k: v for k, v in f.items() if not k.endswith("_b64")} for f in frames])
    reps = [raw(srv, R, tag, "report_point", {"i": 2}) for _ in range(4)]
    R.check("R", [r.get("accepted") for r in reps] == [True, True, True, False] and reps[3] == {
        "accepted": False, "reports_left": 0, "error": "budget_exhausted"} and srv.budgets.reports[2] == 3
        and sum(1 for x in srv.reports if not x["accepted"]) == 1, site=tag, expect="4th report refused", replies=reps)
    tl = raw(srv, R, tag, "telemetry", {})
    R.check("W", proto.check_reply("telemetry", tl) is None and tl["n"] > 0, site=tag, what="telemetry reply")
    raw(srv, R, tag, "finish", {})
    after = {}
    for method, p in (("step", {"dt": 1.0}), ("run_primitive", {"name": "ccw", "bottom_nodes": true_b}),
                      ("set_targets", {"lengths_mm": [200] * 6}), ("stop", {}), ("report_point", {"i": 1})):
        after[method] = raw(srv, R, tag, method, p)
    R.check("R", all(v.get("error") == "episode_over" for v in after.values()), site=tag,
            expect="state calls after finish -> episode_over", replies=after)
    info = {m: raw(srv, R, tag, m, {}) for m in ("mission", "status", "telemetry")}
    R.check("W", all("error" not in v for v in info.values()), site=tag, what="information calls after the end answer")
    srv.close()
    del st


def check_privileged(R, tag):
    srv = server(tag, render=False)
    s0 = state(srv)
    res = {m: raw(srv, R, tag, m, p) for m, p in (("reference_run", {}), ("reset", {"site": "h01"}),
                                                   ("debrief", {}), ("pose_truth", {}), ("__init__", {}),
                                                   ("_m_step", {"dt": 1.0}), ("finalize", {}))}
    ok = all(v == {"error": "forbidden"} for v in res.values()) and state(srv) == s0 and srv.reference_result is None
    R.check("P", ok, site=tag, what="privileged / practice / unknown / internal method names without the token",
            replies=res, illegal_calls=srv.illegal_calls)
    st = vclient.connect_handler(srv.handle)
    try:
        st._call("reference_run")
        raised = None
    except vclient.StationError as e:
        raised = e.code
    R.check("P", raised == "forbidden", site=tag, what="client _call('reference_run') without the token raises",
            code=raised)
    sig = [n for n in dir(srv) if "token" in n.lower()]
    R.check("P", not any(proto.ARGS.get(m) and any("token" in a for a in proto.ARGS[m]) for m in proto.ARGS),
            site=tag, what="no call takes a token argument (the token is a file of the submission, read by verify)",
            server_attrs_with_token=sig)
    srv.close()


def check_pose(R, tag):
    est = server(tag, render=False)
    cel = server(tag, render=False, pose_source="cell")
    for srv, name in ((est, "estimated"), (cel, "cell")):
        m = raw(srv, R, tag, "mission", {})
        R.check("O", m["pose_source"] == name, site=tag, what="mission pose_source", value=m["pose_source"])
    r = raw(est, R, tag, "pose", {})
    R.check("O", r.get("error") == "not_available", site=tag, what="estimated: pose() refused", reply=r)
    b = list(est.robot_truth.bottom()["nodes"])
    for srv in (est, cel):
        raw(srv, R, tag, "step", {"dt": 1.0})
        raw(srv, R, tag, "run_primitive", {"name": "roll_120_120", "bottom_nodes": b})
        while raw(srv, R, tag, "status", {})["busy"]:
            raw(srv, R, tag, "step", {"dt": 0.5})
    p = raw(cel, R, tag, "pose", {})
    ok = proto.check_reply("pose", p) is None
    if p.get("nodes_xyz") is not None:
        err = float(np.abs(np.asarray(p["nodes_xyz"]) - cel.S.nodes()).max())
    else:
        err = None
    R.check("O", ok and p.get("nodes_xyz") is not None and len(p["visible_caps"]) >= 4 and err < 0.045, site=tag,
            what="cell: pose() in sight", max_err_m=err, visible_caps=p.get("visible_caps"), index=p.get("index"))
    R.check("O", est.pose.n_updates == 0 and cel.pose.n_updates > 0 and est.budgets.frames_used == 0 and
            cel.budgets.frames_used == 0, site=tag, what="the pose source updates only under 'cell', uses no frame",
            updates=[est.pose.n_updates, cel.pose.n_updates])
    R.check("O", est.S.state_hash() == cel.S.state_hash(), site=tag, what="the switch does not change the physics",
            hashes=[est.S.state_hash(), cel.S.state_hash()])
    q = cel.S.d.qpos.copy()
    c = cel.S.nodes()[:, :2].mean(axis=0)
    sp3 = [s for s in cel.site["search_points"] if int(s["id"]) == 3][0]["xy"]
    for k in range(3):
        a = cel.S.ids.jnt_qpos[k]
        q[a:a + 2] += np.asarray(sp3) - c
    _n, vis = cel.pose.visibility(q)
    cel.pose.on_exposure(10 ** 6, 1.0e4, q)
    rp = cel.pose.reply()
    R.check("O", sum(vis) == 0 and rp["nodes_xyz"] is None, site=tag,
            what="cell: robot moved (in a copy of the state) onto search point 3 under the slab -> no caps, null",
            visible=int(sum(vis)))
    est.close()
    cel.close()


def check_debrief(R):
    sys.path.insert(0, TESTS)
    from voidscore import debrief
    refused = []

    class _Hidden(object):
        mode = "hidden"

    class _Practice(object):
        mode = "practice"
    for srv_like, kind in ((_Practice(), "hidden"), (_Hidden(), "practice"), (_Hidden(), "cal")):
        try:
            debrief.build(srv_like, None, kind, None)
            refused.append([kind, srv_like.mode, False, None])
        except debrief.DebriefError as e:
            refused.append([kind, srv_like.mode, True, str(e)[:120]])
        except Exception as e:  # noqa: BLE001  (refused for another reason: it must be the DebriefError)
            refused.append([kind, srv_like.mode, False, repr(e)[:120]])
    refused_ok = [r[2] for r in refused]
    R.check("D", all(refused_ok), what="voidscore.debrief.build refuses kind hidden and a server in mode hidden", cases=refused)
    tree = ast.parse(open(os.path.join(TESTS, "verify.py")).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    deb = sorted(n for n in names if "debrief" in n.lower())
    R.check("D", not deb, what="verify.py has no name or import with 'debrief'", hits=deb)
    R.check("D", not any("debrief" in n.lower() for n in dir(VoidCellServer)), what="the server has no debrief method")
    wl = json.dumps([proto.REPLY_KEYS, {"%s.%s" % k: v for k, v in proto.NESTED_KEYS.items()}]).lower()
    hits = [w for w in LAYOUT_WORDS if '"%s"' % w in wl or '"%s_' % w in wl]
    R.check("W", not hits, what="the reply whitelist itself has no layout word", hits=hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sites", default=",".join("h%02d" % k for k in range(1, 11)))
    ap.add_argument("--detail-sites", default="h01,h04")
    ap.add_argument("--plate-root", default=None)
    a = ap.parse_args()
    global PLATE_ROOT
    if a.plate_root:
        if not os.path.isabs(a.plate_root) or not os.path.isdir(a.plate_root):
            raise SystemExit("q4: --plate-root must be an absolute path of an existing folder")
        PLATE_ROOT = a.plate_root
    elif load_cfg().get("render_backend") == "plate_filament":
        raise SystemExit("q4: the plate backend needs --plate-root (the installed bundles of the hidden sites)")
    os.makedirs(a.out, exist_ok=True)
    R = Rec(a.out)
    tags = a.sites.split(",")
    det = a.detail_sites.split(",")
    for t in tags:
        check_mission(R, t)
    check_bad_arguments(R, det[0])
    check_refusals(R, det[0])
    check_privileged(R, det[0])
    check_pose(R, det[min(1, len(det) - 1)])
    check_debrief(R)
    R.check("W", not R.bad_replies, what="every reply inside the whitelist and the code enumeration",
            n_replies=R.n_replies, bad=R.bad_replies[:20])
    R.fh.close()
    txt = open(os.path.join(a.out, "replies.jsonl"), encoding="utf-8").read()
    R.check("W", not any(h in txt for h in HIDDEN_NAMES), what="no reply holds a hidden box name or void_debris",
            hits=[h for h in HIDDEN_NAMES if h in txt])
    n_ok = sum(1 for r in R.rows if r["ok"])
    out = {"package": "QC Q4 (version 2.1)", "sites": tags, "detail_sites": det, "n_checks": len(R.rows), "n_ok": n_ok, "pass": n_ok == len(R.rows),
           "n_replies": R.n_replies, "rows": R.rows,
           "replies_sha256": hashlib.sha256(open(os.path.join(a.out, "replies.jsonl"), "rb").read()).hexdigest()}
    with open(os.path.join(a.out, "q4_interface.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("n_checks", "n_ok", "pass", "n_replies")}))
    for r in R.rows:
        if not r["ok"]:
            print("FAIL", json.dumps(r, default=str)[:600])
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
