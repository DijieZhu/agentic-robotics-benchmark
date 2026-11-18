"""No-leak test of the cell: every reply of a full mission is checked.

    MUJOCO_GL=osmesa python -I tests/review/cell/test_no_leak.py SITE [--reference] [--json OUT]

The replies are recorded at the cell's `handle` (exactly what the wire carries): those of a scripted program that
makes every public call (out of the link's coverage, a queued report, back into coverage, every sensor, the
refusals) and, with --reference, those of the whole privileged reference mission (every item visited and reported,
crossings, events). Checks, per reply of a public call:
  * strict key allow-lists per call (sensor frames, lidar, get_state, step, reports, finish), so a new key fails;
  * no key that names ground truth or a grader-side record anywhere (truth, states, temp_C, T_world_cam, base_true,
    pose_true, evidence records, cited, credit, score, slot, model_names, readable_region, observation_pose, ...);
  * no float in a dynamic reply equals a true value of the site's instruments (keys of poses and times excepted);
  * events carry only t, type and report_id, and never a tag;
  * no string anywhere names a decoy's tag, a model or geom name, an element id, the generator, the robot's origin,
    a physics setting or a noise process;
  * describe() is a datasheet (no generator bands, no capability file), mission() has only the public documents.
And the practice debrief carries no total and no as-built drawing.
Version 2: an instrument is known by its tag alone: no `label_code`, no `tag_codes`, `marker_id` only in
the drawing's grid marker list, no `36h11-NNN` label anywhere, the AprilTag dictionary named only for the grid markers;
no reply names the missing sheet's rooms (site.json `missing_sheet.rooms`) as drawn rooms, nor a variant, a candidate
number or the generator's table (`ladder.v2`); the hold after `blocked` adds `held`, `hold_until`, `detail` to `walk`.
"""

import argparse
import json
import math
import os
import re
import sys

TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TESTS)

from plantcell.client import CellError, connect_handler  # noqa: E402
from plantcell.server import PlantCellServer  # noqa: E402

ALLOWED = {
    "get_camera": {"frame_id", "sensor", "t", "width", "height", "K", "T_body_cam", "gain", "image_png_b64",
                   "depth_png_b64", "depth_unit_m", "depth_invalid"},
    "get_depth": {"frame_id", "sensor", "t", "width", "height", "K", "T_body_cam", "depth_png_b64", "depth_unit_m",
                  "depth_invalid"},
    "get_thermal": {"frame_id", "sensor", "t", "width", "height", "K", "T_body_cam", "thermal_ck_png_b64",
                    "thermal_unit_k"},
    "get_lidar": {"t", "beams", "ranges_f32_b64", "angle_step_rad", "T_body_lidar"},
    "get_state": {"sim_time", "done", "done_reason", "status", "odometry", "imu", "joint_pos", "foot_contact", "mode",
                  "body", "head", "lamp", "dosimeter", "link", "reports_pending", "frames_used", "reports_used",
                  # nested
                  "x", "y", "yaw", "distance", "roll", "pitch", "gyro", "height", "pitch_cmd", "pan", "tilt", "counts",
                  "dwell_s", "dose_gy", "dose_fraction", "up", "rssi_dbm", "nav", "inspect", "depth", "thermal",
                  "lidar", "instrument", "mark", "source"},
    "step": {"sim_time", "done", "events", "t", "type", "report_id"},
    "walk": {"ok", "held", "hold_until", "detail"}, "set_mode": {"ok"}, "set_body": {"ok"}, "set_lamp": {"ok"}, "set_head": {"ok", "pan", "tilt"},
    "report": {"ok", "sent", "report_id"}, "report_mark": {"ok", "sent", "report_id"},
    "report_source": {"ok", "sent", "report_id"}, "finish": {"ok", "sim_time"},
}
MISSION_TOP = {"frame", "datum", "drawing", "building", "grid", "rooms", "walls", "doors", "equipment", "grid_markers",
               "zoning", "link", "work_order", "round_sheet", "modification_log", "facility_facts", "catalog",
               "budgets"}
FORBIDDEN_KEYS = {"truth", "states", "temp_C", "angle_deg", "opening_pct", "level_pct", "dose_rate_uSv_h", "strength",
                  "peak_dose_rate_uSv_h", "hot_index", "hot_label", "T_world_cam", "base_true", "pose_true",
                  "credit", "score", "true_pose", "model_names", "readable_region", "observation_pose",
                  "discrepancy", "sources", "site_json", "true", "slot", "ladder", "acceptance_region",
                  "mod_log_truth", "cited", "evidence_record", "episode_score", "R", "frames_json", "geoms",
                  "face_geoms", "body", "accept", "proof", "proofs", "capability", "reference_p95", "schema",
                  "dose_units", "unit_factor_gy_per_site_uSv", "dose_budget_uSv", "dose_rating_site_uSv",
                  # version 2: plate codes, the missing sheet's and the variants' records of site.json
                  "label_code", "tag_codes", "missing_sheet", "on_missing_sheet", "b_step", "b_low", "sheet_place",
                  "b_step_variant", "b_low_variant", "approach_zones", "corridor_cuts"}
# 'body' is a key of get_state (body height): allowed there only
KEY_EXCEPT = {("get_state", "body")}
WORDS = re.compile(r"\b(solo ?12|solo|odri|open dynamic|menagerie|site_gen|bricks|mujoco|mjspec|solref|solimp|"
                   r"timestep|poisson|gaussian|noise|random|seed|uSv|site units?|generator|stand-?in|"
                   r"needs_high_step|needs_crouch|free_max|barrier_min|capability|benchmark|grader|harbor|"
                   r"door_sills|in_room|b_step|b_low|sheet_place|candidate)\b", re.I)
LABEL = re.compile(r"\b36h11-\d+\b")
# phrases of the gait's measured tables (describe()["modes"]["tables"], verbatim) that describe how a trial was
# run, not a process of the cell: removed before the word scan (the robot manual prints the same words)
TRIAL_PROTOCOL = re.compile(r"\brandom walking commands \(mostly forward, some backward, with sideways and turning parts "
                            r"and stops\), redrawn every 2-4 s\b")
POSE_KEYS = {"x", "y", "yaw", "t", "sim_time", "pan", "tilt", "roll", "pitch", "height", "pitch_cmd", "distance",
             "dose_gy", "dose_fraction", "dwell_s", "rssi_dbm", "gain",
             # constants of the encodings (the same in every reply)
             "thermal_unit_k", "depth_unit_m", "angle_step_rad"}


def walk(obj, path="", out=None):
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append((path, k, v))
            walk(v, path + "/" + str(k), out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            walk(v, path + "[%d]" % i, out)
    return out


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from strings(v)


class Recorder(object):
    def __init__(self, server):
        self.server = server
        self.inner = server.handle
        self.replies = []

    def __call__(self, method, params):
        res = self.inner(method, params)
        self.replies.append((method, json.loads(json.dumps(res))))
        return res


def site_secrets(world):
    site = world.site
    nums = set()
    for ip in site["ips"] + site.get("decoys", []):
        for _, k, v in walk(ip.get("truth") or {}):
            if isinstance(v, float) and not isinstance(v, bool) and k not in ("pos",) and abs(v) > 1e-6:
                nums.add(round(v, 9))
    decoys = {d["tag"] for d in site.get("decoys", []) if d.get("tag")}
    names = set()
    for ip in site["ips"] + site.get("decoys", []):
        mn = ip.get("model_names") or {}
        for g in (mn.get("geoms") or [])[:3]:
            names.add(g)
        if mn.get("body"):
            names.add(mn["body"])
    elems = {e["id"] for e in site.get("elements", [])}
    return nums, decoys, names, elems


def identity_tokens(server):
    """Strings and numbers that identify the site package or the mission's draws: the package's name and file
    stem, the generator's seed, the noise seed, the package's sha256 (and its first 12 hex digits)."""
    w = server.world
    toks = {str(server.site_name), os.path.basename(str(w.pkg.source)).split(".")[0], str(server.site_sha)[:12],
            str(server.site_sha)}
    for v in (w.site.get("seed"), (w.pkg.manifest or {}).get("seed"), server.seed):
        if v is not None:
            toks.add(str(v))
    nums = set()
    for v in (w.site.get("seed"), server.seed):
        try:
            nums.add(int(v))
        except (TypeError, ValueError):
            pass
    return {t for t in toks if t and len(t) >= 4}, nums


def check(replies, world, debrief=None, identity=None):
    problems = []
    nums, decoys, names, elems = site_secrets(world)
    id_strings, id_numbers = identity or (set(), set())
    for method, res in replies:
        for path, k, v in walk(res):
            if isinstance(v, int) and not isinstance(v, bool) and v in id_numbers and v > 1000:
                problems.append("%s: %s=%r is the package's or the mission's seed" % (method, k, v))
        for st in strings(res):
            if method in ("get_camera", "get_depth", "get_thermal", "get_lidar") and len(st) > 200:
                continue
            for t in id_strings:
                if t in st:
                    problems.append("%s names the package (%r)" % (method, t))
    listed = {ip["tag"] for ip in world.site["ips"]}
    seen = {}
    for method, res in replies:
        seen[method] = seen.get(method, 0) + 1
        if isinstance(res, dict) and set(res) <= {"error", "detail"}:
            for s in strings(res):
                for t in decoys | names:
                    if t and t in s:
                        problems.append("%s error names %s" % (method, t))
            continue
        allow = ALLOWED.get(method)
        for path, k, v in walk(res):
            if k in FORBIDDEN_KEYS and (method, k) not in KEY_EXCEPT:
                if method == "mission" and k in ("body",):
                    continue
                if method == "mission" and path == "/catalog/lettering/glyphs":
                    continue                       # the plant lettering's glyph table: its keys are the characters
                                                   # (the letter R is not the return factor)
                problems.append("%s: key %r at %s" % (method, k, path or "/"))
            if allow is not None and k not in allow:
                problems.append("%s: key %r not on the call's allow-list (at %s)" % (method, k, path or "/"))
        if method == "mission":
            extra = set(res) - MISSION_TOP
            if extra:
                problems.append("mission(): top-level keys beyond the public documents: %s" % sorted(extra))
            gone = set(str(x) for x in ((world.site.get("missing_sheet") or {}).get("rooms") or []))
            for r_ in res.get("rooms", []):
                if str(r_.get("number")) in gone or str(r_.get("id")) in gone:
                    problems.append("mission(): room %s of the missing sheet is drawn" % r_.get("number"))
        # version 2 (text plates): the plate codes
        for path, k, v in walk(res):
            if k == "marker_id" and not (method == "mission" and re.fullmatch(r"/grid_markers\[\d+\]", path)):
                problems.append("%s: marker_id outside the drawing's grid marker list (at %s)" % (method, path))
            if k == "dictionary" and not ((method == "describe" and path == "/tags") or
                                          (method == "mission" and path == "/catalog/tags/grid_markers")):
                problems.append("%s: a marker dictionary at %s" % (method, path))
        for st_ in strings(res):
            if method in ("get_camera", "get_depth", "get_thermal", "get_lidar") and len(st_) > 200:
                continue
            if LABEL.search(st_):
                problems.append("%s: a plate code %r" % (method, LABEL.search(st_).group(0)))
        if method not in ("describe", "mission"):
            for path, k, v in walk(res):
                if isinstance(v, float) and k not in POSE_KEYS and not path.startswith(("/K", "/T_body")):
                    if round(v, 9) in nums:
                        problems.append("%s: %s=%r equals a true value (at %s)" % (method, k, v, path))
        if method == "step":
            for e in res.get("events", []):
                if set(e) - {"t", "type", "report_id"}:
                    problems.append("event with extra keys %s" % e)
                s = json.dumps(e)
                for t in listed | decoys:
                    if t in s:
                        problems.append("event names %s" % t)
        for s in strings(res):
            if method in ("get_camera", "get_depth", "get_thermal", "get_lidar") and len(s) > 200:
                continue                                 # base64 payloads
            for t in decoys:
                if re.search(r"\b%s\b" % re.escape(t), s):
                    problems.append("%s names the unlisted instrument %s" % (method, t))
            for t in names:
                if t in s:
                    problems.append("%s names the model's %s" % (method, t))
            if method != "mission":
                for e in elems:
                    if re.fullmatch(r"E\d{3}", e) and re.search(r"\b%s\b" % e, s):
                        problems.append("%s names the element %s" % (method, e))
            for m in WORDS.finditer(TRIAL_PROTOCOL.sub("", s)):
                problems.append("%s: word %r in %r" % (method, m.group(0), s[:120]))
    if debrief is not None:
        d = json.dumps(debrief).lower()
        for k in ('"score', '"total', '"rooms"', '"walls"', '"doors"', '"reward', '"ratio'):
            if k in d:
                problems.append("debrief contains %s" % k)
    return problems, seen


def scripted(r):
    """Every public call, out of coverage and back, the refusals."""
    d = r.describe()
    m = r.mission()
    items = m["work_order"]["items"]
    inst = [it for it in items if it["type"] not in ("dose_point", "hotspot")]
    marks = [it for it in items if it["type"] == "dose_point"]
    r.get_state()
    r.set_mode("walk")
    for _ in range(40):
        st = r.get_state()
        o = st["odometry"]
        r.walk(0.35, -0.6 * o["y"], -1.5 * o["yaw"])
        r.step(50)
        if o["x"] > 4.5:
            break
    r.walk(0.0, 0.0, 0.0)
    r.step(100)
    r.report(inst[-1]["tag"], 1.0, [])
    fr = r.get_camera("inspect")
    r.report(inst[0]["tag"], None, [fr["frame_id"]])
    for _ in range(80):
        st = r.get_state()
        o = st["odometry"]
        e = math.atan2(math.sin(math.pi - o["yaw"]), math.cos(math.pi - o["yaw"]))
        r.walk(0.3 if abs(e) < 0.3 else 0.0, 0.0, 1.5 * e)
        r.step(50)
        if o["x"] < 0.6:
            break
    r.walk(0.0, 0.0, 0.0)
    r.step(100)
    r.set_mode("stand")
    r.set_body(height=0.22, pitch=0.05)
    r.set_head(pan=0.3, tilt=-0.2)
    r.set_lamp(True)
    r.step(80)
    r.get_camera("nav", depth=True)
    r.get_camera("nav", depth=False)
    r.get_depth()
    r.get_thermal()
    r.get_lidar()
    if marks:
        r.report_mark(marks[0]["tag"], 1.0)
    r.report_source(1.0, 2.0)
    for bad in (lambda: r.get_camera("mast"), lambda: r.set_mode("run"), lambda: r.step(0),
                lambda: r.report("NO-SUCH-TAG", 1.0, ["F1", "F2", "F3", "F4"])):
        try:
            bad()
        except CellError:
            pass
    for _ in range(4):
        r.step(100)
    r.get_state()
    r.finish()
    return d


def run(site, reference=False):
    srv = PlantCellServer(mode="hidden")
    srv.reset(site)
    rec = Recorder(srv)
    scripted(connect_handler(rec))
    problems, seen = check(rec.replies, srv.world, srv.debrief(true_pose_hz=2.0), identity_tokens(srv))
    out = {"scripted": {"replies": len(rec.replies), "calls": seen, "problems": problems}}
    srv.close()
    if reference:
        tests = TESTS
        srv = PlantCellServer(mode="hidden", privileged=True, reference_dir=os.path.join(tests, "reference"))
        srv.reset(site)
        rec = Recorder(srv)
        srv.handle = rec                      # the reference's proxy picks up the recorder
        rec.inner = PlantCellServer.handle.__get__(srv)
        res = srv.handle("reference_run", {})
        replies = [(m, r) for m, r in rec.replies if m not in ("reference_run", "get_state_privileged")]
        problems2, seen2 = check(replies, srv.world, srv.debrief(), identity_tokens(srv))
        summ = srv.finalize()
        out["reference"] = {"replies": len(replies), "calls": seen2, "problems": problems2,
                            "episode_score": summ["episode_score"], "result": res}
        srv.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    out = run(a.site, a.reference)
    n = 0
    for part, d in out.items():
        print("%s: %d replies, calls %s" % (part, d["replies"], d["calls"]))
        for p in d["problems"][:60]:
            print("PROBLEM [%s] %s" % (part, p))
        n += len(d["problems"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
    print("no-leak: %s (%d problems)" % ("OK" if n == 0 else "FAIL", n))
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
