"""Maintainers' generator of the fixed hidden set (`tests/hidden/fixed_set.json`): pxr and numpy only, in the task
image's /opt/venv/bin/python3 on the CPU (no Kit, no simulation). Nothing here is part of the grade; the grader never
runs it.

    python3 tests/review/make_fixed_set.py draw --n N --out FILE
    python3 tests/review/make_fixed_set.py corners --out FILE
    python3 tests/review/make_fixed_set.py select --candidates FILE --results DIR --k 2 --frozen FILE --out PATH

draw: a batch of N candidates, all drawn before any of them runs, each from OS entropy (`secrets`): an id, a noise
seed (63 bits: the sensors' noise), a lock key (256 bits: the match lock's stream) and a draw seed (63 bits), from
which its day is drawn from `control_sim.DEFAULT["hidden"]` (design 1.2): the vehicle's offset, the lever, the pan,
the water, the gusts (one after lift-off, one or two more at fixed times), and the six parts, placed on the real bay's
table tops (`hidden/bay_facts.json`: oriented rectangles) by geometry alone, each with its own box and authored pose
from the real bay's layer (`hidden/bay_true.usda`, /World/Facility/Items/<name>): its table (each part on Table A or B
with probability 1/2, redrawn until 1 to 5 stand on each), its rest (upright as authored; the four lying-capable
parts on a broad side with probability 1/2: turned 90 degrees, either way, about the horizontal axis along the longer
side of the footprint, so the shorter side stands up), a heading uniform over 360 degrees, a place (the footprint's
centre uniform on the top; the whole footprint at least edge_m inside the top's edges and gap_m from every other
part's, by separating axes) and the height that puts the turned box's lowest point 1 mm above the top. A part that
finds no place within `tries` makes the whole arrangement be drawn again (never a fallback). Each day is written with
its values (the bodies' world poses), its draw seed for the record; the file is never overwritten.

corners: the development days of design 3.2 (fixed values, the noise seed and lock key derived from the day's id): a
plain day, then one change each: the pan at (-0.45, -0.45) and at (+0.45, +0.45), the vehicle at (+0.30, -0.30, +25)
and (-0.30, +0.30, -25), the lever at 60 degrees, 5 parts on Table A with every lying-capable part lying, the same on
Table B, a 10 m/s 5 s gust 2 s after lift-off; and a last day with the extremes combined.

select: the pre-registered selection (design 1.3) of a batch run in drawn order through `run_mission.py` (the grader's
path, privileged, no time limit; each result as `<results>/<id>.json`): the first k candidates that pass are kept. A
pass: the run ends "done" (all three jobs, everything still), 6 of 6 parts in the crate, steam and fire out, no robot
lost or crashed, the bucket never dropped, no abort or error, the day's parts exactly the stage's, and no part moved
more than 2 cm during the pre-roll. Every result must come from the frozen reference (`--frozen`: sha256sum lines of
the reference's files; each one the submission holds must match). Writes the set with a `source` that lists every
candidate's outcome.

Round 4 (spec section 4.2): a candidate is considered only if it passes the observability check first
(`check_observability.py`, run on every candidate of the batch before any candidate runs: `<results>/<id>.obs.json`,
its code hashed into `--frozen` as `tests/review/check_observability.py`); one that fails it is dropped before it
runs and recorded with the failing row. More than one candidate in six failing the check stops the selection: the
thresholds or the generator are revisited before a new batch is drawn. `--no-observability` selects as round 3 did.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import secrets
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, TESTS)
import control_sim as cs  # noqa: E402  (its geometry helpers and DEFAULT; pxr and numpy only at import)
from pxr import Usd, UsdPhysics  # noqa: E402

BAY_LAYER = os.path.join(TESTS, "hidden", "bay_true.usda")
BAY_FACTS = os.path.join(TESTS, "hidden", "bay_facts.json")
SET_FORMAT = "hazard_bay_build fixed set 1"
CANDIDATES_FORMAT = "hazard_bay_build candidates 1"
DEV_FORMAT = "hazard_bay_build development days 1"
SCENE_PROBES = ("layout", "settle", "steam", "hoist", "fleet")
LIFT = 0.001  # [m] the box's lowest point above the top


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def day_digest(day):
    """The sha256 of a day as a set holds it (canonical JSON; `run_mission.py` records the same of the day it ran)."""
    return hashlib.sha256(json.dumps(day, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------------------------------- the real bay
def part_geometry():
    """Each real part (a rigid body under /World/Facility/Items of the real bay's layer, in the layer's order): its own
    box (lo, hi in its body's frame: the grader's `own_box`) and its authored world orientation (w, x, y, z)."""
    st = Usd.Stage.Open(BAY_LAYER)
    items = st.GetPrimAtPath(f"{cs.FAC}/Items")
    out = {}
    for c in items.GetChildren():
        if not c.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        path = c.GetPath().pathString
        box = cs.own_box(st, path)
        if box is None:
            raise SystemExit(f"{path}: no box (are the asset library's files there?)")
        q = np.array(cs.usd_pose(st, path)[1], dtype=float)
        out[c.GetName()] = dict(lo=np.array(box[0], dtype=float), hi=np.array(box[1], dtype=float), q=q / np.linalg.norm(q))
    if len(out) != 6:
        raise SystemExit(f"the real bay holds {len(out)} parts, not 6")
    return out


def table_tops():
    tops = {t["name"]: t for t in json.load(open(BAY_FACTS))["table_tops"]}
    if set(tops) != set(cs.TABLES):
        raise SystemExit(f"bay_facts.json: table tops {sorted(tops)}")
    return tops


def rest_quat(g, rest, way):
    """The part's orientation before its heading: upright as authored, or lying on a broad side: turned 90 degrees
    (`way` +1 or -1) about the horizontal axis along the longer side of its footprint as it stands."""
    q = g["q"]
    if rest == "upright":
        return q
    size = g["hi"] - g["lo"]
    axes = [cs.rot(q, e) for e in np.eye(3)]
    up = int(np.argmax([abs(a[2]) for a in axes]))
    along = max((k for k in range(3) if k != up), key=lambda k: size[k])
    u = np.array([axes[along][0], axes[along][1], 0.0])
    u = u / np.linalg.norm(u)
    half = way * math.pi / 4.0
    return cs.qmul(np.array([math.cos(half), *(math.sin(half) * u)]), q)


def footprint(g, q):
    """The box turned by `q`: its footprint's convex hull about the box's centre, the centre's offset from the body's
    origin, the lowest corner's height above the origin."""
    lo, hi = g["lo"], g["hi"]
    corners = np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    cw = np.array([cs.rot(q, c) for c in corners])
    cen = cs.rot(q, (lo + hi) / 2)
    return cs.convex_hull(cw[:, :2]) - cen[:2], cen, float(cw[:, 2].min())


def place(rng, g, q_rest, top, placed, spec):
    """One part on one top: a heading, then a place, `tries` times at most (None: no place found)."""
    c0, yaw = np.array(top["center"], dtype=float), math.radians(float(top["yaw_deg"]))
    ax, ay = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
    hw, hd = top["size"][0] / 2 - spec["edge_m"], top["size"][1] / 2 - spec["edge_m"]
    for _ in range(int(spec["tries"])):
        heading = rng.uniform(0.0, 360.0)
        q = cs.qmul(cs.quat_z(math.radians(heading)), q_rest)
        hull, cen, zmin = footprint(g, q)
        pt = c0 + rng.uniform(-hw, hw) * ax + rng.uniform(-hd, hd) * ay
        world = hull + pt
        rel = world - c0
        if np.any(np.abs(rel @ ax) > hw) or np.any(np.abs(rel @ ay) > hd):
            continue
        if any(cs.sat_gap(world, other) < spec["gap_m"] for other in placed):
            continue
        pos = [float(pt[0] - cen[0]), float(pt[1] - cen[1]), float(top["height"]) + LIFT - zmin]
        return dict(pos=pos, quat=[float(v) for v in q], heading=heading, hull=world)
    return None


def draw_parts(rng, spec, geom, tops, tables=None, rests=None):
    """The six parts' arrangement (see the module's text). `tables` / `rests` ({name: ...} for every part) fix those
    instead of drawing them (the development days)."""
    names = list(geom)
    lo_n, hi_n = spec["per_table"]
    for _ in range(1000):
        if tables:
            table = dict(tables)
        else:
            while True:
                on_a = [rng.random() < 0.5 for _ in names]
                if lo_n <= sum(on_a) <= hi_n and lo_n <= len(names) - sum(on_a) <= hi_n:
                    break
            table = {n: ("Table_A" if a else "Table_B") for n, a in zip(names, on_a)}
        rest, way = {}, {}
        for n in names:
            if rests:
                rest[n] = rests[n]
            else:
                rest[n] = "side" if n in spec["lying"] and rng.random() < spec["p_lying"] else "upright"
            way[n] = (1 if rng.random() < 0.5 else -1) if rest[n] == "side" else 0
        q_rest = {n: rest_quat(geom[n], rest[n], way[n]) for n in names}
        area = {}
        for n in names:
            hull = footprint(geom[n], q_rest[n])[0]
            area[n] = float(np.ptp(hull[:, 0]) * np.ptp(hull[:, 1]))
        placed, poses = {cs.TABLES[0]: [], cs.TABLES[1]: []}, {}
        for n in sorted(names, key=lambda m: -area[m]):  # (the largest footprints first)
            got = place(rng, geom[n], q_rest[n], tops[table[n]], placed[table[n]], spec)
            if got is None:
                break
            poses[n] = got
            placed[table[n]].append(got["hull"])
        if len(poses) == len(names):
            return {n: dict(table=table[n], rest=rest[n], heading_deg=poses[n]["heading"], pos=poses[n]["pos"], quat=poses[n]["quat"])
                    for n in names}
    raise RuntimeError("no arrangement of the parts found in 1000 draws")


def gust(rng, spec, **when):
    g = dict(when)
    g["duration_s"] = rng.uniform(*spec["gust_duration_s"])
    g["speed_mps"] = rng.uniform(*spec["gust_speed_mps"])
    g["dir_deg"] = rng.uniform(0.0, 360.0)
    return g


def draw_day(rng, spec, geom, tops):
    """One day from the spec (design 1.2), in this order: the vehicle, the lever, the pan, the water, the gusts, the
    parts."""
    u, y, p = spec["ugv_xy_m"], spec["ugv_yaw_deg"], spec["pan_m"]
    day = dict(ugv=[rng.uniform(-u, u), rng.uniform(-u, u), rng.uniform(-y, y)])
    day["lever_deg"] = rng.uniform(*spec["lever_deg"])
    day["pan"] = [rng.uniform(-p, p), rng.uniform(-p, p)]
    day["water_l"] = rng.uniform(*spec["water_l"])
    gusts = [gust(rng, spec, after_liftoff_s=rng.uniform(*spec["gust_after_liftoff_s"]))]
    for _ in range(rng.randint(*spec["gusts_extra"])):
        gusts.append(gust(rng, spec, t0_s=rng.uniform(*spec["gust_t0_s"])))
    day["gusts"] = gusts
    day["parts"] = draw_parts(rng, spec, geom, tops)
    return day


def describe(m):
    """A day in one paragraph (printed, never written to a set)."""
    d = m["day"]
    gs = []
    for g in d["gusts"]:
        when = f"lift-off +{g['after_liftoff_s']:.1f} s" if "after_liftoff_s" in g else f"at {g['t0_s']:.1f} s"
        gs.append(f"{when} {g['speed_mps']:.1f} m/s {g['duration_s']:.1f} s toward {g['dir_deg']:.0f} deg")
    by = {t: [f"{n}{' (side)' if p['rest'] == 'side' else ''}" for n, p in d["parts"].items() if p["table"] == t] for t in cs.TABLES}
    return (f"{m['id']}: vehicle ({d['ugv'][0]:+.3f}, {d['ugv'][1]:+.3f} m, {d['ugv'][2]:+.1f} deg), lever {d['lever_deg']:.1f} deg, "
            f"pan ({d['pan'][0]:+.3f}, {d['pan'][1]:+.3f} m), water {d['water_l']:.2f} L\n    gusts: {'; '.join(gs)}\n    "
            + "; ".join(f"{t}: {', '.join(v)}" for t, v in by.items()))


def generator_hashes():
    return {os.path.relpath(p, TESTS): sha256_file(p) for p in (os.path.abspath(__file__), os.path.join(TESTS, "control_sim.py"), BAY_LAYER, BAY_FACTS)}


def check_missions(missions):
    """The grader's own check of a missions list (`verify.read_missions`, the real bay's six parts)."""
    import verify
    truth = json.load(open(os.path.join(TESTS, "truth.json")))
    verify.read_missions(missions, truth["groups"]["settle"]["items"])


# --------------------------------------------------------------------------------------------- the commands
def cmd_draw(a):
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: a drawn batch is never redrawn (write a new file)")
    spec, geom, tops = dict(cs.DEFAULT["hidden"]), part_geometry(), table_tops()
    batch = secrets.token_hex(2)
    missions = []
    for i in range(a.n):
        noise_seed, lock_key, draw_seed = secrets.randbits(63), secrets.token_hex(32), secrets.randbits(63)
        day = draw_day(random.Random(draw_seed), spec, geom, tops)
        missions.append(dict(id=f"{batch}-{i + 1:02d}", noise_seed=noise_seed, lock_key=lock_key, draw_seed=draw_seed, day=day))
    check_missions(missions)
    out = dict(format=CANDIDATES_FORMAT, batch=batch, drawn_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               spec=spec, generator_sha256=generator_hashes(), missions=missions)
    with open(a.out, "x") as f:
        json.dump(out, f, indent=1)
    for m in missions:
        print(describe(m))
    print(f"{len(missions)} candidates (batch {batch}) -> {a.out}")


def dev_day(key, spec, geom, tops):
    """One development day by its key (see `cmd_corners`)."""
    base = dict(ugv=[0.0, 0.0, 0.0], lever_deg=30.0, pan=[0.0, 0.0], water_l=9.25,
                gusts=[dict(after_liftoff_s=7.0, duration_s=3.25, speed_mps=7.0, dir_deg=45.0), dict(t0_s=124.0, duration_s=3.25, speed_mps=7.0, dir_deg=200.0)])
    names = list(geom)
    as_authored = {n: ("Table_A" if n in ("cracker_box", "mustard_bottle", "sugar_box") else "Table_B") for n in names}
    five = {n: ("Table_B" if n == "soup_can" else "Table_A") for n in names}
    lying = {n: ("side" if n in spec["lying"] else "upright") for n in names}
    arrangements = dict(plain=(as_authored, {n: "upright" for n in names}), a5=(five, lying),
                        b5=({n: ("Table_A" if t == "Table_B" else "Table_B") for n, t in five.items()}, lying))

    def parts(which):
        tables, rests = arrangements[which]
        return draw_parts(random.Random(f"hazard_bay_build development parts: {which}"), spec, geom, tops, tables, rests)

    gust10 = dict(after_liftoff_s=2.0, duration_s=5.0, speed_mps=10.0, dir_deg=150.0)
    changes = {
        "plain": dict(parts=parts("plain")),
        "pan-mm": dict(pan=[-0.45, -0.45], parts=parts("plain")),
        "pan-pp": dict(pan=[0.45, 0.45], parts=parts("plain")),
        "ugv-pm": dict(ugv=[0.30, -0.30, 25.0], parts=parts("plain")),
        "ugv-mp": dict(ugv=[-0.30, 0.30, -25.0], parts=parts("plain")),
        "lever60": dict(lever_deg=60.0, parts=parts("plain")),
        "a5-lying": dict(parts=parts("a5")),
        "b5-lying": dict(parts=parts("b5")),
        "gust10": dict(gusts=[gust10, base["gusts"][1]], parts=parts("plain")),
        "extremes": dict(ugv=[0.30, -0.30, 25.0], lever_deg=60.0, pan=[-0.45, -0.45], water_l=10.0,
                         gusts=[gust10, dict(t0_s=60.0, duration_s=5.0, speed_mps=10.0, dir_deg=330.0),
                                dict(t0_s=240.0, duration_s=5.0, speed_mps=10.0, dir_deg=240.0)], parts=parts("b5")),
    }
    day = dict(base, **changes[key])
    mid = f"dev-{key}"
    noise_seed = int(hashlib.sha256(f"noise: {mid}".encode()).hexdigest()[:16], 16) & ((1 << 63) - 1)
    return dict(id=mid, noise_seed=noise_seed, lock_key=hashlib.sha256(f"lock: {mid}".encode()).hexdigest(), day=day)


DEV_KEYS = ("plain", "pan-mm", "pan-pp", "ugv-pm", "ugv-mp", "lever60", "a5-lying", "b5-lying", "gust10", "extremes")


def cmd_corners(a):
    spec, geom, tops = dict(cs.DEFAULT["hidden"]), part_geometry(), table_tops()
    missions = [dev_day(k, spec, geom, tops) for k in DEV_KEYS]
    check_missions(missions)
    out = dict(format=DEV_FORMAT, note="Development days (design 3.2): fixed corner values of the day's ranges, for the reference's "
               "development missions (run_mission.py --set FILE --index K); never part of the fixed hidden set. "
               "Index: " + ", ".join(f"{k} {m['id']}" for k, m in enumerate(missions)),
               spec=spec, generator_sha256=generator_hashes(), missions=missions)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for k, m in enumerate(missions):
        print(f"[{k}] " + describe(m))
    print(f"{len(missions)} development days -> {a.out}")


def fails(res):
    """The pre-registered criterion (design 1.3) on one result of run_mission.py: the reasons it fails (none: a pass)."""
    d = res.get("data") or {}
    h = d.get("hidden") or {}
    why = []
    if d.get("error"):
        why.append("a grader error")
    if d.get("abort"):
        why.append("aborted")
    if d.get("end") != "done":
        why.append(f"ended '{d.get('end')}'")
    if d.get("n_parts") != 6 or d.get("parts_in_crate") != 6:
        why.append(f"{d.get('parts_in_crate')} of {d.get('n_parts')} parts in the crate")
    if d.get("steam_end") is not False:
        why.append("the steam on")
    if d.get("fire_end") is not False:
        why.append("the fire burning")
    if d.get("lost"):
        why.append(f"lost: {', '.join(sorted(d['lost']))}")
    if d.get("crash"):
        why.append(f"crashed: {', '.join(d['crash'])}")
    if d.get("dropped"):
        why.append("the bucket dropped")
    if h.get("parts_missing") or h.get("parts_unplaced") or not h.get("parts_moved"):
        why.append("the day's parts are not the stage's")
    moved = h.get("parts_moved") or {}
    far = sorted(n for n, v in moved.items() if not isinstance(v, (int, float)) or v > 0.02)
    if far:
        why.append(f"moved more than 2 cm in the pre-roll: {', '.join(far)}")
    return why


def read_frozen(path):
    """sha256sum lines (`<hex>  <path>`): {file name: sha256}."""
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        h, name = line.split(None, 1)
        out[os.path.basename(name.lstrip("*"))] = h.lower()
    if not out:
        raise SystemExit(f"{path}: no hashes")
    return out


def cmd_select(a):
    cands = json.load(open(a.candidates))
    if cands.get("format") != CANDIDATES_FORMAT:
        raise SystemExit(f"{a.candidates}: not a batch of candidates ({CANDIDATES_FORMAT})")
    check_missions([{k: m[k] for k in ("id", "noise_seed", "lock_key", "day")} for m in cands["missions"]])
    frozen = read_frozen(a.frozen)
    kept, lines = [], []
    obs_failed, obs_hash = 0, None
    for m in cands["missions"]:
        path = os.path.join(a.results, f"{m['id']}.json")
        label = f"{m['id']} (draw seed {m['draw_seed']})"
        if not a.no_observability and len(kept) < a.k:  # (the observability check, run on every candidate before any ran)
            op = os.path.join(a.results, f"{m['id']}.obs.json")
            if not os.path.isfile(op):
                raise SystemExit(f"{m['id']} has no observability result ({op}): run check_observability.py on the batch first")
            obs = json.load(open(op))
            want = next((v for k, v in frozen.items() if k.endswith("check_observability.py")), None)
            if want is None or obs.get("check_file_sha256") != want:
                raise SystemExit(f"{op}: not the frozen observability check")
            obs_hash = obs.get("check_sha256")
            if not obs.get("ok"):
                obs_failed += 1
                worst = "; ".join(f"row {k}: {json.dumps({kk: vv for kk, vv in obs['rows'][k].items() if kk in ('share', 'px2', 'px', 'n', 'needle_px', 'dial_px')})}"
                                  for k in obs.get("failed", [])[:3])
                lines.append(f"{label}: not considered, a target not observable ({worst})")
                continue
        if len(kept) >= a.k:
            if os.path.isfile(path):
                why = fails(json.load(open(path)))
                lines.append(f"{label}: not considered, {a.k} kept before it in drawn order (it ran: {'pass' if not why else '; '.join(why)})")
            else:
                lines.append(f"{label}: not run, {a.k} kept before it in drawn order")
            continue
        if not os.path.isfile(path):
            raise SystemExit(f"{m['id']} has no result ({path}): run the candidates in drawn order before the selection")
        res = json.load(open(path))
        mm = res.get("mission") or {}
        if mm.get("id") != m["id"] or mm.get("day_sha256") != day_digest(m["day"]):
            raise SystemExit(f"{path}: not the run of candidate {m['id']} as drawn")
        opts = res.get("options") or {}
        if not res.get("privileged") or opts.get("time_limit") or opts.get("no_token") or opts.get("program_env"):
            raise SystemExit(f"{path}: not the grader's path (the privileged reference, no time limit, no extra environment)")
        hashes = {os.path.basename(k): v for k, v in (res.get("submission_sha256") or {}).items()}
        checked = [n for n in frozen if n in hashes]
        if not checked or any(hashes[n] != frozen[n] for n in checked):
            raise SystemExit(f"{path}: not the frozen reference ({', '.join(n for n in checked if hashes[n] != frozen[n]) or 'no file in common'})")
        why = fails(res)
        if why:
            lines.append(f"{label}: dropped ({'; '.join(why)})")
        else:
            kept.append(m)
            lines.append(f"{label}: kept")
    if obs_failed > len(cands["missions"]) / 6.0:
        for line in lines:
            print(line)
        raise SystemExit(f"{obs_failed} of {len(cands['missions'])} candidates failed the observability check: revisit the thresholds or the "
                         "generator before a new batch is drawn")
    if len(kept) < a.k:
        for line in lines:
            print(line)
        raise SystemExit(f"{len(kept)} of {len(cands['missions'])} candidates passed: draw a new batch (and look at why they dropped)")
    missions = [{k: m[k] for k in ("id", "noise_seed", "lock_key", "day")} for m in kept]
    check_missions(missions)
    source = (f"Drawn {cands['drawn_utc']} by tests/review/make_fixed_set.py draw: batch {cands['batch']}, {len(cands['missions'])} candidates, "
              "each its id, noise seed, lock key and draw seed from os.urandom (secrets), all drawn before any of them ran; each day "
              "drawn from its draw seed by control_sim.DEFAULT['hidden'] (generator sha256: "
              + ", ".join(f"{k} {v}" for k, v in sorted(cands["generator_sha256"].items())) + "). "
              "The reference was frozen first (" + ", ".join(f"{k} {v}" for k, v in sorted(frozen.items())) + "). "
              "The candidates ran in drawn order through tests/review/run_mission.py (the grader's path: the privileged reference, "
              "the mission's lock key and noise seed, no time limit); kept: the first " + str(a.k) + " that passed the criterion fixed "
              "before any ran: the run ends done (all three jobs, everything still), 6 of 6 parts in the crate, steam and fire out, "
              "no robot lost or crashed, the bucket never dropped, no abort or error, the day's parts exactly the stage's, no part "
              "moved more than 2 cm during the pre-roll"
              + ("" if a.no_observability else
                 f"; every candidate first passed the pre-registered observability check (tests/review/check_observability.py, its code "
                 f"and thresholds sha256 {obs_hash}), run on the whole batch before any candidate ran")
              + ". Candidates: " + " | ".join(lines) + ".")
    out = dict(format=SET_FORMAT, scene_probes={g: 0 for g in SCENE_PROBES}, missions=missions, source=source)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    for line in lines:
        print(line)
    print(f"kept {', '.join(m['id'] for m in kept)} -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--n", type=int, required=True)
    d.add_argument("--out", required=True)
    c = sub.add_parser("corners")
    c.add_argument("--out", required=True)
    s = sub.add_parser("select")
    s.add_argument("--candidates", required=True)
    s.add_argument("--results", required=True)
    s.add_argument("--k", type=int, default=2)
    s.add_argument("--frozen", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--no-observability", action="store_true")
    a = ap.parse_args()
    dict(draw=cmd_draw, corners=cmd_corners, select=cmd_select)[a.cmd](a)


if __name__ == "__main__":
    main()
