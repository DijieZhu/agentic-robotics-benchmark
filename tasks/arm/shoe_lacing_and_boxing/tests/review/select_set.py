"""Pre-registered selection of the fixed hidden set (tests/hidden/seeds.json): a maintainer's tool, the grader never
runs it.

The rule, fixed before any candidate ran (the owner's scoring rule for contributed tasks, 2026-11-04, and the owner's
note of 2026-11-06 that a test runs once):

1. Freeze. `freeze` records in tests/hidden/selection.json the sha256 of the reference (solution/solve.sh,
   solution/reference/*, tests/reference/*), of the cell and the grader (tests/shoecell/*, tests/cellkit/*,
   tests/verify.py, tests/test.sh, tests/grader_config.json, tests/given_hashes.json, which pins every given file the
   grader compiles and serves, tests/hidden/oracle_token.sha256), of the image's inputs (environment/Dockerfile,
   requirements.txt, the Panda's two sha256 lists, obj2stl.py) and the task image the runs use (tag and id). The
   freeze is committed before the draw. `draw` refuses to start unless every frozen file equals the record and its
   committed version; `run-one` refuses to run when a frozen file it can see differs from the record
   (review/run_candidate.sh refuses another image).
2. Draw. `draw` takes the candidates from os.urandom (8 bytes each, little-endian, the top bit cleared: the 63-bit
   seeds verify.py runs; a draw below 2**32 or a duplicate is drawn again) and writes them in drawn order, with the
   draw's time and this tool's sha256, to selection.json before any of them runs. Batch 1: 24 candidates.
3. Run. Each candidate ONCE (no repeats under other BLAS kernels or nudged states), in a fresh container of the
   recorded image with no network (review/run_candidate.sh), through the grader's own episode call: verify.run_episode
   with the grader's configuration (tests/grader_config.json), a fresh cell in hidden mode and the submission that
   solution/solve.sh installs (the reference packer and the token, so the cell is privileged), run as the runner
   account over the wire. The only additions are read-only: the scene after the cell's reset, a sha256 of the final
   state and, once the episode is over, the start-state check of item 4. Candidates run in parallel; their order is the
   draw index.
4. Keep. A candidate passes when its episode scores work_score 1.0 (the pair's two shoes laced through every row and
   snug, the box erected, packed, closed and shipped, no safety violation), with no internal error of the cell and no
   physics fault, and when its scene starts physical (`start_state`, read only, on a separate cell instance of the
   same code after the graded run): at t = 0 no contact deeper than 1 mm between two bodies (lace segments among
   themselves aside), the lid tray flat on the infeed tray's plate (tilted less than 1 degree, touching no rim, its
   footprint inside the rim), no lace inside the lid tray (a lace capsule reaching inside its footprint below its top,
   or touching it inside its walls), and over 1 s with nothing commanded no shoe, the lid or the blank moves more than
   2 mm. (Added for the second selection, 2026-11-07, before its draw: in the first, hidden scene 4's lid tray had been
   set down with a wall inside the tray's near rim and started jammed on it, 6 contacts 2.5-5.5 mm deep, 219 N, moving
   at 0.55 m/s; and in a development episode of the fixed cell an aglet that had slid under a lid wall dragged the lid
   off the tray when its shoe was lifted. Speed at t = 0 is not the test: a side-lying shoe can read 3-11 cm/s there
   and stop within 0.1 s, less than 1 mm on.) The first 6 that pass, in drawn order (6 = grader_config.json's episodes), are the set, in that
   order. Every candidate is recorded: kept; dropped, with its outcome and the reason; or ran after the 6th kept and
   not used. With fewer than 6 passing, `draw --batch 2` adds 12 candidates once every candidate of batch 1 has run.
5. Record and check. Every run's outcome (score, milestones, violations, slips, times, the final state's sha256) and
   its scene (the work order drawn from the seed: the batch's shoe model and size, the box, the colours, each shoe's
   arrival (upright or on its side, which way round), the blank, the lid, the cradle and light jitter; and where the lid
   rests after the reset: flat, tilted on the infeed tray's rim, or pushed off the tray). `report` prints the hollowing
   check: the kept scenes' features against all candidates' and the dropped ones'. The set is hollow when a count of
   side-lying arrivals per pair (0, 1 or 2) that some candidate has is missing from the kept scenes; the report says
   whether a drop (the pass criterion) or the cutoff (the drawn order) left it out, and lists the other features (the
   size, the colours, the jitter, the lid's rest) for the reader.

    python3 -B tests/review/select_set.py freeze --image TAG
    python3 -B tests/review/select_set.py draw [--batch 1] [--count 24]
    review/run_candidate.sh ...        (a container per candidate; inside it: select_set.py run-one)
    python3 -B tests/review/select_set.py collect --results DIR
    python3 -B tests/review/select_set.py make [--size 6]
    python3 -B tests/review/select_set.py report [--size 6]

freeze, draw, collect, make and report need only the standard library; run-one needs the task image's interpreter (it
imports verify.py). Paths are found from this file, so it runs from any checkout.
"""

import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
TASK = os.path.dirname(TESTS)
SELECTION = os.path.join(TESTS, "hidden", "selection.json")
SEEDS = os.path.join(TESTS, "hidden", "seeds.json")
SEL_FORMAT = "shoe_lacing_and_boxing selection 2"
FROZEN_GLOBS = ("solution/solve.sh", "solution/reference/*", "tests/reference/*.py", "tests/shoecell/*.py",
                "tests/cellkit/*.py", "tests/verify.py", "tests/test.sh", "tests/grader_config.json",
                "tests/given_hashes.json", "tests/hidden/oracle_token.sha256", "environment/Dockerfile",
                "environment/requirements.txt", "environment/panda_assets.sha256", "environment/panda_converted.sha256",
                "environment/obj2stl.py")
TOOL = ("tests/review/select_set.py", "tests/review/run_candidate.sh")
BATCH_SIZE = {1: 24}
LATER_BATCH = 12
SET_SIZE = 6
MASK63 = (1 << 63) - 1
START_DEEP_MM = 1.0           # start state (rule item 4): no contact deeper than this at t = 0 (lace-lace aside)
START_TILT_DEG = 1.0          # the lid tray tilted less than this
START_IDLE_S = 1.0            # with nothing commanded for this long ...
START_MOVE_MM = 2.0           # ... no shoe, the lid or the blank moves farther than this


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen_paths(root=TASK):
    out = set()
    for pattern in FROZEN_GLOBS:
        for p in glob.glob(os.path.join(root, pattern)):
            if os.path.isfile(p) and not os.path.islink(p) and "__pycache__" not in p:
                out.add(os.path.relpath(p, root))
    return sorted(out)


def frozen_hashes(root=TASK):
    return {p: sha256_file(os.path.join(root, p)) for p in frozen_paths(root)}


def load(path=SELECTION):
    with open(path) as f:
        return json.load(f)


def save(sel, path=SELECTION):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sel, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args):
    return subprocess.run(["git", "-C", TASK] + list(args), capture_output=True, text=True, timeout=60)


def committed(paths):
    """True when every path is tracked and equals its committed version."""
    tracked = git("ls-files", "--error-unmatch", "--", *paths)
    clean = git("diff", "--quiet", "HEAD", "--", *paths)
    return tracked.returncode == 0 and clean.returncode == 0


# ------------------------------------------------------------------------------------------------ freeze and draw
def cmd_freeze(a):
    if os.path.exists(SELECTION):
        raise SystemExit("{} exists: a freeze starts a selection".format(SELECTION))
    image_id = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", a.image], capture_output=True,
                              text=True, timeout=60).stdout.strip()
    if not image_id:
        raise SystemExit("no image {}".format(a.image))
    sel = {"format": SEL_FORMAT, "frozen_utc": utc_now(), "frozen": frozen_hashes(),
           "image": {"tag": a.image, "id": image_id, "built_from": "environment/ (its files are frozen above)"},
           "rule": "tests/review/select_set.py (module docstring)"}
    save(sel)
    print("frozen {} files, image {} {} -> {}".format(len(sel["frozen"]), a.image, image_id[:19], SELECTION))


def cmd_draw(a):
    sel = load()
    now = frozen_hashes()
    if sel["frozen"] != now:
        raise SystemExit("the frozen files differ from the record: {}".format(
            sorted(set(sel["frozen"]) ^ set(now)) or [p for p in now if sel["frozen"][p] != now[p]]))
    rel_sel = os.path.relpath(SELECTION, TASK)
    if a.batch == 1:
        if sel.get("candidates"):
            raise SystemExit("batch 1 is drawn already")
        if not committed(list(now) + [rel_sel]):
            raise SystemExit("the freeze is not committed as it is: commit it first")
        sel["frozen_commit"] = git("rev-parse", "HEAD").stdout.strip()
        sel["batches"], sel["candidates"] = [], []
    else:
        if any(c["status"] == "drawn" for c in sel["candidates"]):
            raise SystemExit("an earlier batch has not run yet")
    count = a.count or BATCH_SIZE.get(a.batch, LATER_BATCH)
    seen = {c["seed"] for c in sel["candidates"]}
    redrawn, new = 0, []
    while len(new) < count:
        s = int.from_bytes(os.urandom(8), "little") & MASK63
        if s < (1 << 32) or s in seen:
            redrawn += 1
            continue
        seen.add(s)
        new.append(s)
    for s in new:
        sel["candidates"].append({"draw": len(sel["candidates"]), "batch": a.batch, "seed": s, "status": "drawn"})
    sel["batches"].append({"batch": a.batch, "drawn_utc": utc_now(), "count": count, "redrawn": redrawn,
                           "source": "os.urandom(8) per seed, little-endian, the top bit cleared (63 bits); a draw "
                                     "below 2**32 or a duplicate drawn again",
                           "tool_sha256": {p: sha256_file(os.path.join(TASK, p)) for p in TOOL}})
    save(sel)
    print("batch {}: {} candidates drawn ({} redrawn) -> {}".format(a.batch, count, redrawn, SELECTION))


# ------------------------------------------------------------------------------------------------ one run (container)
def scene_features(server):
    """The scene of an episode: its work order (a function of the seed) and where the lid rests after the reset. Reads
    only."""
    import numpy as np
    o = server.order
    p = o["pairs"][0]
    sc = server.scene
    lp, _ = sc.lid_pose()
    R = np.array(sc.data.xmat[sc.lid_body]).reshape(3, 3)
    tilt = float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0.0, 1.0))))
    moved = 1000.0 * math.hypot(float(lp[0]) - p["lid"]["pos"][0], float(lp[1]) - p["lid"]["pos"][1])
    rest = "off the tray" if moved > 50.0 else ("on the rim" if tilt > 1.0 else "flat")

    def way(yaw):
        return "turned" if abs(math.remainder(float(yaw), 2 * math.pi)) > math.pi / 2 else "as drawn"

    return {
        "batch": {k: o["batch"][k] for k in ("length", "width", "row_spacing", "throat_gap")},
        "carton": dict(o["carton"]),
        "colours": {"upper": p["upper_colour"], "sole": p["sole_colour"], "lace": p["lace_colour"]},
        "shoes": [{"side": s["side"], "on_side": bool(s["on_side"]), "side_sign": int(s["side_sign"]),
                   "yaw": float(s["yaw"]), "way": way(s["yaw"]), "pos": [float(v) for v in s["pos"]]} for s in p["shoes"]],
        "side_lying": int(sum(bool(s["on_side"]) for s in p["shoes"])),
        "blank": {"pos": list(p["blank"]["pos"]), "yaw": float(p["blank"]["yaw"]), "way": way(p["blank"]["yaw"])},
        "lid": {"pos": list(p["lid"]["pos"]), "yaw": float(p["lid"]["yaw"]), "rest": rest, "tilt_deg": tilt,
                "moved_mm": moved, "rest_pos": [float(v) for v in lp]},
        "jitter": {"cradle": list(o["jitter"]["cradle"]), "light": list(o["jitter"]["light"])},
    }


def start_state(seed, cfg, given):
    """Rule item 4's start-state check: the scene the program meets at t = 0, rebuilt from the seed on a separate cell
    instance (hidden mode, no reference, no rendering; the graded run is not touched), then START_IDLE_S of control
    ticks with nothing commanded (the arms hold their home posture). Reads only."""
    import numpy as np
    import mujoco
    from shoecell.server import ShoeCellServer

    s = ShoeCellServer(given, config=cfg, mode="hidden", privileged=False, render=False, log=lambda msg: None)
    s.reset(int(seed))
    sc, m, d = s.scene, s.scene.model, s.scene.data
    f6 = np.zeros(6)
    lid_geoms = {j for j in range(m.ngeom) if int(m.geom_bodyid[j]) == sc.lid_body}
    rim_geoms = {j for j in range(m.ngeom) if m.geom(j).name.startswith("infeed/rim")}
    deep, rim_n = {}, 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.body(int(m.geom_bodyid[c.geom1])).name, m.body(int(m.geom_bodyid[c.geom2])).name
        if (c.geom1 in lid_geoms and c.geom2 in rim_geoms) or (c.geom2 in lid_geoms and c.geom1 in rim_geoms):
            mujoco.mj_contactForce(m, d, i, f6)
            rim_n += abs(float(f6[0]))
        if c.dist >= -START_DEEP_MM / 1000.0 or (b1.startswith("lace") and b2.startswith("lace")):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        key = "|".join(sorted(n.split("/")[0] if n.startswith("lace") else n for n in (b1, b2)))
        e = deep.setdefault(key, {"depth_mm": 0.0, "normal_N": 0.0, "contacts": 0})
        e["depth_mm"] = min(e["depth_mm"], 1000.0 * float(c.dist))
        e["normal_N"] += abs(float(f6[0]))
        e["contacts"] += 1
    R = np.array(d.xmat[sc.lid_body]).reshape(3, 3)
    tilt = float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0.0, 1.0))))
    corners = []
    for j in lid_geoms:
        if int(m.geom_contype[j]) == 0 and int(m.geom_conaffinity[j]) == 0:
            continue                                    # (the printed logo: no collision)
        Rg, sz = d.geom_xmat[j].reshape(3, 3), m.geom_size[j]
        corners += [d.geom_xpos[j] + Rg @ np.array([sx * sz[0], sy * sz[1], sz[2]]) for sx in (-1, 1) for sy in (-1, 1)]
    (cx, cy), (hx, hy) = sc.layout["infeed"]["centre"], sc.layout["infeed"]["half"]
    inner = 0.004                                       # the rim's half thickness (physics.Scene: rim boxes at +-half)
    inside_mm = 1000.0 * min(min(float(p[0]) - (cx - hx + inner), (cx + hx - inner) - float(p[0]),
                                 float(p[1]) - (cy - hy + inner), (cy + hy - inner) - float(p[1])) for p in corners)
    # lace inside the lid tray: a lace capsule (cord or aglet) with a point of its axis inside the tray's outer footprint
    # below its top plate, or touching it inside the walls' inner faces (an aglet that slid under a wall in the settle
    # drags the lid along when its tail is pulled)
    lp, Rl = d.xpos[sc.lid_body].copy(), np.array(d.xmat[sc.lid_body]).reshape(3, 3)
    dims = sc.lid["dims"]
    lace_geoms = {}
    for i, ids in enumerate(sc.lace_bodies):
        for b in ids:
            lace_geoms[int(b)] = "lace%d" % i
    in_lid = {}
    for g in range(m.ngeom):
        name = lace_geoms.get(int(m.geom_bodyid[g]))
        if name is None or int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            continue
        ax = d.geom_xmat[g].reshape(3, 3)[:, 2]
        for u in np.linspace(-1.0, 1.0, 5):
            q = Rl.T @ (d.geom_xpos[g] + u * float(m.geom_size[g][1]) * ax - lp)
            if abs(q[0]) < dims["Lo"] / 2 and abs(q[1]) < dims["Wo"] / 2 and q[2] < 0.0:
                in_lid[name] = in_lid.get(name, 0) + 1
                break
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 in lid_geoms or c.geom2 in lid_geoms:
            name = lace_geoms.get(int(m.geom_bodyid[c.geom2 if c.geom1 in lid_geoms else c.geom1]))
            q = Rl.T @ (np.asarray(c.pos) - lp)
            if name and abs(q[0]) <= dims["Li"] / 2 + 0.0005 and abs(q[1]) <= dims["Wi"] / 2 + 0.0005:
                in_lid[name + " (contact inside the walls)"] = in_lid.get(name + " (contact inside the walls)", 0) + 1
    bodies = {"shoe%d" % i: b for i, b in enumerate(sc.shoe_body)}
    bodies.update({"lid": sc.lid_body, "blank": sc.carton_floor_body})
    vadr = {"shoe%d" % i: a for i, a in enumerate(sc.shoe_vadr)}
    vadr.update({"lid": sc.lid_vadr, "blank": sc.carton_vadr})
    p0 = {k: d.xpos[b].copy() for k, b in bodies.items()}
    speed0 = {k: float(np.linalg.norm(d.qvel[a:a + 3])) for k, a in vadr.items()}
    for _ in range(int(round(START_IDLE_S / s.tick_s))):
        s.physics_tick()
    moved = {k: 1000.0 * float(np.linalg.norm(d.xpos[b] - p0[k])) for k, b in bodies.items()}
    out = {"deep": deep, "lid_tilt_deg": tilt, "lid_rim_contact_N": rim_n, "lid_inside_rim_mm": inside_mm,
           "lace_in_lid": in_lid, "idle_s": START_IDLE_S, "moved_mm": moved, "speed_t0_m_s": speed0}
    why_not = []
    if deep:
        why_not.append("contacts deeper than %.0f mm: %s" % (START_DEEP_MM, sorted(deep)))
    if tilt >= START_TILT_DEG:
        why_not.append("lid tilted %.1f deg" % tilt)
    if rim_n > 0.0 or inside_mm <= 0.0:
        why_not.append("lid on the rim (%.1f N, %.1f mm inside)" % (rim_n, inside_mm))
    if in_lid:
        why_not.append("lace inside the lid tray: %s" % sorted(in_lid))
    far = sorted(k for k, v in moved.items() if v > START_MOVE_MM)
    if far:
        why_not.append("moved over %.0f mm in %.0f s idle: %s" % (START_MOVE_MM, START_IDLE_S, far))
    out["ok"] = not why_not
    out["why_not"] = why_not
    return out


def cmd_run_one(a):
    sys.path.insert(0, TESTS)
    import numpy as np
    import mujoco
    import verify
    from shoecell.server import ShoeCellServer

    started, t0 = utc_now(), time.time()
    frozen_ok, differs = None, []
    if a.selection:
        sel = load(a.selection)
        for rel, want in sorted(sel["frozen"].items()):
            if rel.startswith("tests/"):
                path = os.path.join(TESTS, rel[len("tests/"):])
            elif rel == "solution/reference/packer.py":
                path = os.path.join(verify.SUBMISSION, "packer.py")
            elif rel == "solution/reference/oracle_token":
                path = os.path.join(verify.SUBMISSION, ".oracle_token")
            elif a.task:
                path = os.path.join(a.task, rel)
            else:
                continue
            if not os.path.isfile(path) or sha256_file(path) != want:
                differs.append(rel)
        frozen_ok = not differs
        if differs and not a.dev:
            raise SystemExit("frozen files differ: {}".format(differs))
    cfg = verify.load_json(os.path.join(verify.STAGE_DIR, "grader_config.json"))
    ok, problems = verify.root_payload_ok(verify.load_json(os.path.join(verify.STAGE_DIR, "given_hashes.json")))
    if not ok:
        raise SystemExit("the root-owned givens are unusable: {}".format(problems[:5]))
    if verify.token_state() != (True, False):
        raise SystemExit("the submission carries no matching token")
    rec = {}

    class Recorded(ShoeCellServer):
        def reset(self, seed):
            out = ShoeCellServer.reset(self, seed)
            rec["scene"] = scene_features(self)
            return out

        def finalize(self):
            out = ShoeCellServer.finalize(self)
            rec["final_qpos_sha256"] = hashlib.sha256(np.asarray(self.scene.data.qpos, np.float64).tobytes()).hexdigest()
            return out

    verify.ShoeCellServer = Recorded
    rn = verify.Runner(cfg)
    summary = verify.run_episode(cfg, int(a.seed), rn, True, tag="ep01")
    start = start_state(int(a.seed), cfg, verify.ROOT_GIVEN)      # (after the graded run: it cannot touch it)
    events = summary.get("events") or []
    pairs = summary.get("pairs") or []
    res = {
        "seed": int(a.seed), "label": a.label, "started_utc": started, "finished_utc": utc_now(),
        "wall_s": time.time() - t0, "image_id": os.environ.get("PR48_IMAGE_ID"), "frozen_ok": frozen_ok,
        "frozen_differs": differs, "python": sys.version.split()[0], "numpy": np.__version__,
        "mujoco": mujoco.__version__, "cpus_visible": len(os.sched_getaffinity(0)),
        "blas_env": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_CORETYPE")},
        "work_score": summary["work_score"], "safety_ok": summary["safety_ok"], "violations": summary["violations"],
        "done_reason": summary["done_reason"], "end_reason": summary.get("end_reason"),
        "internal_errors": summary["internal_errors"], "illegal_call_count": summary["illegal_call_count"],
        "sim_time_s": summary["sim_time_s"], "ticks": summary["ticks"], "step_calls": summary["step_calls"],
        "calls": summary["calls"], "camera_frames_total": summary["camera_frames_total"],
        "program_time_s": summary["program_time_s"], "cell_work_s": summary["cell_work_s"],
        "episode_wall_s": summary["episode_wall_s"],
        "pairs": [{k: p[k] for k in ("credit", "safety_void", "milestones", "detail", "shipped_at")} for p in pairs],
        "slips": sum(1 for _, e in events if str(e).startswith("aglet_slipped")),
        "n_events": len(events), "events_sha256": hashlib.sha256(json.dumps(events).encode()).hexdigest(),
        "final_qpos_sha256": rec.get("final_qpos_sha256"), "scene": rec.get("scene"), "start_state": start,
    }
    res["passes"] = passes(res)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
        f.write("\n")
    full = (a.out[:-len(".json")] if a.out.endswith(".json") else a.out) + ".summary.json"
    with open(full, "w") as f:                  # the cell's whole summary, its events included
        json.dump(summary, f, indent=1, default=float)
    print("seed {} work_score {:.4f} sim {:.2f} s ticks {} slips {} void {} start {} passes {} wall {:.0f} s".format(
        a.seed, res["work_score"], res["sim_time_s"], res["ticks"], res["slips"], not res["safety_ok"],
        "ok" if start["ok"] else start["why_not"], res["passes"], res["wall_s"]), flush=True)


# ------------------------------------------------------------------------------------------------ the rule
def passes(res):
    return (float(res["work_score"]) == 1.0 and bool(res["safety_ok"]) and int(res["internal_errors"]) == 0
            and res["done_reason"] != "physics_fault" and bool((res.get("start_state") or {}).get("ok")))


def why(res):
    """What a run that does not pass lost, from its record."""
    st = res.get("start_state") or {}
    if not st.get("ok"):
        return "start state: " + ("; ".join(st.get("why_not") or []) or "not checked")
    if res["done_reason"] == "physics_fault":
        return "physics fault"
    if int(res["internal_errors"]):
        return "{} internal error(s) of the cell".format(res["internal_errors"])
    if not res["safety_ok"]:
        return "voided: " + "; ".join(v["what"] for v in res["violations"][:3])
    ms = (res["pairs"][0] if res["pairs"] else {"milestones": {}})["milestones"]
    short = ["{} {:.2f}".format(k, float(ms.get(k, 0.0))) for k in ("lacing_a", "lacing_b", "erected", "packed",
                                                                     "closed", "shipped") if float(ms.get(k, 0.0)) < 1.0]
    return "work_score {:.4f}; milestones below 1: {}".format(res["work_score"], ", ".join(short) or "none")


def classify(sel, size):
    kept, dropped, after = [], [], []
    for c in sorted(sel["candidates"], key=lambda c: c["draw"]):
        if c["status"] == "drawn":
            raise SystemExit("draw {} has not run".format(c["draw"]))
        if len(kept) == size:
            after.append(c)
        elif c["result"]["passes"]:
            kept.append(c)
        else:
            dropped.append(c)
    return kept, dropped, after


def cmd_collect(a):
    sel = load()
    got = 0
    runs = []
    for c in sel["candidates"]:
        path = os.path.join(a.results, "{:02d}.json".format(c["draw"]))
        if c["status"] != "drawn" or not os.path.isfile(path):
            continue
        res = load(path)
        if res["seed"] != c["seed"]:
            raise SystemExit("{}: seed {} is not draw {}'s".format(path, res["seed"], c["draw"]))
        if not res["frozen_ok"]:
            raise SystemExit("{}: ran with frozen files that differ: {}".format(path, res["frozen_differs"]))
        if res["image_id"] != sel["image"]["id"]:
            raise SystemExit("{}: ran in image {}, not the recorded one".format(path, res["image_id"]))
        res["passes"] = passes(res)
        c["result"] = res
        c["status"] = "ran"
        runs.append(res)
        got += 1
    if runs:
        sel.setdefault("runs", []).append({
            "collected_utc": utc_now(), "candidates": got, "first_start_utc": min(r["started_utc"] for r in runs),
            "last_finish_utc": max(r["finished_utc"] for r in runs), "image_id": sel["image"]["id"],
            "python": sorted({r["python"] for r in runs}), "cpus_visible": sorted({r["cpus_visible"] for r in runs}),
            "blas_env": runs[0]["blas_env"], "results": os.path.abspath(a.results)})
    save(sel)
    print("collected {} runs; {} still drawn".format(got, sum(1 for c in sel["candidates"] if c["status"] == "drawn")))


def hollowing(kept, dropped, after):
    every = kept + dropped + after

    def kinds(cs):
        return sorted({c["result"]["scene"]["side_lying"] for c in cs})

    missing = [k for k in kinds(every) if k not in kinds(kept)]
    cause = {}
    for k in missing:
        by = []
        if any(c["result"]["scene"]["side_lying"] == k for c in dropped):
            by.append("dropped (the pass criterion): draws {}".format(
                [c["draw"] for c in dropped if c["result"]["scene"]["side_lying"] == k]))
        if any(c["result"]["scene"]["side_lying"] == k for c in after):
            by.append("after the cutoff (the drawn order): draws {}".format(
                [c["draw"] for c in after if c["result"]["scene"]["side_lying"] == k]))
        cause[str(k)] = "; ".join(by)
    return {"side_lying_kinds_all": kinds(every), "side_lying_kinds_kept": kinds(kept), "missing": missing,
            "missing_cause": cause, "hollow": bool(missing)}


def cmd_make(a):
    sel = load()
    if sel["frozen"] != frozen_hashes():
        raise SystemExit("the frozen files changed since the draw")
    kept, dropped, after = classify(sel, a.size)
    if len(kept) < a.size:
        raise SystemExit("only {} kept: draw another batch".format(len(kept)))
    for c in kept:
        c["status"] = "kept"
    for c in dropped:
        c["status"] = "dropped"
        c["reason"] = why(c["result"])
    for c in after:
        c["status"] = "after_cutoff"
        c["outcome"] = "passes" if c["result"]["passes"] else why(c["result"])
    seeds = [c["seed"] for c in kept]
    doc = {"seeds": seeds,
           "source": "the fixed hidden set: the first {} candidates in drawn order on which the frozen reference scores "
                     "1.0 (tests/hidden/selection.json; the rule in tests/review/select_set.py)".format(a.size)}
    with open(SEEDS, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    sel["set"] = {"made_utc": utc_now(), "size": a.size, "kept": [c["draw"] for c in kept],
                  "dropped": [c["draw"] for c in dropped], "after_cutoff": [c["draw"] for c in after],
                  "seeds_sha256": sha256_file(SEEDS), "hollowing": hollowing(kept, dropped, after)}
    save(sel)
    print("set: {} kept (draws {}), {} dropped, {} after the cutoff -> {}".format(
        len(kept), sel["set"]["kept"], len(dropped), len(after), SEEDS))


def _fmt(v):
    return "{:.4f}".format(v) if isinstance(v, float) else str(v)


def cmd_report(a):
    sel = load()
    kept, dropped, after = classify(sel, a.size)
    every = kept + dropped + after
    groups = (("kept", kept), ("dropped", dropped), ("after cutoff", after), ("all", every))
    print("kept {}, dropped {}, after the cutoff {}".format(len(kept), len(dropped), len(after)))
    print("\n| draw | seed | status | work_score | start | side-lying | shoe L / W (mm) | colours | lid | sim s | slips | wall s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in every:
        r, sc = c["result"], c["result"]["scene"]
        st = "kept" if c in kept else ("dropped" if c in dropped else "after cutoff")
        ss = r.get("start_state") or {}
        print("| {} | {} | {} | {:.4f} | {} | {} | {:.1f} / {:.1f} | {}/{}/{} | {} | {:.2f} | {} | {:.0f} |".format(
            c["draw"], c["seed"], st, r["work_score"], "ok" if ss.get("ok") else "; ".join(ss.get("why_not") or ["-"]),
            "+".join(s["side"] for s in sc["shoes"] if s["on_side"]) or "0", 1000 * sc["batch"]["length"],
            1000 * sc["batch"]["width"], sc["colours"]["upper"], sc["colours"]["sole"], sc["colours"]["lace"],
            sc["lid"]["rest"], r["sim_time_s"], r["slips"], r["wall_s"]))
    ss = [c["result"].get("start_state") or {} for c in every]
    if all(ss):
        print("\nstart state (all candidates): lid inside the rim {:.1f}-{:.1f} mm, tilt <= {:.2f} deg, rim contact <= {:.3f} N; "
              "largest move in {:.0f} s idle {:.2f} mm; contacts deeper than {:.0f} mm in {} candidates".format(
                  min(s["lid_inside_rim_mm"] for s in ss), max(s["lid_inside_rim_mm"] for s in ss),
                  max(s["lid_tilt_deg"] for s in ss), max(s["lid_rim_contact_N"] for s in ss), START_IDLE_S,
                  max(max(s["moved_mm"].values()) for s in ss), START_DEEP_MM, sum(1 for s in ss if s["deep"])))
    print("\nfeature | " + " | ".join(name for name, _ in groups))
    num = (("shoe length (m)", lambda s: s["batch"]["length"]), ("shoe width (m)", lambda s: s["batch"]["width"]),
           ("row spacing (m)", lambda s: s["batch"]["row_spacing"]), ("throat gap (m)", lambda s: s["batch"]["throat_gap"]),
           ("box L (m)", lambda s: s["carton"]["L"]), ("box W (m)", lambda s: s["carton"]["W"]),
           ("box H (m)", lambda s: s["carton"]["H"]), ("cradle dx (m)", lambda s: s["jitter"]["cradle"][0]),
           ("cradle dy (m)", lambda s: s["jitter"]["cradle"][1]), ("cradle dyaw (rad)", lambda s: s["jitter"]["cradle"][2]),
           ("light 0", lambda s: s["jitter"]["light"][0]), ("lid y (m)", lambda s: s["lid"]["pos"][1]))
    for name, fn in num:
        cells = []
        for _, cs in groups:
            v = [fn(c["result"]["scene"]) for c in cs]
            cells.append("{:.4f}..{:.4f}".format(min(v), max(v)) if v else "-")
        print("{} | {}".format(name, " | ".join(cells)))
    cat = (("side-lying per pair", lambda s: [str(s["side_lying"])]),
           ("side-lying shoe", lambda s: [x["side"] for x in s["shoes"] if x["on_side"]] or ["none"]),
           ("upper colour", lambda s: [s["colours"]["upper"]]), ("sole colour", lambda s: [s["colours"]["sole"]]),
           ("lace colour", lambda s: [s["colours"]["lace"]]), ("lid rest", lambda s: [s["lid"]["rest"]]),
           ("shoe way round", lambda s: [x["way"] for x in s["shoes"]]), ("blank way round", lambda s: [s["blank"]["way"]]))
    for name, fn in cat:
        cells = []
        for _, cs in groups:
            counts = {}
            for c in cs:
                for v in fn(c["result"]["scene"]):
                    counts[v] = counts.get(v, 0) + 1
            cells.append(", ".join("{} {}".format(k, counts[k]) for k in sorted(counts)) or "-")
        print("{} | {}".format(name, " | ".join(cells)))
    h = hollowing(kept, dropped, after)
    print("\nhollow: {} (side-lying kinds: all {}, kept {}; missing {} {})".format(
        h["hollow"], h["side_lying_kinds_all"], h["side_lying_kinds_kept"], h["missing"], h["missing_cause"]))
    for c in dropped:
        print("drop draw {} seed {}: {}".format(c["draw"], c["seed"], why(c["result"])))
    for c in after:
        if not c["result"]["passes"]:
            print("after the cutoff, draw {} seed {}: {}".format(c["draw"], c["seed"], why(c["result"])))
    ran = [c["result"] for c in every]
    print("\nall {} candidates: {} pass, mean work_score {:.4f}, voids {}, slips {}, sim {:.0f}-{:.0f} s, wall {:.0f}-{:.0f} s".format(
        len(ran), sum(r["passes"] for r in ran), sum(r["work_score"] for r in ran) / len(ran),
        sum(not r["safety_ok"] for r in ran), sum(r["slips"] for r in ran), min(r["sim_time_s"] for r in ran),
        max(r["sim_time_s"] for r in ran), min(r["wall_s"] for r in ran), max(r["wall_s"] for r in ran)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--image", required=True)
    p.set_defaults(fn=cmd_freeze)
    p = sub.add_parser("draw")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--count", type=int, default=0)
    p.set_defaults(fn=cmd_draw)
    p = sub.add_parser("run-one")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--selection", default=None, help="check the frozen files against this record")
    p.add_argument("--task", default=None, help="a copy of the task folder (solution/, environment/) for the check")
    p.add_argument("--dev", action="store_true", help="record a frozen-file difference instead of refusing")
    p.set_defaults(fn=cmd_run_one)
    p = sub.add_parser("collect")
    p.add_argument("--results", required=True)
    p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("make")
    p.add_argument("--size", type=int, default=SET_SIZE)
    p.set_defaults(fn=cmd_make)
    p = sub.add_parser("report")
    p.add_argument("--size", type=int, default=SET_SIZE)
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
