"""Gates of the duct model: procedures and lines, fixed before the gate runs.

Impact protocol (every impact number of this package): one straight duct of 15 pieces of 0.508 m on a flat floor of
grip g; the drive model's car (default tyre model, drive_params.json) on y = 0 heading +x, steering 0, speed target v from rest
with a run-up of 0.6 + 0.35 v^2 m to the duct's skin; the duct's axis makes the incidence angle with the car's
heading (90 deg = head on) and crosses the car's path at the hit station (7.5 piece lengths from its tail = the
centre of piece 7, unless stated). Protocol B0: at the first physics step with a contact between a car geom and a
duct collision geom the speed target becomes 0 (the drive model's full brake), steering stays 0; protocol B50: the same 50 ms
(20 steps) later. The run goes on 1.5 s after the contact (longer, up to 3 s, while a piece still moves faster than
1 mm/s). Measured: the car's ground speed in the step before the contact (`v_contact`), the largest displacement of
any piece at any physics step (`dmax`, DuctTracker), the verdict crash = dmax >= 0.05 m, the largest
pin stretch, the final state's sha256 (qpos || qvel).

Lines (pre-registered; never moved after a result):
  G1 impact table: head on / 30 deg / 10 deg x v_cmd 0.3 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0 m/s x grip 0.5 0.7 0.9, B0;
     the same at grip 0.7 with B50. The 5 cm line `v50` (linear interpolation of dmax over v_contact) head on at
     grip 0.7 (B0) must lie in [0.80, 1.25] m/s ("near 1 m/s"). Other cells are reported.
  G2 sensitivity: mass_per_m_kg and floor_friction x0.5 and x1.5 (one at a time), head on, grip 0.7, B0, v_cmd 0.5 ..
     2.0: v50 and dmax at 1.0 m/s against the base set. Reported, no pass line (bending stiffness x0.5 / x1.5 and
     the car-duct contact softness are reported as extra rows).
  G3 cost: `stadium_world(902)` (902 pieces in 61 ducts, 230 m lap), the car lapping by pure pursuit on the true
     pose at 4.5 m/s, 20 s after a 2 s warm-up: median over physics steps of car.pre_step + mj_step +
     DuctTracker.update <= 160 us on the selection host (load as found, nice 10, one thread).
     Reported besides: the cell's kinematics refresh (the cell calls mj_kinematics, mj_comPos, mj_comVel after every
     step), p25, the same on a second machine, every mode, and a near-wall variant (pursuit line 0.6 m off centre).
  G4 same-machine determinism: the 20 cases of `DET_CASES` twice in one process and once in a fresh process give
     the same final-state sha256 and the same dmax bit for bit, in every mode.
  G5 cross-machine verdicts: the 20 cases give the same touch / crash verdict on this host and on two further machines with other CPUs in mode `sleep`; if not, the default becomes `active_window`. Hashes and dmax are compared and reported for every mode.
  G6 put-back: after a 3 m/s head-on crash, put_back(displaced(0.05)) leaves every end point of the re-laid duct within
     0.03 m of the world file (200 draws of the put-back error), the duct's displacement from its new reference is
     below 1 mm 1 s later, and a LiDAR ray finds the re-laid piece where the tracker says it is (1 mm); every mode.

Run (from the folder that holds sketch/):
    python -m sketch.ducts.gates --all --write          # G1 .. G4, G6 on this host, writes the tables beside this file
    python -m sketch.ducts.gates --det --out FILE       # G4 / G5 cases only (what the other machines run)
    python -m sketch.ducts.gates --merge-det FILE ...   # G5: merge the machines' files into determinism_table.json
"""

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time

import mujoco
import numpy as np

from sketch.examples import constants as K

from . import scenes
from .params import load_params, with_overrides

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = 0.05
SPEEDS = (0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
ANGLES = (90, 30, 10)
GRIPS = (0.5, 0.7, 0.9)
V50_LINE = (0.80, 1.25)
COST_LINE_US = 160.0
MODES = ("active_window", "sleep", "awake")
REACT_B50 = 20                       # physics steps (50 ms at 2.5 ms)

# the 20 cases of G4 / G5: near the 5 cm line, where a verdict can flip, plus two far from it
DET_CASES = (
    [{"angle": 90, "v": v, "grip": 0.7} for v in (0.9, 1.0, 1.1, 1.2, 1.3)]
    + [{"angle": 90, "v": 1.1, "grip": g} for g in (0.5, 0.9)]
    + [{"angle": 30, "v": v, "grip": 0.7} for v in (1.3, 1.5, 1.7)]
    + [{"angle": 10, "v": v, "grip": 0.7} for v in (2.4, 2.7, 3.0)]
    + [{"angle": 90, "v": 1.1, "grip": 0.7, "hit": h} for h in (7.0, 1.0, 14.5)]
    + [{"angle": 90, "v": v, "grip": 0.7, "react": REACT_B50} for v in (0.8, 1.0)]
    + [{"angle": 90, "v": v, "grip": 0.7} for v in (0.5, 3.0)]
)


def env_info():
    info = {"host": socket.gethostname(), "python": platform.python_version(), "mujoco": mujoco.__version__,
            "numpy": np.__version__, "load1": round(os.getloadavg()[0], 1)}
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu"] = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        info["cpu"] = platform.processor()
    info["libc"] = " ".join(platform.libc_ver())
    return info


def state_hash(data):
    return hashlib.sha256(np.asarray(data.qpos).tobytes() + np.asarray(data.qvel).tobytes()).hexdigest()


def _params(mode, params=None, **over):
    p = params if params is not None else load_params()
    return with_overrides(p, mode=mode, **over)


# ------------------------------------------------------------------------------------------------ one impact
def impact_case(angle, v, grip=0.7, hit=7.5, react=0, mode="active_window", params=None, t_after=1.5, t_max_after=3.0):
    """One impact by the protocol of the module text; returns its measurements."""
    p = _params(mode, params)
    w = scenes.impact_world(angle, hit_piece=hit, base_grip=grip)
    pose = scenes.impact_car_pose(angle, v)
    m, d, car, ducts = scenes.compile_with_car(w, pose, params=p, mode=mode, grip=grip)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
    is_car = np.array([n.startswith(K.EGO_PREFIX) for n in names])
    is_duct = np.array([(K.geom_rule(n) or (None, None))[1] == "duct" for n in names])
    car.set_targets(0.0, v)
    dt = m.opt.timestep
    t_contact = v_contact = brake_at = None
    dmax, step, v_prev = 0.0, 0, 0.0
    n_max = int((6.0 + 0.5 * v + t_max_after) / dt)
    gap = 0.0
    while step < n_max:
        v_prev = car.state()["speed"]
        car.pre_step()
        mujoco.mj_step(m, d)
        step += 1
        dmax = max(dmax, ducts.update())
        if t_contact is None:
            for i in range(d.ncon):
                c = d.contact[i]
                if (is_car[c.geom1] and is_duct[c.geom2]) or (is_car[c.geom2] and is_duct[c.geom1]):
                    t_contact, v_contact, brake_at = d.time, v_prev, step + int(react)
                    break
        if brake_at is not None and step >= brake_at:
            car.set_targets(0.0, 0.0)
        if t_contact is not None:
            gap = max(gap, float(ducts.joint_gaps().max()))
            after = d.time - t_contact
            if after >= t_after:
                moving = _piece_speed(ducts)
                if moving < 1e-3 or after >= t_max_after:
                    break
    disp = ducts.displacements()
    out = {"angle_deg": angle, "v_cmd_mps": v, "grip": grip, "hit_station_pieces": hit, "react_steps": int(react),
           "mode": mode, "v_contact_mps": None if v_contact is None else round(v_contact, 6),
           "t_contact_s": None if t_contact is None else round(t_contact, 6), "dmax_m": float(dmax),
           "d_final_m": float(disp.max()), "crash": bool(dmax >= THRESHOLD), "piece_of_max": int(np.argmax(disp)),
           "pin_stretch_max_m": gap, "car_speed_end_mps": round(car.state()["speed"], 6), "sim_s": round(d.time, 4),
           "state_sha256": state_hash(d)}
    if ducts.window is not None:
        out["window"] = ducts.window.stats()
    return out


def _piece_speed(ducts):
    """Largest joint speed of any dynamic duct piece now (m/s or rad/s)."""
    d = ducts.data
    if ducts.window is not None:
        v = 0.0
        for j, si in ducts.window.active.items():
            s = ducts.window.slots[si]
            n = ducts.slices[j].stop - ducts.slices[j].start
            v = max(v, float(np.abs(d.qvel[s.vidx[:n].reshape(-1)]).max()))
        return v
    if hasattr(ducts, "vidx"):
        return float(np.abs(d.qvel[ducts.vidx]).max())
    return 0.0


def v50(rows):
    """Speed at which dmax first reaches the threshold, by linear interpolation over v_contact; None if never."""
    pts = sorted((r["v_contact_mps"], r["dmax_m"]) for r in rows if r["v_contact_mps"] is not None)
    for i, (vv, dd) in enumerate(pts):
        if dd >= THRESHOLD:
            if i == 0:
                return {"v50_mps": None, "below_grid": True}
            v0, d0 = pts[i - 1]
            return {"v50_mps": round(v0 + (THRESHOLD - d0) * (vv - v0) / (dd - d0), 4), "below_grid": False}
    return {"v50_mps": None, "above_grid": True}


# ------------------------------------------------------------------------------------------------ G1, G2
def impact_table(mode="active_window", speeds=SPEEDS, angles=ANGLES, grips=GRIPS, log=print):
    rows = []
    for g in grips:
        for a in angles:
            for v in speeds:
                rows.append(impact_case(a, v, grip=g, mode=mode))
                log("G1 B0 grip %.1f angle %d v %.2f: v_contact %.3f dmax %.4f"
                    % (g, a, v, rows[-1]["v_contact_mps"] or -1, rows[-1]["dmax_m"]))
    for a in angles:
        for v in speeds:
            rows.append(impact_case(a, v, grip=0.7, react=REACT_B50, mode=mode))
            log("G1 B50 angle %d v %.2f: dmax %.4f" % (a, v, rows[-1]["dmax_m"]))
    positions = []
    for h in (7.5, 7.0, 3.5, 1.0, 0.5, 14.5):
        positions.append(impact_case(90, 1.0, grip=0.7, hit=h, mode=mode))
        log("G1 position %.1f: dmax %.4f" % (h, positions[-1]["dmax_m"]))
    lines = {}
    for g in grips:
        for a in angles:
            sel = [r for r in rows if r["grip"] == g and r["angle_deg"] == a and r["react_steps"] == 0]
            lines["B0 grip %.1f angle %d" % (g, a)] = v50(sel)
    for a in angles:
        sel = [r for r in rows if r["grip"] == 0.7 and r["angle_deg"] == a and r["react_steps"] == REACT_B50]
        lines["B50 grip 0.7 angle %d" % a] = v50(sel)
    head = lines["B0 grip 0.7 angle 90"]["v50_mps"]
    gate = {"line": "v50 head on, grip 0.7, B0 in [%.2f, %.2f] m/s" % V50_LINE, "v50_mps": head,
            "pass": head is not None and V50_LINE[0] <= head <= V50_LINE[1]}
    return {"protocol": __doc__.split("Lines")[0].strip(), "mode": mode, "rows": rows, "positions": positions,
            "v50": lines, "gate_G1": gate}


def sensitivity_table(mode="active_window", speeds=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0), log=print):
    base = load_params()
    variants = [("base", {})]
    for key in ("mass_per_m_kg", "floor_friction"):
        for f in (0.5, 1.5):
            variants.append(("%s x%.1f" % (key, f), {key: base[key] * f}))
    for f in (0.5, 1.5):
        variants.append(("bend_stiffness_nm_per_rad x%.1f (extra)" % f,
                         {"bend_stiffness_nm_per_rad": base["bend_stiffness_nm_per_rad"] * f}))
    for tc in (0.01, 0.04):
        c = dict(base["contact"])
        c["solref"] = [tc, 1.0]
        variants.append(("contact solref %.2f s (extra)" % tc, {"contact": c}))
    out = []
    for label, over in variants:
        p = with_overrides(base, mode=mode, **over)
        rows = [impact_case(90, v, grip=0.7, mode=mode, params=p) for v in speeds]
        at1 = next(r for r in rows if abs(r["v_cmd_mps"] - 1.0) < 1e-9)
        res = {"variant": label, "overrides": over, "v50": v50(rows), "dmax_at_1mps_m": at1["dmax_m"],
               "rows": [{k: r[k] for k in ("v_cmd_mps", "v_contact_mps", "dmax_m", "crash")} for r in rows]}
        out.append(res)
        log("G2 %s: v50 %s dmax@1 %.4f" % (label, res["v50"], res["dmax_at_1mps_m"]))
    b = out[0]
    for r in out:
        if r["v50"]["v50_mps"] and b["v50"]["v50_mps"]:
            r["v50_ratio_to_base"] = round(r["v50"]["v50_mps"] / b["v50"]["v50_mps"], 4)
        r["dmax_at_1mps_ratio_to_base"] = round(r["dmax_at_1mps_m"] / b["dmax_at_1mps_m"], 4)
    return {"mode": mode, "variants": out}


# ------------------------------------------------------------------------------------------------ G3
def lap_cost(mode="active_window", n_pieces=902, seconds=20.0, warmup=2.0, offset_m=0.0, speed=4.5, params=None,
             slots=None):
    p = _params(mode, params, **({"window__slots": slots} if slots else {}))
    w = scenes.stadium_world(n_pieces)
    m, d, car, ducts = scenes.compile_with_car(w, w["truth"]["start_pose"]["pose"], params=p, mode=mode)
    pp = scenes.Pursuit(w, speed_mps=speed, offset_m=offset_m)
    n0, n = int(warmup / m.opt.timestep), int((warmup + seconds) / m.opt.timestep)
    T = np.zeros((n, 4))
    dmax = 0.0
    act = 0
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
    is_car = np.array([x.startswith(K.EGO_PREFIX) for x in names])
    is_duct = np.array([(K.geom_rule(x) or (None, None))[1] == "duct" for x in names])
    touch_steps = 0
    dist = 0.0
    pc = time.perf_counter
    for i in range(n):
        if i % 10 == 0:
            s, v = pp.command(car.state()["pose"])
            car.set_targets(s, v)
        t0 = pc()
        car.pre_step()
        t1 = pc()
        mujoco.mj_step(m, d)
        t2 = pc()
        dm = ducts.update()
        t3 = pc()
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        t4 = pc()
        T[i] = [t1 - t0, t2 - t1, t3 - t2, t4 - t3]
        dmax = max(dmax, dm)
        act = max(act, ducts.awake_pieces())
        if i >= n0:
            dist += car.state()["speed"] * m.opt.timestep
        for k in range(d.ncon):
            c = d.contact[k]
            if (is_car[c.geom1] and is_duct[c.geom2]) or (is_car[c.geom2] and is_duct[c.geom1]):
                touch_steps += 1
                break
    T = T[n0:] * 1e6
    own = T[:, 0] + T[:, 1] + T[:, 2]
    res = {"mode": mode, "n_pieces": sum(len(x["pieces"]) for x in w["truth"]["ducts"]), "nv": int(m.nv),
           "offset_m": offset_m, "speed_mps": speed, "sim_s": seconds,
           "us_per_step": {"pre_step_med": round(float(np.median(T[:, 0])), 2),
                           "mj_step_med": round(float(np.median(T[:, 1])), 2),
                           "update_med": round(float(np.median(T[:, 2])), 2), "refresh_med": round(float(np.median(T[:, 3])), 2),
                           "gate_sum_med": round(float(np.median(own)), 2),
                           "gate_sum_p25": round(float(np.percentile(own, 25)), 2),
                           "gate_sum_p90": round(float(np.percentile(own, 90)), 2),
                           "with_refresh_med": round(float(np.median(own + T[:, 3])), 2)},
           "distance_m": round(dist, 2), "dmax_m": dmax, "max_dynamic_pieces": act, "touch_steps": touch_steps,
           "load1": round(os.getloadavg()[0], 1)}
    if ducts.window is not None:
        res["window"] = ducts.window.stats()
    return res


def cost_table(log=print, modes=MODES, seconds=20.0):
    rows = []
    for mode in modes:
        r = lap_cost(mode, seconds=seconds if mode != "awake" else 6.0)
        rows.append(r)
        log("G3 %s: %s" % (mode, r["us_per_step"]))
    for slots in (2, 4):
        r = lap_cost("active_window", seconds=seconds, slots=slots)
        r["variant"] = "slots %d" % slots
        rows.append(r)
        log("G3 active_window slots %d: %s" % (slots, r["us_per_step"]))
    for mode in ("active_window", "sleep"):
        r = lap_cost(mode, seconds=seconds, offset_m=-0.6)
        r["variant"] = "near the outer wall (pursuit line 0.6 m right of centre)"
        rows.append(r)
        log("G3 %s near wall: %s window %s" % (mode, r["us_per_step"], r.get("window")))
    gate_row = rows[0]
    return {"env": env_info(), "rows": rows,
            "gate_G3": {"line": "median car.pre_step + mj_step + update <= %.0f us, default mode, 902 pieces" % COST_LINE_US,
                        "mode": gate_row["mode"], "value_us": gate_row["us_per_step"]["gate_sum_med"],
                        "pass": gate_row["us_per_step"]["gate_sum_med"] <= COST_LINE_US}}


# ------------------------------------------------------------------------------------------------ G4, G5
def det_cases(modes=MODES, repeat=2, log=print):
    out = {"env": env_info(), "cases": []}
    for i, case in enumerate(DET_CASES):
        for mode in modes:
            runs = []
            for _ in range(repeat):
                r = impact_case(case["angle"], case["v"], grip=case["grip"], hit=case.get("hit", 7.5),
                                react=case.get("react", 0), mode=mode)
                runs.append(r)
            row = {"case": i, "mode": mode, **case, "dmax_m": [r["dmax_m"] for r in runs],
                   "crash": [r["crash"] for r in runs], "state_sha256": [r["state_sha256"] for r in runs],
                   "v_contact_mps": runs[0]["v_contact_mps"]}
            out["cases"].append(row)
            log("case %2d %-13s %s dmax %.6f crash %s same %s" % (i, mode, case, runs[0]["dmax_m"], runs[0]["crash"],
                                                                len(set(row["state_sha256"])) == 1))
    return out


def merge_det(files):
    runs = [json.load(open(f)) for f in files]
    base = runs[0]
    rows = []
    for k, row in enumerate(base["cases"]):
        entry = {"case": row["case"], "mode": row["mode"], "angle": row["angle"], "v": row["v"], "grip": row["grip"],
                 "hit": row.get("hit", 7.5), "react": row.get("react", 0), "machines": {}}
        for r in runs:
            rr = r["cases"][k]
            assert rr["case"] == row["case"] and rr["mode"] == row["mode"]
            entry["machines"][r["env"]["host"]] = {"dmax_m": rr["dmax_m"][0], "crash": rr["crash"][0],
                                                   "sha256": rr["state_sha256"][0],
                                                   "repeat_identical": (len(set(rr["state_sha256"])) == 1
                                                                        and len(set(rr["dmax_m"])) == 1)}
        ms = list(entry["machines"].values())
        entry["same_verdict"] = len({x["crash"] for x in ms}) == 1
        entry["same_hash"] = len({x["sha256"] for x in ms}) == 1
        entry["same_dmax"] = len({x["dmax_m"] for x in ms}) == 1
        rows.append(entry)
    summary = {}
    for mode in sorted({r["mode"] for r in rows}):
        sel = [r for r in rows if r["mode"] == mode]
        summary[mode] = {"cases": len(sel), "same_verdict": sum(r["same_verdict"] for r in sel),
                         "same_hash": sum(r["same_hash"] for r in sel), "same_dmax": sum(r["same_dmax"] for r in sel),
                         "crashes": sum(r["machines"][list(r["machines"])[0]]["crash"] for r in sel),
                         "repeat_identical_everywhere": all(m["repeat_identical"] for r in sel for m in r["machines"].values())}
    sleep_ok = summary.get("sleep", {}).get("same_verdict") == summary.get("sleep", {}).get("cases")
    return {"machines": [r["env"] for r in runs], "summary": summary, "rows": rows,
            "gate_G5": {"line": "20 cases, same verdict on every machine, mode sleep", "pass": bool(sleep_ok)}}


def same_process_vs_fresh(modes=MODES, log=print):
    """G4's third run: every case once more in a fresh interpreter; compares with this process."""
    here = det_cases(modes=modes, repeat=2, log=lambda *a: None)
    cmd = [sys.executable, "-B", "-m", "sketch.ducts.gates", "--det", "--repeat", "1", "--out", "-", "--modes"] + list(modes)
    env = dict(os.environ)
    res = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(HERE)), capture_output=True, text=True, env=env, check=True)
    fresh = json.loads(res.stdout.strip().splitlines()[-1])
    rows = []
    for a, b in zip(here["cases"], fresh["cases"]):
        same = len(set(a["state_sha256"] + b["state_sha256"])) == 1 and len(set(a["dmax_m"] + b["dmax_m"])) == 1
        rows.append({"case": a["case"], "mode": a["mode"], "identical": same, "sha256": a["state_sha256"][0][:16]})
    ok = all(r["identical"] for r in rows)
    log("G4: %d / %d identical" % (sum(r["identical"] for r in rows), len(rows)))
    return {"env": env_info(), "rows": rows, "gate_G4": {"line": "two runs in one process + one in a fresh process identical",
                                                         "pass": ok}}


# ------------------------------------------------------------------------------------------------ G6
def restore_gate(mode="active_window", draws=200, log=print):
    from . import geom
    p = _params(mode)
    w = scenes.impact_world(90, hit_piece=7.5, base_grip=0.7)
    pose = scenes.impact_car_pose(90, 3.0)
    m, d, car, ducts = scenes.compile_with_car(w, pose, params=p, mode=mode, grip=0.7)
    car.set_targets(0.0, 3.0)
    contact = False
    for i in range(int(4.0 / m.opt.timestep)):
        car.pre_step()
        mujoco.mj_step(m, d)
        dm = ducts.update()
        if dm > 0.0 and not contact:
            contact = True
            car.set_targets(0.0, 0.0)
    dmax_crash = float(ducts.displacements().max())
    pieces = ducts.displaced(THRESHOLD)
    # the staff: car away, ducts re-laid
    car.place([-3.0, -2.0, 0.0])
    car.set_targets(0.0, 0.0)
    ducts.put_back(pieces)
    mujoco.mj_forward(m, d)
    tail0, head0 = geom.ends(ducts.world_poses, ducts.lengths)
    t, h = geom.ends(ducts.poses(), ducts.lengths)
    err_first = float(max(np.linalg.norm(t - tail0, axis=1).max(), np.linalg.norm(h - head0, axis=1).max()))
    for i in range(int(1.0 / m.opt.timestep)):
        car.pre_step()
        mujoco.mj_step(m, d)
        ducts.update()
    mujoco.mj_forward(m, d)
    drift = float(ducts.displacements().max())
    # a LiDAR ray at piece 7's cylinder from the side: hit distance against the tracker's pose
    k = 7
    px, py, pyaw = ducts.poses()[k]
    nx, ny = -math.sin(pyaw), math.cos(pyaw)
    origin = np.array([px - 2.0 * nx, py - 2.0 * ny, K.DUCT_RADIUS_M])
    gid = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(m, d, origin, np.array([nx, ny, 0.0]), np.array(K.LIDAR_GROUPMASK, np.uint8), 1, -1, gid)
    ray_err = abs(float(dist) - (2.0 - K.DUCT_RADIUS_M))
    ray_geom = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0]))
    # many draws of the put-back error (the error model alone, with the tracker's own generator)
    errs = []
    for i in range(draws):
        ducts.put_back(None)
        t, h = geom.ends(ducts.poses(), ducts.lengths)
        errs.append(float(max(np.linalg.norm(t - tail0, axis=1).max(), np.linalg.norm(h - head0, axis=1).max())))
    ok = (max(errs) <= 0.03 + 1e-12 and err_first <= 0.03 + 1e-12 and drift < 1e-3 and ray_err < 1e-3
          and ray_geom == K.duct_geom("D0", k, "lidar"))
    res = {"mode": mode, "dmax_crash_m": dmax_crash, "pieces_displaced": len(pieces), "err_first_m": err_first,
           "drift_after_1s_m": drift, "ray_error_m": ray_err, "ray_geom": ray_geom,
           "draws": draws, "err_max_m": max(errs), "err_mean_m": float(np.mean(errs)), "pass": bool(ok)}
    log("G6 %s: %s" % (mode, {k: v for k, v in res.items() if k != "mode"}))
    return res


def creep_check(mode="active_window", seconds=60.0):
    """A 1 m/s head-on touch, then the car parked away for `seconds`: drift of the touched duct afterwards."""
    p = _params(mode)
    w = scenes.impact_world(90, hit_piece=7.5, base_grip=0.7)
    m, d, car, ducts = scenes.compile_with_car(w, scenes.impact_car_pose(90, 1.0), params=p, mode=mode, grip=0.7)
    car.set_targets(0.0, 1.0)
    hit = False
    for i in range(int(4.0 / m.opt.timestep)):
        car.pre_step()
        mujoco.mj_step(m, d)
        if ducts.update() > 0 and not hit:
            hit = True
            car.set_targets(0.0, 0.0)
    after_touch = ducts.displacements().copy()
    car.place([-4.0, -3.0, 0.0])
    for i in range(int(seconds / m.opt.timestep)):
        car.pre_step()
        mujoco.mj_step(m, d)
        ducts.update()
    end = ducts.displacements()
    return {"mode": mode, "dmax_after_touch_m": float(after_touch.max()), "dmax_after_rest_m": float(end.max()),
            "drift_max_m": float(np.abs(end - after_touch).max()), "rest_s": seconds}


# ------------------------------------------------------------------------------------------------ command line
def _write(name, obj):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
        f.write("\n")
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="S4 gates")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--impact", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--det", action="store_true", help="the 20 cases (G4 / G5)")
    ap.add_argument("--g4", action="store_true", help="same-machine determinism incl. a fresh process")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--creep", action="store_true")
    ap.add_argument("--merge-det", nargs="+")
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if a.merge_det:
        res = merge_det(a.merge_det)
        print(json.dumps(res["summary"], indent=1), res["gate_G5"])
        if a.write:
            _write("determinism_table.json", res)
        return 0
    if a.det:
        quiet = a.out == "-"
        res = det_cases(modes=a.modes, repeat=a.repeat, log=(lambda *x: None) if quiet else print)
        if a.out == "-":
            print(json.dumps(res))
        elif a.out:
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1)
            print("wrote", a.out)
        return 0
    if a.all or a.impact:
        t0 = time.time()
        res = impact_table()
        res["env"] = env_info()
        res["wall_s"] = round(time.time() - t0, 1)
        print("G1", res["gate_G1"], json.dumps(res["v50"]))
        if a.write:
            _write("impact_table.json", res)
    if a.all or a.sensitivity:
        res = sensitivity_table()
        res["env"] = env_info()
        if a.write:
            _write("sensitivity_table.json", res)
    if a.all or a.cost:
        res = cost_table()
        print("G3", res["gate_G3"])
        if a.write:
            _write("cost_table.json", res)
    if a.all or a.g4:
        res = same_process_vs_fresh(modes=a.modes)
        print("G4", res["gate_G4"])
        if a.write:
            _write("determinism_same_machine.json", res)
    if a.all or a.restore or a.creep:
        res = {"env": env_info(), "restore": [restore_gate(md) for md in MODES],
               "creep": [creep_check(md) for md in MODES]}
        res["gate_G6"] = {"line": "put-back error <= 0.03 m, drift < 1 mm, ray within 1 mm, every mode",
                          "pass": all(r["pass"] for r in res["restore"])}
        print("G6", res["gate_G6"], [(c["mode"], c["drift_max_m"]) for c in res["creep"]])
        if a.write:
            _write("restore_table.json", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
