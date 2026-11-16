"""The privileged reference's arm postures over a whole episode (tests/review/posture.md). Runs one episode the way
run_ref.py does (CellServer privileged, the four camera frames, reference_clean, finalize), inside the task image,
and samples both arms every 0.2 s of simulated time. Usage: posture_run.py STAGE SEED OUT.json

A sample of an arm is unnatural when (torso frame; the shoulder is the shoulder_roll_link's origin, the elbow the
elbow_link's, the upper arm the line between them):
  elbow_high   the elbow less than 5 cm below the shoulder (at or above it: the arm raised over the head);
  limit        a joint within 0.10 rad of its limit;
  out          the upper arm swung out to the side more than 75 deg from hanging down (in the frontal plane);
  back         the upper arm swung back more than 40 deg (in the sagittal plane).
The finer measure of round 4 (posture.md; "unnatural2": any of these or the four above):
  swivel       the elbow more than 55 deg from its natural direction about the shoulder-wrist line (down and a
               little out: the "chicken wing", or the elbow into the body), the arm bent more than 25 deg (the
               shoulder the shoulder_pitch_link's origin, the wrist the wrist_roll_link's);
  hyperext     the elbow straightened past straight (joint above pi/2 + 0.08);
  wrist_bent   wrist pitch or yaw beyond 0.95 rad, or wrist roll beyond 1.55 rad;
  twist        shoulder yaw beyond 1.45 rad;
  flip         a wrist joint turning by more than 1.2 rad within 0.6 s.
"""
import json, math, os, sys, time
import numpy as np

stage, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, stage)
sys.dont_write_bytecode = True
import mujoco
from house_cell.server import CellServer
from house_cell.robot import ARM_JOINTS

EVERY = 20                      # ticks (0.2 s)
cfg = json.load(open(os.path.join(stage, "grader_config.json")))
t0 = time.time()
srv = CellServer(seed, privileged=True, nominal=False, config=cfg, log=lambda s: None, render=True)
cell = srv.cell
m, d, r = cell.m, cell.d, cell.robot
B = mujoco.mjtObj.mjOBJ_BODY
SH = {s: mujoco.mj_name2id(m, B, f"{r.name}/{s}_shoulder_roll_link") for s in ("left", "right")}
EL = {s: mujoco.mj_name2id(m, B, f"{r.name}/{s}_elbow_link") for s in ("left", "right")}
OUT = {"left": 1.0, "right": -1.0}
SP = {s: mujoco.mj_name2id(m, B, f"{r.name}/{s}_shoulder_pitch_link") for s in ("left", "right")}
WR = {s: mujoco.mj_name2id(m, B, f"{r.name}/{s}_wrist_roll_link") for s in ("left", "right")}
FLAGS = ("elbow_high", "limit", "out", "back")
FLAGS2 = ("swivel", "hyperext", "wrist_bent", "twist", "flip")
sec2 = {s: {f: 0.0 for f in FLAGS2} for s in ("left", "right")}
either2 = [0.0]
hist = {s: [] for s in ("left", "right")}
sec = {s: {f: 0.0 for f in FLAGS} for s in ("left", "right")}
sec_any = {f: 0.0 for f in FLAGS}
either = [0.0]
worst = {s: {"el_dz_max": -1.0, "margin_min": 9.0} for s in ("left", "right")}
dt = EVERY * 0.01


def sample():
    Rt = d.xmat[r.torso].reshape(3, 3)
    fl_any = set()
    fl2_any = set()
    for s in ("left", "right"):
        q = d.qpos[r.arm_q[s]]
        margin = float(np.min(np.minimum(q - r.arm_lo[s], r.arm_hi[s] - q)))
        u = Rt.T @ (d.xpos[EL[s]] - d.xpos[SH[s]])
        el_dz = float(d.xpos[EL[s]][2] - d.xpos[SH[s]][2])
        n = u / max(1e-9, float(np.linalg.norm(u)))
        out_deg = math.degrees(math.atan2(OUT[s] * n[1], -n[2]))
        back_deg = math.degrees(math.atan2(-n[0], -n[2]))
        fl = set()
        if el_dz > -0.05:
            fl.add("elbow_high")
        if margin < 0.10:
            fl.add("limit")
        if out_deg > 75.0:
            fl.add("out")
        if back_deg > 40.0:
            fl.add("back")
        for f in fl:
            sec[s][f] += dt
        fl_any |= fl
        # the finer measure
        S_, E_, W_ = Rt.T @ d.xpos[SP[s]], Rt.T @ d.xpos[EL[s]], Rt.T @ d.xpos[WR[s]]
        nv = W_ - S_
        dn = float(np.linalg.norm(nv))
        nv = nv / dn
        e = (E_ - S_) - float((E_ - S_) @ nv) * nv
        want = np.array([0.0, 0.3 * OUT[s], -1.0])
        w = want - float(want @ nv) * nv
        if np.linalg.norm(w) < 0.2:
            want = np.array([0.0, OUT[s], -0.3])
            w = want - float(want @ nv) * nv
        ue, uw = e / max(1e-9, float(np.linalg.norm(e))), w / float(np.linalg.norm(w))
        dev = math.degrees(math.acos(float(np.clip(ue @ uw, -1, 1))))
        a_, b_ = float(np.linalg.norm(E_ - S_)), float(np.linalg.norm(W_ - E_))
        bend = 180.0 - math.degrees(math.acos(float(np.clip((a_ * a_ + b_ * b_ - dn * dn) / (2 * a_ * b_), -1, 1))))
        f2 = set()
        if bend > 25.0 and dev > 55.0:
            f2.add("swivel")
        if q[3] > math.pi / 2 + 0.08:
            f2.add("hyperext")
        if abs(q[5]) > 0.95 or abs(q[6]) > 0.95 or abs(q[4]) > 1.55:
            f2.add("wrist_bent")
        if abs(q[2]) > 1.45:
            f2.add("twist")
        hist[s].append(np.asarray(q, float).copy())
        if len(hist[s]) > 4 and float(np.max(np.abs(hist[s][-1][4:] - hist[s][-4][4:]))) > 1.2:
            f2.add("flip")
        hist[s] = hist[s][-4:]
        for f in f2:
            sec2[s][f] += dt
        fl2_any.update(f2 | fl)
        worst[s]["el_dz_max"] = max(worst[s]["el_dz_max"], el_dz)
        worst[s]["margin_min"] = min(worst[s]["margin_min"], margin)
    for f in fl_any:
        sec_any[f] += dt
    if fl_any:
        either[0] += dt
    if fl2_any:
        either2[0] += dt


orig = cell._tick


def tick(record=True):
    orig(record)
    if record and cell.ticks % EVERY == 0:
        sample()


cell._tick = tick
for cam in ("ceiling_kitchen", "ceiling_dining", "ceiling_living", "head"):
    srv.handle("get_camera", {"name": cam})
srv.handle("reference_clean", {})
rec = srv.finalize()
T = float(rec["sim_time_s"])
res = {"seed": seed, "stage_score": rec["stage_score"], "safety_ok": rec["safety_ok"], "sim_time_s": T,
       "credit_by_part": rec["credit_by_part"], "damage": rec.get("damage"),
       "unnatural_s": round(either[0], 1), "unnatural_fraction": round(either[0] / T, 4),
       "by_flag_s": {f: round(v, 1) for f, v in sec_any.items()},
       "per_arm_s": {s: {f: round(v, 1) for f, v in sec[s].items()} for s in sec},
       "worst": worst, "wall_s": round(time.time() - t0, 1),
       "unnatural2_s": round(either2[0], 1), "unnatural2_fraction": round(either2[0] / T, 4),
       "per_arm2_s": {s: {f: round(v, 1) for f, v in sec2[s].items()} for s in sec2}}
json.dump(res, open(out, "w"), indent=1, default=str)
print("seed %d score %.4f unnatural %.0f s of %.0f (%.1f %%) %s; finer: %.1f %% %s" % (
    seed, rec["stage_score"], either[0], T, 100 * either[0] / T, res["by_flag_s"], 100 * either2[0] / T,
    res["per_arm2_s"]))
