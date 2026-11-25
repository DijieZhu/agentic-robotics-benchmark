"""Health checks of the simulation base: stable rest, long mixed-primitive run, timestep study (displacement,
heading and peak line tension against the timestep, with the same-dt perturbation floor), determinism (state
hashes), real-time factor on one core.
The settle/validity ideas follow an earlier tensegrity kit that is not published (a settle detector, a state
hash, a light monitor), re-implemented here for this robot (not imported).

usage: from ysim import health; health.check_rest() / check_long() / check_dt() / scripted_hashes() / check_rtf()
"""
import json
import math
import random
import sys
import time

import mujoco
import numpy as np

from . import gaits
from . import model as mdl
from .sim import Simulator


class Monitor:
    """Light validity monitor: NaN, MuJoCo warnings, penetration depth, speeds, node heights."""

    def __init__(self, sim):
        self.sim = sim
        self.min_dist = 0.0
        self.min_dist_pair = None
        self.max_speed = 0.0
        self.min_node_z = 1e9
        self.nan = False
        self.n = 0

    def update(self):
        s = self.sim
        d = s.d
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            self.nan = True
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist < self.min_dist:
                self.min_dist = float(c.dist)
                self.min_dist_pair = (mujoco.mj_id2name(s.m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                                      mujoco.mj_id2name(s.m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))
        self.max_speed = max(self.max_speed, s.max_node_speed())
        self.min_node_z = min(self.min_node_z, float(s.nodes()[:, 2].min()))
        self.n += 1

    def report(self):
        w = self.sim.d.warning
        return {"nan": self.nan, "warnings": {str(mujoco.mjtWarning(i)).split(".")[-1]: int(w[i].number)
                                              for i in range(len(w)) if w[i].number},
                "max_penetration_mm": -1000.0 * self.min_dist, "worst_pair": self.min_dist_pair,
                "max_node_speed_m_s": self.max_speed, "min_node_height_mm": 1000.0 * self.min_node_z, "samples": self.n}


def check_rest(t_hold=30.0, cfg_kw=None):
    S = Simulator(mdl.ModelConfig(visual=False, **(cfg_kw or {})))
    ok = S.stand()
    c0 = S.com().copy()
    n0 = S.nodes().copy()
    mon = Monitor(S)
    for _ in range(int(t_hold / 0.5)):
        S.step(int(round(0.5 / S.dt)))
        mon.update()
    c1 = S.com()
    n1 = S.nodes()
    return {"stand_ok": ok, "com_height_mm": 1000 * float(c1[2]), "top_mm": 1000 * S.top_z(),
            "com_drift_mm": 1000 * float(np.linalg.norm(c1 - c0)),
            "node_drift_max_mm": 1000 * float(np.linalg.norm(n1 - n0, axis=1).max()),
            "final_max_node_speed_mm_s": 1000 * S.max_node_speed(), "node_height_mm": (1000 * n1[:, 2]).round(2).tolist(),
            "bottom": list(S.bottom()[0]), "sensor_length_mm": (1000 * S.sensor_lengths()).round(2).tolist(),
            "cable_tension_N": np.round(S.cable_tensions(), 3).tolist(),
            "line_stiffness_N_m": np.round(S.k_line, 1).tolist(), "monitor": mon.report(),
            "pass": bool(ok and np.linalg.norm(c1 - c0) < 1e-3 and S.max_node_speed() < 1e-3)}


def check_long(t_total=600.0, seed=7, mode=None, cfg_kw=None):
    """Mixed random primitives (the 11 published ones) for t_total simulated seconds."""
    S = Simulator(mdl.ModelConfig(visual=False, **(cfg_kw or {})))
    S.stand()
    rng = random.Random(seed)
    mon = Monitor(S)
    seq, peakT, tout, path, not_rest, faces = [], 0.0, 0, 0.0, 0, {}
    t0 = time.time()
    c_prev = S.com()[:2].copy()
    stopped = None
    while S.time < t_total:
        name = rng.choice(gaits.NAMES)
        try:
            r = S.run_primitive(name, mode=mode)
        except gaits.NotRollableFace as e:
            stopped = str(e)
            break
        mon.update()
        seq.append(name)
        peakT = max(peakT, r["peak_cable_tension_N"])
        tout += r["timeouts"]
        not_rest += 0 if r["rest_reached"] else 1
        faces[str(tuple(r["bottom_end"]))] = faces.get(str(tuple(r["bottom_end"])), 0) + 1
        c = S.com()[:2]
        path += float(np.linalg.norm(c - c_prev))
        c_prev = c.copy()
    rep = mon.report()
    e = S.energy_report()
    return {"sim_s": S.time, "wall_s": time.time() - t0, "n_primitives": len(seq), "sequence_head": seq[:20],
            "timeouts": tout, "rest_not_reached": not_rest, "end_faces": faces, "stopped_not_rollable": stopped,
            "peak_cable_tension_N": peakT, "com_path_m": path, "monitor": rep, "energy": e,
            "final_hash": S.state_hash(),
            "pass": bool(not rep["nan"] and not rep["warnings"] and rep["max_penetration_mm"] < 5.0
                         and rep["max_node_speed_m_s"] < 3.0 and stopped is None)}


def _dt_runs(dt, mode=None, perturb=0.0, names=None):
    out = {}
    for name in (names or gaits.NAMES):
        S = Simulator(mdl.ModelConfig(visual=False, timestep=dt))
        if perturb:
            S.d.qpos[S.ids.jnt_qpos[0]] += perturb
        S.stand(mode=mode)
        rows = []
        for _ in range(2):
            r = S.run_primitive(name, mode=mode)
            rows.append({"dx": r["dx"], "dy": r["dy"], "dtheta_deg": r["dtheta_deg"],
                         "peak_T": r["peak_cable_tension_N"], "faces": [list(r["bottom_start"]), list(r["bottom_end"])]})
        out[name] = rows
    return out


def _dt_diff(a, b):
    dt_mm, dth, dT = [], [], []
    for name in a:
        for x, y in zip(a[name], b[name]):
            dt_mm.append(1000 * math.hypot(x["dx"] - y["dx"], x["dy"] - y["dy"]))
            dth.append(abs(x["dtheta_deg"] - y["dtheta_deg"]))
            dT.append(100 * abs(x["peak_T"] - y["peak_T"]) / y["peak_T"])
    st = lambda v: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(np.max(v))}
    return {"translation_mm": st(dt_mm), "heading_deg": st(dth), "peak_tension_pct": st(dT)}


def check_dt(dts=(0.001, 0.0005, 0.00025), ref=0.000125, mode=None, perturb=1e-7, names=None):
    """Timestep study (ysim 0.3.0): each primitive from stand, two executions, at every dt and at the reference
    dt; differences against the reference; the floor = the same runs at the chosen candidate dts with the start
    shifted by `perturb` metres (how much the result moves for a negligible change of the start)."""
    R = _dt_runs(ref, mode, names=names)
    rows = {}
    for dt in dts:
        t0 = time.process_time()
        a = _dt_runs(dt, mode, names=names)
        rows[str(dt)] = {"vs_ref": _dt_diff(a, R), "cpu_s": time.process_time() - t0}
        if perturb:
            rows[str(dt)]["floor"] = _dt_diff(_dt_runs(dt, mode, perturb, names=names), a)
    return {"ref_dt": ref, "rows": rows}


def scripted_hashes(seq=("roll_100_100", "ccw", "roll_140_120", "cw"), mode=None):
    S = Simulator(mdl.ModelConfig(visual=False))
    S.stand()
    hs = [S.state_hash()]
    for p in seq:
        S.run_primitive(p, mode=mode)
        hs.append(S.state_hash())
    return hs


def check_rtf(t_sim=60.0, visual=False, mode=None):
    S = Simulator(mdl.ModelConfig(visual=visual))
    S.stand()
    t_start = S.time
    t0 = time.process_time()
    w0 = time.time()
    rng = random.Random(3)
    while S.time - t_start < t_sim:
        S.run_primitive(rng.choice(gaits.NAMES), mode=mode)
    cpu = time.process_time() - t0
    return {"visual_meshes_loaded": visual, "dt": S.dt, "sim_s": S.time - t_start, "cpu_s": cpu,
            "wall_s": time.time() - w0, "rtf_cpu": (S.time - t_start) / cpu}


if __name__ == "__main__":
    out = {}
    for chk in sys.argv[2].split(","):
        t0 = time.time()
        if chk == "rest":
            out["rest"] = check_rest()
        elif chk == "long":
            out["long"] = check_long()
        elif chk == "dt":
            out["dt"] = check_dt()
        elif chk == "det":
            out["det"] = {"hashes": scripted_hashes()}
        elif chk == "rtf":
            out["rtf"] = [check_rtf(visual=False), check_rtf(visual=True)]
        out.setdefault("wall", {})[chk] = time.time() - t0
        print(chk, json.dumps(out[chk])[:600], flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
