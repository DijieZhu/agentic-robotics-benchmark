"""The acceptance run of the motion-controller task (grader side): one unit, three cells, every job once.

    acc = Acceptance(design_dir, seed, log=print)
    out = acc.run(make_controller)       # make_controller() -> an object with call(op, payload, cpu_s, wall_s) and close()
    out["scores"] (metrics.episode), out["records"] (one per job), out["stats"]

`make_controller` returns a fresh controller connection (wire.Runner in verify, LocalController in development); after a
crash the grader makes a new one and calls `setup` again for the current cell (CONTRACT.md 5). The job loop follows
CONTRACT.md 4-6: `plan`, then 10 ms bus cycles (`step`) until done or the cap, a contact stops the arm and fails the
cell (its later jobs score 0), a 0.5 s settle after done, measurements on the true unit. `scenario` = a fixed hidden
scenario (scenarios.py) instead of drawing the unit and the cells from the seed. `privileged=True` adds the truth payloads (the oracle's run; only when
the submission carries the oracle token).
"""

import copy
import math
import time

import numpy as np

from . import cells as C
from . import jobs as JB
from . import metrics as M
from . import plant as P
from . import privileged as PV
from . import unit as U

SETUP_CPU_S, SETUP_WALL_S = 60.0, 120.0
PLAN_CPU_S, PLAN_WALL_S = 20.0, 60.0
STEP_CPU_HARD_S, STEP_WALL_S = 1.0, 5.0
REHOME_S = 20.0
MIN_FEASIBLE_JOBS = 8          # a cell layout with fewer feasible jobs is redrawn (at most MAX_CELL_ATTEMPTS layouts; the
MIN_PATH_JOBS = 5              # best one is kept), and a dispensing cell with fewer feasible paths
MIN_MAIN_JOBS = 5              # ... and a shelf / bins cell with fewer compartment reaches / bin placements (its main work)
MAIN_KIND = {"shelf": "compartment", "bins": "bin"}
MAX_CELL_ATTEMPTS = 10
CONTACT_ENDS_CELL = True       # a contact fails the cell's acceptance: that job and the cell's later jobs score 0
NEAR_MM = 1.0                  # overshoot: the largest later excursion after the tool point first came within 1 mm of
                               # its final position, minus that 1 mm


class LocalController(object):
    """In-process controller connection (development, references, tests): calls the object directly and measures the
    calling thread's CPU time."""

    def __init__(self, ctl):
        self.ctl = ctl
        self.dead = False
        self.stats = {"local": True}

    def call(self, op, payload=None, cpu_s=1.0, wall_s=10.0):
        payload = payload or {}
        t0, c0 = time.time(), time.thread_time()
        try:
            if op == "setup":
                self.ctl.setup(payload["unit"], payload["cell"], payload["tool"], **({"truth": payload["truth"]} if "truth" in payload else {}))
                rep = {"ok": True}
            elif op == "plan":
                from .client import _takes_state
                kw = {"truth": payload["truth"]} if "truth" in payload else {}
                if _takes_state(self.ctl.plan):
                    r = self.ctl.plan(payload["job"], payload.get("state"), **kw)
                else:
                    r = self.ctl.plan(payload["job"], **kw)
                rep = {"ok": True, "result": "infeasible" if r == "infeasible" else "ok"}
            elif op == "step":
                rep = {"ok": True, "cmd": self.ctl.step(payload["state"])}
            else:
                rep = {"ok": True}
            status = "ok"
        except Exception as e:  # noqa: BLE001
            rep, status = {"ok": False, "error": "{}: {}".format(type(e).__name__, e)}, "error"
        cpu = time.thread_time() - c0
        if cpu > cpu_s:
            status = "cpu"
        return status, rep, cpu, time.time() - t0

    def close(self):
        return dict(self.stats)


def tool_public(gen, design):
    """The cell's tool as the controller receives it (CONTRACT.md 3): the part's size and seat (its -x face on the
    fixed finger's pad face: centre at TCP + seat x along the TCP's x and seat z along its z), the needle and its holder."""
    t = gen["tool"]
    if t is None:
        return {"kind": "gripper"}
    seat = U.tool_seat(design, t)
    if t["kind"] == "part":
        part = gen["part"]
        return {"kind": "part", "half_mm": [round(v * 1e3, 2) for v in part["half"]],
                "seat_mm": {"x": round(seat["seat_x"] * 1e3, 2), "z": round(seat["seat_z"] * 1e3, 2)},
                "mass_range_kg": [0.03, 0.15], "com_note": "the centre of mass is inside the part; not known"}
    p = t["params"]
    return {"kind": "needle", "length_mm": round(p["length_m"] * 1e3, 2), "radius_mm": round(p["radius_m"] * 1e3, 2),
            "mass_kg": p["mass_kg"], "axis_offset_x_mm": round(seat["seat_x"] * 1e3, 3),
            "holder_half_mm": [round(v * 1e3, 2) for v in U.NEEDLE_HOLDER_HALF_M], "holder_centre_z_mm": round(U.NEEDLE_HOLDER_Z_M * 1e3, 2)}


def unit_public(spec):
    return {"serial": spec["serial"], "calibration": copy.deepcopy(spec["record"])}


def _angle(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


class Acceptance(object):
    def __init__(self, design_dir, seed=None, log=None, privileged=False, kinds=C.KINDS, counts=None, docs_ideal=False,
                 scenario=None):
        """scenario: a fixed hidden scenario (scenarios.load): its unit, cells and jobs instead of drawing them."""
        self.design_dir = design_dir
        self.xml = design_dir.rstrip("/") + "/robot.xml"
        self.design = U.load_design(design_dir)
        self.scenario = scenario
        if scenario is not None:
            self.seed = int(scenario["seed"])
            self.spec = scenario["spec"]
        else:
            self.seed = U.os_seed() if seed is None else int(seed)
            self.spec = U.draw_unit(self.seed, self.design)
        self.nominal = U.nominal_spec(self.design)
        self.log = log or (lambda s: None)
        self.privileged = bool(privileged) or bool(docs_ideal)
        self.docs_ideal = bool(docs_ideal)        # band calibration: the truth payload computed from the documents only
        self.kinds = list(kinds)
        self.counts = counts
        self._prev_exempt = []
        self.stats = {"cycles": 0, "cycles_late": 0, "restarts": 0, "contacts": 0, "rehomes": 0, "truth_payloads": 0,
                      "controller_errors": 0, "gen_s": 0.0, "run_s": 0.0}

    # --------------------------------------------------------------------------------------------- controller
    def _connect(self, make_controller):
        self.ctl = make_controller()
        return self.ctl

    def _setup(self, cell, gen, plant=None):
        payload = {"unit": unit_public(self.spec), "cell": cell.description(), "tool": tool_public(gen, self.design)}
        if self.privileged:
            payload["truth"] = {"serial": self.spec["serial"]}
            self.stats["truth_payloads"] += 1
        st, rep, cpu, wall = self.ctl.call("setup", payload, SETUP_CPU_S, SETUP_WALL_S)
        return st == "ok"

    def _revive(self, make_controller, cell, gen, plant=None):
        try:
            self.ctl.close()
        except Exception:  # noqa: BLE001
            pass
        self.stats["restarts"] += 1
        self._connect(make_controller)
        return self._setup(cell, gen, plant)

    def _restart(self, make_controller, cell, gen):
        """A restart in the middle of a cell, with the cell-start rule: setup retried once with a fresh controller,
        then the cell's remaining jobs are not run."""
        if not self._revive(make_controller, cell, gen) and not self._revive(make_controller, cell, gen):
            self._cell_dead = "setup failed"

    # -------------------------------------------------------------------------------------------------- run
    def run(self, make_controller):
        records = []
        self._connect(make_controller)
        q_home = [P.HOME_Q[j] for j in U.JOINTS]
        for k, kind in enumerate(self.kinds):
            t0 = time.time()
            cell, gen = self.layout(kind, q_home)
            self.stats["gen_s"] += time.time() - t0
            t1 = time.time()
            plant = P.Plant(self.xml, self.spec, cell, tool=gen["tool"], design=self.design)
            plant.power_up()
            self._prev_exempt = []
            if self.docs_ideal:
                self.believed = U.believed_spec(self.design, self.spec["record"])
                truth_view = JB.View(self.xml, self.believed, cell.described(), tool=gen["tool"], design=self.design)
            else:
                truth_view = JB.View(self.xml, self.spec, cell, tool=gen["tool"], design=self.design) if self.privileged else None
            ok = self._setup(cell, gen, plant)
            if not ok and not self._revive(make_controller, cell, gen):
                for job in gen["jobs"]:
                    records.append(self._blank(job, cell, "setup failed"))
                continue
            self._cell_dead = None
            for job in gen["jobs"]:
                if self._cell_dead:                     # the cell has stopped: its later jobs are not run (score 0)
                    rec = self._blank(job, cell, self._cell_dead)
                else:
                    rec = self._run_job(plant, cell, gen, job, truth_view, make_controller)
                    if CONTACT_ENDS_CELL and str(rec.get("status", "")).startswith("contact"):
                        self._cell_dead = "not run: the cell stopped at the contact in {}".format(job["id"])
                        self.stats["cells_stopped"] = self.stats.get("cells_stopped", 0) + 1
                rec["cell"] = kind
                records.append(rec)
                self.log("  {} {} {}: {}".format(kind, job["id"], job["type"], _brief(rec)))
            self.stats["run_s"] += time.time() - t1
        try:
            self.stats["runner"] = self.ctl.close()
        except Exception:  # noqa: BLE001
            pass
        scores = M.episode(records)
        cyc = max(1, self.stats["cycles"])
        self.stats["on_time_fraction"] = 1.0 - self.stats["cycles_late"] / cyc
        return {"seed": self.seed, "serial": self.spec["serial"], "records": records, "scores": scores, "stats": dict(self.stats)}

    def layout(self, kind, q_home):
        """(cell, gen) of a kind: the scenario's, or drawn from the seed; a drawn layout in which too few jobs can be
        made is redrawn (deterministically: attempt k uses a shifted seed; the best of MAX_CELL_ATTEMPTS is kept)."""
        if self.scenario is not None:
            c = self.scenario["cells"][kind]
            return c["cell"], c["gen"]
        kidx = C.KINDS.index(kind)                  # by kind, so a single-kind run reproduces the full run's cell
        if self.counts:
            JB.COUNTS[kind] = dict(self.counts.get(kind, JB.COUNTS[kind]))
        need_p = MIN_PATH_JOBS if JB.COUNTS[kind].get("path") else 0
        need_r = 1 if (JB.COUNTS[kind].get("path") and JB.COUNTS[kind].get("reach")) else 0   # dispense: a purge
        need_m = MIN_MAIN_JOBS if kind in MAIN_KIND else 0
        best = None
        for attempt in range(MAX_CELL_ATTEMPTS):
            cell = C.draw_cell(self.seed + 1009 * (kidx + 1) + 7919 * attempt, kind)
            gen = self._generate(cell, self.seed + kidx + 7919 * attempt, q_home)
            feas = [j for j in gen["jobs"] if not j.get("infeasible")]
            nf = len(feas)
            n_p = sum(1 for j in feas if j["type"] == "path")
            n_r = nf - n_p
            n_m = sum(1 for j in feas if j["type"] == "reach" and j["target"].get("kind") == MAIN_KIND.get(kind))
            key = (min(nf, MIN_FEASIBLE_JOBS) + min(n_p, need_p) + min(n_r, need_r) + min(n_m, need_m), nf)
            if best is None or key > best[0]:
                best = (key, cell, gen)
            if nf >= MIN_FEASIBLE_JOBS and n_p >= need_p and n_r >= need_r and n_m >= need_m:
                break
            self.stats["cell_redraws"] = self.stats.get("cell_redraws", 0) + 1
        return best[1], best[2]

    def _generate(self, cell, gseed, q_home):
        """jobs.generate, or (development only: $ARB_GEN_CACHE set) its pickled result for the same seed and code."""
        import os
        cache = os.environ.get("ARB_GEN_CACHE")
        if not cache:
            return JB.generate(self.xml, self.spec, self.nominal, cell, gseed, q_home, log=None)
        import hashlib
        import pickle
        here = os.path.dirname(os.path.abspath(__file__))
        h = hashlib.sha256()
        for fn in ("jobs.py", "cells.py", "unit.py", "kin.py", "plant.py", "privileged.py", "metrics.py"):
            with open(os.path.join(here, fn), "rb") as f:
                h.update(f.read())
        key = "{}_{}_{}_{}.pkl".format(self.seed, cell.kind, gseed, h.hexdigest()[:12])
        path = os.path.join(cache, key)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        gen = JB.generate(self.xml, self.spec, self.nominal, cell, gseed, q_home, log=None)
        os.makedirs(cache, exist_ok=True)
        with open(path + ".tmp", "wb") as f:
            pickle.dump(gen, f)
        os.replace(path + ".tmp", path)
        return gen

    def _blank(self, job, cell, why):
        return {"id": job["id"], "type": job["type"], "infeasible": bool(job.get("infeasible")), "refused": False,
                "completed": False, "contact": None, "status": why, "cell": cell.kind, "time_target_s": job["time_target_s"]}

    # ---------------------------------------------------------------------------------------------- one job
    def _run_job(self, plant, cell, gen, job, truth_view, make_controller):
        rec = self._blank(job, cell, "ok")
        pub = JB.public(job, cell)
        st0 = plant.read()
        payload = {"job": pub, "state": {"position": st0["position"], "speed": st0["speed"], "load": st0["load"], "moving": st0["moving"]}}
        if self.privileged:
            tr = {"infeasible": bool(job.get("infeasible"))}
            if not job.get("infeasible"):
                job = self._replan_from_actual(plant, job, truth_view)
                jaw = plant.seat["jaw_q"] if gen["tool"] else 0.15
                if self.docs_ideal:
                    jb = dict(job)

                    def same_roll(qn, qref):
                        # the jaw axis has no sign: take the nominal solution's roll branch nearest the plan's (+-pi)
                        qn = list(qn)
                        while qn[4] - qref[4] > math.pi / 2:
                            qn[4] -= math.pi
                        while qref[4] - qn[4] > math.pi / 2:
                            qn[4] += math.pi
                        return qn
                    if job["type"] == "reach":
                        jb["plan_true"] = list(job["plan_true"][:-1]) + [same_roll(job["q_nominal"], job["plan_true"][-1])]
                    else:
                        qref = job["plan_true"][-1]
                        jb["q_true"] = [same_roll(q, qref) for q in job["q_nominal"]]
                        jb["plan_true"] = list(job["plan_true"][:-1]) + [jb["q_true"][0]]
                    tr["traj"] = PV.trajectory(truth_view.kin, self.believed, jb, jaw_q=jaw)
                else:
                    tr["traj"] = PV.trajectory(truth_view.kin, self.spec, job, jaw_q=jaw)
            payload["truth"] = tr
            self.stats["truth_payloads"] += 1
        st, rep, cpu, wall = self.ctl.call("plan", payload, PLAN_CPU_S, PLAN_WALL_S)
        rec["plan_cpu_s"] = round(cpu, 3)
        if st != "ok":
            self.stats["controller_errors"] += 1
            rec["status"] = "plan " + st
            if getattr(self.ctl, "dead", False) or st in ("cpu", "timeout", "dead", "protocol"):
                self._restart(make_controller, cell, gen)
            return rec
        if rep.get("result") == "infeasible":
            rec["refused"] = True
            rec["status"] = "refused"
            return rec
        # the motion; the clearance exemptions are this job's support and the previous job's (the arm leaves it)
        plant.reset_monitor()
        plant.set_exempt(list(job.get("exempt", [])) + list(self._prev_exempt))
        self._prev_exempt = list(job.get("exempt", []))
        t0 = plant.t
        T = float(job["time_target_s"])
        if job["type"] == "reach":
            cap = max(2.0 * T, T + 2.0)
        else:
            cap = 2.0 * float(job["approach_time_target_s"]) + float(job["path"]["t"][-1]) + 2.0
        samples, conds = [], []
        path_t0 = None
        done_t = None
        cycle = 0
        status = "ok"
        while True:
            state = plant.read()
            state["cycle"] = cycle
            state["t"] = round(plant.t - t0, 4)
            st, rep, cpu, wall = self.ctl.call("step", {"state": state}, STEP_CPU_HARD_S, STEP_WALL_S)
            if st != "ok":
                self.stats["controller_errors"] += 1
                status = "step " + st
                if getattr(self.ctl, "dead", False) or st in ("cpu", "timeout", "dead", "protocol"):
                    self._restart(make_controller, cell, gen)
                break
            cmd = rep.get("cmd")
            if cmd is None:
                cmd = {}
            parsed = _parse_cmd(cmd)
            if parsed is None:
                self.stats["controller_errors"] += 1
                status = "step protocol (malformed command)"
                break
            late = int(cpu // P.CYCLE_S) if cpu > P.CYCLE_S else 0
            self.stats["cycles"] += 1 + late
            if late:
                self.stats["cycles_late"] += 1 + late
            for _ in range(late):
                plant.cycle(None)
                samples.append(self._sample(plant, t0))
                conds.append(samples[-1]["cond"])
                if plant.contact:
                    break
            if plant.contact is None:
                goal, spd, acc = parsed
                plant.cycle(goal, spd, acc)
                samples.append(self._sample(plant, t0))
                conds.append(samples[-1]["cond"])
            cycle += 1 + late
            if plant.contact is not None:
                status = "contact"
                break
            if job["type"] == "path" and path_t0 is None and cmd.get("path_start"):
                path_t0 = plant.t
            if cmd.get("done"):
                done_t = plant.t
                break
            if plant.t - t0 > cap + 1e-9:
                status = "cap"
                break
        rec["status"] = status
        rec["cycles"] = cycle
        if status == "contact":
            rec["contact"] = dict(plant.contact)
            self.stats["contacts"] += 1
            self._rehome(plant)
            return rec
        if done_t is None:
            return rec
        # settle with the last goals on the bus, still monitoring contact
        tip_done = np.array(samples[-1]["tip"])
        settle = []
        for _ in range(int(round(M.SETTLE_S / P.CYCLE_S))):
            plant.cycle(None)
            settle.append(self._sample(plant, t0))
            if plant.contact is not None:
                break
        if plant.contact is not None:
            rec["status"] = "contact (settle)"
            rec["contact"] = dict(plant.contact)
            self.stats["contacts"] += 1
            self._rehome(plant)
            return rec
        rec["t_used"] = round(done_t - t0, 3)
        rec["min_clear_m"] = float(plant.min_clear)
        rec["min_clear_pair"] = plant.min_clear_pair and [str(plant.min_clear_pair[0]), str(plant.min_clear_pair[1])]
        rec["conds"] = conds
        rec["cond_min"] = float(min(conds)) if conds else None
        end = settle[-1]
        tip_end = np.array(end["tip"])
        if job["type"] == "reach":
            tg = job["target"]
            err = {"pos_mm": float(np.linalg.norm(tip_end - tg["pos"]) * 1e3), "app_deg": _angle(end["axis"], tg["app"]), "jaw_deg": None}
            if tg.get("jaw") is not None:
                jw = np.asarray(tg["jaw"], float)
                x = np.asarray(end["jaw_axis"], float)
                err["jaw_deg"] = min(_angle(x, jw), _angle(-x, jw))
            rec["err"] = err
            rec["settle_mm"] = float(np.linalg.norm(tip_end - tip_done) * 1e3)
            d = [float(np.linalg.norm(np.array(s["tip"]) - tip_end) * 1e3) for s in samples + settle]
            k1 = next((i for i, v in enumerate(d) if v < NEAR_MM), len(d))
            rec["overshoot_mm"] = max(0.0, max(d[k1:]) - NEAR_MM) if k1 < len(d) else float(d[-1])
            rec["completed"] = True
        else:
            if path_t0 is None:
                rec["status"] = "no path_start"
                return rec
            p = job["path"]
            Tp = float(p["t"][-1])
            if done_t + 1e-9 < path_t0 + Tp:
                rec["status"] = "done before the path ended"
                return rec
            devs, axis_out, act, ref = [], 0, [], []
            n_ref = len(p["tips"])
            for s in samples:
                tr = s["t_abs"] - path_t0
                if tr <= 1e-9 or tr > Tp + 1e-9:
                    continue
                i = min(n_ref - 1, int(round(tr / 0.01)))
                act.append(np.array(s["tip"], float))
                ref.append(np.asarray(p["tips"][i], float))
                devs.append(float(np.linalg.norm(act[-1] - ref[-1]) * 1e3))
                if _angle(s["axis"], -np.asarray(p["normal"], float)) > JB.AXIS_TOL_DEG:
                    axis_out += 1
            if not devs:
                rec["status"] = "no path samples"
                return rec
            rec["rms_mm"] = float(math.sqrt(np.mean(np.square(devs))))
            rec["peak_mm"] = float(max(devs))
            rec["axis_out_frac"] = axis_out / len(devs)
            # along-track progress: the tip's motion projected on the reference's motion, over the reference's (a needle
            # parked anywhere scores 0, however short the path)
            dA, dR = np.diff(np.array(act), axis=0), np.diff(np.array(ref), axis=0)
            den = float(np.sum(dR * dR))
            rec["progress"] = float(np.clip(np.sum(dA * dR) / den, 0.0, 1.0)) if den > 0 else 0.0
            rec["n_path_samples"] = len(devs)
            rec["completed"] = True
        rec["score"], rec["parts"] = M.job_score(rec)
        return rec

    def _replan_from_actual(self, plant, job, view):
        """The privileged plan starts where the generator assumed the arm would be (the previous job's end); after a
        stop, a refused job or a miss it is re-planned from the arm's actual link angles (privileged runs only)."""
        q_act = plant.q().copy()
        q_act[:5] += plant.data.qpos[plant.slack_qadr]
        if float(np.max(np.abs(q_act[:5] - np.asarray(job["plan_true"][0])[:5]))) < math.radians(3.0):
            return job
        jb = dict(job)
        if job["type"] == "reach":
            goal, pos, app = job["q_true"] if not self.docs_ideal else job["q_nominal"], job["target"]["pos"], job["target"]["app"]
        else:
            goal = (job["q_true"] if not self.docs_ideal else job["q_nominal"])[0]
            pos, app = job["path"]["tips"][0], -np.asarray(job["path"]["normal"], float)
        ex = None
        p, R = plant.tool_tip()
        if job.get("exit_from"):
            ef = job["exit_from"]
            if np.linalg.norm(p - np.asarray(ef[0])) < 0.01:
                ex = tuple(ef)
        rng = np.random.default_rng(int(self.seed) & 0xFFFFFFFF)
        plan = JB.plan_to(view, q_act, np.asarray(goal, float), pos, app, rng, exit_from=ex,
                          entry_exempt_min=(0.0015 if job["type"] == "path" else None))
        if plan is not None:
            jb["plan_true"] = [list(map(float, q)) for q in plan]
            jb["plan_slow"] = list(JB.PLAN_INFO["slow"])
            self.stats["privileged_replans"] = self.stats.get("privileged_replans", 0) + 1
        return jb

    def _sample(self, plant, t0):
        p, R = plant.tool_tip()
        return {"t_abs": plant.t, "tip": p.tolist(), "axis": R[:, 2].tolist(), "jaw_axis": R[:, 0].tolist(), "cond": plant.conditioning()}

    def _rehome(self, plant):
        self._prev_exempt = []
        plant.power_up(keep_time=True)
        plant.t = round(plant.t + REHOME_S, 9)
        self.stats["rehomes"] += 1


def _parse_cmd(cmd):
    """(goal, speed, acc) of a step reply, or None when it is malformed (not a dict, a list of the wrong length, a
    non-finite or non-numeric entry, a goal outside 0..4095 x 2): a protocol error, the job is aborted."""
    if not isinstance(cmd, dict):
        return None

    def vec(key, lo, hi):
        v = cmd.get(key)
        if v is None:
            return True, None
        if not isinstance(v, (list, tuple)) or len(v) != 6:
            return False, None
        out = []
        for x in v:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return False, None
            try:
                f = float(x)
            except (OverflowError, ValueError):
                return False, None
            if not math.isfinite(f) or not (lo <= f <= hi):
                return False, None
            out.append(f)
        return True, out
    ok, goal = vec("goal", -8192, 8192)
    if not ok:
        return None
    ok, spd = vec("speed", -1e6, 1e6)
    if not ok:
        return None
    ok, acc = vec("acc", -1e6, 1e6)
    if not ok:
        return None
    if goal is not None:
        goal = [int(round(x)) for x in goal]
    if spd is not None:
        spd = [abs(x) for x in spd]
    if acc is not None:
        acc = [abs(x) for x in acc]
    return goal, spd, acc


def _brief(rec):
    if rec.get("refused"):
        return "refused ({})".format("right" if rec.get("infeasible") else "WRONG")
    if rec.get("infeasible"):
        return "attempted an infeasible job: " + rec.get("status", "")
    if not rec.get("completed"):
        return rec.get("status", "")
    if rec["type"] == "reach":
        e = rec["err"]
        return "score {:.3f} pos {:.2f} mm app {:.2f} deg clear {:.1f} mm t {:.2f}/{:.2f}".format(
            rec.get("score", 0), e["pos_mm"], e["app_deg"], rec["min_clear_m"] * 1e3, rec["t_used"], rec["time_target_s"])
    return "score {:.3f} rms {:.2f} mm peak {:.2f} axis_out {:.2f} clear {:.1f} mm".format(
        rec.get("score", 0), rec["rms_mm"], rec["peak_mm"], rec["axis_out_frac"], rec["min_clear_m"] * 1e3)
