"""Simulator: MuJoCo physics + six gearmotors + the authors' PID + gait executor + telemetry.

Loop (single-threaded, deterministic): every physics step each motor turns its spool from the duty it holds and
the current line tension; the cable rest length changes by r * dtheta and the line stiffness follows the material
rule k = EA / (L0 + line_inner_length); every 1/control_rate the PID reads the short-sensor lengths (true
hook-to-hook distances) and sets new duties (zero-order hold, as the Arduino holds the last PWM until the next UDP
packet).

A primitive (ysim 0.3.0, as on the real robot): from a rest state on a rollable face (gaits.ROLLABLE_FACES; any
other bottom triple raises gaits.NotRollableFace), the primitive's own target states (the authors' control code,
symmetry-mapped once from the three lowest end caps), then the authors' re-plan phase: state [1]*6 at RANGE 100,
i.e. every short sensor back to 200 mm (run_tensegrity_Astar.py lines 438-443, 471-477; RA-L Fig. 5: 're-planning
occurs at the end of each primitive as the robot returns to a rest state'), then a braked hold of params
settle_after (ysim 0.3.1: ASSUMPTION, the authors' re-planning delay 1/7 s + 0.768 s; ysim's reading of the hardware
loop: the rest phase completes every time, then the hold; on the robot the rest phase and the planning overlap and a
roll that follows a roll cuts an unfinished rest phase short, run_tensegrity_Astar.py lines 548-553).
"""
import hashlib
import math

import mujoco
import numpy as np

from . import gaits
from . import geometry as geom
from . import metrics
from . import model as mdl
from . import params as prm
from .controller import FirmwarePID
from .motor import GearMotor

G0 = 9.80665
MODES = ("hardware", "tracked", "authors_sim")


def bands(mode=None, code_tol=None):
    """(band of targets >= 0.5, band of targets < 0.5, rest-phase band (tol, low_tol), slack rule) of an execution
    mode; code_tol = the authors' band history (hardware mode; default the code's initial 0.15)."""
    mode = mode or prm.value("execution_mode")
    if mode == "tracked":
        t = prm.value("track_tol")
        return t, t, (t, t), True
    if mode == "hardware":
        ct = prm.value("pid_tol") if code_tol is None else code_tol
        lo = prm.value("pid_low_tol")
        return ct, lo, (ct, lo), False
    raise ValueError(mode)


class Simulator:
    def __init__(self, cfg=None, poses=None, v_bat=None, control_rate=None, pid_variant="hardware", pid_P=None,
                 spool_r=None, motor_tau=None, hold_torque=None):
        """spool_r, motor_tau, hold_torque: sensitivity knobs of the motor ASSUMPTIONs (motor.GearMotor)."""
        self.cfg = cfg or mdl.ModelConfig()
        self.xml, self.info = mdl.build_xml(self.cfg, poses)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        self.ids = mdl.Ids(self.m)
        self.dt = float(self.m.opt.timestep)
        rate = control_rate or prm.value("control_rate")
        self.ctrl_every = max(1, int(round(1.0 / (rate * self.dt))))
        self.motors = [GearMotor(v_bat, spool_r, motor_tau, hold_torque) for _ in range(6)]
        self.pid = FirmwarePID(pid_variant, pid_P)
        c = self.info["config"]
        self.EA = float(c["line_EA"])
        self.k_const = c["cable_k"]               # diagnostic constant stiffness (None = material rule)
        self.c_c = float(self.m.tendon_damping[self.ids.cable[0]])
        cal = _read_calibration()
        self.cal_m, self.cal_b = cal
        self.nstep = 0
        self.duty = [0.0] * 6
        self.exec = None
        self.peak_tension = 0.0
        self.log = None
        self.code_tol = prm.value("pid_tol")      # the authors' self.tol, set by rolls only (params code_tol_history)
        self.tick_hooks = []      # callables h(sim) run at every control tick before the PID (read-only observers:
                                  # camera exposure snapshots, capability sampling); physics unaffected
        mujoco.mj_forward(self.m, self.d)
        self._lift_to_floor(0.002)
        mujoco.mj_forward(self.m, self.d)
        L = self.d.ten_length[self.ids.cable].copy()
        self.L0 = list(L)
        self.L0_max = [x + prm.value("line_capacity") for x in L]
        self.L0_min = prm.value("line_min_rest_length")
        self.k_line = [0.0] * 6
        for i, t in enumerate(self.ids.cable):
            self.m.tendon_lengthspring[t] = [0.0, self.L0[i]]
            self._set_line_k(i)
        mujoco.mj_forward(self.m, self.d)

    # ---------------- state access ----------------
    def nodes(self):
        return self.d.site_xpos[self.ids.node_site].copy()

    def com(self):
        return np.mean([self.d.xipos[b] for b in self.ids.body], axis=0)

    def sensor_lengths(self):
        """Nine stretch-sensor lengths [m]: 0-5 short (parallel to cables 0-5), 6-8 long (td_12..td_14)."""
        return self.d.ten_length[self.ids.sensor_tendons].copy()

    def cable_tensions(self):
        out = []
        for i, t in enumerate(self.ids.cable):
            L, v = self.d.ten_length[t], self.d.ten_velocity[t]
            T = self.k_line[i] * (L - self.L0[i]) if L > self.L0[i] else 0.0
            out.append(max(T + self.c_c * v, 0.0) if T > 0 else 0.0)
        return out

    def bottom(self):
        """The three lowest end caps (the authors' bottom3) and whether they form a rollable face."""
        b = metrics.bottom3(self.nodes())
        return b, b in gaits.SYMMETRY

    def _set_line_k(self, i):
        k = mdl.line_stiffness(self.L0[i], self.EA, self.k_const)
        self.k_line[i] = k
        self.m.tendon_stiffness[self.ids.cable[i]] = k

    def _lift_to_floor(self, clearance):
        zmin = 1e9
        for g in self.ids.col_geoms:
            zmin = min(zmin, self._geom_zmin(g))
        dz = clearance - zmin
        for a in self.ids.jnt_qpos:
            self.d.qpos[a + 2] += dz

    def _geom_zrange(self, g):
        m, d = self.m, self.d
        t = m.geom_type[g]
        p = d.geom_xpos[g]
        R = d.geom_xmat[g].reshape(3, 3)
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = m.geom_size[g][0]
            return p[2] - r, p[2] + r
        if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, h = m.geom_size[g][0], m.geom_size[g][1]
            ax = R[:, 2]
            e = abs(ax[2]) * h + r * math.sqrt(max(0.0, 1 - ax[2] ** 2))
            return p[2] - e, p[2] + e
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            z = (m.mesh_vert[v0:v0 + nv] @ R.T + p)[:, 2]
            return float(z.min()), float(z.max())
        return p[2], p[2]

    def _geom_zmin(self, g):
        return self._geom_zrange(g)[0]

    def top_z(self):
        """Highest point of the robot's collision geometry [m] (same as capability.top_z)."""
        return max(self._geom_zrange(g)[1] for g in self.ids.col_geoms)

    # ---------------- stepping ----------------
    def _motors(self):
        d, m = self.d, self.m
        for i, t in enumerate(self.ids.cable):
            L, v = d.ten_length[t], d.ten_velocity[t]
            T = self.k_line[i] * (L - self.L0[i]) if L > self.L0[i] else 0.0
            if T > 0.0:
                T = max(T + self.c_c * v, 0.0)
            if T > self.peak_tension:
                self.peak_tension = T
            dL = self.motors[i].step(self.duty[i], T, self.dt)
            L0 = self.L0[i] + dL
            L0 = min(max(L0, self.L0_min), self.L0_max[i])
            self.L0[i] = L0
            m.tendon_lengthspring[t, 1] = L0
            self._set_line_k(i)

    def step(self, n=1):
        for _ in range(n):
            if self.nstep % self.ctrl_every == 0:
                self._control()
            self._motors()
            mujoco.mj_step(self.m, self.d)
            self.nstep += 1

    @property
    def time(self):
        return self.nstep * self.dt

    def _control(self):
        for h in self.tick_hooks:
            h(self)
        if self.log is not None:
            self.log.append(self.telemetry())
        ex = self.exec
        if ex is None or ex["finished"]:
            self.duty = [0.0] * 6
            return
        st = ex["seq"][ex["k"]]
        L_mm = (self.d.ten_length[self.ids.short] * 1000.0).tolist()
        slack = None
        if ex["slack_rule"]:
            Lc = self.d.ten_length[self.ids.cable]
            slack = [bool(Lc[i] < self.L0[i] - 1e-4) for i in range(6)]
        self.duty = self.pid.update(L_mm, st["target"], st["ranges"], st["tol"], st["low_tol"], slack)
        t_in = self.time - ex["t0"]
        complete = self.pid.state_complete()
        if complete or t_in > ex["timeout"]:
            ex["log"].append({"state": ex["k"], "phase": st["phase"], "target": st["target"], "ranges": st["ranges"],
                              "tol": st["tol"], "duration": t_in, "timeout": not complete, "lengths_mm": L_mm,
                              "done": list(self.pid.done), "slack_zeroed": list(self.pid.zeroed),
                              "nodes": self.nodes().tolist()})
            ex["k"] += 1
            self.pid.reset()
            ex["t0"] = self.time
            self.duty = [0.0] * 6
            if ex["k"] >= len(ex["seq"]):
                ex["finished"] = True
            elif ex["redetect"] and ex["seq"][ex["k"]]["phase"] == "primitive":
                b = gaits.ground_endcaps(self.nodes())
                if gaits.SYMMETRY.get(b) is not None:
                    ms = gaits.map_state(ex["base"][ex["k"]], b)
                    ex["seq"][ex["k"]]["target"] = gaits.reverse_state(ms, b) if ex.get("reverse") else ms

    # ---------------- gait execution ----------------
    def start_states(self, states, left=100, right=100, tol=0.15, low_tol=0.15, bottom=None, timeout=None,
                     redetect=False, reverse=False, raw=False, rest_phase=None, slack_rule=False):
        """Execute a list of base target states (six normalised targets each, motor order of the authors' code).

        Symmetry map: chosen once from the three lowest end caps at the start (hardware: run_tensegrity_Astar.py
        lines 556-561) or, with redetect=True, from the lower end cap of each rod at the start of every state (the
        authors' simulator). A bottom triple outside the authors' table raises gaits.NotRollableFace (ysim 0.3.0:
        no silent fallback). raw=True: the states are motor targets as given (no mapping, no face check; for
        recovery sequences). reverse=True re-indexes the mapped states with the authors' reverse mapping.
        rest_phase=(tol, low_tol): append the authors' re-plan phase, [1]*6 at RANGE 100 with that band.
        slack_rule=True: the authors' simulator slack rule (controller.FirmwarePID.update)."""
        nodes = self.nodes()
        if raw:
            b = None
            mapped = [list(map(float, s)) for s in states]
        else:
            b = tuple(bottom) if bottom is not None else (gaits.ground_endcaps(nodes) if redetect
                                                           else metrics.bottom3(nodes))
            if gaits.SYMMETRY.get(b) is None:
                raise gaits.NotRollableFace("bottom triple %s is not a rollable face %s" % (b, gaits.ROLLABLE_FACES))
            mapped = [gaits.map_state(s, b) for s in states]
            if reverse:
                mapped = [gaits.reverse_state(s, b) for s in mapped]
        rng = [left] * 3 + [right] * 3
        seq = [{"target": s, "ranges": list(rng), "tol": tol, "low_tol": low_tol, "phase": "primitive"} for s in mapped]
        if rest_phase is not None:
            r0 = prm.value("default_range") * 1000.0
            seq.append({"target": list(gaits.REST), "ranges": [r0] * 6, "tol": rest_phase[0],
                        "low_tol": rest_phase[1], "phase": "rest"})
        self.pid.reset()
        self.exec = {"base": [list(s) for s in states], "seq": seq, "redetect": redetect, "k": 0, "t0": self.time,
                     "finished": not seq, "log": [], "bottom": b, "timeout": timeout or prm.value("state_timeout"),
                     "reverse": bool(reverse), "slack_rule": bool(slack_rule)}

    def run_until_done(self, t_max=60.0):
        n = 0
        nmax = int(t_max / self.dt)
        while not self.exec["finished"] and n < nmax:
            self.step(self.ctrl_every)
            n += self.ctrl_every
        return self.exec["finished"]

    def settle(self, t=1.0, vtol=None):
        """Hold (brake) for t seconds; if vtol is given, stop early once the max node speed < vtol for 0.5 s."""
        self.exec = None
        n = int(round(t / self.dt))
        quiet = 0
        for _ in range(0, n, self.ctrl_every):
            self.step(self.ctrl_every)
            if vtol is not None:
                v = self.max_node_speed()
                quiet = quiet + self.ctrl_every if v < vtol else 0
                if quiet * self.dt >= 0.5:
                    break

    def max_node_speed(self):
        vmax = 0.0
        for k, b in enumerate(self.ids.body):
            dof = self.ids.jnt_dof[k]
            v = self.d.qvel[dof:dof + 3]
            w = self.d.qvel[dof + 3:dof + 6]
            vmax = max(vmax, float(np.linalg.norm(v)) + float(np.linalg.norm(w)) * prm.value("node_offset"))
        return vmax

    def mode_bands(self, mode=None):
        """bands(mode) with this simulator's controller history (the band the last roll set)."""
        return bands(mode, self.code_tol)

    def stand(self, t_hold=2.0, t_settle=15.0, vtol=5e-4, mode=None):
        """Standing procedure: settle with the motors braked, the rest phase of the execution mode (all six short
        sensors to the rest target [1]*6 at RANGE 100 = 200 mm, as at the end of every primitive), then settle
        until every node moves slower than vtol for 0.5 s (the prism's twist mode relaxes within a few seconds)."""
        self.settle(t_hold)
        _, _, rest, slack = self.mode_bands(mode)
        self.start_states([], raw=True, rest_phase=rest, slack_rule=slack)
        ok = self.run_until_done(30.0)
        ok = ok and not any(s["timeout"] for s in self.exec["log"])
        self.settle(t_settle, vtol=vtol)
        return ok

    def run_primitive(self, name, mode=None, settle_after=None, tol=None, min_length_mm=None, grouping="triangle",
                      reverse=False, end_rest=None):
        """Run one named primitive from the current rest state; returns a result dict (see the package README).

        mode "hardware": the authors' control code: band 0.15 (0.35 for rolls with a range >= 130; a turn and the
        rest phase keep the band of the last roll, params code_tol_history), no slack rule.
        mode "tracked": the same targets, ranges, PID and motors with the band params.track_tol for every target
        plus the authors' simulator slack rule (params slack_rule).
        Default mode: params execution_mode. Both end with the authors' re-plan rest phase (end_rest).
        mode "authors_sim": the authors' simulator definition (gaits.SIM_*; diagnostics; no rest phase).
        tol / min_length_mm / grouping override the bands and lengths (diagnostics only; grouping "parity" = motors
        0,2,4 left as in hybrid_mppi get_3bar_prims_se2_data).
        reverse=True (rolls only, as in the authors' control code): the authors' reverse gait (gaits.REVERSE).
        settle_after: braked hold after the rest phase [s]; None = params settle_after (ASSUMPTION: the authors'
        re-planning delay, one 7 Hz pose period + the planner's mean time).
        Raises gaits.NotRollableFace if the robot does not rest on a rollable face."""
        p = gaits.get(name)
        if reverse and p.kind != "roll":
            raise ValueError("the authors' code reverses rolls only (run_tensegrity_Astar.py lines 561-563)")
        mode = mode or prm.value("execution_mode")
        if mode not in MODES:
            raise ValueError(mode)
        if end_rest is None:
            end_rest = mode != "authors_sim"
        n0 = self.nodes()
        b0 = metrics.bottom3(n0)
        t0 = self.time
        e0 = sum(mo.energy_elec for mo in self.motors)
        self.peak_tension = 0.0
        saved = (self.pid.min_length, self.pid.P, self.pid.variant)
        states, left, right, redetect = p.states(), p.left, p.right, False
        new_code_tol = self.code_tol
        if mode == "hardware":
            if p.kind == "roll":
                new_code_tol = p.tol                    # lines 569-572 (set by every roll)
            t_hi, t_lo, slack = new_code_tol, prm.value("pid_low_tol"), False
            rest = (new_code_tol, t_lo)
        elif mode == "tracked":
            t_hi, t_lo, rest, slack = self.mode_bands("tracked")
        else:
            states = [list(x) for x in gaits.SIM_BASE[p.kind]]
            left, right = gaits.SIM_RANGES.get(p.kind, (p.left, p.right))
            t_hi = t_lo = gaits.SIM_TOL
            rest, slack, redetect = (gaits.SIM_TOL, gaits.SIM_TOL), True, True
        if tol is not None:
            t_hi = t_lo = tol
            rest = (tol, tol)
        self.start_states(states, left, right, t_hi, t_lo, redetect=redetect, reverse=reverse,
                          rest_phase=rest if end_rest else None, slack_rule=slack)
        if mode == "hardware":
            self.code_tol = new_code_tol
        if mode == "authors_sim":
            self.pid.min_length = gaits.SIM_MIN_LENGTH[p.kind]
            self.pid.P = gaits.SIM_P
            self.pid.variant = "authors_sim"
        if min_length_mm is not None:
            self.pid.min_length = min_length_mm
        if grouping == "parity":
            for s in self.exec["seq"]:
                if s["phase"] == "primitive":
                    s["ranges"] = [left if i % 2 == 0 else right for i in range(6)]
        ok = self.run_until_done(120.0)
        self.pid.min_length, self.pid.P, self.pid.variant = saved
        log = self.exec["log"]
        t_exec = self.time - t0
        self.settle(prm.value("settle_after") if settle_after is None else settle_after)
        n1 = self.nodes()
        b1 = metrics.bottom3(n1)
        dx, dy, dth, dist = metrics.relative(n0, n1)
        prim_log = [s for s in log if s["phase"] == "primitive"]
        rest_log = [s for s in log if s["phase"] == "rest"]
        pre = None
        if prim_log and rest_log:
            n_pre = np.array(prim_log[-1]["nodes"])
            pdx, pdy, pdth, pdist = metrics.relative(n0, n_pre)
            pre = {"dx": pdx, "dy": pdy, "dtheta_deg": pdth, "dist": pdist, "bottom": metrics.bottom3(n_pre)}
        Ls = self.sensor_lengths() * 1000.0
        return {"name": name, "mode": mode, "reverse": bool(reverse), "finished": ok, "dx": dx, "dy": dy,
                "dtheta_deg": dth, "dist": dist, "t_exec": t_exec, "t_total": self.time - t0, "states": log,
                "timeouts": sum(1 for s in log if s["timeout"]),
                "rest_timeout": bool(rest_log and rest_log[-1]["timeout"]),
                "rest_reached": bool(rest_log and not rest_log[-1]["timeout"]),
                "t_rest_phase": float(rest_log[-1]["duration"]) if rest_log else 0.0,
                "bottom_start": b0, "bottom_end": b1, "rollable_end": b1 in gaits.SYMMETRY,
                "end_triangle": b1 in gaits.END_TRIANGLES, "se2_before_rest_phase": pre,
                "top_z_end": self.top_z(), "com_z_end": float(self.com()[2]),
                "short_sensors_end_mm": Ls[:6].tolist(), "long_sensors_end_mm": Ls[6:].tolist(),
                "energy_J": sum(mo.energy_elec for mo in self.motors) - e0,
                "peak_cable_tension_N": self.peak_tension, "nodes_start": n0.tolist(), "nodes_end": n1.tolist()}

    # ---------------- telemetry (what the real robot reports) ----------------
    def telemetry(self):
        L = self.sensor_lengths()
        Lmm = L * 1000.0
        cap = [self.cal_m[i] * Lmm[i] + self.cal_b[i] for i in range(9)]
        sd = self.d.sensordata
        imu = []
        for k in range(3):
            a = sd[self.ids.acc[k]:self.ids.acc[k] + 3] / G0
            g = np.degrees(sd[self.ids.gyr[k]:self.ids.gyr[k] + 3])
            imu.append({"accel_g": a.tolist(), "gyro_dps": g.tolist()})
        return {"t": self.time, "sensor_length_m": L.tolist(), "capacitance": cap,
                "encoder_counts": [mo.encoder_counts() for mo in self.motors], "imu": imu,
                "duty": list(self.duty), "motor_energy_J": [mo.energy_elec for mo in self.motors]}

    def energy_report(self):
        cap = prm.value("motor_pack_capacity")
        per = [0.0, 0.0, 0.0]
        for i, (mot, line) in enumerate(geom.CABLES):
            per[geom.node_strut(mot)] += self.motors[i].energy_elec
        return {"motor_energy_J": [mo.energy_elec for mo in self.motors],
                "mech_work_J": [mo.work_mech for mo in self.motors],
                "per_strut_J": per, "per_strut_fraction_of_motor_pack": [e / cap for e in per]}

    # ---------------- determinism ----------------
    def state_hash(self):
        """SHA-256 of the full MuJoCo integration state + cable rest lengths and stiffnesses + motor and controller
        state (idea taken from an earlier kit that is not published)."""
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        buf = np.zeros(mujoco.mj_stateSize(self.m, spec))
        mujoco.mj_getState(self.m, self.d, buf, spec)
        h = hashlib.sha256()
        h.update(buf.tobytes())
        h.update(np.array(self.L0).tobytes())
        h.update(np.array(self.k_line).tobytes())
        h.update(np.array([s for mo in self.motors for s in mo.state()]).tobytes())
        h.update(np.array(self.pid.prev + self.pid.cum + self.duty + [self.code_tol], float).tobytes())
        return h.hexdigest()


def _read_calibration():
    """The authors' sensor calibration (rss25 calibration/new_calibration.json, copied as data)."""
    import json
    from . import paths
    with open(paths.data_path("authors_calibration.json")) as fh:
        d = json.load(fh)
    return list(d["m"]), list(d["b"])
