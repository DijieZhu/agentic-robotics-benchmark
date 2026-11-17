"""Plant: the real unit in a real cell (grader side only; never agent-readable).

    p = Plant(design_xml, unit_spec, cell, tool)        # cells.Cell or None; the cell's clamped tool or None
    p.power_up()                                      # the home pose commanded THROUGH THE UNIT'S RECORD, settle
    p.cycle(goal_ticks, speed=None, acc=None)          # one 10 ms bus cycle: queue the goals, step 5 x 2 ms, monitor
    p.read()                                           # what the bus returns: ticks, speed (ticks/s), load (per mille)
    p.tcp(), p.tool_tip()                              # TRUE poses (grader measurements)

Physics: the unit's MJCF (`unit.unit_mjcf`, implicitfast, 2 ms) with the cell's true geometry and the clamped tool
compiled in (its geoms collide with the cell), the servo firmware layer (`gt_arm.servo.ServoBus`: tick quantisation,
EEPROM clamp, latency FIFO, trapezoidal profile) extended with the goal-speed and acceleration registers (whole
numbers) and a load register that holds its whole reading for 20 ms, and the series compliance with its play operator
(as `ctl_info.sim`). The home pose and the jaw's clamp are commanded through the unit's calibration RECORD (as an
operator's tool would), so the first readings say nothing about the record's error.

Monitoring: CONTACT (every physics step) = any active penetration deeper than PEN_M between the robot (links, the
clamped part, the needle holder and needle) and the cell / table, or between non-adjacent robot bodies. CLEARANCE (every
cycle, on a probe copy whose cell geoms report contacts up to unit.CLEAR_DETECT_M away) = the smallest distance between a
robot geom and a cell geom, with per-job exemptions (the job's support).
"""

import copy
import math
import os

import numpy as np

from arm.gt_arm import servo as S

from . import unit as U

CYCLE_S = 0.010
STEPS_PER_CYCLE = 5
PEN_M = 1e-5
HOME_Q = {"base_yaw": 0.0, "shoulder_pitch": 0.35, "elbow_pitch": -1.45, "wrist_pitch": -1.25, "wrist_roll": 0.0, "jaw": 0.15}
ACC_UNIT_TICKS_S2 = 100.0
COND_L_M = 0.1                        # the characteristic length that scales the approach-pitch row of the task Jacobian
TOOL_GEOMS = ("part_geom", "holder_geom", "needle_geom")


def exempt_pairs(model, geom_obj, exempt):
    """{(robot geom id, cell geom id)} for exempt = [{"who": "robot" | "part" | "needle", "what": object or primitive}];
    geom_obj = {cell geom id: object name}. Geom ids are the same in a unit's physics and probe models."""
    import mujoco
    m = model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
    robot = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
    ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in TOOL_GEOMS}
    who = {"part": [ids["part_geom"]], "needle": [ids["needle_geom"], ids["holder_geom"]], "robot": robot}
    out = set()
    for e in exempt:
        for rg in who[e["who"]]:
            if rg < 0:
                continue
            for cg, obj in geom_obj.items():
                if obj == e["what"] or names[cg] == "cell_" + e["what"]:
                    out.add((rg, cg))
    return out


def record_ticks(spec, q):
    """Ticks of design angles through the unit's calibration RECORD (what a tool that trusts the record commands)."""
    rec = spec["record"]["servos"]
    out = []
    for j, qj in zip(U.JOINTS, q):
        r = rec[j]
        t = int(round(r["zero_ticks"] + r["direction"] * float(qj) / U.TICK_RAD))
        out.append(min(max(t, r["min_ticks"]), r["max_ticks"]))
    return np.array(out, int)


class Bus(S.ServoBus):
    """ServoBus with per-servo goal-speed (ticks/s, 0 = max) and acceleration (x 100 ticks/s^2, 0 = default) registers
    (whole numbers) and a load register refreshed every LOAD_REFRESH_S (the whole reading is held)."""

    def __init__(self, *a, **k):
        super(Bus, self).__init__(*a, **k)
        self.acc = np.full(len(self.joints), S.ACCEL_RAD_S2)
        self._load_hold = (None, None)

    def command_regs(self, t_now, goal_ticks, speed=None, acc=None):
        gt = self.clamp_ticks(np.asarray(goal_ticks, int))
        spd = None if speed is None else np.floor(np.abs(np.asarray(speed, float)))
        ac = None if acc is None else np.floor(np.abs(np.asarray(acc, float)))
        self.fifo.append((float(t_now) + self.latency, gt, spd, ac))
        return gt

    def _release(self, t):
        while self.fifo and self.fifo[0][0] <= t + 1e-12:
            item = self.fifo.pop(0)
            gt, spd = item[1], item[2]
            acc = item[3] if len(item) > 3 else None
            self.goal = S.ticks_to_bus(gt)
            if spd is None:
                self.vlim = self.vmax.copy()
            else:
                v = spd * S.TICK_RAD
                self.vlim = np.where(v > 0, np.minimum(v, self.vmax), self.vmax)
            if acc is None:
                self.acc = np.full(len(self.joints), S.ACCEL_RAD_S2)
            else:
                a = acc * ACC_UNIT_TICKS_S2 * S.TICK_RAD
                self.acc = np.where(a > 0, np.minimum(a, S.ACCEL_RAD_S2), S.ACCEL_RAD_S2)

    def before_step(self, t, dt):
        self._release(t)
        d = self.goal - self.sp
        vstop = np.sqrt(2.0 * self.acc * np.abs(d))
        vdes = np.sign(d) * np.minimum(self.vlim, vstop)
        dv = np.clip(vdes - self.sv, -self.acc * dt, self.acc * dt)
        self.sv = self.sv + dv
        step = self.sv * dt
        arrive = (np.sign(step) == np.sign(d)) & (np.abs(step) >= np.abs(d))
        self.sp = np.where(arrive, self.goal, self.sp + step)
        self.sv = np.where(arrive | (np.abs(self.goal - self.sp) < 1e-12), 0.0, self.sv)
        return self.bus_to_q(self.sp)

    def loads(self, data, t):
        k = int(math.floor(t / S.LOAD_REFRESH_S + 1e-9))
        if self._load_hold[0] != k:
            self._load_hold = (k, super(Bus, self).loads(data, t))
        return self._load_hold[1].copy()

    def speed_ticks(self, data):
        return np.rint(self.sign * data.qvel[self.dadr] / S.TICK_RAD).astype(int)


class Plant(object):
    def __init__(self, design_xml, spec, cell=None, tool=None, design=None):
        import mujoco
        self.mj = mujoco
        self.spec = copy.deepcopy(spec)
        self.cell = cell
        self.tool = copy.deepcopy(tool) if tool else None
        self.design = design if design is not None else U.load_design(os.path.dirname(os.path.abspath(design_xml)))
        self.xml = U.unit_mjcf(design_xml, spec, cell, tool=self.tool, design=self.design)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.probe_model = mujoco.MjModel.from_xml_string(U.unit_mjcf(design_xml, spec, cell, probe=True, tool=self.tool, design=self.design))
        self.probe_data = mujoco.MjData(self.probe_model)
        m = self.model
        self.dt = float(m.opt.timestep)
        J = U.JOINTS
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in J]
        self.qadr = np.array([m.jnt_qposadr[i] for i in jid])
        self.dadr = np.array([m.jnt_dofadr[i] for i in jid])
        sj = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j + "_slack") for j in U.ARM_JOINTS]
        self.slack_qadr = np.array([m.jnt_qposadr[i] for i in sj])
        self.slack_dadr = np.array([m.jnt_dofadr[i] for i in sj])
        self.act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_" + j) for j in J])
        self.tcp_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.pitch_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "wrist_pitch")
        self.robot_geom = np.array([m.geom_bodyid[g] != 0 for g in range(m.ngeom)])
        self.geom_object = {}
        if cell is not None:
            for g in cell.geoms():
                self.geom_object[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g["name"])] = g["object"]
        self.table_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        D = spec["D"]
        self.k_slack = np.array([float(D["compliance_nm_per_rad"][j]) for j in U.ARM_JOINTS])
        self.h_slack = np.radians([float(D["hysteresis_deg"][j]) for j in U.ARM_JOINTS])
        self.K_slack = np.array([float(m.jnt_stiffness[i]) for i in sj])
        self.play = np.zeros(5)
        self.bus = Bus(m, J, U.bus_cal(spec), U.dynamics(spec), self.qadr, self.dadr, self.act, spec["D"]["load_noise_seed"])
        self.seat = U.tool_seat(self.design, self.tool) if self.tool else {"seat_x": 0.0, "seat_z": 0.0, "jaw_q": None}
        # the jaw while a tool is clamped: the clamp angle commanded through the record (the true angle differs by its error)
        self.jaw_clamp = None
        self.jaw_clamp_ticks = None
        if self.seat["jaw_q"] is not None:
            q = np.zeros(6)
            q[5] = self.seat["jaw_q"]
            t = self.bus.clamp_ticks(record_ticks(self.spec, q))
            self.jaw_clamp_ticks = int(t[5])
            b = S.ticks_to_bus(t)
            self.jaw_clamp = float(self.bus.bus_to_q(b)[5])
        self.needle_len = float(self.tool["params"]["length_m"]) if (self.tool and self.tool["kind"] == "needle") else 0.0
        self.t = 0.0
        self.exempt = set()
        self.reset_monitor()
        self.recording = False
        self.log = []

    # ------------------------------------------------------------------------------------------------------ tools
    def tool_tip(self):
        """(position, R) of the active tool point: the needle tip (the needle lies along the TCP's +z, at the seat's
        offset along x), else the TCP."""
        p, R = self.tcp()
        if self.needle_len > 0:
            return p + R[:, 0] * float(self.seat["seat_x"]) + R[:, 2] * self.needle_len, R
        return p, R

    def set_exempt(self, exempt):
        """exempt = [{"who": "robot" | "part" | "needle", "what": object or primitive name}]: those robot-to-cell
        distances do not count as clearance (contact still does)."""
        self.exempt = exempt_pairs(self.probe_model, self.geom_object_all(), exempt)

    def geom_object_all(self):
        d = dict(self.geom_object)
        d[self.table_geom] = "table"
        return d

    # --------------------------------------------------------------------------------------------------- stepping
    def power_up(self, q=None, settle_s=1.0, keep_time=False):
        """The arm powered up at the home pose (or q), commanded through the unit's RECORD: the true pose is the home
        pose plus the record's error, as on a real unit after its homing routine."""
        mj = self.mj
        t_keep = self.t
        mj.mj_resetData(self.model, self.data)
        qd = np.array([(q or HOME_Q)[j] for j in U.JOINTS], float)
        if self.seat["jaw_q"] is not None:
            qd[5] = self.seat["jaw_q"]
        b = S.ticks_to_bus(self.bus.clamp_ticks(record_ticks(self.spec, qd)))
        qt = self.bus.bus_to_q(b)
        self.data.qpos[self.qadr] = qt
        self.data.ctrl[self.act] = qt
        self.bus.fifo = []
        self.bus.sp = b.copy()
        self.bus.goal = b.copy()
        self.bus.sv = np.zeros(len(b))
        self.bus.vlim = self.bus.vmax.copy()
        self.bus.acc = np.full(len(b), S.ACCEL_RAD_S2)
        self.bus._load_hold = (None, None)
        self.t = t_keep if keep_time else 0.0
        self.centre_play()
        for _ in range(int(round(settle_s / self.dt))):
            self._step(monitor=False)
        self.reset_monitor()

    def _slack_load(self):
        return -self.data.qfrc_bias[self.slack_dadr]

    def _apply_play(self):
        g = self._slack_load()
        eq = g / self.k_slack
        self.play = np.minimum(np.maximum(self.play, eq - self.h_slack), eq + self.h_slack)
        self.model.qpos_spring[self.slack_qadr] = self.play - g / self.K_slack

    def centre_play(self, iters=3):
        mj = self.mj
        for _ in range(iters):
            mj.mj_forward(self.model, self.data)
            g = self._slack_load()
            self.play = g / self.k_slack
            self.data.qpos[self.slack_qadr] = self.play
        self.model.qpos_spring[self.slack_qadr] = self.play - self._slack_load() / self.K_slack
        mj.mj_forward(self.model, self.data)

    def reset_monitor(self):
        self.contact = None
        self.min_clear = float("inf")
        self.min_clear_pair = None

    def _monitor(self):
        """Contact check of one physics step: any penetration deeper than PEN_M involving the robot."""
        d, m = self.data, self.model
        if self.contact is not None:
            return
        for k in range(d.ncon):
            c = d.contact[k]
            if float(c.dist) >= -PEN_M:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, r2 = self.robot_geom[g1], self.robot_geom[g2]
            if not (r1 or r2):
                continue
            cellg = g2 if r1 and not r2 else (g1 if r2 and not r1 else None)
            mj = self.mj
            bodies = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "world" for g in (g1, g2)]
            self.contact = {"t": round(self.t, 4), "kind": "cell" if cellg is not None else "self", "pen_mm": round(-float(c.dist) * 1e3, 4),
                            "bodies": bodies, "object": (self.geom_object.get(cellg, "table") if cellg is not None else None)}
            return

    def clearance_now(self):
        """Smallest robot-to-cell distance now (m; capped at unit.CLEAR_DETECT_M), the exemptions skipped, and the pair."""
        mj, pm, pd = self.mj, self.probe_model, self.probe_data
        pd.qpos[:] = self.data.qpos
        mj.mj_kinematics(pm, pd)
        mj.mj_collision(pm, pd)
        best, pair = U.CLEAR_DETECT_M, None
        for k in range(pd.ncon):
            c = pd.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, r2 = self.robot_geom[g1], self.robot_geom[g2]
            if r1 == r2:
                continue
            robg, cellg = (g1, g2) if r1 else (g2, g1)
            if (robg, cellg) in self.exempt:
                continue
            if float(c.dist) < best:
                best, pair = float(c.dist), (robg, self.geom_object.get(cellg, "table" if cellg == self.table_geom else "?"))
        return best, pair

    def _step(self, monitor=True):
        q_set = self.bus.before_step(self.t, self.dt)
        self.data.ctrl[self.act] = q_set
        if self.jaw_clamp is not None:
            self.data.ctrl[self.act[5]] = self.jaw_clamp
        self._apply_play()
        self.mj.mj_step(self.model, self.data)
        self.t = round(self.t + self.dt, 9)
        if monitor:
            self._monitor()

    def cycle(self, goal_ticks=None, speed=None, acc=None):
        """One bus cycle. goal_ticks None = no new command (the servos keep their goals)."""
        if goal_ticks is not None:
            if self.jaw_clamp_ticks is not None:     # servo 6 holds the clamp: its goal is ignored (no profile starts)
                goal_ticks = list(goal_ticks)
                goal_ticks[5] = self.jaw_clamp_ticks
            self.bus.command_regs(self.t, goal_ticks, speed, acc)
        for _ in range(STEPS_PER_CYCLE):
            self._step()
        c, pair = self.clearance_now()
        if c < self.min_clear:
            self.min_clear, self.min_clear_pair = c, pair
        if self.recording:
            self.log.append(self.measure())

    def hold(self, seconds):
        for _ in range(int(round(seconds / CYCLE_S))):
            self.cycle(None)

    # --------------------------------------------------------------------------------------------------- readings
    def read(self):
        return {"t": round(self.t, 4), "position": [int(v) for v in self.bus.encoder_ticks(self.data)],
                "speed": [int(v) for v in self.bus.speed_ticks(self.data)],
                "load": [int(v) for v in self.bus.loads(self.data, self.t)], "moving": bool(self.bus.moving(self.data))}

    def tcp(self):
        return self.data.site_xpos[self.tcp_id].copy(), self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()

    def q(self):
        return self.data.qpos[self.qadr].copy()

    def q_link(self):
        q = self.q()
        q[:5] += self.data.qpos[self.slack_qadr]
        return q

    def conditioning(self):
        """Smallest singular value of the task Jacobian at the tool point: position rows and the approach-pitch row
        (scaled by COND_L_M), columns = base yaw, shoulder, elbow, wrist pitch (the output hinges)."""
        mj, m, d = self.mj, self.model, self.data
        p, R = self.tool_tip()
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        body = int(m.site_bodyid[self.tcp_id])
        mj.mj_jac(m, d, jp, jr, p, body)
        cols = self.dadr[:4]
        axis = d.xaxis[self.pitch_jid].copy()
        axis /= np.linalg.norm(axis)
        Jt = np.vstack([jp[:, cols], COND_L_M * (axis @ jr[:, cols])[None, :]])
        return float(np.linalg.svd(Jt, compute_uv=False)[-1])

    def measure(self):
        p, R = self.tool_tip()
        return {"t": round(self.t, 4), "tip": p.tolist(), "axis": R[:, 2].tolist(), "jaw_axis": R[:, 0].tolist(),
                "q": self.q().tolist(), "qd": self.data.qvel[self.dadr].tolist(), "cond": self.conditioning(),
                "clear": self.min_clear}
