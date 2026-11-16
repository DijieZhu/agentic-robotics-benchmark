"""One episode of the house-cleaning cell: model, controller, physical rules, cleaning processes, cameras, budgets and
the episode's score.

The public surface (what a program can learn) is `describe`, `get_state`, `command`, `step`, `get_camera`, `finish`;
everything else here is the grader's (the true object poses, the grease, the sponge's and the crumbs' states, the
score).
"""

from __future__ import annotations

import math
import time

import mujoco
import numpy as np

from . import cleaning as CL
from . import layout as LAYOUT
from . import objects as O
from . import room as RM
from .control import TICK, V_MAX, RobotController
from .physics import (GRIP_MAX_FORCE, GRIP_MAX_TORQUE, GRIP_MAX_TORQUE_BAR, SAFETY, Clips, Fragile, Grips, Mechanisms,
                      Safety)
from .render import CameraRig, camera_pose_world, encode_frame, intrinsics
from .robot import (ARM_HOME, ARM_JOINTS, D435_PITCH, D435_POS, GRASP_SITE_POS, HAND_JOINTS, HEAD_CAM, PELVIS_HEIGHT,
                    SIDES, STAND_RANGE, WAIST_JOINTS, robot_index)
from .scene import ROBOT_NAME, build
from .walk_core import LegWalker, support_from_geom

SUBSTEPS = 5                               # 5 x 2 ms physics per 10 ms control tick
LOC_NOISE = (0.003, math.radians(0.2))     # localisation noise (m, rad), per reading
# the motor state get_state reports for every motor a program can read (the waist's 3, the arms' 2 x 7, the hands'
# 2 x 7), as a G1's low-level state does (the body's LowState and the Dex3-1 hands' HandState, one entry per motor):
#   q        the joint angle, exact (the encoders' resolution is far below anything the task needs);
#   dq       the joint velocity, with reading-to-reading noise (DQ_NOISE rad/s, 1 sigma), quantised to DQ_STEP;
#   tau_est  the joint torque estimated from the motor's current: the drive's torque (the servo's plus the gravity
#            feed-forward that holds the robot's own links), read with a calibration error that stays for the episode
#            (a scale error within TAU_GAIN and an offset of TAU_OFFSET x the joint's torque limit, 1 sigma) and
#            reading-to-reading noise (TAU_NOISE x the limit, 1 sigma), quantised to TAU_STEP
TAU_GAIN = 0.05
TAU_OFFSET = 0.015
TAU_NOISE = 0.005
TAU_STEP = 0.01
DQ_NOISE = 0.01
DQ_STEP = 0.001
PUBLIC_CAMERAS = ("ceiling_kitchen", "ceiling_dining", "ceiling_living", "head")
API = "house-cell/1.0"


def _cam_mjname(name: str) -> str:
    return f"{ROBOT_NAME}/head_cam" if name == "head" else name


def _drawn_supports(m: mujoco.MjModel) -> list:
    """What the drawn feet stand on rather than in: the rug as the cameras draw it."""
    G = mujoco.mjtObj.mjOBJ_GEOM
    (cx, cy), (sx, sy) = RM.RUG["center"], RM.RUG["size"]
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    out = []
    for g in range(m.ngeom):
        if not (mujoco.mj_id2name(m, G, g) or "").startswith("decor_") or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(m.geom_dataid[g])
        V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
        V = V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        c = V[:, :2].mean(axis=0)
        if V[:, 2].min() > -0.005 and V[:, 2].max() < 0.03 and abs(c[0] - cx) < sx / 2 and abs(c[1] - cy) < sy / 2:
            out.append(support_from_geom(m, g))
    if not out:
        g = mujoco.mj_name2id(m, G, "rug")
        if g >= 0:
            out.append(support_from_geom(m, g))
    return out


class Cell:
    def __init__(self, seed: int, *, nominal: bool = False, render: bool = True, config: dict | None = None):
        cfg = config or {}
        self.seed = int(seed)
        self.cfg = cfg
        self.layout = LAYOUT.draw(self.seed, nominal=nominal)
        self.spec, self.meta = build(self.layout)
        self.m = self.spec.compile()
        self.d = mujoco.MjData(self.m)
        self.robot = robot_index(self.m, ROBOT_NAME)
        self.ctrl = RobotController(self.m, self.robot)
        self.grips = Grips(self.m, self.d, self.robot, self.meta)
        self.clips = Clips(self.m, self.d, self.meta, self.grips)
        self.mech = Mechanisms(self.m, self.d, self.meta)
        self.fragile = Fragile(self.m, self.d)
        self.safety = Safety(self.m, self.d, self.robot, self.grips)
        self.rng = np.random.default_rng((self.seed * 2654435761 + 97) & ((1 << 64) - 1))
        self._tau_init()
        self.process = CL.Process(self.m, self.d, self.meta, self.layout, self.grips, self.mech,
                                  np.random.default_rng(self.rng.integers(1 << 62)))
        B = mujoco.mjtObj.mjOBJ_BODY
        self.body = {n: mujoco.mj_name2id(self.m, B, n) for n in O.LOOSE}
        sizes = {n: RM.CEILING_CAMERAS[n]["size"] for n in RM.CEILING_CAMERAS}
        sizes[_cam_mjname("head")] = HEAD_CAM["size"]
        self.cams = CameraRig(self.m, sizes, np.random.default_rng(self.rng.integers(1 << 62))) if render else None
        sup = _drawn_supports(self.m) if render else []
        self.walker = LegWalker(self.m, f"{ROBOT_NAME}/", supports=sup) if render else None
        self._q_kin = None
        self._rd = None
        self.sim_cap = float(cfg.get("sim_cap_s", 4800.0))
        self.frame_budget = int(cfg.get("camera_budget_frames", 1500))
        self.max_ticks = int(cfg.get("max_ticks_per_step", 500))
        self.frames = {}
        self.frames_total = 0
        self.render_wall = 0.0
        self.work_wall = 0.0
        self.ticks = 0
        self.done = False
        self.done_reason = None
        self._reset()

    # ------------------------------------------------------------------------------------------------ setup
    def _reset(self):
        m, d, r = self.m, self.d, self.robot
        mujoco.mj_resetData(m, d)
        x, y, yaw = self.layout["robot"]
        d.qpos[r.base_q[:3]] = (x, y, yaw)
        d.qpos[r.base_q[3]] = 0.0
        for s in SIDES:
            d.qpos[r.arm_q[s]] = ARM_HOME[s]
        for name, (pos, quat) in self.layout["objects"].items():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
            a = m.jnt_qposadr[jid]
            d.qpos[a:a + 3] = pos
            d.qpos[a + 3:a + 7] = quat
        mujoco.mj_forward(m, d)
        for tool in RM.CLIPS:
            self.clips.engage(tool, 0.0)
        self.clips.events.clear()
        if self.walker is not None:
            self._q_kin = d.qpos.copy()
        self.ctrl.reset_targets(d)
        for _ in range(int(float(self.cfg.get("settle_s", 0.6)) / TICK)):
            self._tick(record=False)
        self.ticks = 0
        self.t0 = float(d.time)
        self.safety.violations.clear()
        self.safety.count = 0
        for k in self.safety.peaks:
            self.safety.peaks[k] = 0.0
        self.safety.peak_who.clear()
        self.fragile.broken.clear()
        self.process.wipe_s = self.process.sweep_s = 0.0
        if self.walker is not None:
            self.walker.reset(self.sim_time, d.qpos[self.walker.base_adr])

    @property
    def sim_time(self) -> float:
        return self.ticks * TICK

    def _tick(self, record: bool = True):
        m, d = self.m, self.d
        for s in SIDES:
            self.ctrl.hand_shape[s] = self.grips.shape(s)
        self.ctrl.tick(d)
        self.mech.pre_step()
        for i in range(SUBSTEPS):
            if i == SUBSTEPS - 1 and self.walker is not None:
                self._q_kin[:] = d.qpos
            mujoco.mj_step(m, d)
        t = self.sim_time
        self.grips.tick(t)
        self.clips.tick(t)
        stroke = self.mech.post_step()
        self.process.tick(stroke)
        if record:
            self.fragile.tick(t, self.grips.held())
            self.safety.tick(t)
            self.ticks += 1
            if self.walker is not None:
                self.walker.step(self.sim_time, d.qpos[self.walker.base_adr])

    # ------------------------------------------------------------------------------------------------ public API
    def describe(self) -> dict:
        joints = {"waist": [f"{w}_joint" for w in WAIST_JOINTS],
                  "left_arm": [f"left_{w}_joint" for w in ARM_JOINTS],
                  "right_arm": [f"right_{w}_joint" for w in ARM_JOINTS],
                  "left_hand": [f"left_hand_{w}_joint" for w in HAND_JOINTS],
                  "right_hand": [f"right_hand_{w}_joint" for w in HAND_JOINTS]}
        ranges = {}
        for names in joints.values():
            for jn in names:
                j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{ROBOT_NAME}/" + jn)
                ranges[jn] = [float(self.m.jnt_range[j, 0]), float(self.m.jnt_range[j, 1])]
        cams = {}
        for n, c in RM.CEILING_CAMERAS.items():
            cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, n)
            p, R = camera_pose_world(self.m, self.d, cid)
            cams[n] = {"intrinsics": intrinsics(c["size"][0], c["size"][1], c["fovy"]),
                       "pose_in_room": {"position": [float(v) for v in p], "rotation": R.tolist()},
                       "depth_min_m": 0.40}
        cams["head"] = {"intrinsics": intrinsics(HEAD_CAM["size"][0], HEAD_CAM["size"][1], HEAD_CAM["fovy"]),
                        "mount": {"body": "torso_link", "position": list(D435_POS), "pitch_down_rad": D435_PITCH,
                                  "optical_axis": "body +x pitched down"},
                        "depth_min_m": 0.28,
                        "note": "every head frame carries the camera's pose in the room at capture time"}
        return {
            "api": API,
            "robot": {"model": "unitree_g1_29dof_dex3 (cell variant)", "start_hint": "just inside the door",
                      "legs_in_images": "the cameras draw the legs stepping while the base moves (drawn only)",
                      "motor_state": {"joints": list(self.tau_names),
                                      "fields": {"q": "rad, exact", "dq": "rad/s, noisy",
                                                 "tau_est": "N m, estimated from the motor current"},
                                      "torque_limits_Nm": {n: float(v) for n, v in zip(self.tau_names, self.tau_limit)},
                                      "note": "get_state()['robot']['q'], ['dq'] and ['tau_est']: {joint name: value} "
                                              "for every one of these motors, as the G1's low-level state reports "
                                              "them (cell_manual.md section 6)"}},
            "joints": joints, "joint_ranges": ranges,
            "control": {"tick_s": TICK, "max_ticks_per_step": self.max_ticks, "base_velocity_max": list(V_MAX),
                        "stand_height_range": list(STAND_RANGE), "stand_height_nominal": PELVIS_HEIGHT,
                        "arm_joint_rate_rad_s": 1.5, "waist_rate_rad_s": 0.8,
                        "grasp_site": {"body": "<side>_wrist_yaw_link",
                                       "position": {s: list(GRASP_SITE_POS[s]) for s in SIDES},
                                       "axes": "x = approach (along the fingers), y = pinch axis, z = x cross y"}},
            "grip": {"max_force_N": GRIP_MAX_FORCE, "max_torque_Nm": GRIP_MAX_TORQUE,
                     "max_torque_bar_Nm": GRIP_MAX_TORQUE_BAR,
                     "rule": "see cell_manual.md section 4: closing a hand whose grasp site is at a grasp feature of "
                             "an object (a bar or cylinder, a slab's edge, a disc's rim, a lump) holds the object"},
            "cameras": cams, "camera_budget_frames": self.frame_budget,
            "room": room_survey(),
            "budgets": {"sim_cap_s": self.sim_cap,
                        "program_time_cap_s": float(self.cfg.get("program_time_cap_s", 3600.0)),
                        "program_time": "the episode's wall clock minus the cell's own time inside step (simulation) "
                                        "and get_camera (rendering)",
                        "silence_s": float(self.cfg.get("silence_s", 120.0)),
                        "cpu_time_cap_s": float(self.cfg.get("cpu_cap_factor", 4.0))
                        * float(self.cfg.get("program_time_cap_s", 3600.0)),
                        "cpu_time": "the program process's CPU time, all threads (SIGXCPU beyond it); OMP_NUM_THREADS, "
                                    "OPENBLAS_NUM_THREADS and MKL_NUM_THREADS are 2"},
            "safety": {"robot_contact_N": SAFETY["robot_contact_N"], "robot_contact_s": SAFETY["robot_contact_ticks"]
                       * TICK, "throw_speed_m_s": SAFETY["throw_speed"], "strike_speed_m_s": SAFETY["strike_speed"]},
        }

    def _localised(self):
        r = self.robot
        x, y, yaw = (float(v) for v in self.d.qpos[r.base_q[:3]])
        return (x + float(self.rng.normal(0, LOC_NOISE[0])), y + float(self.rng.normal(0, LOC_NOISE[0])),
                yaw + float(self.rng.normal(0, LOC_NOISE[1])))

    def _tau_init(self):
        """The motors whose state get_state reports (the waist's, the arms' and the hands'), their joints' position and
        velocity addresses, their drives' torque limits, and the episode's calibration errors: drawn from a generator
        of their own (the localisation noise and the scene's draws are untouched)."""
        m, r = self.m, self.robot
        names = [f"{w}_joint" for w in WAIST_JOINTS] + [f"{s}_{w}_joint" for s in SIDES for w in ARM_JOINTS] \
            + [f"{s}_hand_{w}_joint" for s in SIDES for w in HAND_JOINTS]
        qadr = list(r.waist_q) + [a for s in SIDES for a in r.arm_q[s]] + [a for s in SIDES for a in r.hand_q[s]]
        acts = list(r.waist_act) + [a for s in SIDES for a in r.arm_act[s]] + [a for s in SIDES for a in r.hand_act[s]]
        dofs = [int(m.jnt_dofadr[m.actuator_trnid[a, 0]]) for a in acts]
        self.tau_names = names
        self.tau_qadr = np.array(qadr, int)
        self.tau_dof = np.array(dofs, int)
        self.tau_act = np.array(acts, int)
        self.tau_limit = np.array([float(m.actuator_forcerange[a, 1]) for a in acts])
        g = np.random.default_rng([self.seed & ((1 << 63) - 1), 0x7A0E57])
        self.tau_gain = 1.0 + g.uniform(-TAU_GAIN, TAU_GAIN, len(names))
        self.tau_offset = g.normal(0.0, TAU_OFFSET, len(names)) * self.tau_limit
        self.tau_rng = np.random.default_rng(g.integers(1 << 62))
        self.dq_rng = np.random.default_rng(g.integers(1 << 62))

    def tau_true(self) -> np.ndarray:
        """The drives' torques now (N m): the servo's force plus the gravity compensation of the robot's links (the
        real robot's motors carry both)."""
        return self.d.actuator_force[self.tau_act] + self.d.qfrc_gravcomp[self.tau_dof]

    def tau_est(self) -> dict:
        tau = self.tau_true() * self.tau_gain + self.tau_offset \
            + self.tau_rng.normal(0.0, 1.0, len(self.tau_names)) * TAU_NOISE * self.tau_limit
        tau = np.round(tau / TAU_STEP) * TAU_STEP
        return {n: round(float(v), 3) for n, v in zip(self.tau_names, tau)}

    def dq_est(self) -> dict:
        dq = self.d.qvel[self.tau_dof] + self.dq_rng.normal(0.0, DQ_NOISE, len(self.tau_names))
        dq = np.round(dq / DQ_STEP) * DQ_STEP
        return {n: round(float(v), 4) for n, v in zip(self.tau_names, dq)}

    def get_state(self) -> dict:
        d, r, c = self.d, self.robot, self.ctrl
        q = {n: float(d.qpos[a]) for n, a in zip(self.tau_names, self.tau_qadr)}
        hands = {}
        for s in SIDES:
            hands[s] = {"closed": c.hand_cmd[s] == "close", "tactile_N": round(self.grips.tactile(s), 2),
                        "grasp_pose_in_robot": self._grasp_in_base(s),
                        "ik_residual": [round(c.arm_ik_err[s][0], 5), round(c.arm_ik_err[s][1], 5)]
                        if c.arm_mode[s] == "pose" else None}
        return {"sim_time": round(self.sim_time, 4), "done": self.done,
                "robot": {"base_pose": [round(v, 5) for v in self._localised()],
                          "stand_height": round(PELVIS_HEIGHT + float(d.qpos[r.base_q[3]]), 4),
                          "base_moving": bool(np.linalg.norm(c.vel) > 1e-4),
                          "q": {k: round(v, 5) for k, v in q.items()}, "dq": self.dq_est(), "tau_est": self.tau_est(),
                          "hands": hands}}

    def _grasp_in_base(self, side):
        d, r = self.d, self.robot
        x, y, yaw = (float(v) for v in d.qpos[r.base_q[:3]])
        c, s = math.cos(yaw), math.sin(yaw)
        Rb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        sid = r.grasp_site[side]
        p = Rb.T @ (d.site_xpos[sid] - np.array([x, y, 0.0]))
        Rs = Rb.T @ d.site_xmat[sid].reshape(3, 3)
        qq = np.zeros(4)
        mujoco.mju_mat2Quat(qq, Rs.reshape(-1))
        return [round(float(v), 5) for v in p] + [round(float(v), 6) for v in qq]

    def command(self, cmd: dict) -> dict:
        notes = self.ctrl.command(cmd)
        for s in SIDES:
            h = cmd.get(f"{s}_hand")
            if h in ("open", "close"):
                self.grips.command(s, h == "close", self.sim_time)
        return {"ok": True, "notes": notes}

    def step(self, n_ticks: int) -> dict:
        events = []
        n_viol = len(self.safety.violations)
        t = time.time()
        try:
            for _ in range(int(n_ticks)):
                if self.done:
                    break
                self._tick()
                if self.sim_time >= self.sim_cap - 1e-9:
                    self._end("sim_cap")
                    events.append("episode_over")
                    break
        finally:
            self.work_wall += time.time() - t
        for v in self.safety.violations[n_viol:]:
            events.append("safety:" + v["what"])
        return {"sim_time": round(self.sim_time, 4), "done": self.done, "events": events}

    def get_camera(self, name: str, depth: bool = True) -> dict:
        if name not in PUBLIC_CAMERAS:
            return {"error": "bad_params"}
        if self.frames_total >= self.frame_budget:
            return {"error": "camera_budget"}
        self.frames_total += 1
        self.frames[name] = self.frames.get(name, 0) + 1
        t_frame = time.time()
        try:
            return self._frame(name, depth)
        finally:
            self.work_wall += time.time() - t_frame

    def _frame(self, name: str, depth: bool) -> dict:
        t = time.time()
        mjname = _cam_mjname(name)
        kind = "head" if name == "head" else "ceiling"
        rgb, dep = self.cams.render(self.render_state(), mjname, depth=depth, kind=kind)
        self.render_wall += time.time() - t
        if name == "head":
            w, h = HEAD_CAM["size"]
            intr = intrinsics(w, h, HEAD_CAM["fovy"])
        else:
            c = RM.CEILING_CAMERAS[name]
            intr = intrinsics(c["size"][0], c["size"][1], c["fovy"])
        out = encode_frame(name, self.frames[name] - 1, rgb, dep, intr)
        cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, mjname)
        p, R = camera_pose_world(self.m, self.d, cid)
        if name == "head":
            r = self.robot
            x, y, yaw = (float(v) for v in self.d.qpos[r.base_q[:3]])
            lx, ly, lyaw = self._localised()
            dyaw = lyaw - yaw
            cz, sz = math.cos(dyaw), math.sin(dyaw)
            Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
            p = Rz @ (p - np.array([x, y, 0.0])) + np.array([lx, ly, 0.0])
            R = Rz @ R
        out["pose_in_room"] = {"position": [float(v) for v in p], "rotation": R.tolist()}
        out["sim_time"] = round(self.sim_time, 4)
        return out

    def finish(self) -> dict:
        if not self.done:
            self._end("finish")
        return {"ok": True}

    def render_state(self) -> mujoco.MjData:
        """The state every camera image is drawn from: the physics state's kinematics with the robot's 12 leg joints as
        its walker draws them now (everything else is the physics' own)."""
        if self.walker is None:
            return self.d
        if self._rd is None:
            self._rd = mujoco.MjData(self.m)
        rd = self._rd
        rd.qpos[:] = self._q_kin
        rd.mocap_pos[:] = self.d.mocap_pos
        rd.mocap_quat[:] = self.d.mocap_quat
        rd.time = self.d.time
        w = self.walker
        rd.qpos[w.qpos_adr] = w.pose_at(self.sim_time, self._q_kin[w.base_adr])
        mujoco.mj_kinematics(self.m, rd)
        mujoco.mj_comPos(self.m, rd)
        mujoco.mj_camlight(self.m, rd)
        return rd

    def _end(self, reason: str):
        self.done = True
        self.done_reason = reason

    # ------------------------------------------------------------------------------------------------ grader side
    def pose(self, name):
        b = self.body[name]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def score(self) -> dict:
        from .scoring import score_cell
        return score_cell(self)


def room_survey() -> dict:
    """The room's fixed geometry as describe() publishes it (the cell manual's section 1 says the same)."""
    K, S, B = RM.KITCHEN, RM.SINK, RM.BIN
    return {
        "frame": "origin at the south-west floor corner, x east, y north, z up, metres",
        "size": [RM.ROOM_X, RM.ROOM_Y, RM.WALL_H], "door_x": list(RM.DOOR_X), "window_y": list(RM.WINDOW_Y),
        "kitchen": {"cabinets": [K["x0"], K["y0"], 0.0, K["top_x1"], K["y1"], K["top_z"]], "top_z": K["top_z"],
                    "toe_kick_x1": K["plinth_x1"],
                    "sink_basin": [S["x0"], S["y0"], S["bottom_z"], S["x1"], S["y1"], K["top_z"]],
                    "faucet": {k: (list(v) if isinstance(v, tuple) else v) for k, v in RM.FAUCET.items()},
                    "soap_pump": {k: (list(v) if isinstance(v, tuple) else v) for k, v in RM.SOAP.items()},
                    "sponge_tray": {"center": list(RM.SPONGE_TRAY["center"]), "size": list(RM.SPONGE_TRAY["size"]),
                                    "rim_h": RM.SPONGE_TRAY["rim_h"]},
                    "hob": list(RM.HOB), "wall_cabinets": [RM.UPPER[k] for k in ("x0", "y0", "z0", "x1", "y1", "z1")]},
        "bins": {"slots_y": list(RM.BIN_SLOTS_Y), "x": [B["x0"], B["x1"]], "half_y": B["half_y"], "rim_z": B["rim_z"],
                 "wall_t": B["wall_t"], "lid": {"hinge_x": B["hinge_x"], "hinge_z": B["rim_z"] + 0.002,
                                                "thickness": B["lid_t"], "range_rad": list(RM.LID_RANGE),
                                                "hold_open_above_rad": RM.LID_HOLD,
                                                "tab": {"radius": B["tab_r"], "length": B["tab_len"],
                                                        "beyond_lid_front": B["tab_out"]}},
                 "note": "which slot holds the blue (recycling) bin and which the grey (general waste) one is seen "
                         "in the cameras"},
        "tool_rail": {"x": [RM.TOOL_RAIL["x0"], RM.TOOL_RAIL["x1"]], "z": RM.TOOL_RAIL["z"],
                      "clips": {k: list(v) for k, v in RM.CLIPS.items()}, "capture_m": RM.CLIP_CAPTURE},
        "dining_table": {"center": list(RM.DINING["center"]), "size": list(RM.DINING["size"]),
                         "top_z": RM.DINING["top_z"], "leg": RM.DINING["leg"], "leg_inset": RM.DINING["leg_inset"]},
        "chairs": {k: {"center": list(v["center"]), "yaw": v["yaw"]} for k, v in RM.CHAIRS.items()},
        "chair": {k: (list(v) if isinstance(v, tuple) else v) for k, v in RM.CHAIR.items()},
        "coffee_table": {"center": list(RM.COFFEE["center"]), "size": list(RM.COFFEE["size"]),
                         "top_z": RM.COFFEE["top_z"]},
        "bookshelf": {k: (list(v) if isinstance(v, tuple) else v) for k, v in RM.SHELF.items()},
        "book_target": dict(RM.BOOK_TARGET),
        "couch": {"footprint_xy": list(RM.COUCH["footprint"]),
                  "boxes": {k: list(v) for k, v in RM.COUCH["boxes"].items()}},
        "plant": {k: {"center": list(v["center"]), "radius": v["radius"], "z": list(v["z"])}
                  for k, v in RM.PLANT.items()},
        "rug": {"center": list(RM.RUG["center"]), "size": list(RM.RUG["size"]), "note": "drawn only"},
    }
