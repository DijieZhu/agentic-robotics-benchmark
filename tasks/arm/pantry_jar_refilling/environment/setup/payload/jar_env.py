"""Pantry jar refilling: the closed_loop environment (two arms).

One episode is one refill list on one counter (`jar_sim.py`) with one unit: the jars' hidden properties (thread
turns and running torque, mass, friction, sizes) are the unit's; the rest (where each jar stands, the list, the tray's
pieces) comes from the seed (`jar_sim.draw_episode`).

Control at 10 Hz. An action is a JSON object {"l": [...], "r": [...], "cams": mask}; each arm's list is one of:
- 7 numbers [dx, dy, dz, drx, dry, drz, grip]: the operational-space controller's step: the hand's target moves by
  (dx, dy, dz) m (at most MAX_DPOS) and turns by the rotation vector (drx, dry, drz) rad (at most MAX_DROT), both in
  the world frame, from the hand's current pose; grip -1 open .. +1 closed;
- 14 numbers: the same and seven joint angles, the posture the controller's null space pulls the arm toward from now
  on (the start posture until set);
- 8 numbers [q1..q7, grip]: a joint step: a gravity-compensated joint PD toward q, each joint's target at most MAX_DQ
  from its current angle; q also becomes the null-space posture.
`cams` is a bitmask of the images the next observation carries: 1 overhead, 2 front, 4 tray, 8 left wrist, 16 right
wrist.

The observation: t; per arm ("l", "r"): q, dq (7), ee_pos, ee_quat (wxyz), gripper (the fingers' opening, m),
wrench (the wrist's force and torque in the world frame, with noise); each camera's pose T_world_<cam> (4 x 4,
OpenCV convention: x right, y down, z forward) and intrinsics K_<cam>; the requested images (the others empty); the
refill list {kind: pieces}.
"""

import base64
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import jar_sim as S  # noqa: E402
from jar_sim import RenderError  # noqa: E402,F401 - for the grader

CAMS = tuple(S.CAMERA_BITS)
TORQUE_NOISE = S.WRENCH_NOISE / 10


def encode_image(img):
    return {"dtype": "uint8", "shape": list(img.shape), "b64": base64.b64encode(np.ascontiguousarray(img)).decode()}


def decode_image(d):
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=np.uint8).reshape(d["shape"])


def to_wire(obs):
    out = {}
    for k, v in obs.items():
        if k in CAMS:
            out[k] = encode_image(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def check_action(action):
    """The action as {"l": floats, "r": floats, "cams": int}; ValueError if malformed."""
    if not isinstance(action, dict) or set(action) - {"l", "r", "cams"} or not {"l", "r"} <= set(action):
        raise ValueError("malformed action")
    out = {}
    for s in ("l", "r"):
        a = [float(x) for x in action[s]]
        if len(a) not in (7, 8, 14) or not all(math.isfinite(x) for x in a):
            raise ValueError("malformed action")
        out[s] = a
    cams = action.get("cams", 0)
    if cams != int(cams) or not 0 <= int(cams) <= S.ALL_CAMS:
        raise ValueError("malformed action")
    out["cams"] = int(cams)
    return out


def to_arm_command(a):
    """An arm's action as jar_sim.Arm.command expects: 7 -> 8 (osc + grip), 14 -> 15, 8 -> 9 (joint)."""
    if len(a) == 7:
        return a[:7] + [0.0]
    if len(a) == 14:
        return a[:7] + [0.0] + a[7:14]
    return a[:8] + [0.0]


class JarEnv:
    def __init__(self, unit, unit_id="dev", render=True):
        self.unit = unit
        self.unit_id = unit_id
        self.render_images = render

    def reset(self, seed):
        self.seed = int(seed)
        self.episode = S.draw_episode(self.seed, self.unit)
        self.w = S.World(self.unit, self.episode)
        self.rng = np.random.default_rng([self.seed % 2 ** 32, self.seed >> 32, 31])
        self.render_next = S.ALL_CAMS
        self.faulted = False
        self.refill = dict(self.episode["refill"])
        return self.observation()

    @property
    def t(self):
        return self.w.t

    def step(self, action):
        """Apply one action (see the module's doc); returns the observation. A malformed action raises ValueError;
        an unstable simulation sets `faulted` (the episode then scores 0)."""
        a = check_action(action)
        self.render_next = a["cams"]
        try:
            self.w.step({s: to_arm_command(a[s]) for s in ("l", "r")})
        except FloatingPointError:
            self.faulted = True
        return self.observation()

    def unstable(self):
        return bool(self.faulted or self.w.unstable())

    def camera_matrix(self, cam):
        w, h, fovy = S.camera_size(cam)
        return S.intrinsics(fovy, w, h)

    def camera_pose(self, cam, data=None):
        d = self.w.d if data is None else data
        R, pc = S.cam_pose(d, self.w.m.camera(S.MJ_CAMERA[cam]).id)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, pc
        return T

    def arm_obs(self, s):
        a, d = self.w.arms[s], self.w.d
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, a.R().flatten())
        wr = a.wrench_world()
        noise = np.concatenate([self.rng.normal(0, S.WRENCH_NOISE, 3), self.rng.normal(0, TORQUE_NOISE, 3)])
        return {"q": d.qpos[a.jq].copy(), "dq": d.qvel[a.jv].copy(), "ee_pos": a.pos(), "ee_quat": q,
                "gripper": a.gripper(), "wrench": wr + noise}

    def observation(self):
        obs = {"t": int(self.w.t), "refill": self.refill, "l": self.arm_obs("l"), "r": self.arm_obs("r")}
        for cam in CAMS:
            obs[f"T_world_{cam}"] = self.camera_pose(cam)
            obs[f"K_{cam}"] = self.camera_matrix(cam)
            if self.render_images and self.render_next & S.CAMERA_BITS[cam]:
                obs[cam] = S.render(self.w.m, self.w.d, cam, self.rng)
            else:
                obs[cam] = np.zeros((0, 0, 3), np.uint8)
        return obs

    def perturbed_observation(self, obs, rng):
        """The probe lock's other copy: each arm's joints offset by up to 1 mrad, the hand's and the wrist camera's
        poses recomputed from them, each force offset by up to 0.2 N; images unchanged."""
        w, m = self.w, self.w.m
        out = dict(obs)
        d = mujoco.MjData(m)
        d.qpos[:] = w.d.qpos
        for s in ("l", "r"):
            a = w.arms[s]
            d.qpos[a.jq] = np.asarray(obs[s]["q"], float) + rng.uniform(-0.001, 0.001, 7)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        for s in ("l", "r"):
            a = w.arms[s]
            arm = dict(obs[s])
            Rs = d.site_xmat[a.site].reshape(3, 3)
            qq = np.empty(4)
            mujoco.mju_mat2Quat(qq, Rs.flatten())
            wr = np.asarray(obs[s]["wrench"], float).copy()
            wr[:3] += rng.uniform(-0.2, 0.2, 3)
            arm.update(q=d.qpos[a.jq].copy(), ee_pos=d.site_xpos[a.site].copy(), ee_quat=qq, wrench=wr)
            out[s] = arm
            out[f"T_world_wrist_{s}"] = self.camera_pose(f"wrist_{s}", d)
        return out

    def truth(self):
        """The full state, for a token run only (the reference's shadow): the unit, the seed, the simulation state
        and the threads' state (which lid is in its thread, armed to re-enter it, has been opened, its closing
        torque's peak since it last entered)."""
        w, d = self.w, self.w.d
        return {"unit": self.unit, "seed": self.seed, "t": int(w.t), "qpos": d.qpos.tolist(), "qvel": d.qvel.tolist(),
                "engaged": [bool(x) for x in w.engaged], "armed": [bool(x) for x in w.armed],
                "opened": [bool(x) for x in w.opened], "seat_peak": [float(x) for x in w.seat_peak],
                "jar_start": [p.tolist() for p in w.jar_start]}

    def close(self):
        """Nothing to release (the renderer is made and closed per image)."""
