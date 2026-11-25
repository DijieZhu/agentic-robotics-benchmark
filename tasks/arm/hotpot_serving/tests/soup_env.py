"""Hotpot serving: the closed_loop environment (two arms).

One episode is one order at one counter (`soup_sim.py`) with one unit: the pot's, the broth's and the burner's hidden
properties (the pot's mass, how much broth it holds and how fast it pours, the burner's full power, the pieces' size
and friction) are the unit's; the rest (which bowl stands where, the order, the pot's place, which dish holds which
kind, the pieces on the dishes) comes from the seed (`soup_sim.draw_episode`).

Control at 10 Hz. An action is a JSON object {"l": [...], "r": [...], "cams": mask}; each arm's list is one of:
- 7 numbers [dx, dy, dz, drx, dry, drz, grip]: the operational-space controller's step: the hand's target moves by
  (dx, dy, dz) m (at most MAX_DPOS) and turns by the rotation vector (drx, dry, drz) rad (at most MAX_DROT), both in
  the world frame, from the hand's current pose; grip -1 open .. +1 closed;
- 14 numbers: the same and seven joint angles, the posture the controller's null space pulls the arm toward from now
  on (the start posture until set);
- 8 numbers [q1..q7, grip]: a joint step: a gravity-compensated joint PD toward q, each joint's target at most MAX_DQ
  from its current angle; q also becomes the null-space posture.
Joint angles in an action (a posture, a joint step's q) are clipped to the joints' limits.
`cams` is a bitmask of the images the next observation carries: 1 overhead, 2 front, 4 left wrist, 8 right wrist.

The observation: t, unit, order ({bowl: {"items": {kind: n}, "soup_ml": v}}, kind in soup_sim.KINDS), knob (the
knob's angle, rad: 0 off .. KNOB_MAX full); per arm ("l",
"r"): q, dq (7), ee_pos, ee_quat (wxyz), gripper (the fingers' opening, m), wrench (the wrist's force and torque in
the world frame, with noise); each camera's pose T_world_<cam> (4 x 4, OpenCV convention: x right, y down,
z forward) and intrinsics K_<cam>; the requested images (the others empty).
"""

import base64
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import soup_sim as S  # noqa: E402
from soup_sim import RenderError  # noqa: E402,F401 - for the grader

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
    """An arm's action as soup_sim.Arm.command expects: 7 -> 8 (osc + grip), 14 -> 15, 8 -> 9 (joint)."""
    if len(a) == 7:
        return a[:7] + [0.0]
    if len(a) == 14:
        return a[:7] + [0.0] + a[7:14]
    return a[:8] + [0.0]


class SoupEnv:
    def __init__(self, unit, unit_id="dev", render=True):
        self.unit = unit
        self.unit_id = unit_id
        self.render_images = render
        self.renderers = {}                     # one renderer per image size, for this episode's model (soup_sim.render)

    def reset(self, seed):
        self.seed = int(seed)
        self.episode = S.draw_episode(self.seed, self.unit)
        S.close_renderers(self.renderers)       # a new episode is a new model
        self.w = S.World(self.unit, self.episode)
        self.rng = np.random.default_rng([self.seed % 2 ** 32, self.seed >> 32, 31])
        self.render_next = S.ALL_CAMS
        self.faulted = False
        self.order = {k: {"items": dict(v["items"]), "soup_ml": v["soup_ml"]} for k, v in self.episode["order"].items()}
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
        obs = {"t": int(self.w.t), "unit": self.unit_id, "order": self.order, "knob": round(self.w.knob(), 4),
               "l": self.arm_obs("l"), "r": self.arm_obs("r")}
        for cam in CAMS:
            obs[f"T_world_{cam}"] = self.camera_pose(cam)
            obs[f"K_{cam}"] = self.camera_matrix(cam)
            if self.render_images and self.render_next & S.CAMERA_BITS[cam]:
                obs[cam] = S.render(self.w.m, self.w.d, cam, self.rng, self.renderers)
            else:
                obs[cam] = np.zeros((0, 0, 3), np.uint8)
        return obs

    def perturbed_observation(self, obs, rng):
        """The probe lock's other copy: each arm's joints offset by up to 1 mrad, the hand's and the wrist camera's
        poses moved by what those offsets move them, each force offset by up to 0.2 N; images unchanged.

        The observed hand pose is the simulation's own, computed one physics step before the observed joint angles,
        so it is not exactly the forward kinematics of `q`. The copy keeps that relation: its poses are the observed
        ones moved by the difference between the kinematics of the offset joints and of the observed joints. (A copy
        whose poses were the exact kinematics of its joints could be told from the real observation by anyone holding
        the model.)"""
        w, m = self.w, self.w.m
        out = dict(obs)
        d0, d1 = mujoco.MjData(m), mujoco.MjData(m)           # kinematics of the observed and of the offset joints
        d0.qpos[:] = w.d.qpos
        d1.qpos[:] = w.d.qpos
        for s in ("l", "r"):
            a = w.arms[s]
            q = np.asarray(obs[s]["q"], float)
            d0.qpos[a.jq] = q
            d1.qpos[a.jq] = q + rng.uniform(-0.001, 0.001, 7)
        for d in (d0, d1):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
        for s in ("l", "r"):
            a = w.arms[s]
            arm = dict(obs[s])
            R0, R1 = d0.site_xmat[a.site].reshape(3, 3), d1.site_xmat[a.site].reshape(3, 3)
            Robs = np.empty(9)
            mujoco.mju_quat2Mat(Robs, np.asarray(obs[s]["ee_quat"], float))
            qq = np.empty(4)
            mujoco.mju_mat2Quat(qq, (R1 @ R0.T @ Robs.reshape(3, 3)).flatten())
            wr = np.asarray(obs[s]["wrench"], float).copy()
            wr[:3] += rng.uniform(-0.2, 0.2, 3)
            arm.update(q=d1.qpos[a.jq].copy(), ee_quat=qq, wrench=wr,
                       ee_pos=np.asarray(obs[s]["ee_pos"], float) + d1.site_xpos[a.site] - d0.site_xpos[a.site])
            out[s] = arm
            cam = f"wrist_{s}"
            T0, T1 = self.camera_pose(cam, d0), self.camera_pose(cam, d1)
            T = np.array(obs[f"T_world_{cam}"], float)
            T[:3, :3] = T1[:3, :3] @ T0[:3, :3].T @ T[:3, :3]
            T[:3, 3] += T1[:3, 3] - T0[:3, 3]
            out[f"T_world_{cam}"] = T
        return out

    def truth(self):
        """The full state, for a token run only (the reference's shadow): the unit, the seed, the simulation state
        and the broth's state (each vessel's volume, the stream's parcels, the spill, the temperatures)."""
        w, d = self.w, self.w.d
        return {"unit": self.unit, "seed": self.seed, "t": int(w.t), "qpos": d.qpos.tolist(), "qvel": d.qvel.tolist(),
                "liquid": w.liquid.state(), "pot_lifted": bool(w.pot_lifted)}

    def close(self):
        """Close the episode's renderers."""
        S.close_renderers(self.renderers)
