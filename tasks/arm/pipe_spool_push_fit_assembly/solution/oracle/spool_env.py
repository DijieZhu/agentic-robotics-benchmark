"""Pipe spool push-fit assembly: the closed_loop environment (two arms).

One episode builds one pipe spool (`spool_sim.py`) from one unit's parts: two copper pipes lying loose and two push-fit
elbows on the bench. The unit's hidden properties (insertion force, stop depth, turning torque, bore clearance, elbow
mass, pipe mass per metre) are the unit's; the rest (the pipes' lengths and places, A's banded end, the elbows' places,
poses and turns, the work order's twist) comes from the seed (`spool_sim.draw_episode`).

Control at 10 Hz. An action is a JSON object {"l": [...], "r": [...], "cams": mask}; each arm's list is one of:
- 7 numbers [dx, dy, dz, drx, dry, drz, grip]: the operational-space controller's step: the hand's target moves by
  (dx, dy, dz) m (at most MAX_DPOS) and turns by the rotation vector (drx, dry, drz) rad (at most MAX_DROT), both in
  the world frame, from the hand's current pose; grip -1 is open, +1 closed (squeezing with the grasp force);
- 14 numbers: the same and seven joint angles, the posture the controller's null space pulls the arm toward from now
  on (the start posture until set);
- 8 numbers [q1..q7, grip]: a joint step: a gravity-compensated joint PD toward q, each joint's target at most MAX_DQ
  from its current angle; q also becomes the null-space posture.
Joint angles in an action are clipped to the joints' limits. `cams` is a bitmask of the images the next observation
carries: 1 overhead, 2 front, 4 side, 8 left wrist, 16 right wrist.

The observation: t; order ({"twist_deg"}); per arm ("l", "r"): q, dq (7), ee_pos, ee_quat (wxyz) of the grip site,
wrench (the wrist's force and torque in the world frame, with noise), gripper (the fingers' opening, m); each camera's
pose T_world_<cam> (4 x 4, OpenCV convention: x right, y down, z forward) and intrinsics K_<cam>; the requested
images (the others empty).
"""

import base64
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import spool_sim as S  # noqa: E402
from spool_sim import RenderError  # noqa: E402,F401 - for the grader

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


class SpoolEnv:
    def __init__(self, unit, unit_id="dev", render=True):
        self.unit = unit
        self.unit_id = unit_id
        self.render_images = render
        self.renderers = {}

    def reset(self, seed):
        self.close()
        self.seed = int(seed)
        self.episode = S.draw_episode(self.seed, self.unit)
        self.w = S.World(self.unit, self.episode)
        self.rng = np.random.default_rng([self.seed % 2 ** 32, self.seed >> 32, 31])
        self.render_next = S.ALL_CAMS
        self.faulted = False
        self.true_wrench = {}
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
            self.w.step({s: a[s] for s in ("l", "r")})
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
        self.true_wrench[s] = wr.copy()                       # for the probe lock's copy (never on the wire)
        noise = np.concatenate([self.rng.normal(0, S.WRENCH_NOISE, 3), self.rng.normal(0, TORQUE_NOISE, 3)])
        out = {"q": d.qpos[a.jq].copy(), "dq": d.qvel[a.jv].copy(), "ee_pos": a.pos(), "ee_quat": q,
               "wrench": wr + noise}
        out["gripper"] = a.gripper()
        return out

    def observation(self):
        obs = {"t": int(self.w.t), "order": {"twist_deg": int(self.episode["twist_deg"])},
               "l": self.arm_obs("l"), "r": self.arm_obs("r")}
        for cam in CAMS:
            obs[f"T_world_{cam}"] = self.camera_pose(cam)
            obs[f"K_{cam}"] = self.camera_matrix(cam)
            if self.render_images and self.render_next & S.CAMERA_BITS[cam]:
                obs[cam] = S.render(self.w, cam, self.rng, self.renderers)
            else:
                obs[cam] = np.zeros((0, 0, 3), np.uint8)
        return obs

    def perturbed_observation(self, obs, rng):
        """The probe lock's other copy: each arm's joints offset by up to 1 mrad, the hand's and the wrist camera's
        poses recomputed from them, each wrist's wrench read once more (the true wrench of this observation with noise
        drawn anew, as a second reading of the sensor would be); images unchanged. A real observation's poses
        are those of the joint angles one physics substep before its `q` (MuJoCo computes the poses at the start of
        the last substep), so the copy's poses are computed the same way: from that substep's joint angles moved by
        the same offsets."""
        w, m = self.w, self.w.m
        out = dict(obs)
        d = mujoco.MjData(m)
        d.qpos[:] = w.d.qpos
        dq = {}
        for s in ("l", "r"):
            a = w.arms[s]
            dq[s] = rng.uniform(-0.001, 0.001, 7)
            d.qpos[a.jq] = w.q_prev[s] + dq[s]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        for s in ("l", "r"):
            a = w.arms[s]
            arm = dict(obs[s])
            Rs = d.site_xmat[a.site].reshape(3, 3)
            qq = np.empty(4)
            mujoco.mju_mat2Quat(qq, Rs.flatten())
            wr = self.true_wrench[s] + np.concatenate([rng.normal(0, S.WRENCH_NOISE, 3), rng.normal(0, TORQUE_NOISE, 3)])
            arm.update(q=np.asarray(obs[s]["q"], float) + dq[s], ee_pos=d.site_xpos[a.site].copy(), ee_quat=qq,
                       wrench=wr)
            out[s] = arm
            out[f"T_world_wrist_{s}"] = self.camera_pose(f"wrist_{s}", d)
        return out

    def truth(self):
        """The full state, for a token run only (the reference's shadow): the unit, the seed and the world's state
        (spool_sim.World.state)."""
        return dict(self.w.state(), unit=self.unit, seed=self.seed)

    def close(self):
        S.close_renderers(self.renderers)
