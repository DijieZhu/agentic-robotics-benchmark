"""Cabinet door hanging: the closed_loop environment (two arms).

One episode replaces one cabinet's door (`door_sim.py`) of one unit: the old door, hung on two old pins, comes off and
the new door goes on. The unit's hidden properties (knuckle count, pin class, clearance, hinge heights, door and
handle mass) are the unit's; the rest (the carcass's pose, the racks' places, which stands hold which pins) comes
from the seed (`door_sim.draw_episode`).

Control at 10 Hz. An action is a JSON object {"l": [...], "r": [...], "cams": mask}; each arm's list is one of:
- 7 numbers [dx, dy, dz, drx, dry, drz, grip]: the operational-space controller's step: the hand's target moves by
  (dx, dy, dz) m (at most MAX_DPOS) and turns by the rotation vector (drx, dry, drz) rad (at most MAX_DROT), both in
  the world frame, from the hand's current pose. For the left arm grip -1 is open, +1 closed; for the right arm grip
  > 0 asks for vacuum, <= 0 releases it;
- 14 numbers: the same and seven joint angles, the posture the controller's null space pulls the arm toward from now
  on (the start posture until set);
- 8 numbers [q1..q7, grip]: a joint step: a gravity-compensated joint PD toward q, each joint's target at most MAX_DQ
  from its current angle; q also becomes the null-space posture.
`cams` is a bitmask of the images the next observation carries: 1 overhead, 2 front, 4 side, 8 left wrist, 16 right
wrist (the cup tool's camera).

The observation: t; per arm ("l", "r"): q, dq (7), ee_pos, ee_quat (wxyz) of the tool's frame (the gripper's grip
site; the cup's face centre, its z axis along the cup), wrench (the wrist's force and torque in the world frame, with
noise); the left arm's gripper (the fingers' opening, m), the right arm's vacuum (true while the cup holds a door);
each camera's pose T_world_<cam> (4 x 4, OpenCV convention: x right, y down, z forward) and intrinsics K_<cam>; the
requested images (the others empty).
"""

import base64
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import door_sim as S  # noqa: E402
from door_sim import RenderError  # noqa: E402,F401 - for the grader

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


class DoorEnv:
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
        noise = np.concatenate([self.rng.normal(0, S.WRENCH_NOISE, 3), self.rng.normal(0, TORQUE_NOISE, 3)])
        out = {"q": d.qpos[a.jq].copy(), "dq": d.qvel[a.jv].copy(), "ee_pos": a.pos(), "ee_quat": q,
               "wrench": wr + noise}
        if a.tool == "gripper":
            out["gripper"] = a.gripper()
        else:
            out["vacuum"] = bool(self.w.held is not None)
        return out

    def observation(self):
        obs = {"t": int(self.w.t), "l": self.arm_obs("l"), "r": self.arm_obs("r")}
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
        """The full state, for a token run only (the reference's shadow): the unit, the seed and the world's state
        (door_sim.World.state)."""
        return dict(self.w.state(), unit=self.unit, seed=self.seed)

    def close(self):
        S.close_renderers(self.renderers)
