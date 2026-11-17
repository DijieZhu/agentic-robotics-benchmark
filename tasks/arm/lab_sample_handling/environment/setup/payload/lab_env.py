"""Lab sample handling: the closed_loop environment.

One episode is one worklist on one bench (`lab_sim.py`) with one unit: the turn's hidden properties
(hinge friction, handle play, rotor inertia) are the unit's; the rest (the rotor's start angle, where the handle
sits in its play, the racks' places, the fills, which two tubes go to which cold positions) comes from the seed
(`lab_sim.draw_episode`).

Control at 10 Hz. An action is one of three shapes (JSON numbers):
- 8: [dx, dy, dz, drx, dry, drz, grip, cams]: the operational-space controller's step (box transport's and the
  appliance task's): the hand's target moves by (dx, dy, dz) m (at most MAX_DPOS) and turns by the rotation
  vector (drx, dry, drz) rad (at most MAX_DROT), both in the world frame; grip -1 open .. +1 closed;
- 15: the same and seven joint angles, the posture the controller's null space pulls the arm toward from now on
  (the start posture until set);
- 9: [q1..q7, grip, cams]: a joint step: a gravity-compensated joint PD toward q, each joint's target at most
  MAX_DQ from its current angle; q also becomes the null-space posture.
`cams` is a bitmask of the images the next observation carries: 1 left, 2 right, 4 wrist, 8 rack.

The observation (`observation`, on the wire `to_wire`): t, q, dq (7), ee_pos, ee_quat (wxyz), gripper (the
fingers' opening, m), wrench (the wrist's force and torque in the world frame, with noise), each camera's pose
T_world_<cam> (4 x 4, OpenCV convention: x right, y down, z forward) and intrinsics K_<cam>, the requested images
(the others empty), and the worklist: {"cold": {incoming position: cold position}, "spin": [incoming positions]}.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import base64

import mujoco
import numpy as np

import lab_sim as S
from lab_sim import B, RenderError, cam_pose, intrinsics, render  # noqa: F401 - RenderError for the grader

MAX_DPOS, MAX_DROT, MAX_DQ = B.MAX_DPOS, B.MAX_DROT, S.MAX_DQ
CAMS = {"left": (1, B.CAMERAS["left"][3], B.CAMERAS["left"][4], B.CAMERAS["left"][2], "left"),
        "right": (2, B.CAMERAS["right"][3], B.CAMERAS["right"][4], B.CAMERAS["right"][2], "right"),
        "wrist": (4, B.WRIST["width"], B.WRIST["height"], B.WRIST["fovy"], "r_eye_in_hand"),
        "rack": (8, S.RACK_CAM["width"], S.RACK_CAM["height"], S.RACK_CAM["fovy"], "rack_cam")}
ALL_CAMS = 15
TORQUE_NOISE = 0.03


def a_mass_g(a, fill_ml):
    """A tube's mass in grams: the empty tube's and its fill's (1 mL weighs 1 g)."""
    return a.mass * 1000.0 + float(fill_ml)


def encode_image(img):
    return {"dtype": "uint8", "shape": list(img.shape), "b64": base64.b64encode(np.ascontiguousarray(img)).decode()}


def decode_image(d):
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=np.uint8).reshape(d["shape"])


def to_wire(obs):
    """The observation as JSON-ready values: images base64-encoded, arrays as lists."""
    out = {}
    for k, v in obs.items():
        if k in CAMS:
            out[k] = encode_image(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


class LabEnv:
    def __init__(self, props, unit_id="dev", render=True):
        self.props = {k: float(props[k]) for k in S.UNIT_PROPS}
        self.unit_id = unit_id
        self.render_images = render

    def reset(self, seed):
        self.seed = int(seed)
        self.a = S.sim_args(self.seed)
        self.episode = S.draw_episode(self.seed, self.props)
        self.p, self.src, self.names, self.cold, self.masses = S.world(self.a, self.episode)
        self.rng = np.random.default_rng([self.seed % 2**32, self.seed >> 32, 31])
        self.render_next = ALL_CAMS
        self.faulted = False
        slots = S.slot_names()
        ep = self.episode
        # rotor positions are numbered 1-8 from the marked position, counter-clockwise seen from above; the
        # tubes already in the rotor are listed with their masses (g), as on a sample sheet
        self.worklist = {"cold": {slots[i]: n for i, n in ep["cold"].items()},
                         "spin": [slots[i] for i in ep["spin"]],
                         "preloaded": {str(int(h) + 1): round(a_mass_g(self.a, f), 2)
                                       for h, f in sorted(ep["preloaded"].items())}}
        return self.observation()

    @property
    def t(self):
        return self.p.t

    def step(self, action):
        """Apply one action (see the module's doc); returns the observation. A malformed action raises
        ValueError; an unstable simulation sets `faulted` (the episode then scores 0)."""
        a = [float(x) for x in action]
        if len(a) not in (8, 9, 15) or not all(math.isfinite(x) for x in a):
            raise ValueError("malformed action")
        cams = a[8] if len(a) == 9 else a[7]
        if cams != int(cams) or not 0 <= cams <= ALL_CAMS:
            raise ValueError("malformed action")
        self.render_next = int(cams)
        try:
            if len(a) == 9:
                self.p.joint_step(np.array(a[:7]), a[7])
            else:
                if len(a) == 15:
                    self.p.q_null = np.array(a[8:15])
                self.p.step(np.array(a[0:3]), np.array(a[3:6]), a[6])
        except FloatingPointError:
            self.faulted = True
        return self.observation()

    def unstable(self):
        d = self.p.data
        return bool(self.faulted or not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)))
                    or np.max(np.abs(d.qvel[self.p.jv])) > B.MAX_ARM_SPEED)

    def camera_matrix(self, cam):
        _bit, w, h, fovy, _name = CAMS[cam]
        return intrinsics(fovy, w, h)

    def camera_pose(self, cam, data=None):
        d = self.p.data if data is None else data
        R, pc = cam_pose(d, self.p.model.camera(CAMS[cam][4]).id)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, pc
        return T

    def wrench(self):
        d, m = self.p.data, self.p.model
        R = d.site_xmat[self.p.ft_site].reshape(3, 3)
        ta = m.sensor("r_g_torque_ee").adr[0]
        f = R @ d.sensordata[self.p.ft_force:self.p.ft_force + 3]
        tq = R @ d.sensordata[ta:ta + 3]
        return np.concatenate([f + self.rng.normal(0, B.WRENCH_NOISE, 3), tq + self.rng.normal(0, TORQUE_NOISE, 3)])

    def observation(self):
        p, d = self.p, self.p.data
        R = p.ee_R()
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        obs = {"t": int(p.t), "q": d.qpos[p.jq].copy(), "dq": d.qvel[p.jv].copy(), "ee_pos": p.ee(), "ee_quat": q,
               "gripper": float(d.qpos[p.gq[0]] - d.qpos[p.gq[1]]), "wrench": self.wrench(),
               "scale_g": p.scale_display(), "worklist": self.worklist}
        for cam, (bit, w, h, _fovy, name) in CAMS.items():
            obs[f"T_world_{cam}"] = self.camera_pose(cam)
            obs[f"K_{cam}"] = self.camera_matrix(cam)
            if self.render_images and self.render_next & bit:
                obs[cam] = render(p.model, d, name, w, h, self.rng)
            else:
                obs[cam] = np.zeros((0, 0, 3), np.uint8)
        return obs

    def perturbed_observation(self, obs, rng):
        """The probe lock's other copy: joints offset by up to 1 mrad, the hand's and the wrist camera's poses
        recomputed from them, the force offset by up to 0.2 N; images unchanged."""
        p, m = self.p, self.p.model
        out = dict(obs)
        q = np.asarray(obs["q"], float) + rng.uniform(-0.001, 0.001, 7)
        d = mujoco.MjData(m)
        d.qpos[:] = p.data.qpos
        d.qpos[p.jq] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        Rs = d.site_xmat[p.site].reshape(3, 3)
        qq = np.empty(4)
        mujoco.mju_mat2Quat(qq, Rs.flatten())
        out.update(q=q, ee_pos=d.site_xpos[p.site].copy(), ee_quat=qq, T_world_wrist=self.camera_pose("wrist", d))
        w = np.asarray(obs["wrench"], float).copy()
        w[:3] += rng.uniform(-0.2, 0.2, 3)
        out["wrench"] = w
        return out

    def close(self):
        """Nothing to release (the renderer is made and closed per image)."""

    def truth(self):
        """The full state, for a token run only (the reference's shadow): the unit, the seed, qpos and qvel."""
        d = self.p.data
        return {"unit": self.props, "seed": self.seed, "qpos": d.qpos.tolist(), "qvel": d.qvel.tolist()}
