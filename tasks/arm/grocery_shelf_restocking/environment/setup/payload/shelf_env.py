"""Shelf restocking: the closed_loop environment (what a controller sees and does).

One episode is one delivery on one bay (`shelf_world.py`) with one unit (the bay's design and its SKUs); the
seed draws the rest (`draw_episode`). Control at 10 Hz. An action is one of four shapes (JSON numbers):
- 9: [dx, dy, dz, drx, dry, drz, grip, dlift, cams]: the operational-space step (box transport's):
  the hand's target moves by (dx, dy, dz) m (at most MAX_DPOS) and turns by the rotation vector (drx, dry, drz)
  rad (at most MAX_DROT), world frame; grip -1 open .. +1 closed; the lift's set point moves by dlift m (at most
  MAX_LIFT), the hand's target riding with it;
- 16: the same and seven joint angles, the posture the controller's null space pulls the arm toward from now on;
- 10: [q1..q7, grip, dlift, cams]: a joint step (each joint's target at most MAX_DQ from its angle); q also
  becomes the null-space posture;
- 1: [0]: finished (the episode ends and is scored).
`cams` is a bitmask of the images the next observation carries: 1 left, 2 right (both on the lift's carriage),
4 wrist, 8 overview.

The observation: t, q, dq (7), lift (the base's height, m), ee_pos, ee_quat (wxyz), gripper (the fingers'
opening, m), wrench (the wrist's force and torque, world frame, with noise), each camera's pose T_world_<cam>
(4 x 4, OpenCV convention) and intrinsics K_<cam>, the requested images (the others empty), and the order:
{"planogram": {"<shelf>:<lane>": sku}, "delivery": {"<shelf>:<lane>": units}, "today": day} (shelves from the
bottom, lanes 0-4 from the right as seen from the aisle), and on the first observation only, "catalog": {sku: front-view
image}.
"""

import base64
import math

import mujoco
import numpy as np

import box_transport_env as B
import shelf_sim as S
import shelf_world as W

MAX_DPOS, MAX_DROT, MAX_DQ, MAX_LIFT = B.MAX_DPOS, B.MAX_DROT, W.MAX_DQ, W.MAX_LIFT
CAMS = W.CAMS
ALL_CAMS = W.ALL_CAMS
TORQUE_NOISE = 0.03
CATALOG_PX = 128


def encode_image(img):
    return {"dtype": "uint8", "shape": list(img.shape), "b64": base64.b64encode(np.ascontiguousarray(img)).decode()}


def decode_image(d):
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=np.uint8).reshape(d["shape"])


class RenderError(RuntimeError):
    """The renderer failed: a task error (the grader's), never the submission's."""


def to_wire(obs):
    out = {}
    for k, v in obs.items():
        if k in CAMS:
            out[k] = encode_image(v)
        elif k == "catalog":
            out[k] = {s: encode_image(img) for s, img in v.items()}
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def intrinsics(fovy, w, h):
    f = (h / 2.0) / math.tan(math.radians(fovy) / 2.0)
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def cam_pose(d, cid):
    T = np.eye(4)
    T[:3, :3] = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = d.cam_xpos[cid]
    return T


def catalog(unit):
    """One front-view image per SKU: the product alone on a grey ground, lit as the bay is."""
    out = {}
    for name, sku in unit["skus"].items():
        w = mujoco.MjSpec()
        w.visual.global_.offwidth = CATALOG_PX
        w.visual.global_.offheight = CATALOG_PX
        wb = w.worldbody
        light = wb.add_light()
        light.pos = [-0.6, 0.0, 0.8]
        light.dir = [0.6, 0.0, -0.6]
        g = wb.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_PLANE
        g.size = [1, 1, 0.1]
        g.rgba = [0.6, 0.6, 0.62, 1]
        hh = W.half_h(sku)
        spec = W.product_spec("p", sku, (0.0, 0.0, hh), 0.0)
        body = S.add_product(wb, spec)
        W.add_look(body, sku)
        c = wb.add_camera()
        c.name = "front"
        pos = (-0.45, 0.0, hh + 0.02)
        c.pos = list(pos)
        c.quat = list(B.quat_from_mat(B.look_at(pos, (0.0, 0.0, hh))))
        c.fovy = 35.0
        m = w.compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        try:
            r = mujoco.Renderer(m, CATALOG_PX, CATALOG_PX)
            r.update_scene(d, camera="front")
            out[name] = r.render().copy()
            r.close()
        except Exception as exc:
            raise RenderError(f"catalog: {exc}") from exc
    return out


class ShelfEnv:
    def __init__(self, unit, unit_id="dev", render=True):
        self.unit = unit
        self.unit_id = unit_id
        self.render_images = render
        self._renderers = {}

    def reset(self, seed):
        self.close()
        self.seed = int(seed)
        self.episode = W.draw_episode(self.seed, self.unit)
        self.w = W.World(self.unit, self.episode)
        self.w.settle()
        self.rng = W.rng_of(self.seed, 31)
        self.render_next = ALL_CAMS
        self.faulted = False
        self.finished = False
        self.order = {"planogram": dict(self.episode["planogram"]), "delivery": dict(self.episode["deliver"]),
                      "today": int(self.episode["today"])}
        self.first = True
        return self.observation()

    @property
    def t(self):
        return self.w.t

    def step(self, action):
        """Apply one action (see the module's doc); returns the observation. A malformed action raises
        ValueError; an unstable simulation sets `faulted` (the episode then scores 0)."""
        a = [float(x) for x in action]
        if len(a) not in (1, 9, 10, 16) or not all(math.isfinite(x) for x in a):
            raise ValueError("malformed action")
        if len(a) == 1:
            if a[0] != 0:
                raise ValueError("malformed action")
            self.finished = True
            return self.observation()
        cams = a[9] if len(a) == 10 else a[8]
        if cams != int(cams) or not 0 <= cams <= ALL_CAMS:
            raise ValueError("malformed action")
        self.render_next = int(cams)
        try:
            if len(a) == 10:
                self.w.joint_step(np.array(a[:7]), a[7], a[8])
            else:
                if len(a) == 16:
                    self.w.q_null = np.array(a[9:16])
                self.w.step(np.array(a[0:3]), np.array(a[3:6]), a[6], dlift=a[7])
        except FloatingPointError:
            self.faulted = True
        return self.observation()

    def unstable(self):
        return bool(self.faulted or self.w.unstable())

    def camera_name(self, cam):
        return "r_eye_in_hand" if cam == "wrist" else cam

    def camera_pose(self, cam, data=None):
        d = self.w.data if data is None else data
        return cam_pose(d, self.w.model.camera(self.camera_name(cam)).id)

    def render(self, cam):
        _bit, w, h, _fovy, *_ = CAMS[cam]
        try:
            if (w, h) not in self._renderers:
                self._renderers[(w, h)] = mujoco.Renderer(self.w.model, h, w)
            r = self._renderers[(w, h)]
            r.update_scene(self.w.data, camera=self.w.model.camera(self.camera_name(cam)).id)
            img = r.render().astype(np.float32)
        except Exception as exc:
            raise RenderError(f"{cam}: {exc}") from exc
        img += self.rng.normal(0, B.RGB_NOISE, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def wrench(self):
        w = self.w
        d, m = w.data, w.model
        ft = m.site("r_g_ft_frame").id
        R = d.site_xmat[ft].reshape(3, 3)
        fa, ta = m.sensor("r_g_force_ee").adr[0], m.sensor("r_g_torque_ee").adr[0]
        f = R @ d.sensordata[fa:fa + 3]
        tq = R @ d.sensordata[ta:ta + 3]
        return np.concatenate([f + self.rng.normal(0, B.WRENCH_NOISE, 3), tq + self.rng.normal(0, TORQUE_NOISE, 3)])

    def observation(self):
        w = self.w
        d = w.data
        R = w.ee_R()
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        obs = {"t": int(w.t), "q": d.qpos[w.jq].copy(), "dq": d.qvel[w.jv].copy(), "lift": w.lift(),
               "ee_pos": w.ee(), "ee_quat": q, "gripper": float(d.qpos[w.gq[0]] - d.qpos[w.gq[1]]),
               "wrench": self.wrench(), "order": self.order}
        if self.first:
            obs["catalog"] = catalog(self.unit)
            self.first = False
        for cam, (bit, cw, ch, fovy, *_r) in CAMS.items():
            obs[f"T_world_{cam}"] = self.camera_pose(cam)
            obs[f"K_{cam}"] = intrinsics(fovy, cw, ch)
            if self.render_images and self.render_next & bit:
                obs[cam] = self.render(cam)
            else:
                obs[cam] = np.zeros((0, 0, 3), np.uint8)
        return obs

    def perturbed_observation(self, obs, rng):
        """The probe lock's other copy: joints offset by up to 1 mrad, the hand's and the wrist camera's poses
        recomputed from them, the force offset by up to 0.2 N; images unchanged."""
        w = self.w
        m = w.model
        out = dict(obs)
        q = np.asarray(obs["q"], float) + rng.uniform(-0.001, 0.001, 7)
        d = mujoco.MjData(m)
        d.qpos[:] = w.data.qpos
        d.qpos[w.jq] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        Rs = d.site_xmat[w.site].reshape(3, 3)
        qq = np.empty(4)
        mujoco.mju_mat2Quat(qq, Rs.flatten())
        out.update(q=q, ee_pos=d.site_xpos[w.site].copy(), ee_quat=qq, T_world_wrist=self.camera_pose("wrist", d))
        wr = np.asarray(obs["wrench"], float).copy()
        wr[:3] += rng.uniform(-0.2, 0.2, 3)
        out["wrench"] = wr
        return out

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers = {}

    def truth(self):
        """The full state, for a token run only (the reference's shadow): the unit, the seed, qpos and qvel."""
        d = self.w.data
        return {"unit": self.unit, "seed": self.seed, "qpos": d.qpos.tolist(), "qvel": d.qvel.tolist(),
                "lift_set": self.w.lift_set}
