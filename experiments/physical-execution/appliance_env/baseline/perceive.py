"""Camera-only perception for the practice-model baseline (P3): render-and-compare against the two fixed
cameras with the unit's KNOWN model (an agent has the practice models in its home).

  * `PoseEstimator.estimate(obs)`: the unit's placement jitter (dx, dy, dyaw) from the first observation's
    `left` / `right` images: coarse grid, then three local refinements, each candidate rendered at half
    resolution with the unit body (and its support and the back wall) moved to the candidate pose;
  * `PoseEstimator.knob_angle(obs, loc, jitter)`: a knob's angle from a window around its projection.

Only observations go in: images, joint readings, the published camera poses and intrinsics (the fixed
cameras move a little per episode, and the light changes: images are compared after normalization). No
simulator state.
"""
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import appliance_env as E  # noqa: E402

SCALE = 2                              # render at half the camera resolution (refinement, knob angles)
COARSE = 4                             # the placement grid search at a quarter


def gray(img):
    return np.asarray(img, np.float32).mean(axis=2)


def norm(g):
    """Edges, normalized: gradient magnitude of the zero-mean, unit-spread image, so the comparison
    ignores the episode's light strength, tint and (mostly) its direction."""
    g = (g - g.mean()) / (g.std() + 1e-6)
    gy, gx = np.gradient(g)
    e = np.hypot(gx, gy)
    return e / (e.mean() + 1e-6)


def down(img, k=SCALE):
    g = gray(img)
    h, w = g.shape
    return g[:h - h % k, :w - w % k].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


class PoseEstimator(object):

    def __init__(self, spec):
        self.env = E.ApplianceEnv(spec, None, render=False, jitter_scale=(0, 0, 0))
        self.env.reset(0, jitter=(0.0, 0.0, 0.0))
        m = self.env.model
        self.m, self.d = m, self.env.data
        self.ubody = m.body("u_object").id
        self.support = m.geom("support").id if "support" in [m.geom(i).name for i in range(m.ngeom)] else None
        self.wall = m.geom("back_wall").id
        lay = self.env.layout
        self.lo, self.hi, self.mt = lay["bbox_lo"], lay["bbox_hi"], lay["mount"]
        self.sup_size0 = float(m.geom_size[self.support][0]) if self.support is not None else 0.0
        self.renderers = {}
        for k in (SCALE, COARSE):
            for cam in ("left", "right"):
                _p, _t, _f, w, h = E.CAMERAS[cam]
                self.renderers[cam, k] = (mujoco.Renderer(m, h // k, w // k), m.camera(cam).id)

    def set_cameras(self, obs):
        """Put this scene's fixed cameras where the observation says they are (T_world_*, OpenCV)."""
        m = self.m
        for cam in ("left", "right"):
            T = np.asarray(obs["T_world_" + cam], float)
            cid = m.camera(cam).id
            m.cam_pos[cid] = T[:3, 3]
            m.cam_quat[cid] = E.quat_from_mat(T[:3, :3] @ np.diag([1.0, -1.0, -1.0]))

    def set_state(self, jitter, q, gripper, knobs=None):
        m, d, env = self.m, self.d, self.env
        pos, yaw, mt = E.mount_pose(env.spec["kind"], self.lo, self.hi, jitter)
        m.body_pos[self.ubody] = pos
        m.body_quat[self.ubody] = E.yaw_quat(yaw)
        front = mt["front_x"] + jitter[0]
        if self.support is not None:
            m.geom_pos[self.support] = [front + self.sup_size0 - 0.03, jitter[1], m.geom_pos[self.support][2]]
        depth = self.hi[1] - self.lo[1]
        m.geom_pos[self.wall][0] = front + max(depth, 0.6) + 0.15
        d.qpos[env.jq] = q
        d.qpos[env.gq[0]], d.qpos[env.gq[1]] = gripper / 2, -gripper / 2
        for loc, (qa, _dof) in env.unit.knobs.items():
            d.qpos[qa] = (knobs or {}).get(loc, 0.0)
        if env.unit.temp_knob is not None:
            d.qpos[env.unit.temp_knob[0]] = (knobs or {}).get("temp", 0.0)
        mujoco.mj_forward(m, d)
        env.unit.update(m, d, set())                  # burner flames follow the hypothesized knob angles

    def render(self, cam, k=SCALE):
        r, cid = self.renderers[cam, k]
        r.update_scene(self.d, camera=cid)
        return gray(r.render())

    def cost(self, targets, window=None, k=SCALE):
        c = 0.0
        for cam, tgt in targets.items():
            img = self.render(cam, k)
            if window is not None:
                (r0, r1, c0, c1) = window[cam]
                img, tgt = img[r0:r1, c0:c1], tgt[r0:r1, c0:c1]
            c += float(np.mean(np.abs(norm(img) - norm(tgt))))
        return c

    def estimate(self, obs, knobs=None):
        self.set_cameras(obs)
        targets = {cam: down(E.decode_image(obs[cam])) for cam in ("left", "right")}
        coarse = {cam: down(E.decode_image(obs[cam]), COARSE) for cam in ("left", "right")}
        q, g = np.asarray(obs["q"], float), float(obs["gripper"])
        span = np.array([0.03, 0.04, math.radians(10)])
        best, best_c = np.zeros(3), None
        grid = np.linspace(-1, 1, 7)
        for a in grid:
            for b in grid:
                for c in grid:
                    j = np.array([a, b, c]) * span
                    self.set_state(j, q, g, knobs)
                    cc = self.cost(coarse, k=COARSE)
                    if best_c is None or cc < best_c:
                        best, best_c = j, cc
        self.set_state(best, q, g, knobs)
        best_c = self.cost(targets)
        step = span / 4
        for _ in range(4):
            centre = best.copy()
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    for c in (-1, 0, 1):
                        j = centre + np.array([a, b, c]) * step
                        self.set_state(j, q, g, knobs)
                        cc = self.cost(targets)
                        if cc < best_c:
                            best, best_c = j, cc
            step = step / 2
        return best, best_c

    def knob_angle(self, obs, loc, jitter, knobs=None, angles=None):
        """A knob's angle (rad) from the fixed cameras, comparing a window around its projection."""
        self.set_cameras(obs)
        targets = {cam: down(E.decode_image(obs[cam])) for cam in ("left", "right")}
        q, g = np.asarray(obs["q"], float), float(obs["gripper"])
        knobs = dict(knobs or {})
        self.set_state(jitter, q, g, knobs)
        gid = self.m.geom("u_knob_%s_main" % loc).id
        X = self.d.geom_xpos[gid]
        window = {}
        for cam in ("left", "right"):
            K = self.env.camera_matrix(cam) / SCALE
            K[2, 2] = 1.0
            T = self.env.fixed_pose(cam)
            Xc = np.linalg.inv(T) @ np.r_[X, 1.0]
            u, v, _ = K @ (Xc[:3] / Xc[2])
            h, w = targets[cam].shape
            r = 12
            window[cam] = (max(0, int(v) - r), min(h, int(v) + r), max(0, int(u) - r), min(w, int(u) + r))
        hi = float(self.m.joint("u_knob_%s_joint" % loc).range[1])
        lo = float(self.m.joint("u_knob_%s_joint" % loc).range[0])
        cands = angles if angles is not None else np.linspace(lo, hi, 61)
        best, best_c = 0.0, None
        for a in cands:
            knobs[loc] = float(a)
            self.set_state(jitter, q, g, knobs)
            cc = self.cost(targets, window)
            if best_c is None or cc < best_c:
                best, best_c = float(a), cc
        return best
