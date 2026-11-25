"""The pose switch: the task configuration's `pose_source`, default "estimated".

  "estimated"  pose() answers not_available; the program estimates the pose from the RGB-D frames itself.
  "cell"       pose() answers {t, index, nodes_xyz, visible_caps} of the newest camera exposure: when at least 4 of
               the 6 end caps have a line of sight to the camera, nodes_xyz = the six true node positions (site
               frame, m) plus independent Gaussian noise of 10 mm per axis; otherwise null (under the slab always
               null). visible_caps = the node numbers whose cap is in sight. It is updated at the camera's exposures
               (7 Hz) and uses no frame budget.

Line of sight: one ray from the camera's true optical centre to the node (end-cap centre), against the world's
collision boxes (geom group 0) and the robot's own collision bodies (group 3; the end caps are spheres of 17.5 mm
around the nodes): the cap is in sight when the first thing the ray meets is no nearer than the cap's own surface
(distance >= |node - camera| - cap radius - 1 mm). Rays are cast on a copy of the physics model's data set to the
exposure's joint positions (nothing of the running physics is touched). The noise is seeded per (episode noise seed,
exposure index).
"""
import numpy as np

VALUES = ("estimated", "cell")
DEFAULT = "estimated"
NOISE_M = 0.010
MIN_VISIBLE = 4
CAP_SLACK_M = 0.001
NOISE_STREAM = 0x5A


class PoseSource:
    def __init__(self, S, cam_pos, noise_seed, cap_radius):
        import mujoco
        self.mj = mujoco
        self.m = S.m
        self.d = mujoco.MjData(S.m)
        self.node_site = list(S.ids.node_site)
        self.cam = np.asarray(cam_pos, float).copy()
        self.seed = int(noise_seed) & ((1 << 63) - 1)
        self.r_cap = float(cap_radius)
        self.groups = np.array([1, 0, 0, 1, 0, 0], np.uint8)     # world boxes (0) and robot collision bodies (3)
        self.latest = None
        self.n_updates = 0

    def visibility(self, qpos):
        """(node positions 6 x 3, visible flags[6]) at qpos."""
        mj = self.mj
        self.d.qpos[:] = qpos
        mj.mj_kinematics(self.m, self.d)
        nodes = self.d.site_xpos[self.node_site].copy()
        vis = []
        gid = np.zeros(1, np.int32)
        for p in nodes:
            v = p - self.cam
            L = float(np.linalg.norm(v))
            u = v / L
            dist = mj.mj_ray(self.m, self.d, self.cam, u, self.groups, 1, -1, gid)
            vis.append(bool(dist < 0 or dist >= L - self.r_cap - CAP_SLACK_M))
        return nodes, vis

    def on_exposure(self, index, t, qpos):
        nodes, vis = self.visibility(qpos)
        rng = np.random.default_rng([self.seed, NOISE_STREAM, int(index)])
        noisy = nodes + rng.standard_normal(nodes.shape) * NOISE_M
        caps = [i for i, f in enumerate(vis) if f]
        self.latest = {"t": round(float(t), 6), "index": int(index),
                       "nodes_xyz": (np.round(noisy, 5).tolist() if len(caps) >= MIN_VISIBLE else None),
                       "visible_caps": caps}
        self.n_updates += 1

    def reply(self):
        return dict(self.latest) if self.latest is not None else {"t": None, "index": None, "nodes_xyz": None,
                                                                  "visible_caps": []}
