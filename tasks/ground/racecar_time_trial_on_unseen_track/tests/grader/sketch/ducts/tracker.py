"""`DuctTracker`: displacement of every duct piece from where it was
last set down, the staff's put-back, the duct state carried between models, and (mode `active_window`) the window.

    ducts = DuctTracker(model, data, world, seed=None)   # seed: int / sequence / numpy Generator; default seeds.putback
    after every physics step:  dmax = ducts.update()     # largest displacement now (m)
    ducts.displaced(0.05) -> [(duct_id, k), ...]         # pieces at or above the threshold
    ducts.put_back(pieces)                               # re-lay every duct that holds one of these pieces
    ducts.poses() / ducts.reference_poses() / ducts.set_poses(poses, reference=None)
    ducts.total_displacement()                           # sum over pieces of their distance from the world file

Semantics:
* A piece's tracked points are the two end points of its axis; its displacement is the larger horizontal distance of
  the two from where they were when the piece was last set down (by the venue builder, the staff or set_poses with a
  reference). The reference never moves on a touch, so light pushes add up.
* Poses are computed from qpos (`geom.pose_from_q`), not from xpos, so `update()` is right whether or not the
  caller refreshed the kinematics after `mj_step`.
* `put_back(pieces)` re-lays each duct that holds one of `pieces` (None: every duct) as one object: its world-file
  shape moved rigidly by the put-back error (geom.putback_pose, at most `putback.max_err_m` at every end point), at
  rest. That pose becomes the duct's new reference. A chain of pinned pieces can only be re-laid as a whole, so its
  other pieces are repaired too; ducts not named stay where they are (touches below the threshold are not repaired).
* After put_back / set_poses the caller runs `mj_forward` before the next scan (the cell does): in mode `sleep`
  MuJoCo wakes a sleeping piece whose qpos was written, and a frozen piece of the window has its geoms moved here.
"""

import math

import mujoco
import numpy as np

from sketch.examples import constants as K

from . import geom
from .build import TEXT_MODE
from .params import load_params


def _model_text(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, name)
    if i < 0:
        return None
    a, n = int(model.text_adr[i]), int(model.text_size[i])
    return bytes(model.text_data[a:a + n - 1]).decode("ascii")


def _rng(world, seed, params):
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None:
        seed = world["truth"]["seeds"]["putback"]
    if isinstance(seed, (list, tuple)):
        return np.random.default_rng(list(seed))
    return np.random.default_rng([int(seed), int(params["putback"]["stream_key"])])


class DuctTracker(object):
    """Displacement tracking, put-back and state transfer of the duct pieces of one model."""

    def __init__(self, model, data, world, seed=None, params=None, putback_err_m=None):
        self.model, self.data, self.world = model, data, world
        self.params = params if params is not None else load_params()
        self.rng = _rng(world, seed, self.params)
        self.err = float(self.params["putback"]["max_err_m"] if putback_err_m is None else putback_err_m)
        mode = _model_text(model, TEXT_MODE)
        if mode is None:
            raise ValueError("the model was not built by sketch.ducts.add_ducts (no custom text %r)" % TEXT_MODE)
        self.mode = mode
        ducts = world["truth"]["ducts"]
        self.ids = [(d["id"], k) for d in ducts for k in range(len(d["pieces"]))]
        self.index = {pid: i for i, pid in enumerate(self.ids)}
        self.duct_ids = [d["id"] for d in ducts]
        self.duct_of = np.array([j for j, d in enumerate(ducts) for _ in d["pieces"]], dtype=int)
        self.slices = []
        a = 0
        for d in ducts:
            self.slices.append(slice(a, a + len(d["pieces"])))
            a += len(d["pieces"])
        self.n = a
        self.lengths = np.array([float(d["piece_length_m"]) for d in ducts for _ in d["pieces"]], dtype=float)
        self.world_poses = np.array([p for d in ducts for p in d["pieces"]], dtype=float).reshape(-1, 3)
        self.ref = self.world_poses.copy()
        self.cur = self.world_poses.copy()
        self.disp = np.zeros(self.n)
        self.dmax = 0.0
        self.window = None
        if mode in ("sleep", "awake"):
            self._bind_bodies()
        elif mode == "active_window":
            from .window import ActiveWindow
            self.window = ActiveWindow(model, data, world, self.params, self)
        elif mode == "static":
            pass
        else:
            raise ValueError("unknown duct mode %r in the model" % mode)
        self._refresh_all()

    # ------------------------------------------------------------------------------------------ dynamic bodies
    def _bind_bodies(self):
        m = self.model
        qa, va, p0 = [], [], []
        self.tree_of = np.zeros(self.n, dtype=int)
        for i, (did, k) in enumerate(self.ids):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, K.duct_body(did, k))
            if bid < 0:
                raise ValueError("body %r is missing" % K.duct_body(did, k))
            jx = int(m.body_jntadr[bid])
            if int(m.body_jntnum[bid]) != 3:
                raise ValueError("body %r has not the three joints of a planar piece" % K.duct_body(did, k))
            qa.append(int(m.jnt_qposadr[jx]))
            va.append(int(m.jnt_dofadr[jx]))
            self.tree_of[i] = int(m.body_treeid[bid])
            p0.append([float(m.body_pos[bid][0]), float(m.body_pos[bid][1]),
                       2.0 * math.atan2(float(m.body_quat[bid][3]), float(m.body_quat[bid][0]))])
        self.qadr = np.array(qa, dtype=int)
        self.vadr = np.array(va, dtype=int)
        self.qidx = (self.qadr[:, None] + np.arange(3)[None, :]).reshape(-1)
        self.vidx = (self.vadr[:, None] + np.arange(3)[None, :]).reshape(-1)
        self.p0 = np.array(p0, dtype=float)
        self.sleep = bool(m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP))
        self._was_awake = np.ones(self.n, dtype=bool)

    def _dynamic_poses(self, sel=None):
        q = self.data.qpos[self.qidx].reshape(-1, 3)
        if sel is None:
            return geom.pose_from_q(self.p0, q)
        return geom.pose_from_q(self.p0[sel], q[sel])

    # ------------------------------------------------------------------------------------------ measurement
    def _refresh_all(self):
        if self.window is not None:
            self.cur = self.window.poses()
        elif self.mode in ("sleep", "awake"):
            self.cur = self._dynamic_poses()
        self.disp = geom.displacement(self.cur, self.ref, self.lengths)
        self.dmax = float(self.disp.max()) if self.n else 0.0

    def update(self):
        """Measure after a physics step; returns the largest displacement of any piece now (m)."""
        if self.n == 0:
            return 0.0
        if self.mode == "awake":
            self._refresh_all()
        elif self.mode == "sleep":
            awake = self.data.tree_awake[self.tree_of].astype(bool)
            sel = awake | self._was_awake          # a piece that fell asleep in this step: its last motion counts
            self._was_awake = awake
            if sel.any():
                idx = np.nonzero(sel)[0]
                self.cur[idx] = self._dynamic_poses(idx)
                self.disp[idx] = geom.displacement(self.cur[idx], self.ref[idx], self.lengths[idx])
                self.dmax = float(self.disp.max())
        elif self.window is not None:
            changed = self.window.step()
            if changed is not None and len(changed):
                self.cur[changed] = self.window.poses(changed)
                self.disp[changed] = geom.displacement(self.cur[changed], self.ref[changed], self.lengths[changed])
                self.dmax = float(self.disp.max())
        return self.dmax

    def displaced(self, threshold_m):
        """(duct_id, k) of every piece whose displacement is at least threshold_m, in world-file order."""
        return [self.ids[i] for i in np.nonzero(self.disp >= float(threshold_m))[0]]

    def displacements(self):
        """Per-piece displacement (m) from the reference, world-file order (diagnostics)."""
        return self.disp.copy()

    def total_displacement(self):
        """Sum over all pieces of their displacement from the world file's pose (the reported statistic)."""
        return float(geom.displacement(self.cur, self.world_poses, self.lengths).sum())

    # ------------------------------------------------------------------------------------------ state
    def poses(self):
        """[x, y, yaw] of every piece now, (n_pieces, 3), world-file order."""
        return self.cur.copy()

    def reference_poses(self):
        """The poses every piece was last set down at, (n_pieces, 3)."""
        return self.ref.copy()

    def set_poses(self, poses, reference=None):
        """Put every piece at `poses` (n_pieces, 3), at rest; `reference` (n_pieces, 3) becomes the set-down poses
        (None: the reference stays as it is). Used to carry the walls from one model of a venue to the next."""
        p = np.asarray(poses, dtype=float).reshape(self.n, 3)
        self._write(np.arange(self.n), p)
        if reference is not None:
            self.ref = np.asarray(reference, dtype=float).reshape(self.n, 3).copy()
        self._refresh_all()

    def put_back(self, pieces=None):
        """Re-lay every duct that holds one of `pieces` ((duct_id, k) pairs; None: every duct) at its world-file
        shape plus the put-back error; the new pose is its reference. Draws three numbers per duct, in world-file
        order of the ducts."""
        if pieces is None:
            ducts = list(range(len(self.duct_ids)))
        else:
            want = set()
            for pid in pieces:
                pid = (pid[0], int(pid[1]))
                if pid not in self.index:
                    raise KeyError("no duct piece %r" % (pid,))
                want.add(int(self.duct_of[self.index[pid]]))
            ducts = sorted(want)
        for j in ducts:
            sl = self.slices[j]
            new = geom.putback_pose(self.world_poses[sl], self.lengths[sl], self.err, self.rng)
            idx = np.arange(sl.start, sl.stop)
            self._write(idx, new)
            self.ref[sl] = new
        self._refresh_all()

    def _write(self, idx, poses):
        """Put pieces `idx` at `poses`, at rest."""
        if self.mode in ("sleep", "awake"):
            q = geom.q_from_pose(self.p0[idx], poses)
            qi = (self.qadr[idx][:, None] + np.arange(3)[None, :]).reshape(-1)
            vi = (self.vadr[idx][:, None] + np.arange(3)[None, :]).reshape(-1)
            self.data.qpos[qi] = q.reshape(-1)
            self.data.qvel[vi] = 0.0
            self.cur[idx] = poses
            if self.mode == "sleep":
                self._was_awake[idx] = True
        elif self.window is not None:
            self.window.write(idx, poses)
            self.cur[idx] = poses
        else:
            raise ValueError("static ducts cannot be moved")

    # ------------------------------------------------------------------------------------------ diagnostics
    def joint_gaps(self):
        """Largest distance between the head of a piece and the tail of the next one, per duct (pin stretch)."""
        tail, head = geom.ends(self.cur, self.lengths)
        out = []
        for sl in self.slices:
            if sl.stop - sl.start < 2:
                out.append(0.0)
                continue
            out.append(float(np.linalg.norm(head[sl.start:sl.stop - 1] - tail[sl.start + 1:sl.stop], axis=1).max()))
        return np.array(out)

    def awake_pieces(self):
        """Number of pieces that are dynamic now (mode sleep: awake; window: in a slot; awake: all)."""
        if self.mode == "awake":
            return self.n
        if self.mode == "sleep":
            return int(self.data.tree_awake[self.tree_of].sum())
        if self.window is not None:
            return self.window.n_active_pieces()
        return 0
