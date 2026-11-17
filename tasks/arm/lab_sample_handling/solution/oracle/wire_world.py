"""Lab sample handling: the reference driven over the closed_loop wire .

A token run, as the appliance task's oracle: each observation carries `truth` (the unit, the seed, the full
state). On an episode's first observation the oracle builds a shadow of the world from the unit and the seed (the
same bench, `worklist_probe.world`) and runs probe 8's reference (`worklist_probe.run_worklist`) on it in a
thread. The shadow is a `WireWorld`: its `step` and `joint_step` send the action instead of simulating, then wait
for the next observation, which sets the shadow's state: the arm's joints from the observation, everything else
from `truth`. The reference's force readings come from the observation's wrench. Its images are rendered from the
shadow, whose state is the truth, so it requests none.

A repeated step index (a probe of the lock) is answered without the reference: a Cartesian step's target (the
hand's pose it aimed at) re-based on that copy's hand pose; a joint step as sent.

v2 : the scale's reading is the observation's `scale_g`. The reference thinks between some steps for
longer than a reply may take (its reach table, about 30 s): when it has not sent its next command within THINK_S,
the oracle answers with a hold, and keeps answering holds without syncing the shadow (which the thread is reading)
until the command comes; a Cartesian command is a target pose, re-based on the observation it answers.
"""

import queue
import threading

import mujoco
import numpy as np

import centrifuge_probe as C
import worklist_probe as W
from tube_insert_probe import B


class EpisodeOver(BaseException):
    """The grader moved on (reset or close): ends the reference's thread wherever it waits."""


HOLD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]
THINK_S = 10.0                  # s, the longest the oracle waits for the reference's next command before a hold


class WireWorld(C.World):
    """A `World` whose steps go over the wire (see the module's doc)."""

    def attach(self, out_q, in_q):
        self.out_q, self.in_q = out_q, in_q
        self.sent_null = np.asarray(self.q_null, float).copy()
        self.obs_wrench = np.zeros(6)
        m = self.model
        self.arm_q = np.asarray(self.jq)
        self.arm_v = np.asarray(self.jv)
        self.other_q = np.setdiff1d(np.arange(m.nq), self.arm_q)
        self.other_v = np.setdiff1d(np.arange(m.nv), self.arm_v)

    def sync(self, obs, truth):
        d = self.data
        qpos, qvel = np.asarray(truth["qpos"], float), np.asarray(truth["qvel"], float)
        d.qpos[self.other_q] = qpos[self.other_q]
        d.qvel[self.other_v] = qvel[self.other_v]
        d.qpos[self.arm_q] = np.asarray(obs["q"], float)
        d.qvel[self.arm_v] = np.asarray(obs["dq"], float)
        mujoco.mj_forward(self.model, d)
        self.obs_wrench = np.asarray(obs["wrench"], float)
        self.obs_scale = obs.get("scale_g")

    def scale_display(self):
        return getattr(self, "obs_scale", None)

    def wrench_force(self):
        return self.obs_wrench[:3].copy()

    def _exchange(self, cmd):
        self.out_q.put(cmd)
        msg = self.in_q.get()
        if msg is None:
            raise EpisodeOver()
        self.t += 1
        if self.track_force:
            self.max_force = max(self.max_force, float(np.linalg.norm(self.wrench_force() - self.force_ref)))

    def step(self, dp, dr, grip):
        d = self.data
        dp = np.asarray(dp, float)
        n = np.linalg.norm(dp)
        if n > B.MAX_DPOS:
            dp = dp * (B.MAX_DPOS / n)
        dr = np.asarray(dr, float)
        n = np.linalg.norm(dr)
        if n > B.MAX_DROT:
            dr = dr * (B.MAX_DROT / n)
        target_pos = d.site_xpos[self.site] + dp
        target_R = B.axis_angle_mat(dr) @ d.site_xmat[self.site].reshape(3, 3)
        null = None
        if not np.allclose(self.q_null, self.sent_null):
            null = np.asarray(self.q_null, float).copy()
            self.sent_null = null
        self._exchange({"kind": "cart", "pos": target_pos, "R": target_R, "grip": float(grip), "null": null})

    def joint_step(self, q_target, grip):
        q = np.asarray(q_target, float).copy()
        self.q_null = q.copy()
        self.sent_null = q.copy()
        self._exchange({"kind": "joint", "q": q, "grip": float(grip)})


def action_for(cmd, obs):
    """The wire action for a stored command, re-based on observation `obs` (a Cartesian target's offset from the
    hand's pose in `obs`)."""
    if cmd is None:
        return list(HOLD)
    if cmd["kind"] == "joint":
        return [*map(float, cmd["q"]), cmd["grip"], 0.0]
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(obs["ee_quat"], float))
    dp = cmd["pos"] - np.asarray(obs["ee_pos"], float)
    dr = C.rotvec(cmd["R"] @ R.reshape(3, 3).T)
    act = [*map(float, dp), *map(float, dr), cmd["grip"], 0.0]
    if cmd["null"] is not None:
        act += [*map(float, cmd["null"])]
    return act


class Oracle:
    """The reference as a wire submission: `act(t, obs)` returns the action for step t."""

    def __init__(self):
        self.thread = None
        self.last_t, self.last_cmd = None, None
        self.finished = False
        self.result = None
        self.pending = False

    def reset(self):
        self.stop()
        self.last_t, self.last_cmd, self.finished, self.result, self.pending = None, None, False, None, False

    def stop(self):
        if self.thread is not None and self.thread.is_alive():
            self.in_q.put(None)               # it ends at its next exchange (a daemon; not waited for while thinking)
            self.thread.join(timeout=1)
        self.thread = None

    def _start(self, obs):
        truth = obs["truth"]
        a = W.make_args(int(truth["seed"]))
        ep = W.draw_episode(a.seed, truth["unit"])
        for k in ("rotor_friction", "handle_play", "rotor_inertia", "rotor_start"):
            setattr(a, k, ep[k])
        p, src, names, cold, masses = W.world(a, ep, world_cls=WireWorld)
        self.out_q, self.in_q = queue.Queue(), queue.Queue()
        p.attach(self.out_q, self.in_q)
        p.sync(obs, truth)
        self.p = p

        def run():
            try:
                self.result = W.run_worklist(p, a, ep, src, names, cold, masses)
            except EpisodeOver:
                return
            self.out_q.put("done")

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def act(self, t, obs):
        if "truth" not in obs:
            return list(HOLD)
        if t == self.last_t:                       # a probe: the same step again, another copy
            return action_for(self.last_cmd, obs)
        if self.thread is None and not self.finished:
            self._start(obs)
        elif not self.finished and not self.pending:
            self.p.sync(obs, obs["truth"])
            self.in_q.put("go")
        cmd = None
        if not self.finished:
            try:
                got = self.out_q.get(timeout=THINK_S)
            except queue.Empty:
                self.pending = True            # still thinking: hold this step
                self.last_t, self.last_cmd = t, None
                return list(HOLD)
            self.pending = False
            if got == "done":
                self.finished = True
            else:
                cmd = got
        self.last_t, self.last_cmd = t, cmd
        return action_for(cmd, obs)
