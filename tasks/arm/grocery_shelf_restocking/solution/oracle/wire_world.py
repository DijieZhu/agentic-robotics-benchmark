"""Shelf restocking: the reference driven over the closed_loop wire.

A token run: each observation carries `truth` (the unit, the seed, the full state). On an episode's first
observation the oracle builds a shadow of the world from the unit and the seed (`shelf_world.World`) and runs the
reference (`reference.Reference.run`) on it in a thread. The shadow is a `WireWorld`: its `step` sends the
action instead of simulating, then waits for the next observation, which sets the shadow's state: the arm's joints
from the observation, everything else from `truth`. A repeated step index (a probe of the lock) is answered
without the reference: the last command re-based on that copy's hand pose.
"""

import queue
import threading

import mujoco
import numpy as np

import box_transport_env as B
import reference as R
import shelf_sim as S
import shelf_world as W
from shelf_ctl import rotvec


class EpisodeOver(BaseException):
    """The grader moved on (reset or close): ends the reference's thread wherever it waits."""


HOLD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]
FINISH = [0]


class WireWorld(W.World):
    def attach(self, out_q, in_q):
        self.out_q, self.in_q = out_q, in_q
        self.sent_null = np.asarray(self.q_null, float).copy()
        m = self.model
        self.other_q = np.setdiff1d(np.arange(m.nq), self.jq)
        self.other_v = np.setdiff1d(np.arange(m.nv), self.jv)

    def sync(self, obs, truth):
        d = self.data
        qpos, qvel = np.asarray(truth["qpos"], float), np.asarray(truth["qvel"], float)
        d.qpos[self.other_q] = qpos[self.other_q]
        d.qvel[self.other_v] = qvel[self.other_v]
        d.qpos[self.jq] = np.asarray(obs["q"], float)
        d.qvel[self.jv] = np.asarray(obs["dq"], float)
        self.lift_set = float(truth["lift_set"])
        mujoco.mj_forward(self.model, d)

    def _exchange(self, cmd):
        self.out_q.put(cmd)
        msg = self.in_q.get()
        if msg is None:
            raise EpisodeOver()
        self.t += 1

    def step(self, dp, dr, grip, dlift=0.0):
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
        dl = float(np.clip(dlift, -S.LIFT_STEP, S.LIFT_STEP))
        null = None
        if not np.allclose(self.q_null, self.sent_null):
            null = np.asarray(self.q_null, float).copy()
            self.sent_null = null
        self._exchange({"kind": "cart", "pos": target_pos, "R": target_R, "grip": float(grip), "dlift": dl,
                        "null": null})


def action_for(cmd, obs):
    """The wire action for a stored command, re-based on observation `obs`."""
    if cmd is None:
        return list(HOLD)
    if cmd == "finish":
        return list(FINISH)
    Rm = np.empty(9)
    mujoco.mju_quat2Mat(Rm, np.asarray(obs["ee_quat"], float))
    dp = cmd["pos"] - np.asarray(obs["ee_pos"], float)
    dr = rotvec(cmd["R"] @ Rm.reshape(3, 3).T)
    act = [*map(float, dp), *map(float, dr), cmd["grip"], cmd["dlift"], 0.0]
    if cmd["null"] is not None:
        act += [*map(float, cmd["null"])]
    return act


class Oracle:
    def __init__(self):
        self.thread = None
        self.last_t, self.last_cmd = None, None
        self.finished = False

    def reset(self):
        self.stop()
        self.last_t, self.last_cmd, self.finished = None, None, False

    def stop(self):
        if self.thread is not None and self.thread.is_alive():
            self.in_q.put(None)
            self.thread.join(timeout=1)
        self.thread = None

    def _start(self, obs):
        truth = obs["truth"]
        unit = truth["unit"]
        ep = W.draw_episode(int(truth["seed"]), unit)
        p = WireWorld(unit, ep)
        self.out_q, self.in_q = queue.Queue(), queue.Queue()
        p.attach(self.out_q, self.in_q)
        p.sync(obs, truth)
        p.t = int(obs["t"])
        self.p = p

        def run():
            try:
                R.Reference(p).run()
            except EpisodeOver:
                return
            except Exception:  # noqa: BLE001 - a failed reference ends its episode instead of leaving it waiting
                pass
            self.out_q.put("finish")

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def act(self, t, obs):
        if "truth" not in obs:
            return list(HOLD)
        if t == self.last_t:
            return action_for(self.last_cmd, obs)
        if self.thread is None and not self.finished:
            self._start(obs)
        elif not self.finished:
            self.p.sync(obs, obs["truth"])
            self.in_q.put("go")
        cmd = None
        if not self.finished:
            # wait for the reference's next command however long it takes (no wall-clock limit of its own: a
            # reply is then a function of the observations alone, and two runs of an episode are the same run on
            # a busy machine too; the grader's own deadline per reply still applies)
            got = self.out_q.get()
            if got == "finish":
                self.finished = True
            cmd = got
        else:
            cmd = "finish"
        self.last_t, self.last_cmd = t, cmd
        return action_for(cmd, obs)
