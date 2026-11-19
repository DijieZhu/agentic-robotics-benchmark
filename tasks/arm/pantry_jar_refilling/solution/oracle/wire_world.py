"""Pantry jar refilling: the privileged reference driven over the closed_loop wire (a token run).

Each observation of a token run carries `truth` (the unit, the seed, the simulation state and the threads' state;
`JarEnv.truth`). On an episode's first observation the oracle builds a shadow of the world from the unit and the
seed (the same counter, `jar_sim.World`) and runs the reference (`reference.Reference`) on it in a thread. The
shadow's episode does not simulate: its `step` hands the reference's action to the wire and waits for the next
observation, whose `truth` then sets the shadow's whole state. The reference requests no images (its state is the
truth).

An action leaves as targets: each arm's hand pose it aims at (the shadow's hand pose moved by the action's step) and
its grip. `action_for` turns them back into steps from the hand poses of the observation being answered. On the
observation the shadow was synced to, that is the reference's own step; on the perturbed copy of a repeated step
index (a probe of the lock) it is the same target seen from the copy's hand pose.

If the reference has not sent its next action within THINK_S, the oracle answers with a hold and keeps answering
holds without syncing the shadow (which the thread is reading) until the action comes.
"""

import queue
import threading

import mujoco
import numpy as np

import jar_sim as S
import reference as R

THINK_S = 20.0                  # s, the longest the oracle waits for the reference's next action before a hold
HOLD = {"l": [0.0] * 6 + [-1.0], "r": [0.0] * 6 + [-1.0], "cams": 0}


class EpisodeOver(BaseException):
    """The grader moved on (reset or close): ends the reference's thread wherever it waits."""


class ShadowEnv:
    """What the reference reads of `JarEnv`: the world `w`, the `unit` and the `episode`, set from `truth`."""

    def __init__(self, unit, seed):
        self.unit = unit
        self.seed = int(seed)
        self.episode = S.draw_episode(self.seed, unit)
        self.w = S.World(unit, self.episode)

    def sync(self, truth):
        w, d = self.w, self.w.d
        d.qpos[:] = np.asarray(truth["qpos"], float)
        d.qvel[:] = np.asarray(truth["qvel"], float)
        for j, e in enumerate(truth["engaged"]):
            if bool(e) != w.engaged[j]:
                w.engaged[j] = bool(e)
                w._show(j, bool(e))
        w.armed = [bool(x) for x in truth["armed"]]
        w.opened = [bool(x) for x in truth["opened"]]
        w.seat_peak = [float(x) for x in truth["seat_peak"]]
        w.jar_start = [np.asarray(p, float) for p in truth["jar_start"]]
        w.t = int(truth["t"])
        mujoco.mj_forward(w.m, d)


class ShadowEpisode:
    """`Episode` for the reference: `step` sends the action as targets and returns the next observation."""

    def __init__(self, env, out_q, in_q):
        self.env = env
        self.out_q, self.in_q = out_q, in_q

    def step(self, action):
        self.out_q.put(targets(self.env.w, action))
        obs = self.in_q.get()
        if obs is None:
            raise EpisodeOver()
        self.env.sync(obs["truth"])
        return obs, False, {}


def targets(w, action):
    """An action (7-number steps) as each arm's target hand pose and grip, from the world's hand poses now."""
    out = {}
    for s in ("l", "r"):
        a = np.asarray(action[s], float)
        arm = w.arms[s]
        dp, dr = a[0:3], a[3:6]
        n = np.linalg.norm(dp)
        if n > S.MAX_DPOS:
            dp = dp * (S.MAX_DPOS / n)
        n = np.linalg.norm(dr)
        if n > S.MAX_DROT:
            dr = dr * (S.MAX_DROT / n)
        out[s] = (arm.pos() + dp, S.axis_angle_mat(dr) @ arm.R(), float(a[6]))
    return out


def action_for(cmd, obs):
    """The wire action for stored targets, as steps from the hand poses in observation `obs`."""
    if cmd is None:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in HOLD.items()}
    act = {"cams": 0}
    for s in ("l", "r"):
        pos, Rt, grip = cmd[s]
        Rm = np.empty(9)
        mujoco.mju_quat2Mat(Rm, np.asarray(obs[s]["ee_quat"], float))
        dp = pos - np.asarray(obs[s]["ee_pos"], float)
        dr = R.rotvec(Rt @ Rm.reshape(3, 3).T)
        act[s] = [*map(float, dp), *map(float, dr), grip]
    return act


class Oracle:
    """The reference as a wire submission: `act(t, obs)` returns the action for step t."""

    def __init__(self):
        self.thread = None
        self.reset_state()

    def reset_state(self):
        self.last_t, self.last_cmd, self.pending = None, None, False

    def reset(self):
        self.stop()
        self.reset_state()

    def stop(self):
        if self.thread is not None and self.thread.is_alive():
            self.in_q.put(None)               # it ends at its next exchange
            self.thread.join(timeout=2)
        self.thread = None

    def _start(self, obs):
        truth = obs["truth"]
        env = ShadowEnv(truth["unit"], truth["seed"])
        env.sync(truth)
        self.out_q, self.in_q = queue.Queue(), queue.Queue()
        ref = R.Reference(ShadowEpisode(env, self.out_q, self.in_q))

        def run():
            try:
                ref.run()
            except EpisodeOver:
                return

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def act(self, t, obs):
        if "truth" not in obs:
            return action_for(None, obs)
        if t == self.last_t:                       # a repeated step index: the same targets, this copy's poses
            return action_for(self.last_cmd, obs)
        if self.thread is None:
            self._start(obs)
        elif not self.pending:
            self.in_q.put(obs)
        try:
            cmd = self.out_q.get(timeout=THINK_S)
            self.pending = False
        except queue.Empty:
            cmd, self.pending = None, True         # still thinking: hold this step
        self.last_t, self.last_cmd = t, cmd
        return action_for(cmd, obs)
