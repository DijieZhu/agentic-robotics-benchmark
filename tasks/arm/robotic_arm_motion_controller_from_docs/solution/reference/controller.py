"""The privileged reference controller (the task's oracle): it replays the grader's truth payload (the tick
trajectory of each job, computed from the unit's true parameters) and refuses the jobs the truth marks infeasible.
It only runs with the oracle token next to it (the grader sends no truth otherwise). stdlib only."""


class Controller(object):
    def __init__(self, workdir):
        self.workdir = workdir
        self.traj = None
        self.k = 0

    def setup(self, unit, cell, tool, truth=None):
        self.truth_seen = truth is not None

    def plan(self, job, truth=None):
        if truth is None:
            raise RuntimeError("the oracle needs the truth payload (oracle token)")
        if truth["infeasible"]:
            self.traj = None
            return "infeasible"
        self.traj = truth["traj"]
        self.k = 0
        return "ok"

    def step(self, state):
        tr = self.traj
        k = self.k
        self.k += 1
        i = min(k, len(tr["ticks"]) - 1)
        cmd = {"goal": tr["ticks"][i], "speed": tr["speed"][i], "acc": tr["acc"][i],
               "path_start": tr["path_start_cycle"] is not None and k == tr["path_start_cycle"],
               "done": k >= tr["done_cycle"]}
        return cmd
