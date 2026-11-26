"""The grader's same-seed re-run, tried for real (a review tool: run by a maintainer in the verifier image as root,
in a staged copy of tests/ with a submission in /home/user/submission, exactly where verify.py would run; the
grader never runs it).

    cd /opt/ale/tests && /opt/venv/bin/python3 review/test_rerun.py once      # one process ends by itself: a re-run
    cd /opt/ale/tests && /opt/venv/bin/python3 review/test_rerun.py twice     # the same episode's two processes: a task error

`once` runs verify.main() twice: first undisturbed, then with harness_sim.EpisodeProcess replaced by a subclass
whose child is killed (SIGSEGV) at control step KILL_AT of the first graded episode's first attempt (the render
self-test's process is left alone). It then ASSERTS that the disturbed run's verdict has `sim_reruns` 1 and equals
the undisturbed one in the reward and in every metric but `sim_reruns`, `controller_time_s` (the clock is set
back to the episode's start, to the time the controller really took the second time) and `root_path_dirs_secured`
(counted once per process), and prints RERUN TEST PASS.
That holds for a controller that clears its state at `reset` (the reference does).
`twice` kills the first episode's first two processes: no reward is written and the exit status is 1.
"""
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = [os.path.join(sys.path[0], "verify.py")] + sys.argv[1:]       # the kit finds the stage folder from argv[0]

import harness_sim as S  # noqa: E402
import verify  # noqa: E402

KILL_AT = 40
MODE = sys.argv[1] if len(sys.argv) > 1 else "once"
made = [0]


Base = S.EpisodeProcess


class Crashing(Base):
    def __init__(self, *a, **k):
        Base.__init__(self, *a, **k)
        made[0] += 1
        self.number = made[0]            # 1 = the render self-test, 2 = episode 0's first attempt, 3 = its second
        self.steps = 0

    def step(self, action):
        self.steps += 1
        if self.steps == KILL_AT and self.number in ((2,) if MODE == "once" else (2, 3)):
            os.kill(self.proc.pid, signal.SIGSEGV)
        return Base.step(self, action)


import json  # noqa: E402

VERDICT = os.environ.get("ROBOTICS_GRADER_VERDICT_PATH") or "/tmp/rerun_verdict.json"


def run(path, crashing):
    os.environ["ROBOTICS_GRADER_VERDICT_PATH"] = path
    made[0] = 0
    verify.S.EpisodeProcess = Crashing if crashing else Base
    verify.main()
    with open(path) as fh:
        return json.load(fh)


if MODE == "twice":
    run(VERDICT, True)                     # ends in robotics_grader.stage.fail: exit status 1, no reward
    raise SystemExit("the second failure in a row did not end the run")
plain = run(VERDICT + ".undisturbed", False)
again = run(VERDICT, True)
# not compared: the re-run count itself, the controller's clock, and how many of root's PATH folders the first
# main() of this process had to take back (the second finds them taken)
skip = ("sim_reruns", "controller_time_s", "root_path_dirs_secured")
assert again["metrics"]["sim_reruns"] == 1.0, again["metrics"]["sim_reruns"]
assert plain["metrics"]["sim_reruns"] == 0.0
assert again["rewards"] == plain["rewards"], (again["rewards"], plain["rewards"])
differ = sorted(k for k in set(plain["metrics"]) | set(again["metrics"])
                if k not in skip and plain["metrics"].get(k) != again["metrics"].get(k))
assert not differ, "metrics that differ between the undisturbed run and the re-run: {}".format(differ)
print("RERUN TEST PASS: one re-run, reward {} and {} other metrics equal to the undisturbed run's".format(
    again["rewards"], len(again["metrics"]) - len(skip)), file=sys.stderr)
