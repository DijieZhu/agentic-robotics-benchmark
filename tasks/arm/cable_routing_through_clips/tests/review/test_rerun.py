"""The grader's same-seed re-run, tried for real (a review tool: run by a maintainer in the verifier image as root,
in a staged copy of tests/ with a submission in /home/user/submission, exactly where verify.py would run; the
grader never runs it).

    cd /opt/ale/tests && /opt/venv/bin/python3 review/test_rerun.py once      # one process ends by itself: a re-run
    cd /opt/ale/tests && /opt/venv/bin/python3 review/test_rerun.py twice     # the same episode's two processes: a task error

It runs verify.main() with harness_sim.EpisodeProcess replaced by a subclass whose child is killed (SIGSEGV) at
control step KILL_AT of the first graded episode's first attempt (`once`), or of its first two attempts (`twice`);
the render self-test's process is left alone. Expected with `once`: the log says the episode was run again, the
verdict's `sim_reruns` is 1 and the episode's result is the one of an undisturbed run (the controller's clock and
the lock's counts are set back). Expected with `twice`: no reward is written and the exit status is 1.
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


verify.S.EpisodeProcess = Crashing
verify.main()
