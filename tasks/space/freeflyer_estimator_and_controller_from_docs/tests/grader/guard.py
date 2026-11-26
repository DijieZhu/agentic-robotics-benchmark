"""Stages under a guard, and the one rule that decides what a failure means.

    guard = Guard(deadline)
    out = guard.run("session_1.fly", fn, *args)        # any failure: GraderVoid (a task error)

Every object built from a session's records (the flight, the record's checks, the measures, the score) is built inside
guard.run. In this task the submission's own faults never raise out of a stage: the runner answers them as the
printed rules say (a protocol error, an invalid plan, a set-up failure) and the world records their cost in its
records, which the formula reads. So a guard that trips is always the grader's own cause: the verify deadline, the
address-space limit (MemoryError), a hidden-set, configuration, world or record defect, a record that contradicts
itself, or any other exception (an internal error). The verification is then void: a task error, and no score is
written. The alarm (SIGALRM) fires between Python bytecodes; the messages name a stage only, never a time read from
the clock.
"""
import math
import signal
import time


class StageTimeout(Exception):
    pass


class GraderVoid(Exception):
    """The verification is void (a task error): a stage ran out of time or memory, or the grader failed."""


class Guard(object):
    def __init__(self, deadline=None, enabled=True):
        self.deadline = deadline
        self.enabled = enabled

    def left(self):
        return math.inf if self.deadline is None else self.deadline - time.time()

    def run(self, name, fn, *args, **kwargs):
        cap = self.left() if self.enabled else math.inf
        if self.enabled and cap <= 0.0:
            raise GraderVoid("no time was left for stage %s before the verify deadline" % name)

        def _alarm(_sig, _frm):
            raise StageTimeout("stage %s ran out of the verify's time" % name)

        armed = self.enabled and math.isfinite(cap)
        if armed:
            old = signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, cap)
        try:
            return fn(*args, **kwargs)
        except StageTimeout as exc:
            raise GraderVoid("%s: the verification is void (no outcome may come from the clock)" % exc) from None
        except MemoryError:
            raise GraderVoid("stage %s ran out of memory: the verification is void" % name) from None
        except GraderVoid:
            raise
        except BaseException as exc:  # noqa: BLE001 - every other failure is the grader's own
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise GraderVoid("stage %s: %s: %s" % (name, type(exc).__name__, str(exc)[:400])) from None
        finally:
            if armed:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old)
