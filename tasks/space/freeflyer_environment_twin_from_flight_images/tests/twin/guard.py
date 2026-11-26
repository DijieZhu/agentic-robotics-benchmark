"""Stages under a guard: an alarm per stage, and the one rule that decides what a failure means.

    guard = Guard(deadline, caps)
    out = guard.run("rays", fn, *args)                    # any failure: GraderVoid (a task error)
    out = guard.run("scene", fn, *args, faults=(ValueError,))   # the listed exception types: SubmissionFault
    out = guard.run("rays", fn, *args, limit_rule="Z7")   # a time-out at the stage's OWN limit: LimitFault
    guard.limit("scene")                                  # a stage's configured cap (math.inf when disabled or absent)

Every object built from a submission is built inside guard.run. A guard trips in one of two ways:
  - on the submission: the stage raises SubmissionFault (or an exception type the stage lists as the submission's
    own, such as the simulator refusing a geometry): that flight scores 0 with the stage's named reason;
  - on the submission's cost, for a stage that names a rule (limit_rule): the stage's alarm at the stage's OWN limit
    (its configured cap, reached while the verification's deadline was still further away) raises LimitFault, a
    SubmissionFault that names the rule and the limit: that flight scores 0 under the rule (rule Z7 for the rays);
  - for any other cause: the stage's alarm (StageTimeout) of a stage that names no rule, or at the verification's
    deadline, the address-space limit (MemoryError), a configuration, given or truth defect, or any other exception
    (an internal error): GraderVoid. The verification is void, a task error, and no score is written.
The alarm (SIGALRM) fires between Python bytecodes, so the callers cast rays in row chunks: one simulator call is
short. The messages name a stage and its configured cap only, never a time that depends on the clock.
The grader's scene and rays stages do not run under guard.run: grade.scene_in_child runs them in a child process with
the child's own limits (the stage's cap from guard.limit, the deadline, the worker's address space), and grade.py
reads their stops (rule Z7 at a stage's own limit or by memory; void otherwise). guard.stats collects, per flight,
that child's step, outcome, seconds and peak memory for the maintainers' timing record (never the verdict).
"""
import math
import signal
import time


class SubmissionFault(Exception):
    """The submission made this stage fail: its flight scores 0 with this reason."""


class LimitFault(SubmissionFault):
    """A stage that names a rule did not finish inside its own limit: its flight scores 0 under that rule."""

    def __init__(self, msg, rule):
        super().__init__(msg)
        self.rule = rule


class StageTimeout(Exception):
    pass


class GraderVoid(Exception):
    """The verification is void (a task error): a stage ran out of time or memory, or the grader failed."""


class Guard:
    def __init__(self, deadline=None, caps=None, enabled=True):
        self.deadline = deadline
        self.caps = dict(caps or {})
        self.enabled = enabled
        self.stats = {}

    def limit(self, name):
        return float(self.caps[name]) if self.enabled and name in self.caps else math.inf

    def left(self):
        return math.inf if self.deadline is None else self.deadline - time.time()

    def run(self, name, fn, *args, faults=(), limit_rule=None):
        cap_conf = float(self.caps[name]) if name in self.caps else math.inf
        cap = min(cap_conf, self.left()) if self.enabled else math.inf
        if self.enabled and cap <= 0.0:
            raise GraderVoid("no time was left for stage %s before the verify deadline" % name)
        own_limit = cap >= cap_conf
        why = ("stage %s timed out (its cap of %g s)" % (name, cap_conf) if own_limit
               else "stage %s ran out of the verify's time" % name)

        def _alarm(_sig, _frm):
            raise StageTimeout(why)

        armed = self.enabled and math.isfinite(cap)
        if armed:
            old = signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, cap)
        try:
            return fn(*args)
        except SubmissionFault:
            raise
        except StageTimeout as exc:
            if limit_rule is not None and own_limit:
                raise LimitFault("stage %s did not finish inside its limit of %g s of the grader's wall clock"
                                 % (name, cap_conf), limit_rule) from None
            raise GraderVoid("%s: the verification is void (no outcome may come from the clock)" % exc) from None
        except MemoryError:
            raise GraderVoid("stage %s ran out of memory (the grader's address-space limit): the verification is void"
                             % name) from None
        except GraderVoid:
            raise
        except BaseException as exc:  # noqa: BLE001 - classified below
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if faults and isinstance(exc, faults):
                raise SubmissionFault("stage %s: %s: %s" % (name, type(exc).__name__, str(exc)[:300])) from None
            raise GraderVoid("stage %s: internal error %s: %s" % (name, type(exc).__name__, str(exc)[:300])) from None
        finally:
            if armed:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old)
