"""The reference's one exception: a phase that cannot go on stops the program and says where and why."""


class ReferenceFailure(RuntimeError):
    def __init__(self, phase, reason, data=None):
        RuntimeError.__init__(self, "%s: %s" % (phase, reason))
        self.phase, self.reason, self.data = phase, reason, dict(data or {})
