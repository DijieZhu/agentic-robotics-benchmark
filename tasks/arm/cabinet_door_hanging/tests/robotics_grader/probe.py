"""The probe copy of an observation whose poses are one physics step older than its joints.

A simulator that steps and then reports reads its bodies' poses from the last step's
kinematics, which were computed from the joint positions *before* that step's integration,
while the joint positions it reports are the ones *after* it. So a real observation's hand
pose is not the forward kinematics of its own joint reading: it is the kinematics of the
reading one physics step earlier. A probe copy (`ProbeLock.probe`) that offsets the joint
reading and then recomputes the poses as the exact kinematics of the offset reading is easy to
tell from the real observation by anyone who holds the model: its residual
`pose - kinematics(joints)` is exactly zero, the real one's is not.

`stale_copy` builds the copy the way the real observation is built: the poses of the copy are
the kinematics of the *older* joint positions moved by the same offset the copy's reading has.
Its residual then has the real observation's size and direction to every order, at rest and
in motion.

The kit knows no simulator. The task hands in its own forward kinematics as a callable.
"""

__all__ = ["stale_copy"]


def stale_copy(kinematics, stale, observed, copy):
    """Poses for a probe copy whose joint reading is `copy` where the real one is `observed`.

    `kinematics(q)` returns the poses for joint positions `q` (any object: the caller reads it
    as it reads the simulator's own poses). `stale` are the joint positions the simulator's
    reported poses were computed from (the positions before the last physics step); `observed`
    the reported ones; `copy` the copy's. All three are sequences of the same length.

    Returns `kinematics(stale + (copy - observed))`.
    """
    stale, observed, copy = list(stale), list(observed), list(copy)
    if not len(stale) == len(observed) == len(copy):
        raise ValueError("stale, observed and copy must have the same length")
    return kinematics([s + (c - o) for s, o, c in zip(stale, observed, copy)])
