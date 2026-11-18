"""A reading over several frames: the value the reference reports from the readings of a few frames of one instrument.

    from bricks import consensus
    value, conf, dbg = consensus.combine("level", [(71.3, 1.0), (70.0, 1.0), (88.4, 0.9)])

Kinds and units are the round sheet's (the reference reports in these units): 'gauge' in the dial's unit (pass
`span` = hi - lo of the scale), 'level' and 'valve_handwheel' in percent, 'valve_lever' in degrees, 'lamps' as a list
of booleans. A reading is (value, confidence) as a reader returns it (value None = no reading).

Rule: readings below `min_conf` are dropped; the largest group of readings that agree with each other (every one
within AGREE of the group's median; lamps: identical) is taken when it holds at least `min_agree` readings and more
than half of the usable ones; the value is the group's median (lamps: the group's pattern) and the confidence the
group's mean confidence. Otherwise there is no value (`dbg['why']`), and `dbg['need_more']` says whether another
frame, preferably from another place, could settle it. Agreement tolerances are about half the scoring tolerances
of acceptance_v1.yaml (gauge 2 % of span, level / gate 5 points, lever 5 degrees), so two readings that agree are
within tolerance of each other's median. A vote cannot remove an error that every frame of one view shares (a
plate misfound the same way in each frame): that is the readers' catalogue checks' job, and a second view from
another place is the remedy (the readers are deterministic for one image, not for one place).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

__all__ = ["AGREE", "combine", "agree"]

# kind -> (mode, tolerance): 'span' = fraction of the scale span, 'abs' = in the reported unit
AGREE = {"gauge": ("span", 0.010), "level": ("abs", 2.5), "valve_handwheel": ("abs", 2.5),
         "valve_lever": ("abs", 2.5)}


def _tol(kind: str, span: float | None) -> float:
    mode, t = AGREE[kind]
    if mode == "span":
        if span is None or not span > 0:
            raise ValueError("combine(%r): pass the scale span" % kind)
        return t * float(span)
    return t


def agree(kind: str, a, b, span: float | None = None) -> bool:
    """Whether two readings of one instrument agree (lamps: identical)."""
    if kind == "lamps":
        return list(a) == list(b)
    return abs(float(a) - float(b)) <= _tol(kind, span)


def combine(kind: str, readings, span: float | None = None, min_conf: float = 0.5, min_agree: int = 2):
    """(value or None, confidence, debug) from a list of (value, confidence) readings of one instrument."""
    usable = [(v, float(c)) for v, c in readings if v is not None and c is not None and float(c) >= min_conf]
    dbg = {"n": len(readings), "usable": len(usable)}
    if not usable:
        dbg.update(why="no usable reading", need_more=True)
        return None, 0.0, dbg
    if kind == "lamps":
        cnt = Counter(tuple(bool(x) for x in v) for v, _ in usable)
        (pat, k), *rest = cnt.most_common()
        dbg["groups"] = [[list(p), n] for p, n in cnt.most_common()]
        if k < min_agree or (rest and rest[0][1] == k) or k * 2 <= len(usable):
            dbg.update(why="no majority pattern", need_more=True)
            return None, 0.0, dbg
        confs = [c for v, c in usable if tuple(bool(x) for x in v) == pat]
        return list(pat), float(np.mean(confs)), dbg
    if kind not in AGREE:
        raise ValueError("combine: unknown kind %r" % kind)
    tol = _tol(kind, span)
    vals = np.array([float(v) for v, _ in usable])
    confs = np.array([c for _, c in usable])
    best = None
    for i in range(len(vals)):
        # grow a group around reading i: readings within tol of the group's running median
        grp = [i]
        for j in np.argsort(np.abs(vals - vals[i])):
            if j == i:
                continue
            med = float(np.median(vals[grp + [int(j)]]))
            if all(abs(vals[g] - med) <= tol for g in grp + [int(j)]):
                grp.append(int(j))
        key = (len(grp), -float(np.ptp(vals[grp])) if len(grp) > 1 else 0.0)
        if best is None or key > best[0]:
            best = (key, sorted(grp))
    grp = best[1]
    dbg["group"] = [float(vals[g]) for g in grp]
    if len(grp) < min_agree or len(grp) * 2 <= len(vals):
        dbg.update(why="readings disagree" if len(vals) >= min_agree else "one reading only", need_more=True)
        return None, 0.0, dbg
    dbg["need_more"] = False
    return float(np.median(vals[grp])), float(np.mean(confs[grp])), dbg
