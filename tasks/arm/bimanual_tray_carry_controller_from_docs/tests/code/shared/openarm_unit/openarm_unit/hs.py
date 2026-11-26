import math
from typing import NamedTuple

from .state import HS_TABLE_ENTRIES, N_ARM_JOINTS, HSState

TWO_PI = 6.283185307179586


class HSError(ValueError):
    ""


class HSLaw(NamedTuple):
    harmonic_weights: tuple
    relative_rms: float
    period_range_rad: tuple
    table_bounds: tuple
    table_entries: int

    @classmethod
    def from_law(cls, law):
        q = law["quantities"]["HS1"] if "quantities" in law else law
        v = lambda k: q[k]["value"]
        try:
            w = tuple(float(x) for x in v("harmonic_weights"))
            pr = tuple(float(x) for x in v("period_range_rad"))
            tb = tuple(float(x) for x in v("table_bounds"))
            out = cls(w, float(v("relative_rms")), pr, tb, int(v("table_entries")))
        except (KeyError, TypeError, ValueError) as exc:
            raise HSError(f"params_law.json quantities.HS1 is incomplete: {exc!r}") from exc
        if len(w) != 2 or len(pr) != 2 or len(tb) != 2 or not (0.0 < pr[0] <= pr[1]) or not (tb[0] < 1.0 < tb[1]):
            raise HSError(f"HS1 constants out of form: {out}")
        if abs(w[0] * w[0] + w[1] * w[1] - 1.0) > 1e-12:
            raise HSError(f"the harmonic weights' squares must sum to 1 (RMS of rho = r), not {w}")
        if out.table_entries != HS_TABLE_ENTRIES:
            raise HSError(f"the law's table has {out.table_entries} entries, the state type {HS_TABLE_ENTRIES}")
        return out


def amplitude(r):
    return float(r) * math.sqrt(2.0)


def table(r, period, phase1, phase2, law):
    if not (period > 0.0):
        raise HSError("the period is positive")
    n = law.table_entries
    a = amplitude(r)
    w1, w2 = law.harmonic_weights
    out = []
    for j in range(n):
        u = TWO_PI * j / n
        out.append(1.0 + a * (w1 * math.sin(u + phase1) + w2 * math.sin(2.0 * u + phase2)))
    tab = tuple(out)
    lo, hi = law.table_bounds
    if min(tab) < lo or max(tab) > hi:
        raise HSError(f"a table value lies outside [{lo}, {hi}] (r = {r})")
    return tab


def from_uniforms(u_period, u_phase1, u_phase2, law):
    lo, hi = law.period_range_rad
    if not (len(u_period) == len(u_phase1) == len(u_phase2) == N_ARM_JOINTS):
        raise HSError("fourteen uniforms per draw")
    periods = tuple(lo + (hi - lo) * float(u) for u in u_period)
    phases = tuple((TWO_PI * float(a), TWO_PI * float(b)) for a, b in zip(u_phase1, u_phase2))
    return periods, phases


def hs_state(periods, phases, law, r=None):
    rr = law.relative_rms if r is None else float(r)
    if len(periods) != N_ARM_JOINTS or len(phases) != N_ARM_JOINTS:
        raise HSError("fourteen periods and fourteen phase pairs")
    lo, hi = law.period_range_rad
    for lam in periods:
        if not (lo <= lam <= hi):
            raise HSError(f"period {lam} outside {law.period_range_rad}")
    tabs = tuple(table(rr, lam, p1, p2, law) for lam, (p1, p2) in zip(periods, phases))
    return HSState(rr, periods, phases, tabs)


def off_state(period=0.035):
    if not (period > 0.0):
        raise HSError("the period is positive")
    ones = (1.0,) * HS_TABLE_ENTRIES
    return HSState(0.0, (float(period),) * N_ARM_JOINTS, ((0.0, 0.0),) * N_ARM_JOINTS, (ones,) * N_ARM_JOINTS)


def table_handout(state):
    if not isinstance(state, HSState):
        raise HSError("an HSState is handed out")
    return tuple((float(lam), tuple(tab)) for lam, tab in zip(state.period, state.tables))
