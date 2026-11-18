"""C7 scale: band(|s* - 1|, C7.band) with s* the final alignment's scale; the orchestrator multiplies it
by C1 (criteria.C7.gate)."""
from .config import band_down


def score(ctx):
    s = float(ctx.align.s)
    lo, hi = ctx.sec["band"]
    err = abs(s - 1.0)
    return {"value": band_down(err, float(lo), float(hi)), "detail": {"s_star": s, "abs_err": err,
                                                                         "refined": bool(ctx.align.refined)}}
