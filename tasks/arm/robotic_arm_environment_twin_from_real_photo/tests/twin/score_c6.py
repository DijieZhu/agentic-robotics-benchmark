"""C6 scale: band(|s* - 1|, 0.10, 0.30) when at least 5 pairs are matched, else 0."""
from .config import band_down


def score(s_star, n_pairs, cfg):
    b = cfg["scale_band"]
    lo, hi = b["band"]
    err = abs(float(s_star) - 1.0)
    if n_pairs < int(b["min_pairs"]):
        return 0.0, {"s_star": float(s_star), "abs_err": err, "n_pairs": int(n_pairs), "reason": "too few pairs"}
    return band_down(err, lo, hi), {"s_star": float(s_star), "abs_err": err, "n_pairs": int(n_pairs)}
