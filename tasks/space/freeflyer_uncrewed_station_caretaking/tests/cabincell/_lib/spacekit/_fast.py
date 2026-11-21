"""Small-array helpers without numpy's per-call overhead (np.cross costs ~25-100 us per call)."""
from __future__ import annotations

import numpy as np


def cross(a, b) -> np.ndarray:
    """Cross product of two 3-vectors."""
    a0, a1, a2 = a.tolist() if isinstance(a, np.ndarray) else a
    b0, b1, b2 = b.tolist() if isinstance(b, np.ndarray) else b
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


class ScalarTable:
    """Firmware lookup table on Python floats (scalar queries without numpy overhead)."""

    def __init__(self, breakpoints, values):
        self.bp = [float(v) for v in breakpoints]
        self.y = [float(v) for v in values]
        self.n = len(self.bp)

    def __call__(self, x: float) -> float:
        import bisect
        bp, y = self.bp, self.y
        if x < bp[0]:
            return y[0]
        if x > bp[-1]:
            return y[-1]
        k = min(max(bisect.bisect_right(bp, x) - 1, 0), self.n - 2)
        r = (x - bp[k]) / (bp[k + 1] - bp[k])
        return (1.0 - r) * y[k] + r * y[k + 1]


def cross_rows(a, b) -> np.ndarray:
    """Row-wise cross product of (N, 3) arrays (either may be a single (3,) vector)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.ndim == 1:
        a = np.broadcast_to(a, b.shape)
    if b.ndim == 1:
        b = np.broadcast_to(b, a.shape)
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out
