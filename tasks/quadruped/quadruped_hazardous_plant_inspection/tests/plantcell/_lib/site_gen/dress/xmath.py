"""Small vector helpers without BLAS (cross-machine determinism).

numpy's matrix products and norms go through OpenBLAS, whose kernels are chosen by the CPU: their sums can differ in
the last bit between two CPU generations, and the dressing writes such numbers into the model (mesh vertices, poses),
which then hashes differently on the grading machine (measured 2026-11-16 on two machines with different CPUs). These use elementwise
IEEE operations only, so the same inputs give the same bits on every CPU.
"""

from __future__ import annotations

import math

import numpy as np


def matvec_rows(P, M_T):
    """P (n, 3) @ M_T (3, 3) as explicit products and sums (rows of P times the columns of M_T)."""
    P = np.asarray(P, float)
    M_T = np.asarray(M_T, float)
    return (P[:, 0:1] * M_T[0][None, :] + P[:, 1:2] * M_T[1][None, :]) + P[:, 2:3] * M_T[2][None, :]


def mat_vec(R, v):
    """R (3, 3) @ v (3,)."""
    R = np.asarray(R, float)
    v = np.asarray(v, float)
    return (R[:, 0] * v[0] + R[:, 1] * v[1]) + R[:, 2] * v[2]


def dot3(a, b):
    return (float(a[0]) * float(b[0]) + float(a[1]) * float(b[1])) + float(a[2]) * float(b[2])


def rows_dot(P, v):
    """(n, 3) . (3,) per row."""
    P = np.asarray(P, float)
    return (P[:, 0] * float(v[0]) + P[:, 1] * float(v[1])) + P[:, 2] * float(v[2])


def norm(v):
    v = [float(x) for x in np.asarray(v, float).ravel()]
    s = 0.0
    for x in v:
        s += x * x
    return math.sqrt(s)


def rows_norm(D):
    D = np.asarray(D, float)
    return np.sqrt((D[:, 0] * D[:, 0] + D[:, 1] * D[:, 1]) + D[:, 2] * D[:, 2])
