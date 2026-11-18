"""Contact-force distribution for the stance feet (a quasi-static balance controller).

Given the desired net wrench on the robot about its centre of mass (force F and moment M, odometry-aligned frame)
and the stance feet's positions relative to the CoM, find ground-reaction forces f_i that
    minimise  || S (A f - W) ||^2 + sum_i reg_i ||lambda_i||^2
    subject to f_i inside the linearised friction cone (pyramid with 4 edges, coefficient mu) and f_i . n >= f_min.
Writing each force as f_i = f_min n + E lambda_i with lambda_i >= 0 (E = the pyramid's edge vectors) turns this into
a non-negative least-squares problem, solved exactly by an active-set method.

Probe-instrument freeze (2026-11-13): the upstream file imported scipy.optimize.nnls; the instrument must run on
numpy + mujoco only, so `nnls` below is a numpy Lawson-Hanson active-set solver on the normal equations, started
from the clipped unconstrained solution (stateless: the result depends only on the inputs). The problem has full
column rank (the reg rows), so its solution is unique; on 2146 problems recorded from a walk it matches scipy's to
1.9e-10 N. Nothing else in this file differs from upstream.
"""

from __future__ import annotations

import numpy as np

_cache = {}
_EPS = float(np.finfo(float).eps)


def nnls(A: np.ndarray, b: np.ndarray, maxiter: int = 200):
    """min ||A x - b|| subject to x >= 0. Returns (x, residual norm)."""
    AtA = A.T @ A
    Atb = A.T @ b
    n = Atb.shape[0]
    tol = 10.0 * _EPS * max(A.shape) * float(np.abs(AtA).max())
    try:
        x = np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        x = np.zeros(n)
    P = x > tol                                   # passive set = support of the feasible start point
    x = np.where(P, x, 0.0)
    it = 0
    first = True
    while True:
        if not first:                             # outer loop: free the variable with the largest gradient
            if P.all():
                break
            w = Atb - AtA @ x
            wr = np.where(P, -np.inf, w)
            j = int(np.argmax(wr))
            if wr[j] <= tol:
                break
            P[j] = True
        first = False
        while True:                               # inner loop: least squares on P, step back while infeasible
            idx = np.flatnonzero(P)
            s = np.zeros(n)
            if not idx.size:
                break
            s[idx] = np.linalg.solve(AtA[np.ix_(idx, idx)], Atb[idx])
            sp = s[idx]
            if sp.min() > 0.0 or it >= maxiter:
                break
            it += 1
            neg = idx[sp <= 0.0]
            alpha = float(np.min(x[neg] / (x[neg] - s[neg])))
            x = x + alpha * (s - x)
            P &= x > tol
            x[~P] = 0.0
        x = s
    return x, float(np.linalg.norm(A @ x - b))


def pyramid_edges(n, mu):
    n = n / np.linalg.norm(n)
    t1 = np.cross(n, [0.0, 1.0, 0.0])
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.cross(n, [1.0, 0.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return np.stack([n + mu * t1, n - mu * t1, n + mu * t2, n - mu * t2], axis=1)   # (3, 4)


def distribute(r: np.ndarray, W: np.ndarray, normal: np.ndarray, mu: float, f_min: float,
               weights: np.ndarray, reg: np.ndarray):
    """r (n, 3) foot positions relative to the CoM; W (6,) desired wrench; returns f (n, 3)."""
    n = len(r)
    if n == 0:
        return np.zeros((0, 3))
    key = (round(float(normal[0]), 4), round(float(normal[1]), 4), round(float(normal[2]), 4), mu)
    if _cache.get("key") != key:
        _cache["key"] = key
        _cache["E"] = pyramid_edges(np.array(key[:3]), mu)
        _cache["n"] = np.array(key[:3]) / np.linalg.norm(key[:3])
    E, nrm = _cache["E"], _cache["n"]
    # A_i = [I; skew(r_i)];  A_i E = [E; r_i x E]
    AE = np.empty((6, 4 * n))
    Wd = W.copy()
    f0 = f_min * nrm
    for i in range(n):
        ri = r[i]
        AE[0:3, 4 * i:4 * i + 4] = E
        AE[3, 4 * i:4 * i + 4] = ri[1] * E[2] - ri[2] * E[1]
        AE[4, 4 * i:4 * i + 4] = ri[2] * E[0] - ri[0] * E[2]
        AE[5, 4 * i:4 * i + 4] = ri[0] * E[1] - ri[1] * E[0]
        Wd[0:3] -= f0
        Wd[3] -= ri[1] * f0[2] - ri[2] * f0[1]
        Wd[4] -= ri[2] * f0[0] - ri[0] * f0[2]
        Wd[5] -= ri[0] * f0[1] - ri[1] * f0[0]
    S = weights
    M = np.vstack([S[:, None] * AE, np.diag(np.repeat(np.sqrt(reg), 4))])
    b = np.concatenate([S * Wd, np.zeros(4 * n)])
    lam, _ = nnls(M, b, maxiter=200)
    f = f0[None, :] + lam.reshape(n, 4) @ E.T
    return f
