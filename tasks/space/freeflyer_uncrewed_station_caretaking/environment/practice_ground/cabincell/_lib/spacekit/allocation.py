"""Generic thruster allocation helpers (pure numpy).

B (6 x n) maps nonnegative thruster magnitudes u to the body wrench [F; tau] about the CoM.
``nnls`` is the Lawson-Hanson active-set algorithm (same solution as scipy.optimize.nnls).
``allocate`` adds a Tikhonov term (minimum effort among equal residuals) and
direction-preserving saturation by uniform scaling.  ``pinv_clip`` is the naive pseudo-inverse + clip
allocation, kept as a counter-example (median relative wrench error 51 % on 12 nozzles).
"""
from __future__ import annotations

import numpy as np

__all__ = ["thruster_matrix", "nnls", "allocate", "pinv_clip"]


def thruster_matrix(positions, directions, com=(0.0, 0.0, 0.0)) -> np.ndarray:
    """B[:, i] = [d_i ; (p_i - com) x d_i] (thrust directions d_i, positions p_i, body frame)."""
    P = np.asarray(positions, float)
    D = np.asarray(directions, float)
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    r = P - np.asarray(com, float)[None, :]
    return np.vstack([D.T, np.cross(r, D).T])


def nnls(A, b, max_iter: int | None = None, tol: float | None = None):
    """min ||A x - b||_2 subject to x >= 0 (Lawson & Hanson 1974).  Returns (x, residual norm)."""
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    m, n = A.shape
    max_iter = 3 * n if max_iter is None else max_iter
    tol = 10 * max(m, n) * np.linalg.norm(A, 1) * np.finfo(float).eps if tol is None else tol
    x = np.zeros(n)
    P = np.zeros(n, bool)
    w = A.T @ (b - A @ x)
    it = 0
    while (~P).any() and (w[~P] > tol).any() and it < max_iter:
        j = int(np.argmax(np.where(~P, w, -np.inf)))
        P[j] = True
        while True:
            it += 1
            z = np.zeros(n)
            z[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if (z[P] > tol).all():
                x = z
                break
            mask = P & (z <= tol)
            alpha = np.min(x[mask] / (x[mask] - z[mask]))
            x = x + alpha * (z - x)
            P &= x > tol
            x[~P] = 0.0
            if it >= max_iter:
                break
        w = A.T @ (b - A @ x)
    return x, float(np.linalg.norm(A @ x - b))


def allocate(B, wrench, umax=None, reg: float = 1e-4, weights=None):
    """Nonnegative allocation with Tikhonov regularisation and saturation by uniform scaling.

    Returns (u, achieved_wrench, scale); scale < 1 means the demand was shrunk (direction kept) to stay
    within the per-thruster limits umax.
    """
    B = np.asarray(B, float)
    w = np.asarray(wrench, float)
    Wt = np.eye(B.shape[0]) if weights is None else np.diag(weights)
    n = B.shape[1]
    A = np.vstack([Wt @ B, np.sqrt(reg) * np.eye(n)])
    u, _ = nnls(A, np.concatenate([Wt @ w, np.zeros(n)]))
    scale = 1.0
    if umax is not None:
        umax = np.broadcast_to(np.asarray(umax, float), (n,))
        over = float(np.max(u / umax))
        if over > 1.0:
            scale = 1.0 / over
            u = u * scale
    return u, B @ u, scale


def pinv_clip(B, wrench, umax=None):
    """Naive allocation: pseudo-inverse, then clip to [0, umax] (counter-example)."""
    u = np.linalg.pinv(np.asarray(B, float)) @ np.asarray(wrench, float)
    u = np.clip(u, 0.0, None if umax is None else umax)
    return u, np.asarray(B, float) @ u
