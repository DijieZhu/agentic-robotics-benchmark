# Part of the quadruped's on-board gait firmware.
"""Small-vector helpers that avoid numpy's per-call overhead in the 1 kHz loop."""

import numpy as np


def cross3(a, b):
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def cross_rows(A, B):
    """Row-wise cross product of (n, 3) arrays (B may be (3,) broadcast)."""
    B = np.broadcast_to(B, A.shape)
    out = np.empty_like(A, dtype=float)
    out[:, 0] = A[:, 1] * B[:, 2] - A[:, 2] * B[:, 1]
    out[:, 1] = A[:, 2] * B[:, 0] - A[:, 0] * B[:, 2]
    out[:, 2] = A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]
    return out


def solve_damped_batch(J, V, lam=1e-4):
    """Damped least squares x_i = (J_i^T J_i + lam I)^-1 J_i^T v_i for J (n, 3, 3), V (n, 3)."""
    Jt = np.transpose(J, (0, 2, 1))
    A = Jt @ J + lam * np.eye(3)[None]
    b = (Jt @ V[:, :, None])
    return np.linalg.solve(A, b)[:, :, 0]
