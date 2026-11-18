"""The raceline QP without BLAS: cyclic banded Hessians, a banded Cholesky in plain Python, a primal-dual
interior-point method for box constraints.

    H = CyclicBand([d0, d1, d2, d3])      # H[i,i] = d0[i], H[i,i+k] = H[i+k,i] = dk[i] (indices mod n)
    x = box_qp(H, f, lo, hi)             # min 0.5 x'Hx + f'x  s.t.  lo <= x <= hi
    H, f = stencil_qp(c, nl, terms)      # sum_t sum_i w_t[i] |sum_m a_m P_{i+o_m}|^2 with P_i = c_i + x_i nl_i

Why no BLAS: with numpy 2.5.3's bundled OpenBLAS 0.3.34, the Haswell kernel pin (contract 1.0's setting for
comparable runs, removed in 1.1) makes large GEMM, inv and solve segfault on Zen 4 and Zen 5; and kernels chosen
per CPU would make the line differ in the last bits between machines. Here every reduction is plain Python
(IEEE double, the same on every x86-64 machine) or an element-wise numpy operation.

A cyclic band of half-width k is made an ordinary band by the interleaved order 0, n-1, 1, n-2, 2, ...: loop
neighbours up to k apart are at most 2k apart in that order, so the Cholesky factor has half-bandwidth 2k.
"""

import math

import numpy as np


class CyclicBand(object):
    def __init__(self, diags):
        self.d = [np.asarray(x, dtype=float) for x in diags]
        self.k = len(self.d) - 1
        self.n = len(self.d[0])

    @property
    def d0(self):
        return self.d[0]

    def matvec(self, x):
        x = np.asarray(x, dtype=float)
        y = self.d[0] * x
        for k in range(1, self.k + 1):
            dk = self.d[k]
            y = y + dk * np.roll(x, -k) + np.roll(dk * x, k)
        return y

    def dense(self):
        n = self.n
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] += self.d[0][i]
            for k in range(1, self.k + 1):
                j = (i + k) % n
                A[i, j] += self.d[k][i]
                A[j, i] += self.d[k][i]
        return A


def stencil_qp(c, nl, terms):
    """(H, f) of sum_t sum_i w_t[i] |sum_m a_m P_{i+o_m}|^2 (objective 0.5 x'Hx + f'x + const), P_i = c_i + x_i nl_i.
    `terms`: [(offsets, coeffs, w)] with w an array (n,) or a scalar."""
    c = np.asarray(c, dtype=float)
    nl = np.asarray(nl, dtype=float)
    n = len(c)
    kmax = max(max(o) - min(o) for o, _a, _w in terms)
    diags = [np.zeros(n) for _ in range(kmax + 1)]
    f = np.zeros(n)
    nn = [np.ones(n)] + [nl[:, 0] * np.roll(nl[:, 0], -k) + nl[:, 1] * np.roll(nl[:, 1], -k) for k in range(1, kmax + 1)]
    for offsets, coeffs, w in terms:
        w = np.broadcast_to(np.asarray(w, dtype=float), (n,))
        # (S c)_i = sum_m a_m c_{i+o_m}
        Sc = np.zeros((n, 2))
        for o, a in zip(offsets, coeffs):
            Sc += a * np.roll(c, -o, axis=0)
        for m1, (o1, a1) in enumerate(zip(offsets, coeffs)):
            # rows j = i + o1  ->  i = j - o1: W_i = roll(w, o1)[j]
            wj = np.roll(w, o1)
            Scj = np.roll(Sc, o1, axis=0)
            f += 2.0 * a1 * wj * (nl[:, 0] * Scj[:, 0] + nl[:, 1] * Scj[:, 1])
            for o2, a2 in zip(offsets, coeffs):
                dlt = o2 - o1
                if dlt < 0:
                    continue
                diags[dlt] += 2.0 * a1 * a2 * wj * nn[dlt]
    return CyclicBand(diags), f


def interleave(n):
    """perm[p] = loop index at position p of the order 0, n-1, 1, n-2, ...; pos = its inverse."""
    perm = []
    a, b = 0, n - 1
    while a <= b:
        perm.append(a)
        if a != b:
            perm.append(b)
        a += 1
        b -= 1
    pos = [0] * n
    for p, i in enumerate(perm):
        pos[i] = p
    return perm, pos


class BandCholesky(object):
    """Cholesky factor of the symmetric positive definite matrix H + diag(diag_add) (H a CyclicBand) in the
    interleaved order (half-bandwidth 2k). Plain Python floats."""

    def __init__(self, H, diag_add, order):
        perm, pos = order
        n = H.n
        HB = 2 * H.k
        self.HB = HB
        d = [x.tolist() for x in H.d]
        d[0] = (H.d[0] + diag_add).tolist()
        A = [[0.0] * (HB + 1) for _ in range(n)]        # A[p][k] = A(p, p - k)
        for i in range(n):
            p = pos[i]
            A[p][0] += d[0][i]
            for k in range(1, H.k + 1):
                j = (i + k) % n
                q = pos[j]
                if q == p:
                    A[p][0] += 2.0 * d[k][i]
                    continue
                hi, lo = (p, q) if p > q else (q, p)
                off = hi - lo
                if off > HB:
                    raise ValueError("band too wide")
                A[hi][off] += d[k][i]
        L = [[0.0] * (HB + 1) for _ in range(n)]
        for p in range(n):
            row = L[p]
            Ap = A[p]
            for k in range(min(HB, p), 0, -1):         # column q = p - k, increasing q
                q = p - k
                rq = L[q]
                acc = Ap[k]
                for m in range(max(p - HB, 0), q):
                    acc -= row[p - m] * rq[q - m]
                row[k] = acc / rq[0]
            acc = Ap[0]
            for k in range(1, min(HB, p) + 1):
                acc -= row[k] * row[k]
            if acc <= 0.0:
                raise ValueError("matrix not positive definite at position %d (%.3e)" % (p, acc))
            row[0] = math.sqrt(acc)
        self.L = L
        self.perm, self.pos = perm, pos
        self.n = n

    def solve(self, b):
        n, L, perm, HB = self.n, self.L, self.perm, self.HB
        bl = np.asarray(b, dtype=float).tolist()
        y = [0.0] * n
        for p in range(n):
            acc = bl[perm[p]]
            row = L[p]
            for k in range(1, min(HB, p) + 1):
                acc -= row[k] * y[p - k]
            y[p] = acc / row[0]
        x = [0.0] * n
        for p in range(n - 1, -1, -1):
            acc = y[p]
            for k in range(1, min(HB, n - 1 - p) + 1):
                acc -= L[p + k][k] * x[p + k]
            x[p] = acc / L[p][0]
        out = [0.0] * n
        for p in range(n):
            out[perm[p]] = x[p]
        return np.asarray(out)


def fdot(a, b):
    return math.fsum((np.asarray(a, dtype=float) * np.asarray(b, dtype=float)).tolist())


def box_qp(H, f, lo, hi, tol=1e-9, max_iter=80, sigma=0.15, info=None):
    """min 0.5 x'Hx + f'x subject to lo <= x <= hi, H a CyclicBand (positive definite). Primal-dual path
    following with a fixed centring parameter and the fraction-to-boundary rule. Where hi - lo is below 1e-6
    the variable is fixed at the midpoint."""
    n = H.n
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    f = np.asarray(f, dtype=float)
    fixed = (hi - lo) < 1e-6
    mid = 0.5 * (lo + hi)
    lo = np.where(fixed, mid - 5e-7, lo)
    hi = np.where(fixed, mid + 5e-7, hi)
    order = interleave(n)
    x = 0.5 * (lo + hi)
    s1, s2 = x - lo, hi - x
    scale = max(1.0, float(np.max(np.abs(H.d0))))
    z1 = np.full(n, 1e-3 * scale)
    z2 = np.full(n, 1e-3 * scale)
    fscale = 1.0 + float(np.max(np.abs(f)))
    it = 0
    mu = rdm = float("nan")
    for it in range(1, max_iter + 1):
        rd = H.matvec(x) + f - z1 + z2
        mu = (fdot(s1, z1) + fdot(s2, z2)) / (2.0 * n)
        rdm = float(np.max(np.abs(rd)))
        if rdm <= tol * fscale and mu <= tol * 1e-3 * scale:
            break
        smu = sigma * mu
        D = z1 / s1 + z2 / s2
        ch = BandCholesky(H, D, order)
        rhs = -rd + smu / s1 - z1 - smu / s2 + z2
        dx = ch.solve(rhs)
        dz1 = (smu - s1 * z1 - z1 * dx) / s1
        dz2 = (smu - s2 * z2 + z2 * dx) / s2
        alpha = 1.0
        for v, dv in ((s1, dx), (s2, -dx), (z1, dz1), (z2, dz2)):
            neg = dv < 0.0
            if np.any(neg):
                alpha = min(alpha, float(np.min(-v[neg] / dv[neg])))
        alpha = min(1.0, 0.995 * alpha)
        x = x + alpha * dx
        s1 = np.maximum(x - lo, 1e-300)
        s2 = np.maximum(hi - x, 1e-300)
        z1 = z1 + alpha * dz1
        z2 = z2 + alpha * dz2
    if info is not None:
        info.update({"iterations": it, "mu": mu, "rd_max": rdm})
    return np.clip(x, lo, hi)
