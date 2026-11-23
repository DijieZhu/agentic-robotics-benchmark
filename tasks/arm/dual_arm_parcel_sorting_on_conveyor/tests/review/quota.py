"""The first subset under quotas (standard library only).

``first_subset(cands, size, quotas)`` returns the earliest set of ``size`` candidates, in the candidates' order
(lexicographically: the first candidate as early as possible, then the second, ...), whose summed contributions meet
every quota, or None. A candidate is a dict with ``"contrib"`` {quota name: number}; a quota is (name, threshold):
the sum of that name over the set must be at least the threshold. Contributions may be negative (a share written as
``soft - share * parcels >= 0``). Depth-first search in candidate order, pruned by the best completion each quota
can still reach, so the first set found is the lexicographically first one.
"""

from __future__ import annotations


def first_subset(cands, size, quotas, limit=2_000_000):
    n = len(cands)
    names = [q[0] for q in quotas]
    need = {q[0]: float(q[1]) for q in quotas}
    vec = [[float(c["contrib"].get(k, 0.0)) for k in names] for c in cands]
    calls = [0]

    def bound_ok(acc, start, r):
        """The best each quota can still reach: its sum so far plus the r largest contributions left."""
        for j, k in enumerate(names):
            rest = sorted((vec[i][j] for i in range(start, n)), reverse=True)[:r]
            if acc[j] + sum(rest) < need[k] - 1e-9:
                return False
        return True

    def dfs(chosen, acc, start):
        calls[0] += 1
        if calls[0] > limit:
            raise RuntimeError("quota search too long")
        r = size - len(chosen)
        if r == 0:
            return list(chosen) if all(acc[j] >= need[k] - 1e-9 for j, k in enumerate(names)) else None
        if n - start < r or not bound_ok(acc, start, r):
            return None
        for i in range(start, n):
            if n - i < r:
                break
            acc2 = [a + v for a, v in zip(acc, vec[i])]
            got = dfs(chosen + [i], acc2, i + 1)
            if got is not None:
                return got
        return None

    idx = dfs([], [0.0] * len(names), 0)
    return None if idx is None else [cands[i] for i in idx]
