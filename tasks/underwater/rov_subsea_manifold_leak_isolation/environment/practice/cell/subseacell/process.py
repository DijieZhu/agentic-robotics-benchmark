"""Process model of the manifold: connectivity graph, leak-segment pressure, bubble emission scale (cell side).

Graph (DESIGN 2.3.5; 14 nodes, 18 valves):
    per branch i = 1..6:  well W_i --BV_i-- S_i (leak segment, carries TP-i) --PV_i-- production header HP
                                                                     S_i --TV_i-- test header HT
    pressure sources: the six wells and the two headers (the production system keeps them at pressure).
A valve conducts when its hinge angle < 80 deg (`valve.closed_min_deg`: closed = >= 80 deg; no partial opening).
Every 0.1 s (grid point j at mission time j * 0.1 s):
    S_i connected to any source through conducting valves -> p(S_i) = p0 (the site's segment pressure);
    otherwise the LEAK segment decays p <- p exp(-dt / tau) (the leak drains it); a healthy isolated segment keeps its
    trapped pressure (no leak path; `healthy_isolated="decay"` reproduces the G6 sketch, which decayed every segment).
The bubble emission rate of the leak is proportional to p(S_leak) / p0 (`rate_scale_step`), read from the 0.1 s
history by grid index, so a plume that re-runs from a checkpoint reads the same value whatever the query order.  Asking
for a grid point that has not been computed yet raises (a world-loop ordering error must not pass silently).

The isolation set = the leak branch's {BV, PV, TV} that are open as found (1-3 valves); closing exactly those isolates
the leak segment.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

N_BRANCHES = 6
KINDS = ("BV", "PV", "TV")
CLOSED_MIN_DEG = 80.0          # valve.closed_min_deg
DT_S = 0.1                     # process.update_hz = 10
P0_RANGE_BAR = (30.0, 60.0)    # process.p0_bar (hidden)
TAU_RANGE_S = (40.0, 90.0)     # process.tau_s (hidden)
_EPS = 1e-9


def valve_tags() -> List[str]:
    """The 18 tags in the canonical order BV1..BV6, PV1..PV6, TV1..TV6."""
    return [f"{k}{b}" for k in KINDS for b in range(1, N_BRANCHES + 1)]


class ProcessGraph:
    """Nodes W1..W6, S1..S6, HP, HT; one edge per valve.  `segments_connected(cond)` -> bool (6,) for S1..S6."""

    def __init__(self):
        self.nodes = [f"W{b}" for b in range(1, 7)] + [f"S{b}" for b in range(1, 7)] + ["HP", "HT"]
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.sources = [self.index[f"W{b}"] for b in range(1, 7)] + [self.index["HP"], self.index["HT"]]
        self.edges: Dict[str, tuple] = {}
        for b in range(1, N_BRANCHES + 1):
            s = self.index[f"S{b}"]
            self.edges[f"BV{b}"] = (self.index[f"W{b}"], s)
            self.edges[f"PV{b}"] = (s, self.index["HP"])
            self.edges[f"TV{b}"] = (s, self.index["HT"])
        self.tags = valve_tags()
        self._seg = np.array([self.index[f"S{b}"] for b in range(1, 7)])
        self._cache: Dict[int, np.ndarray] = {}

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_valves(self) -> int:
        return len(self.edges)

    def bits(self, cond: Dict[str, bool]) -> int:
        b = 0
        for i, t in enumerate(self.tags):
            if cond[t]:
                b |= 1 << i
        return b

    def segments_connected_bits(self, bits: int) -> np.ndarray:
        out = self._cache.get(bits)
        if out is not None:
            return out
        parent = list(range(self.n_nodes))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for i, t in enumerate(self.tags):
            if bits >> i & 1:
                a, b = self.edges[t]
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
        live = {find(s) for s in self.sources}
        out = np.array([find(int(s)) in live for s in self._seg], bool)
        out.setflags(write=False)
        self._cache[bits] = out
        return out

    def segments_connected(self, cond: Dict[str, bool]) -> np.ndarray:
        return self.segments_connected_bits(self.bits(cond))


class Process:
    def __init__(self, leak_branch: int, p0_bar: float, tau_s: float, dt: float = DT_S,
                 healthy_isolated: str = "hold", closed_min_deg: float = CLOSED_MIN_DEG):
        if not 1 <= int(leak_branch) <= N_BRANCHES:
            raise ValueError("leak_branch must be 1..6")
        if healthy_isolated not in ("hold", "decay"):
            raise ValueError("healthy_isolated: 'hold' or 'decay'")
        self.leak = int(leak_branch)
        self.p0 = float(p0_bar)
        self.tau = float(tau_s)
        self.dt = float(dt)
        self.decay = math.exp(-self.dt / self.tau)
        self.healthy_isolated = healthy_isolated
        self.closed_min_deg = float(closed_min_deg)
        self.graph = ProcessGraph()
        self.tags = self.graph.tags
        self.k = 0                                       # last computed grid index (time k * dt)
        self._cap = 4096
        self._p = np.zeros((self._cap, N_BRANCHES))     # p[j, branch-1]
        self._bits = np.zeros(self._cap, np.int64)      # conduction bit mask used for grid point j (j >= 1)
        self._conn = np.zeros((self._cap, N_BRANCHES), bool)
        self._p[0] = self.p0                             # as found: every branch has an open valve (J2 / DESIGN 2.1)
        self._conn[0] = True
        self._bits[0] = -1

    @classmethod
    def from_site(cls, site: dict, **kw) -> "Process":
        """Site keys (G6 sketch schema): leak_branch (1..6), leak_p0_bar, leak_tau_s."""
        return cls(site["leak_branch"], site["leak_p0_bar"], site["leak_tau_s"], **kw)

    # ---------------------------------------------------------------- valve state -> conduction
    def conducting(self, angles_deg: Dict[str, float]) -> Dict[str, bool]:
        return {t: (float(angles_deg[t]) < self.closed_min_deg) for t in self.tags}

    def isolation_set(self, as_found_conducting: Dict[str, bool]) -> List[str]:
        return [f"{k}{self.leak}" for k in KINDS if as_found_conducting[f"{k}{self.leak}"]]

    # ---------------------------------------------------------------- stepping
    def _grow(self, need: int):
        if need < self._cap:
            return
        cap = max(2 * self._cap, need + 1)
        for name in ("_p", "_bits", "_conn"):
            a = getattr(self, name)
            b = np.zeros((cap,) + a.shape[1:], a.dtype)
            b[:self._cap] = a
            setattr(self, name, b)
        self._cap = cap

    def step_to(self, t: float, angles_deg: Optional[Dict[str, float]] = None,
                conducting: Optional[Dict[str, bool]] = None) -> np.ndarray:
        """Advance the 0.1 s grid to the last point <= t with the given valve state (angles or conduction flags).
        Returns the segments' connection flags (6,) of the last computed point."""
        if conducting is None:
            if angles_deg is None:
                raise ValueError("give angles_deg or conducting")
            conducting = self.conducting(angles_deg)
        bits = self.graph.bits(conducting)
        conn = self.graph.segments_connected_bits(bits)
        kt = int(math.floor(float(t) / self.dt + _EPS))
        if kt <= self.k:
            return self._conn[self.k].copy()
        self._grow(kt)
        leak_i = self.leak - 1
        for j in range(self.k + 1, kt + 1):
            prev = self._p[j - 1]
            if self.healthy_isolated == "decay":
                p = np.where(conn, self.p0, prev * self.decay)
            else:
                p = np.where(conn, self.p0, prev)
                if not conn[leak_i]:
                    p[leak_i] = prev[leak_i] * self.decay
            self._p[j] = p
            self._bits[j] = bits
            self._conn[j] = conn
        self.k = kt
        return conn.copy()

    # ---------------------------------------------------------------- read-out
    def _check(self, j: int):
        if j > self.k:
            raise RuntimeError(f"process grid point {j} (t = {j * self.dt:.1f} s) not computed yet "
                               f"(process at {self.k * self.dt:.1f} s): step the process before querying the plume")

    def pressure_step(self, j: int, branch: Optional[int] = None) -> float:
        """Segment pressure [bar] at grid point j (mission time j * dt); j < 0 -> as found (p0)."""
        b = self.leak if branch is None else int(branch)
        if j < 0:
            return self.p0
        self._check(j)
        return float(self._p[j, b - 1])

    def rate_scale_step(self, j: int) -> float:
        """Bubble emission scale p(S_leak) / p0 at grid point j (j < 0: before the run, 1)."""
        if j < 0:
            return 1.0
        self._check(j)
        return float(self._p[j, self.leak - 1]) / self.p0

    def rate_scale(self, t: float) -> float:
        return self.rate_scale_step(int(math.floor(float(t) / self.dt + _EPS)))

    def leak_pressure(self) -> float:
        return float(self._p[self.k, self.leak - 1])

    def segment_pressure(self, branch: int) -> float:
        """Current pressure of segment S_branch [bar] (the hot-stab insertion force uses it)."""
        return float(self._p[self.k, int(branch) - 1])

    def pressures(self) -> np.ndarray:
        return self._p[self.k].copy()

    def connected(self, j: Optional[int] = None) -> np.ndarray:
        j = self.k if j is None else int(j)
        self._check(j)
        return self._conn[j].copy()

    def leak_isolated(self) -> bool:
        return not bool(self._conn[self.k, self.leak - 1])

    def history(self, branch: Optional[int] = None) -> np.ndarray:
        """Pressure history of one segment (default: the leak) on the grid 0 .. k (copy)."""
        b = self.leak if branch is None else int(branch)
        return self._p[:self.k + 1, b - 1].copy()

    def conduction_bits_history(self) -> np.ndarray:
        """Conduction bit mask per grid point 1 .. k (bit i = valve_tags()[i]); point 0 = as found (-1)."""
        return self._bits[:self.k + 1].copy()

    def times(self) -> np.ndarray:
        return np.arange(self.k + 1) * self.dt


def fit_tau(t: np.ndarray, p: np.ndarray) -> float:
    """Least-squares tau of p(t) = p(t0) exp(-(t - t0) / tau) (log-linear fit)."""
    t = np.asarray(t, float)
    lp = np.log(np.asarray(p, float))
    A = np.column_stack([np.ones_like(t), t])
    coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
    return float(-1.0 / coef[1])
