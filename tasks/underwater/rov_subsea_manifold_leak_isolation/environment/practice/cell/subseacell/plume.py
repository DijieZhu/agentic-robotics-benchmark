"""Lagrangian bubble plume of the seep, drifting clutter clumps, and the deterministic mean path (cell side).

    plume = Plume.from_site(site, flow)          # flow = flow.StationFlow.from_site(site)
    plume.attach_process(process)                # emission rate x p(S_leak) / p0 (process.Process)
    plume.advance_to(t)                          # state at the last 0.1 s step <= t (mission time)
    plume.P, plume.ids, plume.w                  # live particles (views), positions in W

Model (DESIGN 2.3.5; rise speed M14):
  * low-rate seep: one particle = a small cluster of 3-8 mm bubbles; `rate` particles per second at full segment
    pressure (100-300 /s), times process.rate_scale_step (deterministic count per step, fractional carry);
  * emitted at the leak, initial jitter +-3 cm; rise speed w_i = w_b (1 + 0.15 (2u - 1)) (uniform +-15 % around
    w_b, bubble size sorting), w_b in 0.20-0.28 m/s (ruling M14);
  * Euler at 10 Hz:  x += (u_flow(x, t) + w_i e_z) dt + uniform random walk (variance 2 K dt per axis, K = 0.01 m^2/s);
    u_flow = the station current (flow.StationFlow fast path: 1/7 profile x free stream x structure factor); the roof
    grating does not stop bubbles; removed 12 m above the source;
  * spin-up: the leak is older than the dive, the plume starts `warmup_s` (150 s) before the run (step 0 at mission
    time -warmup_s; steps at (k - k_warm) * dt);
  * every random number is a counter-based hash of (seed, particle id, step index): the state at step k is a pure
    function of k.  Checkpoints every 100 steps; a query earlier than the current step restores the latest checkpoint
    at or before it and re-runs, so the state is bitwise independent of the query order (test_plume.py);
  * emission at step k uses the process pressure of grid point j = k - k_warm (mission time of the step's start), which
    is always <= t - dt when the plume is advanced to t: the world must step the process up to t first (it raises
    otherwise).
Clutter (sonar only, DESIGN 2.3.5 [placeholder values of G3]): 0-3 clumps of 24 point scatterers (diameter 0.2-0.5 m,
total TS -38 to -28 dB) at roof + 0.8 .. 7 m, drifting with the local current (no rise) in a 30 m periodic domain
around the structure (a wrap re-draws the other horizontal coordinate and the height from the hash).

The sonar coupling (sensing.PlumeSonar) asks for `horizontal_index()`: the live particles pre-binned on a horizontal
grid (0.5 m cells, built once per plume step on demand) with each cell's height range, so a ping only projects the
particles of the cells that can intersect its beam.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, fields
from typing import Callable, Optional

import numpy as np

from .flow import BL_DELTA_M, BL_EXPONENT, Z_REF_M, StationFlow, StructureEnvelope

# ------------------------------------------------------------------------------------------------ design values
RISE_RANGE_M_S = (0.20, 0.28)      # plume.rise_m_s (M14)
RISE_SPREAD = 0.15                 # plume.rise_spread_frac: uniform +-15 % per particle
EMISSION_RANGE_PER_S = (100.0, 300.0)
DIFFUSIVITY_M2_S = 0.01
CULL_DZ_M = 12.0
STEP_S = 0.1
CHECKPOINT_STEPS = 100
JITTER_M = 0.03
WARMUP_S = 150.0
CULL_EVERY = 5
INDEX_BIN_M = 0.5

_M64 = 0xFFFFFFFFFFFFFFFF
_G = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_INV21 = 1.0 / float(1 << 21)
_EPS = 1e-9


def _splitmix(x):
    x = x + _G
    x = (x ^ (x >> np.uint64(30))) * _M1
    x = (x ^ (x >> np.uint64(27))) * _M2
    return x ^ (x >> np.uint64(31))


def hash3(seed: int, ids: np.ndarray, k: int, salt: int = 0):
    """Three uniforms in [0, 1) (21-bit resolution) per id, a pure function of (seed, id, k, salt)."""
    with np.errstate(over="ignore"):
        x = np.asarray(ids).astype(np.uint64) * _G
        x ^= np.uint64(((int(k) * 0x2545F4914F6CDD1D) ^ (int(salt) * 0x5851F42D4C957F2D) ^ int(seed)) & _M64)
        x = _splitmix(_splitmix(x))
    m = np.uint64(0x1FFFFF)
    u1 = ((x >> np.uint64(43)) & m).astype(np.float64) * _INV21
    u2 = ((x >> np.uint64(22)) & m).astype(np.float64) * _INV21
    u3 = ((x >> np.uint64(1)) & m).astype(np.float64) * _INV21
    return u1, u2, u3


# ------------------------------------------------------------------------------------------------ specs
@dataclass
class PlumeSpec:
    src_w: tuple               # leak position in W
    rate: float                # particles per second at full segment pressure
    w_b: float                 # mean rise speed [m/s]
    seed: int = 1
    K: float = DIFFUSIVITY_M2_S
    jitter: float = JITTER_M
    w_spread: float = RISE_SPREAD   # half-width of the uniform per-particle spread (fraction of w_b)
    cull_dz: float = CULL_DZ_M
    dt: float = STEP_S
    warmup_s: float = WARMUP_S

    @classmethod
    def from_site(cls, site: dict) -> "PlumeSpec":
        """site["plume"] = {rate, w_b, seed[, K]}; the leak position is site["src_W"] (G6 schema) or plume["src_w"]."""
        p = dict(site["plume"])
        src = p.pop("src_w", None) or site["src_W"]
        names = {f.name for f in fields(cls)}
        return cls(src_w=tuple(float(v) for v in src), **{k: v for k, v in p.items() if k in names and k != "src_w"})

    def problems(self) -> list:
        out = []
        if not RISE_RANGE_M_S[0] - 1e-12 <= self.w_b <= RISE_RANGE_M_S[1] + 1e-12:
            out.append(f"w_b {self.w_b} outside {RISE_RANGE_M_S} (M14)")
        if not EMISSION_RANGE_PER_S[0] - 1e-9 <= self.rate <= EMISSION_RANGE_PER_S[1] + 1e-9:
            out.append(f"rate {self.rate} outside {EMISSION_RANGE_PER_S}")
        if abs(self.w_spread - RISE_SPREAD) > 1e-12 or abs(self.K - DIFFUSIVITY_M2_S) > 1e-12 \
                or abs(self.cull_dz - CULL_DZ_M) > 1e-12 or abs(self.dt - STEP_S) > 1e-12:
            out.append("non-design spread / diffusivity / cull height / step")
        return out


@dataclass
class ClutterSpec:
    n: int                     # 0-3 clumps
    radius: tuple              # (n,) clump radius [m] (0.10-0.25 -> 0.2-0.5 m across)
    ts_db: tuple               # (n,) total target strength of a clump [dB re 1 m^2]
    z_band: tuple              # heights (W z) the clumps live in
    domain: float = 30.0       # periodic square domain (m) around the structure
    n_pts: int = 24
    seed: int = 2

    @classmethod
    def from_site(cls, site: dict) -> Optional["ClutterSpec"]:
        c = site.get("clutter")
        if not c:
            return None
        return cls(n=int(c["n"]), radius=tuple(c["radius"]), ts_db=tuple(c["ts_db"]), z_band=tuple(c["z_band"]),
                   domain=float(c.get("domain", 30.0)), n_pts=int(c.get("n_pts", 24)), seed=int(c["seed"]))


# ------------------------------------------------------------------------------------------------ horizontal index
class HIndex:
    """Live particles binned on a horizontal grid: per non-empty cell its centre, height range and its slice of
    `order` (particle indices sorted by cell).  `gather(mask)` -> particle indices of the selected cells, ascending."""
    __slots__ = ("bin", "cx", "cy", "zmin", "zmax", "start", "count", "order", "n", "centre", "hz")

    def __init__(self, b, cx, cy, zmin, zmax, start, count, order, n):
        self.bin, self.cx, self.cy, self.zmin, self.zmax = b, cx, cy, zmin, zmax
        self.start, self.count, self.order, self.n = start, count, order, n
        self.centre = np.stack([cx, cy, 0.5 * (zmin + zmax)])            # (3, M) cell box centres
        self.hz = 0.5 * (zmax - zmin)                                    # (M,) half heights

    @classmethod
    def build(cls, P: np.ndarray, bin_m: float = INDEX_BIN_M) -> "HIndex":
        n = int(P.shape[0])
        if n == 0:
            e = np.zeros(0)
            return cls(bin_m, e, e, e, e, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), 0)
        x, y = P[:, 0], P[:, 1]                     # strided 1-D views: per-column ops are much faster than (n, 2)
        xlo, ylo = float(x.min()), float(y.min())
        b = float(bin_m)
        while True:
            x0 = math.floor(xlo / b) * b
            y0 = math.floor(ylo / b) * b
            ib = 1.0 / b
            ix = ((x - x0) * ib).astype(np.int32)
            iy = ((y - y0) * ib).astype(np.int32)
            nx = int(ix.max()) + 1
            ny = int(iy.max()) + 1
            if nx * ny <= 65535:
                break
            b *= 2.0
        iy *= nx
        iy += ix
        key = iy.astype(np.uint16)
        order = np.argsort(key, kind="stable")
        cnt = np.bincount(key, minlength=nx * ny)
        ne = np.flatnonzero(cnt)
        c = cnt[ne]
        st = np.cumsum(c) - c
        zs = P[order, 2]
        zmin = np.minimum.reduceat(zs, st)
        zmax = np.maximum.reduceat(zs, st)
        cx = x0 + ((ne % nx) + 0.5) * b
        cy = y0 + ((ne // nx) + 0.5) * b
        return cls(b, cx, cy, zmin, zmax, st, c, order, n)

    def gather(self, sel: np.ndarray) -> np.ndarray:
        cells = np.flatnonzero(sel)
        if not cells.size:
            return np.zeros(0, np.int64)
        st = self.start[cells]
        c = self.count[cells]
        if cells.size <= 48:                       # few cells: slices are cheaper than the vectorised gather
            o = self.order
            idx = np.concatenate([o[a:a + b] for a, b in zip(st.tolist(), c.tolist())])
        else:
            tot = int(c.sum())
            offs = np.repeat(st - (np.cumsum(c) - c), c) + np.arange(tot)
            idx = self.order[offs]
        idx.sort()
        return idx


# ------------------------------------------------------------------------------------------------ the plume
class Plume:
    def __init__(self, ps: PlumeSpec, flow: StationFlow, clutter: Optional[ClutterSpec] = None,
                 rate_fn: Optional[Callable[[int], float]] = None, checkpoint_every: int = CHECKPOINT_STEPS,
                 allow_out_of_range: bool = False):
        probs = ps.problems()
        if probs and not allow_out_of_range:
            raise ValueError("plume spec outside the design ranges: " + "; ".join(probs))
        self.ps, self.flow, self.cs = ps, flow, clutter
        self.src = np.asarray(ps.src_w, float).reshape(3)
        self.k_warm = int(round(ps.warmup_s / ps.dt))
        self.t0 = -self.k_warm * ps.dt
        self.rate_fn = rate_fn
        self.ck_every = int(checkpoint_every)
        self._version = 0
        self._hidx = None
        self._hidx_key = None
        self._reset()
        self.ckpts = {0: self._snapshot()}
        self.n_steps_run = 0
        self.t_step_s = 0.0
        self.c_step_s = 0.0
        self.c_index_s = 0.0          # thread CPU of the horizontal index builds (once per step, on demand)
        self.n_index_builds = 0

    @classmethod
    def from_site(cls, site: dict, flow: StationFlow, **kw) -> "Plume":
        return cls(PlumeSpec.from_site(site), flow, ClutterSpec.from_site(site), **kw)

    def attach_process(self, process) -> None:
        """Emission x process.rate_scale_step(j) (j = mission grid index of the step)."""
        self.rate_fn = process.rate_scale_step

    # -------------------------------------------------------------- state (capacity buffers; P / ids / w are views)
    @property
    def P(self) -> np.ndarray:
        return self._P[:self.n]

    @property
    def ids(self) -> np.ndarray:
        return self._ids[:self.n]

    @property
    def w(self) -> np.ndarray:
        return self._w[:self.n]

    def _alloc(self, cap):
        P, I, Wb = np.zeros((cap, 3)), np.zeros(cap, np.int64), np.zeros(cap)
        if hasattr(self, "_P"):
            P[:self.n], I[:self.n], Wb[:self.n] = self._P[:self.n], self._ids[:self.n], self._w[:self.n]
        self._P, self._ids, self._w = P, I, Wb

    def _rise(self, ids: np.ndarray) -> np.ndarray:
        ps = self.ps
        v1, _, _ = hash3(ps.seed, ids, -2, salt=2)
        return ps.w_b * (1.0 + ps.w_spread * (2.0 * v1 - 1.0))

    def _reset(self):
        self.k = 0
        self.n = 0
        self._alloc(4096)
        self.next_id = 0
        self.emit_acc = 0.0
        cs = self.cs
        if cs is not None and cs.n > 0:
            self.cP = self._clump_spawn(np.arange(cs.n))
            self.cwrap = np.zeros(cs.n, np.int64)
        else:
            self.cP = np.zeros((0, 3))
            self.cwrap = np.zeros(0, np.int64)
        self._version += 1

    def _snapshot(self):
        return (self.k, self.P.copy(), self.ids.astype(np.int32), self.next_id, self.emit_acc,
                self.cP.copy(), self.cwrap.copy())

    def _restore(self, s):
        k, P, ids32, self.next_id, self.emit_acc, cP, cw = s
        self.k = k
        n = P.shape[0]
        self.n = 0
        if self._P.shape[0] < n:
            self._alloc(2 * n)
        self.n = n
        self._P[:n] = P
        self._ids[:n] = ids32
        self._w[:n] = self._rise(self._ids[:n])          # a pure function of the id: bitwise the same as stored
        self.cP, self.cwrap = cP.copy(), cw.copy()
        self._version += 1

    def step_time(self, k: Optional[int] = None) -> float:
        """Mission time of step k (default: the current state)."""
        return ((self.k if k is None else int(k)) - self.k_warm) * self.ps.dt

    def time(self) -> float:
        return self.step_time()

    def step_index(self, t: float) -> int:
        """The state index reached by advance_to(t)."""
        return max(int(math.floor(float(t) / self.ps.dt + _EPS)) + self.k_warm, 0)

    # -------------------------------------------------------------- clutter clumps
    def _clump_spawn(self, j):
        cs = self.cs
        u1, u2, u3 = hash3(cs.seed, j.astype(np.int64), 7_000_000, salt=17)
        o = self.flow.pose.o
        L = cs.domain
        z = cs.z_band[0] + (cs.z_band[1] - cs.z_band[0]) * u3
        return np.column_stack([o[0] - L / 2 + L * u1, o[1] - L / 2 + L * u2, z])

    def _clump_step(self, V):
        if self.cP.shape[0] == 0:
            return
        f = self.flow.factor_fast(self.cP)
        self.cP[:, 0] += f * (V[0] * self.ps.dt)
        self.cP[:, 1] += f * (V[1] * self.ps.dt)
        o = self.flow.pose.o
        L = self.cs.domain
        for a in (0, 1):
            lo = o[a] - L / 2
            rel = self.cP[:, a] - lo
            out = (rel < 0) | (rel >= L)
            if np.any(out):
                jj = np.nonzero(out)[0]
                self.cwrap[jj] += 1
                self.cP[jj, a] = lo + np.mod(rel[jj], L)
                v1, v2, _ = hash3(self.cs.seed, jj.astype(np.int64) * 1000 + self.cwrap[jj], 3, salt=31)
                b = 1 - a
                self.cP[jj, b] = o[b] - L / 2 + L * v1
                zb = self.cs.z_band
                self.cP[jj, 2] = zb[0] + (zb[1] - zb[0]) * v2

    def clutter_points(self):
        """(M, 3) scatterer positions and (M,) cross-sections [m^2] of all clumps at the current step."""
        cs = self.cs
        if cs is None or cs.n == 0:
            return np.zeros((0, 3)), np.zeros(0)
        if getattr(self, "_c_off", None) is None:
            n = cs.n_pts
            offs, sig = [], []
            for j in range(cs.n):
                ids = j * 10000 + np.arange(n, dtype=np.int64)
                u1, u2, u3 = hash3(cs.seed, ids, 0, salt=41)
                r = cs.radius[j] * np.cbrt(u1)          # uniform in a ball
                ct = 2 * u2 - 1
                ph = 2 * math.pi * u3
                st = np.sqrt(1 - ct * ct)
                offs.append(np.column_stack([r * st * np.cos(ph), r * st * np.sin(ph), r * ct]))
                sig.append(np.full(n, 10 ** (cs.ts_db[j] / 10.0) / n))
            self._c_off = np.stack(offs)                # (n_clumps, n_pts, 3)
            self._c_sig = np.concatenate(sig)
        return (self.cP[:, None, :] + self._c_off).reshape(-1, 3), self._c_sig

    # -------------------------------------------------------------- stepping
    def _emit_count(self):
        ps = self.ps
        j = self.k - self.k_warm
        rs = 1.0 if self.rate_fn is None else float(self.rate_fn(j))
        self.emit_acc += ps.rate * rs * ps.dt
        n = int(math.floor(self.emit_acc + 1e-12))
        self.emit_acc -= n
        return n

    def _step(self):
        ps = self.ps
        t = self.step_time()
        n = self._emit_count()
        if n > 0:
            ids = np.arange(self.next_id, self.next_id + n, dtype=np.int64)
            self.next_id += n
            u1, u2, u3 = hash3(ps.seed, ids, -1, salt=1)
            if self.n + n > self._P.shape[0]:
                self._alloc(2 * (self.n + n))
            a, b = self.n, self.n + n
            j = ps.jitter
            self._P[a:b, 0] = self.src[0] + j * (2.0 * u1 - 1.0)
            self._P[a:b, 1] = self.src[1] + j * (2.0 * u2 - 1.0)
            self._P[a:b, 2] = self.src[2] + j * (2.0 * u3 - 1.0)
            self._ids[a:b] = ids
            self._w[a:b] = self._rise(ids)
            self.n = b
        V = self.flow.V(t)
        if self.n:
            P = self.P
            pr = self.flow.factor_fast(P)
            a = math.sqrt(6.0 * ps.K * ps.dt)
            r1, r2, r3 = hash3(ps.seed, self.ids, self.k, salt=3)
            dt = ps.dt
            P[:, 0] += pr * (V[0] * dt) + (2.0 * a) * r1 - a
            P[:, 1] += pr * (V[1] * dt) + (2.0 * a) * r2 - a
            P[:, 2] += pr * (V[2] * dt) + self.w * dt + (2.0 * a) * r3 - a
            if (self.k % CULL_EVERY) == CULL_EVERY - 1:
                keep = P[:, 2] < self.src[2] + ps.cull_dz
                if not np.all(keep):
                    m = int(keep.sum())
                    self._P[:m] = P[keep]
                    self._ids[:m] = self.ids[keep]
                    self._w[:m] = self.w[keep]
                    self.n = m
        self._clump_step(V)
        self.k += 1
        self._version += 1
        if self.k % self.ck_every == 0 and self.k not in self.ckpts:
            self.ckpts[self.k] = self._snapshot()

    def advance_to(self, t: float) -> "Plume":
        """State at the last step <= t (mission time).  Earlier times restore a checkpoint and re-run."""
        kt = self.step_index(t)
        if kt < self.k:
            kc = max(k for k in self.ckpts if k <= kt)
            self._restore(self.ckpts[kc])
        if self.k < kt:
            t1 = time.perf_counter()
            c1 = time.thread_time()
            n0 = self.k
            while self.k < kt:
                self._step()
            self.n_steps_run += self.k - n0
            self.t_step_s += time.perf_counter() - t1
            self.c_step_s += time.thread_time() - c1
        return self

    # -------------------------------------------------------------- read-out
    def horizontal_index(self, bin_m: float = INDEX_BIN_M) -> HIndex:
        key = (self._version, float(bin_m))
        if self._hidx is None or self._hidx_key != key:
            c0 = time.thread_time()
            self._hidx = HIndex.build(self.P, bin_m)
            self._hidx_key = key
            self.c_index_s += time.thread_time() - c0
            self.n_index_builds += 1
        return self._hidx

    def layer(self, z_lo_w: float, z_hi_w: float) -> np.ndarray:
        """Live particles with z_lo <= z < z_hi (W), copy."""
        P = self.P
        return P[(P[:, 2] >= z_lo_w) & (P[:, 2] < z_hi_w)].copy()

    def state_digest(self) -> str:
        h = hashlib.sha256()
        for a in (self.P, self.ids, self.w, self.cP, self.cwrap, np.array([self.k, self.n, self.next_id]),
                  np.array([self.emit_acc])):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def checkpoint_bytes(self) -> int:
        return int(sum(s[1].nbytes + s[2].nbytes + s[5].nbytes for s in self.ckpts.values()))


# ------------------------------------------------------------------------------------------------ mean path
def profile_integral(a, b, delta: float = BL_DELTA_M, z_ref: float = Z_REF_M):
    """int_a^b (min(h, delta) / z_ref)^(1/7) dh, heights above the sea bed (vectorised)."""
    def F(h):
        h = np.maximum(np.asarray(h, float), 0.0)
        hc = np.minimum(h, delta)
        k = 1.0 + BL_EXPONENT
        return hc ** k / (k * z_ref ** BL_EXPONENT) + np.maximum(h - delta, 0.0) * (delta / z_ref) ** BL_EXPONENT
    return F(b) - F(a)


def drift_offsets(u2_s, z_src_s, w_b, f_inside, dh_above_roof: float = 2.0,
                  env: Optional[StructureEnvelope] = None):
    """Analytic mean drift (no gusts, no diffusion, uniform structure factor inside, sharp envelope) of the exit point
    through the roof and of the section dh above the roof, relative to the source, for a 2 m current u2_s (..., 2) in
    S.  Heights in S (above the mudmat top).  Returns (exit_offset, section_offset) (..., 2)."""
    env = env or StructureEnvelope()
    hb = env.mudmat_thk
    u = np.asarray(u2_s, float)
    oin = (np.asarray(f_inside, float) * profile_integral(np.asarray(z_src_s, float) + hb, env.roof_z + hb)
           / np.asarray(w_b, float))[..., None] * u
    oab = (profile_integral(env.roof_z + hb, env.roof_z + dh_above_roof + hb) / np.asarray(w_b, float))[..., None] * u
    return oin, oin + oab


def mean_path(flow: StationFlow, src_s, w_b: float, t: float, heights_s=(), dz: float = 0.01) -> dict:
    """Deterministic mean trajectory (no gusts, no diffusion) under the instantaneous mean flow at time t, in S.
    Integrates dx/dz = u(x, z) / w_b with the same 1/7 profile, envelope and structure factor as the particles.
    Returns {height_above_mudmat: (x_S, y_S)} for the roof (exit) and each requested height, plus 'inside_exit':
    whether the path is still inside the roof footprint (0.3 m from its edge) where it reaches the roof."""
    pose, env = flow.pose, flow.env
    v2 = flow.v2m(t)
    vS = pose.vec_to_s(np.array([v2[0], v2[1], 0.0]))[:2]
    hs = sorted(set([env.roof_z] + [float(h) for h in heights_s]))
    p = np.array([src_s[0], src_s[1], src_s[2]], float)
    out = {}
    zmax = max(hs)
    zb = flow.seabed_depth + pose.o[2]                   # mudmat top above the sea bed
    hx, hy = env.lx / 2, env.ly / 2
    while p[2] < zmax - 1e-9:
        zm = p[2] + 0.5 * dz
        prof = float(((min(max(zm + zb, 0.0), BL_DELTA_M)) / Z_REF_M) ** BL_EXPONENT)
        fac = float(flow.sfac(pose.to_w(np.array([p[0], p[1], zm]))[None, :])[0])
        p[:2] += vS * prof * fac * dz / w_b
        p[2] += dz
        for hh in hs:
            if hh not in out and p[2] >= hh - 1e-9:
                out[hh] = p[:2].copy()
                if hh == env.roof_z:
                    out["inside_exit"] = bool(abs(p[0]) < hx - 0.3 and abs(p[1]) < hy - 0.3)
    return out
