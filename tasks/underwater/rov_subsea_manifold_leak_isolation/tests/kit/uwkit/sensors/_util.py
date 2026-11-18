"""Small helpers shared by the sensor models (numpy + mujoco only).

Conventions (project-wide):
  * World W: right-handed, z up, free surface at z = 0, depth = -z.  For magnetics and headings W is taken as
    ENU (x east, y north) unless a sensor is given another north vector.
  * Body B: FLU (x forward, y left/port, z up).
  * Quaternions are (w, x, y, z) (MuJoCo order).
  * All randomness comes from explicitly passed numpy.random.Generator objects (PCG64).  Nothing here reads the
    wall clock or the global numpy RNG.

These helpers live in uwkit/sensors/ because the root uwkit/frames.py (owned by another lane) did not exist when
this package was written.  They are deliberately tiny and dependency-free.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import mujoco

# ----------------------------------------------------------------------------------------------------------- rotations


def quat_to_mat(q: Sequence[float]) -> np.ndarray:
    """(w, x, y, z) unit quaternion -> 3x3 rotation matrix (columns = rotated frame axes in parent)."""
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).reshape(9))
    return q


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def euler_zyx_from_mat(R: np.ndarray) -> tuple[float, float, float]:
    """R = Rz(yaw) Ry(pitch) Rx(roll) (FLU body in ENU-like world).  Returns (roll, pitch, yaw) in rad.

    FLU sign convention: positive pitch (about +y = port axis) = nose DOWN; positive roll (about +x) = port side up
    / starboard side down; positive yaw (about +z) = nose turns to port (counter-clockwise seen from above).
    Compass-style angles (heading clockwise from north) are produced by the AHRS model, not here.
    """
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def mat_from_euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = w / th
    K = skew(k)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def orthonormalize(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    return u @ vt


# ----------------------------------------------------------------------------------------------------------- mujoco


def site_id(m: mujoco.MjModel, name: str) -> int:
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
    if i < 0:
        raise KeyError(f"site '{name}' not found in model")
    return i


def site_pose(d: mujoco.MjData, sid: int) -> tuple[np.ndarray, np.ndarray]:
    """World position (3,) and rotation (3x3, columns = site axes in W) of a site (copies)."""
    return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()


def site_velocity_world(m: mujoco.MjModel, d: mujoco.MjData, sid: int) -> tuple[np.ndarray, np.ndarray]:
    """(linear velocity of the site point in W, angular velocity in W).  Valid after mj_forward / mj_step."""
    res = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, res, 0)
    return res[3:].copy(), res[:3].copy()


def root_body_of_site(m: mujoco.MjModel, sid: int) -> int:
    b = int(m.site_bodyid[sid])
    return int(m.body_rootid[b])


class RayCaster:
    """Thin wrapper around mj_multiRay / mj_ray.

    geomgroup: optional 6-vector of 0/1 (which geom groups the rays see).  The project convention for excluding the
    vehicle's own geoms is to pass bodyexclude = vehicle root body *and* keep vehicle sub-bodies (gripper etc.) in a
    geom group that is masked out here.
    """

    def __init__(self, m: mujoco.MjModel, geomgroup: Optional[Sequence[int]] = None, bodyexclude: int = -1,
                 flg_static: bool = True):
        self.m = m
        self.geomgroup = None if geomgroup is None else np.asarray(geomgroup, dtype=np.uint8).reshape(6)
        self.bodyexclude = int(bodyexclude)
        self.flg_static = bool(flg_static)

    def cast(self, d: mujoco.MjData, origin: np.ndarray, dirs: np.ndarray, cutoff: float):
        """dirs: (N, 3) unit vectors (world).  Returns geomid (N,) int32 (-1 = miss), dist (N,) (-1 = miss),
        normal (N, 3) world unit normals (garbage where missed)."""
        dirs = np.ascontiguousarray(dirs, dtype=np.float64).reshape(-1, 3)
        n = dirs.shape[0]
        gid = np.empty(n, np.int32)
        dist = np.empty(n, np.float64)
        nrm = np.empty(3 * n, np.float64)
        if n:
            mujoco.mj_multiRay(self.m, d, np.ascontiguousarray(origin, dtype=np.float64), dirs.reshape(-1),
                               self.geomgroup, self.flg_static, self.bodyexclude, gid, dist, nrm, n, float(cutoff))
        return gid, dist, nrm.reshape(n, 3)

    def cast_one(self, d: mujoco.MjData, origin: np.ndarray, direction: np.ndarray):
        gid = np.array([-1], np.int32)
        nrm = np.zeros(3)
        dist = mujoco.mj_ray(self.m, d, np.ascontiguousarray(origin, dtype=np.float64),
                             np.ascontiguousarray(direction, dtype=np.float64), self.geomgroup, self.flg_static,
                             self.bodyexclude, gid, nrm)
        return int(gid[0]), float(dist), nrm


# ----------------------------------------------------------------------------------------------------------- RNG


def stable_hash32(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def child_rng(parent: np.random.Generator, n: int = 1) -> list[np.random.Generator]:
    """Spawn n independent child Generators from a parent (numpy >= 1.25)."""
    return list(parent.spawn(n))


def rng_from_seed(seed: int, name: str = "") -> np.random.Generator:
    """Deterministic PCG64 Generator for (seed, name): different names give independent streams."""
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=(stable_hash32(name),) if name else ())
    return np.random.Generator(np.random.PCG64(ss))


_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLD = np.uint64(0x9E3779B97F4A7C15)


def _splitmix(x: np.ndarray) -> np.ndarray:
    x = x + _GOLD
    x = (x ^ (x >> np.uint64(30))) * _M1
    x = (x ^ (x >> np.uint64(27))) * _M2
    return x ^ (x >> np.uint64(31))


def hash_uniform(seed: int, *keys: np.ndarray) -> np.ndarray:
    """Counter-based uniform [0, 1) numbers: a pure function of (seed, keys...), vectorised.

    Used for world-anchored random fields (marine snow particles, sea-bed micro texture) so that the result does not
    depend on frame order.  `seed` should itself be drawn from the world's Generator.
    """
    with np.errstate(over="ignore"):
        h = np.full(np.broadcast(*keys).shape if keys else (), np.uint64(seed & 0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        for k in keys:
            h = _splitmix(h ^ np.asarray(k).astype(np.int64).astype(np.uint64))
        h = _splitmix(h)
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


# ----------------------------------------------------------------------------------------------------------- timing


class RateScheduler:
    """Fires at t0, t0 + T, t0 + 2T, ... in *simulation* time (no drift).  `period` may be changed on the fly (the
    next firing is then scheduled from the last firing)."""

    def __init__(self, rate_hz: float, t0: float = 0.0):
        self.period = 1.0 / float(rate_hz)
        self.next_t = float(t0)
        self.last_t: Optional[float] = None

    def due(self, t: float) -> bool:
        return t + 1e-12 >= self.next_t

    def fire(self, t_fire: Optional[float] = None) -> float:
        """Mark a firing at the scheduled time (or t_fire) and schedule the next one.  Returns the firing time."""
        tf = self.next_t if t_fire is None else float(t_fire)
        self.last_t = tf
        self.next_t = tf + self.period
        return tf

    def set_rate(self, rate_hz: float) -> None:
        self.period = 1.0 / float(rate_hz)
        if self.last_t is not None:
            self.next_t = self.last_t + self.period


class DelayLine:
    """FIFO of (release_time, item).  Items become available once sim time >= release_time."""

    def __init__(self):
        self._q: list[tuple[float, int, Any]] = []
        self._n = 0

    def push(self, release_t: float, item: Any) -> None:
        self._q.append((float(release_t), self._n, item))
        self._n += 1
        self._q.sort(key=lambda e: (e[0], e[1]))

    def pop_ready(self, t: float) -> list[Any]:
        out = []
        while self._q and self._q[0][0] <= t + 1e-12:
            out.append(self._q.pop(0)[2])
        return out

    def __len__(self):
        return len(self._q)


class GaussMarkov:
    """First-order Gauss-Markov process (vector).  tau = inf gives a random walk with sigma_rw [unit/sqrt(s)];
    finite tau gives stationary std `sigma` with correlation time tau."""

    def __init__(self, rng: np.random.Generator, dim: int, sigma: float = 0.0, tau: float = math.inf,
                 sigma_rw: float = 0.0, x0: Optional[np.ndarray] = None):
        self.rng = rng
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.sigma_rw = float(sigma_rw)
        self.x = np.zeros(dim) if x0 is None else np.array(x0, dtype=float)

    def step(self, dt: float) -> np.ndarray:
        if dt <= 0:
            return self.x
        if math.isinf(self.tau):
            self.x = self.x + self.sigma_rw * math.sqrt(dt) * self.rng.standard_normal(self.x.shape)
        else:
            phi = math.exp(-dt / self.tau)
            q = self.sigma * math.sqrt(max(0.0, 1.0 - phi * phi))
            self.x = phi * self.x + q * self.rng.standard_normal(self.x.shape)
        return self.x


# ----------------------------------------------------------------------------------------------------------- spec


@dataclass
class SpecItem:
    """One line of a datasheet-level public spec table."""
    key: str
    value: Any
    unit: str = ""
    note: str = ""


def spec_table(items: Iterable[SpecItem]) -> list[dict]:
    return [asdict(i) for i in items]


def dataclass_to_dict(obj) -> dict:
    out = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, np.ndarray):
            v = v.tolist()
        out[f.name] = v
    return out


# ----------------------------------------------------------------------------------------------------------- PNG


def write_png(path: str, img: np.ndarray) -> None:
    """Minimal PNG writer (stdlib zlib only).  img: (H, W) or (H, W, 3|4) uint8."""
    import struct
    a = np.ascontiguousarray(img)
    if a.dtype != np.uint8:
        raise TypeError("uint8 expected")
    if a.ndim == 2:
        ctype, ch = 0, 1
        a = a[:, :, None]
    else:
        ch = a.shape[2]
        ctype = {3: 2, 4: 6}[ch]
    h, w = a.shape[:2]
    raw = b"".join(b"\x00" + a[r].tobytes() for r in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


__all__ = [n for n in dir() if not n.startswith("_") or n in ("_splitmix",)]
