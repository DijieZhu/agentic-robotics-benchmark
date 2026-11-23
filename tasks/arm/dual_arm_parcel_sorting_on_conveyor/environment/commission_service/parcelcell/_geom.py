"""Geometry helpers shared by the station builder and the parcel slots.

Parcel face ids (parcel body frame): 0:+z 1:-z 2:+x 3:-x 4:+y 5:-y.
"""

from __future__ import annotations

import mujoco
import numpy as np

FACE_NORMALS = np.array([
    [0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
], dtype=float)
FACE_NAMES = ("+z", "-z", "+x", "-x", "+y", "-y")
# the two size axes spanning each face
FACE_DIMS = {0: (0, 1), 1: (0, 1), 2: (1, 2), 3: (1, 2), 4: (0, 2), 5: (0, 2)}

LABEL_T = 0.0015          # label half thickness
PAD_HIDDEN = 0.001        # rigid parcels: pad shrunk inside the box (no contact)

# contact parameters of the parcel models
RIGID = dict(solimp=[0.95, 0.99, 0.001, 0.5, 2], solref=[0.005, 1])
SOFT_GEOM = dict(solimp=[0.6, 0.95, 0.008, 0.5, 2], solref=[0.02, 0.6])
SOFT_PAD = dict(solimp=[0.5, 0.9, 0.01, 0.5, 2], solref=[0.03, 0.5])


def box_inertia(m: float, half) -> np.ndarray:
    """Diagonal inertia of a solid box with half sizes ``half`` (the formula
    MuJoCo's compiler uses)."""
    a, b, c = half
    return m / 3.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])


def ellipsoid_inertia(m: float, semi) -> np.ndarray:
    a, b, c = semi
    return m / 5.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])


def zaxis_quat(n: np.ndarray) -> np.ndarray:
    """Minimal rotation taking +z to ``n`` (the meaning of MJCF ``zaxis=``)."""
    z = np.array([0.0, 0.0, 1.0])
    d = float(np.dot(z, n))
    if d > 1 - 1e-12:
        return np.array([1.0, 0, 0, 0])
    if d < -1 + 1e-12:
        return np.array([0.0, 1, 0, 0])          # 180 deg about x
    axis = np.cross(z, n)
    axis /= np.linalg.norm(axis)
    half = np.arccos(np.clip(d, -1, 1)) / 2
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])


def refresh_bounds(m: mujoco.MjModel, gid: int) -> None:
    """geom_rbound / geom_aabb are compile-time quantities; after a geom_size
    write they must be refreshed by hand, else the broadphase uses the old
    bounds and misses contacts (seen in the source: a rigid parcel's bottom
    9 mm invisible, parcel resting tilted by 11 deg)."""
    size = m.geom_size[gid]
    if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        m.geom_rbound[gid] = float(np.max(size))
    else:
        m.geom_rbound[gid] = float(np.linalg.norm(size))
    m.geom_aabb[gid, :3] = 0.0
    m.geom_aabb[gid, 3:] = size


def label_layout(size, label_face: int, label_uv=None) -> dict:
    """Label sheet + two dark code blocks + centre site on one face.

    ``size``: full parcel size (x, y, z). ``label_uv``: off-centre placement
    as fractions in [-1, 1] of the free margin along the face's two tangents
    (None = centred). Returns positions / half sizes in the parcel frame."""
    dims = np.asarray(size, dtype=float) / 2
    n = FACE_NORMALS[label_face]
    ax = int(np.argmax(np.abs(n)))
    t1, t2 = (ax + 1) % 3, (ax + 2) % 3
    pos = n * (dims[ax] + LABEL_T)
    lsize = np.full(3, LABEL_T)
    lsize[t1] = dims[t1] * 0.62
    lsize[t2] = dims[t2] * 0.62
    tang1 = np.zeros(3); tang1[t1] = 1
    tang2 = np.zeros(3); tang2[t2] = 1
    if label_uv is not None:
        u = float(np.clip(label_uv[0], -1, 1))
        v = float(np.clip(label_uv[1], -1, 1))
        pos = pos + tang1 * u * (dims[t1] - lsize[t1]) \
                  + tang2 * v * (dims[t2] - lsize[t2])
    bsize = lsize * 0.42; bsize[ax] = LABEL_T
    off1 = tang1 * lsize[t1] * 0.45 + n * LABEL_T
    off2 = -tang1 * lsize[t1] * 0.4 - tang2 * lsize[t2] * 0.3 + n * LABEL_T
    bsize2 = lsize * 0.3; bsize2[ax] = LABEL_T
    return {
        "label": (pos, lsize),
        "label_qr": (pos + off1, bsize * 0.9),
        "label_bar": (pos + off2, bsize2),
        "site_pos": pos,
        "site_quat": zaxis_quat(n),
    }


def contact_values(overrides: dict | None = None) -> dict:
    """The parcel models' contact values {"rigid", "soft_geom", "soft_pad"},
    each {"solimp": [5], "solref": [2]}, with optional overrides."""
    out = {"rigid": dict(RIGID), "soft_geom": dict(SOFT_GEOM),
           "soft_pad": dict(SOFT_PAD)}
    for k, v in (overrides or {}).items():
        if k not in out:
            raise ValueError(f"unknown parcel contact class {k!r}")
        for f, x in v.items():
            if f not in ("solimp", "solref"):
                raise ValueError(f"unknown parcel contact field {f!r}")
            out[k][f] = [float(a) for a in x]
    return out


def parcel_fields(kind: str, size, mass: float, friction: float,
                  contact: dict | None = None) -> dict:
    """Collision geometry, contact parameters and mass properties of the
    simulator's two parcel models.

    rigid: homogeneous box; the pad ellipsoid is shrunk inside the box.
    soft (mailer bag): flattened box (85 % height) + ellipsoid pad with soft
    contact; total mass split 0.9 / 0.1 between them. ``contact``: overrides
    of the contact values (``contact_values``)."""
    cv = contact_values(contact)
    RIGID, SOFT_GEOM, SOFT_PAD = cv["rigid"], cv["soft_geom"], cv["soft_pad"]
    dims = np.asarray(size, dtype=float) / 2
    if kind == "rigid":
        return dict(
            box_size=dims.copy(), box_friction=[friction, 0.008, 0.0002],
            box_solimp=RIGID["solimp"], box_solref=RIGID["solref"],
            pad_size=np.full(3, PAD_HIDDEN), pad_friction=None,
            pad_solimp=None, pad_solref=None,
            mass=float(mass), inertia=box_inertia(mass, dims))
    if kind == "soft":
        box_half = np.array([dims[0], dims[1], dims[2] * 0.85])
        pad_semi = np.array([dims[0] * 0.92, dims[1] * 0.92, dims[2]])
        f = [friction + 0.15, 0.01, 0.0003]
        return dict(
            box_size=box_half, box_friction=f,
            box_solimp=SOFT_GEOM["solimp"], box_solref=SOFT_GEOM["solref"],
            pad_size=pad_semi, pad_friction=list(f),
            pad_solimp=SOFT_PAD["solimp"], pad_solref=SOFT_PAD["solref"],
            mass=float(mass),
            inertia=(box_inertia(mass * 0.9, box_half)
                     + ellipsoid_inertia(mass * 0.1, pad_semi)))
    raise ValueError(f"unknown parcel kind {kind!r}")
