"""Loose items of the station scene for robot-free runs and checks: the soft-sheet kinds, the pouch sizes, MJCF of a
free sheet / a sheet or pouch lying on a face, and the geometric sealed area of an item on a face (the measure the
grader's K / L predicates use: back-face area within 10 mm of the face plane and inside the opening).

Contents: KINDS, strip_points, strip_metrics, free_sheet_xml, sheet_on_screen, pouch_xml.  Faces are
``stowage.Face`` objects (holding screen or pouch pad); the K-style predicate is the sealed area, the strip count
stays a metric.
"""
from __future__ import annotations

import numpy as np

from .util import quat_from_mat

# soft-sheet kinds of the family (0.28-0.30 x 0.26-0.30 m, 0.08-0.14 kg, 5 strips, hinge 0.002 N m/rad);
# one corner of the family per kind
KINDS = {
    "bag": dict(size=(0.30, 0.28, 0.004), mass=0.12, colour="0.90 0.55 0.20 1"),
    "cover": dict(size=(0.28, 0.26, 0.004), mass=0.08, colour="0.15 0.55 0.55 1"),
    "towel": dict(size=(0.30, 0.30, 0.004), mass=0.14, colour="0.50 0.35 0.65 1"),
}
POUCHES = {
    "smallest": dict(size=(0.18, 0.11, 0.02), mass=0.06),
    "reference": dict(size=(0.20, 0.12, 0.025), mass=0.10),
    "largest": dict(size=(0.22, 0.14, 0.03), mass=0.15),
}
N_STRIPS = 5
NEAR = 0.010            # a back-face sub-point counts as sealed within this gap above the face plane [m]


def kind_item(kind: str, rng=None, jitter: bool = True) -> dict:
    """Size / mass of one sheet of a kind; with an rng the size is drawn within 1 cm and the mass within 10 % of the
    kind's corner, clipped to the family range."""
    K = KINDS[kind]
    lx, ly, lz = K["size"]
    m = K["mass"]
    if rng is not None and jitter:
        lx = float(np.clip(lx + rng.uniform(-0.01, 0.01), 0.28, 0.30))
        ly = float(np.clip(ly + rng.uniform(-0.01, 0.01), 0.26, 0.30))
        m = float(np.clip(m * (1 + rng.uniform(-0.1, 0.1)), 0.08, 0.14))
    return dict(size=[lx, ly, lz], mass=m, colour=K["colour"], kind=kind)


def sheet_xml(name: str, centre, n, chain, size, mass, hinge_k: float = 0.002, colour: str = KINDS["bag"]["colour"]):
    """A soft sheet (spacekit soft_item, 5 strips, carry strap on the middle strip, the handle loop at the middle of the
    end strip, loop='edge_mid') with its centre at ``centre``, face normal ``n`` (the handle side; the back face looks
    along -n), chain direction ``chain``.  Returns (xml, contact exclusions, strip half sizes)."""
    from spacekit import fixtures as FX
    lx, ly, lz = size
    nn = np.asarray(n, float) / np.linalg.norm(n)
    x = np.asarray(chain, float) - (np.asarray(chain, float) @ nn) * nn
    x /= np.linalg.norm(x)
    y = np.cross(nn, x)
    R = np.column_stack([x, y, nn])
    sw = lx / N_STRIPS
    root = np.asarray(centre, float) - x * sw * (N_STRIPS // 2)
    xml, excl, half = FX.soft_item(name, root, quat_from_mat(R), n=N_STRIPS, size=(lx, ly, lz), mass=mass, hinge_k=hinge_k,
                                   centre_handle=True, loop="edge_mid", material="mat_item_soft")
    xml = xml.replace('material="mat_item_soft"', f'rgba="{colour}"')
    return xml, excl, list(half)


free_sheet_xml = sheet_xml      # the earlier name


def sheet_on_face_xml(name: str, face, item: dict, offset=(0.0, 0.0), chain: str = "b", gap: float = 0.0003, lift: float = 0.0):
    """A sheet lying on a face: back face ``gap`` (+ ``lift``) above the face plane, centre shifted by offset (a, b),
    chain along the face's b (or a)."""
    ch = face.b if chain == "b" else face.a
    c = face.c + face.a * offset[0] + face.b * offset[1] + face.n * (gap + lift + item["size"][2] / 2)
    return sheet_xml(name, c, face.n, ch, item["size"], item["mass"], colour=item.get("colour", KINDS["bag"]["colour"]))


def pouch_on_face_xml(name: str, face, size=POUCHES["reference"]["size"], mass: float = POUCHES["reference"]["mass"],
                      offset=(0.0, 0.0), long_axis: str = "a", gap: float = 0.0003, lift: float = 0.0,
                      colour: str = "0.15 0.45 0.85 1"):
    """A pouch (spacekit small_item with its handle) lying on a face, long side along the face's a (or b).  Returns
    (xml, half sizes)."""
    from spacekit import fixtures as FX
    x = face.a if long_axis == "a" else face.b
    R = np.column_stack([x, np.cross(face.n, x), face.n])
    c = face.c + face.a * offset[0] + face.b * offset[1] + face.n * (gap + lift + size[2] / 2)
    xml, half = FX.small_item(name, c, quat_from_mat(R), size=tuple(size), mass=mass)
    xml = xml.replace('material="mat_item_bag"', f'rgba="{colour}"')
    return xml, list(half)


# ----------------------------------------------------------------------------------------------- measures
def back_points(data, body: int, half, nu: int = 4, nv: int = 8):
    """Sub-points of the back face (local -z) of a box body in world coordinates, and each sub-point's cell area."""
    hx, hy, hz = half
    R = data.xmat[body].reshape(3, 3)
    xs = ((np.arange(nu) + 0.5) / nu * 2 - 1) * hx
    ys = ((np.arange(nv) + 0.5) / nv * 2 - 1) * hy
    loc = np.array([(x, y, -hz) for x in xs for y in ys])
    return data.xpos[body] + loc @ R.T, (2 * hx / nu) * (2 * hy / nv)


strip_points = back_points      # the earlier name


def sealed_area(data, bodies, half, face, near: float = NEAR, nu: int = 4, nv: int = 8) -> float:
    """Geometric sealed area of an item on a face [m^2]: back-face area within ``near`` of the face plane (and not
    more than 5 mm behind it) inside the opening, summed over the item's bodies."""
    area = 0.0
    for b in bodies:
        P, dA = back_points(data, b, half, nu, nv)
        g, _, _ = face.local(P)
        ok = (g <= near) & (g >= -0.005) & face.in_opening(P)
        area += float(ok.sum()) * dA
    return area


def strip_metrics(data, bodies, half, face, near: float = NEAR) -> dict:
    """Per strip of a sheet relative to a face: the back-face centre's gap above the face plane, whether the centre is
    inside the opening, (a, b) of the centre, the strip normal against the face normal; and the geometric sealed area
    of the whole sheet (``sealed_area``)."""
    gaps, inside, nrm, ab = [], [], [], []
    for b in bodies:
        R = data.xmat[b].reshape(3, 3)
        c = data.xpos[b] - R[:, 2] * half[2]
        g, s1, s2 = face.local(c[None, :])
        gaps.append(float(g[0]))
        ab.append((float(s1[0]), float(s2[0])))
        inside.append(bool(face.in_opening(c[None, :])[0]))
        nrm.append(float(R[:, 2] @ face.n))
    return dict(gaps=gaps, inside=inside, covered_m2=sealed_area(data, bodies, half, face, near), normal_dot=nrm, ab=ab)


def n_on(metrics: dict, near: float = NEAR) -> int:
    """Strips whose back-face centre is inside the opening and within -5 mm .. ``near`` of the face plane (metric)."""
    return int(sum(1 for g, i in zip(metrics["gaps"], metrics["inside"]) if i and -0.005 <= g <= near))


def footprint_fraction_on_face(data, body: int, half, face, near: float = NEAR, n: int = 12) -> float:
    """Fraction of an item's own projected footprint (its back face) that lies sealed on a face (the pouch criterion
    measures with it; the grader owns the threshold)."""
    P, dA = back_points(data, body, half, n, n)
    g, _, _ = face.local(P)
    ok = (g <= near) & (g >= -0.005) & face.in_opening(P)
    return float(ok.mean())


def item_centre(data, bodies) -> np.ndarray:
    return np.array([data.xipos[b] for b in bodies]).mean(0)
