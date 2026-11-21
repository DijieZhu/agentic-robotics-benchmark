"""Placement of the items on the return inlets and the measures a generator draws on (the site rules of
the task: placement, hold and cover windows).

Blocker (both tiers): natural placement -- plane rotation U[0, 360) deg, centre offset U(+-0.10) m along the
opening's long axis a and U(+-0.09) m along its short axis b (station_site/2 convention: ``theta_deg`` 0 = the strip
chain along b, ``offset_m`` = [along a, along b]); r_face drawn with the placement from the unit-size range
(``r_face_range``); accepted when the sealed area is in 0.026-0.036 m^2 and the hold in the tier's window.  The sealed
area is what the suction sampler of the scene sees (the geometry and the sampler differ by up to +4 % when a
strip's long side crosses the opening's short side), so a draw is filtered on the sampler's own area:

* ``sampler_area_from_poses`` -- the suction sampler's sealed area (``spacekit.suction.SuctionFaces``: per item body a
  grid of cells with 4 x 4 sub-points on the face that opposes the inlet normal; a sub-point counts inside the opening
  and the seal box) for given body poses, with the sampler's own arithmetic (same arrays, same order, same sums): on
  a scene's body poses it is bit-identical to the scene's sampler at seal 1;
* ``sampler_area`` -- the same on the poses the builder writes for an item lying flat on its inlet (the builder's
  6-significant-digit MJCF numbers mirrored), i.e. the analytic replica of the scene's sealed area at t = 0
  (``covered_flat``; bit-identical to the scene at seal 1 unless a sub-point lies within ~1e-15 m of the opening edge);
* ``geometric_overlap`` -- the exact area of the item's footprint inside the opening (convex clipping), the
  "network calculation" side of the hold check;
* ``unit_hold`` -- the hold of one item = the unit network's inlet pressure difference x its sealed area (the same
  FanNetwork as the scene's ``StationVentilation``: K_face 8, the unit's r_face, 40 cfm per inlet x flow scale,
  design-point fan scaling), with every sealed area of the unit's inlets in place.

Pouches: long side along the opening's long axis, over one end of the opening so that 2-4
sub-point columns of its 12 lie inside (``columns``; the opening edge midway between two columns, 8 mm from both for
the reference pouch), never over an end whose outside has a wall box (``station_v03.json`` pouch_ends: RI-H2-1 +a,
RI-L1-2 -a), the handle's bar centre at least ``HANDLE_CLEAR_M`` beyond the end bar of the frame (a handle over
the end bar 5 / 6 capture time-outs), hold 0.10-0.30 N with the unit's other items in place.

numpy, mujoco (quaternion helpers) and the station library's inlet tables.
"""
from __future__ import annotations

import math

import numpy as np

from . import spec as SPEC

SUB = 4                         # SuctionFaces' sub-points per cell side (default)
GRID_SHEET = (2, 6)             # station.site_rig.ITEM_GRID_SHEET (per strip)
GRID_POUCH = (3, 2)             # station.site_rig.ITEM_GRID_POUCH
G_MAX = 0.012                   # SuctionFaces' seal box: gap <= g_max ...
G_MIN = -0.01                   # ... and > -0.01 m
FRAME_END_BAR = 0.03            # width of the frame bar beyond each end of the opening [m] (world.inlet.frame_m)
HANDLE_CLEAR_M = 0.01           # [C] the pouch handle's bar centre this far beyond the end bar's outer edge
POUCH_COLUMNS = 12              # sub-point columns along a pouch's long side (3 cells x 4 sub-points)
PREFER_COLUMNS = (3, 2, 4)


# ----------------------------------------------------------------------------------------------- site_spec numbers
def opening_half(values=None) -> tuple:
    v = values or SPEC.values()
    a, b = v["world.inlet.opening_m"]
    return (a / 2.0, b / 2.0)


def r_face_range(build: str, n_inlets: int, values=None) -> tuple:
    """(lo, hi) r_face [Pa s/m] of a build on a unit with n parallel inlets (site_spec world.inlet.r_face)."""
    r = (values or SPEC.values())["world.inlet.r_face"]
    if build == "element":
        return tuple(float(x) for x in r["element"])
    if build != "pad":
        raise ValueError(build)
    return tuple(float(x) for x in r["pad_two_inlet_unit" if int(n_inlets) == 2 else "pad_three_inlet_unit"])


def tier_window(tier: str, values=None) -> tuple:
    return tuple(float(x) for x in (values or SPEC.values())["world.blocker.hold_N"][tier])


def cover_window(values=None) -> tuple:
    return tuple(float(x) for x in (values or SPEC.values())["world.blocker.placement"]["covered_m2"])


def natural_placement(rng, values=None) -> dict:
    """theta_deg U[0, 360), offset_m [U(+-offset_long), U(+-offset_short)] (world.blocker.placement)."""
    P = (values or SPEC.values())["world.blocker.placement"]
    th = float(rng.uniform(float(P["theta_deg"][0]), float(P["theta_deg"][1])))
    oa = float(rng.uniform(-float(P["offset_long_m"]), float(P["offset_long_m"])))
    ob = float(rng.uniform(-float(P["offset_short_m"]), float(P["offset_short_m"])))
    return dict(theta_deg=th % 360.0, offset_m=[oa, ob])


def chain_angle_fold(theta_deg: float) -> float:
    """Angle of the strip chain from the opening's short axis, folded to [0, 90] deg (the line, both ends)."""
    t = float(theta_deg) % 180.0
    return float(min(t, 180.0 - t))


# ----------------------------------------------------------------------------------------------- inlet geometry
def inlet(feats: dict, iid: str) -> dict:
    for i in feats["v02"]["inlets"]:
        if i["id"] == iid:
            return i
    raise KeyError(iid)


def unit_of(feats: dict, iid: str) -> str:
    """'cau_<m>' of an inlet."""
    return inlet(feats, iid)["cab_air_unit"]


def unit_inlets(feats: dict, unit: str) -> list:
    """Inlet ids of a cabin air unit, in the station's order (the network's inlet order)."""
    return [i["id"] for i in feats["v02"]["inlets"] if i["cab_air_unit"] == unit]


def n_inlets_of(feats: dict, iid: str) -> int:
    return len(unit_inlets(feats, unit_of(feats, iid)))


def inlet_frame(inl: dict):
    """(face centre, n, a, b) of an inlet's suction face (the station's frame front)."""
    from station import inlets as IN
    n, a, b = IN.frame_axes(inl["normal"], inl["t1"])
    return IN.face_center(inl), n, a, b


class InletFace:
    """An inlet's opening as a face (the interface of ``station.stowage.Face``: c, n, a, b, half, local, in_opening),
    with the numbers of the suction face the station's ventilation builds for it (``spacekit.station`` from the inlet
    table: centre = wall point + n x frame depth, t1 from the inlet's quaternion, t2 = n x t1; ``spacekit.suction.Inlet``
    normalisations) so that the sampler arithmetic on it is the scene's."""

    def __init__(self, inl: dict, values=None):
        from spacekit.suction import Inlet
        from spacekit.station import _quat_to_mat
        self.name, self.kind = inl["id"], "return_inlet"
        R = _quat_to_mat(inl["quat_wxyz"])
        n = np.asarray(inl["normal"], float)
        fr = inl.get("frame_m")
        c = np.asarray(inl["wall_point"], float) + n * float(fr[2] if fr is not None else inl.get("face_offset", 0.030))
        op = inl.get("opening_m", (0.40, 0.12))
        I = Inlet(inl["id"], c, n, R[:, 0], (op[0] / 2, op[1] / 2))
        self.c, self.n, self.a, self.b = I.c, I.n, I.t1, I.t2
        self.half = (float(I.half[0]), float(I.half[1]))

    @property
    def area(self) -> float:
        return 4.0 * self.half[0] * self.half[1]

    def local(self, x):
        d = np.atleast_2d(np.asarray(x, float)) - self.c
        return d @ self.n, d @ self.a, d @ self.b

    def in_opening(self, x, margin: float = 0.0):
        _, s1, s2 = self.local(x)
        return (np.abs(s1) <= self.half[0] - margin) & (np.abs(s2) <= self.half[1] - margin)

    face_local = local


# ----------------------------------------------------------------------------------------------- the sampler's area
def _cells(half, grid, sub: int = SUB):
    """The sampler's cell centres (x, y, -hz) of the back face, the sub-point offsets (cells, sub^2, 3) and the cell
    area -- the arithmetic of ``SuctionFaces.__init__`` (rows of the face with sz = -1)."""
    hx, hy, hz = (float(v) for v in half)
    nx, ny = grid
    cx, cy = 2 * hx / nx, 2 * hy / ny
    xs = ((np.arange(nx) + 0.5) / nx * 2 - 1) * hx
    ys = ((np.arange(ny) + 0.5) / ny * 2 - 1) * hy
    u = (np.arange(sub) + 0.5) / sub - 0.5
    o = np.stack(np.meshgrid(u * cx, u * cy, indexing="ij"), -1).reshape(-1, 2)
    o3 = np.column_stack([o, np.zeros(len(o))])
    rl = np.array([(x, y, -1.0 * hz) for x in xs for y in ys], float)
    return rl, np.repeat(o3[None], len(rl), axis=0), cx * cy


def sampler_area_from_poses(xpos, xmat, half, grid, face, values=None, sub: int = SUB) -> float:
    """Sealed area [m^2] the suction sampler gives at seal 1 for item bodies at (xpos (k, 3), xmat (k, 9) or (k, 3, 3))
    on ``face`` (an ``InletFace`` or a ``station.stowage.Face``; centre c, n, a = t1, b = t2, half): the cells of the
    face of each body that opposes the face normal (cos < -0.5), a sub-point counts when it is inside the opening and
    in the seal box (-0.01 m < gap <= 0.012 m), the cell's share is the mean over its sub-points, the area the sum of
    cell area x share over the cells with a share > 0, bodies in the given order (ascending body id in a scene)."""
    rl, offs, A = _cells(half, grid, sub)
    hx_o, hy_o = face.half
    c, n, t1, t2 = (np.asarray(v, float) for v in (face.c, face.n, face.a, face.b))
    P = np.asarray(xpos, float).reshape(-1, 3)
    R = np.asarray(xmat, float).reshape(-1, 3, 3)
    nl = np.array([0.0, 0.0, -1.0])
    rows_P, rows_R, rows_rl, rows_off = [], [], [], []
    for k in range(len(P)):
        if not (float((R[k] @ nl) @ n) < -0.5):
            continue
        rows_P.append(np.repeat(P[k][None], len(rl), axis=0))
        rows_R.append(np.repeat(R[k][None], len(rl), axis=0))
        rows_rl.append(rl)
        rows_off.append(offs)
    if not rows_P:
        return 0.0
    Rr = np.concatenate(rows_R)
    # the sampler's own expressions (SuctionFaces.accumulate): cell centres, then the sub-points
    p = np.concatenate(rows_P) + np.einsum("bij,bj->bi", Rr, np.concatenate(rows_rl))
    sp = p[:, None, :] + np.einsum("cij,ckj->cki", Rr, np.concatenate(rows_off))
    d = sp - c
    gs = d @ n
    a1 = d @ t1
    a2 = d @ t2
    ins = (np.abs(a1) <= hx_o) & (np.abs(a2) <= hy_o) & (gs <= G_MAX) & (gs > G_MIN)
    sl = (ins * 1.0).mean(axis=1)
    keep = sl > 0
    if not keep.any():
        return 0.0
    Aa = np.full(len(sl), A)
    return 0.0 + float(np.sum(Aa[keep] * sl[keep]))


def _fmt6(x) -> np.ndarray:
    """The numbers as spacekit.fixtures writes them into MJCF (``f'{x:.6g}'``), read back."""
    return np.array([float(f"{float(v):.6g}") for v in np.ravel(x)])


def _quat_xmat(q) -> np.ndarray:
    """Rotation matrix of a written quaternion as the compiled model holds it (normalised, then mju_quat2Mat)."""
    import mujoco
    q = np.array(q, float)
    q = q / math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    mujoco.mju_normalize4(q)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def sheet_poses(inl: dict, B: dict):
    """Body poses (xpos (5, 3), xmat (5, 3, 3)) and strip half sizes of a soft sheet the builder lays on its inlet
    (station.sites.soft_sheet_xml): root = centre - chain x sw x 2, then each strip sw further along the chain."""
    from station import sites as SI
    from station.util import quat_from_mat
    lx, ly, lz = (float(v) for v in B["size"])
    n_s = int(B.get("n_strips", 5))
    sw = lx / n_s
    c, R = SI.item_frame_on_inlet(inl, B.get("theta_deg", 0.0), B.get("offset_m", [0, 0]), lz / 2)
    root = c - R[:, 0] * sw * (n_s // 2)
    Rq = _quat_xmat(_fmt6(quat_from_mat(R)))
    p0 = _fmt6(root)
    xpos = [p0]
    for k in range(1, n_s):
        xpos.append(xpos[-1] + Rq @ np.array([sw, 0.0, 0.0]))
    return np.array(xpos), np.repeat(Rq[None], n_s, axis=0), (sw / 2, ly / 2, lz / 2)


def pouch_pose(inl: dict, S: dict):
    """Body pose (xpos (1, 3), xmat (1, 3, 3)) and half sizes of a pouch the builder lays on its inlet."""
    from station import sites as SI
    from station.util import quat_from_mat
    sx, sy, sz = (float(v) for v in S["size"])
    c, R = SI.item_frame_on_inlet(inl, S.get("theta_deg", 0.0), S.get("offset_m", [0, 0]), sz / 2, chain_along_b=False)
    Rq = _quat_xmat(_fmt6(quat_from_mat(R)))
    return _fmt6(c)[None, :], Rq[None], (sx / 2, sy / 2, sz / 2)


def sampler_area(feats: dict, item: dict, kind: str = "sheet", values=None) -> float:
    """Analytic replica of the scene's sealed area at t = 0 (seal 1) of a blocker (kind 'sheet') or a pouch on its
    inlet, from the site entry alone."""
    inl = inlet(feats, item["inlet"])
    face = InletFace(inl, values)
    if kind == "sheet":
        xp, xm, half = sheet_poses(inl, item)
        return sampler_area_from_poses(xp, xm, half, GRID_SHEET, face, values)
    xp, xm, half = pouch_pose(inl, item)
    return sampler_area_from_poses(xp, xm, half, GRID_POUCH, face, values)


# ----------------------------------------------------------------------------------------------- exact geometry
def _clip(poly: np.ndarray, hx: float, hy: float) -> np.ndarray:
    """Sutherland-Hodgman clipping of a convex polygon (k, 2) to |x| <= hx, |y| <= hy."""
    def clip_edge(P, axis, lim, sign):
        out = []
        n = len(P)
        for i in range(n):
            A, B = P[i], P[(i + 1) % n]
            ina, inb = sign * A[axis] <= lim, sign * B[axis] <= lim
            if ina:
                out.append(A)
            if ina != inb:
                t = (lim - sign * A[axis]) / (sign * (B[axis] - A[axis]))
                out.append(A + t * (B - A))
        return np.array(out) if out else np.zeros((0, 2))
    P = np.asarray(poly, float)
    for axis, lim in ((0, hx), (1, hy)):
        for sign in (1.0, -1.0):
            if len(P) == 0:
                return P
            P = clip_edge(P, axis, lim, sign)
    return P


def _area(P: np.ndarray) -> float:
    if len(P) < 3:
        return 0.0
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def footprint_ab(item: dict, kind: str = "sheet") -> np.ndarray:
    """Corners (4, 2) in the inlet's (a, b) of a flat item's footprint (sheet: theta 0 = chain along b; pouch: theta 0
    = long side along a)."""
    sx, sy = float(item["size"][0]), float(item["size"][1])
    th = math.radians(float(item.get("theta_deg", 0.0)))
    if kind == "sheet":
        x = np.array([-math.sin(th), math.cos(th)])          # chain = cos th b - sin th a
    else:
        x = np.array([math.cos(th), math.sin(th)])           # long side = cos th a + sin th b
    y = np.array([-x[1], x[0]])
    c = np.asarray(item.get("offset_m", [0.0, 0.0]), float)
    return np.array([c + sx / 2 * s1 * x + sy / 2 * s2 * y for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def geometric_overlap(item: dict, kind: str = "sheet", values=None) -> float:
    """Exact area [m^2] of a flat item's footprint inside its inlet's opening."""
    hx, hy = opening_half(values)
    return _area(_clip(footprint_ab(item, kind), hx, hy))


# ----------------------------------------------------------------------------------------------- the unit network
def unit_network(n_inlets: int, build: str, r_face: float, flow_cfm: float = 40.0):
    """The FanNetwork of one cabin air unit (station.inlets.network: n identical 0.40 x 0.12 m inlets)."""
    from station import inlets as IN
    return IN.network(int(n_inlets), build, float(r_face), flow_cfm=float(flow_cfm))


def unit_pc(n_inlets: int, build: str, r_face: float, sealed: list, flow_cfm: float = 40.0) -> tuple:
    """(P_c with the sealed areas on inlets 0.. of the list, P_c unobstructed) [Pa]."""
    net, inls = unit_network(n_inlets, build, r_face, flow_cfm)
    pc0 = float(net.Pc)
    for k, a in enumerate(sealed):
        net.set_blocked(inls[k].name, float(a))
    net._key = None
    net.solve()
    return float(net.Pc), pc0


def unit_hold(n_inlets: int, build: str, r_face: float, sealed: float, others=(), flow_cfm: float = 40.0) -> float:
    """Hold [N] of an item sealing ``sealed`` on one inlet of the unit, the unit's other inlets carrying ``others``."""
    pc, _ = unit_pc(n_inlets, build, r_face, [sealed] + list(others), flow_cfm)
    return pc * float(sealed)


def r_face_for_hold(target_N: float, n_inlets: int, build: str, sealed: float, others=(), flow_cfm: float = 40.0,
                    lo: float = 1.0, hi: float = 600.0) -> float:
    """r_face [Pa s/m] giving the target hold (bisection; the bracket's top is cut to what the fan can serve)."""
    def f(r):
        return unit_hold(n_inlets, build, r, sealed, others, flow_cfm) - float(target_N)
    while hi > lo:
        try:
            f(hi)
            break
        except ValueError:
            hi -= 10.0
    a, b = lo, hi
    fa, fb = f(a), f(b)
    if fa > 0 or fb < 0:
        raise ValueError(f"no r_face in [{lo}, {hi}] gives {target_N} N (holds {fa + target_N:.4f}..{fb + target_N:.4f})")
    for _ in range(80):
        m = 0.5 * (a + b)
        if f(m) > 0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)


# ----------------------------------------------------------------------------------------------- blocker draws
def sample_blocker(rng, tier: str, iid: str, feats: dict, sheet=None, r_face=None, others=(), flow_cfm: float = 40.0,
                   register_ref: str = "LI-01", max_tries: int = 20000, values=None) -> dict:
    """Joint draw: natural placement + r_face from the unit-size range (or the given r_face), until
    the replica's sealed area is in the cover window and the unit network's hold is in the tier window.  ``sheet``: an
    ``items.draw_sheet`` dict (drawn once here when None).  Returns dict(blocker (station_site/2 entry), r_face,
    cover_replica_m2, cover_geometric_m2, hold_N, Pc_Pa, tries)."""
    from . import items as IT
    v = values or SPEC.values()
    build = "element" if tier == "S" else "pad"
    n = n_inlets_of(feats, iid)
    rr = r_face_range(build, n, v)
    cw, hw = cover_window(v), tier_window(tier, v)
    sh = sheet or IT.draw_sheet(rng)
    for k in range(max_tries):
        pl = natural_placement(rng, v)
        r = float(rng.uniform(*rr)) if r_face is None else float(r_face)
        B = IT.blocker_entry(iid, sh, pl["theta_deg"], pl["offset_m"], register_ref)
        A = sampler_area(feats, B, "sheet", v)
        if not (cw[0] <= A <= cw[1]):
            continue
        pc, pc0 = unit_pc(n, build, r, [A] + list(others), flow_cfm)
        h = pc * A
        if hw[0] <= h <= hw[1]:
            return dict(blocker=B, r_face=r, cover_replica_m2=A, cover_geometric_m2=geometric_overlap(B, "sheet", v), hold_N=h,
                        Pc_Pa=pc, Pc_clear_Pa=pc0, tries=k + 1, n_inlets=n, build=build)
    raise RuntimeError(f"no {tier} blocker on {iid} in {max_tries} draws")


# ----------------------------------------------------------------------------------------------- pouches
def pouch_offset_a(size_x: float, columns: int, end: int, values=None) -> float:
    """Centre offset along a that puts ``columns`` of the pouch's 12 sub-point columns inside the opening, the
    opening's end midway between two columns: end x (half_a + size_x / 2 - columns x size_x / 12)."""
    ha, _ = opening_half(values)
    return float(end) * (ha + float(size_x) / 2 - int(columns) * float(size_x) / POUCH_COLUMNS)


def handle_clear(size_x: float, columns: int, values=None, margin: float = HANDLE_CLEAR_M) -> bool:
    """The handle's bar centre (the pouch centre along a) at least ``margin`` beyond the frame's end bar."""
    ha, _ = opening_half(values)
    return abs(pouch_offset_a(size_x, columns, 1, values)) >= ha + FRAME_END_BAR + margin


def end_allowed(iid: str, end: int) -> bool:
    """Whether a pouch may hang over that end of the inlet's opening (station_v03.json pouch_ends; False = a wall box
    outside that end)."""
    from station import data
    ends = data.v03().get("pouch_ends", {})
    key = "+a" if int(end) > 0 else "-a"
    return bool(ends.get(iid, {}).get(key, True))


def pouch_at_end(pid: str, iid: str, pouch: dict, columns: int, end: int, register_ref=None, values=None) -> dict:
    """station_site/2 pouch entry over one end of the opening (theta 0, centred across)."""
    from . import items as IT
    off = [pouch_offset_a(pouch["size"][0], columns, end, values), 0.0]
    return IT.pouch_entry(pid, iid, pouch, off, end, columns, register_ref)


def pouch_options(iid: str, pouch: dict, values=None, margin: float = HANDLE_CLEAR_M) -> list:
    """(columns, end) choices of a pouch on an inlet that respect the end rule and the handle rule, in the preferred
    order (3, 2, 4 columns; +a before -a)."""
    out = []
    for c in PREFER_COLUMNS:
        if not handle_clear(pouch["size"][0], c, values, margin):
            continue
        for e in (1, -1):
            if end_allowed(iid, e):
                out.append((c, e))
    return out


def place_pouch(rng, pid: str, iid: str, pouch: dict, feats: dict, r_face: float, build: str = "pad", others=(),
                flow_cfm: float = 40.0, register_ref=None, values=None) -> dict | None:
    """A pouch on an inlet of a unit whose r_face is fixed: the first (columns, end) option -- the ends in random
    order -- whose replica sealed area is in the pouch cover window and whose hold (the unit's other sealed areas in
    place) is in the hold window; None when no option fits.  Returns dict(pouch entry, cover_replica_m2, hold_N)."""
    v = values or SPEC.values()
    PW = v["world.pouch.placement"]
    cw, hw = PW["covered_m2"], PW["hold_N"]
    n = n_inlets_of(feats, iid)
    opts = pouch_options(iid, pouch, v)
    if len(opts) > 1 and rng is not None and float(rng.uniform()) < 0.5:
        opts = sorted(opts, key=lambda o: (PREFER_COLUMNS.index(o[0]), -o[1]))
    for c, e in opts:
        S = pouch_at_end(pid, iid, pouch, c, e, register_ref, v)
        A = sampler_area(feats, S, "pouch", v)
        if not (cw[0] <= A <= cw[1]):
            continue
        h = unit_hold(n, build, r_face, A, others, flow_cfm)
        if hw[0] <= h <= hw[1]:
            return dict(pouch=S, cover_replica_m2=A, hold_N=h)
    return None


def pouch_for_hold(feats: dict, iid: str, target_N: float, pouch: dict, others=(), flow_cfm: float = 40.0,
                   r_range=None, pid: str = "pouch_1", values=None) -> dict:
    """(r_face, columns, end, entry) that gives ``target_N`` on ``iid`` (pouch_for_hold on the replica's
    sealed area, the pad r_face range of the unit size): the first option of ``pouch_options`` whose r_face solves the
    target inside the range."""
    v = values or SPEC.values()
    n = n_inlets_of(feats, iid)
    rr = r_range or r_face_range("pad", n, v)
    last = None
    for c, e in pouch_options(iid, pouch, v):
        S = pouch_at_end(pid, iid, pouch, c, e, None, v)
        A = sampler_area(feats, S, "pouch", v)
        try:
            r = r_face_for_hold(target_N, n, "pad", A, others, flow_cfm, lo=rr[0], hi=rr[1])
        except ValueError as ex:
            last = str(ex)
            continue
        return dict(r_face=r, columns=c, end=e, pouch=S, cover_replica_m2=A, n_inlets=n)
    raise ValueError(f"no pouch option on {iid} gives {target_N} N in r_face {rr}: {last}")
