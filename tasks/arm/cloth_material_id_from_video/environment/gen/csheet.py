"""csheet: the drawn fabric (maintainers' side: the renderer and its checks; never in the image or the agent's home).

The real model's 41 x 21 vertices are a coarse polyline across a bend: straight facets between rows 7.5 mm apart cut
the table's corner (up to 2-3 mm of a drawn fabric inside the table at the front edge). Here the grid is refined
twice (Catmull-Rom, 161 x 81, 1.9 mm), the under side is laid clear of the table's box (0.3 mm) by moving every
under-side point that is nearer than that out of the box along the box's distance gradient, the points between the
jaws are kept on the lower jaw, and the face is the under side plus the drawn thickness along the normals of the
moved under side. The drawn surface therefore differs from the model's vertices: by the contact half-thickness
everywhere (the drawn fabric lies on the table, the model's vertex centres float 1.5 mm above it) and by up to about
4 mm at the table's edge and between the jaws; the fabric's free edges are not moved (tests/review/observability.py
measures what that does to the two scored quantities).
"""
import numpy as np

CLEAR_FACE, CLEAR_EDGE, EDGE_NEAR, EDGE_FAR = 0.0003, 0.0003, 0.003, 0.008      # one clearance everywhere: a larger one near the edge drew a ridge across a stiff sheet
CLEAR_MID, ROUND, ITER = 0.0002, 0.0010, 40
JAW_X0, JAW_X1 = -0.013, 0.012          # the jaws' extent along the clamp's x (into the fabric)
BLADE_TOP = -0.0018                     # the blade's upper face in the clamp frame
PAD_GAP = 0.0001
BLADE_T = 0.00015
JAW_H, JAW_CLEAR = 0.0052, 0.0005


def upsample(Pg):
    def mid(A):
        Ap = np.concatenate([2 * A[:1] - A[1:2], A, 2 * A[-1:] - A[-2:-1]], 0)
        return (-Ap[:-3] + 9 * Ap[1:-2] + 9 * Ap[2:-1] - Ap[3:]) / 16.0
    nx = Pg.shape[0]
    X = np.empty((2 * nx - 1,) + Pg.shape[1:], Pg.dtype)
    X[0::2] = Pg
    X[1::2] = mid(Pg)
    Xt = X.transpose(1, 0, 2)
    ny = Xt.shape[0]
    Y = np.empty((2 * ny - 1,) + Xt.shape[1:], Pg.dtype)
    Y[0::2] = Xt
    Y[1::2] = mid(Xt)
    return Y.transpose(1, 0, 2)


def grid_normals(Pg):
    n = np.cross(np.gradient(Pg, axis=0), np.gradient(Pg, axis=1))
    return n / (np.linalg.norm(n, axis=2, keepdims=True) + 1e-15)


def box_sdf(P, c, h):
    """signed distance of points P (.., 3) to the box of centre c and half sizes h, its gradient, and the distance to
    the box's nearest edge line measured inside the faces (the second largest of the per-axis distances)"""
    q = P - c
    d = np.abs(q) - h
    out = np.maximum(d, 0.0)
    lo = np.linalg.norm(out, axis=-1)
    inside = d.max(-1)
    sd = np.where(lo > 0, lo, inside)
    g = np.where((lo > 0)[..., None], out / (lo[..., None] + 1e-15), np.eye(3)[d.argmax(-1)]) * np.sign(q + 1e-15)
    e = np.sort(d, axis=-1)[..., 1]
    return sd, g, e


def quat_mat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def shell(Vrow, grid, rad, t_f, table, clamp_row, half_w):
    """Vrow (nx * ny, 3) -> dict: bot, top (NX, NY, 3 grids), n (the moved under side's normals), pad_z (the clamp-frame
    height of the pad's lower face over the sheet between the jaws, or None while the jaws hold nothing there)."""
    Pg = upsample(upsample(Vrow.reshape(grid[0], grid[1], 3).astype(np.float64)))
    n = grid_normals(Pg)
    bot = Pg - (rad - CLEAR_FACE) * n
    c, h = np.asarray(table[0], float), np.asarray(table[1], float)

    def state(P):
        sd, _g, e = box_sdf(P, c, h)                      # the true box: the distance that counts
        _s, g, _e = box_sdf(P, c, h - ROUND)              # the direction: the box with its edges rounded, radial at an edge
        w = np.clip((e + EDGE_FAR) / (EDGE_FAR - EDGE_NEAR), 0.0, 1.0)
        return sd, g, w

    def project(bot):
        for _ in range(ITER):
            sd, g, w = state(bot)
            need = CLEAR_FACE + (CLEAR_EDGE - CLEAR_FACE) * w
            push = np.where(sd > 0.02, 0.0, np.maximum(need - sd, 0.0))
            bot = bot + push[..., None] * g
            # the facets: the middle of every grid edge and of every cell must be clear as well (a facet across the
            # table's corner): where one is not, its corner points move out along the direction at that middle
            add = np.zeros_like(bot)
            for mid, parts in ((0.5 * (bot[1:] + bot[:-1]), ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))),
                               (0.5 * (bot[:, 1:] + bot[:, :-1]), ((slice(None), slice(1, None)), (slice(None), slice(None, -1)))),
                               (0.25 * (bot[1:, 1:] + bot[:-1, 1:] + bot[1:, :-1] + bot[:-1, :-1]),
                                ((slice(1, None), slice(1, None)), (slice(None, -1), slice(1, None)), (slice(1, None), slice(None, -1)), (slice(None, -1), slice(None, -1))))):
                sdm, gm, _w = state(mid)
                d = (np.where(sdm > 0.02, 0.0, np.maximum(CLEAR_MID - sdm, 0.0))[..., None] * gm)
                for sl in parts:
                    cur = add[sl]
                    bigger = np.linalg.norm(d, axis=-1) > np.linalg.norm(cur, axis=-1)
                    add[sl] = np.where(bigger[..., None], d, cur)
            if not np.any(add) and not np.any(push):
                break
            bot = bot + 0.7 * add
        return bot

    def blur(D):
        k = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
        for ax in (0, 1):
            Dp = np.concatenate([np.repeat(D.take([0], axis=ax), 2, axis=ax), D, np.repeat(D.take([-1], axis=ax), 2, axis=ax)], axis=ax)
            n_ = D.shape[ax]
            D = sum(k[j] * Dp.take(np.arange(j, j + n_), axis=ax) for j in range(5))
        return D

    # A point moved alone leaves a tent across the sheet at the table's edge (a stiff sheet pivoting on the edge: the
    # model lets it sink 1.2 mm into its contact skin there). The moves are spread over about a centimetre of the
    # sheet, then the sheet is laid clear once more.
    bot0 = bot
    D = bot0 * 0.0
    first = project(bot0) - bot0
    if np.any(first):
        D = 1.25 * blur(blur(blur(first)))
    bot = project(bot0 + D)
    R, p = quat_mat(clamp_row[3:7]), np.asarray(clamp_row[:3], float)
    L = (bot - p) @ R
    foot = (L[..., 0] >= JAW_X0) & (L[..., 0] <= JAW_X1) & (np.abs(L[..., 1]) <= half_w) & (L[..., 2] - BLADE_TOP >= -0.008) & (L[..., 2] - BLADE_TOP <= 0.020)
    closed = bool(clamp_row[7] > 0.5)
    held = closed and bool(foot.any())
    # The model holds the strip by constraints, without jaws: under load the strip tips a few millimetres about the
    # jaw axis. Drawn, it lies flat on the blade between closed jaws, blended back to the model's shape over 8 mm in
    # front of the jaws and 4 mm beside them (a part of the sheet that swings in far under the clamp is left alone).
    # Over an OPEN clamp (sliding in under the edge, the states right after the release) the sheet is only kept above the blade.
    wx = np.clip((JAW_X1 + 0.010 - L[..., 0]) / 0.008, 0.0, 1.0) * (L[..., 0] >= JAW_X0 - 0.004)      # full weight to 2 mm past the jaws: the facets there
    wy = np.clip((half_w + 0.006 - np.abs(L[..., 1])) / 0.004, 0.0, 1.0)
    dz = L[..., 2] - BLADE_TOP
    w = wx * wy * ((dz >= -0.008) & (dz <= (0.020 if closed else 0.008)))      # a held strip can be tipped up by more than 8 mm in a stroke
    flat = w * (BLADE_TOP + PAD_GAP) + (1.0 - w) * L[..., 2]
    L[..., 2] = flat if closed else np.maximum(L[..., 2], np.where(w > 0, flat, L[..., 2]))
    bot = L @ R.T + p
    n2 = grid_normals(bot)
    top = bot + t_f * n2
    pad_z = None
    if held:
        Lt = (top - p) @ R
        pad_z = float(Lt[..., 2][foot].max() + PAD_GAP)
        # the upper jaw is an obstacle for the sheet's face: a thick sheet that bends up right in front of the jaws
        # (a shake's down stroke) would otherwise be drawn through the jaw's front
        lo, hi = np.array([JAW_X0, -half_w, pad_z]), np.array([JAW_X1, half_w, pad_z + JAW_H])
        for _ in range(3):
            sd, g, _e = box_sdf(Lt, (lo + hi) / 2, (hi - lo) / 2)
            under = (Lt[..., 2] <= pad_z) & (Lt[..., 0] <= JAW_X1)
            need = np.where(under, 0.0, JAW_CLEAR)
            push = np.where(sd > 0.01, 0.0, np.maximum(need - sd, 0.0))
            Lt = Lt + push[..., None] * g
        top = Lt @ R.T + p
    return {"bot": bot, "top": top, "n": n2, "pad_z": pad_z}


def topology(nx, ny):
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    uv = np.stack([ii / (nx - 1.0), jj / (ny - 1.0)], -1).reshape(-1, 2)
    q = np.arange(nx * ny).reshape(nx, ny)
    f_top = np.concatenate([np.stack([q[:-1, :-1], q[1:, :-1], q[1:, 1:]], -1).reshape(-1, 3), np.stack([q[:-1, :-1], q[1:, 1:], q[:-1, 1:]], -1).reshape(-1, 3)])
    ring = np.concatenate([q[:-1, 0], q[-1, :-1], q[::-1, -1][:-1], q[0, ::-1][:-1]])
    inner = np.concatenate([q[:-1, 1], q[-2, :-1], q[::-1, -2][:-1], q[1, ::-1][:-1]])
    nr = len(ring)
    r_i = np.arange(nr)
    r_j = (r_i + 1) % nr
    f_edge = np.concatenate([np.stack([r_i, r_j, r_j + nr], 1), np.stack([r_i, r_j + nr, r_i + nr], 1)])
    return uv, f_top, ring, inner, f_edge


def samples(G):
    """points of a drawn surface grid for a distance measurement: its vertices, the middles of its grid edges and of its cells"""
    return np.concatenate([G.reshape(-1, 3), (0.5 * (G[1:] + G[:-1])).reshape(-1, 3), (0.5 * (G[:, 1:] + G[:, :-1])).reshape(-1, 3),
                           (0.25 * (G[1:, 1:] + G[:-1, 1:] + G[1:, :-1] + G[:-1, :-1])).reshape(-1, 3)])
