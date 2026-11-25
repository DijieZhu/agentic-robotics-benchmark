"""Per-heap checks on the compiled model shared by the generator (a placement is rejected when one fails) and the
predicates (G20 class geometry, G24 piece bound); definitions in the predicates' docstring and in module measure.

G20 by class (heap library version 3):
  both    placed rectangle in the class's w and d ranges; compiled footprint inside the rectangle (to within one cell
          of the placed field, 5 mm x scale: the field's roll-off beyond its last raised vertex; the overhang is
          recorded); compiled top in the class's top range
  R       edge rule >= 95 % on every exposed side; core: >= 90 % of the core >= 0.10 m (no low place in the core)
  M       (natural toe, no rim: the edge rule and the core are measured and reported, not judged) compiled footprint
          depth >= 0.28 m; compiled gap between the heap and its line <= 0.05 m (largest over the middle 60 % of the
          width); scale inside the piece's registered interval; and the piece's trial verdict in the registry is true
          (version 3: trials.ok, trials.lines a, b, c, d, verdict.trials, the piece in `usable`) - the only item read
          from the registry, every other one is measured on the compiled model
  No slope band is judged for either class (the version 3 registry has none); the slope median and whether a heap has
  upright faces are measured and recorded.
G24's fit line per class is the registry's (doc.classes.<cls>.fit) and else FIT_SHARE_BY_CLASS.
"""
import numpy as np

from . import library as libmod
from . import measure as ms

FIT_SHARE, FIT_MAX_MM, VERTEX_MM = 0.99, 8.0, 0.5
# Per class (the library's accepted fit lines, as in sitegen_cal/calgen.py): rubble heaps are built of real fragments whose
# crevices are narrower than a height-field cell, so their line is 0.98; mounds keep 0.99. Used when the registry's
# classes carry no `fit`.
FIT_SHARE_BY_CLASS = {"R": 0.98, "M": 0.99}
M_DEPTH_MIN = 0.28           # mounds of fill: compiled footprint depth (the class's depth floor of library version 3)
M_LINE_GAP_MAX = 0.05        # mounds of fill: compiled gap to the line over the middle 60 % of the width
JUDGED = {"R": ("w_ok", "d_ok", "footprint_in_rect", "top_ok", "edge_ok", "core_ok"),
          "M": ("w_ok", "d_ok", "footprint_in_rect", "top_ok", "depth_ok", "line_gap_ok", "scale_registered_ok",
                "piece_trials_ok")}


def fit_line(lib_cls, cls):
    v = (lib_cls.get(cls) or {}).get("fit")
    return float(v) if v is not None else FIT_SHARE_BY_CLASS.get(cls, FIT_SHARE)


def vertex_check(m, d, gid, h, piece, H, meta, seed, k):
    """200 compiled vertices (100 interior, 100 raised) by vertical rays against the piece grid mapped back."""
    cg = ms.compiled_grid(m, gid)
    nrow, ncol = cg["nrow"], cg["ncol"]
    rx, ry = cg["size"][0], cg["size"][1]
    cx, cy = float(d.geom_xpos[gid][0]), float(d.geom_xpos[gid][1])
    xs = cx - rx + 2 * rx * np.arange(ncol) / (ncol - 1)
    ys = cy - ry + 2 * ry * np.arange(nrow) / (nrow - 1)
    g = np.random.default_rng([int(seed), 99, int(k)])
    ii = g.integers(1, nrow - 1, 100)
    jj = g.integers(1, ncol - 1, 100)
    ri, rj = np.nonzero(cg["data"][1:-1, 1:-1] > 0)
    sel = g.integers(0, len(ri), 100)
    ii = np.concatenate([ii, ri[sel] + 1])
    jj = np.concatenate([jj, rj[sel] + 1])
    XY = np.c_[xs[jj], ys[ii]]
    z = ms.vertical_z(m, d, gid, XY)
    place = dict(scale=h["scale"], mirror=h["mirror"], yaw_deg=h["yaw_deg"], cx=h["cx"], y_line=h["y_line"])
    xp, yp = libmod.to_piece(XY[:, 0], XY[:, 1], piece, place)
    sig = h["scale"]
    zref = np.maximum(0.0, sig * libmod.bilinear(H, meta, xp, yp) - libmod.SINK * (sig - 1.0)) - libmod.SINK
    dz = np.abs(z - zref)
    zl = ms.lattice_hits(m, d, gid, XY)[:, 2]
    return dict(n=int(len(z)), nan=int(np.isnan(z).sum()),
                maxdiff_mm=float(1000 * np.max(np.where(np.isnan(dz), np.inf, dz))),
                multiray_vs_vertical_max_mm=float(1000 * np.nanmax(np.abs(zl - z))))


def heap_shape(m, d, gid, h, lib_cls, piece, H, meta, seed, k, mesh):
    sh = ms.shape(m, d, gid, h)
    C = lib_cls[h["cls"]]
    x0, x1, y0, y1 = h["rect"]
    fr = sh.get("footprint_rect_world")
    tol = 1e-6
    # a height field rolls off over one cell beyond its last raised vertex: a piece whose mesh reaches its registered
    # rectangle (library version 3 rectangles are tight to the mesh) has a compiled footprint up to one cell of the
    # placed field (5 mm x scale) outside it; that much is the field's own edge, more is a placement error
    tfp = libmod.GRID * float(h["scale"]) + tol
    vert = sh.get("vertical_share") is not None and sh["vertical_share"] > 0.05
    lo_r, hi_r = (piece.get("scale_registered") or piece["scale"])[:2]
    tv = piece.get("trials_verdict") or {}
    g20 = dict(cls=h["cls"], rect_w=round(x1 - x0, 6), rect_d=round(y1 - y0, 6),
               w_ok=bool(C["w"][0] - tol <= x1 - x0 <= C["w"][1] + tol),
               d_ok=bool(C["d"][0] - tol <= y1 - y0 <= C["d"][1] + tol),
               footprint_in_rect=bool(fr is not None and fr[0] >= x0 - tfp and fr[1] <= x1 + tfp and fr[2] >= y0 - tfp
                                      and fr[3] <= y1 + tfp),
               footprint_overhang_m=(round(max(0.0, x0 - fr[0], fr[1] - x1, y0 - fr[2], fr[3] - y1), 6)
                                     if fr is not None else None),
               footprint_tolerance_m=round(tfp, 6),
               top_ok=bool(sh.get("top_meas") is not None and C["top"][0] - tol <= sh["top_meas"] <= C["top"][1] + tol),
               edge_ok=bool(sh.get("edge_rule_min_side") is not None and sh["edge_rule_min_side"] >= ms.EDGE_FRAC),
               core_ok=bool((sh.get("core_ge010_frac") or 0.0) >= 0.90),
               depth_ok=bool(sh.get("D_meas") is not None and sh["D_meas"] >= M_DEPTH_MIN - tol),
               line_gap_ok=bool(sh.get("line_gap_mid60_max_m") is not None
                                and sh["line_gap_mid60_max_m"] <= M_LINE_GAP_MAX + tol),
               scale_registered_ok=bool(float(lo_r) - 1e-6 <= h["scale"] <= float(hi_r) + 1e-6),
               piece_trials_ok=bool(tv.get("ok") is True),
               piece_trials_verdict=dict(tv),
               upright_faces=bool(vert),
               measured={kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in sh.items()})
    g20["judged"] = list(JUDGED[h["cls"]])
    g20["pass_"] = bool(all(g20[q] for q in JUDGED[h["cls"]]))
    cg = ms.compiled_grid(m, gid)
    vc = vertex_check(m, d, gid, h, piece, H, meta, seed, k)
    V, F = mesh
    place = dict(scale=h["scale"], mirror=h["mirror"], yaw_deg=h["yaw_deg"], cx=h["cx"], y_line=h["y_line"])
    Vs = libmod.to_site(V, piece, place)
    fb = ms.fit_bound(m, d, gid, h, Vs, F)
    lo, hi = piece["scale"]
    line = fit_line(lib_cls, h["cls"])
    g24 = dict(zero_ring=cg["ring_zero"], max_is_size_z=cg["max_is_one"], size_z=cg["size"][2],
               max_height_above_base_m=cg["max_height_m"], vertices=vc,
               fit=dict({kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in fb.items()}), fit_line=line,
               scale=h["scale"], scale_interval=[lo, hi], scale_ok=bool(lo - 1e-6 <= h["scale"] <= hi + 1e-6))
    g24["pass_"] = bool(g24["zero_ring"] and g24["max_is_size_z"] and vc["nan"] == 0 and vc["maxdiff_mm"] <= VERTEX_MM
                        and fb["share_le_3mm"] >= line and fb["max_mm"] <= FIT_MAX_MM and g24["scale_ok"])
    return g20, g24
