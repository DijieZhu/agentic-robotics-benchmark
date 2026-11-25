"""Rubble boxes in the straight runs under the slab, rule "A" (hidden side; the rule's final text: the probe's draws
plus one rejection R4).

Notation (the void of one mouth tau; the "true mouth" below is mouth tau, also on a counterfactual site):
  P            the P class clear width W_SWEEP + 0.14 (the mouths' width); hz the slab underside
  mouth        [m0, m1] x [y2, yg]; inner end = the segment (m0, yg)-(m1, yg)
  G1           [gx0, gx1] x [yg, yg + g1]; connector N [nx0, nx1] x [yg + g1, yc0]; back corridor K
               [kx0, kx1] x [yc0, yc0 + wK] (none on an L site); chamber C [cx0, cx1] x [yc0, LY], inner wall x = xc
  corner points  G1 / N inner (N's mouth-side wall x G1's north wall) and outer (G1's end wall x G1's south wall);
               (nx0, yc0), (nx1, yc0); with K: (xc, yc0 + wK)
  straight runs  G1 from the mouth's FAR edge to the connector's near edge (connector east: [m0, nx0]; west:
               [nx1, m1]) across [yg, yg + g1]; N along y in [yg + g1, yc0] across [nx0, nx1]; K from the connector's
               square to the chamber's inner wall ([nx1, xc] or [xc, nx0]) across [yc0, yc0 + wK]

Draws (one fresh numpy default_rng([seed, 13]) per void; runs in the order G1, N, K):
  n = 1 if rng.random() < 0.5 else 2
  for j in 0 .. n - 1, at most 300 draws:   L = U(0.20, 0.30); stop the run if its length < L;
      s0 = U(s_lo, s_hi - L); side = low if rng.random() < 0.5 else high (against the wall at c_lo or c_hi);
      box = [s0, s0 + L] along x [c_lo, c_lo + (w - P)] or [c_hi - (w - P), c_hi] across, z 0 .. hz
      reject  R1  Euclidean distance box - inner end of the mouth < 0.5 m (exact; the inner end is horizontal)
              R2  Euclidean distance box - any corner point < 0.5 m
              R3  |centre - centre'| < 0.8 + (L + L') / 2 for a box already placed in the same run
              R4  the run is G1 and the box's interval along it meets (m0, m1) (the mouth's junction square)
      else keep it; 300 rejected draws (or a run too short): no more boxes in this run.
Three points the text leaves open, fixed here: (i) the G1 interval starts at the mouth's far edge and R4 rejects
(the interval is not shortened); (ii) N draws its coin and its tries like any run (it can never hold a box, but the
numbers it consumes decide K's boxes); (iii) R1 is the exact distance.
The N run never holds a box (both ends are corner points and N is only as long as the partition).
"""
import math

import numpy as np

RULE = "A"
STREAM = "default_rng([seed, 13])"
CLEAR_M = 0.5        # to the mouth's inner end and to every corner point
PAIR_M = 0.8         # free stretch between two boxes of one run
LEN_M = (0.20, 0.30)
TRIES = 300


def _rect_hseg_dist(r, x0, x1, y):
    """Exact Euclidean distance between the rectangle r = (x0, x1, y0, y1) and the horizontal segment
    (x0, y)-(x1, y)."""
    dx = max(x0 - r[1], 0.0, r[0] - x1)
    dy = max(r[2] - y, 0.0, y - r[3])
    return math.hypot(dx, dy)


def _rect_pt_dist(r, p):
    return math.hypot(max(r[0] - p[0], 0.0, p[0] - r[1]), max(r[2] - p[1], 0.0, p[1] - r[3]))


def geometry(vis, hid):
    """The runs (name, axis, s_lo, s_hi, c_lo, c_hi), the corner points and the mouth of one void."""
    M = vis["mouths"][hid["tau"]]
    m0, m1 = M["x0"], M["x1"]
    xM = 0.5 * (m0 + m1)
    _gx0, _gx1, gy0, gy1 = hid["G1"]
    nx0, nx1, ny0, ny1 = hid["N"]
    cx0, cx1 = hid["chamber"][0], hid["chamber"][1]
    east = hid["xN"] > xM
    corners = [((nx0 if east else nx1), gy1), ((nx1 if east else nx0), gy0), (nx0, ny1), (nx1, ny1)]
    runs = [("G1", "x", (m0 if east else nx1), (nx0 if east else m1), gy0, gy1), ("N", "y", ny0, ny1, nx0, nx1)]
    if hid["K"] is not None:
        _kx0, _kx1, ky0, ky1 = hid["K"]
        k_east = 0.5 * (cx0 + cx1) > hid["xN"]
        xc = cx0 if k_east else cx1
        corners.append((xc, ky1))
        runs.append(("K", "x", (nx1 if k_east else cx1), (cx0 if k_east else nx0), ky0, ky1))
    return dict(runs=runs, corners=corners, mouth=(m0, m1), yg=gy0)


def place(seed, vis, hid, P):
    """[(name, kind, (x0, x1, y0, y1)), ...] of the rubble boxes of one void and the record of the draws."""
    rng = np.random.default_rng([int(seed), 13])
    U = lambda lo, hi: float(rng.uniform(lo, hi))
    g = geometry(vis, hid)
    m0, m1 = g["mouth"]
    yg = g["yg"]
    boxes, recs, not_placed = [], [], []
    rejections = {"R1": 0, "R2": 0, "R3": 0, "R4": 0}
    for name, ax, s_lo, s_hi, c_lo, c_hi in g["runs"]:
        width = c_hi - c_lo
        across = width - P
        n_want = 1 if rng.random() < 0.5 else 2
        placed = []
        for j in range(n_want):
            done, short = False, False
            for _ in range(TRIES):
                L = U(*LEN_M)
                if s_hi - s_lo < L:
                    short = True
                    break
                s0 = U(s_lo, s_hi - L)
                low = rng.random() < 0.5
                q0, q1 = (c_lo, c_lo + across) if low else (c_hi - across, c_hi)
                bx = (s0, s0 + L, q0, q1) if ax == "x" else (q0, q1, s0, s0 + L)
                if _rect_hseg_dist(bx, m0, m1, yg) < CLEAR_M:
                    rejections["R1"] += 1
                    continue
                if any(_rect_pt_dist(bx, q) < CLEAR_M for q in g["corners"]):
                    rejections["R2"] += 1
                    continue
                if any(abs(s0 + 0.5 * L - pc) < PAIR_M + 0.5 * (L + pl) for pc, pl in placed):
                    rejections["R3"] += 1
                    continue
                if name == "G1" and bx[1] > m0 + 1e-9 and bx[0] < m1 - 1e-9:
                    rejections["R4"] += 1
                    continue
                placed.append((s0 + 0.5 * L, L))
                wall = ("south" if low else "north") if ax == "x" else ("west" if low else "east")
                boxes.append(("void_debris_%s_%d" % (name, j), bx))
                recs.append(dict(run=name, j=j, side="low" if low else "high", wall=wall, length_m=L, across_m=across,
                                 gap_m=P, run_width_m=width))
                done = True
                break
            if not done:
                not_placed.append(dict(run=name, j=j, wanted=n_want,
                                       reason="the run is shorter than the drawn length" if short
                                       else "no admissible position in %d draws" % TRIES))
                break
    return boxes, dict(rule=RULE, stream=STREAM, boxes=recs, not_placed=not_placed, rejections=rejections)
