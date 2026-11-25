#!/usr/bin/env python3
"""QC Q8 (task version 2.0): is every heap on the open floor of a frozen site really impassable?

For every heap of a site package (site.json "heaps" block; the calibration site's two heaps as well), in THAT site's
own compiled world (the cell's own builder cellcore.world.new_simulator on the package's site.xml: the site's floor,
patch and heap friction, every wall, box, ridge, slab, crate and the other heaps), privileged trials from each EXPOSED
side, TRIALS_PER_SIDE (4) per side, so usually 12 for a heap that leans on a line with both flanks open.

Measuring protocol = the heap capability tool's (heapcap 1.0.0, not published; imported read-only: its cell loader, its
primitive bookkeeping, its flat-floor table, its rolls, its turn-away rule), written for a heap in a site instead of a
heap in front of a 0.9 m wall:
  local frame of a heap: u along its line (local west -> east), v from its front (v = 0, the side away from the line)
  towards the line (v = depth); a north-line heap's local frame is the world's shifted, a south-line heap's is the
  world's turned by 180 deg.
  sides: F (front, robot from v < 0), L (local west, robot from u < u0), R (local east); the line side is never
  exposed. A side is EXPOSED when no obstacle rectangle (site.json boxes standing on the floor below 0.30 m: walls,
  supports, rubble, plugs, crates, the other heaps) lies within STRIP (0.05 m) of it and the open floor before it is
  >= RUNUP_MIN (0.45 m) deep (the generator's spacing rule makes every gap 0 or >= 0.60 m).
  protocols by k (trial index within the side): F: k even 'into' (heapcap: 5 rolls of one roll primitive, start
  0.37-0.72 m before the front, offset across the width +- 0.12 m beyond it, heading +- 20 deg), k odd 'push'
  (heapcap: the base's PRIVILEGED steering device, line through cu + 0.3 (start - cu) heading at the line, 12
  primitives, start 0.37-0.62 m, offset +- 0.35 of the half width, heading +- 15 deg); L / R: k even 'along'
  (heapcap's along from that side: 6 rolls, alpha 0-45 deg towards the line, run-up 0.35-0.70 m), k odd 'push'
  from that side (the same device, line through cv + 0.3 (start - cv) heading across the heap). The push protocol
  does NOT stop when the old 'crossed' fires (as the library's 60 trials per piece: the criterion is the depth).
  Start distances are drawn as in heapcap and capped by the open floor (robot centre >= CLEAR 0.20 m from every
  obstacle rectangle and >= 0.30 m outside the heap's rectangle); a draw that does not fit is redrawn (sub-draw
  stream), at most 40 times; when none fits, the same draws in order, each moved sideways (across its approach) by
  the smallest 0.01 m step up to 0.30 m that fits (heading and steering line as drawn). Then heapcap's turn-away for
  every side: turn until the rolling direction is within 50 deg of the front's outward normal (away from the line;
  <= 30 turns), 3 rolls away; turn-away ok = rollable, third-lowest cap <= 0.03 m, centroid >= 0.25 m beyond the
  footprint's front (heapcap: 'centroid >= 0.25 m south of the front of the footprint').
  every trial: the site's own friction (no draw), pre-rolls 0-2, trial stream default_rng([Q8_TAG, crc32(site name),
  heap index, side code, k, sub-draw]).
Heap shape measured ON THE COMPILED MODEL (that heap's geom alone in geom group 5, vertical mj_ray nudged by
(1.1e-6, 0.7e-6) m off the target): footprint = heights > 1 mm on a 2.5 mm lattice over the rectangle + 0.02 m;
centroid depth inside the outline = distance transform of the footprint with the line side counted as inside;
measured depth D = the footprint's extent away from the line; crest region = footprint cells >= 0.85 x measured top;
core (old 'crossed', reported only) = footprint shrunk by half about its area centroid.

Per-trial criteria (the heap library's lines for R and for M, the mounds with a natural toe; every trial judged):
  R  centroid depth <= 0.10 m and <= 40 % of D
  M  (a) centroid depth <= 0.16 m and <= 50 % of D; (d) no primitive ends with all three lowest end caps above
     0.03 m (lowest cap centre z1 > 0.03 m), and no centroid sample inside the crest region
Ratios (the library's lines (b) and (c): per piece at most 1 of 60 trials fails), counted by class and reported,
never a per-trial failure: turn-away ok; an end triangle after a primitive that touched the heap. The tool's old
'crossed' is reported, never a criterion. The rubble under the slab is not in Q8 (D9 and R8 as in v1.1).

State record of every trial (real states beside the heaps, for the end-cap colour check): one npz per trial with qpos, qvel and the
simulation time at the start and at the end of every primitive, the primitive names and phases.

Commands (numerics as test.sh: OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 single-threaded BLAS; python -I -B;
nice -n 10; the cell copy = tests/voidcell + tests/reference only, never tests/hidden):
  q8_heap_trials.py plan --manifest M.json --out DIR             # heaps, exposed sides, jobs (no simulation)
  q8_heap_trials.py work --manifest M.json --out DIR --worker I [--cell CELL]   # claim and run jobs (resumable)
  q8_heap_trials.py summary --manifest M.json --out DIR --public P.json --private Q.json [--accepted A.json]
manifest: {"sites": [{"name": "h01", "package": "/path/h01.tar.xz", "kind": "hidden"|"practice"|"cal"|"dev"}, ...]}.
Everything per heap or per trial goes to DIR (keep it mode 700); the console and the public summary carry only
neutral site names, counts and pass / fail (hidden sites: pass / fail per site only).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
import zlib

import numpy as np

TOOL_VERSION = "1.1.0"     # 1.1.0: turn-away towards the front for every side (heapcap's), the sideways start fall-back
HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(DEV, "tools", "heapcap"))
import heapcap as hc  # noqa: E402  (the measuring protocol; read-only import)

Q8_TAG = 20261122
TRIALS_PER_SIDE = 4
RAY_STEP = 0.0025
PAD = 0.02
NUDGE = (1.1e-6, 0.7e-6)
GROUP = 5
STRIP = 0.05
RUNUP_MIN = 0.45
RUNUP_MAX = 1.20
CLEAR = 0.20
OUTSIDE = 0.30
OBST_Z0 = 0.30
CREST_FRAC = 0.85
MAX_SUB = 40
LINES = {"R": dict(depth_m=0.10, share=0.40), "M": dict(depth_m=0.16, share=0.50)}
SIDE_CODE = {"F": 0, "L": 1, "R": 2}

mujoco = cw = None


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(cell):
    global mujoco, cw
    if mujoco is None:
        hc.load_cell(cell)
        import mujoco as _mj
        from cellcore import world as _cw
        mujoco, cw = _mj, _cw


def open_site(path):
    import sitepkg   # the cell's own reader (vendored _lib is on sys.path after load())
    return sitepkg.open_site(path)


# ===================================================================== frames and sides
class Frame:
    """Local frame of a heap: u along the line, v from the front (0) towards the line (depth)."""

    def __init__(self, heap):
        x0, x1, y0, y1 = map(float, heap["rect"])
        self.north = heap["line"] == "north"
        self.cx = 0.5 * (x0 + x1)
        self.rect = (x0, x1, y0, y1)
        self.depth = y1 - y0
        self.u0, self.u1 = (x0 - self.cx, x1 - self.cx) if self.north else (self.cx - x1, self.cx - x0)
        self.y0, self.y1 = y0, y1

    def to_world(self, u, v):
        if self.north:
            return (self.cx + u, self.y0 + v)
        return (self.cx - u, self.y1 - v)

    def heading(self, h_local):
        return float((h_local + (0.0 if self.north else 180.0)) % 360.0)

    def normal(self, side):
        """Outward normal of a side in the world."""
        n = {"F": (0.0, -1.0), "L": (-1.0, 0.0), "R": (1.0, 0.0)}[side]
        return n if self.north else (-n[0], -n[1])


def obstacles(site, own):
    out = []
    for b in site.get("boxes", []):
        if b.get("name") == own:
            continue
        a = b.get("aabb")
        if not a or len(a) < 6 or float(a[4]) >= OBST_Z0:
            continue
        out.append((b["name"], (float(a[0]), float(a[1]), float(a[2]), float(a[3]))))
    return out


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _rect_pt(r, p):
    dx = max(r[0] - p[0], 0.0, p[0] - r[1])
    dy = max(r[2] - p[1], 0.0, p[1] - r[3])
    return math.hypot(dx, dy)


def side_strip(fr, side, t0, t1, shrink=0.02):
    """World rectangle of the floor before a side, from t0 to t1 outside it."""
    x0, x1, y0, y1 = fr.rect
    if side == "F":
        ru = (fr.u0 + shrink, fr.u1 - shrink)
        corners = [fr.to_world(u, -t) for u in ru for t in (t0, t1)]
    else:
        vv = (shrink, fr.depth - shrink)
        uu = (fr.u0 - t0, fr.u0 - t1) if side == "L" else (fr.u1 + t0, fr.u1 + t1)
        corners = [fr.to_world(u, v) for u in uu for v in vv]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), max(xs), min(ys), max(ys))


def sides_of(site, heap):
    fr = Frame(heap)
    obs = obstacles(site, heap["name"])
    lx, ly = map(float, site["size_m"])
    out = {}
    for side in ("F", "L", "R"):
        strip = side_strip(fr, side, 0.001, STRIP)
        blocked = [n for n, r in obs if _overlap(strip, r)]
        inside = strip[0] >= -1e-9 and strip[1] <= lx + 1e-9 and strip[2] >= -1e-9 and strip[3] <= ly + 1e-9
        run = 0.0
        t = 0.01
        while t <= RUNUP_MAX + 1e-9:
            st = side_strip(fr, side, 0.001, t)
            if any(_overlap(st, r) for _n, r in obs) or st[0] < 0 or st[1] > lx or st[2] < 0 or st[3] > ly:
                break
            run = t
            t += 0.01
        exposed = (not blocked) and inside and run >= RUNUP_MIN
        out[side] = dict(exposed=bool(exposed), runup_m=round(run, 3), flush_with=len(blocked),
                         why=None if exposed else ("flush" if blocked or not inside else "run-up %.2f m" % run))
    return out


# ===================================================================== compiled shape
def heap_shape(S, heap):
    m, d = S.m, S.d
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, heap["name"])
    if gid < 0:
        raise ValueError("no geom %s in the compiled site" % heap["name"])
    x0, x1, y0, y1 = map(float, heap["rect"])
    lx = np.arange(x0 - PAD, x1 + PAD + 1e-9, RAY_STEP)
    ly = np.arange(y0 - PAD, y1 + PAD + 1e-9, RAY_STEP)
    saved = m.geom_group.copy()
    try:
        m.geom_group[m.geom_group == GROUP] = 0
        m.geom_group[gid] = GROUP
        mujoco.mj_forward(m, d)
        gg = np.zeros(6, np.uint8)
        gg[GROUP] = 1
        hit = np.zeros(1, np.int32)
        v = np.array([0.0, 0.0, -1.0])
        Z = np.full((len(ly), len(lx)), -1.0)
        for i, y in enumerate(ly):
            for j, x in enumerate(lx):
                dist = mujoco.mj_ray(m, d, np.array([x + NUDGE[0], y + NUDGE[1], 2.0]), v, gg, 1, -1, hit)
                if dist >= 0 and hit[0] == gid:
                    Z[i, j] = 2.0 - dist
    finally:
        m.geom_group[:] = saved
        mujoco.mj_forward(m, d)
    from scipy import ndimage
    F = Z > 0.001
    if not F.any():
        raise ValueError("heap %s: empty footprint on the compiled model" % heap["name"])
    rows, cols = np.nonzero(F)
    north = heap["line"] == "north"
    padn = 12
    if north:
        Fp = np.concatenate([F, np.ones((padn, F.shape[1]), bool)], axis=0)
        Fp = np.pad(Fp, ((8, 0), (8, 8)), constant_values=False)
        s = (ndimage.distance_transform_edt(Fp)[8:8 + F.shape[0], 8:8 + F.shape[1]] - 0.5) * RAY_STEP
    else:
        Fp = np.concatenate([np.ones((padn, F.shape[1]), bool), F], axis=0)
        Fp = np.pad(Fp, ((0, 8), (8, 8)), constant_values=False)
        s = (ndimage.distance_transform_edt(Fp)[padn:padn + F.shape[0], 8:8 + F.shape[1]] - 0.5) * RAY_STEP
    s = np.where(F, np.maximum(s, 0.0), 0.0)
    top = float(Z.max())
    C = F & (Z >= CREST_FRAC * top)
    LX, LY = np.meshgrid(lx, ly)
    cxf, cyf = float(LX[F].mean()), float(LY[F].mean())
    QX, QY = cxf + 2.0 * (LX - cxf), cyf + 2.0 * (LY - cyf)
    qi = np.round((QY - ly[0]) / RAY_STEP).astype(int)
    qj = np.round((QX - lx[0]) / RAY_STEP).astype(int)
    okq = (qi >= 0) & (qi < F.shape[0]) & (qj >= 0) & (qj < F.shape[1])
    core = np.zeros_like(F)
    core[okq] = F[qi[okq], qj[okq]]
    fx = LX[F]
    fy = LY[F]
    shape = dict(W_meas=float(fx.max() - fx.min()), D_meas=float(fy.max() - fy.min()), top_meas=top,
                 footprint_rect=[float(fx.min()), float(fx.max()), float(fy.min()), float(fy.max())],
                 footprint_centroid=[cxf, cyf], crest_cells=int(C.sum()), n_ray=int(Z.size),
                 grid_sha256=hashlib.sha256(np.ascontiguousarray(np.round(Z, 6)).tobytes()).hexdigest())
    geo = dict(lx0=float(lx[0]), ly0=float(ly[0]), F=F, s=s, C=C, core=core, pts=np.c_[fx, fy])
    return shape, geo


def _look(geo, key, x, y):
    i = int(round((y - geo["ly0"]) / RAY_STEP))
    j = int(round((x - geo["lx0"]) / RAY_STEP))
    A = geo[key]
    if 0 <= i < A.shape[0] and 0 <= j < A.shape[1]:
        return A[i, j]
    return 0


# ===================================================================== trial
class Probe:
    """Control-tick hook: robot contact ticks with this heap, with the other heaps, with any other solid (floor and
    patches not counted); penetration into this heap; node-centroid path every `every` ticks."""

    def __init__(self, S, heap_gid, every=10):
        m = S.m
        self.robot = np.array(sorted(int(x) for x in S.ids.col_geoms))
        self.heap = heap_gid
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        self.other_heaps = np.array([i for i, n in enumerate(names) if n.startswith("heap__") and i != heap_gid], int)
        self.ground = np.array([i for i, n in enumerate(names) if n == "floor" or n.startswith("patch__")], int)
        self.every = every
        self.tick = 0
        self.path = []
        self.maxpen = 0.0
        self.contact_samples = 0
        self.reset()
        S.tick_hooks.append(self)

    def reset(self):
        self.heap_ticks = 0
        self.wall_ticks = 0      # heapcap's name: any other solid (walls, boxes, ridge, slab, crates)
        self.oheap_ticks = 0

    def __call__(self, S):
        d = S.d
        self.tick += 1
        n = d.ncon
        if n:
            G = d.contact.geom[:n]
            rob = np.isin(G, self.robot)
            r0, r1 = rob[:, 0], rob[:, 1]
            other0 = np.where(r1, G[:, 0], -1)
            other1 = np.where(r0, G[:, 1], -1)
            oth = np.concatenate([other0[r1 & ~r0], other1[r0 & ~r1]])
            if oth.size:
                hh = oth == self.heap
                if hh.any():
                    self.heap_ticks += 1
                    self.contact_samples += int(hh.sum())
                    sel = ((G[:, 0] == self.heap) & r1) | ((G[:, 1] == self.heap) & r0)
                    self.maxpen = max(self.maxpen, float(-d.contact.dist[:n][sel].min()))
                if self.other_heaps.size and np.isin(oth, self.other_heaps).any():
                    self.oheap_ticks += 1
                rest = oth[~np.isin(oth, self.ground) & (oth != self.heap) & ~np.isin(oth, self.other_heaps)]
                if rest.size:
                    self.wall_ticks += 1
        if self.tick % self.every == 0:
            p = S.d.site_xpos[S.ids.node_site][:, :2].mean(0)
            self.path.append((float(p[0]), float(p[1])))


def _prim(S, name, probe, phase, rec):
    r = hc._prim(S, name, probe, phase)
    nd = S.nodes()
    z = np.sort(nd[:, 2])
    r["z1"] = float(z[0])
    r["oheap"] = probe.oheap_ticks
    rec["qpos"].append(S.d.qpos.copy())
    rec["qvel"].append(S.d.qvel.copy())
    rec["t"].append(float(S.time))
    rec["name"].append(name)
    rec["phase"].append(phase)
    return r


def draw_start(g, fr, side, k, site, heap, sides):
    """(protocol, local start (u, v, heading_local), plan line local (point, goal) or None, n_main, sub-draws)."""
    obs = obstacles(site, heap["name"])
    lx_, ly_ = map(float, site["size_m"])
    hu = 0.5 * (fr.u1 - fr.u0)
    cu = 0.5 * (fr.u0 + fr.u1)
    cv = 0.4 * fr.depth
    runF = sides["F"]["runup_m"]
    run = sides[side]["runup_m"]
    tried = []

    def fits(st):
        wx, wy = fr.to_world(st[0], st[1])
        return bool(all(_rect_pt(r, (wx, wy)) >= CLEAR for _n, r in obs) and _rect_pt(fr.rect, (wx, wy)) >= OUTSIDE
                    and CLEAR <= wx <= lx_ - CLEAR and CLEAR <= wy <= ly_ - CLEAR)
    for sub in range(MAX_SUB):
        gs = g if sub == 0 else np.random.default_rng([Q8_TAG, int(g.integers(0, 2 ** 31)), sub])
        shrink = 1.0 - 0.6 * sub / MAX_SUB
        if side == "F":
            dmax = min(run - CLEAR, (0.72 if k % 2 == 0 else 0.62))
            dmin = min(0.37, dmax)
            dd = dmin + (dmax - dmin) * float(gs.random()) * shrink
            if k % 2 == 0:
                us = float(gs.uniform(fr.u0 - 0.12, fr.u1 + 0.12))
                start = (us, -dd, 90.0 + float(gs.uniform(-20, 20)))
                proto, n_main, line = "into", 5, None
            else:
                off = float(gs.uniform(-0.35, 0.35)) * hu
                start = (cu + off, -dd, 90.0 + float(gs.uniform(-15, 15)))
                proto, n_main = "push", 12
                line = ((cu + 0.3 * off, 0.0), 90.0)
        else:
            sgn = 1.0 if side == "L" else -1.0
            uedge = fr.u0 if side == "L" else fr.u1
            if k % 2 == 0:
                alpha = float(gs.uniform(0.0, 45.0))
                lmax = min(0.70, run - CLEAR)
                Ls = min(0.35, lmax) + (lmax - min(0.35, lmax)) * float(gs.random()) * shrink
                vhit = float(gs.uniform(0.0, fr.depth))
                vs = min(vhit - Ls * math.tan(math.radians(alpha)), fr.depth - 0.23)
                vs = max(vs, -(runF - CLEAR)) if sides["F"]["exposed"] else max(vs, 0.0)
                us = uedge - sgn * Ls
                start = (us, vs, alpha if side == "L" else 180.0 - alpha)
                proto, n_main, line = "along", 6, None
            else:
                dmax = min(run - CLEAR, 0.62)
                dmin = min(0.37, dmax)
                dd = dmin + (dmax - dmin) * float(gs.random()) * shrink
                vs = cv + float(gs.uniform(-0.35, 0.35)) * 0.5 * fr.depth
                vs = min(vs, fr.depth - 0.23)
                us = uedge - sgn * dd
                h0 = 0.0 if side == "L" else 180.0
                start = (us, vs, h0 + float(gs.uniform(-15, 15)))
                proto, n_main = "push", 12
                line = ((0.0, cv + 0.3 * (vs - cv)), h0)
        tried.append((proto, start, line, n_main))
        if fits(start):
            return proto, start, line, n_main, sub
    # 1.0.1: none of the MAX_SUB draws fits (a narrow heap beside a wall: the push's start near the heap's middle
    # is closer than CLEAR to the wall). The draws again, in order, each moved sideways (across its approach) by the
    # smallest step of 0.01 m (up to 0.30 m; + before -) that fits; heading and steering line stay as drawn. A trial
    # whose first MAX_SUB draws fit is unchanged from 1.0.0.
    for sub, (proto, start, line, n_main) in enumerate(tried):
        for k_ in range(1, 31):
            for sg in (1.0, -1.0):
                dlt = sg * 0.01 * k_
                st = (start[0] + dlt, start[1], start[2]) if side == "F" else (start[0], start[1] + dlt, start[2])
                if fits(st):
                    return proto, st, line, n_main, MAX_SUB + sub * 1000 + k_
    return None, None, None, None, MAX_SUB


def trial(site, xml, name, hidx, heap, side, k, shape_cache, states_dir=None):
    fr = Frame(heap)
    sides = sides_of(site, heap)
    code = zlib.crc32(name.encode()) & 0x7FFFFFFF
    g = np.random.default_rng([Q8_TAG, code, int(hidx), SIDE_CODE[side], int(k)])
    pre = int(g.integers(0, 3))
    roll = hc.ROLLS[(k // 2 + SIDE_CODE[side] + hidx) % 3]
    t0 = time.time()
    S = cw.new_simulator(xml)
    key = heap["name"]
    if key not in shape_cache:
        shape_cache[key] = heap_shape(S, heap)
    shape, geo = shape_cache[key]
    out = dict(tool=TOOL_VERSION, site=name, heap_index=int(hidx), heap=heap["name"], cls=heap["cls"], side=side,
               k=int(k), roll=roll, pre_rolls=pre, shape=shape, sides=sides)
    proto, st, line, n_main, sub = draw_start(g, fr, side, k, site, heap, sides)
    out.update(protocol=proto, sub_draws=sub)
    if proto is None:
        out.update(no_start=True, wall_s=round(time.time() - t0, 1))
        out["verdict"] = judge(out)
        return out
    wxy = fr.to_world(st[0], st[1])
    rd = fr.heading(st[2])
    hc.cap.start(S, wxy, rd, pre_rolls=pre)
    gid = mujoco.mj_name2id(S.m, mujoco.mjtObj.mjOBJ_GEOM, heap["name"])
    probe = Probe(S, gid)
    mon = hc.Monitor(S)
    s0 = hc._st(S)
    probe.path.append((s0["x"], s0["y"]))
    rec = dict(qpos=[S.d.qpos.copy()], qvel=[S.d.qvel.copy()], t=[float(S.time)], name=["start"], phase=["start"])
    plan = None
    if line is not None:
        p0 = fr.to_world(*line[0])
        plan = hc.cap.privileged_steer_plan(S, hc.TABLE, line_p0=p0, goal_deg=fr.heading(line[1]))
    rows = []
    stop = None
    first_cross = None
    for i in range(n_main):
        if not hc._st(S)["roll"]:
            stop = "not_rollable"
            break
        r = _prim(S, plan(i) if plan else roll, probe, "main", rec)
        rows.append(r)
        if r.get("err"):
            stop = "not_rollable"
            break
        if first_cross is None and any(_look(geo, "core", x, y) for x, y in probe.path):
            first_cross = i
    nrm = fr.normal("F")      # heapcap's turn-away: away from the line, whatever side the trial came from
    th_n = math.degrees(math.atan2(nrm[1], nrm[0]))
    turns = 0
    while hc._st(S)["roll"] and turns < hc.MAX_TURNS:
        e = (hc._st(S)["rd"] - th_n + 180.0) % 360.0 - 180.0
        if abs(e) <= 50.0:
            break
        r = _prim(S, "ccw" if e < 0 else "cw", probe, "turn", rec)
        rows.append(r)
        turns += 1
        if r.get("err"):
            break
    for _ in range(3):
        if not hc._st(S)["roll"]:
            break
        r = _prim(S, hc.REC_ROLL, probe, "away", rec)
        rows.append(r)
        if r.get("err"):
            break
    fin = hc._st(S)
    ext = float((geo["pts"] @ np.asarray(nrm)).max())
    clear_n = float(np.dot([fin["x"], fin["y"]], nrm) - ext)
    rec_ok = bool(fin["roll"] and fin["z3"] < hc.LEAN_Z and clear_n >= 0.25)
    last = rows[-1] if rows else {}
    if rec_ok:
        rec_why = None
    elif not fin["roll"]:
        rec_why = "not rollable"
    elif fin["z3"] >= hc.LEAN_Z and last.get("heap"):
        rec_why = "leaning on this heap"
    elif fin["z3"] >= hc.LEAN_Z and last.get("oheap"):
        rec_why = "leaning on another heap"
    elif fin["z3"] >= hc.LEAN_Z and last.get("wall"):
        rec_why = "leaning on another solid"
    elif clear_n < 0.25:
        rec_why = "not 0.25 m beyond the front of this heap"
    else:
        rec_why = "other"
    path = probe.path
    infp = [(x, y) for x, y in path if _look(geo, "F", x, y)]
    depth = max([0.0] + [float(_look(geo, "s", x, y)) for x, y in infp])
    crest = sum(1 for x, y in path if _look(geo, "C", x, y))
    crossed = any(_look(geo, "core", x, y) for x, y in path)
    on_mound = sum(1 for r in rows if r.get("z1", 0.0) > hc.LEAN_Z)
    rep = mon.report()
    contact = [r for r in rows if r.get("heap") and r.get("dev") is not None]
    Dm = shape["D_meas"]
    out.update(start_world=[float(wxy[0]), float(wxy[1]), rd], start_local=[float(st[0]), float(st[1]), float(st[2])],
               stop=stop, crossed_old=bool(crossed), first_cross_prim=first_cross,
               centroid_depth_m=depth, depth_share=(depth / Dm if Dm > 0 else None), crest_samples=int(crest),
               on_mound_prims=int(on_mound), n_main=sum(1 for r in rows if r["phase"] == "main"), n_prims=len(rows),
               contact_prims=sum(1 for r in rows if r.get("heap")), contact_samples=int(probe.contact_samples),
               other_heap_prims=sum(1 for r in rows if r.get("oheap")), solid_prims=sum(1 for r in rows if r.get("wall")),
               lean_any=any(r["z3"] > hc.LEAN_Z for r in rows),
               endtri_after_contact=sum(1 for r in rows if r.get("heap") and r.get("endtri")),
               endtri=sum(1 for r in rows if r.get("endtri")), notroll=any(r.get("err") for r in rows),
               turns=turns, turns_capped=turns >= hc.MAX_TURNS, rec_ok=rec_ok, rec_clear_m=clear_n, rec_why=rec_why,
               final=dict(x=fin["x"], y=fin["y"], z3=fin["z3"], face=fin["face"], roll=fin["roll"]),
               dev_contact=[r["dev"] for r in contact], maxpen_mm=1000.0 * probe.maxpen,
               nan=bool(rep["nan"] or not np.all(np.isfinite(S.d.qpos)) or not np.all(np.isfinite(S.d.qvel))),
               warnings=rep["warnings"],
               prims=[[r["name"], r["phase"], r["f0"], r["f1"], r.get("heap", 0), r.get("wall", 0), r.get("oheap", 0),
                       round(r["z3"], 5), round(r.get("z1", float("nan")), 5), bool(r.get("endtri")), round(r["x1"], 5),
                       round(r["y1"], 5)] for r in rows],
               path=[[round(x, 4), round(y, 4)] for x, y in path[::3]], sim_s=float(S.time))
    out["verdict"] = judge(out)
    if states_dir:
        os.makedirs(states_dir, exist_ok=True)
        fn = os.path.join(states_dir, "%s_%s_%s%d.npz" % (name, heap["name"], side, k))
        tmp = fn + ".part.npz"
        np.savez_compressed(tmp, qpos=np.array(rec["qpos"]), qvel=np.array(rec["qvel"]), t=np.array(rec["t"]),
                            name=np.array(rec["name"]), phase=np.array(rec["phase"]))
        os.replace(tmp, fn)
        out["state_file"] = os.path.basename(fn)
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def judge(o):
    if o.get("no_start"):
        return dict(pass_=None, why=["no start pose fits the open floor"])
    if o.get("exception"):
        return dict(pass_=False, why=["exception"])
    L = LINES[o["cls"]]
    dep, share = o["centroid_depth_m"], o["depth_share"]
    a = dep <= L["depth_m"] + 1e-12 and share is not None and share <= L["share"] + 1e-12
    d_ok = True
    if o["cls"] == "M":
        d_ok = o["on_mound_prims"] == 0 and o["crest_samples"] == 0
    ok = bool(a and d_ok and not o.get("nan"))
    why = [] if ok else [w for w, f in (("depth", not a), ("d: on mound / crest", not d_ok), ("nan", o.get("nan")))
                         if f]
    return dict(pass_=ok, a=bool(a), d=bool(d_ok), why=why)


# ===================================================================== jobs
def read_manifest(p):
    with open(p) as fh:
        return json.load(fh)["sites"]


def plan_jobs(man):
    jobs, plan = [], {}
    for e in man:
        site, xml, prov = open_site(e["package"])
        hs = site.get("heaps", {}).get("heaps", []) if isinstance(site.get("heaps"), dict) else []
        rows = []
        for hidx, h in enumerate(hs):
            sd = sides_of(site, h)
            rows.append(dict(heap=h["name"], cls=h["cls"], line=h["line"], sides=sd))
            for side in ("F", "L", "R"):
                if sd[side]["exposed"]:
                    for k in range(TRIALS_PER_SIDE):
                        jobs.append((e["name"], hidx, side, k))
        plan[e["name"]] = dict(kind=e.get("kind"), package_sha256=prov["sha256"], heaps=rows)
    return jobs, plan


def cmd_plan(a):
    load(a.cell)
    man = read_manifest(a.manifest)
    jobs, plan = plan_jobs(man)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "plan.json"), "w") as fh:
        json.dump(dict(tool=TOOL_VERSION, plan=plan, jobs=jobs), fh, indent=1)
    per = {}
    for j in jobs:
        per[j[0]] = per.get(j[0], 0) + 1
    print("q8 plan: %d sites, %d trials" % (len(plan), len(jobs)))
    return 0


def done_keys(out, ran_only=False):
    done = set()
    for f in glob.glob(os.path.join(out, "raw", "*.jsonl")):
        for line in open(f):
            try:
                o = json.loads(line)
                if ran_only and o.get("no_start"):
                    continue
                done.add((o["site"], o["heap_index"], o["side"], o["k"]))
            except Exception:  # noqa: BLE001
                pass
    return done


def cmd_work(a):
    load(a.cell)
    man = {e["name"]: e for e in read_manifest(a.manifest)}
    with open(os.path.join(a.out, "plan.json")) as fh:
        jobs = [tuple(j) for j in json.load(fh)["jobs"]]
    cdir = os.path.join(a.out, "claims_redo" if a.redo_no_start else "claims")
    os.makedirs(cdir, exist_ok=True)
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    cache, sites = {}, {}
    n = 0
    done = done_keys(a.out, ran_only=a.redo_no_start)
    for job in jobs:
        if job in done:
            continue
        cp = os.path.join(cdir, "%s_%d_%s%d" % job)
        try:
            fd = os.open(cp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(fd, ("w%s %s" % (a.worker, time.strftime("%H:%M:%S"))).encode())
            os.close(fd)
        except FileExistsError:
            continue
        name, hidx, side, k = job
        if name not in sites:
            site, xml, _prov = open_site(man[name]["package"])
            sites[name] = (site, xml)
            cache[name] = {}
        site, xml = sites[name]
        heap = site["heaps"]["heaps"][hidx]
        try:
            o = trial(site, xml, name, hidx, heap, side, k, cache[name],
                      states_dir=os.path.join(a.out, "states", name))
        except Exception as ex:  # noqa: BLE001
            import traceback
            o = dict(site=name, heap_index=hidx, heap=heap["name"], cls=heap["cls"], side=side, k=k,
                     exception=repr(ex)[:300], tb=traceback.format_exc()[-1500:])
            o["verdict"] = judge(o)
        with open(os.path.join(a.out, "raw", "%s.w%s.jsonl" % (name, a.worker)), "a") as fh:
            fh.write(json.dumps(o, sort_keys=True, default=float) + "\n")
        n += 1
        pv = o["verdict"]["pass_"]
        print("q8 %s trial %s: %s" % (name, n, "pass" if pv else ("not run" if pv is None else "FAIL")), flush=True)
    return 0


# ===================================================================== summary
def load_rows(out):
    """One row per job; a row of a trial that ran wins over a 'no start' row of the same job (the redo pass)."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(out, "raw", "*.jsonl"))):
        for line in open(f):
            o = json.loads(line)
            if "verdict" not in o:
                o["verdict"] = judge(o)
            k = (o["site"], o["heap_index"], o["side"], o["k"])
            if k not in rows or (rows[k].get("no_start") and not o.get("no_start")):
                rows[k] = o
    return [rows[k] for k in sorted(rows)]


def cmd_summary(a):
    man = read_manifest(a.manifest)
    kinds = {e["name"]: e.get("kind") for e in man}
    with open(os.path.join(a.out, "plan.json")) as fh:
        P = json.load(fh)
    jobs = [tuple(j) for j in P["jobs"]]
    rows = load_rows(a.out)
    have = {(o["site"], o["heap_index"], o["side"], o["k"]) for o in rows}
    missing = [j for j in jobs if tuple(j) not in have]
    per_heap, per_site = {}, {}
    cls_tot = {}
    for o in rows:
        v = o["verdict"]
        key = (o["site"], o["heap_index"])
        h = per_heap.setdefault(key, dict(site=o["site"], heap=o["heap"], cls=o["cls"], trials=0, fail=0,
                                          depth_max=0.0, share_max=0.0, rec_fail=0, endtri_trials=0,
                                          crossed_old=0, sides={}, failed=[], no_turn_away=[], no_start=0))
        h["trials"] += 1
        h["sides"][o["side"]] = h["sides"].get(o["side"], 0) + 1
        h.setdefault("trial_rows", []).append(dict(
            side=o["side"], k=o["k"], protocol=o.get("protocol"), pass_=v["pass_"], why=v.get("why"),
            depth_m=o.get("centroid_depth_m"), share=o.get("depth_share"), on_mound_prims=o.get("on_mound_prims"),
            crest_samples=o.get("crest_samples"), rec_ok=o.get("rec_ok"), rec_why=o.get("rec_why"),
            endtri_after_contact=o.get("endtri_after_contact"), crossed_old=o.get("crossed_old"),
            contact_prims=o.get("contact_prims"), start_world=o.get("start_world"), state_file=o.get("state_file"),
            n_prims=o.get("n_prims"), tool=o.get("tool")))
        if v["pass_"] is False:
            h["fail"] += 1
            h["failed"].append(dict(side=o["side"], k=o["k"], why=v.get("why")))
        elif v["pass_"] is None:
            h["no_start"] += 1
        if "centroid_depth_m" in o:
            h["depth_max"] = max(h["depth_max"], o["centroid_depth_m"])
            h["share_max"] = max(h["share_max"], o["depth_share"] or 0.0)
            h["D_meas"] = o["shape"]["D_meas"]
            h["top_meas"] = o["shape"]["top_meas"]
            if not o["rec_ok"]:
                h["rec_fail"] += 1
                h["no_turn_away"].append(dict(side=o["side"], k=o["k"], clear_m=round(o["rec_clear_m"], 3),
                                              turns=o["turns"], z3=round(o["final"]["z3"], 4),
                                              why=o.get("rec_why")))
            if o["endtri_after_contact"] > 0:
                h["endtri_trials"] += 1
            if o["crossed_old"]:
                h["crossed_old"] += 1
            h["on_mound_prims"] = h.get("on_mound_prims", 0) + o["on_mound_prims"]
            h["crest_samples"] = h.get("crest_samples", 0) + o["crest_samples"]
            h["contact_samples"] = h.get("contact_samples", 0) + o["contact_samples"]
        c = cls_tot.setdefault((kinds.get(o["site"]), o["cls"]), dict(trials=0, fail=0, rec_ok=0, endtri_trials=0,
                                                                     depth_max=0.0, crossed_old=0, no_start=0))
        c["trials"] += 1
        c["fail"] += 1 if v["pass_"] is False else 0
        c["no_start"] += 1 if v["pass_"] is None else 0
        if "centroid_depth_m" in o:
            c["rec_ok"] += 1 if o["rec_ok"] else 0
            if not o["rec_ok"]:
                rw = c.setdefault("turn_away_fail_why", {})
                rw[str(o.get("rec_why"))] = rw.get(str(o.get("rec_why")), 0) + 1
            c["endtri_trials"] += 1 if o["endtri_after_contact"] > 0 else 0
            c["depth_max"] = max(c["depth_max"], o["centroid_depth_m"])
            c["crossed_old"] += 1 if o["crossed_old"] else 0
    for (s, _i), h in per_heap.items():
        ps = per_site.setdefault(s, dict(kind=kinds.get(s), trials=0, fail=0, heaps=0, no_start=0))
        ps["trials"] += h["trials"]
        ps["fail"] += h["fail"]
        ps["no_start"] += h["no_start"]
        ps["heaps"] += 1
    for s in P["plan"]:
        per_site.setdefault(s, dict(kind=kinds.get(s), trials=0, fail=0, heaps=0, no_start=0))
    priv = dict(tool=TOOL_VERSION, per_heap=[per_heap[k] for k in sorted(per_heap)], per_site=per_site,
                missing=missing, plan=P["plan"],
                by_kind_and_class={"%s/%s" % k: v for k, v in sorted(cls_tot.items(), key=lambda t: str(t[0]))})
    with open(a.private + ".part", "w") as fh:
        json.dump(priv, fh, indent=1, sort_keys=True, default=float)
    os.replace(a.private + ".part", a.private)
    # accepted exceptions (--accepted FILE: {"accepted": [{"site", "line", "trials", "decision", ...}]}): failing trials of
    # a listed site and line, up to the listed number, make the site's result "pass (accepted exception)"; the trials'
    # verdicts are not changed
    acc = json.load(open(a.accepted))["accepted"] if getattr(a, "accepted", None) else []
    fails_by = {}
    for h in per_heap.values():
        for f in h["failed"]:
            why = f.get("why") or []
            line = "%s(%s)" % (h["cls"], "d" if any(str(w).startswith("d:") for w in why) and "depth" not in why
                               else "a" if why == ["depth"] else "?")
            fails_by.setdefault(h["site"], []).append(line)
    acc_rows = []
    accepted_sites = set()
    for e in acc:
        got = sum(1 for x in fails_by.get(e["site"], []) if x == e["line"])
        acc_rows.append(dict(e, trials_matched=got))
    for s_, lines in fails_by.items():
        lim = {}
        for e in acc:
            if e["site"] == s_:
                lim[e["line"]] = lim.get(e["line"], 0) + int(e["trials"])
        if all(lines.count(x) <= lim.get(x, 0) for x in set(lines)):
            accepted_sites.add(s_)
    # public: hidden sites pass / fail only; the ten together in one line; practice / cal / dev per site
    pub_sites = {}
    hidden_line = dict(trials=0, fail=0, depth_max_R=0.0, depth_max_M=0.0)
    for s, ps in sorted(per_site.items()):
        miss = sum(1 for j in missing if j[0] == s)
        ok = ps["fail"] == 0 and miss == 0 and ps["no_start"] == 0
        res = "pass" if ok else ("FAIL" if ps["fail"] else "incomplete")
        if res == "FAIL" and s in accepted_sites and miss == 0 and ps["no_start"] == 0:
            res = "pass (accepted exception)"
        if ps["kind"] == "hidden":
            pub_sites[s] = res
            hidden_line["trials"] += ps["trials"]
            hidden_line["fail"] += ps["fail"]
        else:
            pub_sites[s] = dict(result=res, trials=ps["trials"], fail=ps["fail"], missing=miss,
                                not_run=ps["no_start"])
    for h in per_heap.values():
        if kinds.get(h["site"]) == "hidden":
            k = "depth_max_" + h["cls"]
            hidden_line[k] = max(hidden_line[k], h["depth_max"])
    ratios = {}
    for (kind, cl), c in sorted(cls_tot.items(), key=lambda t: str(t[0])):
        grp = "hidden" if kind == "hidden" else "public"
        r = ratios.setdefault("%s/%s" % (grp, cl), dict(trials=0, turn_away_ok=0, endtri_after_contact_trials=0,
                                                        crossed_old=0))
        r["trials"] += c["trials"]
        r["turn_away_ok"] += c["rec_ok"]
        r["endtri_after_contact_trials"] += c["endtri_trials"]
        r["crossed_old"] += c["crossed_old"]
        for kk, vv in c.get("turn_away_fail_why", {}).items():
            r.setdefault("turn_away_fail_why", {})
            r["turn_away_fail_why"][kk] = r["turn_away_fail_why"].get(kk, 0) + vv
    pub_heaps = []
    for h in sorted(per_heap.values(), key=lambda t: (t["site"], t["heap"])):
        if kinds.get(h["site"]) == "hidden":
            continue
        pub_heaps.append(dict(site=h["site"], heap=h["heap"], cls=h["cls"], trials=h["trials"], fail=h["fail"],
                              sides=h["sides"], depth_max_m=round(h["depth_max"], 4),
                              share_max=round(h["share_max"], 4), D_meas_m=round(h.get("D_meas", 0.0), 4),
                              turn_away_fail=h["rec_fail"], endtri_trials=h["endtri_trials"],
                              crossed_old=h["crossed_old"], no_turn_away=h["no_turn_away"], not_run=h["no_start"],
                              failed=h["failed"]))
    pub = dict(tool="q8_heap_trials %s" % TOOL_VERSION, sites=pub_sites,
               hidden_ten_together=dict(trials=hidden_line["trials"], failed_trials=hidden_line["fail"],
                                        deepest_entry_R_m=round(hidden_line["depth_max_R"], 4),
                                        deepest_entry_M_m=round(hidden_line["depth_max_M"], 4)),
               ratios_by_class=ratios, public_heaps=pub_heaps, missing_trials=len(missing), accepted_exceptions=acc_rows,
               all_pass=all(((v if isinstance(v, str) else v["result"]) in ("pass", "pass (accepted exception)"))
                            for v in pub_sites.values()),
               all_pass_without_exceptions=all((v if isinstance(v, str) else v["result"]) == "pass"
                                               for v in pub_sites.values()))
    with open(a.public + ".part", "w") as fh:
        json.dump(pub, fh, indent=1, sort_keys=True, default=float)
    os.replace(a.public + ".part", a.public)
    print("q8 summary: %s" % ("pass" if pub["all_pass"] else "FAIL"))
    for s, v in sorted(pub_sites.items()):
        print("  %s %s" % (s, v if isinstance(v, str) else v["result"]))
    return 0 if pub["all_pass"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["plan", "work", "summary"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default=os.environ.get("HEAPCAP_CELL"))
    ap.add_argument("--worker", default="0")
    ap.add_argument("--public")
    ap.add_argument("--private")
    ap.add_argument("--redo-no-start", action="store_true", help="work: run again the jobs whose rows are all 'no start'")
    ap.add_argument("--accepted", help="summary: a JSON file of accepted exceptions (site, line, trials, decision)")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        return cmd_plan(a)
    if a.cmd == "work":
        return cmd_work(a)
    return cmd_summary(a)


if __name__ == "__main__":
    sys.exit(main())
