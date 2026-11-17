"""The cleaning job's process models (the cell manual states every rule and number here; the program is told none of
the states): the grease film on the dining table, the sponge's soap, water and grease load, the crumbs.

Grease (a grid of 20 mm cells over the stain and a margin round it; grease per cell g in [0, 1]; loads in units of a
25 mm cell's worth):
  a sponge pressed flat on the table (its face within 25 deg of the table, a normal force N above 2 N) and sliding
  takes grease off every cell under its face: per tick g -> g * exp(-K_WIPE * min(N - 2, 25) * ds * f), ds the
  distance it slid, f = (scouring pad down 1.0 | foam down 0.35) x (soap on the sponge 1.0 | wet, no soap 0.30 |
  dry 0.12) x (1 - load / LOAD_MAX); what it takes off loads the sponge; a loaded sponge puts grease back on the
  cells it passes (DEPOSIT x (load / LOAD_MAX)^3 per sponge length slid), so a saturated sponge smears. Soap is used
  up as grease comes off (SOAP_PER_UNIT doses per cell's worth).
Sponge: held in the running water (its centre within 45 mm of the stream's axis, between the basin's bottom and the
  outlet), it sheds its load (time constant 0.6 s) and its soap (2.5 s) and is wet. A stroke of the soap pump made by
  pressing the sponge down on the pump's head (12 mm) puts a dose of soap into it (at most 2).
Crumbs (kinematic: drawn-only bodies the cell moves): they rest on the highest surface below them. The broom's
  bristles, flat on the floor (their bottom within 20 mm of it, the head within 35 deg of level) and moving, push the
  floor crumbs under them out over their leading edge (faster than 1.0 m/s they scatter instead); the sponge does
  the same with the crumbs on the surface it is pressed on, and a crumb pushed over the surface's edge flies on 10-20 cm
  the way it was pushed and falls. A
  crumb stops at whatever stands on the floor (the walls, the toe kick, the bins, legs). The dustpan resting flat
  on the floor (its lip within 8 mm of it, the pan within 12 deg of level) takes in the crumbs pushed over its lip;
  they stay in the pan as it moves unless it is tipped past 55 deg, when they pour out over its lowest edge (the
  lip, or a side past 65 deg; over the back wall only upside down) and fall. A crumb that falls into an open bin
  stays there; onto a closed lid it stays on the lid.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import objects as O
from . import room as RM

TICK = 0.01
K_WIPE = 0.8
N_MIN, N_MAX = 2.0, 25.0
FLAT_COS = math.cos(math.radians(25.0))       # the sponge's face on the surface it wipes
BROOM_FLAT_COS = math.cos(math.radians(35.0))  # the broom's head (its bristles flex)
FACE = {"pad": 1.0, "foam": 0.35}
SOAP_F, WET_F, DRY_F = 1.0, 0.30, 0.12
LOAD_MAX = 32.0                     # cells' worth of grease a sponge can take up
DEPOSIT = 0.25                      # grease per cell a saturated sponge leaves per sponge length slid
SOAP_PER_UNIT = 1.0 / 38.0          # soap doses used per cell's worth of grease taken off
SOAP_MAX = 2.0
RINSE_TAU, SOAP_RINSE_TAU = 0.6, 2.5
SCATTER_SPEED = 1.0
EDGE_FLY = (0.10, 0.20)          # m: a crumb pushed over a surface's edge flies on this far the way it was pushed
BROOM_ZTOL = 0.020                 # m: the bristles' bottom within this of the floor (they flex) sweeps
PAN_ZTOL, PAN_LEVEL_COS = 0.008, math.cos(math.radians(12.0))
PAN_SPILL_COS, PAN_SIDE_COS = math.cos(math.radians(55.0)), math.cos(math.radians(65.0))
LID_OPEN = math.radians(35.0)       # a lid opened further than this lets a falling crumb into its bin

FREE, PAN, BIN_GENERAL, BIN_RECYCLING, SINK, LID = 0, 1, 2, 3, 4, 5
STATE_NAMES = {FREE: "free", PAN: "pan", BIN_GENERAL: "bin_general", BIN_RECYCLING: "bin_recycling", SINK: "sink",
               LID: "lid"}
FOAM_RGB = np.array([0.96, 0.80, 0.22])
DIRTY_RGB = np.array([0.42, 0.34, 0.18])


class Sponge:
    def __init__(self, m):
        self.m = m
        self.soap = 0.0
        self.wet = 1.0
        self.load = 0.0
        self.peak_load = 0.0
        self.doses = 0
        self.rinse_s = 0.0
        G = mujoco.mjtObj.mjOBJ_GEOM
        self.foam = mujoco.mj_name2id(m, G, "sponge_foam")
        self.suds = mujoco.mj_name2id(m, G, "sponge_suds")
        self.col = mujoco.mj_name2id(m, G, "sponge_col")

    def sync(self):
        a = min(1.0, self.load / LOAD_MAX) ** 0.8
        self.m.geom_rgba[self.foam, :3] = FOAM_RGB * (1 - a) + DIRTY_RGB * a
        self.m.geom_rgba[self.suds, 3] = 0.9 * min(1.0, self.soap)

    def record(self):
        return {"soap": round(self.soap, 3), "wet": round(self.wet, 3), "load": round(self.load, 3),
                "load_frac": round(self.load / LOAD_MAX, 3), "peak_load": round(self.peak_load, 3),
                "doses": self.doses, "rinse_s": round(self.rinse_s, 2)}


class Grease:
    def __init__(self, m, meta):
        gr = meta["grease"]
        self.m = m
        self.cell = float(gr["cell"])
        self.xy = np.array([(c[2], c[3]) for c in gr["cells"]], float)
        self.g0 = np.array(gr["g0"], float)
        self.g = self.g0.copy()
        self.geom0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "grease0")
        self.total0 = float(self.g0.sum()) * (self.cell / 0.025) ** 2
        self._shown = np.full(len(self.g), -1.0)
        self.removed = 0.0
        self.deposited = 0.0
        self.unit = (self.cell / 0.025) ** 2          # a cell's grease in 25 mm cells' worth (the load's unit)

    def residual(self):
        return float(self.g.sum()) * self.unit

    def removed_fraction(self):
        return 1.0 - self.residual() / max(self.total0, 1e-9)

    def under(self, c, R, half):
        """Cells whose centres lie under a face (centre c, rotation R whose x, y axes span the face, half sizes)."""
        d = self.xy - c[:2][None, :]
        u = d @ R[:2, 0]
        w = d @ R[:2, 1]
        return (np.abs(u) <= half[0]) & (np.abs(w) <= half[1])

    def wipe(self, sponge: Sponge, c, R, half, N, ds, face):
        sel = self.under(c, R, half)
        if not sel.any() or ds <= 0:
            return 0.0
        f = FACE[face] * (SOAP_F if sponge.soap > 0.02 else (WET_F if sponge.wet > 0.5 else DRY_F))
        cap = max(0.0, 1.0 - sponge.load / LOAD_MAX)
        k = K_WIPE * min(max(N - N_MIN, 0.0), N_MAX) * ds * f * cap
        g = self.g[sel]
        taken = g * (1.0 - math.exp(-k))
        g = g - taken
        lf = sponge.load / LOAD_MAX
        dep = DEPOSIT * lf ** 3 * (ds / O.SPONGE["size"][0])
        put = np.minimum(np.maximum(0.0, dep * (1.0 - g)), 1.0 - g)
        g = g + put
        self.g[sel] = g
        t_sum, p_sum = float(taken.sum()) * self.unit, float(put.sum()) * self.unit
        sponge.load = max(0.0, sponge.load + t_sum - p_sum)
        sponge.peak_load = max(sponge.peak_load, sponge.load)
        if sponge.soap > 0:
            sponge.soap = max(0.0, sponge.soap - t_sum * SOAP_PER_UNIT)
        self.removed += t_sum
        self.deposited += p_sum
        return t_sum

    def sync(self, force=False):
        a = 0.92 * np.power(np.clip(self.g, 0.0, 1.0), 0.55)
        a[self.g < 0.01] = 0.0
        ch = np.abs(a - self._shown) > 0.01 if not force else np.ones(len(a), bool)
        idx = np.nonzero(ch)[0]
        if len(idx):
            self.m.geom_rgba[self.geom0 + idx, 3] = a[idx]
            self._shown[idx] = a[idx]


class Crumbs:
    def __init__(self, m, d, meta, layout, mech, rng):
        self.m, self.d = m, d
        n = meta["crumbs"]
        self.n = n
        self.P = np.array(layout["crumbs"], float).reshape(n, 3)
        self.r = np.array(layout["crumb_r"], float)
        self.state = np.zeros(n, int)
        self.local = np.zeros((n, 3))
        self.pushed = np.zeros(n, bool)        # moved by a tool's face this tick (only those can enter the pan)
        self.mocap = np.array([int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"crumb{i}")])
                               for i in range(n)])
        self.on = np.array(["floor" if p[2] < 0.05 else "dining_top" for p in self.P], dtype=object)
        self.mech = mech
        self.rng = rng
        self.bins = dict(meta["bins"])
        from .layout import floor_obstacles
        self.obstacles = np.array(floor_obstacles(), float)
        self.surfaces = RM.surfaces()
        self.events = {"captured": 0, "spilled": 0, "scattered": 0, "binned_general": 0, "binned_recycling": 0,
                       "to_floor": 0}
        self.pan_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "dustpan")
        self.sync()

    # -------------------------------------------------------------------------------------------- geometry
    def _blocked(self, x, y, r):
        if x < r + 0.005 or y < r + 0.005 or x > RM.ROOM_X - r - 0.005 or y > RM.ROOM_Y - r - 0.005:
            return True
        o = self.obstacles
        return bool(np.any((o[:, 0] - r < x) & (x < o[:, 2] + r) & (o[:, 1] - r < y) & (y < o[:, 3] + r)))

    def pan_frame(self):
        b = self.pan_body
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def pan_down(self):
        p, R = self.pan_frame()
        return p[2] < PAN_ZTOL and R[2, 2] > PAN_LEVEL_COS

    def in_pan_xy(self, x, y, p, R, margin=0.0):
        d = np.array([x - p[0], y - p[1], 0.0])
        lx = float(d @ R[:, 0])
        ly = float(d @ R[:, 1]) / max(1e-6, math.hypot(R[0, 1], R[1, 1]))
        W, D = O.DUSTPAN["width"], O.DUSTPAN["depth"]
        return abs(lx) < W / 2 - 0.005 - margin and 0.006 + margin < ly < D - 0.008 - margin

    def bin_slot_at(self, x, y):
        B = RM.BIN
        for slot, yc in enumerate(RM.BIN_SLOTS_Y):
            if B["x0"] + B["wall_t"] < x < B["x1"] - B["wall_t"] and abs(y - yc) < B["half_y"] - B["wall_t"]:
                return slot
        return None

    # -------------------------------------------------------------------------------------------- falling
    def drop(self, i, x, y, z):
        """Crumb i falls from (x, y, z) onto the highest support below (or into a bin, the sink or the pan)."""
        r = self.r[i]
        slot = self.bin_slot_at(x, y)
        B = RM.BIN
        if slot is not None and z > B["rim_z"] - 0.02:
            if self.mech.lid_angle(slot) > LID_OPEN:
                kind = self.bins[slot]
                self.state[i] = BIN_GENERAL if kind == "general" else BIN_RECYCLING
                self.events["binned_" + kind] += 1
                yc = RM.BIN_SLOTS_Y[slot]
                self.P[i] = (float(np.clip(x, B["x0"] + 0.03, B["x1"] - 0.03)),
                             float(np.clip(y, yc - B["half_y"] + 0.03, yc + B["half_y"] - 0.03)), 0.045 + r * 0.55)
            else:
                self.state[i] = LID
                self.P[i] = (x, y, B["rim_z"] + B["lid_t"] + 0.003 + r * 0.55)
            return
        # the dustpan, held under the falling crumb
        p, R = self.pan_frame()
        if R[2, 2] > math.cos(math.radians(35.0)) and p[2] < z and self.in_pan_xy(x, y, p, R):
            self.state[i] = PAN
            d = np.array([x, y, z]) - p
            loc = R.T @ d
            loc[2] = O.DUSTPAN["t"] + r * 0.55
            self.local[i] = loc
            self.events["captured"] += 1
            return
        S = RM.SINK
        if S["x0"] < x < S["x1"] and S["y0"] < y < S["y1"] and z > S["bottom_z"]:
            self.state[i] = SINK
            self.P[i] = (x, y, S["bottom_z"] + r * 0.55)
            return
        best, on = 0.0, "floor"
        for name, ((x0, y0, x1, y1), h) in self.surfaces.items():
            if name == "sink":
                continue
            if x0 < x < x1 and y0 < y < y1 and h <= z + 1e-4 and h > best:
                best, on = h, name
        if on == "floor":
            # land a little scattered, never inside something that stands on the floor
            for _ in range(8):
                xx, yy = x + self.rng.normal(0, 0.012), y + self.rng.normal(0, 0.012)
                if not self._blocked(xx, yy, r):
                    x, y = xx, yy
                    break
            self.events["to_floor"] += 1
        self.state[i] = FREE
        self.on[i] = on
        self.P[i] = (x, y, best + r * 0.55)

    # -------------------------------------------------------------------------------------------- pushing
    def plow(self, surface, c, R, half, v, zsurf):
        """A tool face (centre c, axes R[:, 0] (long) and R[:, 1] (across) in the surface's plane, half sizes) moving
        at v (world, m/s) on `surface` (the crumbs resting on it)."""
        idx = np.nonzero((self.state == FREE) & (self.on == surface))[0]
        if not len(idx):
            return
        sp = float(math.hypot(v[0], v[1]))
        if sp < 0.004:
            return
        a1 = R[:2, 0] / max(1e-9, np.linalg.norm(R[:2, 0]))
        a2 = np.array([-a1[1], a1[0]])
        d = self.P[idx, :2] - c[None, :2]
        u, w = d @ a1, d @ a2
        hu, hw = half
        inside = (np.abs(u) <= hu + self.r[idx]) & (np.abs(w) <= hw + self.r[idx])
        if not inside.any():
            return
        vu, vw = float(v[:2] @ a1), float(v[:2] @ a2)
        for k in np.nonzero(inside)[0]:
            i = idx[k]
            ri = self.r[i]
            if sp > SCATTER_SPEED:
                ang = math.atan2(v[1], v[0]) + self.rng.uniform(-0.6, 0.6)
                dist = self.rng.uniform(0.10, 0.40)
                nx, ny = self.P[i, 0] + dist * math.cos(ang), self.P[i, 1] + dist * math.sin(ang)
                self.events["scattered"] += 1
            else:
                uu, ww = u[k], w[k]
                if abs(vw) >= 0.25 * abs(vu):
                    ww = math.copysign(hw + ri + 0.003, vw)
                    uu = uu + self.rng.normal(0, 0.003)
                else:
                    uu = math.copysign(hu + ri + 0.003, vu)
                nx, ny = (c[0] + uu * a1[0] + ww * a2[0], c[1] + uu * a1[1] + ww * a2[1])
            self._move(i, nx, ny, surface, zsurf)

    def _move(self, i, x, y, surface, zsurf):
        r = self.r[i]
        if surface == "floor":
            if self._blocked(x, y, r):
                return
            self.P[i, :2] = (x, y)
            self.pushed[i] = True
            return
        (x0, y0, x1, y1), h = self.surfaces[surface]
        if x0 < x < x1 and y0 < y < y1:
            self.P[i, :2] = (x, y)
            return
        # pushed over the edge: it flies on a little the way it was pushed, then falls
        dx, dy = x - float(self.P[i, 0]), y - float(self.P[i, 1])
        n = math.hypot(dx, dy)
        if n > 1e-6:
            fly = float(self.rng.uniform(*EDGE_FLY))
            x, y = x + dx / n * fly, y + dy / n * fly
        self.drop(i, x, y, h)

    def capture(self):
        """The dustpan down on the floor takes in the floor crumbs pushed into its pan this tick (a pan set down on
        crumbs covers them; it does not scoop them up)."""
        pushed = self.pushed.copy()
        self.pushed[:] = False
        if not self.pan_down():
            return
        p, R = self.pan_frame()
        for i in np.nonzero((self.state == FREE) & (self.on == "floor") & pushed)[0]:
            if self.in_pan_xy(self.P[i, 0], self.P[i, 1], p, R):
                self.state[i] = PAN
                loc = R.T @ (self.P[i] - p)
                loc[2] = O.DUSTPAN["t"] + self.r[i] * 0.55
                self.local[i] = loc
                self.events["captured"] += 1

    def carry(self):
        """Crumbs in the pan ride with it; tipped past 55 deg they pour out over its lowest edge."""
        idx = np.nonzero(self.state == PAN)[0]
        if not len(idx):
            return
        p, R = self.pan_frame()
        W, D, H = O.DUSTPAN["width"], O.DUSTPAN["depth"], O.DUSTPAN["wall_h"]
        if R[2, 2] < PAN_SPILL_COS:
            edges = {"lip": np.array([0.0, 0.0, 0.0]), "back": np.array([0.0, D, H]),
                     "west": np.array([-W / 2, D / 2, H * 0.4]), "east": np.array([W / 2, D / 2, H * 0.4])}
            low = min(edges, key=lambda k: float((p + R @ edges[k])[2]))
            pour = (low == "lip") or (low in ("west", "east") and R[2, 2] < PAN_SIDE_COS) or (R[2, 2] < 0.0)
            if pour:
                for i in idx:
                    loc = self.local[i].copy()
                    if low == "lip":
                        loc[1] = 0.0
                    elif low == "back":
                        loc[1] = D
                    else:
                        loc[0] = math.copysign(W / 2, edges[low][0])
                    wp = p + R @ loc
                    self.events["spilled"] += 1
                    self.drop(i, float(wp[0]), float(wp[1]), float(wp[2]))
                idx = np.nonzero(self.state == PAN)[0]
        if len(idx):
            self.P[idx] = p[None, :] + self.local[idx] @ R.T

    # -------------------------------------------------------------------------------------------- output
    def sync(self):
        self.d.mocap_pos[self.mocap] = self.P

    def counts(self):
        return {STATE_NAMES[k]: int((self.state == k).sum()) for k in STATE_NAMES}


class Process:
    """Per tick, after the physics: wiping, rinsing, soaping, sweeping; then the drawn state (grease, sponge, crumbs)."""

    def __init__(self, m, d, meta, layout, grips, mech, rng):
        self.m, self.d = m, d
        self.grips, self.mech = grips, mech
        self.sponge = Sponge(m)
        self.grease = Grease(m, meta)
        self.crumbs = Crumbs(m, d, meta, layout, mech, rng)
        G, B = mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY
        self.sponge_body = mujoco.mj_name2id(m, B, "sponge")
        self.sponge_geom = mujoco.mj_name2id(m, G, "sponge_col")
        self.surface_geoms = {mujoco.mj_name2id(m, G, "dining_top"): "dining_top",
                              mujoco.mj_name2id(m, G, "coffee_top"): "coffee_top"}
        for k in ("s", "n", "w"):
            self.surface_geoms[mujoco.mj_name2id(m, G, f"counter_{k}")] = "counter"
        # the other surfaces a crumb can come to rest on (one pushed off the table's end lands on a chair's seat): the
        # sponge pressed on them pushes their crumbs too
        for geom, surf in (("chair_west_seat", "chair_west"), ("chair_east_seat", "chair_east"),
                           ("dish_tray", "dish_tray"), ("shelf_top", "shelf_top")):
            gid = mujoco.mj_name2id(m, G, geom)
            if gid >= 0:
                self.surface_geoms[gid] = surf
        self.head_body = mujoco.mj_name2id(m, B, "broom_head")
        self.bristle_geom = mujoco.mj_name2id(m, G, "broom_bristle_col")
        self.pump_body = mujoco.mj_name2id(m, B, "soap_pump")
        self.pump_geom = mujoco.mj_name2id(m, G, "soap_pump_col")
        self._prev_sponge = None
        self._prev_bristle = None
        self.wipe_s = 0.0
        self.sweep_s = 0.0
        self.doses_missed = 0
        self.grease.sync(force=True)
        self.sponge.sync()

    def sponge_on_pump(self):
        d = self.d
        for i in range(d.ncon):
            c = d.contact[i]
            if {int(c.geom1), int(c.geom2)} == {self.sponge_geom, self.pump_geom}:
                return True
        return False

    def sponge_contact(self):
        """(surface name, total normal force) of the sponge's contact with a cleanable surface, else (None, 0)."""
        m, d = self.m, self.d
        f6 = np.zeros(6)
        tot, name = {}, None
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == self.sponge_geom and g2 in self.surface_geoms:
                other = g2
            elif g2 == self.sponge_geom and g1 in self.surface_geoms:
                other = g1
            else:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            s = self.surface_geoms[other]
            tot[s] = tot.get(s, 0.0) + abs(float(f6[0]))
        if not tot:
            return None, 0.0
        name = max(tot, key=tot.get)
        return name, tot[name]

    def tick(self, stroke: bool):
        d = self.d
        sp = self.sponge
        # ---- the sponge on a surface
        b = self.sponge_body
        c, R = d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()
        if self._prev_sponge is None:
            self._prev_sponge = c.copy()
        v = (c - self._prev_sponge) / TICK
        ds = float(math.hypot(v[0], v[1])) * TICK
        self._prev_sponge = c.copy()
        surf, N = self.sponge_contact()
        if surf is not None and abs(R[2, 2]) > FLAT_COS:
            lx, ly, _ = O.SPONGE["size"]
            half = (lx / 2, ly / 2)
            face = "pad" if R[2, 2] > 0 else "foam"
            if surf == "dining_top" and N > N_MIN:
                self.grease.wipe(sp, c, R, half, N, ds, face)
                self.wipe_s += TICK
            if N > 0.5:
                zs = RM.surfaces()[surf][1] if surf in RM.surfaces() else c[2]
                self.crumbs.plow(surf, c, R, half, v, zs)
        # ---- rinsing under the running water
        if self.mech.water_on():
            ox, oy, oz = RM.FAUCET["outlet"]
            if math.hypot(c[0] - ox, c[1] - oy) < RM.STREAM_R and RM.SINK["bottom_z"] < c[2] < oz:
                sp.load *= math.exp(-TICK / RINSE_TAU)
                sp.soap *= math.exp(-TICK / SOAP_RINSE_TAU)
                sp.wet = 1.0
                sp.rinse_s += TICK
        # ---- a pump stroke: a dose into the sponge pressing on the pump's head, else into nothing
        if stroke:
            if self.sponge_on_pump():
                sp.soap = min(SOAP_MAX, sp.soap + 1.0)
                sp.doses += 1
            else:
                self.doses_missed += 1
        # ---- the broom's bristles on the floor
        hb = self.head_body
        Rh = d.xmat[hb].reshape(3, 3)
        bottom = d.xpos[hb] + Rh @ np.array([0.0, 0.0, -(O.BROOM["head"][2] + O.BROOM["bristle"][2])])
        if self._prev_bristle is None:
            self._prev_bristle = bottom.copy()
        vb = (bottom - self._prev_bristle) / TICK
        self._prev_bristle = bottom.copy()
        if bottom[2] < BROOM_ZTOL and Rh[2, 2] > BROOM_FLAT_COS:
            bx, by, _ = O.BROOM["bristle"]
            self.crumbs.plow("floor", bottom, Rh, (bx / 2, by / 2), vb, 0.0)
            self.sweep_s += TICK
        # ---- the dustpan: take in, carry, pour
        self.crumbs.capture()
        self.crumbs.carry()
        # ---- what the cameras see
        self.grease.sync()
        sp.sync()
        self.crumbs.sync()

    def record(self):
        return {"grease_total0": round(self.grease.total0, 3), "grease_residual": round(self.grease.residual(), 3),
                "grease_removed_fraction": round(self.grease.removed_fraction(), 4),
                "grease_taken": round(self.grease.removed, 3), "grease_smeared": round(self.grease.deposited, 3),
                "sponge": self.sponge.record(), "crumbs": self.crumbs.counts(), "crumb_events": dict(self.crumbs.events),
                "wipe_s": round(self.wipe_s, 2), "sweep_s": round(self.sweep_s, 2),
                "soap_strokes": self.mech.strokes, "soap_doses_missed": self.doses_missed,
                "water_s": round(self.mech.water_s, 2)}
