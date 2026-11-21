"""Stowage unit of the station scene v0.3: one holding screen in the middle and a pouch pad
on each side, all three faces in the same plane and fed by the unit's own screen blower.

The module holds the screen part with its blower network and air, and the horizontal unit layout: two pouch pads
(flat suction faces, no side walls) on the screen's blower network (one FanNetwork with three faces); the pad
fiducial plate position is a parameter (candidates on the robot's overhang side, chosen by the firmware
package by visibility, site_spec); the unit's approach zone.

The unit (local frame: a = b x n along the rack face, b = 'up' projected into the face plane, n into the cabin; origin
= the rack-face point under the screen centre):
  holding screen  opening 0.36 x 0.36 m (0.1296 m^2), frame 0.42 x 0.42 m, face plane 0.030 m proud      collision
                  of the rack face; perforated plate (collision) + felt + plenum (display)
  pouch pads      two, centres at a = -/+0.35 m; opening 0.20 (a) x 0.26 (b) m (0.052 m^2), frame 0.26 x     collision
                  0.32 m, face in the screen's plane; plate (collision) + felt + plenum (display)
  screen blower   behind the rack face (display housing); exhaust grilles in the screen's two vertical frame bars
                  (half the flow each); one FanNetwork with the three faces; always on, part of no cabin air unit,
                  no off request
  fiducial plates 0.10 x 0.06 m, 12 mm off the frames: above and below the screen ('+b', '-b'); one per pouch
                  pad at a selectable side ('+b' above, '-b' below, 'outer' outboard)                     display
  id plate        on the screen's -b frame bar                                                            display
Blower: the station fan curve affinity-scaled by fan_scale 0.30 (shut-off 67 Pa), K_face 8, r_face 3200 Pa per m/s on
every face, the duct sized so that the open faces pass the same face speed as the screen's 4.0 cfm (0.0146 m/s): 7.2
cfm in all.  All unit numbers are design values of this fictitious station [G]; spec_values
iva.holding_screen.design / .datasheet carry the screen's (tests compare them).
numpy + spacekit only.
"""
from __future__ import annotations

import numpy as np

from .util import fmt as _v
from .util import unit_axes

CFM = 4.71947e-4
SCREEN_OPENING = (0.36, 0.36)       # a x b [m]
SCREEN_FRAME = (0.42, 0.42)
FRAME_DEPTH = 0.030                 # face plane proud of the rack face [m]
FACE_THICK = 0.003
FELT_THICK = 0.008
PLENUM_DEPTH = 0.06
PAD_OPENING = (0.20, 0.26)          # a x b [m] (the pouch pocket's mouth)
PAD_FRAME = (0.26, 0.32)
PAD_CENTRE_A = 0.35                 # pad centres at a = -/+ this from the screen centre [m]
FID_SIZE = (0.10, 0.06)
FID_GAP = 0.012                     # plate edge off the frame [m]
FRICTION = 0.7                      # item on a perforated plate (as on the inlets' debris screen) [G]
BLOWER = dict(screen_cfm=4.0, fan_scale=0.30, K_face=8.0, r_face=3200.0, tau_fan=1.0)
PAD_FID_SIDES = ("+b", "-b", "outer")
PAD_FID_DEFAULT = "+b"              # the overhang side of a pouch held handle-horizontal, palm up
SITING_PAD_SIDES = ("+b", "-b")     # the pad-plate sides the siting footprint covers ('outer' widens the unit by 0.14 m
                                    # and fails the dense check in L1 and H2: choosing it needs a new siting)
APPROACH = dict(grow_m=0.20, depth_m=1.0)   # unit outline grown by 0.20 m, 1.0 m along the normal [G]
HOUSING_HALF = (0.035, 0.08, 0.025)
EXHAUST_HALF_AB = (0.011, 0.06)
COLORS = dict(frame="0.72 0.73 0.75 1", face="0.30 0.31 0.33 1", hole="0.08 0.08 0.09 1", felt="0.55 0.56 0.50 1",
              plenum="0.04 0.04 0.045 1", housing="0.62 0.63 0.66 1", grille="0.15 0.15 0.16 1", plate="0.95 0.95 0.92 1",
              pad_face="0.34 0.35 0.37 1")
CONTACT = 'solref="0.004 1" solimp="0.95 0.99 0.001" condim="4" priority="1"'
FACE_NAMES = ("hs", "pp0", "pp1")   # holding screen, pouch pad at -a, pouch pad at +a


def face_speed(blower: dict | None = None) -> float:
    """Open-face speed of every face of the unit [m/s] (the screen's design flow over its opening)."""
    B = dict(BLOWER, **(blower or {}))
    return B["screen_cfm"] * CFM / (SCREEN_OPENING[0] * SCREEN_OPENING[1])


class Face:
    """One suction face of a unit: centre (face plane), axes, opening half sizes along (a, b)."""

    def __init__(self, name: str, kind: str, centre, n, a, b, half):
        self.name, self.kind = name, kind
        self.c = np.asarray(centre, float)
        self.n, self.a, self.b = (np.asarray(v, float) for v in (n, a, b))
        self.half = (float(half[0]), float(half[1]))

    @property
    def area(self) -> float:
        return 4.0 * self.half[0] * self.half[1]

    def local(self, x):
        """(gap above the face plane, a, b) of world points x (N, 3)."""
        d = np.atleast_2d(np.asarray(x, float)) - self.c
        return d @ self.n, d @ self.a, d @ self.b

    def in_opening(self, x, margin: float = 0.0):
        _, s1, s2 = self.local(x)
        return (np.abs(s1) <= self.half[0] - margin) & (np.abs(s2) <= self.half[1] - margin)

    face_local = local          # the earlier name

    def as_dict(self) -> dict:
        return dict(name=self.name, kind=self.kind, centre=self.c.tolist(), normal=self.n.tolist(), a=self.a.tolist(),
                    b=self.b.tolist(), half=list(self.half), area_m2=self.area)


# ----------------------------------------------------------------------------------------------- network and air
class UnitNetwork:
    """The unit's blower: one spacekit FanNetwork with the three faces (Inlets named '<unit>_hs', '<unit>_pp0',
    '<unit>_pp1') + the exhaust sources (Inlets of kind 'supply', each carrying its share of the network's flow,
    written by ``sync``)."""

    def __init__(self, name: str, faces, exhaust_centres=(), exhaust_axes=None, blower: dict | None = None):
        from spacekit.suction import FanNetwork, Inlet
        B = dict(BLOWER, **(blower or {}))
        self.blower = B
        v = face_speed(B)
        self.inlets = [Inlet(f.name, f.c, f.n, f.a, f.half, flow=v * f.area) for f in faces]
        self.by_face = {f.name: i for f, i in zip(faces, self.inlets)}
        self.net = FanNetwork(self.inlets, build=dict(K_face=B["K_face"], r_face=B["r_face"]), Q_design=sum(i.flow0 for i in self.inlets),
                              tau_fan=B["tau_fan"], fan_scale=B["fan_scale"], name=name)
        self.exhausts = []
        for k, c in enumerate(exhaust_centres):
            n, t1 = exhaust_axes
            self.exhausts.append(Inlet(f"{name}_exh{k}", c, n, t1, (EXHAUST_HALF_AB[1], EXHAUST_HALF_AB[0]), kind="supply",
                                       flow=self.net.Q / len(exhaust_centres)))
        self.sync()

    @property
    def inlet(self):
        """The holding screen's face."""
        return self.inlets[0]

    def sync(self):
        for e in self.exhausts:
            e.flow = float(self.net.Q) / len(self.exhausts)

    def set_on(self, on: bool, t=None):
        """Controls of the maintainers' runs only (the unit has no off request)."""
        self.net.set_fan(on, t)

    def advance(self, dt: float):
        self.net.advance(dt)

    @property
    def Pc(self) -> float:
        return float(self.net.Pc)


class UnitFlow:
    """Cabin air with the units' own air: base(x) + every unit's three face sinks + its exhaust source (callable on
    (N, 3) points, like ``StationVentilation``).  ``units``: list of UnitNetwork."""

    def __init__(self, base, units):
        self.base = base
        self.units = list(units)
        self.calls = 0

    def units_only(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        u = np.zeros_like(x)
        for s in self.units:
            s.sync()
            for inl in s.inlets:
                if inl.flow > 0:
                    u += inl.velocity(x)
            for e in s.exhausts:
                if e.flow > 0:
                    u += e.velocity(x)
        return u

    def sinks_only(self, x):
        """The faces' sinks alone (the datasheet's sink speed)."""
        x = np.atleast_2d(np.asarray(x, float))
        u = np.zeros_like(x)
        for s in self.units:
            for inl in s.inlets:
                if inl.flow > 0:
                    u += inl.velocity(x)
        return u

    screen_only = units_only    # the earlier name

    def __call__(self, x):
        self.calls += 1
        x = np.atleast_2d(np.asarray(x, float))
        b = self.base(x) if self.base is not None else np.zeros_like(x)
        return b + self.units_only(x)


# ----------------------------------------------------------------------------------------------- the part
class UnitPart:
    """One stowage unit at a rack-face point.  ``wall_point``: rack-face point under the screen centre; ``normal`` into
    the cabin; ``up``: the in-plane direction of +b; ``pad_fid_side``: '+b' | '-b' | 'outer' (the pouch pads' plate);
    ``outline_sides``: pad-plate sides the outline (siting footprint) covers (default: SITING_PAD_SIDES)."""

    def __init__(self, name: str, wall_point, normal, up, pad_fid_side: str = PAD_FID_DEFAULT, blower: dict | None = None,
                 outline_sides=SITING_PAD_SIDES):
        if pad_fid_side not in PAD_FID_SIDES:
            raise ValueError(f"pad_fid_side must be one of {PAD_FID_SIDES}, got {pad_fid_side!r}")
        self.name = name
        self.n, self.a, self.b = unit_axes(normal, up)
        self.o = np.asarray(wall_point, float)
        self.blower = dict(BLOWER, **(blower or {}))
        self.pad_fid_side = pad_fid_side
        self.outline_sides = tuple(outline_sides)
        self.face_center = self.o + self.n * FRAME_DEPTH          # the screen's (the earlier name)
        self.faces = [Face(f"{name}_hs", "holding_screen", self.face_center, self.n, self.a, self.b,
                           (SCREEN_OPENING[0] / 2, SCREEN_OPENING[1] / 2))]
        for k, s in enumerate((-1.0, 1.0)):
            self.faces.append(Face(f"{name}_pp{k}", "pouch_pad", self.face_center + self.a * (s * PAD_CENTRE_A), self.n, self.a, self.b,
                                   (PAD_OPENING[0] / 2, PAD_OPENING[1] / 2)))
        self.screen = self.faces[0]
        self.pads = self.faces[1:]
        self.housing_center = self.o - self.n * (PLENUM_DEPTH + 0.03)
        xb = SCREEN_OPENING[0] / 2 + (SCREEN_FRAME[0] - SCREEN_OPENING[0]) / 4
        self.exhaust_centers = [self.o + self.n * FRAME_DEPTH - self.a * xb, self.o + self.n * FRAME_DEPTH + self.a * xb]

    # ---- frames
    def P(self, x, y, z):
        """Local (a, b, n) -> world."""
        return self.o + self.a * x + self.b * y + self.n * z

    def R(self) -> np.ndarray:
        """Columns a, b, n."""
        return np.column_stack([self.a, self.b, self.n])

    def face(self, which: str) -> Face:
        """'hs' | 'pp0' | 'pp1' (or the full face name)."""
        for f in self.faces:
            if f.name == which or f.name.endswith("_" + which):
                return f
        raise KeyError(which)

    def network(self) -> UnitNetwork:
        return UnitNetwork(self.name, self.faces, self.exhaust_centers, (self.n, self.b), self.blower)

    # ---- fiducial plates
    def fiducial_center(self, which: str = "hs+b", side: str | None = None) -> np.ndarray:
        """Plate centre in the face plane: which = 'hs+b' | 'hs-b' (screen, above / below) or 'pp0' | 'pp1' (a pad;
        ``side`` = '+b' | '-b' | 'outer', default the unit's pad_fid_side)."""
        fo = SCREEN_FRAME[1] / 2
        if which in ("hs+b", "hs-b", "+b", "-b"):
            sg = 1.0 if which.endswith("+b") else -1.0
            return self.face_center + self.b * sg * (fo + FID_GAP + FID_SIZE[1] / 2)
        k = int(which[-1])
        side = side or self.pad_fid_side
        pc = self.pads[k].c
        if side in ("+b", "-b"):
            sg = 1.0 if side == "+b" else -1.0
            return pc + self.b * sg * (PAD_FRAME[1] / 2 + FID_GAP + FID_SIZE[1] / 2)
        if side == "outer":
            sg = -1.0 if k == 0 else 1.0
            return pc + self.a * sg * (PAD_FRAME[0] / 2 + FID_GAP + FID_SIZE[1] / 2)
        raise ValueError(side)

    def fiducial_axes(self, which: str = "hs+b", side: str | None = None):
        """(t1 = the plate's long side, normal)."""
        if which.startswith("hs") or which in ("+b", "-b"):
            return self.a, self.n
        side = side or self.pad_fid_side
        return (self.a if side in ("+b", "-b") else self.b), self.n

    def fiducial_points(self, which: str = "hs+b", side: str | None = None, nu: int = 7, nv: int = 3):
        """Marker points (nu x nv cell centres) of a (real or candidate) plate and its normal (like dock.plate_points)."""
        c = self.fiducial_center(which, side)
        t1, n = self.fiducial_axes(which, side)
        t2 = np.cross(n, t1)
        hx, hy = FID_SIZE[0] / 2, FID_SIZE[1] / 2
        pts = [c + t1 * ((-1 + 2 * (i + 0.5) / nu) * hx) + t2 * ((-1 + 2 * (j + 0.5) / nv) * hy) for i in range(nu) for j in range(nv)]
        return np.array(pts), n.copy()

    def fiducials(self) -> dict:
        """{plate name: dict(center, normal, t1, size)} of the plates built for this unit."""
        out = {}
        for which, nm in (("hs+b", "fid_top"), ("hs-b", "fid_bottom"), ("pp0", "pp0_fid"), ("pp1", "pp1_fid")):
            t1, n = self.fiducial_axes(which)
            out[f"{self.name}_{nm}"] = dict(center=self.fiducial_center(which).tolist(), normal=n.tolist(), t1=t1.tolist(),
                                            size=list(FID_SIZE), face=("hs" if which.startswith("hs") else which))
        return out

    def deposit_axes(self, side: str = "+b"):
        """(palm x, palm z) of the robot laying a sheet on the screen: x = the face normal, z = toward the plate side."""
        sg = 1.0 if side[0] == "+" else -1.0
        return self.n.copy(), sg * (self.a if side[1] == "a" else self.b)

    # ---- outline, volumes
    def outline(self) -> tuple:
        """(lo (a, b), hi (a, b)) of the unit's footprint on the rack face: the frames, the screen's plates and the pad
        plates of every side in ``outline_sides`` (so the siting holds whichever of those candidates is chosen)."""
        lo = [-(PAD_CENTRE_A + PAD_FRAME[0] / 2), -(SCREEN_FRAME[1] / 2 + FID_GAP + FID_SIZE[1])]
        hi = [-lo[0], -lo[1]]
        for side in self.outline_sides:
            for k in (0, 1):
                t1, n = self.fiducial_axes(f"pp{k}", side)
                c = self.fiducial_center(f"pp{k}", side) - self.face_center
                ca, cb = float(c @ self.a), float(c @ self.b)
                ha = FID_SIZE[0] / 2 if np.allclose(t1, self.a) else FID_SIZE[1] / 2
                hb = FID_SIZE[1] / 2 if np.allclose(t1, self.a) else FID_SIZE[0] / 2
                lo = [min(lo[0], ca - ha), min(lo[1], cb - hb)]
                hi = [max(hi[0], ca + ha), max(hi[1], cb + hb)]
        return tuple(float(v) for v in lo), tuple(float(v) for v in hi)

    def obb(self, protrusion: float | None = None) -> tuple:
        """(centre, R columns a b n, half sizes) of the unit's volume in front of the rack face."""
        lo, hi = self.outline()
        prot = FRAME_DEPTH if protrusion is None else float(protrusion)
        c = self.o + self.a * ((lo[0] + hi[0]) / 2) + self.b * ((lo[1] + hi[1]) / 2) + self.n * (prot / 2)
        return c, self.R(), np.array([(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, prot / 2])

    def approach_zone(self, grow: float | None = None, depth: float | None = None) -> dict:
        """The unit's approach zone (the firmware's exemption for fly_rel to this unit's plates): the outline grown by
        ``grow`` in the face plane, from the rack face to ``depth`` along the normal.  dict(center, R, half, ...)."""
        g = APPROACH["grow_m"] if grow is None else float(grow)
        dp = APPROACH["depth_m"] if depth is None else float(depth)
        lo, hi = self.outline()
        c = self.o + self.a * ((lo[0] + hi[0]) / 2) + self.b * ((lo[1] + hi[1]) / 2) + self.n * (dp / 2)
        half = [(hi[0] - lo[0]) / 2 + g, (hi[1] - lo[1]) / 2 + g, dp / 2]
        return dict(unit=self.name, center=c.tolist(), R=self.R().tolist(), half=half, grow_m=g, depth_m=dp)

    # ---- MJCF
    def geoms(self, with_fiducial: bool = True) -> list:
        """MJCF worldbody geoms (static).  Collision geoms contype / conaffinity 1; display geoms contype 0."""
        from spacekit import fixtures as FX
        nm = self.name
        xy = f'xyaxes="{_v(self.a)} {_v(self.b)}"'
        coll = f'contype="1" conaffinity="1" friction="{FRICTION} 0.01 0.001" {CONTACT}'
        disp = 'contype="0" conaffinity="0"'
        fd = FRAME_DEPTH
        g = []

        def frame_and_face(tag, ca, opening, frame, face_rgba, n_rows):
            bwa, bwb = (frame[0] - opening[0]) / 2, (frame[1] - opening[1]) / 2
            for k, s in ((0, 1), (1, -1)):
                g.append(f'<geom name="{nm}_{tag}_frame_b{k}" type="box" pos="{_v(self.P(ca, s * (opening[1] / 2 + bwb / 2), fd / 2))}" '
                         f'size="{frame[0] / 2} {bwb / 2} {fd / 2}" {xy} rgba="{COLORS["frame"]}" {coll}/>')
                g.append(f'<geom name="{nm}_{tag}_frame_a{k}" type="box" pos="{_v(self.P(ca + s * (opening[0] / 2 + bwa / 2), 0, fd / 2))}" '
                         f'size="{bwa / 2} {opening[1] / 2} {fd / 2}" {xy} rgba="{COLORS["frame"]}" {coll}/>')
            g.append(f'<geom name="{nm}_{tag}_face" type="box" pos="{_v(self.P(ca, 0, fd - FACE_THICK / 2))}" '
                     f'size="{opening[0] / 2} {opening[1] / 2} {FACE_THICK / 2}" {xy} rgba="{face_rgba}" {coll}/>')
            for k in range(n_rows):            # perforation look (display only, 0.2 mm proud)
                y = -opening[1] / 2 + (k + 0.5) * opening[1] / n_rows
                g.append(f'<geom name="{nm}_{tag}_hole{k}" type="box" pos="{_v(self.P(ca, y, fd + 0.0002))}" '
                         f'size="{opening[0] / 2 - 0.006} 0.0035 0.0001" {xy} rgba="{COLORS["hole"]}" {disp}/>')
            g.append(f'<geom name="{nm}_{tag}_felt" type="box" pos="{_v(self.P(ca, 0, fd - FACE_THICK - FELT_THICK / 2))}" '
                     f'size="{opening[0] / 2} {opening[1] / 2} {FELT_THICK / 2}" {xy} rgba="{COLORS["felt"]}" {disp}/>')
            g.append(f'<geom name="{nm}_{tag}_plenum" type="box" pos="{_v(self.P(ca, 0, -PLENUM_DEPTH / 2 - 0.002))}" '
                     f'size="{frame[0] / 2 - 0.01} {frame[1] / 2 - 0.01} {PLENUM_DEPTH / 2}" {xy} rgba="{COLORS["plenum"]}" {disp}/>')
        frame_and_face("hs", 0.0, SCREEN_OPENING, SCREEN_FRAME, COLORS["face"], 11)
        for k, s in enumerate((-1.0, 1.0)):
            frame_and_face(f"pp{k}", s * PAD_CENTRE_A, PAD_OPENING, PAD_FRAME, COLORS["pad_face"], 8)
        hh = HOUSING_HALF
        g.append(f'<geom name="{nm}_housing" type="box" pos="{_v(self.housing_center)}" size="{hh[0]} {hh[1]} {hh[2]}" {xy} '
                 f'rgba="{COLORS["housing"]}" {disp}/>')
        for k, ec in enumerate(self.exhaust_centers):
            g.append(f'<geom name="{nm}_exhaust{k}" type="box" pos="{_v(ec + self.n * 0.0003)}" '
                     f'size="{EXHAUST_HALF_AB[0]} {EXHAUST_HALF_AB[1]} 0.0002" {xy} rgba="{COLORS["grille"]}" {disp}/>')
        bw = (SCREEN_FRAME[1] - SCREEN_OPENING[1]) / 2
        g.append(f'<geom name="{nm}_idplate" type="box" pos="{_v(self.P(0.12, -(SCREEN_OPENING[1] / 2 + bw / 2), fd + 0.0005))}" '
                 f'size="0.045 0.011 0.0004" {xy} rgba="{COLORS["plate"]}" {disp}/>')
        if with_fiducial:
            for pname, P in self.fiducials().items():
                fx, _ = FX.fiducial_plate(pname, np.asarray(P["center"]) + self.n * 0.0005, self.n, P["t1"], FID_SIZE)
                g.append(fx)
        return g

    def xml(self, with_fiducial: bool = True) -> str:
        return "\n".join(self.geoms(with_fiducial))

    def as_dict(self) -> dict:
        lo, hi = self.outline()
        return dict(name=self.name, wall_point=self.o.tolist(), normal=self.n.tolist(), a=self.a.tolist(), b=self.b.tolist(),
                    pad_fid_side=self.pad_fid_side, faces=[f.as_dict() for f in self.faces], fiducials=self.fiducials(),
                    outline_lo=list(lo), outline_hi=list(hi), approach_zone=self.approach_zone())


def unit_from_site(name: str, site: dict, pad_fid_side: str = PAD_FID_DEFAULT, **kw) -> UnitPart:
    """A UnitPart at a unit site record (``station_v03.json`` unit_sites: wall_point, normal, up)."""
    return UnitPart(name, site["wall_point"], site["normal"], site["up"], pad_fid_side=pad_fid_side, **kw)


PAD_PLATE_KEY = "world.stowage_unit.pad_plate"


def pad_fid_side_of(site_spec: dict) -> str:
    """The pouch pads' plate side from the task's site_spec: ``world.stowage_unit.pad_plate.selected`` of the runtime
    copy (``values[key]`` = the value) or of the contract (``values[key]['value']``), which must be one of the entry's
    candidates; or, in a switch file of the builder's own parameter test, the key
    ``stowage_unit.pouch_pad_fiducial_side`` (a side or ``{value: side}``).  No default: a site_spec without either key
    raises KeyError (contract 1.2.0; before, a missing key silently gave PAD_FID_DEFAULT); an unknown side ValueError."""
    if not isinstance(site_spec, dict):
        raise TypeError(f"site_spec must be a dict, got {type(site_spec).__name__}")
    vals = site_spec.get("values")
    if isinstance(vals, dict) and PAD_PLATE_KEY in vals:
        v = vals[PAD_PLATE_KEY]
        if isinstance(v, dict) and "selected" not in v and isinstance(v.get("value"), dict):
            v = v["value"]                                   # a contract entry
        side = v.get("selected") if isinstance(v, dict) else None
        cands = v.get("candidates") if isinstance(v, dict) else None
        if side not in PAD_FID_SIDES or (cands is not None and side not in cands):
            raise ValueError(f"{PAD_PLATE_KEY}.selected: {side!r} not in {cands if cands is not None else PAD_FID_SIDES}")
        return side
    v = (site_spec.get("stowage_unit") or {}).get("pouch_pad_fiducial_side")
    if isinstance(v, dict):
        v = v.get("value")
    if v is None:
        raise KeyError(f"{PAD_PLATE_KEY}.selected not in the site_spec (no default side)")
    if v not in PAD_FID_SIDES:
        raise ValueError(f"stowage_unit.pouch_pad_fiducial_side: {v!r} not in {PAD_FID_SIDES}")
    return v


# ----------------------------------------------------------------------------------------------- datasheet
SHEET_AREAS = dict(one_strip=0.06 * 0.28, smallest_sheet=0.28 * 0.26, reference_sheet=0.30 * 0.28, largest_sheet=0.30 * 0.30)
POUCH_AREAS = dict(smallest_pouch=0.18 * 0.11, reference_pouch=0.20 * 0.12, largest_pouch=0.22 * 0.14)


def unit_numbers(blower: dict | None = None, distances=(0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)) -> dict:
    """Datasheet numbers of the unit's network: per-face open flow and speed, plenum pressure, shut-off pressure, holds
    (plenum pressure x sealed area, seal 1) for items on the screen and on a pad, the sink speed on the screen's axis
    (the three faces and the exhaust), the exhaust's face speed."""
    from spacekit.suction import fan_dp
    U = UnitPart("u", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), blower=blower)
    N = U.network()
    net = N.net
    out = dict(blower=dict(U.blower), Q_open_cfm=float(net.Q / CFM), Pc_open_Pa=float(net.Pc),
               shutoff_Pa=float(fan_dp(0.0, 1.0, net.size)), fan_size_scale=float(net.size), K_sys=float(net.K_sys))
    out["faces"] = {f.name.split("_", 1)[1]: dict(area_m2=f.area, Q_open_cfm=float(N.by_face[f.name].flow / CFM),
                                                  v_open_ms=float(N.by_face[f.name].flow / f.area)) for f in U.faces}
    out["air_power_open_W"] = float(net.Q * fan_dp(net.Q, 1.0, net.size))

    def solve(blocked: dict) -> dict:
        for f in U.faces:
            net.set_blocked(f.name, blocked.get(f.name.split("_", 1)[1], 0.0))
        net._key = None
        net.solve()
        return dict(Pc_Pa=float(net.Pc), hold_N={k: float(net.Pc * a) for k, a in blocked.items()}, Q_cfm=float(net.Q / CFM))
    rows = {}
    for k, a in SHEET_AREAS.items():
        rows[f"{k}_on_screen"] = dict(sealed_m2=a, **solve({"hs": a}))
    for k, a in POUCH_AREAS.items():
        rows[f"{k}_on_pad"] = dict(sealed_m2=a, **solve({"pp0": a}))
    rows["reference_sheet_and_a_reference_pouch_on_each_pad"] = solve({"hs": SHEET_AREAS["reference_sheet"], "pp0": POUCH_AREAS["reference_pouch"],
                                                                       "pp1": POUCH_AREAS["reference_pouch"]})
    rows["two_reference_pouches_on_one_pad"] = solve({"pp0": 2 * POUCH_AREAS["reference_pouch"]})
    rows["all_faces_sealed"] = solve({"hs": 0.999 * U.screen.area, "pp0": 0.999 * U.pads[0].area, "pp1": 0.999 * U.pads[1].area})
    out["holds"] = rows
    solve({})
    N.sync()
    F = UnitFlow(None, [N])
    prof = []
    for z in distances:
        x = np.array([U.face_center + U.n * max(z, 1e-4)])
        us, ua = F.sinks_only(x)[0], F(x)[0]
        prof.append(dict(distance_m=float(z), sink_speed_ms=float(np.linalg.norm(us)), sink_normal_ms=float(us @ U.n),
                         unit_air_speed_ms=float(np.linalg.norm(ua)), unit_air_normal_ms=float(ua @ U.n)))
    out["screen_axis"] = prof
    out["exhaust_face_speed_ms"] = float(net.Q / len(N.exhausts) / (4 * EXHAUST_HALF_AB[0] * EXHAUST_HALF_AB[1]))
    return out
