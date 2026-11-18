"""Free and static objects of task 07 besides the panel (cell side; DESIGN v1.2 sections 1.2, 2.1, 2.3.3, 2.3.4, 3.3;
decision M13; gate P02; package P07): the dropped-object stands with their two swivel eyes, the deployment cage, and
the world rules for things resting on the mudmat, the guard rail, the roof grating and the cage deck.

Dropped object (the form that passed gate P02, record P2_P02: DESIGN 2.3.3 fallback (1) "a thinner pipe" -> every class
is a small-bore STEEL stand; lift eye moved to the middle of a top cross tube; inner tube 0.20-0.30 m off the face).
  A welded open tubular stand: bottom skid ring L x W (centrelines 1.00 x 0.50 m), four legs, top ring, a cross tube
  across the middle of the top ring, outer height 0.65 m (13 tubes, 9.05 m of tube).  Two swivel eyes, inner diameter
  80 mm, orange, NOT collision geometry (the grasp rule acts on the TCP-to-eye-centre distance, subseacell.grasp):
    drag eye: on the outward long skid tube, centre 0.22 m above the mudmat, 0.08 m outboard of the tube's centreline;
    lift eye: on the middle of the top cross tube, centre 0.20 m above it (hung from it the stand stays level).
  Register (sea water 1025 kg/m^3; the four classes look the same):
    D1 steel 12 x 1.2 mm flooded  24.7 N wet  | D2 steel 13.5 x 1.6 mm flooded  36.2 N
    D3 steel 13.5 x 2.3 mm sealed 43.3 N      | D4 steel 13.5 x 2.3 mm flooded  49.0 N
  Body frame: x along the long side, y across (+y = the outward side, away from the panel), z up; origin in the middle
  of the footprint on the bottom plane (z = 0 = the mudmat contact plane).
  Hydrodynamics (the kit's free-object model has one drag coefficient per body axis, wrong for an open frame): per
  member Morison cross-flow drag 0.5 rho Cd D ds |u_n| u_n on segments <= 0.1 m (the current sampled per segment, so
  the boundary layer is in it) + axial skin friction; buoyancy at the volume centroid; added mass Ca rho V per member,
  merged isotropically (2/3 of the transverse total, the kit's merged-mass convention with gravcomp) + the unsteady
  current force (rho V + m_a) a_c.  Flooded tubes carry their inner water as rigid mass.  FrameHydro.apply(t) must run
  every physics step after the kit's hydro layer and before mj_step2: install_hydro(sim, [...]) wraps the simulator's
  hydro layer (the kit itself is not changed).
  World rule (grader side, not published; G6 / M13): stand, guard rail and mudmat contacts solref (0.004, 1), solimp
  (0.999, 0.9999, 0.001), the stand condim 6 with rolling friction 0.002 m: SURFACE_RULES gives the rail / mudmat /
  laydown geoms the same parameters (MuJoCo mixes the two geoms' parameters).

Deployment cage (DESIGN 1.3 / 2.1 / 3.3 R; G6 measured: hovering in a strong current does not reach "5 s < 2 cm/s",
the vehicle has to sit on the deck).  A steel box open towards +x (the entrance): back wall, two side walls, roof,
base frame; a deck with rubber landing pads (friction cage.deck_friction) and a pocket under the start position for
the marker block that hangs below the jaws (P05: 58 mm below the vehicle's lowest point); a coded plate (the panel
standard's tag plate, code index tag_code) inside on the back wall facing the entrance; two sonar reflector poles
1.2 m apart (cage.reflector_spacing_m) just outside the entrance.  Frame: origin on the sea bed under the centre of the
inner box, x towards the entrance, y left, z up.  The work order gives the deck height (deck_top_m above the sea bed).

    from subseacell import objects as OB
    fs = OB.make_spec("D4");  info = OB.add_frame(vb.spec, "debris", fs, pos, quat)
    eyes = OB.frame_eyes(info)                             # {"debris_drag": ..., "debris_lift": ...} for subseacell.grasp
    cage = OB.add_cage(vb.spec, "cage", pos_w, yaw)        # static geometry + descriptions (tag, reflectors, deck)
    ... sim = uwkit.UnderwaterSim(vb, ...)
    fh = OB.FrameHydro(sim.model, sim.data, "debris", fs, sim.current);  OB.install_hydro(sim, [fh])
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import mujoco

RHO_SEA = 1025.0
GRAV = 9.80665

# ---------------------------------------------------------------------------------------------------- world rules
DEBRIS_CONTACT = dict(solref=(0.004, 1.0), solimp=(0.999, 0.9999, 0.001, 0.5, 2.0))     # task_values debris.contact
DEBRIS_FRICTION = (0.8, 0.02, 0.002)            # sliding 0.8, torsional 0.02, rolling 0.002 m (condim 6)
SURFACE_FRICTION = (0.8, 0.02, 0.001)           # mudmat / rail / face
SURFACE_RULES = dict(rail=dict(DEBRIS_CONTACT), mudmat=dict(DEBRIS_CONTACT), rail_support=dict(DEBRIS_CONTACT))
# the roof grating's collision proxy (a continuous plate: DESIGN 2.3.4, the marker cannot fall through) keeps MuJoCo's
# default contact; measured by tests/test_stab_objects.py (the marker rests on it in a 0.4 m/s current)
ROOF_FRICTION = (0.8, 0.02, 0.001)

# ---------------------------------------------------------------------------------------------------- the stands
FRAME_DEFAULTS = dict(L=1.00, W=0.50, H=0.65, cd=1.2, cf_axial=0.01, ca=1.0, seg_max=0.10,
                      eye_drag_h=0.22, eye_drag_out=0.08, eye_lift_up=0.20, eye_ring_r=0.045, eye_id=0.080)
# P02 revision of DESIGN 2.3.3 (record P2_P02 section 6; the DESIGN text still prints aluminium D1 / D2 and 17.2 mm steel)
REGISTER = {
    "D1": dict(material="steel", od=0.0120, wall=0.0012, rho_m=7850.0, sealed=False),
    "D2": dict(material="steel", od=0.0135, wall=0.0016, rho_m=7850.0, sealed=False),
    "D3": dict(material="steel", od=0.0135, wall=0.0023, rho_m=7850.0, sealed=True),
    "D4": dict(material="steel", od=0.0135, wall=0.0023, rho_m=7850.0, sealed=False),
}
# the published register table of record P2_P02 section 6 (mass in air [kg], wet weight [N]; what P22 prints)
REGISTER_TABLE = {"D1": dict(mass_air_kg=2.9, wet_N=24.7), "D2": dict(mass_air_kg=4.2, wet_N=36.2),
                  "D3": dict(mass_air_kg=5.8, wet_N=43.3), "D4": dict(mass_air_kg=5.8, wet_N=49.0)}
LIFTABLE = ("D1", "D2")                          # D2 only above about 13.8 V (P02 estimate)
COL = dict(steel=(0.66, 0.67, 0.69, 1.0), eye=(1.0, 0.45, 0.05, 1.0))


@dataclass
class FrameSpec:
    kind: str
    material: str
    od: float
    wall: float
    rho_m: float
    sealed: bool
    L: float = FRAME_DEFAULTS["L"]
    W: float = FRAME_DEFAULTS["W"]
    H: float = FRAME_DEFAULTS["H"]
    cd: float = FRAME_DEFAULTS["cd"]
    rho_w: float = RHO_SEA
    members: list = field(default_factory=list)          # [(a_b, b_b, tag)]
    r: float = 0.0
    length_total: float = 0.0
    mass_rigid: float = 0.0
    mass_metal: float = 0.0
    V_outer: float = 0.0
    V_metal: float = 0.0
    m_added: float = 0.0
    wet_weight_N: float = 0.0
    com_b: tuple = (0.0, 0.0, 0.0)
    eyes_b: dict = field(default_factory=dict)

    def __post_init__(self):
        F = FRAME_DEFAULTS
        r = self.od / 2
        self.r = r
        hx, hy = self.L / 2, self.W / 2
        zb, zt = r, self.H - r
        M = []
        for z, lvl in ((zb, "bottom"), (zt, "top")):
            for y, side in ((-hy, "inner"), (hy, "outer")):
                M.append(((-hx, y, z), (hx, y, z), f"{lvl}_long_{side}"))
            for x, side in ((-hx, "left"), (hx, "right")):
                M.append(((x, -hy, z), (x, hy, z), f"{lvl}_short_{side}"))
        for x in (-hx, hx):
            for y in (-hy, hy):
                M.append(((x, y, zb), (x, y, zt), f"leg_{'L' if x < 0 else 'R'}{'i' if y < 0 else 'o'}"))
        M.append(((0.0, -hy, zt), (0.0, hy, zt), "top_cross"))
        self.members = [(np.array(a, float), np.array(b, float), t) for a, b, t in M]
        Ls = np.array([np.linalg.norm(b - a) for a, b, _ in self.members])
        self.length_total = float(Ls.sum())
        a_o = math.pi * r * r
        a_i = math.pi * (r - self.wall) ** 2
        self.V_outer = a_o * self.length_total
        self.V_metal = (a_o - a_i) * self.length_total
        self.mass_metal = self.rho_m * self.V_metal
        self.mass_rigid = self.mass_metal + (0.0 if self.sealed else self.rho_w * a_i * self.length_total)
        self.m_added = F["ca"] * self.rho_w * self.V_outer * 2.0 / 3.0
        self.wet_weight_N = (self.mass_rigid - self.rho_w * self.V_outer) * GRAV
        c = sum(0.5 * (a + b) * l for (a, b, _), l in zip(self.members, Ls)) / self.length_total
        self.com_b = tuple(float(x) for x in c)
        self.eyes_b = dict(drag=[0.0, hy + F["eye_drag_out"], F["eye_drag_h"]], lift=[0.0, 0.0, zt + F["eye_lift_up"]])

    def inertia(self, mass_per_len):
        """Inertia about the CoM (body axes) of thin rods of the given linear density."""
        c = np.asarray(self.com_b)
        I = np.zeros((3, 3))
        for a, b, _ in self.members:
            l = float(np.linalg.norm(b - a))
            m = mass_per_len * l
            e = (b - a) / l
            p = 0.5 * (a + b) - c
            I += m * l * l / 12.0 * (np.eye(3) - np.outer(e, e)) + m * (p @ p * np.eye(3) - np.outer(p, p))
        return I

    def segments(self, seg_max=None):
        seg_max = FRAME_DEFAULTS["seg_max"] if seg_max is None else seg_max
        P, E, S = [], [], []
        for a, b, _ in self.members:
            l = float(np.linalg.norm(b - a))
            n = max(1, int(math.ceil(l / seg_max - 1e-9)))
            e = (b - a) / l
            for k in range(n):
                P.append(a + (k + 0.5) / n * (b - a))
                E.append(e)
                S.append(l / n)
        return np.array(P), np.array(E), np.array(S)

    def summary(self):
        return dict(kind=self.kind, material=self.material, od_mm=self.od * 1e3, wall_mm=self.wall * 1e3,
                    sealed=self.sealed, L=self.L, W=self.W, H=self.H, tube_length_m=round(self.length_total, 4),
                    mass_metal_kg=round(self.mass_metal, 4), mass_rigid_kg=round(self.mass_rigid, 4),
                    V_outer_m3=self.V_outer, wet_weight_N=round(self.wet_weight_N, 3), m_added_kg=round(self.m_added, 4),
                    cd=self.cd, n_members=len(self.members), eyes_b=self.eyes_b)


def make_spec(kind: str, **over) -> FrameSpec:
    d = dict(REGISTER[kind])
    d.update(over)
    return FrameSpec(kind=kind, **d)


def add_frame(spec: mujoco.MjSpec, name: str, fs: FrameSpec, pos, quat, contact=DEBRIS_CONTACT, visual_eyes=True):
    """Add the stand as a free body `name` (collision tubes `{name}_m{i}`); pos / quat = pose of the body frame.
    Returns dict(body, geoms, eyes_b, spec)."""
    b = spec.worldbody.add_body(name=name, pos=list(pos), quat=list(quat))
    b.add_freejoint(name=f"{name}_root")
    rgba = COL["steel"]
    gnames = []
    for i, (a, bb, tag) in enumerate(fs.members):
        g = b.add_geom(name=f"{name}_m{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[fs.r, 0, 0],
                       fromto=list(a) + list(bb), rgba=list(rgba), friction=list(DEBRIS_FRICTION), condim=6, mass=0.0)
        if contact is not None:
            g.solref = list(contact["solref"])
            g.solimp = list(contact["solimp"])
        gnames.append(g.name)
    lam = (fs.mass_rigid + fs.m_added) / fs.length_total
    I = fs.inertia(lam)
    b.explicitinertial = True
    b.mass = fs.mass_rigid + fs.m_added
    b.ipos = list(fs.com_b)
    b.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
    b.gravcomp = fs.m_added / (fs.mass_rigid + fs.m_added)
    if visual_eyes:
        rr = FRAME_DEFAULTS["eye_ring_r"]
        for en, c in fs.eyes_b.items():
            c = np.asarray(c, float)
            for k in range(12):                      # ring in the plane normal to the long axis: visual only
                a0, a1 = 2 * math.pi * k / 12, 2 * math.pi * (k + 1) / 12
                p0 = c + rr * np.array([0.0, math.cos(a0), math.sin(a0)])
                p1 = c + rr * np.array([0.0, math.cos(a1), math.sin(a1)])
                gg = b.add_geom(name=f"{name}_eye_{en}_{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.0035, 0, 0],
                                fromto=list(p0) + list(p1), rgba=list(COL["eye"]))
                gg.contype, gg.conaffinity, gg.mass = 0, 0, 0.0
            base = np.array([c[0], fs.W / 2, fs.r]) if en == "drag" else np.array([c[0], c[1], fs.H - fs.r])
            top = c - np.array([0.0, 0.0, rr])
            gg = b.add_geom(name=f"{name}_eyepost_{en}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.006, 0, 0],
                            fromto=list(base) + list(top), rgba=list(COL["eye"]))
            gg.contype, gg.conaffinity, gg.mass = 0, 0, 0.0
    return dict(body=name, geoms=gnames, eyes_b={k: list(v) for k, v in fs.eyes_b.items()}, spec=fs)


def frame_eyes(info, inner_diameter_m=None):
    """Eye descriptions for subseacell.grasp ({'<body>_drag': ..., '<body>_lift': ...})."""
    idm = FRAME_DEFAULTS["eye_id"] if inner_diameter_m is None else inner_diameter_m
    out = {}
    for en, p in info["eyes_b"].items():
        n = f"{info['body']}_{en}"
        out[n] = dict(name=n, body=info["body"], point_b=[float(x) for x in p], inner_diameter_m=float(idm))
    return out


def frame_quat(yaw_w: float):
    return (math.cos(yaw_w / 2), 0.0, 0.0, math.sin(yaw_w / 2))


def _cross(a, b):
    a, b = np.asarray(a), np.asarray(b)
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis=-1)


class FrameHydro:
    """Per-member Morison model of one stand.  apply(t) ADDS its wrench to data.xfrc_applied[body]; call it after
    mj_step1 and after xfrc_applied was cleared (install_hydro does this inside the kit's step)."""

    def __init__(self, model, data, body: str, fs: FrameSpec, field, rho=RHO_SEA, unsteady=True):
        F = FRAME_DEFAULTS
        self.m, self.d, self.fs, self.field = model, data, fs, field
        self.bid = model.body(body).id
        j = model.body_jntadr[self.bid]
        self.va = int(model.jnt_dofadr[j])
        self.Pb, self.Eb, self.ds = fs.segments()
        self.kn = 0.5 * rho * fs.cd * fs.od * self.ds
        self.ka = 0.5 * rho * F["cf_axial"] * math.pi * fs.od * self.ds
        self.fb = rho * fs.V_outer * GRAV
        self.m_fk = rho * fs.V_outer + fs.m_added
        self.com_b = np.asarray(fs.com_b, float)
        self.unsteady = unsteady
        self.reset()

    def reset(self):
        self._uc_prev = None
        self._t_prev = None
        self.last = {}

    def apply(self, t):
        d = self.d
        R = d.xmat[self.bid].reshape(3, 3)
        x = d.xpos[self.bid]
        com = d.xipos[self.bid]
        P = x + self.Pb @ R.T
        E = self.Eb @ R.T
        v0 = d.qvel[self.va:self.va + 3]
        w = R @ d.qvel[self.va + 3:self.va + 6]
        vp = v0 + _cross(w, P - x)
        U = self.field.velocity(np.vstack([P, com[None, :]]), t)
        uc = U[-1]
        ur = U[:-1] - vp
        ua = np.einsum("ij,ij->i", ur, E)
        un = ur - ua[:, None] * E
        nn = np.linalg.norm(un, axis=1)
        Fi = (self.kn * nn)[:, None] * un + (self.ka * np.abs(ua) * ua)[:, None] * E
        F = Fi.sum(0)
        M = _cross(P - com, Fi).sum(0)
        F_drag = F.copy()
        F[2] += self.fb
        cob = x + R @ self.com_b
        M += np.cross(cob - com, np.array([0.0, 0.0, self.fb]))
        if self.unsteady and self._uc_prev is not None and t > self._t_prev:
            F += self.m_fk * (uc - self._uc_prev) / (t - self._t_prev)
        self._uc_prev, self._t_prev = uc.copy(), t
        d.xfrc_applied[self.bid, :3] += F
        d.xfrc_applied[self.bid, 3:] += M
        self.last = dict(F_drag=F_drag, M=M, uc=uc)
        return F_drag


def install_hydro(sim, hydros):
    """Run each FrameHydro (or anything with .apply(t)) every physics step right after the kit's hydro layer.
    Wraps this simulator instance's hydro.apply; returns the list it calls (append to add more)."""
    lst = list(hydros)
    hyd = sim.hydro
    orig = hyd.apply

    def apply(dt, t):
        nur = orig(dt, t)
        for h in lst:
            h.apply(t)
        return nur
    hyd.apply = apply
    return lst


def frame_points_w(model, data, body, fs: FrameSpec, n=5):
    """Sample points on the tube centrelines in world (n per member)."""
    bid = model.body(body).id
    R = data.xmat[bid].reshape(3, 3)
    x = data.xpos[bid]
    return np.array([x + R @ (a + u * (b - a)) for a, b, _ in fs.members for u in np.linspace(0.0, 1.0, n)])


def body_speed(model, data, body):
    """Linear speed [m/s] of the body's CoM and its angular speed [rad/s]."""
    v = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, model.body(body).id, v, 0)
    return float(np.linalg.norm(v[3:])), float(np.linalg.norm(v[:3]))


# ---------------------------------------------------------------------------------------------------- the cage
CAGE = dict(
    inner=(1.40, 1.40, 1.20),            # depth (x), width (y), height above the deck [m] (G6 placeholder, kept)
    wall=0.03, base_h=0.05, deck_top=0.15, deck_thk=0.03,
    pocket=dict(x=(-0.04, 0.25), y=(-0.14, 0.10), depth=0.10),     # in the cage frame (start pose below)
    start=dict(x=-0.12, y=0.0, yaw=0.0),                         # hull origin x / y and heading of the start pose
    pad=dict(thk=0.012, friction=(1.0, 0.02, 0.002), solref=(0.05, 1.0), solimp=(0.9, 0.95, 0.001, 0.5, 2.0)),   # rubber
    pole=dict(radius=0.05, height=2.0, out=0.10, spacing=1.2),   # reflectors; spacing = task value cage.reflector_spacing_m
    tag=dict(code_index=26, size=(0.424, 0.200), centre_above_deck=0.40, thickness=0.006),
    rgba=dict(steel=(0.80, 0.42, 0.10, 1.0), pad=(0.12, 0.12, 0.12, 1.0), pole=(0.90, 0.90, 0.90, 1.0),
              plate=(0.95, 0.95, 0.95, 1.0)),
)


def add_cage(spec: mujoco.MjSpec, name="cage", pos_w=(0.0, 0.0, 0.0), yaw=0.0, params=None):
    """Static cage at pos_w (sea-bed point under the inner box centre) facing yaw (+x of the cage = the entrance
    direction).  Geoms are named <name>_<part>_steel / _rubber (acoustic classes by name).  Returns the description:
    pose, deck height, pocket, tag plate (centre, normal, size, code index) and reflector poles in world."""
    C = copy.deepcopy(CAGE if params is None else params)
    dpt, wid, hgt = C["inner"]
    t = C["wall"]
    q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    b = spec.worldbody.add_body(name=name, pos=list(pos_w), quat=list(q))
    BOX, CYL = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    rg = C["rgba"]
    geoms = []

    def box(nm, half, centre, rgba=rg["steel"], **kw):
        g = b.add_geom(name=f"{name}_{nm}", type=BOX, size=list(half), pos=list(centre), rgba=list(rgba), **kw)
        geoms.append(g.name)
        return g
    zd = C["deck_top"]
    hx, hy = dpt / 2, wid / 2
    # base frame on the sea bed (outer footprint)
    box("base_steel", [hx + t, hy + t, C["base_h"] / 2], [0, 0, C["base_h"] / 2])
    # deck plate with a pocket (four plates around the hole) and the pocket floor
    px, py, pdp = C["pocket"]["x"], C["pocket"]["y"], C["pocket"]["depth"]
    zc = zd - C["deck_thk"] / 2 - C["pad"]["thk"]
    hz = C["deck_thk"] / 2
    pieces = [((-hx, px[0]), (-hy, hy)), ((px[1], hx), (-hy, hy)), ((px[0], px[1]), (-hy, py[0])), ((px[0], px[1]), (py[1], hy))]
    for i, ((x0, x1), (y0, y1)) in enumerate(pieces):
        box(f"deck{i}_steel", [(x1 - x0) / 2, (y1 - y0) / 2, hz], [(x0 + x1) / 2, (y0 + y1) / 2, zc])
        pd = C["pad"]
        box(f"deckpad{i}_rubber", [(x1 - x0) / 2, (y1 - y0) / 2, pd["thk"] / 2],
            [(x0 + x1) / 2, (y0 + y1) / 2, zd - pd["thk"] / 2], rgba=rg["pad"], friction=list(pd["friction"]),
            solref=list(pd["solref"]), solimp=list(pd["solimp"]))
    zf = zd - pdp
    box("pocket_floor_steel", [(px[1] - px[0]) / 2, (py[1] - py[0]) / 2, 0.005], [(px[0] + px[1]) / 2, (py[0] + py[1]) / 2, zf - 0.005])
    # columns under the deck to the base (one block each side, outside the pocket)
    box("deckcol_l_steel", [hx, 0.03, (zc - hz - C["base_h"]) / 2], [0, hy - 0.03, (C["base_h"] + zc - hz) / 2])
    box("deckcol_r_steel", [hx, 0.03, (zc - hz - C["base_h"]) / 2], [0, -hy + 0.03, (C["base_h"] + zc - hz) / 2])
    # walls and roof
    zt = zd + hgt
    box("back_steel", [t / 2, hy + t, (zt - C["base_h"]) / 2], [-hx - t / 2, 0, (zt + C["base_h"]) / 2])
    for s, sy in (("l", 1.0), ("r", -1.0)):
        box(f"side_{s}_steel", [hx, t / 2, (zt - C["base_h"]) / 2], [0, sy * (hy + t / 2), (zt + C["base_h"]) / 2])
    box("roof_steel", [hx + t, hy + t, t / 2], [0, 0, zt + t / 2])
    # reflector poles just outside the entrance
    po = C["pole"]
    poles = []
    for s, sy in (("l", 1.0), ("r", -1.0)):
        c = [hx + po["out"], sy * po["spacing"] / 2, po["height"] / 2]
        g = b.add_geom(name=f"{name}_pole_{s}_steel", type=CYL, size=[po["radius"], po["height"] / 2, 0], pos=c,
                       rgba=list(rg["pole"]))
        geoms.append(g.name)
        poles.append(c)
    # coded plate inside on the back wall, facing the entrance (appearance: render twin; a thin non-colliding plate)
    tg = C["tag"]
    tc = [-hx + tg["thickness"] / 2, 0.0, zd + tg["centre_above_deck"]]
    g = b.add_geom(name=f"{name}_tag_plate", type=BOX, size=[tg["thickness"] / 2, tg["size"][0] / 2, tg["size"][1] / 2],
                   pos=tc, rgba=list(rg["plate"]))
    g.contype, g.conaffinity = 0, 0
    Rc = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    pw = np.asarray(pos_w, float)
    to_w = lambda p: (pw + Rc @ np.asarray(p, float)).tolist()
    return dict(name=name, pos_w=pw.tolist(), yaw=float(yaw), R_w=Rc.tolist(), params=C, geoms=geoms,
                deck_top_w=float(pw[2] + zd), deck_top_m=float(zd), inner=list(C["inner"]),
                entrance_centre_w=to_w([hx, 0.0, zd + hgt / 2]), entrance_normal_w=(Rc @ np.array([1.0, 0, 0])).tolist(),
                pocket=dict(C["pocket"]), poles_w=[to_w(c) for c in poles], pole_radius=po["radius"],
                pole_height=po["height"], tag=dict(code_index=tg["code_index"], size_m=list(tg["size"]),
                                                    centre_w=to_w([-hx + tg["thickness"], 0.0, tc[2]]),
                                                    normal_w=(Rc @ np.array([1.0, 0, 0])).tolist()),
                pads=[f"{name}_deckpad{i}_rubber" for i in range(4)],
                acoustic_overrides={f"{name}_deckpad{i}_rubber": "plastic" for i in range(4)})   # MaterialMap(overrides=)


def vehicle_lowest_z(model, data, hull="hull"):
    """Lowest point of the vehicle's colliding geoms (hull frame z, metres) at the current configuration."""
    hid = model.body(hull).id
    R = data.xmat[hid].reshape(3, 3)
    p = data.xpos[hid]
    low = math.inf
    for i in range(model.ngeom):
        if model.body_rootid[model.geom_bodyid[i]] != hid or (model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0):
            continue
        Rg = data.geom_xmat[i].reshape(3, 3)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[i]
            va, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            W = data.geom_xpos[i] + model.mesh_vert[va:va + nv] @ Rg.T
            low = min(low, float(((W - p) @ R)[:, 2].min()))
        else:
            low = min(low, float(((data.geom_xpos[i] - p) @ R)[2] - model.geom_rbound[i]))
    return low


def cage_start_pose(cage: dict, lowest_z: float, clearance=0.002):
    """Hull pose (pos, quat) sitting on the deck at the start position, heading out of the entrance; lowest_z is the
    vehicle's lowest point in the hull frame (vehicle_lowest_z), clearance the gap above the deck pads."""
    st = cage["params"]["start"]
    Rc = np.asarray(cage["R_w"], float)
    p = np.asarray(cage["pos_w"], float) + Rc @ np.array([st["x"], st["y"], 0.0])
    p[2] = cage["deck_top_w"] - lowest_z + clearance
    yaw = cage["yaw"] + st["yaw"]
    return p, np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def in_cage(cage: dict, p_w, margin=0.0):
    """Is the world point inside the cage's inner box (above the deck, below the roof)?"""
    Rc = np.asarray(cage["R_w"], float)
    q = Rc.T @ (np.asarray(p_w, float) - np.asarray(cage["pos_w"], float))
    dpt, wid, hgt = cage["inner"]
    zd = cage["deck_top_m"]
    return bool(abs(q[0]) <= dpt / 2 - margin and abs(q[1]) <= wid / 2 - margin and zd <= q[2] <= zd + hgt)
