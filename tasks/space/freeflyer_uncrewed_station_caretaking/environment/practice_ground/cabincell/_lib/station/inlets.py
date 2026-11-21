"""Return-inlet part family of the station scene (station library v0.3; the changes against
v0.2 are marked "v0.3").

The part (local frame x = long axis t1, y = b = n x t1, z = n into the cabin, origin = the rack-face point under the
part centre):
  frame       0.46 x 0.18 m outer, 0.030 m proud of the rack face, 0.40 x 0.12 m opening (0.048 m^2)    collision
  slats       7 bars across the opening, open-area ratio 0.55, front faces flush with the frame front      collision
  screen      20-mesh debris screen 10 mm behind the face (display)
  media       the removable part that makes the build: a pre-filter pad 6 mm thick right behind the screen,
              or an inlet filter element (pleated, 0.10 m deep) whose face sits right behind the screen       display
  id plate    0.090 x 0.022 m plate on the +b frame bar with the inlet's neutral label (texture)           display
  plenum      box behind the rack face that the duct leaves from (dark)                                     display
The suction face of the simulation kit (``Inlet``) is the frame front: rack-face point + n * FRAME_DEPTH.

Builds (one build per cabin air unit; the inlet pressure is a property of the whole fan network):
  "element" = grille + debris screen + inlet filter element: K_face 8, r_face 396.6 Pa per m/s nominal, 380-460;
  "pad"     = grille + debris screen + one pre-filter pad: K_face 8, r_face 64 Pa per m/s nominal; v0.3 (the hold-force
              classes): the range depends on the unit size, 49-115 on a two-inlet unit, 59-127 on a three-inlet unit.
"""
from __future__ import annotations

import math

import numpy as np

# ----------------------------------------------------------------------------------------------- geometry (metres)
FRAME_OUTER = (0.46, 0.18)
FRAME_DEPTH = 0.030
OPENING = (0.40, 0.12)
N_SLATS = 7
OPEN_RATIO = 0.55
SLAT_DEPTH = 0.020
SCREEN_Z = FRAME_DEPTH - 0.010
PAD_THICK = 0.006
ELEMENT_DEPTH = 0.10
PLENUM = (0.44, 0.16, 0.16)
PLATE = (0.090, 0.022)
DESIGN_FLOW_CFM = 40.0          # per inlet, the station's design value  [C]
CFM = 4.71947e-4
INLET_FRICTION = 0.7            # item on the debris screen / slats (placeholder)  [G]

BUILDS = {
    "element": dict(label="element-type", spacekit_build="filter_face", K_face=8.0, r_face=396.6, r_face_range=(380.0, 460.0),
                    media="inlet filter element", hold_class_N=(5.5, 8.5)),
    "pad": dict(label="pad-type", spacekit_build="prefilter_light", K_face=8.0, r_face=64.0, r_face_range=(49.0, 127.0),
                media="pre-filter pad", hold_class_N=(1.15, 1.60)),
}
# v0.3 (spec_values iva.inlet.hold_classes): r_face range by build and unit size [Pa per m/s]
R_FACE_RANGE = {("pad", 2): (49.0, 115.0), ("pad", 3): (59.0, 127.0), ("element", 2): (380.0, 460.0),
                ("element", 3): (380.0, 460.0)}


def r_face_range(build: str, n_inlets: int) -> tuple:
    """(lo, hi) r_face of a build on a cabin air unit with ``n_inlets`` parallel inlets (v0.3)."""
    return R_FACE_RANGE[(build, int(n_inlets))]

COLORS = dict(frame="0.70 0.71 0.73 1", slat="0.42 0.43 0.45 1", screen="0.20 0.20 0.22 0.30",
              pad="0.84 0.82 0.74 1", element="0.93 0.93 0.90 1", pleat="0.72 0.72 0.69 1", plenum="0.04 0.04 0.045 1",
              plate="0.95 0.95 0.92 1")


def slat_centres():
    """x positions (local) of the slat centres and the slat width: N_SLATS bars, open ratio OPEN_RATIO over the
    opening length, equal gaps (N_SLATS + 1)."""
    L = OPENING[0]
    w = (1.0 - OPEN_RATIO) * L / N_SLATS
    g = OPEN_RATIO * L / (N_SLATS + 1)
    xs = [-L / 2 + g * (k + 1) + w * (k + 0.5) for k in range(N_SLATS)]
    return xs, w


def frame_axes(normal, t1):
    """(n, a, b) unit vectors: n into the cabin, a = t1 projected into the wall plane, b = n x a."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    a = np.asarray(t1, float) - (np.asarray(t1, float) @ n) * n
    a /= np.linalg.norm(a)
    return n, a, np.cross(n, a)


def _v(x, k=6):
    return " ".join(f"{float(v):.{k}g}" for v in np.ravel(x))


def label_of(inlet_id: str) -> str:
    """Neutral printed label of an inlet id: 'ri_l2_1' -> 'RI-L2-1'."""
    return inlet_id.upper().replace("_", "-")


def plate_mesh_asset(name: str, tex_file: str) -> list:
    """Asset lines: a textured quad mesh of the plate size (z = 0) + texture + material."""
    return [f'<texture name="tx_{name}" type="2d" file="{tex_file}"/>',
            f'<material name="mt_{name}" texture="tx_{name}" specular="0.1" shininess="0.1"/>',
            f'<mesh name="ms_{name}" inertia="shell" vertex="{-PLATE[0] / 2} {-PLATE[1] / 2} 0  {PLATE[0] / 2} {-PLATE[1] / 2} 0  '
            f'{PLATE[0] / 2} {PLATE[1] / 2} 0  {-PLATE[0] / 2} {PLATE[1] / 2} 0" '
            f'texcoord="1 0  0 0  0 1  1 1" face="0 1 2  0 2 3"/>']


def inlet_geoms(inl: dict, build: str, plate_mesh: str | None = None, prefix: str = "", text_up=None) -> list:
    """MJCF worldbody geoms of one inlet part (static).  inl: dict with 'id', 'wall_point', 'normal', 't1'.
    build: 'element' | 'pad'.  Collision geoms (frame bars, slats) have contype / conaffinity 1 and group 2; the
    display geoms are contype 0.  Geom names start with '<prefix><id>_'."""
    if build not in BUILDS:
        raise ValueError(f"unknown inlet build {build!r}")
    n, a, b = frame_axes(inl["normal"], inl["t1"])
    o = np.asarray(inl["wall_point"], float)
    nm = prefix + inl["id"]
    xy = f'xyaxes="{_v(a)} {_v(b)}"'

    def P(x, y, z):
        return o + a * x + b * y + n * z

    coll = f'contype="1" conaffinity="1" group="2" friction="{INLET_FRICTION} 0.01 0.001" solref="0.004 1" solimp="0.95 0.99 0.001" condim="4" priority="1"'
    disp = 'contype="0" conaffinity="0" group="2"'
    fo, fd, op = FRAME_OUTER, FRAME_DEPTH, OPENING
    bar_w_long = (fo[1] - op[1]) / 2          # 0.03: the bars above / below the opening
    bar_w_short = (fo[0] - op[0]) / 2         # 0.03: the bars left / right
    g = []
    for k, s in ((0, 1), (1, -1)):
        g.append(f'<geom name="{nm}_frame_b{k}" type="box" pos="{_v(P(0, s * (op[1] / 2 + bar_w_long / 2), fd / 2))}" '
                 f'size="{fo[0] / 2} {bar_w_long / 2} {fd / 2}" {xy} rgba="{COLORS["frame"]}" {coll}/>')
        g.append(f'<geom name="{nm}_frame_a{k}" type="box" pos="{_v(P(s * (op[0] / 2 + bar_w_short / 2), 0, fd / 2))}" '
                 f'size="{bar_w_short / 2} {op[1] / 2} {fd / 2}" {xy} rgba="{COLORS["frame"]}" {coll}/>')
    xs, w = slat_centres()
    for k, x in enumerate(xs):
        g.append(f'<geom name="{nm}_slat{k}" type="box" pos="{_v(P(x, 0, fd - SLAT_DEPTH / 2))}" size="{w / 2} {op[1] / 2} {SLAT_DEPTH / 2}" '
                 f'{xy} rgba="{COLORS["slat"]}" {coll}/>')
    g.append(f'<geom name="{nm}_screen" type="box" pos="{_v(P(0, 0, SCREEN_Z))}" size="{op[0] / 2} {op[1] / 2} 0.0004" {xy} '
             f'rgba="{COLORS["screen"]}" {disp}/>')
    g += media_geoms(inl, build, prefix)
    g.append(f'<geom name="{nm}_plenum" type="box" pos="{_v(P(0, 0, -PLENUM[2] / 2 - 0.002))}" size="{PLENUM[0] / 2} {PLENUM[1] / 2} {PLENUM[2] / 2}" '
             f'{xy} rgba="{COLORS["plenum"]}" {disp}/>')
    pc = P(-0.10, op[1] / 2 + bar_w_long / 2, fd + 0.0006)
    pxy = xy
    if text_up is not None:
        # the label reads upright for a viewer facing the wall with "up" = text_up: plate on the bar on that side,
        # quad axes (n x U, -U) (the quad's texture top is its -y edge, its reading direction -x)
        U = np.asarray(text_up, float) - (np.asarray(text_up, float) @ n) * n
        U /= np.linalg.norm(U)
        side = 1.0 if U @ b > 0 else -1.0
        pc = P(0.0, side * (op[1] / 2 + bar_w_long / 2), fd + 0.0006) + np.cross(U, n) * (-0.10)
        pxy = f'xyaxes="{_v(np.cross(n, U))} {_v(-U)}"'
    if plate_mesh:
        g.append(f'<geom name="{nm}_plate" type="mesh" mesh="ms_{plate_mesh}" material="mt_{plate_mesh}" pos="{_v(pc)}" {pxy} '
                 f'{disp}/>')
    else:
        g.append(f'<geom name="{nm}_plate" type="box" pos="{_v(pc)}" size="{PLATE[0] / 2} {PLATE[1] / 2} 0.0005" {xy} '
                 f'rgba="{COLORS["plate"]}" {disp}/>')
    return g


def media_geoms(inl: dict, build: str, prefix: str = "") -> list:
    """Display geoms of the removable media of one inlet (names '<prefix><id>_media_*'): the pre-filter pad, or the
    inlet filter element with its pleat folds.  A site switches the build by replacing these geoms."""
    if build not in BUILDS:
        raise ValueError(f"unknown inlet build {build!r}")
    n, a, b = frame_axes(inl["normal"], inl["t1"])
    o = np.asarray(inl["wall_point"], float)
    nm = prefix + inl["id"]
    xy = f'xyaxes="{_v(a)} {_v(b)}"'

    def P(x, y, z):
        return o + a * x + b * y + n * z

    disp = 'contype="0" conaffinity="0" group="2"'
    op = OPENING
    g = []
    if build == "pad":
        g.append(f'<geom name="{nm}_media_pad" type="box" pos="{_v(P(0, 0, SCREEN_Z - 0.0006 - PAD_THICK / 2))}" '
                 f'size="{op[0] / 2} {op[1] / 2} {PAD_THICK / 2}" {xy} rgba="{COLORS["pad"]}" {disp}/>')
    else:
        zf = SCREEN_Z - 0.0012
        g.append(f'<geom name="{nm}_media_element" type="box" pos="{_v(P(0, 0, zf - ELEMENT_DEPTH / 2))}" '
                 f'size="{op[0] / 2} {op[1] / 2} {ELEMENT_DEPTH / 2}" {xy} rgba="{COLORS["element"]}" {disp}/>')
        npl = 24
        for k in range(npl):                    # pleat folds seen through the slats
            x = -op[0] / 2 + (k + 0.5) * op[0] / npl
            g.append(f'<geom name="{nm}_media_pleat{k}" type="box" pos="{_v(P(x, 0, zf + 0.0004))}" size="0.0012 {op[1] / 2} 0.0004" '
                     f'{xy} rgba="{COLORS["pleat"]}" {disp}/>')
    return g


def face_center(inl: dict) -> np.ndarray:
    """Centre of the suction face (frame front) of an inlet."""
    n, _, _ = frame_axes(inl["normal"], inl["t1"])
    return np.asarray(inl["wall_point"], float) + n * FRAME_DEPTH


# ----------------------------------------------------------------------------------------------- network numbers
def network(n_inlets: int, build: str, r_face: float | None = None, flow_cfm: float = DESIGN_FLOW_CFM,
            fan_scale="design_point"):
    """A spacekit FanNetwork of one cabin air unit with n identical inlets of this part (opening 0.40 x 0.12 m)."""
    from spacekit.suction import FanNetwork, Inlet

    B = BUILDS[build]
    bd = dict(K_face=B["K_face"], r_face=float(B["r_face"] if r_face is None else r_face))
    inls = [Inlet(f"i{k}", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (OPENING[0] / 2, OPENING[1] / 2),
                  flow=flow_cfm * CFM) for k in range(n_inlets)]
    return FanNetwork(inls, build=bd, Q_design=flow_cfm * CFM * n_inlets, fan_scale=fan_scale), inls


def blocked_state(n_inlets: int, build: str, n_blocked: int, sealed_m2: float, r_face: float | None = None,
                  flow_cfm: float = DESIGN_FLOW_CFM) -> dict:
    """Operating point with ``n_blocked`` of the unit's inlets each sealed over ``sealed_m2`` by an item lying flat
    (seal fraction 1).  Returns the inlet pressure difference, the hold force on one blocking item
    (P_c x sealed area), total and per-inlet flows and face velocities."""
    net, inls = network(n_inlets, build, r_face, flow_cfm)
    for k in range(n_blocked):
        net.set_blocked(inls[k].name, sealed_m2)
    net._key = None
    net.solve()
    A = OPENING[0] * OPENING[1]
    q_open = [i.flow for i in inls[n_blocked:]] or [0.0]
    q_bl = [i.flow for i in inls[:n_blocked]] or [0.0]
    return dict(n_inlets=n_inlets, build=build, r_face=float(net.build[inls[0].name]["r_face"]), n_blocked=n_blocked,
                sealed_m2=sealed_m2 if n_blocked else 0.0, Pc_Pa=float(net.Pc), hold_N=float(net.Pc * sealed_m2) if n_blocked else 0.0,
                Q_total_cfm=float(net.Q / CFM), Q_open_inlet_cfm=float(q_open[0] / CFM),
                Q_blocked_inlet_cfm=float(q_bl[0] / CFM) if n_blocked else None,
                face_velocity_open_ms=float(q_open[0] / A),
                face_velocity_blocked_open_part_ms=float(q_bl[0] / max(A - sealed_m2, 1e-9)) if n_blocked else None,
                fan_size_scale=float(net.size))


def sink_speed_profile(distances=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50), flow_cfm: float = DESIGN_FLOW_CFM) -> list:
    """Speed of the inlet's own near field (spacekit Inlet line of half-space sinks, no cabin circulation) on the
    axis in front of the face centre, at the given distances [m]."""
    from spacekit.suction import Inlet

    inl = Inlet("x", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (OPENING[0] / 2, OPENING[1] / 2), flow=flow_cfm * CFM)
    out = []
    for s in distances:
        u = inl.velocity(np.array([[0.0, 0.0, max(s, 1e-4)]]))[0]
        out.append(dict(distance_m=float(s), speed_ms=float(np.linalg.norm(u))))
    return out


def face_velocity_ms(flow_cfm: float = DESIGN_FLOW_CFM) -> float:
    """Gross face velocity of an open inlet: flow / opening area."""
    return flow_cfm * CFM / (OPENING[0] * OPENING[1])


def hold_table(units=(("l2", 3), ("h2", 2), ("l1", 3)), sealed=(0.026, 0.0336, 0.036)) -> list:
    """Rows for the note: every unit size x build x r_face (range ends and nominal) x 0 / 1 / 2 blocked inlets x
    sealed area."""
    rows = []
    for unit, n in units:
        for build, B in BUILDS.items():
            lo, hi = r_face_range(build, n)
            for r in sorted({lo, B["r_face"], hi}):
                for nb in range(0, min(2, n) + 1):
                    for s in (sealed if nb else (0.0,)):
                        r0 = blocked_state(n, build, nb, s, r_face=r)
                        r0["unit"] = unit
                        lo, hi = B["hold_class_N"]
                        r0["in_class"] = bool(lo <= r0["hold_N"] <= hi) if nb else None
                        rows.append(r0)
    return rows


def wall_offset_note() -> str:
    return (f"frame front {FRAME_DEPTH * 1000:.0f} mm proud of the rack face; opening {OPENING[0]} x {OPENING[1]} m "
            f"({OPENING[0] * OPENING[1]:.3f} m^2), {N_SLATS} slats, open ratio {OPEN_RATIO}")
