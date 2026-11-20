"""The framed-inlet bench of the constant-force fixture (no robot).

The sheet's size and mass are the candidate's (the three families of ``cabincell.site.items``), the loop comes from
spacekit 0.4.0 (``soft_item(loop="edge_mid")``), the plane angle is the site's (station_site/2: ``theta_deg`` 0 = the
chain along the opening's short axis b, the chain = cos(theta) b - sin(theta) a), the flow per inlet is a
parameter, the unit-size r_face ranges come from the site_spec.

Bench: flat wall x = 0 (normal +x), one return-inlet frame 0.46 x 0.18 x 0.03 m (solid box) whose front face x = 0.03
is the opening plane (0.40 x 0.12 m; a = +y, b = +z); one fan network of ``n_inlets`` parallel inlets (this one + the
others without geometry), K_face 8, r_face per candidate.  The sheet lies on the opening plane 1.5 mm off it at the
start; suction on the five strip bodies (3 x 14 cells per strip) every physics step, the network
re-solved every 5 steps (100 Hz).  Pull directions toward both chain ends: ``tiltNN_e0`` / ``tiltNN_e4`` (NN deg off
the face normal toward strip 0 / strip 4), ``drag_e0`` / ``drag_e4`` (80 deg), ``out45`` (the loop on strip 0, 45 deg
toward the strip-0 end), ``normal``.
"""
from __future__ import annotations

import math

import numpy as np

from . import LIB  # noqa: F401  (sys.path)

import mujoco  # noqa: E402

from spacekit import fixtures as FX  # noqa: E402
from spacekit import suction as SU  # noqa: E402

CFM = SU.CFM
FRAME = (0.46, 0.18, 0.03)          # long (y = a), short (z = b), thickness (x) [m]
OPEN_HALF = (0.20, 0.06)            # opening half sizes along a, b [m]
FX_X = FRAME[2]                     # opening plane x [m]
K_FACE = 8.0
WALL_FRICTION = 0.7
SHEET_FRICTION = 0.7
SHEET_GRID = (3, 14)
START_GAP = 0.0015                  # back face off the opening plane at the start [m]
DT = 0.002
T_FIXTURE = 40.0
SETTLE_S = 1.0
PULLS_STRAP = ("normal", "tilt30_e0", "tilt30_e4", "tilt45_e0", "tilt45_e4", "tilt60_e0", "tilt60_e4", "drag_e0", "drag_e4")
PULLS_LOOP = ("normal", "out45")


# ------------------------------------------------------------------------------------------------ geometry / scene
def sheet_frame(theta_deg: float) -> np.ndarray:
    """Columns: chain direction x (= cos th b - sin th a, the site convention), hinge direction y = n x x, n = +x."""
    th = math.radians(theta_deg)
    xl = np.array([0.0, -math.sin(th), math.cos(th)])
    zl = np.array([1.0, 0.0, 0.0])
    yl = np.cross(zl, xl)
    return np.column_stack([xl, yl, zl])


def mat2quat(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).ravel())
    return q


def sheet_xml(cand: dict, name: str = "blk"):
    """The candidate's sheet on the opening plane: (body xml, contact excludes, strip half sizes, R).  Strap on the
    middle strip (site ``<name>_grasp_centre``), loop at the middle of end strip 0 (``<name>_grasp_corner``)."""
    lx, ly, lz = (float(v) for v in cand["size"])
    n = int(cand.get("n_strips", 5))
    sw = lx / n
    R = sheet_frame(float(cand["theta_deg"]))
    off = cand.get("offset_m", (0.0, 0.0))
    centre = np.array([FX_X + lz / 2 + START_GAP, float(off[0]), float(off[1])])
    root = centre - R[:, 0] * sw * (n // 2)
    xml, excl, half = FX.soft_item(name, root, mat2quat(R), n=n, size=(lx, ly, lz), mass=float(cand["mass"]),
                                   hinge_k=float(cand.get("hinge_k", 0.002)), friction=SHEET_FRICTION, centre_handle=True,
                                   loop="edge_mid")
    return xml, excl, half, R


def world_xml(wall_half=(1.0, 1.0)) -> str:
    wall = (f'<geom name="wall" type="box" pos="-0.03 0 0" size="0.03 {wall_half[0]} {wall_half[1]}" rgba="0.8 0.8 0.8 1" '
            f'friction="{WALL_FRICTION} 0.01 0.001"/>')
    fr = (f'<geom name="inlet_frame" type="box" pos="{FRAME[2] / 2} 0 0" size="{FRAME[2] / 2} {FRAME[0] / 2} {FRAME[1] / 2}" '
          f'rgba="0.25 0.27 0.30 1" friction="{WALL_FRICTION} 0.01 0.001" {FX.CONTACT}/>')
    return wall + "\n" + fr


def item_model(bodies: str, contact: str = "", dt: float = DT):
    """Wall + frame + the sheet; no robot, no gravity."""
    xml = f"""<mujoco><option timestep="{dt}" gravity="0 0 0" cone="elliptic" impratio="10" integrator="implicitfast"/>
      <asset>{FX.MATERIALS}</asset>
      <default><geom solref="0.004 1" solimp="0.95 0.99 0.001" condim="4"/></default>
      <worldbody>{world_xml()}
      {bodies}</worldbody><contact>{contact}</contact></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


# ------------------------------------------------------------------------------------------------ fan network
def make_network(r_face: float, n_inlets: int = 3, flow_cfm: float = 40.0, tau_fan: float = 3.0):
    q1 = float(flow_cfm) * CFM
    inl = [SU.Inlet("g0", (FX_X, 0.0, 0.0), (1, 0, 0), (0, 1, 0), OPEN_HALF, flow=q1)]
    inl += [SU.Inlet(f"g{i}", (FX_X, 5.0 * i, 0.0), (1, 0, 0), (0, 1, 0), OPEN_HALF, flow=q1) for i in range(1, int(n_inlets))]
    return SU.FanNetwork(inl, build=dict(K_face=K_FACE, r_face=float(r_face)), Q_design=int(n_inlets) * q1, tau_fan=tau_fan,
                         fan_scale="design_point")


def hold_static(r_face: float, A_sealed: float, n_inlets: int = 3, flow_cfm: float = 40.0) -> tuple:
    """(hold [N], P_c blocked [Pa], P_c unblocked [Pa]) of the network with A_sealed on inlet g0."""
    net = make_network(r_face, n_inlets, flow_cfm)
    net.solve()
    pc0 = float(net.Pc)
    net.set_blocked("g0", float(A_sealed))
    net.solve()
    return float(net.Pc) * float(A_sealed), float(net.Pc), pc0


def r_face_max(n_inlets: int = 3, flow_cfm: float = 40.0, hi: float = 1200.0) -> float:
    lo = 1.0
    while hi - lo > 1.0:
        mid = 0.5 * (lo + hi)
        try:
            make_network(mid, n_inlets, flow_cfm)
            lo = mid
        except ValueError:
            hi = mid
    return lo


def r_face_for_hold(target_N: float, A_sealed: float, n_inlets: int = 3, flow_cfm: float = 40.0, lo: float = 0.5,
                    hi: float = 600.0) -> float:
    """r_face [Pa s/m] that gives the target hold with A_sealed on inlet g0."""
    try:
        make_network(hi, n_inlets, flow_cfm)
    except ValueError:
        hi = r_face_max(n_inlets, flow_cfm, hi)
    return float(SU.brent(lambda r: hold_static(r, A_sealed, n_inlets, flow_cfm)[0] - float(target_N), lo, hi, xtol=1e-6))


def strip_bodies(m, name: str = "blk", n: int = 5) -> list:
    return [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{name}{i}") for i in range(n)]


def make_suction(m, net, half, name: str = "blk", n: int = 5):
    items = [(b, tuple(half), SHEET_GRID) for b in strip_bodies(m, name, n)]
    return SU.SuctionFaces(m, items, net), [it[0] for it in items]


def step_suction(m, d, suc, net, W, k: int, extra=None) -> None:
    """One physics step of the bench: suction (and the caller's extra wrench) between mj_step1 and mj_step2, the
    network every 5 steps."""
    mujoco.mj_step1(m, d)
    W[:] = 0.0
    suc.accumulate(m, d, W)
    if extra is not None:
        extra(W)
    d.xfrc_applied[:] = W
    mujoco.mj_step2(m, d)
    if (k + 1) % 5 == 0:
        net.advance(0.01)
        suc.update_networks()


def settle(cand: dict, r_face: float, t_settle: float = 2.0) -> dict:
    """Settle the candidate's sheet on the frame with r_face; dict(A_sealed, hold, Pc)."""
    xml, excl, half, R = sheet_xml(cand)
    m = item_model(xml, excl)
    d = mujoco.MjData(m)
    net = make_network(r_face, int(cand["n_inlets"]), float(cand.get("flow_cfm", 40.0)))
    suc, bodies = make_suction(m, net, half)
    W = np.zeros((m.nbody, 6))
    k = 0
    while k * m.opt.timestep < t_settle:
        step_suction(m, d, suc, net, W, k)
        k += 1
    return dict(A_sealed=float(suc.covered["g0"]), hold=float(suc.hold_force(bodies)), Pc=float(net.Pc))


# ------------------------------------------------------------------------------------------------ pull directions
def pull_dir(kind: str, R) -> np.ndarray:
    """Unit pull direction in the world; R = sheet_frame.  See the module doc ('tiltNN' / 'drag' are
    aliases of the _e0 ones)."""
    n = np.array([1.0, 0.0, 0.0])
    c = -np.asarray(R)[:, 0]
    if kind == "normal":
        a = 0.0
    elif kind in ("drag", "drag_e0"):
        a = 80.0
    elif kind == "drag_e4":
        a, c = 80.0, -c
    elif kind == "out45":
        a = 45.0
    elif kind.startswith("tilt"):
        body = kind[4:]
        if body.endswith("_e4"):
            a, c = float(body[:-3]), -c
        elif body.endswith("_e0"):
            a = float(body[:-3])
        else:
            a = float(body)
    else:
        raise ValueError(kind)
    a = math.radians(a)
    u = math.cos(a) * n + math.sin(a) * c
    return u / np.linalg.norm(u)


def loop_site_ab(cand: dict) -> np.ndarray:
    """(a, b) of the loop (strip 0 centre) in the face plane before settling."""
    R = sheet_frame(float(cand["theta_deg"]))
    sw = float(cand["size"][0]) / int(cand.get("n_strips", 5))
    off = cand.get("offset_m", (0.0, 0.0))
    c = np.array([0.0, float(off[0]), float(off[1])]) - R[:, 0] * sw * 2
    return c[1:]


def loop_over_opening(cand: dict) -> bool:
    a, b = loop_site_ab(cand)
    return bool(abs(a) <= OPEN_HALF[0] and abs(b) <= OPEN_HALF[1])
