"""The coupling of the humanoid assembly cell: one state machine per interface.

New file of this task. It takes the place of the precedent cells' couplings (arm/assembly_car/coupling.py and
arm/assembly_rov/coupling.py, vendored, unchanged), whose state names it keeps, and drops what this task's design
forbids: the precedents capture a unit with a SOFT WELD that holds it on the feature and PULLS it home; here ENGAGED
applies no force of any kind, nothing is held and nothing is pulled ("no hold,
no pull"). The station fastens a module only once it is held at rest on its own seat: in the real cell the
robot (or gravity) holds it there and the screws are driven; LOCKED is the screws.

Interfaces (three seat classes, one module each; every number from cell_config.json "coupling"):
  leg   the hip-roll bracket's flat flange with its shallow pilot on a hip-pitch OUTPUT ROTOR, 6 bolt positions (60 deg)
  arm   the shoulder ring in the torso shell's socket (a barrel), 12 bolt positions (30 deg)
  head  the head bracket on the neck-yaw output rotor's boss, 6 bolt positions (60 deg; 5 of 6 holes used)

Frames (metres): the interface frame of the module = `frame_in_part` (4 x 4, in the module's flange body: origin =
the bolt circle's centre on the seating face, z = the outward interface axis, from the carrier into the module,
x = towards bolt 0); the SEAT = `seat_in_carrier` (4 x 4, in the carrier body: where that frame lies at the design
pose). Every tick the coupling reads E = seat^-1 . frame (both in the world, from the carrier's and the flange's body
poses after the tick's mj_forward):
  axial    E's translation along z (signed: + = short of home, outboard; - = past home, into the seat)
  lateral  E's translation across z (the distance of the frame's origin from the seat's axis line)
  tilt     the angle between the two z axes
  clock    E's twist about z (the swing-twist decomposition); the nearest bolt position k = round(clock / pitch)
           mod positions, the phase residual = clock - k x pitch (wrapped)
  speeds   the flange body's velocity relative to the carrier body AT THE INTERFACE POINT (the frame's origin): the
           difference of the two rigid-body velocities at that point (|v|), and of the two angular velocities (|w|)
For a rotor seat the carrier is the ROTOR body, so the phase is read against the rotor wherever the rotor stands.

States (the precedent's names): WAITING (before the first tick, or a required interface not achieved; `requires` is
empty for every mate of this task) -> FREE -> ENGAGED (the candidate is inside the ENTER window: |axial|, lateral,
tilt and the phase residual of one bolt position; NO force, no record the agent can read) -> HOLDING (the LOCK
CONDITION holds: inside the enter window with |axial| and tilt inside the LOCK window, the phase within the enter
window of the bound bolt position, relative speed < rel_speed_max and relative angular speed < rel_ang_speed_max)
-> LOCKED (after hold_ticks consecutive control ticks of the condition: the flange body is welded to the carrier body
exactly where it stopped, the weld's 11 data entries written explicitly) or LOCKED_OFF (the same lock at any bolt
position but the design one: it is a lock, it scores nothing, it cannot be undone). VOID: the module was scrapped
(`scrap()`, called by the safety monitor's R5): its mate ends for good (a weld already made stays; it counts 0).
An ENGAGED or HOLDING candidate that leaves release_factor x the enter window is FREE again. A module that does not
belong to an interface is never a candidate there (the mirrored units): each interface evaluates its own module only.

Timing: the coupling ticks after the safety monitor in the cell's control tick (cell.hooks), on the state after the
tick's mj_forward. The lock is made at the end of the hold_ticks-th consecutive tick whose end state satisfies the
condition (0.50 s after the start of the first such tick); the weld acts from the next tick on.

What the coupling writes: nothing in mjModel / mjData before a lock (no weld, no contact bit, no force; the tests
prove it field by field); at a lock, the interface's weld row of eq_data and its eq_active. Its events stay in its
own record (`events`), never in the cell's step() events: nothing tells a program that a part was captured.

The hidden mating graph (the kit's graph/mating_graph.json, schema humanoid-asm-graph/1) is read by ONE function,
`interfaces_from_graph`; a change of the graph's format is a change of that function only.

How the cell uses it: `ifs = interfaces_from_graph(graph, design_phase={...}, body_name=<the adapter's body
naming>)`; the scene data's "welds" = `weld_records(ifs)`; after scene.build_cell: `cp = Coupling(cell, ifs).attach()`
(its tick into cell.hooks, the monitor's locks / on_scrap); at the end `cp.record()`, `cp.metrics()`, `cp.score()`.
A far candidate (farther from its seat than the window can reach) is FREE without the full measurement (the
precedents' pair-distance pre-check; exact: inside a window implies a distance <= hypot(axial, lateral) of it).
"""
import copy
import math

import numpy as np

__all__ = ["STATES", "TERMINAL", "CLASSES", "SCHEMA", "CouplingError", "coupling_config", "check_coupling_config",
           "interfaces_from_graph", "weld_name", "weld_records", "Coupling", "measure_pose", "capture_table"]

STATES = ("WAITING", "FREE", "ENGAGED", "HOLDING", "LOCKED", "LOCKED_OFF", "VOID")
TERMINAL = ("LOCKED", "LOCKED_OFF", "VOID")
CLASSES = ("leg", "arm", "head")
SCHEMA = "humanoid-coupling-config/1"
GRAPH_SCHEMA = "humanoid-asm-graph/1"
TICK_S = 0.01          # the control tick the latch is defined for (the frozen latch.tick_ms 10); checked, not used
CLASS_WORDS = {        # what the cell manual may say about each class (no design pose, no phase, no unit name)
    "leg": "a module's flat flange with a shallow pilot on an output rotor",
    "arm": "a module's ring in a socket (a barrel in the torso shell)",
    "head": "a module's bracket on the boss of an output rotor",
}


class CouplingError(ValueError):
    pass


# ====================================================================================================== small math
def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _wrap(a):
    """an angle in (-pi, pi]"""
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def _quat_wxyz(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).reshape(9))
    return q


def _body_T(d, b):
    T = np.eye(4)
    T[:3, :3] = np.asarray(d.xmat[b]).reshape(3, 3)
    T[:3, 3] = d.xpos[b]
    return T


def measure_pose(E, pitch):
    """(axial, lateral, tilt, clock, k_near, residual) of E = the part's interface frame in the seat frame (4 x 4, m);
    pitch in rad. tilt = the angle between the z axes; clock = the twist about z (2 atan2(qz, qw), wrapped)."""
    t = E[:3, 3]
    axial = float(t[2])
    lateral = float(math.hypot(t[0], t[1]))
    tilt = float(math.atan2(math.hypot(E[0, 2], E[1, 2]), E[2, 2]))
    q = _quat_wxyz(E[:3, :3])
    clock = _wrap(2.0 * math.atan2(float(q[3]), float(q[0])))
    k_near = int(round(clock / pitch))
    res = _wrap(clock - k_near * pitch)
    return axial, lateral, tilt, clock, k_near, res


# ====================================================================================================== configuration
def _num(v, what, positive=True, allow_none=False):
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise CouplingError("coupling: %s must be a finite number, got %r" % (what, v))
    if positive and not float(v) > 0.0:
        raise CouplingError("coupling: %s must be > 0, got %r" % (what, v))
    return float(v)


def check_coupling_config(cfg):
    """the configuration's "coupling" block, checked and converted to SI (m, rad, s): {"latch": {...}, "classes":
    {cls: {...}}, "interfaces": {iface: cls}, "weights": {iface: w}}. Refuses a pull, a hold, a mirrored candidate,
    a lock window wider than its enter window, a release factor below 1, a tick that is not the latch's 10 ms."""
    if "coupling" not in cfg:
        raise CouplingError("the configuration has no 'coupling' block")
    C = cfg["coupling"]
    if C.get("schema") != SCHEMA:
        raise CouplingError("coupling: schema %r, this coupling reads %s" % (C.get("schema"), SCHEMA))
    P = cfg["physics"]
    tick = float(P["timestep"]) * int(P["tick_steps"])
    L = C["latch"]
    if abs(tick - TICK_S) > 1e-12 or abs(_num(L["tick_ms"], "latch.tick_ms") * 1e-3 - TICK_S) > 1e-12:
        raise CouplingError("coupling: the latch is defined for a 10 ms control tick (cell tick %r s)" % tick)
    if L.get("pull") is not False or L.get("hold") != "none" or L.get("mirrored_unit") != "never a candidate":
        raise CouplingError("coupling: no pull, no hold, a mirrored unit is never a candidate")
    if L.get("force_before_lock") != "none":
        raise CouplingError("coupling: ENGAGED applies no force")
    hold_ticks = int(L["hold_ticks"])
    if hold_ticks < 1 or hold_ticks != L["hold_ticks"]:
        raise CouplingError("coupling: latch.hold_ticks must be a whole number >= 1")
    out = {"latch": {"hold_ticks": hold_ticks, "tick_s": TICK_S,
                     "rel_speed_max": _num(L["rel_speed_max_m_s"], "latch.rel_speed_max_m_s"),
                     "rel_ang_speed_max": math.radians(_num(L["rel_ang_speed_max_deg_s"], "latch.rel_ang_speed_max_deg_s")),
                     "release_factor": _num(L["release_factor"], "latch.release_factor")}}
    if out["latch"]["release_factor"] < 1.0:
        raise CouplingError("coupling: latch.release_factor must be >= 1")
    out["classes"] = {}
    for cls in CLASSES:
        if cls not in C["classes"]:
            raise CouplingError("coupling: class %r missing" % cls)
        W = C["classes"][cls]
        en, lk = W["enter"], W["lock"]
        n = int(W["positions"])
        if n < 1 or n != W["positions"]:
            raise CouplingError("coupling: %s.positions must be a whole number >= 1" % cls)
        x = {"positions": n, "pitch": 2.0 * math.pi / n,
             "enter_axial": _num(en["axial_mm"], cls + ".enter.axial_mm") * 1e-3,
             "enter_lateral": (None if en.get("lateral_mm") is None else _num(en["lateral_mm"], cls + ".enter.lateral_mm") * 1e-3),
             "enter_tilt": math.radians(_num(en["tilt_deg"], cls + ".enter.tilt_deg")),
             "enter_clock": math.radians(_num(en["clock_deg"], cls + ".enter.clock_deg")),
             "lock_axial": _num(lk["axial_mm"], cls + ".lock.axial_mm") * 1e-3,
             "lock_tilt": math.radians(_num(lk["tilt_deg"], cls + ".lock.tilt_deg"))}
        if "lateral_mm" not in en:
            raise CouplingError("coupling: %s.enter.lateral_mm must be given (a number, or null = not checked)" % cls)
        if x["lock_axial"] > x["enter_axial"] or x["lock_tilt"] > x["enter_tilt"]:
            raise CouplingError("coupling: %s lock window wider than its enter window" % cls)
        if out["latch"]["release_factor"] * x["enter_clock"] >= x["pitch"] / 2.0:
            raise CouplingError("coupling: %s release window of the phase reaches the next bolt position" % cls)
        out["classes"][cls] = x
    out["interfaces"] = {}
    for iface, cls in C["interfaces"].items():
        if cls not in CLASSES:
            raise CouplingError("coupling: interface %s has class %r" % (iface, cls))
        out["interfaces"][iface] = cls
    out["weights"] = {k: _num(v, "weights." + k) for k, v in C["weights"].items()}
    if set(out["weights"]) != set(out["interfaces"]):
        raise CouplingError("coupling: weights and interfaces name different mates")
    return out


def coupling_config(cfg=None):
    """the checked coupling block of the cell's configuration (default: config.load_config())"""
    if cfg is None:
        from . import config as CF
        cfg = CF.load_config()
    return check_coupling_config(cfg)


# ====================================================================================================== the graph
def _T_m(M_mm):
    T = np.array(M_mm, float).reshape(4, 4).copy()
    T[:3, 3] *= 1e-3
    R = T[:3, :3]
    if np.max(np.abs(R @ R.T - np.eye(3))) > 1e-5 or abs(np.linalg.det(R) - 1.0) > 1e-5:
        raise CouplingError("graph: a frame is not a rotation")
    # re-orthonormalise the 6-decimal rotation of the graph (SVD), so that a frame read at the design pose gives 0
    U, _s, Vt = np.linalg.svd(R)
    T[:3, :3] = U @ Vt
    return T


def interfaces_from_graph(graph, design_phase=None, body_name=None, cfg=None):
    """THE reader of the hidden mating graph (schema humanoid-asm-graph/1, graph/mating_graph.json): the interface
    records of the coupling, one per mate, in the graph's order of reference. A change of the graph's format changes this
    function only.

    design_phase: {mate: option name} for mates whose design phase is still open in the graph (no option
    chosen); a mate with a single option takes it. body_name: f(graph body name) -> the cell's body name (default
    identity; the kit-to-cell adapter knows its naming). cfg: the configuration (default load_config()): the
    graph's copy of the frozen windows must equal the configuration's (two copies of one rule must not drift; the
    configuration is the one source)."""
    if graph.get("schema") != GRAPH_SCHEMA:
        raise CouplingError("graph: schema %r, this reader reads %s" % (graph.get("schema"), GRAPH_SCHEMA))
    if cfg is None:
        from . import config as CF
        cfg = CF.load_config()
    C = check_coupling_config(cfg)
    bn = body_name or (lambda s: s)
    choice = dict(design_phase or {})
    order = list(graph.get("order_of_reference") or list(graph["mates"]))
    out = []
    for mid in order:
        g = graph["mates"][mid]
        cls = g["kind"]
        if cls not in CLASSES or C["interfaces"].get(mid) != cls:
            raise CouplingError("graph: mate %s kind %r, the configuration says %r" % (mid, cls, C["interfaces"].get(mid)))
        W = C["classes"][cls]
        if int(g["bolt_circle"]["positions"]) != W["positions"]:
            raise CouplingError("graph: mate %s has %s bolt positions, the configuration %d" % (mid, g["bolt_circle"]["positions"], W["positions"]))
        wp = g.get("windows_proposed") or {}
        cw = cfg["coupling"]["classes"][cls]
        for part in ("enter", "lock"):
            for k, v in (wp.get(part) or {}).items():
                if abs(float(cw[part][k]) - float(v)) > 1e-12:
                    raise CouplingError("graph: mate %s windows_proposed.%s.%s = %r, the configuration %r" % (mid, part, k, v, cw[part][k]))
        dp = g["design_phase"]
        opts = dp["options"]
        if mid in choice:
            if choice[mid] not in opts:
                raise CouplingError("graph: mate %s has no design-phase option %r (%s)" % (mid, choice[mid], ", ".join(opts)))
            on = choice[mid]
        elif len(opts) == 1:
            on = next(iter(opts))
        else:
            raise CouplingError("graph: mate %s design phase is open (%s): pass design_phase={%r: <one of %s>}"
                                % (mid, dp.get("status"), mid, ", ".join(opts)))
        o = opts[on]
        Fp = _T_m(o["part_frame_in_part_body"])
        Tdes = _T_m(o["T_design_part_body_in_carrier_body"])
        design = [c for c in o["candidates"] if c.get("scores")]
        if len(design) != 1 or len(o["candidates"]) != W["positions"]:
            raise CouplingError("graph: mate %s option %s: %d scoring candidates of %d" % (mid, on, len(design), len(o["candidates"])))
        out.append({"id": mid, "class": cls, "module": g["part_unit"], "part_body": bn(g["part_body"]),
                    "carrier_body": bn(g["carrier_body"]), "carrier_unit": g["carrier_unit"],
                    "seat_in_carrier": (Tdes @ Fp).tolist(), "frame_in_part": Fp.tolist(),
                    "design_k": int(design[0]["k"]) % W["positions"],
                    "requires": list((graph.get("requires") or {}).get(mid) or []),
                    "mirror_of": (g.get("mirrored_unit") or {}).get("mirror_of"),
                    "source": {"graph_schema": graph["schema"], "design_phase_option": on,
                               "rotor_start_q_rad": o.get("rotor_start_q_rad")}})
    return out


# ====================================================================================================== welds
def weld_name(iface_id):
    return "lock__%s" % iface_id


def weld_records(interfaces):
    """the scene data's WELD records (scene.py's data contract): one inactive weld per interface, body1 = the carrier body,
    body2 = the module's flange body; the builder gives it the configuration's weld impedance; the coupling writes
    all 11 data entries at the lock (a weld without explicit data compiles with a wrong anchor)"""
    return [{"name": weld_name(x["id"]), "body1": x["carrier_body"], "body2": x["part_body"], "active": False}
            for x in interfaces]


# ====================================================================================================== the machine
class Coupling(object):
    """The coupling of one episode's cell. `cell` = a built scene.Cell whose data carried `weld_records(interfaces)`;
    `interfaces` = records as `interfaces_from_graph` returns them (or built directly by a test: id, class, module,
    part_body, carrier_body, seat_in_carrier, frame_in_part, design_k, requires). `attach()` plugs it into the cell:
    its tick into cell.hooks (after the safety monitor), the monitor's locks() and on_scrap()."""

    def __init__(self, cell, interfaces, cfg=None):
        import mujoco
        self.mj = mujoco
        self.cell = cell
        self.m, self.d = cell.m, cell.d
        self.cfg = copy.deepcopy(cfg if cfg is not None else cell.cfg)
        self.C = check_coupling_config(self.cfg)
        L = self.C["latch"]
        self.hold_ticks = L["hold_ticks"]
        self.v_max, self.w_max, self.f = L["rel_speed_max"], L["rel_ang_speed_max"], L["release_factor"]
        if abs(float(cell.tick_s) - TICK_S) > 1e-12:
            raise CouplingError("coupling: the cell's tick is %r s, the latch's 10 ms" % cell.tick_s)
        m = self.m
        idx = cell.index
        self.ifaces = []
        seen_mod = set()
        for rec in interfaces:
            x = dict(rec)
            iid = x["id"]
            if x["class"] not in CLASSES:
                raise CouplingError("interface %s: class %r" % (iid, x["class"]))
            if self.C["interfaces"].get(iid) not in (None, x["class"]):
                raise CouplingError("interface %s: class %r, the configuration says %r" % (iid, x["class"], self.C["interfaces"][iid]))
            if x["module"] in seen_mod:
                raise CouplingError("module %s has two interfaces" % x["module"])
            seen_mod.add(x["module"])
            W = self.C["classes"][x["class"]]
            x["W"] = W
            x["bp"] = int(m.body(x["part_body"]).id)
            x["bc"] = int(m.body(x["carrier_body"]).id)
            if idx.body_owner[x["bp"]] != x["module"]:
                raise CouplingError("interface %s: body %s belongs to %r, not to module %s" % (iid, x["part_body"], idx.body_owner[x["bp"]], x["module"]))
            x["carrier_owner"] = idx.body_owner[x["bc"]]
            if x["carrier_owner"] is None or x["carrier_owner"] == x["module"]:
                raise CouplingError("interface %s: carrier body %s has owner %r" % (iid, x["carrier_body"], x["carrier_owner"]))
            x["S"] = np.array(x["seat_in_carrier"], float).reshape(4, 4)
            x["F"] = np.array(x["frame_in_part"], float).reshape(4, 4)
            for k in ("S", "F"):
                R = x[k][:3, :3]
                if np.max(np.abs(R @ R.T - np.eye(3))) > 1e-9:
                    raise CouplingError("interface %s: %s is not a rotation" % (iid, k))
            x["S_inv"] = _inv(x["S"])
            # the cheap pre-check (the precedents' pair distance): inside factor x the enter window implies a distance
            # between the interface point and the seat's origin of at most factor x hypot(axial, lateral); none when the
            # class has no lateral term
            x["reach"] = None if W["enter_lateral"] is None else math.hypot(W["enter_axial"], W["enter_lateral"])
            x["design_k"] = int(x.get("design_k", 0)) % W["positions"]
            x["requires"] = list(x.get("requires") or [])
            x["weight"] = float(self.C["weights"].get(iid, 0.0))
            e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name(iid))
            if e < 0:
                raise CouplingError("interface %s: no weld %s in the cell (put weld_records(...) in the scene data)" % (iid, weld_name(iid)))
            if (int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_WELD) or int(m.eq_obj1id[e]) != x["bc"] or int(m.eq_obj2id[e]) != x["bp"]
                    or int(m.eq_objtype[e]) != int(mujoco.mjtObj.mjOBJ_BODY)):
                raise CouplingError("interface %s: equality %s is not the weld carrier -> flange" % (iid, weld_name(iid)))
            if int(self.d.eq_active[e]):
                raise CouplingError("interface %s: its weld is active at the start" % iid)
            x["eq"] = int(e)
            self.ifaces.append(x)
        self.by_id = {x["id"]: x for x in self.ifaces}
        self.by_module = {x["module"]: x for x in self.ifaces}
        for x in self.ifaces:
            for q in x["requires"]:
                if q not in self.by_id:
                    raise CouplingError("interface %s requires %s, which is not an interface" % (x["id"], q))
        self.total_w = float(sum(x["weight"] for x in self.ifaces))
        self.state = {x["id"]: "WAITING" for x in self.ifaces}
        self.engaged = {x["id"]: None for x in self.ifaces}         # the bound bolt position while engaged
        self.count = {x["id"]: 0 for x in self.ifaces}
        self.rec = {x["id"]: {"id": x["id"], "class": x["class"], "module": x["module"], "weight": x["weight"],
                              "locked": False, "design_ok": None, "k": None, "t_lock_s": None, "t_engaged_first_s": None,
                              "t_holding_start_s": None, "lock_errors": None, "void": None, "n_engaged": 0,
                              "n_released": 0} for x in self.ifaces}
        self.scrapped = {}
        self.events = []                       # (t, kind, interface or module, detail): the coupling's own record
        self.n_tick = 0
        self.last = {}                         # interface -> the last tick's measurement (diagnostics only)
        self.weld_stats = {x["id"]: {"F_max_N": 0.0, "T_max_Nm": 0.0} for x in self.ifaces}
        self._vb = np.zeros(6)
        self._vc = np.zeros(6)

    # ------------------------------------------------------------------ plug-ins
    def attach(self, cell=None):
        """the coupling's tick into cell.hooks (run after the safety monitor), the monitor's locks() / on_scrap()"""
        cell = cell or self.cell
        if self.tick in cell.hooks:
            raise CouplingError("coupling already attached")
        cell.hooks.append(self.tick)
        if cell.monitor is not None:
            cell.monitor.locks = self.locks
            cell.monitor.on_scrap = self.scrap
        cell.coupling = self
        return self

    def locks(self):
        """[(module, carrier owner)] of every weld this coupling made (the monitor's R2 excludes those pairs)"""
        return [(x["module"], x["carrier_owner"]) for x in self.ifaces if self.rec[x["id"]]["locked"]]

    # ------------------------------------------------------------------ measuring
    def measure(self, x):
        """the interface's errors now (module doc); reads mjData only"""
        d, mj = self.d, self.mj
        Tc, Tp = _body_T(d, x["bc"]), _body_T(d, x["bp"])
        E = x["S_inv"] @ _inv(Tc) @ Tp @ x["F"]
        W = x["W"]
        axial, lateral, tilt, clock, k_near, res = measure_pose(E, W["pitch"])
        kb = self.engaged[x["id"]]
        res_b = res if kb is None else _wrap(clock - kb * W["pitch"])
        p = Tp[:3, :3] @ x["F"][:3, 3] + Tp[:3, 3]                 # the interface point in the world
        # mjOBJ_BODY: [w, v] with v the velocity of the body's centre of mass (xipos), world orientation
        mj.mj_objectVelocity(self.m, d, mj.mjtObj.mjOBJ_BODY, x["bp"], self._vb, 0)
        mj.mj_objectVelocity(self.m, d, mj.mjtObj.mjOBJ_BODY, x["bc"], self._vc, 0)
        vp = self._vb[3:] + np.cross(self._vb[:3], p - d.xipos[x["bp"]])
        vc = self._vc[3:] + np.cross(self._vc[:3], p - d.xipos[x["bc"]])
        return {"axial": axial, "lateral": lateral, "tilt": tilt, "clock": clock, "k_near": k_near % W["positions"],
                "k_near_signed": k_near, "res": res, "res_bound": res_b,
                "v_rel": float(np.linalg.norm(vp - vc)), "w_rel": float(np.linalg.norm(self._vb[:3] - self._vc[:3]))}

    def _inside(self, x, r, factor, bound=False):
        """inside factor x the enter window; bound = the phase read against the bound bolt position (the release
        check of an engaged candidate), else against the nearest one"""
        W = x["W"]
        res = r["res_bound"] if bound else r["res"]
        lat_ok = W["enter_lateral"] is None or r["lateral"] <= factor * W["enter_lateral"]
        return (abs(r["axial"]) <= factor * W["enter_axial"] and lat_ok and r["tilt"] <= factor * W["enter_tilt"]
                and abs(res) <= factor * W["enter_clock"])

    def _lock_condition(self, x, r):
        W = x["W"]
        return (self._inside(x, r, 1.0) and r["k_near"] == self.engaged[x["id"]] and abs(r["axial"]) <= W["lock_axial"]
                and r["tilt"] <= W["lock_tilt"] and r["v_rel"] < self.v_max and r["w_rel"] < self.w_max)

    def _requires_met(self, x):
        return all(self.counted(q) for q in x["requires"])

    def _far(self, x, factor):
        """True when the interface point is farther from the seat's origin than factor x the enter window can reach
        (then it is outside that window for sure; the full measurement is skipped)"""
        if x["reach"] is None:
            return False
        d = self.d
        p = np.asarray(d.xmat[x["bp"]]).reshape(3, 3) @ x["F"][:3, 3] + d.xpos[x["bp"]]
        o = np.asarray(d.xmat[x["bc"]]).reshape(3, 3) @ x["S"][:3, 3] + d.xpos[x["bc"]]
        return float(np.linalg.norm(p - o)) > factor * x["reach"] * (1.0 + 1e-9) + 1e-12

    # ------------------------------------------------------------------ the tick
    def tick(self):
        """one control tick (after the cell's mj_forward and the safety monitor)"""
        self.n_tick += 1
        t = float(self.d.time)
        for x in self.ifaces:
            iid = x["id"]
            if self.state[iid] in TERMINAL:
                continue
            if not self._requires_met(x):
                if self.engaged[iid] is not None:
                    self._release(x, t, "a required interface is not achieved")
                self.state[iid] = "WAITING"
                continue
            if self._far(x, self.f if self.engaged[iid] is not None else 1.0):
                self.last[iid] = None
                if self.engaged[iid] is not None:
                    self._release(x, t, "left %.1f x the enter window" % self.f)
                self.state[iid] = "FREE"
                continue
            r = self.measure(x)
            self.last[iid] = r
            if self.engaged[iid] is not None and not self._inside(x, r, self.f, bound=True):
                self._release(x, t, "left %.1f x the enter window" % self.f)
            if self.engaged[iid] is None:
                if not self._inside(x, r, 1.0):
                    self.state[iid] = "FREE"
                    continue
                self.engaged[iid] = r["k_near"]
                rec = self.rec[iid]
                rec["n_engaged"] += 1
                if rec["t_engaged_first_s"] is None:
                    rec["t_engaged_first_s"] = t
                self.events.append((t, "ENGAGED", iid, {"k": r["k_near"]}))
            if self._lock_condition(x, r):
                if self.count[iid] == 0:
                    self.rec[iid]["t_holding_start_s"] = round(t - self.cell.tick_s, 9)
                self.count[iid] += 1
            else:
                self.count[iid] = 0
            self.state[iid] = "HOLDING" if self.count[iid] > 0 else "ENGAGED"
            if self.count[iid] >= self.hold_ticks:
                self._lock(x, r, t)
        self._track_welds()

    def _release(self, x, t, why):
        iid = x["id"]
        self.engaged[iid] = None
        self.count[iid] = 0
        self.rec[iid]["n_released"] += 1
        self.events.append((t, "RELEASED", iid, {"why": why}))

    def _lock(self, x, r, t):
        """weld the flange body to the carrier body exactly where it stopped: body1 = carrier, body2 = flange; the
        anchor = the interface point (data[0:3] in the flange's frame, data[3:6] the same point in the carrier's frame),
        data[6:10] = the flange's orientation in the carrier's frame, data[10] = torquescale 1"""
        iid = x["id"]
        m, d = self.m, self.d
        Tc, Tp = _body_T(d, x["bc"]), _body_T(d, x["bp"])
        Tpc = _inv(Tc) @ Tp
        a2 = x["F"][:3, 3]
        data = np.zeros(11)
        data[0:3] = a2
        data[3:6] = Tpc[:3, :3] @ a2 + Tpc[:3, 3]
        data[6:10] = _quat_wxyz(Tpc[:3, :3])
        data[10] = 1.0
        m.eq_data[x["eq"], :] = data
        d.eq_active[x["eq"]] = 1
        k = self.engaged[iid]
        ok = (k == x["design_k"])
        rec = self.rec[iid]
        rec.update({"locked": True, "design_ok": bool(ok), "k": int(k), "t_lock_s": t,
                    "lock_errors": {"axial_mm": r["axial"] * 1e3, "lateral_mm": r["lateral"] * 1e3,
                                    "tilt_deg": math.degrees(r["tilt"]), "phase_residual_deg": math.degrees(r["res"]),
                                    "rel_speed_mm_s": r["v_rel"] * 1e3, "rel_ang_speed_deg_s": math.degrees(r["w_rel"])},
                    "weld_data": [float(v) for v in data]})
        self.state[iid] = "LOCKED" if ok else "LOCKED_OFF"
        self.events.append((t, self.state[iid], iid, {"k": int(k), "design_k": int(x["design_k"])}))

    # ------------------------------------------------------------------ scrapping
    def scrap(self, module, t=None, why="scrapped"):
        """end a module's mate for good (the safety monitor's R5 calls it: on_scrap(module, t, why)): its interface
        becomes VOID (a weld already made stays where it is; the mate counts 0); the module is never a candidate again.
        Returns False if the module was scrapped before."""
        if module in self.scrapped:
            return False
        t = float(self.d.time) if t is None else float(t)
        self.scrapped[module] = {"t_s": t, "why": str(why)}
        x = self.by_module.get(module)
        self.events.append((t, "SCRAPPED", module, {"why": str(why)}))
        if x is not None:
            iid = x["id"]
            was = self.state[iid]
            self.engaged[iid] = None
            self.count[iid] = 0
            self.rec[iid]["void"] = "module scrapped (%s) while %s" % (why, was)
            self.state[iid] = "VOID"
            self.events.append((t, "VOID", iid, {"was": was}))
        return True

    # ------------------------------------------------------------------ the welds' forces (metrics; read only)
    def weld_forces(self):
        """{interface: {"F_N", "T_Nm"}} of every active weld of this coupling, from the last constraint solve"""
        mj, d = self.mj, self.d
        out = {}
        n = int(d.nefc)
        if not n:
            return out
        typ = np.asarray(d.efc_type[:n])
        ids = np.asarray(d.efc_id[:n])
        for x in self.ifaces:
            if not self.rec[x["id"]]["locked"]:
                continue
            rows = np.nonzero((typ == int(mj.mjtConstraint.mjCNSTR_EQUALITY)) & (ids == x["eq"]))[0]
            f = np.asarray(d.efc_force[rows], float)
            out[x["id"]] = {"F_N": float(np.linalg.norm(f[:3])) if len(f) >= 3 else 0.0,
                            "T_Nm": float(np.linalg.norm(f[3:6])) if len(f) >= 6 else 0.0}
        return out

    def _track_welds(self):
        for iid, w in self.weld_forces().items():
            s = self.weld_stats[iid]
            s["F_max_N"] = max(s["F_max_N"], w["F_N"])
            s["T_max_Nm"] = max(s["T_max_Nm"], w["T_Nm"])

    # ------------------------------------------------------------------ results
    def counted(self, iid):
        """the mate counts: LOCKED at its design bolt position, its module not scrapped"""
        r = self.rec[iid]
        return bool(r["locked"] and r["design_ok"] and r["void"] is None)

    def score(self):
        """the counted mates' weights / the total weight of the interfaces this coupling was given (the episode's scope;
        the episode's stage value before the safety factor; the grader owns the episode's formula)"""
        return sum(x["weight"] for x in self.ifaces if self.counted(x["id"])) / self.total_w if self.total_w > 0 else 0.0

    def record(self):
        out = []
        for x in self.ifaces:
            r = copy.deepcopy(self.rec[x["id"]])
            r["state"] = self.state[x["id"]]
            r["counted"] = self.counted(x["id"])
            r["weld"] = dict(self.weld_stats[x["id"]])
            out.append(r)
        return out

    def metrics(self):
        rec = [self.rec[x["id"]] for x in self.ifaces]
        return {"locked": sum(1 for r in rec if r["locked"] and r["design_ok"] and r["void"] is None),
                "locked_off": sum(1 for r in rec if r["locked"] and r["design_ok"] is False),
                "void": sum(1 for r in rec if r["void"] is not None),
                "scrapped_modules": sorted(self.scrapped), "weight_counted": float(sum(x["weight"] for x in self.ifaces if self.counted(x["id"]))),
                "weight_total": self.total_w, "score": self.score(), "ticks": self.n_tick,
                "weld_F_max_N": max([s["F_max_N"] for s in self.weld_stats.values()] or [0.0]),
                "weld_T_max_Nm": max([s["T_max_Nm"] for s in self.weld_stats.values()] or [0.0])}

    def capture_table(self):
        """the cell manual's fastening table ('the cell manual's table comes from coupling.capture_table()')"""
        return capture_table(self.cfg)


def _measured_rows(cfg):
    """the optional 'measured' part of the coupling block (the entry windows measured on the real seats, per seat
    class: the cell manual prints the measured entry windows): {class: {...}} and the holder's
    description; absent = {} (the table is then exactly the windows' table)"""
    M = (cfg.get("coupling") or {}).get("measured") or {}
    rows = M.get("classes") or {}
    for cls in rows:
        if cls not in CLASSES:
            raise CouplingError("coupling.measured: class %r" % cls)
    return rows, M.get("holder"), M.get("source")


def capture_table(cfg=None):
    """the table of windows the cell manual prints, generated from the configuration alone (mm, deg, s; one row per
    seat class): no design pose, no design phase, no module or mate name; with the entry windows measured on the real
    seats when the configuration's coupling block carries them ('measured')"""
    if cfg is None:
        from . import config as CF
        cfg = CF.load_config()
    C = check_coupling_config(cfg)
    measured, holder, msource = _measured_rows(cfg)
    L = C["latch"]
    f = L["release_factor"]
    rows = []
    for cls in CLASSES:
        W = C["classes"][cls]
        lat = None if W["enter_lateral"] is None else W["enter_lateral"] * 1e3
        rows.append({
            "seat_class": cls, "seat": CLASS_WORDS[cls], "bolt_positions": W["positions"],
            "bolt_pitch_deg": round(math.degrees(W["pitch"]), 9),
            "enter_window": {"axial_mm": round(W["enter_axial"] * 1e3, 9), "lateral_mm": None if lat is None else round(lat, 9),
                             "tilt_deg": round(math.degrees(W["enter_tilt"]), 9), "phase_deg": round(math.degrees(W["enter_clock"]), 9)},
            "release_window": {"factor": f, "axial_mm": round(f * W["enter_axial"] * 1e3, 9),
                               "lateral_mm": None if lat is None else round(f * lat, 9),
                               "tilt_deg": round(f * math.degrees(W["enter_tilt"]), 9), "phase_deg": round(f * math.degrees(W["enter_clock"]), 9)},
            "lock_condition": {"axial_mm": round(W["lock_axial"] * 1e3, 9), "tilt_deg": round(math.degrees(W["lock_tilt"]), 9),
                               "lateral_mm": None if lat is None else round(lat, 9), "phase_deg": round(math.degrees(W["enter_clock"]), 9),
                               "rel_speed_mm_s": round(L["rel_speed_max"] * 1e3, 9),
                               "rel_ang_speed_deg_s": round(math.degrees(L["rel_ang_speed_max"]), 9),
                               "ticks": L["hold_ticks"], "tick_s": L["tick_s"], "duration_s": round(L["hold_ticks"] * L["tick_s"], 9),
                               "speeds_measured": "the module's flange relative to its carrier, at the interface point (the bolt "
                                                  "circle's centre on the seating face); on an output rotor the carrier is the rotor"},
            "force_before_lock": "none", "pull": "none", "hold": "none",
            "candidates": "only the module that belongs to the interface; a mirrored module is never fastened there",
            "at_the_lock": "fastened exactly where it is; not moved to a nominal pose",
            "off_design_position": "fastened for good at a bolt position that does not count; it counts nothing and cannot be undone",
            "scrapped_module": "a module that has touched the floor is never fastened again; its mate counts nothing"})
        if cls in measured:
            rows[-1]["measured_on_the_real_seats"] = copy.deepcopy(measured[cls])
    out = {"schema": "humanoid-capture-table/1", "units": {"length": "mm", "angle": "deg", "time": "s", "speed": "mm/s, deg/s"},
           "rows": rows}
    if measured:
        out["measured_with"] = {"holder": holder, "source": msource}
    return out
