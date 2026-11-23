"""``ParcelCell``: the dual-arm platform at a conveyor carrying a parcel stream.

Compiled once at construction; ``reset(shift)`` and every spawn only write
model / data fields. One ``step`` is one 20 ms control tick of ten 2 ms
physics steps.

    cell = ParcelCell()
    cell.reset(shift)
    while not cell.done:
        cell.step(q_target14, grip2)

Tick order: (1) belt command for the current shift time, slab wrap, idle
slots pinned in the parking area, due arrivals spawned at the entry (FIFO; an
arrival waits while the entry is occupied or no slot is free); (2) arm and
gripper commands (through the command gate, when one is set: a protective
stop replaces them); (3) ten physics steps; (4) truth bookkeeping: line
crossings, falls, exits (parcels past the exit line or fallen are removed);
(5) the tick listeners (the task cell's safety watch, sensors and record),
in the order they were added. Whoever calls ``step`` passes through (2) and
(5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import _geom
from .robot import (GRIPPER, JOINT_NAMES, SIDES, build_robot_spec, home_qpos)
from .sensors import (DEFAULT_CAMERAS, CameraRig, GripperState, JointState)
from .station import (BELT_HARDWARE, BELT_PROBES, STRUCTURE, WRAP_PERIOD,
                      StationConfig, add_station, belt_ctrl_for_speed)
from .stream import Arrival, Parcel, Shift

__all__ = ["CellConfig", "ParcelCell", "CONTROL_DT", "PHYSICS_DT",
           "apply_solver"]

PHYSICS_DT = 0.002
CONTROL_DT = 0.02
N_SUBSTEPS = 10
PARCEL_RGBA = (0.72, 0.55, 0.38, 1.0)          # kraft
_PLACEHOLDER_HALF = (0.12, 0.09, 0.045)        # compile-time slot (soft topology)


_INTEGRATORS = {"euler": mujoco.mjtIntegrator.mjINT_EULER,
                "rk4": mujoco.mjtIntegrator.mjINT_RK4,
                "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST}
_CONES = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
          "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}


def apply_solver(m: mujoco.MjModel, opts: dict | None) -> None:
    """Write solver options into a compiled model (None: keep the model's)."""
    for k, v in (opts or {}).items():
        if k == "integrator":
            m.opt.integrator = _INTEGRATORS[str(v)]
        elif k == "cone":
            m.opt.cone = _CONES[str(v)]
        elif k in ("impratio", "tolerance", "ls_tolerance", "noslip_tolerance"):
            setattr(m.opt, k, float(v))
        elif k in ("iterations", "ls_iterations", "noslip_iterations"):
            setattr(m.opt, k, int(v))
        else:
            raise ValueError(f"unknown solver option {k!r}")


def _park_pos(i: int) -> tuple:
    """Parking spot of idle slot i: below the floor box (no contact,
    hidden from the cameras by the floor)."""
    return (-2.0 + 0.9 * i, -2.5, -0.8)


@dataclass(frozen=True)
class CellConfig:
    station: StationConfig = field(default_factory=StationConfig)
    n_slots: int = 6
    lines: tuple = (("scan", 0.80),)
    """Named lines across the belt (y positions); crossings are logged. The
    exit line (station.exit_y) is always logged as "exit"."""
    cameras: tuple = DEFAULT_CAMERAS
    target_slew_rad_s: float | tuple | None = None
    """Optional per-joint limit on how fast the applied position target may
    move (rad/s, scalar or 14 values). None = off (the source)."""
    soft_finger_pads: bool = True
    """Compliant, high-friction finger pad boxes (solref 0.005 1, mu 1.8): the
    source's decided grasp model (binary full close on soft pad covers)."""
    belt_speed_compensation: bool = True
    home: tuple | dict | None = None
    """Home posture (14 values in JOINT_NAMES order, or a {name: angle}
    dict over the source home). None = the source's home clipped to the
    joint ranges. Note: at the source home the right wrist rests on the near
    lip and the left gripper grazes the slab underside (as in the source)."""
    settle_s: float = 0.3                 # arms settle at home in reset
    entry_clearance: float = 0.06         # extra clearance at the entry (m)
    # contact / solver settings: a held parcel crept and turned in the pads
    # with impratio 10 and soft pads; with the 2F-85 couplers locked
    # (``robot.LOCK_COUPLER``), torsional pad friction (condim 4) and
    # impratio 100 it holds
    pad_solref: float = 0.005
    pad_condim: int = 4
    pad_friction: float = 1.8
    impratio: float = 100.0
    noslip_iterations: int = 0
    lock_couplers: bool | None = None
    """Lock the 2F-85 coupler joints (None = ``robot.LOCK_COUPLER``)."""
    robot_mounts: dict | None = None
    """Robot-mounted hardware (``robot.add_mounts``): the head-camera
    housing, the wrist cameras' mounts and housings. None = none."""
    monitor_substeps: bool = False
    """Instrument every 2 ms physics step: peak contact forces of the robot
    on station structure, on the moving belt surface, arm on arm, arm on the
    column, palm on parcel (``parcelcell.monitor.ContactMonitor``). Off by
    default (cost about 5-10 % of the physics time)."""
    solver: dict | None = None
    """Solver options written into the compiled model's ``opt`` after the
    fields above (keys:
    integrator, cone, impratio, iterations, ls_iterations,
    noslip_iterations, tolerance). None = the robot model file's options."""
    finger_pad: dict | None = None
    """Overrides of the finger pad values {"solref": [t, d], "condim": n,
    "friction": mu} (used when ``soft_finger_pads``). None = the fields
    pad_solref (damping ratio 1), pad_condim and pad_friction."""
    parcel_contact: dict | None = None
    """Overrides of the parcel models' contact values: {"rigid": {solimp,
    solref}, "soft_geom": {...}, "soft_pad": {...}} (see ``_geom``)."""


@dataclass
class _Slot:
    active: bool = False
    arrival: int = -1          # index into shift.arrivals
    parcel: Parcel | None = None
    t_spawn: float = 0.0
    prev_y: float = 0.0
    uid: int = -1              # running parcel number within the shift


class ParcelCell:
    """The simulator. ``model`` / ``data`` are public for tooling but a
    controller under test should only see ``parcelcell.sensors`` quantities
    (``joint_state``, ``gripper_state``, ``render``)."""

    def __init__(self, config: CellConfig | None = None):
        self.cfg = cfg = config or CellConfig()
        self.station = st = cfg.station
        self.n_slots = int(cfg.n_slots)
        self.spec = self._build_spec()
        self.model = m = self.spec.compile()
        # rule 2 (station.py): the midphase BVH is baked at compile time
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
        assert abs(m.opt.timestep - PHYSICS_DT) < 1e-12
        self.data = mujoco.MjData(m)
        self.tick_listeners: list = []
        """Callables f(cell) run after every tick's physics and truth."""
        self.settle_listeners: list = []
        """Callables f(cell, k) run after each settle tick of ``reset``; k is
        the tick's index relative to the shift's start (the last settle tick
        is 0: its end state is the state at t = 0)."""
        self.command_gate = None
        """None or a callable gate(cell, q, grip) -> (q, grip): may replace
        the tick's commands (a protective stop holds them)."""
        self.n_ticks = 0

        name2id = mujoco.mj_name2id
        G, B, J, S, A = (mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY,
                         mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_SITE,
                         mujoco.mjtObj.mjOBJ_ACTUATOR)
        self._gid = lambda n: name2id(m, G, n)

        # ---- robot: joints / actuators strictly by name ----
        self.joint_names = JOINT_NAMES
        self._arm_aid = np.array([name2id(m, A, n) for n in JOINT_NAMES])
        self._arm_jid = np.array([name2id(m, J, n) for n in JOINT_NAMES])
        assert (self._arm_aid >= 0).all() and (self._arm_jid >= 0).all()
        for a, j in zip(self._arm_aid, self._arm_jid):
            assert m.actuator_trnid[a, 0] == j, "actuator / joint mismatch"
        self._arm_qadr = m.jnt_qposadr[self._arm_jid].copy()
        self._arm_dadr = m.jnt_dofadr[self._arm_jid].copy()
        self._ctrl_lo = m.actuator_ctrlrange[self._arm_aid, 0].copy()
        self._ctrl_hi = m.actuator_ctrlrange[self._arm_aid, 1].copy()
        self._grip_aid = np.array([name2id(m, A, f"{s[0]}_{GRIPPER.actuator}")
                                   for s in SIDES])
        self._grip_driver_qadr = np.array([
            m.jnt_qposadr[name2id(m, J, f"{s[0]}_{GRIPPER.driver_joint}")]
            for s in SIDES])
        assert (self._grip_aid >= 0).all()
        self.home = home_qpos(m)
        if isinstance(cfg.home, dict):
            for k, v in cfg.home.items():
                self.home[JOINT_NAMES.index(k)] = float(v)
        elif cfg.home is not None:
            self.home = np.asarray(cfg.home, float).reshape(14).copy()
        self._tcp_sid = {s: name2id(m, S, f"{s}_tcp") for s in SIDES}

        # finger pads (compliant, high friction)
        if cfg.soft_finger_pads:
            fp = dict({"solref": (float(cfg.pad_solref), 1.0),
                       "condim": int(cfg.pad_condim),
                       "friction": float(cfg.pad_friction)},
                      **(cfg.finger_pad or {}))
            for s in SIDES:
                for g in GRIPPER.pad_geoms:
                    gid = self._gid(f"{s[0]}_{g}")
                    assert gid >= 0
                    m.geom_solref[gid] = tuple(float(v) for v in fp["solref"])
                    m.geom_condim[gid] = int(fp["condim"])
                    m.geom_friction[gid][0] = float(fp["friction"])
        m.opt.impratio = float(cfg.impratio)
        m.opt.noslip_iterations = int(cfg.noslip_iterations)
        apply_solver(m, cfg.solver)

        # ---- robot geom set (collision geoms of Base_Link's subtree) ----
        base = name2id(m, B, "Base_Link")

        def under_base(b):
            while b > 0:
                if b == base:
                    return True
                b = m.body_parentid[b]
            return False

        self._robot_body = np.array([under_base(b) for b in range(m.nbody)])
        self.robot_geoms = np.array([
            g for g in range(m.ngeom)
            if self._robot_body[m.geom_bodyid[g]]
            and (m.geom_contype[g] or m.geom_conaffinity[g])])
        # robot collision geoms that can move (not welded to the world)
        self.robot_moving_geoms = np.array([
            g for g in self.robot_geoms if m.body_weldid[m.geom_bodyid[g]] != 0])

        # ---- belt ----
        self._belt_aid = name2id(m, A, "belt_drive")
        self._slab_qadr = m.jnt_qposadr[name2id(m, J, "belt_slide")]
        self._slab_dadr = m.jnt_dofadr[name2id(m, J, "belt_slide")]
        self._belt_surface = self._gid("belt_surface")
        self.belt_geoms = np.array([g for g in (self._gid(n) for n in
                                                BELT_HARDWARE) if g >= 0])
        self.structure_geoms = np.array([g for g in (self._gid(n) for n in
                                                     STRUCTURE + tuple(
                                                         str(h[0]) for h in
                                                         st.hardware))
                                         if g >= 0], dtype=int)
        self.probe_geoms = np.array([self._gid(n) for n in BELT_PROBES]) \
            if (st.frame_probes and not st.solid_frame) else np.zeros(0, int)

        # ---- parcel slots ----
        K = self.n_slots
        self._slot_bid = np.array([name2id(m, B, f"parcel{i}") for i in range(K)], dtype=int)
        self._slot_qadr = np.array([m.jnt_qposadr[name2id(m, J, f"parcel{i}_free")]
                                    for i in range(K)], dtype=int)
        self._slot_dadr = np.array([m.jnt_dofadr[name2id(m, J, f"parcel{i}_free")]
                                    for i in range(K)], dtype=int)
        self._slot_sid = np.array([name2id(m, S, f"parcel{i}_label_site")
                                   for i in range(K)], dtype=int)
        self._slot_g = [self._slot_geoms(i) for i in range(K)]
        for i in range(K):
            self._check_slot_geoms(i)
        self._post_compile()
        self.parcel_geom_slot = np.full(m.ngeom, -1, dtype=int)
        for i in range(K):
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == self._slot_bid[i]:
                    self.parcel_geom_slot[g] = i
        self._geom_is_robot = np.zeros(m.ngeom, bool)
        self._geom_is_robot[self.robot_geoms] = True
        self._geom_is_belt = np.zeros(m.ngeom, bool)
        self._geom_is_belt[self.belt_geoms] = True
        self._geom_is_probe = np.zeros(m.ngeom, bool)
        self._geom_is_probe[self.probe_geoms] = True

        # compile-time copy of every slot field written at run time: reset
        # restores it, so a replay never depends on earlier episodes
        self._slot_snapshot = self._snapshot_slot_fields()

        self.cameras = CameraRig(m, cfg.cameras)
        self.lines = tuple((str(n), float(y)) for n, y in cfg.lines) + \
            (("exit", float(st.exit_y)),)
        if cfg.target_slew_rad_s is None:
            self._slew = None
        else:
            self._slew = np.broadcast_to(
                np.asarray(cfg.target_slew_rad_s, float), (14,)).copy()
        # limiter instrumentation: ticks in which the slew limit changed the
        # applied target, and the largest excess (rad/s above the limit)
        self.n_slew_clip_ticks = 0
        self.slew_clip_max_excess = 0.0
        self.monitor = None
        if cfg.monitor_substeps:
            from .monitor import ContactMonitor
            self.monitor = ContactMonitor(self)

        from .truth import Truth
        self.truth = Truth(self)
        self.shift: Shift | None = None
        self.reset(Shift(arrivals=(), duration=0.0))

    # ================================================================
    # build
    # ================================================================
    def _build_spec(self) -> mujoco.MjSpec:
        spec = build_robot_spec(self.cfg.lock_couplers, self.cfg.robot_mounts)
        spec.modelname = "dual_arm_parcel_station"
        add_station(spec, self.station)
        w = spec.worldbody
        for i in range(self.n_slots):
            self._add_slot_body(w, i)
        return spec

    def _add_slot_body(self, w, i: int, pos=None) -> None:
        """A run-time-written parcel slot body (placeholder fields)."""
        ph = _PLACEHOLDER_HALF
        rgba_pad = [c * 0.95 for c in PARCEL_RGBA[:3]] + [1.0]
        body = w.add_body(name=f"parcel{i}",
                          pos=list(self._park(i) if pos is None else pos))
        body.add_freejoint(name=f"parcel{i}_free")
        # collision pair (soft topology is the superset: box + pad)
        body.add_geom(name=f"parcel{i}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[ph[0], ph[1], ph[2] * 0.85], mass=0.72,
                      rgba=list(PARCEL_RGBA), friction=[0.7, 0.01, 0.0003],
                      solimp=_geom.SOFT_GEOM["solimp"],
                      solref=_geom.SOFT_GEOM["solref"], group=3)
        body.add_geom(name=f"parcel{i}_pad",
                      type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                      size=[ph[0] * 0.92, ph[1] * 0.92, ph[2]], mass=0.08,
                      rgba=rgba_pad, friction=[0.7, 0.01, 0.0003],
                      solimp=_geom.SOFT_PAD["solimp"],
                      solref=_geom.SOFT_PAD["solref"], group=3)
        # visual twins
        body.add_geom(name=f"parcel{i}_geom_visual",
                      type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[ph[0], ph[1], ph[2] * 0.85], mass=0,
                      rgba=list(PARCEL_RGBA), contype=0, conaffinity=0,
                      group=0)
        body.add_geom(name=f"parcel{i}_pad_visual",
                      type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                      size=[ph[0] * 0.92, ph[1] * 0.92, ph[2]], mass=0,
                      rgba=rgba_pad, contype=0, conaffinity=0, group=0)
        # label: white sheet + two dark code blocks + centre site
        n = _geom.FACE_NORMALS[1]
        pos = n * (ph[2] + _geom.LABEL_T)
        body.add_geom(name=f"parcel{i}_label", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=pos.tolist(),
                      size=[ph[0] * 0.62, ph[1] * 0.62, _geom.LABEL_T],
                      rgba=[0.97, 0.97, 0.95, 1], contype=0, conaffinity=0,
                      mass=0, group=0)
        for nm in ("label_qr", "label_bar"):
            body.add_geom(name=f"parcel{i}_{nm}",
                          type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=(pos + [0.02, 0.02, 0]).tolist(),
                          size=[0.02, 0.02, _geom.LABEL_T],
                          rgba=[0.08, 0.08, 0.08, 1], contype=0,
                          conaffinity=0, mass=0, group=0)
        body.add_site(name=f"parcel{i}_label_site", pos=pos.tolist(),
                      zaxis=n.tolist(), size=[0.005] * 3, group=4)

    # ---- slot geoms (overridden by the per-shift world) ----
    _SLOT_GEOM_NAMES = ("geom", "pad", "geom_visual", "pad_visual", "label",
                        "label_qr", "label_bar")

    def _slot_geoms(self, i: int) -> dict:
        """{role: geom id} of slot i (the fields run-time writes touch)."""
        return {n: self._gid(f"parcel{i}_{n}") for n in self._SLOT_GEOM_NAMES}

    def _check_slot_geoms(self, i: int) -> None:
        m = self.model
        g = self._slot_g[i]
        assert min(g.values()) >= 0
        for n in ("label", "label_qr", "label_bar"):
            # rule 1: run-time geom_pos writes need sameframe != 1
            assert m.geom_sameframe[g[n]] != 1, n
        for n in ("geom", "pad", "geom_visual", "pad_visual"):
            # never inertia-framed: a com_offset write must not move them
            assert m.geom_sameframe[g[n]] in (0, 1, 3), n

    def _park(self, i: int) -> tuple:
        """Parking spot of idle slot i."""
        return _park_pos(i)

    def _post_compile(self) -> None:
        """Hook after slot discovery, before the compile-time snapshot."""
        return None

    _SLOT_GEOM_FIELDS = ("geom_size", "geom_pos", "geom_friction",
                         "geom_solref", "geom_solimp", "geom_rbound", "geom_aabb")
    _SLOT_BODY_FIELDS = ("body_mass", "body_inertia", "body_iquat", "body_ipos")
    _SLOT_SITE_FIELDS = ("site_pos", "site_quat")

    def _snapshot_slot_fields(self) -> dict:
        m = self.model
        gids = np.array(sorted({g for d in self._slot_g for g in d.values()}), dtype=int)
        snap = {"gids": gids}
        for f in self._SLOT_GEOM_FIELDS:
            snap[f] = getattr(m, f)[gids].copy()
        for f in self._SLOT_BODY_FIELDS:
            snap[f] = getattr(m, f)[self._slot_bid].copy()
        for f in self._SLOT_SITE_FIELDS:
            snap[f] = getattr(m, f)[self._slot_sid].copy()
        return snap

    def _restore_slot_fields(self) -> None:
        m, s = self.model, self._slot_snapshot
        for f in self._SLOT_GEOM_FIELDS:
            getattr(m, f)[s["gids"]] = s[f]
        for f in self._SLOT_BODY_FIELDS:
            getattr(m, f)[self._slot_bid] = s[f]
        for f in self._SLOT_SITE_FIELDS:
            getattr(m, f)[self._slot_sid] = s[f]

    # ================================================================
    # reset
    # ================================================================
    def reset(self, shift: Shift) -> None:
        """Start a shift: robot at home (grippers open), every slot parked,
        belt at the shift's initial speed, arms settled for ``settle_s``; the
        shift clock (``time``) then starts at 0."""
        m, d = self.model, self.data
        self.shift = shift
        self._restore_slot_fields()
        mujoco.mj_setConst(m, d)
        mujoco.mj_resetData(m, d)
        d.qpos[self._arm_qadr] = self.home
        self._q_applied = self.home.copy()
        d.ctrl[self._arm_aid] = self.home
        self._grip_cmd = np.zeros(2)
        d.ctrl[self._grip_aid] = GRIPPER.ctrl_open
        self.slots = [_Slot() for _ in range(self.n_slots)]
        self._pin_parked()
        self._next_arrival = 0
        self._n_spawned = 0
        self._belt_cmd = shift.speed_at(0.0)
        d.ctrl[self._belt_aid] = belt_ctrl_for_speed(
            self._belt_cmd, self.cfg.belt_speed_compensation)
        mujoco.mj_forward(m, d)
        n_settle = int(round(self.cfg.settle_s / CONTROL_DT))
        for k in range(n_settle):
            self._wrap_slab()
            self._pin_parked()
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(m, d)
            for fn in self.settle_listeners:
                fn(self, k + 1 - n_settle)      # tick index; 0 = the shift's t = 0
        d.time = 0.0
        self.n_ticks = 0
        self.n_slew_clip_ticks = 0
        self.slew_clip_max_excess = 0.0
        if getattr(self, "monitor", None) is not None:
            self.monitor.reset()
        self.truth.reset()

    # ================================================================
    # stepping
    # ================================================================
    @property
    def time(self) -> float:
        """Shift clock (s)."""
        return float(self.data.time)

    @property
    def done(self) -> bool:
        """The shift's duration is over and no parcel remains active or due."""
        return (self.time >= self.shift.duration - 1e-9
                and self._next_arrival >= len(self.shift.arrivals)
                and not any(s.active for s in self.slots))

    def step(self, q_target, grip=(0.0, 0.0)) -> None:
        """One 20 ms control tick. ``q_target``: 14 joint position targets
        (rad, ``JOINT_NAMES`` order) or a {joint name: target} dict (missing
        joints keep their previous target). ``grip``: 2 values, 0 = open ..
        1 = closed (left, right)."""
        m, d = self.model, self.data
        self._pre_tick()
        q = self._target_vector(q_target)
        g = np.asarray(grip, float).reshape(2)
        if self.command_gate is not None:
            q, g = self.command_gate(self, q, g)
        q = np.clip(q, self._ctrl_lo, self._ctrl_hi)
        if self._slew is not None:
            dq = self._slew * CONTROL_DT
            step = q - self._q_applied
            exc = np.abs(step) - dq
            if np.any(exc > 1e-12):
                self.n_slew_clip_ticks += 1
                self.slew_clip_max_excess = max(
                    self.slew_clip_max_excess,
                    float(np.max(exc / CONTROL_DT)))
            q = self._q_applied + np.clip(step, -dq, dq)
        self._q_applied = q
        d.ctrl[self._arm_aid] = q
        g = np.clip(g, 0.0, 1.0)
        self._grip_cmd = g
        d.ctrl[self._grip_aid] = GRIPPER.ctrl_open + \
            (GRIPPER.ctrl_close - GRIPPER.ctrl_open) * g
        mon = self.monitor
        if mon is None:
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(m, d)
        else:
            mon.begin_tick()
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(m, d)
                mon.substep()
            mon.end_tick(self.time)
        self.n_ticks += 1
        self.truth._post_tick()
        for fn in self.tick_listeners:
            fn(self)

    def hold(self) -> None:
        """One tick holding the current targets."""
        self.step(self._q_applied, self._grip_cmd)

    def _target_vector(self, q_target) -> np.ndarray:
        if isinstance(q_target, dict):
            q = self._q_applied.copy()
            for k, v in q_target.items():
                q[JOINT_NAMES.index(k)] = float(v)
            return q
        q = np.asarray(q_target, dtype=float).reshape(-1)
        if q.shape != (14,):
            raise ValueError("q_target must have 14 values (JOINT_NAMES order)")
        return q

    # ---- scheduler ----
    def _pre_tick(self) -> None:
        d = self.data
        self._belt_cmd = self.shift.speed_at(self.time)
        d.ctrl[self._belt_aid] = belt_ctrl_for_speed(
            self._belt_cmd, self.cfg.belt_speed_compensation)
        self._wrap_slab()
        self._pin_parked()
        arr = self.shift.arrivals
        while self._next_arrival < len(arr) and \
                arr[self._next_arrival].t <= self.time + 1e-9:
            if not self._try_spawn(self._next_arrival, arr[self._next_arrival]):
                break                      # FIFO: the head waits
            self._next_arrival += 1

    def _wrap_slab(self) -> None:
        """Slab slide position back by whole periods: the surface is uniform,
        so this is physically neutral and keeps the slab from running out."""
        d, P = self.data, WRAP_PERIOD
        a = self._slab_qadr
        d.qpos[a] = (d.qpos[a] + P / 2) % P - P / 2

    def _pin_parked(self) -> None:
        d = self.data
        for i, s in enumerate(self.slots):
            if s.active:
                continue
            q, v = self._slot_qadr[i], self._slot_dadr[i]
            d.qpos[q:q + 3] = self._park(i)
            d.qpos[q + 3:q + 7] = (1.0, 0.0, 0.0, 0.0)
            d.qvel[v:v + 6] = 0.0

    def entry_pose(self, p: Parcel) -> tuple[np.ndarray, np.ndarray]:
        st = self.station
        z0 = st.belt_height + p.size[2] / 2 + 0.002
        pos = np.array([st.belt_center_x + p.lateral, st.entry_y, z0])
        quat = np.array([math.cos(p.yaw / 2), 0.0, 0.0, math.sin(p.yaw / 2)])
        return pos, quat

    def _entry_clear(self, p: Parcel, pos: np.ndarray) -> bool:
        d = self.data
        for i, s in enumerate(self.slots):
            if not s.active:
                continue
            q = self._slot_qadr[i]
            dist = float(np.linalg.norm(d.qpos[q:q + 2] - pos[:2]))
            if dist < 0.5 * (s.parcel.diag + p.diag) + self.cfg.entry_clearance:
                return False
        return True

    def _try_spawn(self, k: int, a: Arrival) -> bool:
        free = [i for i, s in enumerate(self.slots) if not s.active]
        if not free:
            return False
        pos, quat = self.entry_pose(a.parcel)
        if not self._entry_clear(a.parcel, pos):
            return False
        i = free[0]
        self.place_parcel(i, a.parcel, pos, quat)
        s = self.slots[i]
        s.active, s.arrival, s.parcel = True, k, a.parcel
        s.t_spawn, s.prev_y = self.time, float(pos[1])
        s.uid, self._n_spawned = self._n_spawned, self._n_spawned + 1
        self.truth._on_spawn(i, k, a)
        return True

    def inject(self, parcel: Parcel, pos, quat=(1.0, 0.0, 0.0, 0.0),
               vel=None) -> int:
        """Put a parcel into a free slot at an arbitrary pose (tests, tooling,
        reference solutions); not part of a shift. Returns the slot. The
        spawn event carries ``"injected": True`` and arrival index -1."""
        free = [i for i, s in enumerate(self.slots) if not s.active]
        if not free:
            raise RuntimeError("no free parcel slot")
        i = free[0]
        pos = np.asarray(pos, float)
        self.place_parcel(i, parcel, pos, np.asarray(quat, float), vel)
        s = self.slots[i]
        s.active, s.arrival, s.parcel = True, -1, parcel
        s.t_spawn, s.prev_y = self.time, float(pos[1])
        s.uid, self._n_spawned = self._n_spawned, self._n_spawned + 1
        self.truth._on_spawn(i, -1, Arrival(t=self.time, parcel=parcel),
                             injected=True)
        return i

    # ---- slot field writes ----
    def apply_parcel_fields(self, i: int, p: Parcel) -> None:
        """Write parcel ``p`` into slot ``i``'s model fields (geometry,
        contact parameters, mass properties, label placement)."""
        m = self.model
        g = self._slot_g[i]
        f = _geom.parcel_fields(p.kind, p.size, p.mass, p.friction,
                                self.cfg.parcel_contact)
        for col, vis in (("geom", "geom_visual"), ("pad", "pad_visual")):
            sz = f["box_size"] if col == "geom" else f["pad_size"]
            m.geom_size[g[col]] = sz
            m.geom_size[g[vis]] = sz
        m.geom_friction[g["geom"]] = f["box_friction"]
        m.geom_solimp[g["geom"]] = f["box_solimp"]
        m.geom_solref[g["geom"]] = f["box_solref"]
        if f["pad_friction"] is not None:
            m.geom_friction[g["pad"]] = f["pad_friction"]
            m.geom_solimp[g["pad"]] = f["pad_solimp"]
            m.geom_solref[g["pad"]] = f["pad_solref"]
        b = self._slot_bid[i]
        m.body_mass[b] = f["mass"]
        # body-frame diagonal inertia + identity iquat: the compiler's
        # principal-axis permutation of the placeholder must not survive
        m.body_inertia[b] = f["inertia"]
        m.body_iquat[b] = (1.0, 0.0, 0.0, 0.0)
        m.body_ipos[b] = p.com_offset
        lay = _geom.label_layout(p.size, p.label_face, p.label_uv)
        for nm in ("label", "label_qr", "label_bar"):
            m.geom_pos[g[nm]], m.geom_size[g[nm]] = lay[nm]
        m.site_pos[self._slot_sid[i]] = lay["site_pos"]
        m.site_quat[self._slot_sid[i]] = lay["site_quat"]
        for gid in g.values():
            _geom.refresh_bounds(m, gid)

    def place_parcel(self, i: int, p: Parcel, pos, quat, vel=None) -> None:
        """Put parcel ``p`` into slot ``i`` at pose (pos, quat wxyz) with zero
        (or the given 6-vector) velocity and refresh the derived constants
        while keeping the rest of the world state."""
        self.apply_parcel_fields(i, p)
        d = self.data
        q, v = self._slot_qadr[i], self._slot_dadr[i]
        d.qpos[q:q + 3] = pos
        d.qpos[q + 3:q + 7] = quat
        d.qvel[v:v + 6] = 0.0 if vel is None else vel
        self._set_const_preserving()

    def _set_const_preserving(self) -> None:
        """mj_setConst recomputes derived constants at qpos0 and uses data as
        scratch; keep the world state across it."""
        m, d = self.model, self.data
        snap = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.act.copy(),
                d.time, d.qacc_warmstart.copy())
        mujoco.mj_setConst(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:] = snap[0], snap[1], snap[2]
        d.act[:] = snap[3]
        d.time = snap[4]
        d.qacc_warmstart[:] = snap[5]
        mujoco.mj_forward(m, d)

    def _despawn(self, i: int) -> None:
        s = self.slots[i]
        s.active, s.arrival, s.parcel, s.uid = False, -1, None, -1
        self._pin_parked()

    # ================================================================
    # public side (what a controller may be given)
    # ================================================================
    def joint_state(self) -> JointState:
        d = self.data
        return JointState(names=JOINT_NAMES,
                          q=d.qpos[self._arm_qadr].copy(),
                          qd=d.qvel[self._arm_dadr].copy(),
                          tau=d.actuator_force[self._arm_aid].copy(),
                          q_target=self._q_applied.copy())

    def gripper_state(self) -> GripperState:
        d = self.data
        lo, hi = GRIPPER.driver_range
        return GripperState(
            command=self._grip_cmd.copy(),
            closure=np.clip((d.qpos[self._grip_driver_qadr] - lo) / (hi - lo),
                            0.0, 1.0),
            force=d.actuator_force[self._grip_aid].copy())

    def render(self, camera: str) -> np.ndarray:
        """RGB image (H, W, 3) uint8 of a configured camera."""
        return self.cameras.rgb(self.data, camera)

    def render_depth(self, camera: str) -> np.ndarray:
        return self.cameras.depth(self.data, camera)

    def camera_intrinsics(self, camera: str) -> np.ndarray:
        return self.cameras.intrinsics(camera)

    def camera_extrinsics(self, camera: str) -> np.ndarray:
        return self.cameras.extrinsics(self.data, camera)

    def tcp_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Grasp-centre position and rotation matrix (forward kinematics of
        the robot's own joints; no parcel information)."""
        sid = self._tcp_sid[side]
        return (self.data.site_xpos[sid].copy(),
                self.data.site_xmat[sid].reshape(3, 3).copy())

    def state_digest(self) -> str:
        """sha256 of qpos and qvel (replay checks)."""
        import hashlib
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.data.qpos).tobytes())
        h.update(np.ascontiguousarray(self.data.qvel).tobytes())
        return h.hexdigest()

    def close(self) -> None:
        self.cameras.close()
