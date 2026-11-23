"""``ShiftWorld``: one shift compiled as one model.

Every parcel of the shift is its own body (``parcel<i>`` for arrival i),
compiled with its final collision geometry, contact values, explicit inertial
and label site: exactly the fields the generic simulator's slot writes give
(``_geom.parcel_fields`` / ``_geom.label_layout``; a unit test compares them
field by field). Its looks come from a visuals object (``visuals``), which
adds drawn geoms only. A parcel waits parked below the floor until its
arrival, then lands at the entry exactly as a slot parcel does.

The simulator-level API is ``ParcelCell``'s (this class is a subclass), so a
controller written against the generic simulator runs unchanged:

    world = ShiftWorld(shift, config)
    world.reset(shift)                 # optional: the world starts reset
    while not world.done:
        world.step(q_target14, grip2)  # one 20 ms tick
    world.truth.parcel_states(); world.truth.contacts(); world.truth.events

Differences to the generic simulator, all following from the shift being
compiled in:
* ``done`` is True once the shift's duration has elapsed or the shift was
  ended (``end(reason)``: "finished", "program_time", ..., "physics_fault"
  when MuJoCo flags an unstable state); later ``step`` calls do not advance
  the world (more than ``AFTER_OVER_STEPS`` of them raise ``ShiftOver``);
* ``reset(shift)`` with a different shift recompiles for that shift;
* ``inject`` needs spare slots (``spare_slots``: generic run-time-written
  slots, parked like the others), for tooling and tests.

A shift with a ``setting`` (``Shift.setting``, the line setting) is fed and
driven by its rules instead of stated arrival times:
* feed on settle: one parcel of the list in the station at a time; the first
  is set down at the shift's first tick, each next one at the first tick after
  the one before it was settled (its centre crossed the scan line going
  downstream, it fell, or the safety watch recorded it damaged: the cell's
  ``damage_source``), at its own set-down position (``Arrival.y``) on the
  belt's centre line, once the entry is clear;
* indexing (``setting["indexing"]``): at the start of every tick the belt's
  command is zero while the centre of the parcel in the station (the one set
  down last and not yet settled) lies strictly inside ``setting["zone_y"]``
  (the body position of the last physics step), else the shift's speed;
* the shift ends (``over_reason`` "shift_end") at the end of the tick in
  which the last parcel of its list was settled, or at its duration.
"""

from __future__ import annotations

import time
from dataclasses import replace

import mujoco
import numpy as np

from . import _geom
from .cell import CONTROL_DT, CellConfig, ParcelCell
from .stream import Arrival, Shift
from .visuals import check_visual_only, make_visuals

__all__ = ["ShiftWorld", "ShiftOver", "AFTER_OVER_STEPS", "park_pos"]

AFTER_OVER_STEPS = 50


class ShiftOver(RuntimeError):
    """The shift is over and the caller kept stepping."""


def park_pos(i: int) -> tuple:
    """Parking spot of parcel body i: below the floor box (hidden from every
    camera above the floor, out of contact), nine to a row, inside the floor's
    footprint."""
    return (-3.4 + 0.85 * (i % 9), -2.6 - 0.85 * (i // 9), -0.8)


class ShiftWorld(ParcelCell):
    DEPTH_GROUP = 5
    """Geom group of the depth twin's meshes: drawn in depth images only."""

    def __init__(self, shift: Shift, config: CellConfig | None = None,
                 visuals=None, spare_slots: int = 0, depth_meshes=None):
        cfg = config or CellConfig()
        self.build_shift = shift
        self.depth_meshes = depth_meshes
        self.n_parcels = len(shift.arrivals)
        self.n_spare = int(spare_slots)
        self.visuals = make_visuals(visuals)
        self.over_reason = None
        self._after_over = 0
        self.tick_wall = 0.0           # wall seconds spent inside step()
        self._building = True
        t0 = time.time()
        super().__init__(replace(cfg, n_slots=self.n_parcels + self.n_spare))
        self.build_wall_s = time.time() - t0
        self._building = False
        check_visual_only(self.model, self._visual_geoms)

    NEAR_CLIP_M = 0.02
    """The cameras' near clipping distance (m). MuJoCo's default is 1 % of the
    model's extent (here about 0.13 m), which would hide everything a wrist
    camera sees closer than that (its fingertips, a face being approached);
    drawing only, no physics."""

    def _post_compile(self) -> None:
        """Bounds of the parcels' collision geoms by the slot writes' formula
        (``_geom.refresh_bounds``; the compiler's differs in the last bit);
        the cameras' near clipping distance (``NEAR_CLIP_M``)."""
        m = self.model
        m.vis.map.znear = float(self.NEAR_CLIP_M / max(1e-9, float(m.stat.extent)))
        for i in range(self.n_parcels):
            for n in ("geom", "pad"):
                _geom.refresh_bounds(self.model, self._gid(f"parcel{i}_{n}"))

    # ---------------------------------------------------------------- build
    def _park(self, i: int) -> tuple:
        return park_pos(i)

    def _build_spec(self) -> mujoco.MjSpec:
        from .robot import build_robot_spec
        from .station import add_station
        spec = build_robot_spec(self.cfg.lock_couplers, self.cfg.robot_mounts)
        spec.modelname = "dual_arm_parcel_station"
        add_station(spec, self.station)
        w = spec.worldbody
        shift = self.build_shift
        self.visuals.prepare(spec, shift)
        self._visual_names = []
        contact = self.cfg.parcel_contact
        for i, a in enumerate(shift.arrivals):
            p = a.parcel
            f = _geom.parcel_fields(p.kind, p.size, p.mass, p.friction, contact)
            body = w.add_body(name=f"parcel{i}", pos=list(self._park(i)))
            body.add_freejoint(name=f"parcel{i}_free")
            body.explicitinertial = True
            body.mass = float(f["mass"])
            body.inertia = [float(v) for v in f["inertia"]]
            body.ipos = [float(v) for v in p.com_offset]
            body.iquat = [1.0, 0.0, 0.0, 0.0]
            box_fr = list(f["box_friction"])
            body.add_geom(name=f"parcel{i}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[float(v) for v in f["box_size"]],
                          friction=box_fr, solimp=list(f["box_solimp"]),
                          solref=list(f["box_solref"]), group=3,
                          rgba=[0.72, 0.55, 0.38, 1.0])
            # the pad: the soft cover (soft) or a 1 mm box-inner placeholder
            # (rigid; the slot cell's compiled pad values, never in contact)
            if f["pad_friction"] is not None:
                pad_kw = dict(friction=list(f["pad_friction"]),
                              solimp=list(f["pad_solimp"]),
                              solref=list(f["pad_solref"]))
            else:
                pad_kw = dict(friction=[0.7, 0.01, 0.0003],
                              solimp=list(_geom.SOFT_PAD["solimp"]),
                              solref=list(_geom.SOFT_PAD["solref"]))
            body.add_geom(name=f"parcel{i}_pad",
                          type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                          size=[float(v) for v in f["pad_size"]], group=3,
                          rgba=[0.68, 0.52, 0.36, 1.0], **pad_kw)
            lay = dict(_geom.label_layout(p.size, p.label_face, p.label_uv))
            lay["face"], lay["uv"] = p.label_face, p.label_uv
            # the label site: the visuals' own placement record when they have
            # one (the parcel kit's real paper), else the simulator's layout;
            # its z axis is the label face's normal either way
            site = getattr(self.visuals, "label_site", None)
            site = site(i, p) if site is not None else None
            if site is not None:
                lay["site_pos"], lay["site_quat"] = site
            body.add_site(name=f"parcel{i}_label_site",
                          pos=[float(v) for v in lay["site_pos"]],
                          quat=[float(v) for v in lay["site_quat"]],
                          size=[0.005] * 3, group=4)
            self._visual_names += list(self.visuals.add(spec, body, i, p, lay))
        for j in range(self.n_spare):
            self._add_slot_body(w, self.n_parcels + j)
        self._add_depth_twin(spec)
        return spec

    def _add_depth_twin(self, spec) -> None:
        """The depth twin: the station's surroundings near the cameras (the render bundle's near meshes: fence,
        posts, gantry, frame, photo-eye; world frame) as drawn-only geoms in group ``DEPTH_GROUP`` of one static
        body added after every other body, so depth images show what the colour images show. They take no part in
        contacts, carry no mass and shift no other geom's index: the physics is the same with or without them."""
        dm = self.depth_meshes
        if not dm:
            return
        body = spec.worldbody.add_body(name="depth_twin")
        for i, (V, F) in enumerate(dm):
            me = spec.add_mesh(name="depth_twin_mesh_%d" % i)
            me.uservert = np.asarray(V, np.float64).ravel().tolist()
            me.userface = np.asarray(F, np.int32).ravel().tolist()
            me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            body.add_geom(name="depth_twin_geom_%d" % i, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=me.name,
                          contype=0, conaffinity=0, density=0.0, mass=0.0, group=self.DEPTH_GROUP,
                          rgba=[0.5, 0.5, 0.5, 1.0])

    def _slot_geoms(self, i: int) -> dict:
        if i >= self.n_parcels:                       # a spare slot
            return super()._slot_geoms(i)
        return {"geom": self._gid(f"parcel{i}_geom"),
                "pad": self._gid(f"parcel{i}_pad")}

    def _check_slot_geoms(self, i: int) -> None:
        if i >= self.n_parcels:
            return super()._check_slot_geoms(i)
        g = self._slot_g[i]
        assert min(g.values()) >= 0

    @property
    def _visual_geoms(self) -> list:
        return [self._gid(n) for n in self._visual_names]

    # ---------------------------------------------------------------- reset
    def reset(self, shift: Shift | None = None, force: bool = False) -> None:
        """Start (or restart) the shift: robot at its start posture, every
        parcel parked, the belt at the shift's speed, ``settle_s`` of settling;
        the clock then starts at 0. A different shift recompiles the world.

        A served world (one with tick listeners: the task cell) is reset by its
        cell only (``force``); a controller's own ``reset`` of the same shift
        before the first tick is a no-op (the world is already reset), any
        other is refused."""
        served = bool(self.tick_listeners) or self.command_gate is not None
        same = shift is None or shift is self.build_shift or \
            shift.to_dict() == self.build_shift.to_dict()
        if served and not force and not self._building:
            if same and self.n_ticks == 0:
                return
            raise RuntimeError("a served world is reset by its cell only")
        if self._building:
            shift = self.build_shift          # ParcelCell's constructor reset
        elif not same:
            if served:
                raise RuntimeError("a served world cannot change its shift")
            self.close()
            self.__init__(shift, replace(self.cfg, n_slots=0),
                          visuals=self.visuals, spare_slots=self.n_spare, depth_meshes=self.depth_meshes)
            return
        self.over_reason = None
        self._after_over = 0
        self.tick_wall = 0.0
        super().reset(self.build_shift)
        self._line_reset()

    # ---------------------------------------------------------------- the line setting
    damage_source = None
    """A callable giving the cell's safety watch's {slot: damage record} (set by the task cell); None: no parcel is
    ever damaged."""

    @property
    def setting(self):
        return self.build_shift.setting

    @property
    def scan_y(self) -> float:
        """The scan line's y (the cell's first named line)."""
        for name, y in self.lines:
            if name == "scan":
                return float(y)
        raise KeyError("scan")

    def _line_reset(self) -> None:
        self.settled = {}              # arrival -> {"t", "why"}
        self._feed_ev = 0
        self.list_done_t = None
        self.belt_halted_ticks = 0
        self.belt_halts = 0
        self._halted = False

    def _update_settled(self) -> None:
        """Arrivals settled so far: the first downstream crossing of the scan line, a fall, an exit, or a damage
        record of the safety watch."""
        ev = self.truth.events
        for e in ev[self._feed_ev:]:
            a = e.get("arrival", -1)
            if a is None or a < 0 or e.get("injected") or a in self.settled:
                continue
            why = None
            if e["type"] == "line" and e.get("line") == "scan" and e.get("direction", 0) > 0:
                why = "scan"
            elif e["type"] in ("fall", "exit"):
                why = e["type"]
            if why is not None:
                self.settled[a] = {"t": round(float(e["t"]), 4), "why": why}
        self._feed_ev = len(ev)
        dmg = self.damage_source() if callable(self.damage_source) else self.damage_source
        if dmg:
            for slot, rec in list(dmg.items()):
                if 0 <= int(slot) < self.n_parcels and int(slot) not in self.settled:
                    self.settled[int(slot)] = {"t": round(float(rec.get("t", self.time)), 4), "why": "damaged"}

    def _station_parcel(self):
        """Slot of the parcel in the station: the one set down last, not yet settled (None: none)."""
        k = self._next_arrival - 1
        if k < 0 or k in self.settled:
            return None
        s = self.slots[k]
        return k if (s.active and s.arrival == k) else None

    def _pre_tick(self) -> None:
        st = self.build_shift.setting
        if not st:
            return super()._pre_tick()
        from .station import belt_ctrl_for_speed
        d = self.data
        self._update_settled()
        v = self.shift.speed_at(self.time)
        halted = False
        if st.get("indexing"):
            lo, hi = (float(x) for x in st["zone_y"])
            i = self._station_parcel()
            if i is not None:
                y = float(d.xpos[self._slot_bid[i]][1])
                halted = lo < y < hi
        if halted:
            v = 0.0
            self.belt_halted_ticks += 1
            if not self._halted:
                self.belt_halts += 1
        self._halted = halted
        self._belt_cmd = v
        d.ctrl[self._belt_aid] = belt_ctrl_for_speed(v, self.cfg.belt_speed_compensation)
        self._wrap_slab()
        self._pin_parked()
        arr = self.shift.arrivals
        k = self._next_arrival
        if k < len(arr) and (k == 0 or (k - 1) in self.settled):
            if self._try_spawn(k, arr[k]):
                self._next_arrival += 1

    def _line_end_check(self) -> None:
        if not self.build_shift.setting or self.over_reason is not None:
            return
        self._update_settled()
        n = len(self.build_shift.arrivals)
        if n and len(self.settled) >= n and all(k in self.settled for k in range(n)):
            self.list_done_t = round(float(self.time), 4)
            self.over_reason = "shift_end"

    # ---------------------------------------------------------------- spawn
    def _try_spawn(self, k: int, a: Arrival) -> bool:
        """Arrival k lands on its own body (compiled with its fields): only
        the pose is written, once the entry is clear."""
        i = k
        s = self.slots[i]
        if s.active:
            return False
        pos, quat = self.entry_pose(a.parcel)
        if a.y is not None:
            pos[1] = float(a.y)
        if not self._entry_clear(a.parcel, pos):
            return False
        d = self.data
        q, v = self._slot_qadr[i], self._slot_dadr[i]
        d.qpos[q:q + 3] = pos
        d.qpos[q + 3:q + 7] = quat
        d.qvel[v:v + 6] = 0.0
        s.active, s.arrival, s.parcel = True, k, a.parcel
        s.t_spawn, s.prev_y = self.time, float(pos[1])
        s.uid, self._n_spawned = self._n_spawned, self._n_spawned + 1
        if self.build_shift.setting:
            # fed on settle: due at the shift's first tick (the first parcel) or at the tick after the one before it
            # was settled; the spawn record's delay is measured from then
            due = 0.0 if k == 0 else float(self.settled[k - 1]["t"])
            a = replace(a, t=min(due, float(self.time)))
        self.truth._on_spawn(i, k, a)
        return True

    def inject(self, parcel, pos, quat=(1.0, 0.0, 0.0, 0.0), vel=None) -> int:
        free = [i for i in range(self.n_parcels, self.n_slots)
                if not self.slots[i].active]
        if not free:
            raise RuntimeError("no free spare slot (ShiftWorld(spare_slots=...))")
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

    def set_pose(self, i: int, pos, quat=None, vel=None) -> None:
        """Move active parcel i (tooling and tests: constructed states)."""
        d = self.data
        q, v = self._slot_qadr[i], self._slot_dadr[i]
        d.qpos[q:q + 3] = pos
        if quat is not None:
            d.qpos[q + 3:q + 7] = quat
        if vel is not None:
            d.qvel[v:v + 6] = vel
        self.slots[i].prev_y = float(d.qpos[q + 1])
        mujoco.mj_forward(self.model, d)

    # ---------------------------------------------------------------- ticks
    @property
    def done(self) -> bool:
        """The shift is over: its duration has elapsed, or it was ended."""
        return self.over_reason is not None or \
            self.time >= self.shift.duration - 1e-9

    def end(self, reason: str) -> None:
        if self.over_reason is None:
            self.over_reason = str(reason)

    def step(self, q_target, grip=(0.0, 0.0)) -> None:
        if self.done:
            if self.over_reason is None:
                self.over_reason = "shift_end"
            self._after_over += 1
            if self._after_over > AFTER_OVER_STEPS:
                raise ShiftOver("the shift is over (%s)" % self.over_reason)
            return
        t0 = time.time()
        try:
            super().step(q_target, grip)
        finally:
            self.tick_wall += time.time() - t0
        if self._physics_fault():
            self.end("physics_fault")
        self._line_end_check()
        if self.over_reason is None and \
                self.time >= self.shift.duration - 1e-9:
            self.over_reason = "shift_end"

    _FAULTS = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL,
               mujoco.mjtWarning.mjWARN_BADQPOS)

    def _physics_fault(self) -> bool:
        """MuJoCo flagged an unstable state (it resets such a state itself) or
        the state is not finite: the shift ends as ``physics_fault`` (the cell's
        failure, not the program's)."""
        d = self.data
        if any(d.warning[int(w)].number for w in self._FAULTS):
            return True
        return not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())

    @property
    def tick(self) -> int:
        """Ticks since the shift clock started (time / 20 ms)."""
        return int(self.n_ticks)

    def parcel_index(self) -> list:
        """[(i, arrival time, kind)] for every parcel body (tooling)."""
        return [(i, a.t, a.parcel.kind)
                for i, a in enumerate(self.build_shift.arrivals)]


def world_ticks(duration_s: float) -> int:
    return int(round(duration_s / CONTROL_DT))
