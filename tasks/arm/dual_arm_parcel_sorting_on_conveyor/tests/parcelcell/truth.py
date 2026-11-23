"""Privileged side: world state for graders and for a reference solution.

Never hand any of this to a controller under test.

* ``parcel_states()``  pose, velocity, size, kind, label normal and label
  angle to vertical of every active parcel;
* ``belt_speed()``     commanded (shift) and measured slab speed;
* ``contacts()``       robot-parcel and robot-belt-hardware contacts with
  forces (world frame); robot-belt-frame probe touches (no force);
* ``events``           the event log: ``spawn``, ``line`` (crossing of a
  named line, including ``exit``), ``exit`` (removed past the exit line),
  ``fall`` (removed after falling off).

Pose, velocity and label direction come from qpos / qvel at the end of the
tick; contact flags come from the contact set of the tick's last physics
step (2 ms earlier).

A line crossing record holds: tick time ``t`` and interpolated crossing time
``t_cross``, slot and arrival index, crossing direction (+1 downstream),
position (x, x relative to the belt centre line, y, z), label normal, label
angle to vertical (rad and deg), angular speed (rad/s), linear velocity,
whether the parcel touched the robot at that instant, whether it touched the
belt surface in that tick (``on_belt``), how long ago it last touched the
belt surface (``belt_contact_age_s``: 0 when it touches it in that tick;
None when it never has), the height of its lowest collision point above the
belt surface (``clearance_m``: from the same physics step as the contact
flags; below 0 when it is pressed into the belt) and its speed relative to
the belt. A soft mailer's belt contact comes and goes for a few ticks at a
time while it rides untouched (it rocks on its pad), so the scan line's
"rests on the belt" is judged from the last two (``episode.py``), never from
one tick's contact flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

__all__ = ["Truth", "ParcelState", "ContactRecord"]


@dataclass
class ParcelState:
    slot: int
    arrival: int              # index in the shift's arrivals (-1: injected)
    uid: int                  # running parcel number within the shift
    kind: str
    size: tuple
    pos: np.ndarray           # body origin, world (m)
    quat: np.ndarray          # wxyz
    linvel: np.ndarray        # world (m/s)
    angvel: np.ndarray        # world (rad/s)
    label_normal: np.ndarray  # world unit vector
    label_angle: float        # rad, 0 = label facing straight up
    on_belt: bool
    touching_robot: bool
    touching_parcel: bool = False   # another parcel touches it


@dataclass
class ContactRecord:
    kind: str                 # "robot-parcel" | "robot-belt" | "robot-belt-probe"
    robot_geom: str
    other_geom: str
    slot: int                 # parcel slot or -1
    pos: np.ndarray
    normal: np.ndarray        # unit normal pointing from the other geom to the robot
    dist: float               # signed distance (m, < 0 = penetration)
    force_normal: float       # N, >= 0 (0 for probes)
    force_world: np.ndarray   # contact force acting ON THE ROBOT geom, world (N)


class Truth:
    def __init__(self, cell):
        self.cell = cell
        m = cell.model
        self._gname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or
                       f"geom{g}" for g in range(m.ngeom)]
        self.events: list[dict] = []

    def reset(self) -> None:
        self.events = []
        self._touch = np.zeros(self.cell.n_slots, bool)
        self._on_belt = np.zeros(self.cell.n_slots, bool)
        self._touch_parcel = np.zeros(self.cell.n_slots, bool)
        self._belt_last = np.full(self.cell.n_slots, -np.inf)   # time of the last tick with belt contact
        self._probe_touch = False
        self._probe_hits = []
        self._scan_contacts()

    # ------------------------------------------------------------------
    def belt_speed(self) -> dict:
        c = self.cell
        return {"nominal": float(c._belt_cmd),
                "measured": float(c.data.qvel[c._slab_dadr])}

    def _body_rot(self, i: int) -> np.ndarray:
        """Parcel rotation from qpos (consistent with the reported pose; the
        kinematics in mjData lag one 2 ms physics step after mj_step)."""
        c = self.cell
        q = c._slot_qadr[i]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, c.data.qpos[q + 3:q + 7])
        return R.reshape(3, 3)

    def _label(self, i: int) -> tuple[np.ndarray, float]:
        c = self.cell
        Rs = np.zeros(9)
        mujoco.mju_quat2Mat(Rs, c.model.site_quat[c._slot_sid[i]])
        n = self._body_rot(i) @ Rs.reshape(3, 3)[:, 2]
        n = n / np.linalg.norm(n)
        return n, float(math.acos(max(-1.0, min(1.0, n[2]))))

    def _vel(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        c = self.cell
        v = c._slot_dadr[i]
        lin = c.data.qvel[v:v + 3].copy()
        ang = self._body_rot(i) @ c.data.qvel[v + 3:v + 6]   # free joint: local
        return lin, ang

    def parcel_state(self, i: int) -> ParcelState:
        c = self.cell
        s = c.slots[i]
        q = c._slot_qadr[i]
        n, ang_err = self._label(i)
        lin, ang = self._vel(i)
        return ParcelState(
            slot=i, arrival=s.arrival, uid=s.uid, kind=s.parcel.kind, size=s.parcel.size,
            pos=c.data.qpos[q:q + 3].copy(), quat=c.data.qpos[q + 3:q + 7].copy(),
            linvel=lin, angvel=ang, label_normal=n, label_angle=ang_err,
            on_belt=bool(self._on_belt[i]), touching_robot=bool(self._touch[i]),
            touching_parcel=bool(self._touch_parcel[i]))

    def parcel_states(self) -> list[ParcelState]:
        return [self.parcel_state(i) for i, s in enumerate(self.cell.slots)
                if s.active]

    # ------------------------------------------------------------------
    def contacts(self, kinds=("robot-parcel", "robot-belt",
                              "robot-belt-probe")) -> list[ContactRecord]:
        c = self.cell
        m, d = c.model, c.data
        out = []
        f6 = np.zeros(6)
        for k in range(d.ncon):
            con = d.contact[k]
            g1, g2 = int(con.geom1), int(con.geom2)
            r1, r2 = c._geom_is_robot[g1], c._geom_is_robot[g2]
            if r1 == r2:
                continue
            rg, og = (g1, g2) if r1 else (g2, g1)
            if c.parcel_geom_slot[og] >= 0:
                kind = "robot-parcel"
            elif c._geom_is_belt[og]:
                kind = "robot-belt"
            else:
                continue
            if kind not in kinds:
                continue
            frame = con.frame.reshape(3, 3)
            # MuJoCo: the frame normal points from geom1 to geom2 and the
            # contact force (contact frame) acts on geom2
            sign = 1.0 if rg == g2 else -1.0
            if con.exclude == 0 and con.efc_address >= 0:
                mujoco.mj_contactForce(m, d, k, f6)
                fw = sign * (frame.T @ f6[:3])
                fn = float(f6[0])
            else:
                fw, fn = np.zeros(3), 0.0
            out.append(ContactRecord(
                kind=kind, robot_geom=self._gname[rg], other_geom=self._gname[og],
                slot=int(c.parcel_geom_slot[og]), pos=con.pos.copy(),
                normal=sign * frame[0].copy(), dist=float(con.dist),
                force_normal=fn, force_world=fw))
        if "robot-belt-probe" in kinds:
            for rg, pg, dist, fromto in self._probe_hits:
                n = fromto[:3] - fromto[3:]      # probe -> robot
                nn = float(np.linalg.norm(n))
                out.append(ContactRecord(
                    kind="robot-belt-probe", robot_geom=self._gname[rg],
                    other_geom=self._gname[pg], slot=-1,
                    pos=0.5 * (fromto[:3] + fromto[3:]),
                    normal=n / nn if nn > 0 else np.zeros(3), dist=dist,
                    force_normal=0.0, force_world=np.zeros(3)))
        return out

    def _scan_contacts(self) -> None:
        """Per-slot flags: touching the robot, resting on the belt surface;
        and whether any robot geom is inside the belt frame probes."""
        c = self.cell
        d = c.data
        self._touch[:] = False
        self._on_belt[:] = False
        self._touch_parcel[:] = False
        self._scan_probes()
        n = d.ncon
        if n == 0:
            return
        g = np.asarray(d.contact.geom[:n])
        slot = c.parcel_geom_slot[g]                     # (n, 2)
        rob = c._geom_is_robot[g]
        surf = g == c._belt_surface
        for a, b in ((0, 1), (1, 0)):
            sel = (slot[:, a] >= 0) & rob[:, b]
            self._touch[np.unique(slot[sel, a])] = True
            sel = (slot[:, a] >= 0) & surf[:, b]
            self._on_belt[np.unique(slot[sel, a])] = True
        sel = (slot[:, 0] >= 0) & (slot[:, 1] >= 0) & (slot[:, 0] != slot[:, 1])
        if sel.any():
            self._touch_parcel[np.unique(slot[sel].ravel())] = True
        self._belt_last[self._on_belt] = float(c.time)

    def clearance(self, i: int) -> float:
        """Height (m) of slot i's lowest collision point above the belt
        surface's top (below 0: pressed into it), from the geoms' poses of the
        tick's last physics step (the step the contact flags come from)."""
        c = self.cell
        m, d = c.model, c.data
        bs = c._belt_surface
        top = float(d.geom_xpos[bs][2]) + float(np.abs(d.geom_xmat[bs].reshape(3, 3)[2]) @ m.geom_size[bs])
        low = math.inf
        for role in ("geom", "pad"):
            g = c._slot_g[i][role]
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            rz, sz = d.geom_xmat[g].reshape(3, 3)[2], m.geom_size[g]
            t = int(m.geom_type[g])
            if t == int(mujoco.mjtGeom.mjGEOM_BOX):
                ext = float(np.abs(rz) @ sz)
            elif t == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                ext = float(np.sqrt(np.sum((rz * sz) ** 2)))
            elif t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                ext = float(sz[0])
            else:
                ext = float(m.geom_rbound[g])
            low = min(low, float(d.geom_xpos[g][2]) - ext)
        return low - top

    def _scan_probes(self) -> None:
        """Robot collision geoms overlapping the belt frame probe boxes:
        bounding-sphere prefilter, then an exact distance query."""
        c = self.cell
        m, d = c.model, c.data
        self._probe_hits = []
        rg = c.robot_moving_geoms
        if not len(c.probe_geoms) or not len(rg):
            return
        gx = d.geom_xpos[rg]
        rb = m.geom_rbound[rg]
        fromto = np.zeros(6)
        for p in c.probe_geoms:
            R = d.geom_xmat[p].reshape(3, 3)
            local = (gx - d.geom_xpos[p]) @ R
            out = np.maximum(np.abs(local) - m.geom_size[p], 0.0)
            near = rg[np.linalg.norm(out, axis=1) < rb]
            for g in near:
                fromto[:] = 0.0
                dist = mujoco.mj_geomDistance(m, d, int(g), int(p), 0.01, fromto)
                if dist < 0:
                    self._probe_hits.append((int(g), int(p), float(dist),
                                             fromto.copy()))
        self._probe_touch = bool(self._probe_hits)

    @property
    def robot_in_belt_frame(self) -> bool:
        """A robot collision geom overlaps the belt frame (probes) now."""
        return self._probe_touch

    # ------------------------------------------------------------------
    def _log(self, ev: dict) -> None:
        self.events.append(ev)

    def _on_spawn(self, i: int, k: int, a, injected: bool = False) -> None:
        p = a.parcel
        self._belt_last[i] = -np.inf                     # a new parcel in the slot
        ev = {"type": "spawn", "t": self.cell.time, "t_scheduled": a.t,
              "delay": self.cell.time - a.t, "slot": i, "arrival": k,
              "uid": self.cell.slots[i].uid,
              "kind": p.kind, "size": list(p.size), "mass": p.mass,
              "label_face": p.label_face}
        if injected:
            ev["injected"] = True
        self._log(ev)

    def _record(self, i: int, etype: str, **extra) -> dict:
        c = self.cell
        st = self.parcel_state(i)
        v_belt = float(c.data.qvel[c._slab_dadr])
        rel = st.linvel[:2] - np.array([0.0, v_belt])
        last = float(self._belt_last[i])
        ev = {"type": etype, "t": c.time, "slot": i, "arrival": st.arrival,
              "uid": c.slots[i].uid,
              "kind": st.kind,
              "x": float(st.pos[0]),
              "x_rel": float(st.pos[0] - c.station.belt_center_x),
              "y": float(st.pos[1]), "z": float(st.pos[2]),
              "label_normal": st.label_normal.tolist(),
              "label_angle_rad": st.label_angle,
              "label_angle_deg": math.degrees(st.label_angle),
              "angular_speed": float(np.linalg.norm(st.angvel)),
              "linvel": st.linvel.tolist(),
              "touching_robot": st.touching_robot,
              "touching_parcel": st.touching_parcel,
              "on_belt": st.on_belt,
              "belt_contact_age_s": (round(max(0.0, float(c.time) - last), 6)
                                     if math.isfinite(last) else None),
              "clearance_m": float(self.clearance(i)),
              "speed_rel_belt": float(np.linalg.norm(rel))}
        ev.update(extra)
        return ev

    def _post_tick(self) -> None:
        c = self.cell
        d = c.data
        st = c.station
        self._scan_contacts()
        t1, t0 = c.time, c.time - 0.02
        remove = []
        for i, s in enumerate(c.slots):
            if not s.active:
                continue
            q = c._slot_qadr[i]
            x, y, z = (float(v) for v in d.qpos[q:q + 3])
            y0 = s.prev_y
            for name, yl in c.lines:
                if (y0 < yl <= y) or (y < yl <= y0):
                    direction = 1 if y > y0 else -1
                    frac = (yl - y0) / (y - y0) if y != y0 else 1.0
                    self._log(self._record(
                        i, "line", line=name, y_line=yl, direction=direction,
                        t_cross=t0 + frac * (t1 - t0)))
            s.prev_y = y
            if y > st.exit_y:
                self._log(self._record(i, "exit"))
                remove.append(i)
            elif (z < st.belt_height - st.fall_drop or abs(x) > st.fall_xy
                  or abs(y) > st.fall_xy):
                self._log(self._record(i, "fall"))
                remove.append(i)
        for i in remove:
            c._despawn(i)
            self._touch[i] = self._on_belt[i] = self._touch_parcel[i] = False
            self._belt_last[i] = -np.inf

    # ------------------------------------------------------------------
    def crossings(self, line: str = "scan", first_only: bool = True) -> list[dict]:
        """Downstream crossings of a line (by default the first per parcel)."""
        seen, out = set(), []
        for ev in self.events:
            if ev["type"] == "line" and ev["line"] == line and ev["direction"] > 0:
                if first_only and ev["uid"] in seen:
                    continue
                seen.add(ev["uid"])
                out.append(ev)
        return out
