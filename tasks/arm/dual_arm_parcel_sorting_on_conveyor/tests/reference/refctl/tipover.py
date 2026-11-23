"""Tip over the far bottom edge, for a label on the face towards the robot
(-x): one palm pushes the parcel's near face outward and up; the parcel
pivots about its far bottom edge (the top goes away from the robot) and
comes to rest on its former far face, with the former near face (the label)
on top.

Physics. The palm presses the near face with a normal force N (into the
face) and lifts it with friction T <= mu_pad N (pads 1.8). About the far
bottom edge (the pivot) the push turns the parcel when
    h N + W T > m g (W/2 cos(phi) - H/2 sin(phi))
(h: contact height on the face, W: the parcel's width across the belt, H:
its height, phi: the tip angle), and the pivot does not slide while the
horizontal push stays inside the belt's friction (mu_belt = 0.9, the belt's
own value):
    N cos(phi) + T sin(phi) <= 0.9 (m g + N sin(phi) - T cos(phi)).
With a lift ratio T/N of about 1 both hold for every carton of the
generator (W/H up to 3.75) with margin; the palm follows the rigid rotation
of its contact point about the pivot (a kinematic push), so the parcel turns
rather than slides. Past the balance angle (the centre of mass over the
pivot, atan(W/H)) gravity finishes the last 15-35 degrees; the parcel lands
on its former far face, which then lies at x = x_far .. x_far + H: the tip is
planned only when that stays on the belt (W + H about 0.43 m at most for a
justified parcel). Every point of the parcel moves away from the robot, so
the near lip is never in the way.

Privileged reference: reads the parcel's pose, size and mass from the
simulator's truth; writes only joint position targets and gripper commands
(through the roll package's executor).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .endover import PAD_TIP
from .execu import SL
from .geom import label_dir
from .kin import rot_axis_angle
from .motion import DT, joint_move
from .paths import PlanFail

from .sideturn import G, SideParams, SideTurn, frame_zx


@dataclass
class TipParams:
    h_frac: float = 0.62          # contact height on the near face (fraction of H)
    h_min: float = 0.056          # ... at least this above the belt (the gripper clears the near lip)
    h_top: float = 0.016          # ... at least this below the top edge
    pre_dx: float = 0.03          # the palm waits this far in front of the face (m)
    press: float = 0.004          # target this far into the face (m)
    lift_ratio: float = 0.8       # the push's upward slide along the face per unit of the arc (friction lift)
    y_alt_left: tuple = (0.26, 0.30, 0.22, 0.33, 0.19)    # parcel centre y at the contact, left arm (its reach for the whole arc: y 0.3-0.55)
    y_alt_right: tuple = (-0.34, -0.38, -0.30)            # ... right arm (the column head blocks a level palm near the centre)
    arc_span: float = 0.26        # m of belt travel the arc may take (the left arm's reach along the belt; 0.30: 86 %, 0.34: 61 %, 0.22: 92.5 % of 160 dev parcels)
    t_tip_min: float = 0.9        # s, shortest arc (fast belts)
    t_acc: float = 0.30           # s: the palm reaches the parcel's speed and the face
    t_tip: float = 2.0            # s of the arc
    over_deg: float = 22.0        # the arc ends this far past the balance angle (the parcel lags the palm by 5-8 deg)
    far_margin: float = 0.006     # the tipped parcel's far side stays this far inside the belt's far edge
    f_push: float = 2.0           # N feedforward into the face
    f_push_mg: float = 0.35       # + this x m g
    v_app: float = 2.6            # rad/s of the approach move
    margin_station: float = 0.006  # station clearance of every planned leg (true distance, m)
    back_s: float = 0.30          # s of the back-off after the arc
    settle_s: float = 0.6         # s the parcel is left to settle before the outcome is read
    tilt_max_deg: float = 0.0     # the tool turns with the face up to this; beyond, the pads' lower edge pushes
    leave_deg: float = 10.0       # the palm leaves as soon as the parcel is this far past its balance angle
    above_dz: float = 0.09        # the approach comes down onto the wait pose from this far above (m)
    descend_s: float = 0.35       # s of that descent
    # where the tipped parcel may land, and the alternatives tried
    far_ref: str = "rail"         # the tipped parcel's far side must stay inside: "rail" (the far rail's inner
                                  # face, 1 cm beyond the belt's far edge, minus rail_margin) | "belt" (
                                  # the belt's far edge minus far_margin)
    rail_margin: float = 0.004    # ... clearance from the far rail's inner face (m)
    lean_max: float = 0.0         # > 0: also tip parcels whose far side would land up to this far beyond the
                                  # rail's inner face (they come to rest leaning on the rail)
    alt_spans: tuple = ((0.26, 0.9), (0.22, 0.75), (0.18, 0.6))   # (arc_span, t_tip_min) tried in order when
                                  # no plan is found (fast belts: a shorter arc in belt travel keeps it in reach)
    y_shift_v: float = 0.5        # later alternatives: contact positions moved downstream by this x belt speed
                                  # x t_acc (the wait pose then stays where it is on slow belts, clear of the
                                  # column head); the base positions are always tried first, unshifted (a
                                  # shifted first choice cost two parcels off the belt in a full run)
    y_alt_extra: tuple = (0.28, 0.24)   # further contact positions (left arm) tried after y_alt_left
    y_min_new: float = 0.22       # the extra / shifted alternatives keep the contact at least this far downstream
    over_alt_deg: float = 14.0    # > 0: a last fallback stage plans the base positions and arc with the arc ending
                                  # this far past the balance (instead of over_deg): a shorter arc keeps the palm in
                                  # reach for wide, tall cartons (the palm leaves at leave_deg past the balance anyway)
    fast_v: float = 0.25          # belts at least this fast (m/s) ...
    fast_max_aspect: float = 2.2  # ... decline parcels with W / H at least this: the tipped parcel stands on its
                                  # narrow side (H wide, W tall) and the fast arc's swing toppled 3 of 10 such tips
                                  # over the far rail (0 falls in 975 other tips)


class TipOver(SideTurn):
    """The tip over the far bottom edge by one palm (the right arm upstream,
    the left arm downstream of the station centre)."""

    def __init__(self, rc, eo=None, params: SideParams | None = None, tip: TipParams | None = None, log=None):
        super().__init__(rc, eo, params, log=log)
        self.tp = tip or TipParams()

    # ------------------------------------------------------------------
    def tip_frame(self, slot):
        """Parcel axes of the tip: e_x outward normal of the far face (the
        parcel axis closest to +x), e_z up, e_y = e_z x e_x (along the pivot
        edge); half sizes along them; the pivot point (far bottom edge, at the
        parcel's centre along e_y); the near face's centre."""
        pos, R, size, mass = self.box(slot)
        ix = int(np.argmax(np.abs(R[0, :])))
        iz = int(np.argmax(np.abs(R[2, :])))
        iy = 3 - ix - iz
        ex = R[:, ix] * (1.0 if R[0, ix] > 0 else -1.0)
        ez = R[:, iz] * (1.0 if R[2, iz] > 0 else -1.0)
        ex = ex - ez * float(ex @ ez)
        ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        hx, hy, hz = size[ix] / 2, size[iy] / 2, size[iz] / 2
        pivot = pos + hx * ex - hz * ez
        return dict(pos=pos, R=R, ex=ex, ey=ey, ez=ez, hx=float(hx), hy=float(hy), hz=float(hz),
                    pivot=pivot, mass=mass, iz=iz, sz=(1.0 if R[2, iz] > 0 else -1.0))

    def far_after(self, f):
        """World x of the far side after the tip (the former top face)."""
        return float(f["pivot"][0] + 2 * f["hz"] * abs(f["ex"][0]) + 2 * f["hx"] * 0.0)

    def arc(self, f, phi, h_c, slide=0.0):
        """Contact point and tool frame at tip angle phi (rad) for a contact
        at height h_c above the base on the near face (slid up along the
        face by `slide`), relative to the parcel's present pose."""
        P = f["pivot"]
        ey = f["ey"]
        c0 = f["pos"] - f["hx"] * f["ex"] + (h_c + slide - f["hz"]) * f["ez"]
        Rr = rot_axis_angle(ey, phi)
        c = P + Rr @ (c0 - P)
        z_t = Rr @ f["ex"]
        return c, z_t

    def tool_R(self, z_t, ey, sign):
        return frame_zx(z_t, sign * ey)

    def tilt_of(self, slot, f):
        """Tip angle reached (deg): the planned up axis's lean towards +e_x."""
        pos, R, size, _ = self.box(slot)
        up0 = R[:, f["iz"]] * f["sz"]
        return math.degrees(math.atan2(float(up0 @ f["ex"]), float(up0 @ f["ez"])))

    @staticmethod
    def _s_of(i, n):
        s_ = i / max(1, n)
        return s_ * s_ * s_ * (10 - 15 * s_ + 6 * s_ * s_)

    # ------------------------------------------------------------------
    def plan_tip(self, slot):
        tp = self.tp
        eo = self.eo
        f = self.tip_frame(slot)
        H = 2 * f["hz"]
        Wd = 2 * f["hx"]
        v = max(self.rc.belt_speed(), 0.02)
        x_far_after = float(f["pivot"][0] + H)
        if v >= tp.fast_v and Wd / max(H, 1e-3) >= tp.fast_max_aspect:
            raise PlanFail(f"tip: declined (narrow landing on a fast belt: W/H {Wd / H:.2f}, v {v:.3f})")
        lim = self.far_limit()
        if x_far_after > lim + max(0.0, float(tp.lean_max)):
            raise PlanFail(f"tip: would land beyond the far edge (x {x_far_after:.3f}, limit {lim:.3f})")
        h_c = float(np.clip(tp.h_frac * H, tp.h_min, H - tp.h_top))
        if h_c < tp.h_min - 1e-6:
            raise PlanFail("tip: parcel too low for the palm")
        phi_b = math.atan2(f["hx"], f["hz"])
        phi_end = phi_b + math.radians(tp.over_deg)
        n_acc = int(round(tp.t_acc / DT))
        y0 = float(f["pos"][1])
        q14 = self.ex.q.copy()
        why = []
        spans = tuple(tp.alt_spans) if tp.alt_spans else ((tp.arc_span, tp.t_tip_min),)
        dy_v = float(tp.y_shift_v) * v * tp.t_acc
        base = [("Left", y) for y in tp.y_alt_left] + [("Right", y) for y in tp.y_alt_right]
        new_l = [y for y in tp.y_alt_extra] + [y + dy_v for y in tuple(tp.y_alt_left) + tuple(tp.y_alt_extra)]
        new_l = [y for y in new_l if y >= tp.y_min_new - 1e-9]
        stages = []
        for i_sp, (span, tmin) in enumerate(spans):
            alts = (base if i_sp == 0 else []) + [("Left", y) for y in new_l]
            # stage 0 = the base plan exactly (its positions, its arc), then the alternatives
            if i_sp == 0:
                stages.append((i_sp, span, tmin, base))
                stages.append((i_sp, span, tmin, [("Left", y) for y in new_l]))
            else:
                stages.append((i_sp, span, tmin, [("Left", y) for y in new_l]))
        for i_sp, span, tmin, alts in stages:
            t_tip = float(np.clip(float(span) / v, float(tmin), tp.t_tip))
            n_tip = int(round(t_tip / DT))
            done = self._plan_alts(f, alts, v, y0, q14, h_c, phi_b, phi_end, n_acc, n_tip, why, tag0=f"s{i_sp}")
            if done is not None:
                done["span"] = float(span)
                return done
        if tp.over_alt_deg > 0:
            phi_end2 = phi_b + math.radians(tp.over_alt_deg)
            for span, tmin in spans:
                t_tip = float(np.clip(float(span) / v, float(tmin), tp.t_tip))
                n_tip = int(round(t_tip / DT))
                done = self._plan_alts(f, base + [("Left", y) for y in new_l], v, y0, q14, h_c, phi_b, phi_end2,
                                       n_acc, n_tip, why, tag0="o")
                if done is not None:
                    done["span"] = float(span)
                    done["over_alt"] = True
                    return done
        raise PlanFail("tip: no plan (" + ", ".join(why[:160]) + ")")

    def far_limit(self):
        """Largest x the tipped parcel's far side may reach."""
        tp = self.tp
        st = self.rc.cell.cfg.station
        belt_far = st.belt_center_x + st.belt_width / 2
        if tp.far_ref != "rail":
            return float(belt_far - tp.far_margin)
        if getattr(self, "_rail_x", None) is None:
            try:
                m, d = self.rc.cell.model, self.rc.cell.data
                g = m.geom("belt_rail_far").id
                self._rail_x = float(d.geom_xpos[g][0] - m.geom_size[g][0])
            except Exception:  # noqa: BLE001
                self._rail_x = float(belt_far + 0.010)
        return float(self._rail_x - tp.rail_margin)

    def _plan_alts(self, f, alts, v, y0, q14, h_c, phi_b, phi_end, n_acc, n_tip, why, tag0=""):
        tp = self.tp
        for side, y_c in alts:
            t_c = (y_c - y0) / v                    # s until the parcel's centre reaches y_c
            if t_c < 1.0:
                why.append(f"{y_c}: late")
                continue
            if True:
                for sign in (1.0, -1.0):
                    tag = f"{tag0}:{y_c:.3f}/{side[0]}{int(sign)}"
                    tg = self._tip_targets(f, side, sign, v, t_c, h_c, phi_end, n_acc, n_tip)
                    wait, R0, targets = tg
                    above = wait + np.array([-0.02, 0.0, tp.above_dz])
                    kin = self.kin[side]
                    cands = []
                    for qa in (self.ik_near(side, above, R0, q14[SL[side]]),
                               self.eo.glob_R(side, above, R0, None)):
                        if qa is not None and all(np.abs(qa - o).max() > 0.2 for o in cands):
                            cands.append(qa)
                    if not cands:
                        why.append(f"{tag}: ik above")
                        continue
                    done = None
                    for qa in cands:
                        qg = q14.copy()
                        qg[SL[side]] = qa
                        Qm, _, _ = joint_move(q14, qg, vmax=tp.v_app, tmin=0.6)
                        wl = []
                        if not self.path_clear(Qm, side, margin=tp.margin_station, every=1, why=wl):
                            why.append(f"{tag}: move {wl[:1]}")
                            # through a high waypoint on the robot's side of the
                            # pose above, then straight down to it
                            Qm = None
                            for dxh, dzh in ((-0.05, 0.12), (-0.02, 0.20), (0.04, 0.16)):
                                hp = above + np.array([dxh, 0.0, dzh])
                                qv = self.ik_near(side, hp, R0, q14[SL[side]])
                                if qv is None:
                                    continue
                                g1 = q14.copy()
                                g1[SL[side]] = qv
                                Q1, _, _ = joint_move(q14, g1, vmax=tp.v_app, tmin=0.5)
                                r2 = self._cart_rows(side, above, R0, max(0.25, dzh / 0.5), q0=qv)
                                if r2 is None:
                                    continue
                                Q2 = np.repeat(Q1[-1][None], len(r2), axis=0)
                                Q2[:, SL[side]] = np.array(r2)
                                Qc = np.vstack([Q1, Q2])
                                if self.path_clear(Qc, side, margin=tp.margin_station, every=1):
                                    Qm = Qc
                                    qa = np.array(r2[-1])
                                    break
                            if Qm is None:
                                continue
                        rows_d = self._cart_rows(side, wait, R0, tp.descend_s, q0=qa)
                        if rows_d is None:
                            why.append(f"{tag}: descent ik")
                            continue
                        Qd = np.repeat(q14[None], len(rows_d), axis=0)
                        Qd[:, SL[side]] = np.array(rows_d)
                        Qa = self._trace(side, targets, np.array(rows_d[-1]), q14)
                        if Qa is None:
                            why.append(f"{tag}: arc ik")
                            continue
                        wl = []
                        if not self.path_clear(np.vstack([Qd, Qa]), side, margin=tp.margin_station, every=1, why=wl):
                            why.append(f"{tag}: arc {wl[:1]}")
                            continue
                        t_wait = t_c - tp.t_acc - (len(Qm) + len(Qd)) * DT
                        if t_wait < 0.05:
                            why.append(f"{tag}: late")
                            continue
                        done = dict(side=side, sign=sign, y_c=y_c, Qm=np.vstack([Qm, Qd]), Qa=Qa, t_wait=t_wait,
                                    h_c=h_c, phi_end=phi_end, phi_b=phi_b, f=f, v=v, t_tip=n_tip * DT)
                        break
                    if done is not None:
                        return done
        return None

    def _tip_targets(self, f, side, sign, v, t_c, h_c, phi_end, n_acc, n_tip):
        """The wait pose, its tool frame and the Cartesian targets (one per
        tick) of the speed-matched approach onto the face and of the arc; the
        parcel's pose is extrapolated with the belt (contact at time t_c)."""
        tp = self.tp
        ey = f["ey"]
        yv = np.array([0.0, 1.0, 0.0])
        c0, z0 = self.arc(f, 0.0, h_c)
        R0 = self.tool_R(z0, ey, sign)
        # wait pose: in front of the face, upstream by the distance the palm
        # covers while it accelerates to the belt speed
        wait = c0 + (PAD_TIP - tp.pre_dx) * z0
        wait[1] = float(c0[1] + v * t_c - v * tp.t_acc / 2)
        targets = []
        for i in range(n_acc):
            s = (i + 1) / n_acc
            y_rel = v * tp.t_acc * (s ** 3 - 0.5 * s ** 4)        # smooth ramp 0 -> v
            sx = s * s * (3 - 2 * s)
            targets.append((wait + np.array([0.0, y_rel, 0.0]) + (tp.pre_dx + tp.press) * sx * z0, R0))
        for i in range(n_tip):
            s_ = (i + 1) / n_tip
            s = s_ * s_ * s_ * (10 - 15 * s_ + 6 * s_ * s_)
            phi = phi_end * s
            t = t_c + (i + 1) * DT
            c, z_face = self.arc(f, phi, h_c)
            tau = min(phi, math.radians(tp.tilt_max_deg))
            z_t = rot_axis_angle(ey, tau) @ f["ex"]
            Rt = self.tool_R(z_t, ey, sign)
            # beyond the tool's tilt the face leans away from the pads' tip
            # faces: their lower edge (11 mm below the TCP line) is the contact
            y_t = Rt[:, 1] if Rt[2, 1] > 0 else -Rt[:, 1]
            e = 0.011 * min(1.0, max(0.0, (phi - tau) / math.radians(8.0)))
            targets.append((c + v * t * yv + (PAD_TIP + tp.press) * z_face + e * y_t, Rt))
        return wait, R0, targets

    def _trace(self, side, targets, q_start, q14):
        """Joint rows (14) through the targets from q_start (continuous IK
        with a mild pull to mid-range); None if the arm loses the path."""
        kin = self.kin[side]
        mid = (kin.lo + kin.hi) / 2
        q = np.asarray(q_start, float).copy()
        rows = []
        for tgt, Rt in targets:
            qn, ok, ep, er = kin.ik(tgt, R=Rt, q0=q, q_rest=mid, iters=150, margin=0.01, null_gain=0.02)
            if not ok or ep > 0.004 or np.abs(qn - q).max() > 0.09:
                return None
            q = qn
            r = q14.copy()
            r[SL[side]] = q
            rows.append(r)
        return np.array(rows)

    # ------------------------------------------------------------------
    def tip(self, slot):
        """The tip. Entry: the parcel approaching on the belt (upstream of the
        planned contact line), justified, label on its -x face; both arms
        parked. Exit: the parcel resting on its former far face, released,
        riding downstream; the arm parked. Returns a dict (ok, label after,
        timing, forces, reason)."""
        tp = self.tp
        eo = self.eo
        out = dict(ok=False, reason="", t0=self.cell.time)
        if not self.cell.slots[slot].active:
            out["reason"] = "parcel gone"
            return out
        out["label0"] = label_dir(self.cell.truth.parcel_state(slot).label_normal)[0]
        try:
            plan = self.plan_tip(slot)
        except PlanFail as e:
            out["reason"] = str(e)
            return out
        side = plan["side"]
        f = plan["f"]
        mg = f["mass"] * G
        out.update(side=side, y_c=round(float(plan["y_c"]), 3), span=plan.get("span"), over_alt=plan.get("over_alt"), h_c=round(plan["h_c"], 3),
                   phi_b=round(math.degrees(plan["phi_b"]), 1), phi_end=round(math.degrees(plan["phi_end"]), 1))
        eo.phase = "tip_approach"
        Qm, Qa = plan["Qm"], plan["Qa"]
        for row in Qm:
            eo.tick(row)
        n_wait = int(round(plan["t_wait"] / DT))
        for _ in range(max(0, n_wait)):
            eo.tick(self.ex.q.copy())
        eo.phase = "tip_arc"
        n_acc = int(round(tp.t_acc / DT))
        fmax = 0.0
        trace = []
        Fp = tp.f_push + tp.f_push_mg * mg
        for i, row in enumerate(Qa):
            F = None
            if i >= n_acc - 3:
                # press into the face along the tool axis
                Rt = self.kin[side].fk(row[SL[side]])[1]
                F = {side: Rt[:, 2] * Fp}
            eo.tick(row, F)
            if not self.cell.slots[slot].active:
                break
            fmax = max(fmax, self.contact_side(slot, side))
            if i >= n_acc and (i - n_acc) % 5 == 0:
                trace.append((round(self.cell.time, 2), round(self.tilt_of(slot, f), 1),
                              round(math.degrees(plan["phi_end"]) * self._s_of(i - n_acc + 1, len(Qa) - n_acc), 1),
                              round(self.contact_side(slot, side), 1)))
            # past its balance by a margin the parcel falls on by itself:
            # the palm leaves at once (it would meet the landing parcel)
            if i >= n_acc and self.tilt_of(slot, f) >= math.degrees(plan["phi_b"]) + tp.leave_deg:
                out["left_early_deg"] = round(self.tilt_of(slot, f), 1)
                break
        out["push_force_max"] = round(fmax, 1)
        out["t_arc_end"] = round(self.cell.time, 2)
        out["tilt_at_arc_end"] = round(self.tilt_of(slot, f), 1) if self.cell.slots[slot].active else None
        out["trace"] = trace
        # back off along the face's outward normal at the reached tilt (no
        # drag along the face, which would pull the parcel back over its
        # balance), moving with the belt
        eo.phase = "tip_back"
        v = plan["v"]
        kin = self.kin[side]
        q = self.ex.q[SL[side]].copy()
        p0, R0 = kin.fk(q)
        tilt = math.radians(min(90.0, out["tilt_at_arc_end"] or 0.0))
        n_face = -math.cos(tilt) * f["ex"] + math.sin(tilt) * f["ez"]
        n_b = max(4, int(round(tp.back_s / DT)))
        for i in range(n_b):
            s_ = (i + 1) / n_b
            s = s_ * s_ * (3 - 2 * s_)
            tgt = p0 + 0.05 * s * n_face + np.array([0.0, v * (i + 1) * DT, 0.03 * s])
            qn, ok, ep = eo.ik_R(side, tgt, R0, q, iters=60)
            stp = np.abs(qn - q).max()
            if stp > eo.p.step_max:
                qn = q + (qn - q) * (eo.p.step_max / stp)
            q = qn
            row = self.ex.q.copy()
            row[SL[side]] = q
            eo.tick(row)
        self._escape(side)
        eo.phase = "tip_settle"
        t_s = self.cell.time
        while self.cell.time - t_s < tp.settle_s and self.cell.slots[slot].active and not self.cell.done:
            eo.tick(self.ex.q.copy())
        if self.cell.slots[slot].active:
            lab, err = eo.label_now(slot)
            out["label_after"], out["label_err"] = lab, round(err, 1)
            out["ok"] = lab == "up"
            if not out["ok"]:
                out["reason"] = f"label {lab} ({err:.0f} deg)"
        else:
            out["reason"] = "parcel gone"
        out["t_end"] = round(self.cell.time, 2)
        return out

    def _escape(self, side):
        """The arm up and to its park joints (checked legs; the right arm's
        escape of the side module for the right arm)."""
        if side == "Right":
            self._escape_right(None, dz_min=0.25, back_y=0.0)
            return
        # left arm: straight up, then park
        eo = self.eo
        kin = self.kin["Left"]
        q = self.ex.q[SL["Left"]].copy()
        p0, R0 = kin.fk(q)
        up = np.array([p0[0] - 0.02, p0[1], max(p0[2] + 0.04, self.belt + 0.25)])
        rows = self._cart_rows("Left", up, R0, max(0.3, (up[2] - p0[2]) / 0.3))
        q14 = self.ex.q.copy()
        if rows is not None:
            for r_ in rows:
                row = self.ex.q.copy()
                row[SL["Left"]] = r_
                eo.tick(row)
        qg = self.ex.q.copy()
        qg[SL["Left"]] = self.rc.q_ready[SL["Left"]]
        Q, _, _ = joint_move(self.ex.q, qg, vmax=self.p.v_park, tmin=0.6)
        for row in Q:
            r_ = row.copy()
            r_[SL["Right"]] = self.ex.q[SL["Right"]]
            eo.tick(r_)
