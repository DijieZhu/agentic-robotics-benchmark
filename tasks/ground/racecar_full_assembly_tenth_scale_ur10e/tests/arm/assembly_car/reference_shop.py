"""The UR10e builds the race car on its work stand: the privileged reference program, part II: the four wheels, the
whole episode's flow with its retries, and the server's entry point (`reference_assemble`). Part I (`reference.py`:
the assembler core and the four top-down units) is used as it is; where the wheels need the core to behave
differently, `ShopAssembler` subclasses its `Assembler`.

Privileged state only (as part I): the units' exact poses, the slots' live targets (`CarCell.live_target`: the
carrier as it is now, a hub at its present angle), the hubs' angles, the contact forces of the simulation, the UR's
kinematics and the planner on a copy of the cell (`reference.make_planner`: the per-body margins fixed).

The wheel step (`place_wheel(unit, slot)`; any wheel fits any hub, the four are rigid copies of one part):
  grasps   the frozen wheel grasps of the hub's corner (`car_grasps.FROZEN["whl_<corner>"]`: the kit-validated
           primary, then its five static alternates; top-down across the tyre's diameter, the fingertips 6 mm below
           the outer face, the tool axis on the wheel's axis), each paired with the hex orientations k (0..5) of the
           hub nearest to the grasp's design roll (the tool's roll about the axis at the hub = the grasp's + the hub's
           angle + k x 60 deg); a (grasp, k) is feasible when `pick_checks` holds at the wheel's rest on the bench
           and `place_checks` along the hub's axis (the pre-insertion 60 mm out and inwards in 2 mm steps: fingers 1
           mm, links 3 mm off everything)
  pick     the precedent's top-down pick from the bench (the wheel on its inner face, outer face up): open to the
           width + 15 mm, the pre-grasp 80 mm above, the line down at 30 mm/s, closed to the width - 5 mm (~30 N per
           pad), lifted 60 mm at 50 mm/s, the hold within 3 mm / 4 deg
  carry    planned (the wheel held as it is now) to the pre-insertion: the wheel's pocket 60 mm out along the hub's
           axis, turned onto the hub's hex orientation k (`wheel_clock` "known": the live target from the hub's
           privileged angle and the in-hand pocket angle as held), the tcp <= 0.5 m/s
  insert   Cartesian along the axis at 10 mm/s, the arm's stiffness at 0.5, closed loop on the live target (k
           fixed); the cell captures the wheel within 3.75 mm of home when its pocket is within +-6 deg of a hex
           orientation (its stop is switched off and the coupling pulls it home); a wheel more than 1.5 mm behind the
           command for 10 ticks has stalled, and a wheel 1.8-3.3 mm short of home pushing the hub with more than 6 N
           (STOP_DETECT_N) is read as on its stop at once: there the pocket's turn is off every hex orientation (the hub
           has turned since the plan, or the clock is unknown); a stall anywhere else is a block (backed off 15 mm and
           pushed again, twice)
  search   on the stop (the fallback): the push held at SEARCH_PUSH_N (4 N, <= 5 N: from about 10 N the hub turns
           with the wheel) by an integral law on the tool's target along the axis (the wheel-hub contact force), the
           hand turning the wheel about the axis at 10 deg/s: towards the nearest hex orientation when the turn is
           known (the residual from the coupling's evaluation), else +35 deg then -35 deg (the hex repeats every 60
           deg: one of the two sweeps passes an orientation); captured -> on as below; not captured -> backed out
           20 mm and one more push with the turn re-read
  captured on home (0.5 mm past) and held there until the cell locks it (<= 5 s), opened by 2 mm per pad, backed
           out 60 mm along the axis (checked), the lock awaited (<= 3 s)
A wheel the arm cannot fasten is put back where it was picked (`put_back`: the reverse of the pick) and the episode's
retry pass may try it again; a unit that cannot be picked stays where it is.

The episode (`shop_steps(kit, order)`, `program(cell, order)`):
  shop  the stick pack, the deck onto the car, the compute module, the LiDAR (part I's four steps), then the wheels
        FR, RR, RL, FL (the designed wheel of each corner: WHL2, WHL4, WHL3, WHL1)
  B1    the wheels first: FR, RR, RL, then the stick pack and the deck, then the FL wheel, then the compute module and
        the LiDAR. The FL hub cannot be reached while the deck assembly is on its shelf (every IK branch of the
        hand's pre-insertion, and still 20 mm out, puts wrist 2 12.8-16.8 mm into the deck), so
        its wheel follows the deck
  B2    the shop order with the LiDAR before the compute module
  B3    the shop order with the wheels swapped: WHL4 on FL, WHL3 on FR, WHL2 on RL, WHL1 on RR (no wheel on the hub
        it was designed for)
  B4 (= "sub") the sub-assembly route: the stick pack, the compute module and the LiDAR onto the deck on its shelf,
        the loaded deck onto the car, then the wheels as in shop
  then one retry pass over the steps whose slot is not locked (a wheel put back is picked again; part I's units are
  tried once more where they lie), within the simulated-time budget (`max_sim_s`).

The contact monitor is part I's (`Assembler._monitor`) and, during a wheel step, also flags the moving wheel against
any body of the base but its carrier hub (part I lets a unit whose carrier is on the base touch the whole base: for a
wheel that would hide a tyre on a knuckle).

`reference_assemble(cell, plan=None, log=print, order="shop", **kw)` is the server's entry
(`CarInfoServer.reference_assemble`): it runs `program` inside the cell (every tick through the server's routed tick
and watch) and returns the report; a tick that no longer advances the simulation (the server's sim cap reached) ends
the program with the report so far.
"""
import math
import time

import numpy as np

from . import car_grasps as CG
from . import reference as REF
from .coupling import _inv, quat_wxyz

__all__ = ["ShopAssembler", "WHEEL_SPEC", "ORDERS", "CORNERS", "shop_steps", "program", "reference_assemble",
           "wheel_grasps", "SimStalled"]

CORNERS = ("FL", "FR", "RL", "RR")
WHEEL_SPEC = {"slot": None, "kind": "wheel", "pre_grasp_m": REF.PREGRASP_M, "open_extra_m": REF.OPEN_EXTRA_M,
              "lift_m": REF.LIFT_M, "from_above": True, "carry_v": REF.CARRY_SPEED_M_S, "release_extra_m": REF.RELEASE_EXTRA_M}
WHEEL_STALL_M = 0.0015          # the wheel this far behind the command along the axis ...
WHEEL_STALL_TICKS = 10          # ... for this many ticks: stalled (the wheel-entry trials' rule)
STOP_BAND_M = (0.0018, 0.0033)  # a stall this far short of home rests on the stop (the wheel-entry trials: 2.31-2.46 mm at 0-20 N)
SEARCH_PUSH_N = 4.0             # the search's push on the stop (<= 5 N; the wheel-entry trials: 3-5 N caught, hub creep <= 1.3 deg)
STOP_DETECT_N = 6.0             # a wheel in the stop band pushing the hub harder than this rests on its stop (the push is
                                # read at once: waiting for the lag rule let the arm press the stop with ~20 N first)
SEARCH_GAIN = 0.4e-3            # m per N s of the integral law on the tool's target (the wheel-entry trials' rate)
SEARCH_TURN_DEG_S = 10.0
SEARCH_SWEEP_DEG = 35.0
SEARCH_OVERSHOOT_DEG = 2.0
BACKOFF_M = 0.015
SEARCH_BACKOFF_M = 0.020
WHEEL_RETRIES = 2
K_PER_GRASP = 2                 # the hex orientations tried per grasp (nearest the grasp's design roll first)
FEASIBLE_LIMIT = 3              # the feasibility checks stop after this many feasible (grasp, k) pairs (the first is used)
MAX_SIM_S = 600.0               # no new step or retry is started beyond this simulated time (the server's cap: 650 s)


class SimStalled(RuntimeError):
    """the simulation does not advance any more (the server's sim cap ended the episode)"""


# ------------------------------------------------------------------ the episode's orders
def _designed_wheels(kit):
    """{corner: the wheel designed for it} (the slots' designed_unit; car_grasps.CORNER_WHEEL if absent)"""
    out = {}
    for s in kit["slots"]:
        if s["id"].startswith("whl_"):
            c = s["id"].split("_", 1)[1]
            out[c] = s.get("designed_unit") or CG.CORNER_WHEEL[c]
    return out


SWAP = {"FL": "RR", "FR": "RL", "RL": "FR", "RR": "FL"}      # B3: each hub takes the wheel designed for this corner
WHEEL_CORNER_ORDER = ("FR", "RR", "RL", "FL")
ORDERS = {
    "shop": ("BAT", "DCK", "CMP", "LDR", "whl_FR", "whl_RR", "whl_RL", "whl_FL"),
    "B1": ("whl_FR", "whl_RR", "whl_RL", "BAT", "DCK", "whl_FL", "CMP", "LDR"),
    "B2": ("BAT", "DCK", "LDR", "CMP", "whl_FR", "whl_RR", "whl_RL", "whl_FL"),
    "B3": ("BAT", "DCK", "CMP", "LDR", "whl_FR", "whl_RR", "whl_RL", "whl_FL"),
    "B4": ("BAT", "CMP", "LDR", "DCK", "whl_FR", "whl_RR", "whl_RL", "whl_FL"),
}
ORDERS["sub"] = ORDERS["B4"]


def shop_steps(kit, order="shop", assignment=None):
    """[{"kind": "top" | "wheel", "unit", "slot"}] of an episode: `order` one of ORDERS' names (or a sequence of unit
    types and wheel slots); `assignment` {corner: wheel unit} (default: the designed wheel, B3: swapped)"""
    seq = ORDERS[order] if isinstance(order, str) else tuple(order)
    des = _designed_wheels(kit)
    if assignment is None:
        assignment = {c: des[SWAP[c]] for c in des} if order == "B3" else dict(des)
    by_type = REF.top_down_units(kit)
    top_slot = {t: REF.SPEC[t]["slot"] for t in REF.SPEC}
    out = []
    for item in seq:
        if item.startswith("whl_"):
            c = item.split("_", 1)[1]
            out.append({"kind": "wheel", "unit": assignment[c], "slot": item})
        else:
            out.append({"kind": "top", "unit": by_type[item], "slot": top_slot[item]})
    units = [s["unit"] for s in out]
    if len(set(units)) != len(units):
        raise ValueError("order %r: a unit appears twice (%s)" % (order, units))
    return out


def wheel_grasps(slot, roles=("primary", "alternate")):
    """[(G (m, the wheel frame), width (m), desc, entry)] of a hub's corner: the frozen primary first, then the
    alternates (the four wheels are rigid copies with identical frames: a grasp in the designed wheel's frame holds for
    any wheel)"""
    return CG.frozen_grasps(slot, roles=roles)


# ------------------------------------------------------------------ the assembler
class ShopAssembler(REF.Assembler):
    """`reference.Assembler` with the wheels (module docstring). wheel_clock: "known" (the privileged hub angle and
    pocket angle: the reference), "search" (the hub's angle taken as its model angle; on the stop the light-push
    sweep finds the hex), "blind" (the same believed clock, no search: a wheel on its stop is let go there)"""

    def __init__(self, cell, log=print, wheel_clock="known", max_sim_s=MAX_SIM_S, **kw):
        if wheel_clock not in ("known", "search", "blind"):
            raise ValueError("wheel_clock %r" % (wheel_clock,))
        super().__init__(cell, log=log, **kw)
        self.wheel_clock = wheel_clock
        self.max_sim_s = float(max_sim_s)
        self.wheels = [u for u in self.kit["loose"] if self.type_of.get(u) == "WHL"]
        self.wheel_rest = {u: self.unit_T(u) for u in self.wheels}       # where each wheel was picked from (bench)
        self.carrier_body = {}                                           # the moving wheel's hub body id (monitor)
        self.wheel_cands = {s["id"]: wheel_grasps(s["id"]) for s in self.kit["slots"] if s["id"].startswith("whl_")}

    # ------------------------------------------------------------------ helpers
    def spec_of(self, u):
        return WHEEL_SPEC if self.type_of[u] == "WHL" else super().spec_of(u)

    def sim_left(self):
        return self.max_sim_s - float(self.d.time)

    def hub_of(self, sid):
        return self.cell.slots[sid]["carrier"]

    def wheel_target(self, sid, u, k):
        """(T_home, insertion) of wheel u on slot sid at hex orientation k as the program believes it: "known" the
        live target (the hub at its present angle), else the hub at its model angle (q = 0) on the chassis as it is"""
        if self.wheel_clock == "known":
            tg = self.cell.live_target(sid, u, k=k)
            return tg["T"], np.asarray(tg["insertion"], float)
        s = self.cell.slots[sid]
        c = self.cell.candidate(sid, u)
        kit = self.kit
        hub = s["carrier"]
        Tch = self.cell.body_T("chassis")
        T_hub0 = Tch @ _inv(np.asarray(kit["bodies"]["chassis"]["T_design"], float)) @ np.asarray(kit["bodies"][hub]["T_design"], float)
        F = T_hub0 @ np.asarray(s["frame_in_carrier"], float)
        pitch = math.radians(float(s.get("pitch_deg", 60.0)))
        T = F @ _rotz4(k * pitch) @ _inv(np.asarray(c["frame_in_part"], float))
        return T, F[:3, 2] / np.linalg.norm(F[:3, 2])

    def wheel_eval(self, sid, u):
        """the coupling's evaluation of wheel u on slot sid now (axial, lateral, tilt, clock residual)"""
        x = self.sm.by_id[sid]
        c = [cc for cc in x["cands"] if cc["unit"] == u][0]
        return self.sm.evaluate(x, c)

    def hub_force(self, u, hub):
        """the normal contact force of wheel u on the hub body (N)"""
        m, d = self.m, self.d
        wb, hb = m.body(u).id, m.body(hub).id
        f = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 == wb and b2 == hb) or (b1 == hb and b2 == wb):
                self.mj.mj_contactForce(m, d, i, self._f6)
                f += abs(float(self._f6[0]))
        return f

    def k_order(self, sid, u, G):
        """the hex orientations k of the hub ordered by how far the tool's roll about the axis at the hub is from the
        grasp's design roll (the designed wheel at its design pose on this hub, the hub at q = 0): the arm stays near
        the posture the grasp was validated in"""
        kit = self.kit
        s = self.cell.slots[sid]
        des = s.get("designed_unit") or CG.CORNER_WHEEL[sid.split("_", 1)[1]]
        Tch = self.cell.body_T("chassis")
        T_des = Tch @ _inv(np.asarray(kit["bodies"]["chassis"]["T_design"], float)) @ np.asarray(kit["bodies"][des]["T_design"], float)
        R0 = (T_des @ G)[:3, :3]
        dev = []
        for k in range(int(round(360.0 / float(s.get("pitch_deg", 60.0))))):
            T, _ins = self.wheel_target(sid, u, k)
            dev.append((REF.rotation_angle(R0.T @ (T @ G)[:3, :3]), k))
        dev.sort()
        return [k for _a, k in dev]

    # ------------------------------------------------------------------ the monitor (part I's + the wheel's carrier rule)
    def _monitor(self):
        super()._monitor()
        sid, mov = self.work.get("slot"), (self.work.get("held") or self.work.get("moving"))
        hb = self.carrier_body.get(sid)
        if hb is None or mov is None or self.type_of.get(mov) != "WHL":
            return
        m, d = self.m, self.d
        wb = m.body(mov).id
        ub = self.cell.unit_of_body
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if wb not in (b1, b2):
                continue
            ob = b2 if b1 == wb else b1
            if ob == hb or ub.get(ob) != "base":
                continue
            self.mj.mj_contactForce(m, d, i, self._f6)
            fn = abs(float(self._f6[0]))
            if fn < REF.CONTACT_N:
                continue
            n1, n2 = m.body(b1).name, m.body(b2).name
            key = (self.work["phase"], sid, n1, n2, "wheel-carrier")
            if not self.contacts or self.contacts[-1]["key"] != key:
                self.contacts.append({"key": key, "t": round(float(d.time), 2), "phase": self.work["phase"], "slot": sid,
                                      "bodies": [n1, n2], "geoms": [m.geom(g1).name, m.geom(g2).name], "force_N": round(fn, 1),
                                      "rule": "the moving wheel against the base off its carrier hub"})

    # ------------------------------------------------------------------ the wheel step
    def wheel_feasible(self, u, sid, limit=None):
        """[(G, w, desc, entry, k, q_pick, q_pg, q_pre)] in the candidates' order (grasp tier, then k nearest the design
        roll), and the rejections"""
        T_pick = self.unit_T(u)
        spec = WHEEL_SPEC
        out, rejected = [], {}
        for G, w, desc, e in self.wheel_cands[sid]:
            q_pick, q_pg, r1 = self.pick_checks(u, T_pick, G, w, spec)
            if q_pick is None:
                rejected[desc] = "pick: %s" % (r1,)
                continue
            n_ok = 0
            for k in self.k_order(sid, u, G):
                T_home, ins = self.wheel_target(sid, u, k)
                q_pre, hold, _mrg = self.place_checks(sid, u, G, w, T_home, ins, q_seed=q_pick)
                if q_pre is None:
                    rejected["%s k %d" % (desc, k)] = "hub: %s" % (hold,)
                    continue
                out.append((G, w, desc, e, k, q_pick, q_pg, q_pre))
                n_ok += 1
                if n_ok >= K_PER_GRASP:
                    break
            if limit and len(out) >= limit:
                break
        return out, rejected

    def place_wheel(self, u, sid):
        """pick wheel u where it rests and fasten it on slot sid's hub (module docstring); returns the step's record"""
        sm = self.sm
        spec = WHEEL_SPEC
        hub = self.hub_of(sid)
        rec = {"slot": sid, "unit": u, "kind": "wheel", "t0_s": round(float(self.d.time), 3), "placed_by": "nobody",
               "attempts": [], "carrier_unit": "base", "hub": hub, "hub_deg_start": round(self.cell.hub_deg(hub), 3),
               "wheel_clock": self.wheel_clock}
        stats = self._new_stats(sid, u)
        n_contacts0 = len(self.contacts)
        self.carrier_body = {sid: self.m.body(hub).id}
        self.work.update({"unit": u, "slot": sid, "held": None, "moving": None})
        self._phase("plan")
        t_wall = time.time()
        tried = set()
        for round_ in range(3):
            if self.sim_left() <= 0:
                rec["why"] = "simulated-time budget spent"
                break
            t_chk = time.time()
            feas, rejected = self.wheel_feasible(u, sid, limit=FEASIBLE_LIMIT + len(tried))
            feas = [f for f in feas if (f[3]["row"], f[4]) not in tried] or feas
            rec.setdefault("grasps", []).append({"candidates": len(self.wheel_cands[sid]), "feasible": len(feas),
                                                 "check_wall_s": round(time.time() - t_chk, 1),
                                                 "rejected": {k_: str(v) for k_, v in list(rejected.items())[:8]}})
            if not feas:
                rec["why"] = "no feasible grasp (%s)" % (sorted(set(str(v) for v in rejected.values()))[:4],)
                self.log("  %s %s: %s" % (sid, u, rec["why"]))
                break
            G, w, desc, e, k, q_pick, q_pg, q_pre = feas[0]
            tried.add((e["row"], k))
            att = {"round": round_, "grasp": desc, "row": e["row"], "k": int(k), "width_mm": round(w * 1e3, 2),
                   "feasible": ["%s k %d" % (f[2], f[4]) for f in feas][:8], "hub_deg": round(self.cell.hub_deg(hub), 3)}
            rec["attempts"].append(att)
            ok, why = self.pick(u, G, w, q_pick, q_pg, spec)
            if not ok:
                att["result"] = why
                self.log("  %s %s: pick failed (%s)" % (sid, u, why))
                if self.ct.width() < w + 0.5 * spec["open_extra_m"]:
                    self.let_go(w, spec)
                continue
            att["held"] = self.held_report(u, G)
            ok, why = self.insert_wheel(sid, u, w, att, k, q_branch=q_pre)
            att["result"] = why
            if ok:
                rec["placed_by"] = "ur"
                break
            self.log("  %s %s: not fastened (%s)" % (sid, u, why))
            if self.work["held"] == u:
                att["put_back"] = self.put_back(u, w)
            if sm.state[sid] in ("LOCKED", "LOCKED_OFF"):
                break
        stt = sm.rec[sid]
        rec["slot_state"] = {"state": sm.state[sid], "fastened": stt["fastened"], "achieved": stt["achieved"],
                             "pos_err_mm": stt["pos_err_mm"], "ang_err_deg": stt["ang_err_deg"], "t_s": stt["t_s"],
                             "key_k": stt.get("key_k")}
        rec["coupling_events"] = [(round(float(t), 3), e_) for t, e_ in sm.events if t is not None and t >= rec["t0_s"] - 1e-9
                                  and (sid in e_ or u in e_)]
        stats.pop("_hist", None)
        rec.update({"state": sm.state[sid], "t1_s": round(float(self.d.time), 3), "wall_s": round(time.time() - t_wall, 1),
                    "hub_deg_end": round(self.cell.hub_deg(hub), 3), "timeline": stats["timeline"], "capture": stats["capture"],
                    "forces": {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in stats.items()
                               if k_.endswith("_N") or k_.endswith("_pair")},
                    "unplanned_contacts": [{k_: v for k_, v in c.items() if k_ != "key"} for c in self.contacts[n_contacts0:]]})
        rec["sim_s"] = round(rec["t1_s"] - rec["t0_s"], 3)
        self.work.update({"unit": None, "slot": None, "held": None, "moving": None, "phase": "between"})
        self.carrier_body = {}
        self.step_stats = None
        self.log("  %-8s %-4s -> %s (%s) sim %.1f s wall %.0f s, capture %s, unplanned %d" % (
            sid, u, rec["state"], rec["attempts"][-1].get("result") if rec["attempts"] else rec.get("why"), rec["sim_s"],
            rec["wall_s"], None if rec["capture"] is None else "%.2f of the window" % rec["capture"]["ratio_max"],
            len(rec["unplanned_contacts"])))
        return rec

    def insert_wheel(self, sid, u, w, att, k, q_branch=None):
        """carry to the pre-insertion on hex orientation k, push along the axis until captured (or on the stop: the
        search), on home until locked, let go and back out (module docstring); (ok, why)"""
        cell, ct, sm = self.cell, self.ct, self.sm
        spec = WHEEL_SPEC
        hub = self.hub_of(sid)
        pre = REF.PRE_INSERT_M
        T_home, ins = self.wheel_target(sid, u, k)
        T_pre = T_home.copy()
        T_pre[:3, 3] -= ins * pre
        self._phase("carry")
        if self.carry_to(u, T_pre, w, spec, q_branch=q_branch) is None:
            return False, "no path to the pre-insertion"
        self._phase("insert")
        att["hub_deg_at_insert"] = round(cell.hub_deg(hub), 3)
        captured, x = False, pre
        searched = 0
        for attempt in range(WHEEL_RETRIES + 1):
            Gi = self.tcp_T_inv_unit(u)
            ct.set_stiffness(REF.INSERT_STIFFNESS)
            ct.set_mode("cartesian")
            x = pre
            stall, result, a = 0, None, None
            n_max = int((pre + 0.02) / (REF.INSERT_SPEED_M_S * cell.tick_s)) + 200
            for _ in range(n_max):
                T_home, ins = self.wheel_target(sid, u, k)
                x = max(-REF.CAPTURE_MARGIN_M, x - REF.INSERT_SPEED_M_S * cell.tick_s)
                T_u = T_home.copy()
                T_u[:3, 3] -= ins * x
                T_tcp = T_u @ Gi
                ct.command_cartesian(T_tcp[:3, 3], quat_wxyz(T_tcp[:3, :3]))
                cell.tick()
                if sm.state[sid] in REF.ENGAGED:
                    result = "captured"
                    break
                a = float((T_home[:3, 3] - self.unit_T(u)[:3, 3]) @ ins)       # the wheel short of home along the axis
                if STOP_BAND_M[0] <= a <= STOP_BAND_M[1] and self.hub_force(u, hub) > STOP_DETECT_N:
                    result = "stop"                 # on the stop, read by the push (before the arm presses harder)
                    break
                stall = stall + 1 if a - x > WHEEL_STALL_M else 0
                if stall > WHEEL_STALL_TICKS:
                    result = "stop" if STOP_BAND_M[0] <= a <= STOP_BAND_M[1] else "blocked"
                    break
            if result == "captured":
                captured = True
                break
            r = self.wheel_eval(sid, u)
            att.setdefault("stalls", []).append({"at_mm": None if a is None else round(a * 1e3, 2), "what": result or "no stall",
                                                 "clock_res_deg": round(math.degrees(r["clock_res"]), 2),
                                                 "lateral_mm": round(r["pos_err"] * 1e3, 2), "t_s": round(float(self.d.time), 2)})
            if result == "stop":
                if self.wheel_clock == "blind":
                    break
                if searched < 2:
                    searched += 1
                    self._phase("search")
                    ok_s, srec = self.wheel_search(sid, u, k, a, Gi)
                    att.setdefault("searches", []).append(srec)
                    if ok_s:
                        captured = True
                        break
                    # not caught: back out, re-read the turn, push again
                    self._phase("insert")
                    T_back = self.tcp_T()
                    T_back[:3, 3] -= ins * SEARCH_BACKOFF_M
                    self.move_line(T_back, check=False, speed=0.02)
                    if self.wheel_clock == "known":
                        k = int(self.wheel_eval(sid, u)["k"])
                    pre = SEARCH_BACKOFF_M + max(0.0, a)
                    continue
                break
            T_back = self.tcp_T()
            T_back[:3, 3] -= ins * BACKOFF_M
            self.move_line(T_back, check=False, speed=0.02)
            pre = BACKOFF_M + max(0.0, a if a is not None else x)
        ct.set_stiffness(1.0)
        if not captured:
            ct.set_mode("joint")
            if self.wheel_clock == "blind" and att.get("stalls") and att["stalls"][-1]["what"] == "stop":
                # the blind mode's end: the wheel let go on its stop (it stays there, not fastened)
                att["state_at_release"] = sm.state[sid]
                self.release(sid, u, w, spec)
                return False, "on its stop, let go (blind)"
            return False, "not captured (%s)" % ([s_["what"] for s_ in att.get("stalls", [])],)
        # captured: on home with the wheel in the hand until the cell locks it, then let go
        self._phase("captured")
        att["hub_deg_at_capture"] = round(cell.hub_deg(hub), 3)
        Gi = self.tcp_T_inv_unit(u)
        ct.set_mode("cartesian")
        ct.set_stiffness(REF.INSERT_STIFFNESS)
        kc = int(self.wheel_eval(sid, u)["k"])          # the hex orientation the coupling pulls the wheel to
        for _ in range(REF.LOCK_HOLD_TICKS):
            tg = cell.live_target(sid, u, k=kc)
            T_home, ins = tg["T"], np.asarray(tg["insertion"], float)
            x = max(-REF.CAPTURE_MARGIN_M, x - REF.INSERT_SPEED_M_S * cell.tick_s)
            T_u = T_home.copy()
            T_u[:3, 3] -= ins * x
            T_hold = T_u @ Gi
            ct.command_cartesian(T_hold[:3, 3], quat_wxyz(T_hold[:3, :3]))
            cell.tick()
            if sm.state[sid] in ("LOCKED", "LOCKED_OFF"):
                break
        ct.set_stiffness(1.0)
        ct.set_mode("joint")
        att["state_at_release"] = sm.state[sid]
        return self.release(sid, u, w, spec)

    def wheel_search(self, sid, u, k, a_stop, Gi):
        """on the stop: hold the push at SEARCH_PUSH_N and turn the wheel about the axis (module docstring); returns
        (captured, record)"""
        cell, ct, sm = self.cell, self.ct, self.sm
        hub = self.hub_of(sid)
        r = self.wheel_eval(sid, u)
        res = math.degrees(float(r["clock_res"]))
        if self.wheel_clock == "known":
            first = -res - math.copysign(SEARCH_OVERSHOOT_DEG, res if res != 0 else 1.0)
            legs = [first, -math.copysign(SEARCH_SWEEP_DEG, first)]
        else:
            legs = [SEARCH_SWEEP_DEG, -SEARCH_SWEEP_DEG]
        hub0 = cell.hub_deg(hub)
        T_home, ins = self.wheel_target(sid, u, k)
        xcmd = float(a_stop)
        turned, caught_t = 0.0, None
        F_hist = []
        ct.set_stiffness(REF.INSERT_STIFFNESS)
        ct.set_mode("cartesian")
        t0 = float(self.d.time)
        for leg in legs:
            n = int(abs(leg - turned) / SEARCH_TURN_DEG_S / cell.tick_s) + 30
            for _ in range(n):
                F = self.hub_force(u, hub)
                F_hist.append(F)
                xcmd -= (SEARCH_PUSH_N - F) * SEARCH_GAIN * cell.tick_s * 10.0
                xcmd = min(a_stop + 0.002, max(a_stop - 0.008, xcmd))
                step = SEARCH_TURN_DEG_S * cell.tick_s
                if abs(leg - turned) > step:
                    turned += math.copysign(step, leg - turned)
                else:
                    turned = leg
                T_home, ins = self.wheel_target(sid, u, k)
                T_u = T_home.copy()
                T_u[:3, 3] -= ins * xcmd
                p = T_home[:3, 3]
                Rm = _axis_rot(ins, math.radians(turned))
                T_rot = np.eye(4)
                T_rot[:3, :3] = Rm @ T_u[:3, :3]
                T_rot[:3, 3] = p + Rm @ (T_u[:3, 3] - p)
                T_tcp = T_rot @ Gi
                ct.command_cartesian(T_tcp[:3, 3], quat_wxyz(T_tcp[:3, :3]))
                cell.tick()
                if sm.state[sid] in REF.ENGAGED:
                    caught_t = float(self.d.time)
                    break
            if caught_t is not None:
                break
        Fs = F_hist[30:]                          # the regulated push: after the first 0.3 s
        out = {"clock_res_deg_at_start": round(res, 2), "legs_deg": [round(v, 2) for v in legs], "hand_turn_deg": round(turned, 2),
               "hub_turn_deg": round(cell.hub_deg(hub) - hub0, 3), "push_N_at_start": round(float(F_hist[0]), 2) if F_hist else None,
               "push_N_max_all": round(float(np.max(F_hist)), 2) if F_hist else None,
               "push_N_mean": round(float(np.mean(Fs)), 2) if Fs else None, "push_N_max": round(float(np.max(Fs)), 2) if Fs else None,
               "caught": caught_t is not None, "s": round(float(self.d.time) - t0, 2)}
        return caught_t is not None, out

    def put_back(self, u, w):
        """set a wheel the arm holds back where it was picked from (the reverse of the pick: planned to 80 mm above its
        rest, the line down, opened, backed out upwards); returns what happened"""
        spec = WHEEL_SPEC
        self._phase("put back", held=u)
        T_rest = self.wheel_rest[u]
        Gt = self.tcp_T_inv_unit(u)                           # the tool in the unit as held now
        if self.carry_to(u, _lift(T_rest, spec["pre_grasp_m"] + 0.002), w, spec) is None:
            self.work["held"] = None
            self.let_go(w, spec)
            return "no path above the bench spot: let go where it was"
        self.move_line(_lift(T_rest, 0.002) @ Gt, speed=REF.APPROACH_SPEED_M_S, check=False)
        self.work["held"] = None
        self.grip(min(self.max_width, w + spec["open_extra_m"]), settle_ticks=10)
        T_out = self.tcp_T()
        T_out[2, 3] += spec["pre_grasp_m"]
        self.move_line(T_out, check=False)
        return "put back on its bench spot"

    # ------------------------------------------------------------------ the episode
    def run_steps(self, steps, retry=True, out=None):
        """execute `shop_steps` entries in order, then one retry pass over the slots that are not locked; returns the
        step records (appended to `out` as they finish: a report survives an interrupted run)"""
        recs = out if out is not None else []
        for st in steps:
            if self.sim_left() <= 0:
                recs.append({"slot": st["slot"], "unit": st["unit"], "kind": st["kind"], "state": self.sm.state[st["slot"]],
                             "skipped": "simulated-time budget spent", "attempts": [], "sim_s": 0.0})
                continue
            self.log("step %s <- %s" % (st["slot"], st["unit"]))
            recs.append(self.place_wheel(st["unit"], st["slot"]) if st["kind"] == "wheel" else self.place_unit(st["unit"]))
        if retry:
            for st in steps:
                if self.sm.state[st["slot"]] in ("LOCKED", "LOCKED_OFF") or self.sim_left() <= 0:
                    continue
                if st["unit"] in self.sm.scrapped:
                    continue
                if st["kind"] == "wheel" and self.sm.used.get(st["unit"]) not in (None, st["slot"]):
                    continue
                self.log("retry %s <- %s" % (st["slot"], st["unit"]))
                r = self.place_wheel(st["unit"], st["slot"]) if st["kind"] == "wheel" else self.place_unit(st["unit"])
                r["retry"] = True
                recs.append(r)
        return recs


def _rotz4(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


def _axis_rot(axis, ang):
    """rotation matrix by `ang` (rad) about the unit vector `axis`"""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def _lift(T, dz):
    T2 = np.array(T, float).copy()
    T2[2, 3] += dz
    return T2


def _install_stall_guard(cell):
    """wrap the cell's tick so that a tick which does not advance the simulation (the server's sim cap ended the
    episode: its routed tick becomes a no-op) raises SimStalled; returns the function that removes the guard"""
    orig = cell.tick_one

    def guarded():
        t = float(cell.d.time)
        orig()
        if float(cell.d.time) <= t:
            raise SimStalled("the simulation did not advance at %.2f s" % t)
    cell.tick_one = guarded

    def remove():
        cell.tick_one = orig
    return remove


def program(cell, order="shop", log=print, planner_seed=0, end_ticks=300, assembler=None, wheel_clock="known",
            assignment=None, retry=True, max_sim_s=MAX_SIM_S):
    """run the whole episode on `cell` (a CarCell with the UR): the steps of `shop_steps(kit, order, assignment)` and
    one retry pass; returns the report"""
    t0 = time.time()
    s0 = float(cell.d.time)
    asm = assembler or ShopAssembler(cell, log=log, planner_seed=planner_seed, wheel_clock=wheel_clock, max_sim_s=max_sim_s)
    steps = shop_steps(cell.kit, order, assignment)
    ended = "done"
    recs = []
    remove_guard = _install_stall_guard(cell)
    try:
        asm.move_q(asm.q_ready)
        recs = asm.run_steps(steps, retry=retry, out=recs)
        asm.work.update({"phase": "end", "unit": None, "slot": None, "held": None, "moving": None})
        asm.go(asm.q_ready)
        asm.tick(end_ticks)
    except SimStalled as e:
        ended = "sim stalled: %s" % (e,)
        log("the program ended: %s" % ended)
    finally:
        remove_guard()
    sm = cell.sm
    slots = [s["id"] for s in cell.kit["slots"]]
    rep = {"order": order if isinstance(order, str) else list(order), "wheel_clock": asm.wheel_clock, "ended": ended,
           "assignment": {st["slot"]: st["unit"] for st in steps if st["kind"] == "wheel"},
           "score": round(sm.score(), 4), "states": {s: sm.state[s] for s in slots},
           "all_locked": all(sm.state[s] == "LOCKED" for s in slots), "n_locked": sum(1 for s in slots if sm.state[s] == "LOCKED"),
           "steps": recs, "events": [(round(float(t), 3), e) for t, e in sm.events if t is not None],
           "unplanned_contacts": [{k: v for k, v in c.items() if k != "key"} for c in asm.contacts],
           "sim_s": round(float(cell.d.time) - s0, 2), "wall_s": round(time.time() - t0, 1)}
    log("score %.3f, %d / %d locked, sim %.0f s, wall %.0f s, unplanned contacts %d" % (
        rep["score"], rep["n_locked"], len(slots), rep["sim_s"], rep["wall_s"], len(rep["unplanned_contacts"])))
    return rep


def reference_assemble(cell, plan=None, log=print, order="shop", **kw):
    """the server's entry (`CarInfoServer.reference_assemble`): the whole privileged program on this episode's cell;
    `plan` (the hidden bench plan) is not needed: the program reads the units' poses"""
    return program(cell, order=order, log=log, **kw)
