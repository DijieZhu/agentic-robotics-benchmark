"""recordkit.py -- synthetic cell records for the grader's replay cases (freeflyer_uncrewed_station_caretaking).

A Scenario holds one site -- one of the two fixed example stations:
sites/site_l1xp.json (L1 x P: pad-type units, CAU-L1 inhibited, three pouches) and sites/site_h2xs.json (H2 x S:
element-type CAU-H2 with fan stops allowed, two pouches) -- and piecewise-constant timelines of every item, the
robot's docking, the fans and the events.  record() samples the world every cell.record_period_s from 0 to the end
of the settle (program end + cell.settle_after_exit_s, both read from the runtime copy of the task's numbers) and
returns a caretaking.record/1 dict (the record format).  Nothing is simulated: the replays test how the grader
reads a record.  The loss-of-control events are derived from the samples with the grader's own predicate (the
cell's rule); a case can add events or drop them.

Item states: on_face(face, area, disp[, grip]) -- sealed on a suction face (return inlet, holding screen SB-<M>-S,
pouch pad SB-<M>-P1 / -P2); held(disp) -- in the closed gripper, off every face; free(disp) -- neither.  'disp' is the
centroid's distance from its t = 0 position [m].  The covered fraction of an inlet = the items' sealed area on it
over the opening area (grader_config grader.covered).

Reference runs: reference('P') / reference('S'); the capability ladder: ladder(tier, level).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
CABINCELL = os.path.join(TESTS, "cabincell")
SITE_FILES = {"P": os.path.join(HERE, "sites", "site_l1xp.json"), "S": os.path.join(HERE, "sites", "site_h2xs.json")}
INLETS = ("ri_l2_1", "ri_l2_2", "ri_l2_3", "ri_h2_1", "ri_h2_2", "ri_l1_1", "ri_l1_2", "ri_l1_3")
MODULES = ("l2", "h2", "l1")
UNITS = tuple("cau_" + m for m in MODULES)
REF_ENERGY_WH = {"P": 41.0, "S": 37.3}              # the reference runs' energy (the E basis of every replay)
COLUMN_M2 = 0.002                                    # sealed area of one sub-point column of a pouch on an inlet end


def load_grade():
    """grade.py of this published directory, loaded by path (it loads predicates.py next to it)."""
    name = "caretaking_grade_" + hashlib.sha256(CABINCELL.encode()).hexdigest()[:12]
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, os.path.join(CABINCELL, "grade.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


G = load_grade()
P = G.P
CONFIG = G.load_config()
RUNTIME = G.load_runtime()
PERIOD = float(RUNTIME["values"]["cell.record_period_s"])
SETTLE = float(RUNTIME["values"]["cell.settle_after_exit_s"])
AUTO_RESTART = float(RUNTIME["values"]["station.fan_stop.auto_restart_s"])
OPENING = float(CONFIG["values"]["grader.covered"]["opening_area_m2"])
LOSS = CONFIG["values"]["grader.loss_of_control"]


def on_face(face: str, area: float, disp: float, grip: bool = False) -> dict:
    return dict(grip=bool(grip), seal={face: float(area)}, disp=float(disp))


def held(disp: float) -> dict:
    return dict(grip=True, seal={}, disp=float(disp))


def free(disp: float) -> dict:
    return dict(grip=False, seal={}, disp=float(disp))


def evidence(blocker_cover: float, dist: float = 1.0, inc: float = 20.0, px: float = 410.0, occl: float = 0.0, in_frame: bool = True) -> dict:
    return dict(in_frame=in_frame, dist_m=dist, incidence_deg=inc, long_px=px, occluded=occl, blocker_cover=blocker_cover)


def load_site(tier: str) -> dict:
    with open(SITE_FILES[tier], encoding="utf-8") as fh:
        return json.load(fh)


class Scenario:
    """One run on one site.  Times are mission seconds; every state applies from its time on."""

    def __init__(self, tier: str, label: str = "case"):
        self.tier, self.label = tier, label
        self.site = load_site(tier)
        T = self.site["truth"]
        self.blk, self.unit = T["blocked_inlet"], T["blocked_unit"]
        self.blocker_ref, self.onset = T["blocker_ref"], float(T["onset_min"])
        self.pouches = list(T["pouches"])
        self.small = {p["id"]: p for p in self.site["small_items"]}
        b = self.site["blocker"]
        self.blocker_fp = float(b["size"][0]) * float(b["size"][1])
        self.tracks = {"blocker": [(0.0, on_face(self.blk, float(T["cover_m2"]), 0.0003))]}
        for p in self.pouches:
            s = self.small[p]
            self.tracks[p] = [(0.0, on_face(s["inlet"], COLUMN_M2 * int(s["columns"]), 0.0002))]
        self.undock_t = 31.0                  # None: never leaves the berth
        self.dock_t = None                    # first DOCKED
        self.dock_battery = None
        self.later_undocks = []               # [(t_undock, t_dock or None)] after the first docking
        self.battery0 = float(self.site["robot"]["battery_pct"])
        self.battery_free_end = self.battery0 - 22.0
        self.charge_rate = 0.0                # %/s while docked after DOCKED
        self.program_end = 1400.0
        self.exit = "returned"
        self.energy_wh = REF_ENERGY_WH[tier]
        self.report = None
        self.frames = []
        self.fan_stops = []
        self.violations = []
        self.requests = []
        self.calls = []
        self.hard = []
        self.keepout_s = 0.0
        self.loss_extra = {}
        self.drop_loss = False
        self.contacts_after_dock = {}
        self.final_fans = {}
        self.budgets = dict(program_compute_s=110.0, colour_frames=36.0, depth_frames=30.0, capture_calls=5.0, light_requests=2.0,
                            forbidden_calls=0.0)
        self.metrics = {}
        self.state_cut_s = None               # drop the samples after this time (an incomplete record)
        self.calls.append(dict(t=0.0, call="undock", code="UNDOCKED"))

    # ---------------------------------------------------------------------------------------------------- timelines
    def at(self, item: str, t: float, state: dict) -> "Scenario":
        tr = [(a, s) for a, s in self.tracks[item] if abs(a - t) > 1e-9]
        tr.append((float(t), state))
        tr.sort(key=lambda x: x[0])
        self.tracks[item] = tr
        return self

    def clear_after(self, item: str, t: float) -> "Scenario":
        self.tracks[item] = [(a, s) for a, s in self.tracks[item] if a <= t + 1e-9]
        return self

    def state_at(self, item: str, t: float) -> dict:
        cur = self.tracks[item][0][1]
        for a, s in self.tracks[item]:
            if a <= t + 1e-9:
                cur = s
            else:
                break
        return cur

    def collect(self, p: str, t_grip: float, face: str, t_place: float, frac: float = 0.98, disp: float = 2.3) -> "Scenario":
        """capture DONE on the handle at t_grip (still on its inlet), pulled off 2 s later, pressed onto ``face`` at
        t_place covering ``frac`` of its own footprint, released 1 s later."""
        s = self.small[p]
        fp = float(s["size"][0]) * float(s["size"][1])
        on_inlet = COLUMN_M2 * int(s["columns"])
        self.at(p, t_grip, on_face(s["inlet"], on_inlet, 0.0002, grip=True))
        self.at(p, t_grip + 2.0, held(0.35))
        self.at(p, t_place, on_face(face, frac * fp, disp, grip=True))
        self.at(p, t_place + 1.0, on_face(face, frac * fp, disp))
        self.calls.append(dict(t=t_grip, call="capture", code="DONE"))
        self.calls.append(dict(t=t_place - 5.0, call="fly_rel", code="SUCCESS"))
        return self

    def remove_blocker(self, t_capture: float, t_off: float, disp: float = 0.30) -> "Scenario":
        """capture DONE on the blocker at t_capture (still sealed on the inlet), off the inlet and in the hand at t_off."""
        T = self.site["truth"]
        self.at("blocker", t_capture, on_face(self.blk, float(T["cover_m2"]), 0.0003, grip=True))
        self.at("blocker", t_off, held(disp))
        self.calls.append(dict(t=t_capture, call="capture", code="DONE"))
        return self

    def place_blocker(self, face: str, t_press: float, seal: float, t_release: float, disp: float = 2.29) -> "Scenario":
        self.at("blocker", t_press, on_face(face, seal, disp, grip=True))
        self.at("blocker", t_release, on_face(face, seal, disp))
        self.calls.append(dict(t=t_press - 6.0, call="fly_rel", code="SUCCESS"))
        self.calls.append(dict(t=t_release, call="gripper", code="DONE"))
        return self

    def dock(self, t: float, battery: float) -> "Scenario":
        self.dock_t, self.dock_battery = float(t), float(battery)
        self.calls.append(dict(t=float(t), call="dock", code="DOCKED"))
        return self

    def no_dock(self) -> "Scenario":
        self.dock_t = self.dock_battery = None
        self.calls = [c for c in self.calls if c.get("call") != "dock"]
        return self

    def add_frame(self, fid: int, camera: str, t: float, ev: dict) -> "Scenario":
        self.frames.append(dict(frame_id=int(fid), camera=camera, t=float(t), evidence=dict(ev)))
        self.calls.append(dict(t=float(t), call="camera", code="OK"))
        return self

    def set_report(self, t: float, inlet, item_ref, onset_min, frame_id) -> "Scenario":
        self.report = dict(t=float(t), inlet=inlet, item_ref=item_ref, onset_min=onset_min, frame_id=frame_id)
        self.calls.append(dict(t=float(t), call="report", code="RECORDED"))
        return self

    def fan_stop(self, unit: str, t_stop: float, t_restart, by: str = "robot") -> "Scenario":
        """An accepted stop request at t_stop; restart by the robot (a request) or by the station ('auto')."""
        target = unit[4:].upper()
        self.fan_stops.append(dict(unit=unit, t_stop=float(t_stop), t_restart=(None if t_restart is None else float(t_restart)),
                                   by=(by if t_restart is not None else None)))
        self.requests.append(dict(t=float(t_stop), kind="cau_fan", target=target, value="off", code="ACCEPTED",
                                  auto_restart_at=float(t_stop) + AUTO_RESTART))
        if t_restart is not None and by == "robot":
            self.requests.append(dict(t=float(t_restart), kind="cau_fan", target=target, value="on", code="ACCEPTED"))
        return self

    def request(self, t: float, kind: str, target: str, value: str, code: str) -> "Scenario":
        self.requests.append(dict(t=float(t), kind=kind, target=target, value=value, code=code))
        return self

    # ---------------------------------------------------------------------------------------------------- world
    def docked(self, t: float) -> bool:
        if self.undock_t is None or t < self.undock_t - 1e-9:
            return True
        if self.dock_t is None or t < self.dock_t - 1e-9:
            return False
        for tu, td in self.later_undocks:
            if t >= tu - 1e-9 and (td is None or t < td - 1e-9):
                return False
        return True

    def battery(self, t: float) -> float:
        if self.dock_t is not None:
            if t <= self.dock_t:
                f = t / self.dock_t if self.dock_t > 0 else 1.0
                return self.battery0 + (self.dock_battery - self.battery0) * f
            return self.dock_battery + self.charge_rate * (t - self.dock_t)
        end = self.program_end + SETTLE
        return self.battery0 + (self.battery_free_end - self.battery0) * (t / end if end > 0 else 1.0)

    def fan_running(self, unit: str, t: float) -> bool:
        for e in self.fan_stops:
            if e["unit"] == unit and e["t_stop"] - 1e-9 <= t and (e["t_restart"] is None or t < e["t_restart"] - 1e-9):
                return False
        return True

    def times(self) -> list:
        end = self.program_end + SETTLE
        n = int(math.floor(end / PERIOD + 1e-9))
        ts = [round(k * PERIOD, 6) for k in range(n + 1)]
        if ts[-1] < end - 1e-9:
            ts.append(round(end, 6))
        if self.state_cut_s is not None:
            ts = [t for t in ts if t <= self.state_cut_s + 1e-9]
        return ts

    def record(self) -> dict:
        end = round(self.program_end + SETTLE, 6)
        state = []
        for t in self.times():
            items, cov = {}, {i: 0.0 for i in INLETS}
            for name in self.tracks:
                st = self.state_at(name, t)
                seal = {f: round(a, 6) for f, a in st["seal"].items() if a > 0}
                on = max(seal, key=lambda f: seal[f]) if seal else None
                items[name] = dict(in_gripper=bool(st["grip"]), seal_m2=seal, disp_m=round(st["disp"], 6), on_face=on)
                for f, a in seal.items():
                    if f in cov:
                        cov[f] += a / OPENING
            state.append(dict(t=t, inlets={i: round(min(1.0, v), 6) for i, v in cov.items()}, items=items,
                              fans={u: self.fan_running(u, t) for u in UNITS}, docked=self.docked(t),
                              battery_pct=round(self.battery(t), 4), robot_pos=[0.0, 0.0, 0.0]))
        smp = P.samples(dict(state=state))
        loss = {}
        if not self.drop_loss:
            for name in self.tracks:
                t0 = P.loss_of_control(smp, name, float(LOSS["disp_m"]), float(LOSS["min_s"]))
                if t0 is not None:
                    loss[name] = t0
        loss.update(self.loss_extra)
        last = state[-1]
        fans_end = dict(last["fans"])
        fans_end.update(self.final_fans)
        items_meta = {"blocker": dict(kind="blocker", register_ref=self.blocker_ref, footprint_m2=round(self.blocker_fp, 6), inlet=self.blk)}
        for p in self.pouches:
            s = self.small[p]
            items_meta[p] = dict(kind="pouch", register_ref=s["register_ref"], footprint_m2=round(float(s["size"][0]) * float(s["size"][1]), 6),
                                 inlet=s["inlet"])
        faces = {i: dict(kind="return_inlet", owner="cau_" + i.split("_")[1], module=i.split("_")[1]) for i in INLETS}
        for m in MODULES:
            M = m.upper()
            faces[f"SB-{M}-S"] = dict(kind="holding_screen", owner=f"SB-{M}", module=m)
            faces[f"SB-{M}-P1"] = dict(kind="pouch_pad", owner=f"SB-{M}", module=m)
            faces[f"SB-{M}-P2"] = dict(kind="pouch_pad", owner=f"SB-{M}", module=m)
        state_json = json.dumps(state, sort_keys=True).encode()
        bud = dict(self.budgets)
        bud["sim_s"] = round(self.program_end, 6)
        return {
            "schema": "caretaking.record/1",
            "site_id": self.site["site_id"],
            "run": dict(seed=0, perturb_um=0.0, cpu="replay", label=self.label),
            "spec": dict(runtime_sha256=RUNTIME["sha256"], grader_config_sha256=None, site_package_sha256=None),
            "items": items_meta,
            "faces": faces,
            "times": dict(program_end=round(self.program_end, 6), end=end, settle_s=SETTLE, exit=self.exit),
            "budgets_used": bud,
            "state": state,
            "frames": copy.deepcopy(self.frames),
            "report": copy.deepcopy(self.report),
            "station_requests": copy.deepcopy(self.requests),
            "fan_stops": copy.deepcopy(self.fan_stops),
            "violations": copy.deepcopy(self.violations),
            "dock": dict(t_docked=self.dock_t, battery_pct_at_docked=self.dock_battery, docked_at_end=bool(last["docked"]),
                         item_contacts_after_dock={k: bool(self.contacts_after_dock.get(k, False)) for k in self.tracks}),
            "energy_wh": self.energy_wh,
            "loss_of_control": loss,
            "hard_contacts": copy.deepcopy(self.hard),
            "keepout_s": self.keepout_s,
            "calls": sorted(copy.deepcopy(self.calls), key=lambda c: c["t"]),
            "final": dict(inlets=dict(last["inlets"]), fans=fans_end, docked=bool(last["docked"])),
            "hashes": dict(state_sha256=hashlib.sha256(state_json).hexdigest(),
                           qpos_sha256=hashlib.sha256(("qpos:" + self.label).encode()).hexdigest()),
            "metrics": dict(self.metrics),
        }

    def grade(self, ref_energy_wh=None) -> dict:
        ref = REF_ENERGY_WH[self.tier] if ref_energy_wh is None else ref_energy_wh
        return G.grade(self.record(), self.site, ref, CONFIG, RUNTIME)


# -------------------------------------------------------------------------------------------------------- references
def reference(tier: str, label: str = "reference") -> Scenario:
    """The reference's order: pouches on the way, report at the blocked inlet, the blocker (S: capture,
    fan stop, pull, restart; P: capture the loop, peel), the holding screen of the module, dock."""
    s = Scenario(tier, label)
    if tier == "P":                                        # L1 x P: pouch_3 on RI-L2-2, pouch_2 on RI-H2-1, pouch_1 on RI-L1-3
        s.collect("pouch_3", 150.0, "SB-L2-P1", 190.0)
        s.collect("pouch_2", 330.0, "SB-H2-P2", 370.0)
        s.add_frame(0, "FIC", 650.0, {s.blk: evidence(0.625), "ri_l1_3": evidence(0.0, dist=1.6, inc=35.0, px=250.0)})
        s.set_report(660.0, "RI-L1-2", s.blocker_ref, s.onset + 0.5, 0)
        s.collect("pouch_1", 700.0, "SB-L1-P2", 740.0)
        s.remove_blocker(850.0, 900.0)
        s.place_blocker("SB-L1-S", 940.0, 0.084, 942.0)
        s.dock(1330.0, 26.1)
        s.program_end = 1332.0
    else:                                                  # H2 x S: pouch_1 on RI-L2-1, pouch_2 on RI-L1-1
        s.collect("pouch_1", 150.0, "SB-L2-P1", 190.0)
        s.add_frame(0, "FIC", 520.0, {s.blk: evidence(0.625)})
        s.set_report(530.0, "RI-H2-2", s.blocker_ref, s.onset + 0.5, 0)
        s.remove_blocker(600.0, 614.0)
        s.fan_stop("cau_h2", 605.0, 628.0, by="robot")
        s.place_blocker("SB-H2-S", 680.0, 0.080, 682.0)
        s.collect("pouch_2", 800.0, "SB-L1-P1", 840.0)
        s.dock(1150.0, 31.0)
        s.program_end = 1152.0
    return s


def hold_to_end(tier: str, label: str = "hold") -> Scenario:
    """Ladder level 4 with every pouch stowed: the blocker removed and held in the closed gripper to the end; no docking."""
    s = reference(tier, label)
    s.clear_after("blocker", 900.0 if tier == "P" else 614.0)
    s.calls = [c for c in s.calls if c["call"] != "gripper"]
    s.no_dock()
    return s


def nodock(tier: str, label: str = "nodock") -> Scenario:
    """The reference without docking: the robot hovers 200 s after fixing the sheet, then the program exits."""
    s = reference(tier, label)
    s.no_dock()
    s.program_end = (942.0 if tier == "P" else 682.0) + 200.0
    return s


def u1_isolation(tier: str = "S", label: str = "u1") -> Scenario:
    """S: the fan stopped, the blocker held 2 cm off the face for 35 s (inlet clear), let go and sucked back onto the
    inlet after the restart; pouches stowed, docked.  D, U1, H, L; U2, V, K, E are 0 (the inlet is covered at the end)."""
    s = reference(tier, label)
    T = s.site["truth"]
    s.clear_after("blocker", 0.0)
    s.at("blocker", 600.0, on_face(s.blk, float(T["cover_m2"]), 0.0003, grip=True))
    s.at("blocker", 610.0, held(0.02))
    s.at("blocker", 645.0, free(0.02))
    s.at("blocker", 646.0, on_face(s.blk, float(T["cover_m2"]), 0.001))
    s.calls = [c for c in s.calls if not (c["call"] in ("fly_rel", "gripper") and 670.0 <= c["t"] <= 690.0)]
    return s


# -------------------------------------------------------------------------------------------------------- the ladder
LADDER = {"0a": 0.0, "0b": 0.0, "0c": 0.0, "1": 0.14, "2": 0.24, "3": 0.34, "4": 0.46, "5": 0.74, "6": 0.92, "7": 1.00}
LADDER_TEXT = {
    "0a": "does nothing (only steps until the time is up)", "0b": "undocks and docks again", "0c": "reports from the berth without going there",
    "1": "on site: evidence frame, inlet, item and onset reported", "2": "the blocker removed (then let go, no docking)",
    "3": "removed, let go, docked", "4": "removed and held in the closed gripper to the end (no docking)",
    "5": "the blocker fixed on a holding screen and docked", "6": "every other item on a pouch pad (economy outside the lines)",
    "7": "energy and fan stops of the reference's order"}


def ladder(tier: str, level: str) -> Scenario:
    """One capability more per level; levels 0a-0c are the three ways to score 0."""
    if level == "7":
        return reference(tier, f"ladder_{tier}_7")
    s = Scenario(tier, f"ladder_{tier}_{level}")
    blk_mod = s.blk.split("_")[1].upper()
    t_rep = 660.0 if tier == "P" else 530.0
    if level == "0a":
        s.undock_t = None
        s.calls = []
        s.program_end, s.exit = 2400.0, "time_up"
        s.energy_wh = 28.0
        return s
    if level == "0b":
        s.dock(200.0, s.battery0 - 2.0)
        s.program_end = 202.0
        s.energy_wh = 6.0
        return s
    if level == "0c":
        s.undock_t = None
        s.calls = []
        s.add_frame(0, "FIC", 5.0, {})
        s.set_report(10.0, f"RI-{blk_mod}-{s.blk.split('_')[2]}", s.blocker_ref, s.onset + 0.5, 0)
        s.program_end = 12.0
        s.energy_wh = 0.4
        return s
    s.add_frame(0, "FIC", t_rep - 10.0, {s.blk: evidence(0.625)})
    s.set_report(t_rep, f"RI-{blk_mod}-{s.blk.split('_')[2]}", s.blocker_ref, s.onset + 0.5, 0)
    t_cap, t_off = (850.0, 900.0) if tier == "P" else (600.0, 614.0)
    if level == "1":
        s.dock(1000.0, s.battery0 - 15.0)
        s.program_end = 1002.0
        return s
    s.remove_blocker(t_cap, t_off)
    if tier == "S":
        s.fan_stop("cau_h2", 605.0, 628.0, by="robot")
    if level in ("2", "3"):
        s.at("blocker", t_off + 5.0, free(0.6))
        s.at("blocker", t_off + 60.0, free(3.0))
        if level == "3":
            s.dock(t_off + 300.0, s.battery0 - 20.0)
            s.program_end = t_off + 302.0
        else:
            s.program_end = t_off + 300.0
        return s
    if level == "4":
        s.program_end = t_off + 400.0
        return s
    face = f"SB-{blk_mod}-S"
    s.place_blocker(face, t_off + 40.0, 0.084, t_off + 42.0)
    if level == "5":
        s.dock(t_off + 400.0, s.battery0 - 21.0)
        s.program_end = t_off + 402.0
        return s
    # level 6: the reference with the economy outside both lines (energy 2 x the reference's, 120 s of fan stops)
    s = reference(tier, f"ladder_{tier}_6")
    s.energy_wh = 2.0 * REF_ENERGY_WH[tier]
    if tier == "P":
        s.fan_stop("cau_l2", 400.0, 430.0)
        s.fan_stop("cau_l2", 560.0, 590.0)
        s.fan_stop("cau_h2", 450.0, 480.0)
        s.fan_stop("cau_h2", 610.0, 640.0)
    else:
        s.fan_stop("cau_l2", 300.0, 330.0)
        s.fan_stop("cau_l2", 460.0, 490.0)
        s.fan_stop("cau_l1", 700.0, 737.0)
    return s
