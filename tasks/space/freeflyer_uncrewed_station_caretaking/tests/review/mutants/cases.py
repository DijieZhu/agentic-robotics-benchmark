"""cases.py -- the mutant / benign table as event-replay cases (records only, no simulation).

Every case = a base run (recordkit: 'ref' the reference, 'hold' level 4 with the pouches stowed, 'u1' the U1
isolation run, 'nodock' the reference without docking) + one change.  The runner grades both and checks that the
items that differ from the base are exactly the named ones plus the listed rule consequences ('implied', each with
its reason: a gate or a factor of the scoring rule), that the expected values hold (+-1e-9), that the 'keep' items stay,
and that a benign case loses nothing.  'row' ties a case to its line of the mutant / benign table (M1-M25 mutant rows, B1-B24
benign rows); a case with row None is an extra check that is not a line of the table.  More cases go to
cases_*.py files next to this one, each with a CASES list of the same form.
"""
from __future__ import annotations

import recordkit as RK
from recordkit import evidence, free, held, on_face

W = {k: float(v) for k, v in RK.CONFIG["values"]["grader.weights"].items()}
EC = RK.CONFIG["values"]["grader.E"]
DP = {k: float(v) for k, v in RK.CONFIG["values"]["grader.D.parts"].items()}


def e_with_fan(seconds: float) -> float:
    """E with full e_elec and the e_wind of ``seconds`` of stopped fans."""
    w = 1.0 if seconds <= EC["wind_full_s"] else max(0.0, (EC["wind_zero_s"] - seconds) / (EC["wind_zero_s"] - EC["wind_full_s"]))
    return float(EC["w_elec"]) + float(EC["w_wind"]) * w


def L_of(n_on: int, n_lost: int, n_tot: int, g: float = 1.0) -> float:
    pen = float(RK.CONFIG["values"]["grader.L.formula"]["lost_unstowed_penalty"])
    return W["L"] * g * max(0.0, n_on - pen * n_lost) / n_tot


G_NO_U1 = float(RK.CONFIG["values"]["grader.L.formula"]["g_without_U1"])
R_FACTOR = float(RK.CONFIG["values"]["grader.R.factor"])

DESIGN_ROWS = {
    "M1": "reports the adjacent return inlet → D", "M2": "right inlet, but the cited frame does not show it → D", "M3": "the cited frame was taken after the removal → D",
    "M4": "onset off by 11 min → D loses 0.04", "M5": "a second report corrects the first → the first one counts", "M6": "lets go after the removal → U2, V, K",
    "M7": "the sheet seals only 0.030 m² on the screen → K, U2", "M8": "the sheet reaches the screen only 40 s before the end → K, U2",
    "M9": "the sheet sealed on the screen but still in the closed gripper at the end → K (U2 holds)", "M10": "the sheet grasped again only 30 s before the end → U2 (both branches share one 60 s window)",
    "M11": "the sheet lies on a pouch pad → K", "M12": "a pouch on the holding screen → that item is not counted on a pad, E's gate fails",
    "M13": "a pouch knocked off and not recovered → L loses 1.5 items' credit, E's gate fails", "M14": "every pouch stowed but the blocker never moved → L x 0.25, H = 0",
    "M15": "a CAU still stopped at the end → V", "M16": "a stop reaches 60 s and the station restarts the fan → V", "M17": "a third stop request → V",
    "M18": "a stop request 100 s after the restart (REJECTED_INTERVAL) → V", "M19": "50 s of stopped fan in total → e_wind 0.82",
    "M20": "14 % charge at the moment of docking, charged to 16 % on the dock afterwards → H", "M21": "docks with a pouch in the gripper → H", "M22": "ends perched, not docked → H",
    "M23": "a third hard collision → R = 0.85", "M24": "the fan stopped but the item never moved → U1 fails", "M25": "a NaN in the report → that call is void",
    "B1": "which module's stowage unit and which pouch pad are used", "B2": "a pouch placed askew on a pouch pad (sealed area still ≥ 0.0099 m²)", "B3": "the order in which the pouches are stowed",
    "B4": "on a pad-type unit, the sheet taken off by an oblique pull of the strap or a drag along the strip chain", "B5": "the stop requested only once in position (about 35 s)", "B6": "one stop of ≤ 35 s",
    "B7": "a second stop 120 s or more after the restart (affects e_wind only)", "B8": "an early 'on' during a stop", "B9": "one request to an inhibited CAU",
    "B10": "a second 'off' to a fan already stopped", "B11": "a pouch lifts and is sucked back (displacement < 5 cm)", "B12": "a pouch knocked off, grasped again and placed on a pouch pad",
    "B13": "the blocker's placement fails, it is grasped again and placed", "B14": "a sheet carried onto the screen by the cabin flow (K looks at the final state only)", "B15": "releasing while flying",
    "B16": "REFUSED_HOLDING, REFUSED_ZONE, NO_FIDUCIAL, STALLED or NO_BAR received (only time or calls are lost)", "B17": "lights switched on ≤ 6 times",
    "B18": "berth 2 used", "B19": "keep-out grazed for < 5 s", "B20": "1-2 hard collisions", "B21": "more frames within the budget",
    "B22": "the report submitted at any time before the removal", "B23": "an early end (exit once placed: graded after the 60 s settle)", "B24": "the blocker hides the inlet's slats in the evidence frame",
}

CASES = []


def case(cid, row, kind, tier, base, text, build, named=(), implied=None, expect=None, keep=(), criterion=None, note=""):
    CASES.append(dict(id=cid, row=row, kind=kind, tier=tier, base=base, text=text, build=build, named=tuple(named),
                      implied=dict(implied or {}), expect=dict(expect or {}), keep=tuple(keep), criterion=criterion, note=note))


GATE_K = "E's gate is K"
GATE_U2 = "V's gate is U2"
GATE_U1 = "H's gate is U1"
GATE_PADS = "E's gate needs every pouch on a pad"


# ------------------------------------------------------------------------------------------------ mutants
def m01(s):
    s.frames[0]["evidence"]["ri_l1_1"] = evidence(0.0, dist=1.1, inc=25.0, px=380.0)
    s.report["inlet"] = "RI-L1-1"


case("M01", "M1", "mutant", "P", "ref", "report the adjacent inlet RI-L1-1 (the frame shows it too)", m01, ["D"], expect=dict(D=0.0))


def m02(s):
    s.frames[0]["evidence"] = {"ri_l1_1": evidence(0.0, dist=1.1, inc=25.0, px=380.0), "ri_l1_3": evidence(0.0, dist=1.6, inc=35.0, px=250.0)}


case("M02", "M2", "mutant", "P", "ref", "right inlet, but the cited frame does not show it", m02, ["D"], expect=dict(D=0.0))


def m03(s):
    s.add_frame(1, "FIC", 950.0, {s.blk: evidence(0.0)})
    s.report.update(t=960.0, frame_id=1)


case("M03", "M3", "mutant", "P", "ref", "the cited frame was taken after the removal (the blocker no longer covers the inlet)", m03, ["D"], expect=dict(D=0.0))


def m04(s):
    s.report["onset_min"] = s.onset + 11.0


case("M04", "M4", "mutant", "P", "ref", "onset 11 min late", m04, ["D"], expect=dict(D=W["D"] - DP["onset"]))


def m04b(s):
    s.report["onset_min"] = s.onset - 11.0


case("M04b", "M4", "mutant", "P", "ref", "onset 11 min early", m04b, ["D"], expect=dict(D=W["D"] - DP["onset"]))


def m05(s):
    s.frames[0]["evidence"]["ri_l1_1"] = evidence(0.0, dist=1.1, inc=25.0, px=380.0)
    s.report["inlet"] = "RI-L1-1"
    s.calls.append(dict(t=700.0, call="report", code="ALREADY_REPORTED"))


case("M05", "M5", "mutant", "P", "ref", "first report wrong, a second (right) one answered ALREADY_REPORTED", m05, ["D"], expect=dict(D=0.0))


def m06(s):
    s.clear_after("blocker", 900.0)
    s.at("blocker", 905.0, free(0.6))
    s.at("blocker", 960.0, free(3.0))


case("M06", "M6", "mutant", "P", "ref", "let go of the blocker after the removal", m06, ["U2", "V", "K"], implied={"E": GATE_K},
     expect=dict(U2=0.0, V=0.0, K=0.0, E=0.0))


def m07(s):
    s.clear_after("blocker", 900.0)
    s.place_blocker("SB-L1-S", 940.0, 0.030, 942.0)


case("M07", "M7", "mutant", "P", "ref", "the sheet seals only 0.030 m2 on the holding screen", m07, ["K", "U2"],
     implied={"V": GATE_U2, "E": GATE_K}, expect=dict(K=0.0, U2=0.0, V=0.0, E=0.0))


def m08(s):
    s.clear_after("blocker", 900.0)
    s.at("blocker", 940.0, held(2.2))
    s.at("blocker", 942.0, free(2.25))
    s.at("blocker", s.program_end + RK.SETTLE - 40.0, on_face("SB-L1-S", 0.084, 2.29))


case("M08", "M8", "mutant", "P", "ref", "let go near the screen; the flow fixes it only 40 s before the end", m08, ["K", "U2"],
     implied={"V": GATE_U2, "E": GATE_K}, expect=dict(K=0.0, U2=0.0, V=0.0, E=0.0))


def m09(s):
    s.clear_after("blocker", 900.0)
    s.at("blocker", 940.0, on_face("SB-L1-S", 0.084, 2.29, grip=True))
    s.no_dock()


case("M09", "M9", "mutant", "P", "ref", "sealed on the screen but still in the closed gripper at the end", m09, ["K"],
     implied={"H": "holding the sheet the robot cannot stow the arm, so no DOCKED", "E": GATE_K},
     expect=dict(K=0.0, H=0.0, E=0.0), keep=("U2", "V"))


def m10(s):
    s.at("blocker", 1000.0, free(0.5))
    s.at("blocker", s.program_end + RK.SETTLE - 30.0, held(0.6))


case("M10", "M10", "mutant", "P", "hold", "let go at 1000 s, grasped again 30 s before the end", m10, ["U2"], implied={"V": GATE_U2},
     expect=dict(U2=0.0, V=0.0))


def m11(s):
    s.clear_after("blocker", 900.0)
    s.place_blocker("SB-L1-P1", 940.0, 0.052, 942.0)


case("M11", "M11", "mutant", "P", "ref", "the sheet lies on a pouch pad", m11, ["K"],
     implied={"U2": "U2's branch 'fixed on a holding screen' is K's predicate; a pouch pad is not one", "V": GATE_U2, "E": GATE_K},
     expect=dict(K=0.0, U2=0.0, V=0.0, E=0.0))


def m12(s):
    s.clear_after("pouch_1", 0.0)
    s.collect("pouch_1", 700.0, "SB-L1-S", 740.0)


case("M12", "M12", "mutant", "P", "ref", "a pouch stowed on the holding screen", m12, ["L", "E"], expect=dict(L=L_of(2, 0, 3), E=0.0))


def m13(s):
    s.clear_after("pouch_2", 0.0)
    s.at("pouch_2", 300.0, free(0.4))
    s.at("pouch_2", 330.0, free(1.5))


case("M13", "M13", "mutant", "P", "ref", "a pouch knocked off its inlet and never recovered", m13, ["L", "E"], expect=dict(L=L_of(2, 1, 3), E=0.0))


def m14(s):
    s.clear_after("blocker", 0.0)
    s.calls = [c for c in s.calls if not (c["call"] in ("capture", "fly_rel", "gripper") and 840.0 <= c["t"] <= 950.0)]


case("M14", "M14", "mutant", "P", "ref", "every pouch stowed, the blocker never touched (docked)", m14, ["L", "H"],
     implied={"U1": "the blocker never left its inlet", "U2": "the inlet is covered at the end", "V": GATE_U2,
              "K": "the blocker is not on a holding screen", "E": GATE_K},
     expect=dict(L=L_of(3, 0, 3, G_NO_U1), H=0.0, U1=0.0, U2=0.0, V=0.0, K=0.0, E=0.0))


def m15(s):
    s.fan_stop("cau_l2", s.program_end + RK.SETTLE - 20.0, None)


case("M15", "M15", "mutant", "P", "ref", "CAU-L2 still stopped at the end", m15, ["V"], expect=dict(V=0.0),
     note="not reachable in the cell (the station restarts a fan after 60 s and the settle is 60 s); the grader rule is checked on the record")


def _s_stops(s, stops, extra_requests=(), violations=()):
    s.fan_stops, s.requests = [], [r for r in s.requests if r["kind"] != "cau_fan"]
    for unit, a, b, by in stops:
        s.fan_stop(unit, a, b, by)
    for r in extra_requests:
        s.request(*r)
    s.violations = [dict(t=t, unit=u, kind=k) for t, u, k in violations]


def m16(s):
    _s_stops(s, [("cau_h2", 605.0, 665.0, "auto")], violations=[(665.0, "cau_h2", "overrun_auto_restart")])


case("M16", "M16", "mutant", "S", "ref", "a stop runs into the station's automatic restart at 60 s", m16, ["V"],
     implied={"E": "e_wind of 60 s of stopped fan (over the full line)"}, expect=dict(V=0.0, E=e_with_fan(60.0)))


def m17(s):
    _s_stops(s, [("cau_h2", 605.0, 617.0, "robot"), ("cau_h2", 740.0, 752.0, "robot")],
             extra_requests=[(900.0, "cau_fan", "H2", "off", "REJECTED_QUOTA")], violations=[(900.0, "cau_h2", "third_stop_request")])


case("M17", "M17", "mutant", "S", "ref", "a third stop request (two 12 s stops before it)", m17, ["V"], expect=dict(V=0.0))


def m18(s):
    s.request(728.0, "cau_fan", "H2", "off", "REJECTED_INTERVAL")
    s.violations = [dict(t=728.0, unit="cau_h2", kind="stop_within_gap")]


case("M18", "M18", "mutant", "S", "ref", "a stop request 100 s after the restart (REJECTED_INTERVAL)", m18, ["V"], expect=dict(V=0.0))


def m19(s):
    _s_stops(s, [("cau_h2", 605.0, 655.0, "robot")])


case("M19", "M19", "mutant", "S", "ref", "50 s of stopped fan in total", m19, ["E"], expect=dict(E=e_with_fan(50.0)))


def m20(s):
    s.dock_battery = 14.0
    s.charge_rate = 2.0 / (s.program_end + RK.SETTLE - s.dock_t)


case("M20", "M20", "mutant", "P", "ref", "14 % at DOCKED, charged to 16 % on the berth", m20, ["H"], expect=dict(H=0.0))


def m21(s):
    s.clear_after("pouch_1", 0.0)
    s.collect("pouch_1", 1000.0, "SB-L1-P2", 1040.0)
    s.at("pouch_1", 1045.0, held(2.3))


case("M21", "M21", "mutant", "P", "ref", "the closed gripper caught a pouch that followed the fingers; docked with it", m21, ["H"],
     implied={"L": "the held pouch is not on a pad", "E": GATE_PADS}, expect=dict(H=0.0, L=L_of(2, 0, 3), E=0.0))


def m21b(s):
    s.clear_after("pouch_1", 0.0)
    s.at("pouch_1", 700.0, on_face("ri_l1_3", 0.006, 0.0002, grip=True))
    s.at("pouch_1", 702.0, held(0.35))
    s.at("pouch_1", 740.0, free(2.0))
    s.at("pouch_1", 760.0, free(12.0))
    s.contacts_after_dock["pouch_1"] = True


case("M21b", "M21", "mutant", "P", "ref", "a pouch caught between the robot and the dock", m21b, ["H"],
     implied={"L": "the pouch is not on a pad and was out of control", "E": GATE_PADS}, expect=dict(H=0.0, L=L_of(2, 1, 3), E=0.0))


def m22(s):
    s.no_dock()


case("M22", "M22", "mutant", "P", "ref", "ends away from the dock (hovering; perch is not in this API)", m22, ["H"], expect=dict(H=0.0))


def _contacts(n, cls="structure", speed=0.20, force=1.0):
    return [dict(t=500.0 + 10.0 * k, speed_m_s=speed, force_N=force, other_class=cls, robot_geom="bumper_px", other=f"wall_box_{k}") for k in range(n)]


def m23(s):
    s.hard = _contacts(3)


case("M23", "M23", "mutant", "P", "ref", "a third hard collision (0.20 m/s with structure)", m23, ["R"], expect=dict(R=R_FACTOR))


def m23b(s):
    s.hard = _contacts(3, cls="lamp", speed=0.05, force=6.0)


case("M23b", "M23", "mutant", "P", "ref", "three slow contacts with a lamp over the force line", m23b, ["R"], expect=dict(R=R_FACTOR))


def m23c(s):
    s.keepout_s = float(RK.CONFIG["values"]["grader.R.keepout_s"])


case("M23c", "M23", "mutant", "P", "ref", "keep-out time reaches its limit", m23c, ["R"], expect=dict(R=R_FACTOR))


def m24(s):
    T = s.site["truth"]
    s.clear_after("blocker", 0.0)
    s.at("blocker", 600.0, on_face(s.blk, float(T["cover_m2"]), 0.0003, grip=True))
    s.at("blocker", 640.0, on_face(s.blk, float(T["cover_m2"]), 0.0003))


case("M24", "M24", "mutant", "S", "u1", "the fan stopped but the blocker never moved off the inlet", m24, ["U1"],
     implied={"H": GATE_U1, "L": "g = g_without_U1 without U1"}, expect=dict(U1=0.0, H=0.0, L=L_of(2, 0, 2, G_NO_U1)))


def m25(s):
    s.report["onset_min"] = float("nan")


case("M25", "M25", "mutant", "P", "ref", "a NaN in the report (the call is void)", m25, ["D"], expect=dict(D=0.0))


# ------------------------------------------------------------------------------------------------ extra mutants
def x01(s):
    s.frames[0]["camera"] = "FDC"


case("X01", None, "mutant", "P", "ref", "the cited frame is a depth frame", x01, ["D"], expect=dict(D=0.0))


def x02(s):
    s.frames[0]["t"] = 665.0


case("X02", None, "mutant", "P", "ref", "the cited frame was taken after the report", x02, ["D"], expect=dict(D=0.0))


def x03(s):
    s.clear_after("pouch_2", 0.0)
    s.loss_extra["pouch_2"] = 300.0


case("X03", None, "mutant", "P", "ref", "a pouch left on its inlet, the cell's record lists a loss of control for it", x03, ["L", "E"],
     expect=dict(L=L_of(2, 1, 3), E=0.0), note="the record's event counts even when the 0.5 s samples do not show it")


def x04(s):
    s.state_cut_s = s.program_end + RK.SETTLE - 30.0


case("X04", None, "mutant", "P", "ref", "the record's samples stop 30 s before the end (incomplete window)", x04, ["K", "U2", "L"],
     implied={"V": GATE_U2, "E": GATE_K}, expect=dict(K=0.0, U2=0.0, L=0.0, V=0.0, E=0.0),
     note="missing data never earns a windowed item")


# ------------------------------------------------------------------------------------------------ benign
def b01(s):
    s.clear_after("blocker", 900.0)
    s.place_blocker("SB-L2-S", 940.0, 0.084, 942.0)
    s.clear_after("pouch_1", 0.0)
    s.collect("pouch_1", 700.0, "SB-L1-P1", 740.0)
    s.clear_after("pouch_3", 0.0)
    s.collect("pouch_3", 150.0, "SB-H2-P1", 190.0)


case("B01", "B1", "benign", "P", "ref", "another module's stowage unit and other pads", b01)


def b02(s):
    s.clear_after("pouch_1", 0.0)
    s.collect("pouch_1", 700.0, "SB-L1-P2", 740.0, frac=0.85)


case("B02", "B2", "benign", "P", "ref", "a pouch placed askew, 85 % of its own footprint on the pad (default criterion)", b02)


def b02alt(s):
    s.clear_after("pouch_1", 0.0)
    s.collect("pouch_1", 700.0, "SB-L1-P2", 740.0, frac=0.5)


case("B02alt", "B2", "benign", "P", "ref", "a pouch askew sealing 0.012 m2 (>= 0.0099) under the 'abs_seal_area' criterion", b02alt,
     criterion="abs_seal_area", note="the literal row; under the default criterion the same state loses the pouch")


def b03(s):
    s.clear_after("pouch_2", 0.0)
    s.clear_after("pouch_3", 0.0)
    s.collect("pouch_2", 150.0, "SB-H2-P2", 190.0)
    s.collect("pouch_3", 330.0, "SB-L2-P1", 370.0)


case("B03", "B3", "benign", "P", "ref", "pouches stowed in another order", b03)


def b04(s):
    T = s.site["truth"]
    a = float(T["cover_m2"])
    s.at("blocker", 880.0, on_face(s.blk, a * 0.66, 0.002, grip=True))
    s.at("blocker", 890.0, on_face(s.blk, a * 0.25, 0.01, grip=True))


case("B04", "B4", "benign", "P", "ref", "peeled off in steps (oblique strap pull or a drag along the strip chain)", b04)


case("B05", "B5", "benign", "S", "ref", "one stop of 35.0 s (requested once in position)", lambda s: _s_stops(s, [("cau_h2", 605.0, 640.0, "robot")]))
case("B05b", "B5", "benign", "S", "ref", "one stop of 34.8 s", lambda s: _s_stops(s, [("cau_h2", 605.0, 639.8, "robot")]))
case("B06", "B6", "benign", "S", "ref", "one stop of 30 s", lambda s: _s_stops(s, [("cau_h2", 605.0, 635.0, "robot")]))
case("B07", "B7", "benign", "S", "ref", "a second stop requested 120 s after the restart (20 s + 15 s)",
     lambda s: _s_stops(s, [("cau_h2", 605.0, 625.0, "robot"), ("cau_h2", 745.0, 760.0, "robot")]))
case("B08", "B8", "benign", "S", "ref", "'on' sent early (a 15 s stop)", lambda s: _s_stops(s, [("cau_h2", 605.0, 620.0, "robot")]))
case("B09", "B9", "benign", "P", "ref", "one stop request to the inhibited CAU-L1 (REJECTED_INHIBITED)",
     lambda s: s.request(800.0, "cau_fan", "L1", "off", "REJECTED_INHIBITED"))
case("B10", "B10", "benign", "S", "ref", "'off' again to the stopped fan (REJECTED_STATE)", lambda s: s.request(610.0, "cau_fan", "H2", "off", "REJECTED_STATE"))


def b11(s):
    s.at("pouch_2", 250.0, free(0.04))
    s.at("pouch_2", 256.0, on_face("ri_h2_1", 0.004, 0.003))


case("B11", "B11", "benign", "P", "ref", "a pouch tilts off its inlet by 4 cm and is sucked back", b11)


def b12(s):
    fp = 0.18 * 0.11
    s.clear_after("pouch_2", 0.0)
    s.at("pouch_2", 300.0, free(0.6))
    s.at("pouch_2", 325.0, held(0.9))
    s.at("pouch_2", 370.0, on_face("SB-H2-P2", 0.98 * fp, 2.8, grip=True))
    s.at("pouch_2", 371.0, on_face("SB-H2-P2", 0.98 * fp, 2.8))


case("B12", "B12", "benign", "P", "ref", "a pouch knocked off, grasped again and stowed on a pad", b12)


def b13(s):
    s.clear_after("blocker", 900.0)
    s.at("blocker", 940.0, on_face("SB-L1-S", 0.020, 2.2, grip=True))
    s.at("blocker", 942.0, on_face("SB-L1-S", 0.020, 2.2))
    s.at("blocker", 975.0, held(2.25))
    s.place_blocker("SB-L1-S", 1000.0, 0.084, 1002.0)


case("B13", "B13", "benign", "P", "ref", "the first placement fails (caught on the screen edge), grasped again and fixed", b13)


def b14(s):
    s.clear_after("blocker", 900.0)
    s.at("blocker", 935.0, free(2.2))
    s.at("blocker", 955.0, on_face("SB-L1-S", 0.060, 2.29))


case("B14", "B14", "benign", "P", "ref", "let go 0.15 m from the screen; the cabin flow carries it onto the screen", b14)


def b15(s):
    s.clear_after("blocker", 900.0)
    s.at("blocker", 940.0, held(2.2))
    s.at("blocker", 941.0, free(2.25))
    s.at("blocker", 942.5, on_face("SB-L1-S", 0.084, 2.29))


case("B15", "B15", "benign", "P", "ref", "released 0.10 m from the face while flying", b15)


def b16(s):
    for k, code in enumerate(("REFUSED_HOLDING", "REFUSED_ZONE", "NO_FIDUCIAL", "STALLED", "NO_BAR")):
        s.calls.append(dict(t=860.0 + 2.0 * k, call="arm" if code == "REFUSED_HOLDING" else ("capture" if code in ("STALLED", "NO_BAR") else "fly_rel"), code=code))
    s.budgets["capture_calls"] += 2.0


case("B16", "B16", "benign", "P", "ref", "REFUSED_HOLDING, REFUSED_ZONE, NO_FIDUCIAL, STALLED and NO_BAR answers", b16)


def b17(s):
    for k, m in enumerate(("L2", "H2", "L1", "L2", "H2", "L1")):
        s.request(20.0 + k, "lights", m, "general" if k < 3 else "night", "ACCEPTED")
    s.budgets["light_requests"] = 6.0


case("B17", "B17", "benign", "P", "ref", "six light requests", b17)
case("B18", "B18", "benign", "P", "ref", "docked on berth 2 (the record keeps no berth: nothing may depend on it)",
     lambda s: s.calls.append(dict(t=1329.0, call="fly_to", code="SUCCESS")))
case("B19", "B19", "benign", "P", "ref", "4.9 s in a keep-out zone", lambda s: setattr(s, "keepout_s", 4.9))
case("B20", "B20", "benign", "P", "ref", "two hard collisions (0.20 m/s)", lambda s: setattr(s, "hard", _contacts(2)))


def b21(s):
    s.budgets.update(colour_frames=380.0, depth_frames=280.0)


case("B21", "B21", "benign", "P", "ref", "more frames within the budgets (380 colour, 280 depth)", b21)


def b22(s):
    s.add_frame(1, "FIC", 840.0, {s.blk: evidence(0.625, dist=0.9)})
    s.report.update(t=845.0, frame_id=1)


case("B22", "B22", "benign", "P", "ref", "the report filed just before the capture", b22)


def b22b(s):
    s.add_frame(1, "FWC", 595.0, {s.blk: evidence(0.625, dist=1.8, px=200.0)})
    s.report.update(t=600.0, frame_id=1)


case("B22b", "B22", "benign", "P", "ref", "the report filed early (a wide-camera frame from 1.8 m)", b22b)


def b23(s):
    s.program_end = 943.0
    s.energy_wh = 30.0


case("B23", "B23", "benign", "P", "nodock", "the program exits 1 s after the sheet is fixed: K and U2 are judged in the settle", b23)
case("B23b", "B23", "benign", "P", "ref", "the program exits 0.5 s after DOCKED", lambda s: setattr(s, "program_end", 1330.5))


def b24(s):
    s.frames[0]["evidence"][s.blk]["occluded"] = 0.15


case("B24", "B24", "benign", "P", "ref", "the blocker hides the louvres (not an occluder); 15 % of the rays blocked by other things", b24)


# ------------------------------------------------------------------------------------------------ extra benign
case("B25", None, "benign", "P", "ref", "the report filed after the removal, citing a frame from before it", lambda s: s.report.update(t=1000.0))


def b26(s):
    s.calls.append(dict(t=640.0, call="report", code="INVALID_ARGS"))


case("B26", None, "benign", "P", "ref", "a void report (INVALID_ARGS) before the recorded one", b26)
case("B27", None, "benign", "P", "ref", "onset exactly at the tolerance (10 min)", lambda s: s.report.update(onset_min=s.onset + 10.0))
case("B28", None, "benign", "P", "ref", "docked with exactly the minimum charge (15 %)", lambda s: setattr(s, "dock_battery", 15.0))


def b29(s):
    s.clear_after("pouch_2", 0.0)
    s.collect("pouch_2", 330.0, "SB-L1-P2", 370.0)


case("B29", None, "benign", "P", "ref", "two pouches on one pad, each with >= 80 % of its own footprint", b29)
