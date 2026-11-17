"""The final-assembly-and-trim cell behind the camera API (`RovInfoServer`): grader side, never shipped to the agent.

Sources (the form copied, then rewritten for this cell; none of them is imported):
  docs/plan/driving/tasks/racecar_full_assembly_tenth_scale_ur10e/tests/arm/assembly_car/server.py
      sha256 adc872ebe29d6f3f164cf876c976b06de29e1709ec4104c420a97ef1b23d55e1: the episode setup, the routed control
      tick, the privileged calls, the capture-table texts of describe(), finalize()
  tasks/quadruped/quadruped_full_assembly_12dof_ur10e/tests/arm/assembly_quad/server.py (arb-main-20261113)
      sha256 947f14234fdff276c4b16f44f9bc8ca8230961d75cb031b98fb1d3c6062c0b6f: the robot and camera blocks of describe()
  tests/arm/assembly_info/cell_info.py (vendored): get_state's noisy wrench, get_camera's reply format, the dispatch

Physics: `cell.RovCell` (the base vehicle on the two-position turnover positioner over three load cells, the five
stations with the kit's nineteen loose units, the arm with the wide gripper, the coupling state machine, the watch and
the budgets, run inside every control tick). This module adds the episode around it and the restricted API.

Episode (`reset(entry)`; `entry` = one episode of the fixed hidden set, tests/hidden/episodes.json, format
rov-asm-set/1: {seed, stratum E1..E4, dome_end_deg 0 | 180, work_order {unit, delivery_site {water_temperature_C,
salinity_psu}}, deviation {dm_kg, dx_mm, dy_mm}, layout_digest}):
  * the physical layout `layout.episode_layout(seed, fx, cfg["layout"], stratum="dome0" | "dome180", nominal)` and its
    `layout_digest` (returned: the grader compares it with the set's);
  * THIS vehicle: the base vehicle's mass and centre of mass = the trim table's tabled base vehicle plus the episode's
    per-unit deviation (dm on the mass, dx / dy on the centre of mass in the vehicle frame B; displacement and centre
    of buoyancy as tabled: the trim sheet's acceptance convention, DESIGN v1.2 section 2.4; `vehicle_kit`, the same
    definition as the trim truth `trim_eval.TrimTable.base_row`). Applied to a copy of the kit's `vehicle` body, so the
    load cells weigh this vehicle; nothing else of the kit changes;
  * the information tier's draws (`assembly_info.layout_info.draw_episode`: camera intrinsics and mounts, lights, the
    flange sensor's bias, the tool plate's jitter), the wrist camera's mount resolved from the built robot
    (`render_twin.info_layout`);
  * the load cells' noise (`positioner.LoadCellNoise`: the layout's per-cell zero offsets, sigma from the grader
    configuration): one sample per control tick, the same sample for every get_state() of that tick.

The API (cell_api.md): describe, get_state, set_control, command (q | tcp_pos + tcp_quat, width, force), step,
fixture_index, get_camera, finish. describe() carries this run's work order and no unit pose, no fixture pose beyond
the documented nominal ones, nothing about propellers, captures, fastening or the trim; get_state() carries the robot
and `fixture {index, moving, load_N[3]}`; step() reports only `safety:<what>` and `episode_over`. Anything else answers
{"error": "forbidden"} and is counted (illegal_call_count). With the token (privileged mode) `get_state_privileged`,
`describe_privileged` and `reference_assemble` answer; `reference_assemble` runs the privileged reference program's
whole episode (`reference_ee.reference_assemble(cell, server=..., log=...)`: part III's entry, which runs parts I and
II before the enclosure; imported on first use) inside the cell, through the same routed ticks and the same watch.

Score of an episode (`finalize` -> `score_episode`, DESIGN v1.2 section 3.1):
  stage = (sum of the weights of the counted structural mates + trim_weight x trim) / (their total + trim_weight)
  counted = the coupling's record: fastened in the counting orientation with the place's propeller, neither the unit
  nor its carrier scrapped; trim = `trim_eval.grade` on the trim table, this vehicle's deviation, the work order's
  water and the ballast sites that hold a fastened block at the end (0 / 0.4 / 0.7 / 1.0);
  hard_ok = no robot_contact (H1) and no part_force (H2) violation. The dome rule (H3) and the scrap rules only scrap.
"""

import copy
import json
import math
import time
import traceback

import numpy as np

from ..assembly_info import layout_info as LI
from ..assembly_info.cell_info import DEFAULT_INFO_CONFIG
from . import fixtures as FX
from . import layout as LY
from . import positioner as PZ
from . import render_twin as RT
from . import trim_eval as TE
from .cell import RovCell, check_kit, load_kit

CELL_VERSION = "rov-assembly-cell/1.0.0"
PUBLIC_METHODS = ("describe", "get_state", "set_control", "command", "step", "fixture_index", "get_camera", "finish")
PRIVILEGED_METHODS = ("get_state_privileged", "describe_privileged", "reference_assemble")
# what this cell is built for (DESIGN v1.2 sections 1.1, 1.2, 3.1): 7 scored places of weight 11.0, 13 ballast sites,
# 19 loose units
EXPECTED_KIT = {"scored_slots": 7, "weight": 11.0, "sites": 13, "loose": 19}
HIDDEN_STRATA = ("E1", "E2", "E3", "E4")
DOME_ENDS = (0, 180)
INDEXES = ("keel_up", "deck_up")
# the given kit.json's unit ids per kit body type (the agent's names)
GIVEN_UNIT_IDS = {"electronics_enclosure": "electronics_enclosure", "thruster_a": "ducted thruster", "thruster_b": "ducted thruster",
                  "thruster_guard": "thruster_guard", "ballast_block": "ballast_block"}

# ------------------------------------------------------------------ the agent-visible strings (as the cell manual words them)
TASK_TEXT = ("finish the vehicle on the turnover positioner for this run's work order: fit ballast blocks on the keel so that the "
             "delivered vehicle meets the trim sheet's acceptance in the work order's water, then the electronics enclosure, the "
             "four vertical thrusters and the two thruster guards")
PLACE_WORDS = {
    "ee": ("the electronics enclosure", "its two cradle clips on the two tube cradles (enclosure_mounting.png)", "downwards (the vehicle's -z)"),
    "vt": ("any vertical thruster", "its mounting block against the outer face of the side plate (propulsion_layout.png)",
           "sideways onto the outer face of its side plate (along the vehicle's y)"),
    "guard": ("a thruster guard", "lying on the two lower guard brackets of its side, its inner edge against the side plate "
                                  "(guard_and_brackets.png)", "downwards (the vehicle's -z)"),
    "ballast": ("any ballast block", "flat against the underside of the bottom plate over a hole that keel_hole_map.png marks as free",
                "onto the underside of the bottom plate along the plate's normal"),
}
PLACE_SIDE = {"vt_FR": "front starboard", "vt_FL": "front port", "vt_RR": "rear starboard", "vt_RL": "rear port",
              "guard_P": "port", "guard_S": "starboard"}
FASTENING_SUMMARY = ("a unit inside the capture window of a place it can take, and at rest there relative to the vehicle (every point "
                     "of the unit slower than {still_mm_s} mm/s relative to the vehicle and the unit turning slower than {still_deg_s} "
                     "deg/s relative to it, on {still_ticks} consecutive ticks), is captured: held there and "
                     "pulled home at up to {pull_mm_s} mm/s and {pull_deg_s} deg/s, and fastened once it has rested at home within the "
                     "lock tolerance for {hold_ticks} consecutive ticks; a captured unit moved beyond the release window before it is "
                     "fastened is released; nothing is unfastened; no event reports any of this")
FASTENING_RULES = (
    "the electronics enclosure is captured and fastened either way round; only the way round that enclosure_mounting.png shows counts",
    "a vertical thruster is captured and fastened in any of {vt_n} orientations {vt_pitch} deg apart about the normal of the side "
    "plate; only the one that propulsion_layout.png shows counts, and only with a thruster whose propeller turns the way that "
    "drawing gives for that place; a thruster of the other kind is fastened there all the same, for good, and counts nothing",
    "a thruster guard is captured one way only: turned over, turned end for end or on the other side it does not come to lie in its window",
    "a ballast block counts either way round on any free ballast site; the sites are the holes that keel_hole_map.png marks as free; "
    "fastened ballast blocks count through the trim of the delivered vehicle, not with a weight of their own",
    "a vertical thruster and a thruster guard do not stay on their places without being held: hold such a unit still inside its "
    "window for at least {hold_release_s} s before letting it go; a captured unit does not fall when it is let go",
    "no place has a physical key, and no place needs another one fastened first",
)
SAFETY_RULES = (
    "a robot link other than the gripper's four pads pressing on the vehicle, the positioner, a fixture, the bench or a unit above "
    "robot_force_N (the normal forces of its contacts summed per (link, body it presses on)): the episode scores nothing",
    "a unit pressed on the environment or on another unit with a contact force above part_force_N for part_force_window_s (the "
    "magnitude of the vector sum of the contact forces per (unit, body it presses on), normal and tangential components together; "
    "the gripper's pads on a unit do not count; the force with which the cell holds, pulls or keeps a captured unit is not a "
    "contact force): the episode scores nothing",
    "anything pressing on the electronics enclosure's dome above dome_force_N for dome_window_s scraps the enclosure; the episode goes on",
    "a loose unit whose centre of mass is faster than part_speed_m_s, that falls off the bench, or that is on the vehicle and not "
    "fastened when a turn of the positioner starts is scrapped: it is released, can no longer be fastened and no longer counts; the "
    "episode goes on",
    "a unit let go without being captured that hangs or lies on the vehicle is not scrapped at once: it counts nothing, can be picked "
    "up again and is scrapped at the start of the next turn",
)
MOUNT_TEXT = {
    "top": "overhead, roughly {top_h} m above the bench near the middle of the working area, looking straight down; image width "
           "roughly along the cell's x",
    "wrist": "on the gripper's base, about {wrist_cm} cm off the tool axis on the tool's +x side, tilted about {wrist_tilt} deg "
             "towards the fingertips; the fingers and a held unit are in the lower part of the image",
}
ITEMS_TEXT = (
    "the base vehicle on the turnover positioner, the vehicle frame's origin at ({veh_x}, {veh_y}, {veh_z}) m, its x along the cell's "
    "x (the stern towards the robot arm); the run starts at keel_up; the positioner stands on three load cells",
    "the ballast box at its exact place, {blocks_n} ballast blocks in its pockets",
    "the thruster tray with the four vertical thrusters standing nose cone down in its nests, nominally at ({tray_x}, {tray_y}) m "
    "(+-{tray_xy} mm, turned by up to +-{tray_yaw} deg); which thruster stands in which nest, which way its mounting block points "
    "and at which angle its propeller stands are drawn anew in every run",
    "the enclosure rack with the electronics enclosure on its two V-blocks, nominally at ({rack_x}, {rack_y}) m (+-{rack_xy} mm, "
    "turned by up to +-{rack_yaw} deg); the enclosure turned about its tube axis by up to +-{roll} deg, its dome towards either end",
    "one guard rack on each side, nominally at ({gp_x}, {gp_y}) m (port) and ({gs_x}, {gs_y}) m (starboard) (+-{g_xy} mm, turned "
    "by up to +-{g_yaw} deg), each with a thruster guard lying flat on its shelf as it lies on its side of the vehicle",
)

DEFAULT_ROV_CONFIG = {
    "sim_cap_s": 1200.0,
    "wall_cap_s": 2400.0,
    "privileged_wall_cap_s": 2400.0,
    "silence_s": 120.0,
    "max_ticks_per_step": 500,
    "camera_budget_frames": 600,
    "settle_s": 1.0,
    "safety": {"robot_force_N": 30.0, "part_speed_m_s": 3.0, "dome_force_N": 20.0, "dome_ticks": 5},
    "damage": {"part_force_N": 150.0, "ticks": 20},
    "scrap": {"below_bench_m": 0.05},
    "fixture": {"max_turns": 4, "turn_s": 6.0, "load_cells": {"sigma_N": 0.4, "bias_N": 0.02}},
    "handling": {"tcp_speed_m_s": 0.25, "joint_acc_rad_s2": 8.0},
    "score": {"trim_weight": 4.5, "total_weight": 15.5,
              "trim": {"share_net_buoyancy": 0.4, "share_pitch": 0.3, "share_roll": 0.3, "net_buoyancy_window_N": [0.2, 3.0],
                       "angle_window_deg": 3.0}},
    "photo": json.loads(json.dumps(DEFAULT_INFO_CONFIG["photo"])),
    "info_layout": json.loads(json.dumps(RT.INFO_LAYOUT)),
    "agent_cameras": ["top", "wrist"],
    "agent_renderer": "filament",
    "robot": {"model": "ur10e", "pos": [0.0, 0.0, 0.0], "yaw_deg": 0.0},
    "grip_force_N": 100.0,
    "layout": json.loads(json.dumps(LY.DEFAULT_LAYOUT)),
    "physics": {},
    "oracle_nominal_layout": False,
}


def _deep_update(a, b):
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a


def merge_config(config=None):
    return _deep_update(json.loads(json.dumps(DEFAULT_ROV_CONFIG)), config or {})


def _f(v, decimals):
    """a number as the documents print it: exactly `decimals` digits after the point"""
    s = format(float(v), ".%df" % int(decimals))
    return "0" if s in ("-0", "-0.0", "-0.00") else s


def _r(v, nd=4):
    v = round(float(v), nd)
    return 0.0 if v == 0 else v


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _finite_list(v, n, what):
    """a list of n finite floats, else ValueError (the caller answers bad_params)"""
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise ValueError("%s must be a list of %d numbers" % (what, n))
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError("%s must be a list of %d numbers" % (what, n))
        x = float(x)
        if not math.isfinite(x):
            raise ValueError("%s must be finite" % what)
        out.append(x)
    return out


def _finite_number(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ValueError("%s must be a finite number" % what)
    return float(v)


# ------------------------------------------------------------------ the kit and the episode's vehicle
def check_rov_kit(kit):
    """the kit is the one this cell is built for (EXPECTED_KIT), else ValueError"""
    check_kit(kit)
    slots = kit["slots"]
    scored = [s for s in slots if s.get("scored", True)]
    sites = [s for s in slots if not s.get("scored", True)]
    if len(scored) != EXPECTED_KIT["scored_slots"]:
        raise ValueError("kit: %d scored places, the cell expects %d" % (len(scored), EXPECTED_KIT["scored_slots"]))
    w = sum(float(s["weight"]) for s in scored)
    if abs(w - EXPECTED_KIT["weight"]) > 1e-9:
        raise ValueError("kit: total weight of the scored places %.3f, the cell expects %.1f" % (w, EXPECTED_KIT["weight"]))
    if len(sites) != EXPECTED_KIT["sites"] or any(float(s["weight"]) != 0.0 or not s.get("site") for s in sites):
        raise ValueError("kit: %d unscored ballast sites, the cell expects %d of weight 0 with a site record" % (len(sites), EXPECTED_KIT["sites"]))
    if len(kit["loose"]) != EXPECTED_KIT["loose"]:
        raise ValueError("kit: %d loose units, the cell expects %d" % (len(kit["loose"]), EXPECTED_KIT["loose"]))
    return True


def parse_deviation(dev):
    """(dm_kg, dx_mm, dy_mm) of an episode's per-unit deviation (dict or triple), finite, else ValueError"""
    if isinstance(dev, dict):
        vals = [dev.get("dm_kg"), dev.get("dx_mm"), dev.get("dy_mm")]
    else:
        vals = list(dev or [])
    if len(vals) != 3:
        raise ValueError("deviation: three numbers (dm_kg, dx_mm, dy_mm)")
    return tuple(_finite_number(v, "deviation") for v in vals)


def vehicle_kit(kit, deviation, table):
    """a shallow copy of `kit` whose base vehicle body is THIS vehicle: the trim table's base vehicle (its tabled mass and
    centre of mass, `trim_eval.TrimTable.base_row`) plus the per-unit deviation (mass + dm_kg, centre of mass +
    (dx_mm, dy_mm, 0) in the vehicle frame B, the body frame of the kit's `vehicle`): the one definition the trim truth
    (`trim_eval.grade`), the reference harnesses (`run_reference_r1.episode_kit`) and this server share (ruling of
    group Q, request of P16; tests/README.md). The inertia about the centre of mass, the displacement and everything
    else as the kit has it"""
    dm, dx, dy = parse_deviation(deviation)
    base = kit["base_body"]
    vb = dict(kit["bodies"][base])
    if vb.get("frame") not in (None, "B"):
        raise ValueError("kit: the base body's frame is %r, the deviation is defined in the vehicle frame B" % vb.get("frame"))
    M, _V, cg_mm, _cb = table.base_row((dm, dx, dy))
    M0, _V0, cg0_mm, _cb0 = table.base_row((0.0, 0.0, 0.0))
    if not M > 0:
        raise ValueError("deviation: the vehicle's mass would not be positive")
    m_kit, c_kit = float(vb["mass_kg"]), [float(v) for v in vb["com_m"]]
    vb["mass_kg"] = float(M)
    vb["com_m"] = [float(v) / 1000.0 for v in cg_mm]
    vb["deviation_applied"] = {"dm_kg": dm, "dx_mm": dx, "dy_mm": dy, "mass_kg_tabled": float(M0),
                               "com_m_tabled": [float(v) / 1000.0 for v in cg0_mm], "mass_kg_kit": m_kit, "com_m_kit": c_kit,
                               "definition": "the trim table's base vehicle (mass, centre of mass) + the deviation"}
    out = dict(kit)
    out["bodies"] = dict(kit["bodies"])
    out["bodies"][base] = vb
    return out


# ------------------------------------------------------------------ the score of an episode (pure: the unit tests drive it)
def ballast_holes(records):
    """the hole names (the trim table's) of the unscored ballast sites that hold a fastened block, from coupling.record()"""
    out = []
    for r in records:
        if r.get("scored", True) or not r.get("achieved"):
            continue
        site = r.get("site") or {}
        name = site.get("name_03")
        if not name:
            raise ValueError("ballast site %s has no hole name" % r.get("id"))
        out.append(str(name))
    return out


def stage_score(counted_w, total_w, trim, trim_w):
    """(counted structural weight + trim_w x trim) / (total structural weight + trim_w); DESIGN v1.2 section 3.1"""
    den = float(total_w) + float(trim_w)
    if not den > 0:
        raise ValueError("no weight to score")
    t = float(trim)
    if not 0.0 <= t <= 1.0:
        raise ValueError("trim %r outside [0, 1]" % trim)
    return (float(counted_w) + float(trim_w) * t) / den


def score_episode(records, table, deviation, work_order, trim_weight, hard_ok, rho=None):
    """the episode's score from the coupling's record (scored places: `achieved` = counted; ballast sites: `achieved` =
    holds a fastened block), the trim table, this vehicle's deviation and the work order's water:
    {stage_score, stage_score_gated, counted_w, total_w, trim, trim_eval, holes, n_counted, ...} (finite numbers)"""
    scored = [r for r in records if r.get("scored", True)]
    counted_w = float(sum(float(r["weight"]) for r in scored if r.get("achieved")))
    total_w = float(sum(float(r["weight"]) for r in scored))
    holes = ballast_holes(records)
    site = (work_order or {}).get("delivery_site") or {}
    if rho is None:
        tr = TE.grade(table, parse_deviation(deviation), holes, T_C=float(site["water_temperature_C"]), SP=float(site["salinity_psu"]))
    else:
        tr = TE.grade(table, parse_deviation(deviation), holes, rho=float(rho))
    s = stage_score(counted_w, total_w, tr["trim"], trim_weight)
    return {"stage_score": s, "stage_score_gated": s if hard_ok else 0.0, "hard_ok": bool(hard_ok), "counted_w": counted_w,
            "total_w": total_w, "trim_weight": float(trim_weight), "trim": float(tr["trim"]), "trim_eval": _jsonable(tr),
            "holes": list(tr["holes"]), "n_counted": int(sum(1 for r in scored if r.get("achieved"))), "n_scored": len(scored)}


def check_trim_config(cfg, table):
    """the grader configuration's trim block equals the trim table's acceptance (the table is the truth), else ValueError"""
    tc = cfg["score"]["trim"]
    w = table.window
    want = {"share_net_buoyancy": table.shares[0], "share_pitch": table.shares[1], "share_roll": table.shares[2],
            "net_buoyancy_window_N": list(w["net_buoyancy_N"]), "angle_window_deg": w["pitch_abs_deg"]}
    bad = []
    for k, v in want.items():
        got = tc.get(k)
        if isinstance(v, list):
            if not isinstance(got, list) or len(got) != len(v) or any(abs(float(a) - float(b)) > 1e-12 for a, b in zip(got, v)):
                bad.append(k)
        elif got is None or abs(float(got) - float(v)) > 1e-12:
            bad.append(k)
    if abs(w["pitch_abs_deg"] - w["roll_abs_deg"]) > 1e-12:
        bad.append("angle_window_deg (pitch and roll differ in the table)")
    if abs(float(cfg["score"]["total_weight"]) - (EXPECTED_KIT["weight"] + float(cfg["score"]["trim_weight"]))) > 1e-9:
        bad.append("total_weight")
    if bad:
        raise ValueError("grader_config.json score block differs from the trim table / the kit: %s" % ", ".join(bad))
    return True


# ------------------------------------------------------------------ the server
class RovInfoServer(object):
    """`kit_dir` = the cell's hidden kit (tests/gt: kit.json + meshes/), `robot_dir` = the given robot models
    (universal_robots_ur10e/, wide_gripper/), `config` = the grader configuration (overrides of DEFAULT_ROV_CONFIG).
    `kit` / `fx` / `table`: the loaded kit, fixtures and trim table when the caller has them (one load for several
    episodes)."""

    def __init__(self, kit_dir, robot_dir, config=None, mode="hidden", privileged=False, render=True, log=None, kit=None, fx=None,
                 table=None):
        import mujoco
        if mode != "hidden":
            raise ValueError("this cell has no practice mode")
        self.mujoco = mujoco
        self.cfg = merge_config(config)
        self.kit = kit if kit is not None else load_kit(kit_dir)
        check_rov_kit(self.kit)
        self.kit_dir = self.kit["dir"]
        self.robot_dir = robot_dir
        self.robot_cfg = {}
        lm = bool(self.cfg["layout"].get("lmag", False))
        self.fx = fx if fx is not None else FX.Fixtures(FX.load_cell_layout(), FX.load_facts(), lmag=lm)
        if bool(self.fx.lmag) != lm:
            raise ValueError("the fixtures' L-mag (%s) is not the configuration's (%s)" % (self.fx.lmag, lm))
        self.table = table if table is not None else TE.TrimTable()
        check_trim_config(self.cfg, self.table)
        lc = self.cfg["fixture"]["load_cells"]
        if abs(float(lc["bias_N"]) - float(self.cfg["layout"]["cell_bias_max_N"])) > 1e-12:
            raise ValueError("grader_config.json: fixture.load_cells.bias_N differs from layout.cell_bias_max_N")
        self.mode, self.allow_reset = mode, False
        self.privileged, self.render_enabled = bool(privileged), bool(render)
        self.log = log or (lambda s: None)
        self.camera_names = tuple(self.cfg.get("agent_cameras") or ("top", "wrist"))
        self.model = self.data = self.seed = self.cell = None
        self.layout = self.layout_digest = None
        self.entry = None
        self.done = False
        self.done_reason = None
        self.session = None
        self.renderer = None
        self.calls = {}
        self.illegal_calls = 0
        self.reference_report = None
        self.wall_start = None
        self.wall_used = 0.0

    # ------------------------------------------------------------------ episode setup
    def cell_config(self):
        """the RovCell config of this grader configuration: the robot, the physics, the watch, the positioner, the budget"""
        c = self.cfg
        fxc = c["fixture"]
        return {"robot": dict(c["robot"], models_dir=self.robot_dir),
                "physics": copy.deepcopy(c.get("physics") or {}),
                "lmag": bool(c["layout"].get("lmag", False)),
                "grip_force_N": float(c["grip_force_N"]),
                "watch": {"enabled": True, "robot_force_N": float(c["safety"]["robot_force_N"]),
                          "part_force_N": float(c["damage"]["part_force_N"]), "part_force_ticks": int(c["damage"]["ticks"]),
                          "dome_force_N": float(c["safety"]["dome_force_N"]), "dome_ticks": int(c["safety"]["dome_ticks"]),
                          "scrap_speed_m_s": float(c["safety"]["part_speed_m_s"]),
                          "scrap_below_bench_m": float(c["scrap"]["below_bench_m"]), "scrap": True, "column_while_turning": True},
                "positioner": {"max_turns": int(fxc["max_turns"]), "turn_s": float(fxc["turn_s"]),
                               "noise": {"sigma_N": float(fxc["load_cells"]["sigma_N"]), "bias_max_N": float(fxc["load_cells"]["bias_N"])}},
                "budget": {"sim_s": float(c["sim_cap_s"]), "wall_s": float(c["wall_cap_s"]), "frames": int(c["camera_budget_frames"]),
                           "step_ticks": int(c["max_ticks_per_step"]), "turns": int(fxc["max_turns"])}}

    def reset(self, entry):
        """build the episode's cell from one episode of the fixed hidden set (module doc); returns {"ok", "layout_digest",
        "stratum", "dome_end_deg", "nominal"}"""
        e = dict(entry)
        seed = int(e["seed"])
        dome = int(e["dome_end_deg"])
        if dome not in DOME_ENDS:
            raise ValueError("dome_end_deg must be 0 or 180")
        dev = parse_deviation(e["deviation"])
        wo = e["work_order"]
        site = wo["delivery_site"]
        _finite_number(site["water_temperature_C"], "water_temperature_C")
        _finite_number(site["salinity_psu"], "salinity_psu")
        self.entry = {"seed": seed, "stratum": str(e.get("stratum")), "dome_end_deg": dome,
                      "work_order": {"unit": str(wo["unit"]), "delivery_site": {"water_temperature_C": float(site["water_temperature_C"]),
                                                                                "salinity_psu": float(site["salinity_psu"])}},
                      "deviation": {"dm_kg": dev[0], "dx_mm": dev[1], "dy_mm": dev[2]}}
        nominal = bool(self.privileged and self.cfg.get("oracle_nominal_layout", False))
        t0 = time.time()
        lay = LY.episode_layout(seed, fx=self.fx, cfg=self.cfg["layout"], stratum="dome%d" % dome, nominal=nominal)
        self.layout = lay
        self.layout_digest = LY.layout_digest(lay)
        kit_e = vehicle_kit(self.kit, dev, self.table)
        self.kit_episode = kit_e
        cell = RovCell(self.kit_dir, lay, config=self.cell_config(), log=self.log, settle_s=float(self.cfg["settle_s"]), kit=kit_e, fx=self.fx)
        self.log("[cell] rov cell for seed %d built in %.1f s (stratum %s, dome %d%s)" % (
            seed & 0xFFFFFFFF, time.time() - t0, self.entry["stratum"], dome, ", nominal" if nominal else ""))
        cell.recording = False
        self.cell = cell
        self.model, self.data = cell.m, cell.d
        self.ctrl, self.sm = cell.ctrl, cell.sm
        self.robot = cell.robot
        # the information tier's draws, the wrist camera's mount from the built robot (when the configuration also
        # states the mount's numbers, the documents print them: they must be the robot's)
        stated = {k: self.cfg["info_layout"]["cameras"]["wrist"].get(k) for k in ("offset_m", "z_m", "tilt_deg")}
        self.cfg["info_layout"] = RT.info_layout(self.model, self.data, self.robot, self.cfg["info_layout"])
        got = self.cfg["info_layout"]["cameras"]["wrist"]
        for k, v in stated.items():
            if v is not None and abs(float(v) - float(got[k])) > (1e-4 if k == "tilt_deg" else 1e-5):
                raise ValueError("grader_config.json: info_layout.cameras.wrist.%s %r is not the robot's wrist camera mount %r" % (k, v, got[k]))
        self.episode = LI.draw_episode(seed, self.cfg["info_layout"], strict=False)
        missing = [n for n in self.camera_names if self.episode["cameras"].get(n) is None]
        if missing and self.render_enabled:
            raise LI.MissingMount("camera(s) %s have no mount in the cell's config" % ", ".join(missing))
        self._init_state(seed)
        self._close_session()
        self._raw_tick = cell.tick_one

        def routed():
            self.physics_tick(track=True)
        cell.tick_one = routed
        return {"ok": True, "layout_digest": self.layout_digest, "stratum": self.entry["stratum"], "dome_end_deg": dome, "nominal": nominal}

    def _init_state(self, seed):
        m, cell = self.model, self.cell
        self.seed = self.ep_seed = int(seed)
        self.done = False
        self.done_reason = None
        self.tick_count = 0
        self.events_log = []
        self.illegal_calls = 0
        self.calls = {}
        self.step_calls = 0
        self.steps_clamped = 0
        self.fixture_calls = {"ok": 0, "refused": 0}
        self.wall_start = None
        self.wall_used = 0.0
        self.renderer = None
        self.reference_report = None
        self.reference_calls = 0           # answered (privileged) reference_assemble calls: they stand for the program's steps
        self.frames_used = 0
        self.frames_by_camera = {n: 0 for n in self.camera_names}
        self.render_wall_s = 0.0
        self.render_failures = 0
        self.tick_s = cell.tick_s
        self.tick_steps = cell.tick_steps
        hi = (int(seed) >> 32) & 0xFFFFFFFF
        self._wrench_rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x3E17C4] + ([hi] if hi else []))
        lc = self.cfg["fixture"]["load_cells"]
        self.lc_noise = PZ.LoadCellNoise(int(seed), sigma_N=float(lc["sigma_N"]), bias_N=self.layout["cell_bias_N"])
        self._lc_tick = None
        self._lc_value = None
        self.lc_samples = 0
        self.sensor_force = m.sensor(self.robot["force_sensor"]).id
        self.sensor_torque = m.sensor(self.robot["torque_sensor"]).id

    @property
    def violations(self):
        return self.cell.violations if self.cell is not None else []

    # ------------------------------------------------------------------ physics
    def physics_tick(self, track=True):
        if self.done:
            return []
        cell = self.cell
        n0, w0 = len(self.sm.events), len(cell.watch_events)
        self._raw_tick()
        self.tick_count += 1
        events = []
        if track:
            events += [str(e) for _t, e in self.sm.events[n0:]]
            events += [str(e) for _t, e in cell.watch_events[w0:]]
            for e in events:
                self.events_log.append((float(self.data.time), e))
        if cell.done == "sim_cap" or self.data.time >= float(self.cfg["sim_cap_s"]) - 1e-9:
            self._end("sim_cap")
        return events

    def _end(self, reason):
        if not self.done:
            self.done = True
            self.done_reason = reason

    # ------------------------------------------------------------------ the public API
    def _load_cells(self):
        """this tick's load-cell sample (one per control tick, the same for every call within the tick)"""
        k = int(self.cell.n_tick)
        if self._lc_tick != k:
            self._lc_value = self.lc_noise.sample(self.cell.P.reading())
            self._lc_tick = k
            self.lc_samples += 1
        return [float(v) for v in self._lc_value]

    def get_state(self):
        from .ur_controller import mat_to_quat
        d = self.data
        p, R = self.ctrl.tcp_actual()
        adr = self.model.sensor_adr
        w = np.array(list(d.sensordata[adr[self.sensor_force]:adr[self.sensor_force] + 3]) +
                     list(d.sensordata[adr[self.sensor_torque]:adr[self.sensor_torque] + 3]))
        wn = self.episode["config"]["wrench_noise"]
        w = w + np.asarray(self.episode["wrench_bias"], float) + self._wrench_rng.normal(0.0, 1.0, 6) * np.array([wn["sigma_f_N"]] * 3 + [wn["sigma_t_Nm"]] * 3)
        P = self.cell.P
        return {"sim_time": float(d.time), "done": bool(self.done),
                "robot": {"q": [float(v) for v in self.ctrl.q()], "dq": [float(v) for v in self.ctrl.dq()],
                          "cmd_q": [float(v) for v in self.ctrl.q_cmd], "gripper_width": float(self.ctrl.width()),
                          "tcp_pos": [float(v) for v in p], "tcp_quat": [float(v) for v in mat_to_quat(R)],
                          "wrench": [float(v) for v in w], "mode": self.ctrl.mode, "stiffness_scale": float(self.ctrl.stiffness_scale)},
                "fixture": {"index": P.index(), "moving": bool(P.moving()), "load_N": self._load_cells()}}

    def set_control(self, mode="joint", stiffness_scale=1.0):
        if mode not in ("joint", "cartesian"):
            raise ValueError("mode must be joint or cartesian")
        s = _finite_number(stiffness_scale, "stiffness_scale")
        self.ctrl.set_mode(str(mode))
        self.ctrl.set_stiffness(s)
        return {"ok": True, "mode": self.ctrl.mode, "stiffness_scale": float(self.ctrl.stiffness_scale)}

    def command(self, q=None, tcp_pos=None, tcp_quat=None, width=None, force=None):
        if self.done:
            return {"ok": False, "error": "episode_over"}
        # every argument is checked before any of them is applied (a malformed call changes nothing)
        qv = _finite_list(q, 6, "q") if q is not None else None
        pv = _finite_list(tcp_pos, 3, "tcp_pos") if tcp_pos is not None else None
        rv = _finite_list(tcp_quat, 4, "tcp_quat") if tcp_quat is not None else None
        wv = _finite_number(width, "width") if width is not None else None
        fv = _finite_number(force, "force") if force is not None else None
        if rv is not None and math.sqrt(sum(x * x for x in rv)) < 1e-6:
            raise ValueError("tcp_quat must not be zero")
        if qv is not None and self.ctrl.mode != "joint":
            return {"ok": False, "error": "q target needs joint mode"}
        if (pv is not None or rv is not None) and self.ctrl.mode != "cartesian":
            return {"ok": False, "error": "tcp target needs cartesian mode"}
        if (pv is None) != (rv is None):
            return {"ok": False, "error": "tcp_pos and tcp_quat go together"}
        if qv is not None:
            self.ctrl.command_joint(qv)
        if pv is not None:
            self.ctrl.command_cartesian(pv, rv)
        if wv is not None:
            self.ctrl.command_width(wv)
        if fv is not None:
            self.ctrl.command_force(fv)
        return {"ok": True}

    def step(self, n_ticks=1):
        if isinstance(n_ticks, bool) or not isinstance(n_ticks, (int, float)) or not math.isfinite(float(n_ticks)):
            raise ValueError("n_ticks must be a number")
        n = int(n_ticks)
        if n < 1:
            return {"ok": False, "error": "n_ticks must be >= 1"}
        cap = int(self.cfg["max_ticks_per_step"])
        if n > cap:
            self.steps_clamped += 1
            n = cap
        self.step_calls += 1
        self.cell.budget.charge_step(n)
        events = []
        for _ in range(n):
            if self.done:
                break
            events += self.physics_tick(track=True)
        ev = [e for e in events if e.startswith("safety:")]
        if self.done:
            ev.append("episode_over")
        return {"sim_time": float(self.data.time), "done": bool(self.done), "events": ev}

    def fixture_index(self, target=None):
        if target not in INDEXES:
            raise ValueError("target must be keel_up or deck_up")
        if self.done:
            return {"ok": False, "error": "episode_over"}
        ans = self.cell.fixture_index(str(target))
        self.fixture_calls["ok" if ans.get("ok") else "refused"] += 1
        if ans.get("ok"):
            return {"ok": True}
        return {"refused": str(ans.get("refused"))}

    def finish(self):
        self._end("finish")
        return {"ok": True, "sim_time": float(self.data.time)}

    # ------------------------------------------------------------------ describe()
    def _values(self):
        """the numbers the agent-visible strings print"""
        fx, ct = self.fx, {e["slot"]: e for e in self.sm.capture_table()}
        cams = self.cfg["info_layout"]["cameras"]
        o = fx.vehicle_origin()
        tray = fx.station_spec("thruster_tray")
        rack = fx.station_spec("enclosure_rack")
        gp, gs = fx.station_spec("guard_rack_port"), fx.station_spec("guard_rack_stbd")
        ee, vt = ct["ee"], ct["vt_FR"]
        return {
            "still_mm_s": _f(ee["capture"]["still_mm_s"], 0), "still_deg_s": _f(ee["capture"]["still_deg_s"], 0),
            "still_ticks": _f(ee["capture"]["still_ticks"], 0), "pull_mm_s": _f(ee["pull"]["speed_mm_s"], 0),
            "pull_deg_s": _f(ee["pull"]["rate_deg_s"], 0), "hold_ticks": _f(ee["lock"]["hold_ticks"], 0),
            "vt_n": _f(vt["orientations"]["count"], 0), "vt_pitch": _f(vt["orientations"]["pitch_deg"], 0),
            "hold_release_s": _f(vt["hold_before_release_s"], 1),
            "top_h": _f(cams["top"]["eye"][2], 1), "wrist_cm": _f(100.0 * float(cams["wrist"]["offset_m"]), 0),
            "wrist_tilt": _f(cams["wrist"]["tilt_deg"], 0),
            "veh_x": _f(o[0], 2), "veh_y": _f(o[1], 2), "veh_z": _f(o[2], 2),
            "blocks_n": _f(fx.st["ballast_box"]["blocks"], 0),
            "tray_x": _f(tray[1][0], 2), "tray_y": _f(tray[1][1], 2), "tray_xy": _f(1e3 * tray[2][0], 0), "tray_yaw": _f(tray[2][1], 0),
            "rack_x": _f(rack[1][0], 2), "rack_y": _f(rack[1][1], 2), "rack_xy": _f(1e3 * rack[2][0], 0), "rack_yaw": _f(rack[2][1], 0),
            "roll": _f(fx.st["enclosure_rack"]["roll_range_deg"], 0),
            "gp_x": _f(gp[1][0], 2), "gp_y": _f(gp[1][1], 2), "gs_x": _f(gs[1][0], 2), "gs_y": _f(gs[1][1], 2),
            "g_xy": _f(1e3 * gp[2][0], 0), "g_yaw": _f(gp[2][1], 0),
        }

    @staticmethod
    def _windows(e):
        cap, rel, lock = e["capture"], e["release"], e["lock"]
        return {"capture": {"across_mm": _r(cap["lateral_mm"]), "short_of_home_mm": _r(cap["from_home_mm"]),
                            "past_home_mm": _r(cap["past_home_mm"]), "tilt_deg": _r(cap["tilt_deg"]), "turn_deg": _r(cap["clock_deg"])},
                "at_rest_before_capture": {"slower_than_mm_s": _r(cap["still_mm_s"]), "slower_than_deg_s": _r(cap["still_deg_s"]),
                                           "ticks": int(cap["still_ticks"]), "s": _r(cap["still_s"]),
                                           "speed_of": "%s, relative to the vehicle" % cap["still_speed_of"],
                                           "turn_of": "the unit, relative to the vehicle"},
                "release": {"across_mm": _r(rel["lateral_mm"]), "short_of_home_mm": _r(rel["from_home_mm"]),
                            "tilt_deg": _r(rel["tilt_deg"]), "turn_deg": _r(rel["clock_deg"])},
                "lock": {"pos_mm": _r(lock["pos_mm"]), "ang_deg": _r(lock["ang_deg"]), "along_mm": _r(lock["axial_mm"]),
                         "hold_ticks": int(lock["hold_ticks"]), "rest_speed_mm_s": _r(lock["rel_speed_mm_s"]),
                         "rest_turn_deg_s": _r(lock["rel_ang_speed_deg_s"])},
                "pull": {"speed_mm_s": _r(e["pull"]["speed_mm_s"]), "rate_deg_s": _r(e["pull"]["rate_deg_s"])},
                "physical_key": bool(e.get("physical_key", False)),
                "stays_without_being_held": bool(e["self_standing"]),
                "hold_still_before_release_s": None if e["hold_before_release_s"] is None else _r(e["hold_before_release_s"])}

    def _places(self):
        """describe()["fastening"]["places"]: the seven scored places one by one and the ballast sites as one entry (their
        windows are the same; the sites are the free holes of keel_hole_map.png); no frame, no pose, no part type"""
        out = []
        sites = []
        for e in self.sm.capture_table():
            if e["premounted"]:
                continue
            if not e["scored"]:
                sites.append(e)
                continue
            sid = e["slot"]
            key = "vt" if sid.startswith("vt_") else ("guard" if sid.startswith("guard_") else sid)
            unit, home, along = PLACE_WORDS[key]
            o = e["orientations"]
            p = {"place": sid, "unit": unit, "home": home, "goes_on": along, "weight": float(e["weight"]),
                 "orientations": {"captured": int(o["count"]), "counting": 1, "apart_deg": _r(o["pitch_deg"]) if int(o["count"]) > 1 else None}}
            if sid in PLACE_SIDE:
                p["side"] = PLACE_SIDE[sid]
            p.update(self._windows(e))
            out.append(p)
        if sites:
            e = sites[0]
            unit, home, along = PLACE_WORDS["ballast"]
            o = e["orientations"]
            p = {"place": "ballast site", "unit": unit, "home": home, "goes_on": along, "weight": 0.0,
                 "orientations": {"captured": int(o["count"]), "counting": int(o["count"]), "apart_deg": _r(o["pitch_deg"])}}
            p.update(self._windows(e))
            out.append(p)
        return out

    def _fastening(self):
        v = self._values()
        places = self._places()
        scored = {p["place"]: p["weight"] for p in places if p["place"] != "ballast site"}
        return {"summary": FASTENING_SUMMARY.format(**v), "places": places,
                "rules": [r.format(**v) for r in FASTENING_RULES],
                "weights": scored, "structure_weight": float(sum(scored.values())),
                "trim_weight": float(self.cfg["score"]["trim_weight"]), "total_weight": float(self.cfg["score"]["total_weight"])}

    def _robot_block(self):
        from . import gripper_wide as GW
        from . import ur_controller as UC
        from . import ur_robot as UR
        m, d = self.model, self.data
        rb, ct = self.robot, self.ctrl
        b = m.body(rb["base_body"]).id
        Rb = d.xmat[b].reshape(3, 3)
        turn = math.degrees(math.atan2(Rb[1, 0], Rb[0, 0]))
        wc = GW.DEFAULT_CFG["wrist_camera"]
        g = abs(float(m.opt.gravity[2]))
        tool_mass = float(m.body_subtreemass[m.body(rb["tool_body"]).id])
        wn = self.episode["config"]["wrench_noise"]
        return {
            "model": "6-DoF UR10e-class arm (given model: robot/%s/) with a wide-stroke parallel gripper (robot/%s/)" % (
                UR.UR_MODELS[self.cfg["robot"]["model"]]["dir"], GW.GRIPPER_DIRNAME),
            "base_pos": [float(x) for x in self.cfg["robot"]["pos"]],
            "base_turn_deg": _r(turn, 3),
            "base_axes": "the arm's base frame is the cell frame turned %s deg about z: q = 0 on the first joint points the arm along "
                         "the cell's %s" % (_f(abs(turn), 0), "-x" if abs(abs(turn) - 180.0) < 1.0 else "x turned by that angle"),
            "joints": list(rb["joints"]),
            "joint_range": [[float(m.jnt_range[m.joint(j).id][0]), float(m.jnt_range[m.joint(j).id][1])] for j in rb["joints"]],
            "home_q": [float(x) for x in ct.q_cmd],
            "torque_limits_Nm": [float(x) for x in rb["torque_limits_Nm"]],
            "joint_speed_limits_rad_s": [float(x) for x in rb["speed_limits_rad_s"]],
            "tcp_site": rb["tcp_site"], "tcp_offset_from_flange_m": [0.0, 0.0, _r(rb["tcp_from_flange_m"], 4)],
            "tcp_axes": "z = the approach (the flange axis, out of the flange), y = the closing axis, x across the pads",
            "gripper": {"width_range_m": [_r(GW.width_of_ctrl(255.0), 4), _r(ct.max_width, 4)], "rate_m_s": float(UC.GRIP_RATE_M_S),
                        "width": "the gap between the two pad faces at the faces' middle",
                        "force_range_N": [float(GW.FORCE_RANGE_N[0]), float(GW.FORCE_RANGE_N[1])],
                        "force_default_N": float(self.cfg["grip_force_N"]),
                        "force": "the grip force per pad with which the pads close on a unit (command(force=...)); it stays until "
                                 "the next force",
                        "pad_face_mm": [_r(1e3 * GW.PAD_ACROSS_M, 1), _r(1e3 * GW.PAD_ALONG_M, 1)],
                        "fingertip_blade_mm": {"length": _r(1e3 * GW.TIP_SEG_M, 1), "thickness": _r(1e3 * GW.DESIGN["tip_thick_m"], 1)}
                        if "tip_thick_m" in GW.DESIGN else {"length": _r(1e3 * GW.TIP_SEG_M, 1)},
                        "pinch_centre": "off the tcp along the closing axis; the given gripper module's pad_offset(width) returns it",
                        "mass_kg": _r(tool_mass, 3)},
            "wrist_housing": "a %s x %s x %s mm camera housing on the gripper's base, on the tool's +x side (part of the robot)" % tuple(
                _f(2e3 * float(h), 0) for h in wc["housing_half_m"]),
            "gravity_compensated": True,
            "ft_sensor": ("6-axis at the tool flange's centre in the flange frame (z along the tool axis outwards, x along the closing "
                          "axis = the tcp's y, y along the tcp's -x); what the wrist applies to the tool, the gripper's own weight "
                          "included (about %s N with the tool pointing down): tare it; noise ~%s N / %s N m rms per reading, an "
                          "unknown bias up to %s N / %s N m per episode") % (
                              _f(tool_mass * g, 1), _f(wn["sigma_f_N"], 1), _f(wn["sigma_t_Nm"], 2), _f(wn["bias_f_N"], 0),
                              _f(wn["bias_t_Nm"], 2)),
        }

    def describe(self):
        from . import ur_controller as UC
        v = self._values()
        cams = {}
        for n in self.camera_names:
            c = self.episode["config"]["cameras"][n]
            cams[n] = {"resolution": list(c["size"]), "rgb": True, "depth": bool(c.get("depth", True)),
                       "nominal_hfov_deg": round(float(c["hfov_deg"])), "mount": MOUNT_TEXT.get(n, "").format(**v)}
        dmg, sf, fxc = self.cfg["damage"], self.cfg["safety"], self.cfg["fixture"]
        hd = self.cfg["handling"]
        return {
            "cell_version": CELL_VERSION, "units": {"length": "m", "angle": "rad", "quaternion": "wxyz", "force": "N"},
            "task": TASK_TEXT,
            "work_order": json.loads(json.dumps(self.entry["work_order"])),
            "robot": self._robot_block(),
            "control": {"tick_s": float(self.tick_s), "joint_rate_limit_rad_s": float(min(self.ctrl.rate_limit)),
                        "cartesian_speed_limit_m_s": float(UC.CART_V_MAX), "cartesian_angular_limit_rad_s": float(UC.CART_W_MAX),
                        "stiffness_scale_range": [float(UC.STIFFNESS_RANGE[0]), float(UC.STIFFNESS_RANGE[1])],
                        "modes": ["joint", "cartesian"],
                        "handling_procedure": {"tcp_speed_m_s": float(hd["tcp_speed_m_s"]), "joint_acc_rad_s2": float(hd["joint_acc_rad_s2"]),
                                               "applies_to": "the gripper carrying a unit; the empty gripper is not limited",
                                               "enforced": False}},
            "fixture": {"indexes": list(INDEXES), "start": self.cell.P.cfg["start_index"], "turn_s": float(fxc["turn_s"]),
                        "turns_max": int(fxc["max_turns"]), "column_radius_m": float(self.fx.column()["radius_m"]),
                        "load_cells": {"order": ["A", "B", "C"], "unit": "N", "compression": "positive", "tared": True,
                                       "sample_every_tick": True, "sigma_N": float(fxc["load_cells"]["sigma_N"]),
                                       "zero_offset_up_to_N": float(fxc["load_cells"]["bias_N"]),
                                       "valid_after_s": float(self.cell.P.cfg["reading_valid_after_s"]),
                                       "stations": "positioner_and_cells.png (drawn at deck_up); the readings are not converted "
                                                   "into the vehicle's frame"}},
            "fastening": self._fastening(),
            "cameras": cams, "camera_budget_frames": int(self.cfg["camera_budget_frames"]),
            "datums": rov_datums(self.fx),
            "items": [t.format(**v) for t in ITEMS_TEXT],
            "budgets": {"sim_time_cap_s": float(self.cfg["sim_cap_s"]), "wall_clock_cap_s": float(self.cfg["wall_cap_s"]),
                        "silence_abort_s": float(self.cfg["silence_s"]), "max_ticks_per_step": int(self.cfg["max_ticks_per_step"]),
                        "camera_frames": int(self.cfg["camera_budget_frames"]), "turns": int(fxc["max_turns"])},
            "safety": {"robot_force_N": float(sf["robot_force_N"]), "part_speed_m_s": float(sf["part_speed_m_s"]),
                       "part_force_N": float(dmg["part_force_N"]), "part_force_window_s": _r(self.tick_s * int(dmg["ticks"])),
                       "dome_force_N": float(sf["dome_force_N"]), "dome_window_s": _r(self.tick_s * int(sf["dome_ticks"])),
                       "off_bench_m": float(self.cfg["scrap"]["below_bench_m"]), "rules": list(SAFETY_RULES)},
        }

    # ------------------------------------------------------------------ cameras (render_twin's pipeline)
    def _session(self):
        if self.session is None:
            t = time.time()
            want = str(self.cfg.get("agent_renderer", "filament"))
            if want == "filament":
                from . import agent_render
                ok, why = agent_render.available()
                if not ok:
                    raise RuntimeError("agent renderer filament unavailable: %s" % why)
                self.session = agent_render.open_session(self)
                self.renderer = "filament"
            else:
                self.session = RT.open_session(self)
                self.renderer = "mujoco"
            self.render_wall_s += time.time() - t
            self.log("[cell] %s render session opened in %.1f s" % (self.renderer, time.time() - t))
        return self.session

    def session_open(self):
        return self._session()

    def _close_session(self):
        if getattr(self, "session", None) is not None:
            try:
                self.session.close()
            except Exception:  # noqa: BLE001
                pass
        self.session = None

    def camera_spec(self, name):
        return RT.camera_spec(self, name)

    def photo_spec(self, name):
        return RT.photo_spec(self, name)

    def get_camera(self, name="top", depth=True):
        import base64
        name = str(name)
        if name not in self.camera_names:
            return {"error": "unknown camera (cameras: %s)" % ", ".join(self.camera_names)}
        if not self.render_enabled or self.episode["cameras"].get(name) is None:
            return {"error": "camera_unavailable"}
        if self.frames_used >= int(self.cfg["camera_budget_frames"]):
            return {"error": "camera_budget"}
        k = self.frames_used
        self.frames_used += 1
        self.frames_by_camera[name] = self.frames_by_camera.get(name, 0) + 1
        self.cell.budget.charge_frames(name)
        want_depth = bool(depth) and bool(self.episode["cameras"][name].get("depth", True))
        outputs = ("rgb", "sensor_depth") if want_depth else ("rgb",)
        t = time.time()
        try:
            r = RT.render_frame(self, name, outputs=outputs, k=k)
        except Exception as e:  # noqa: BLE001 - a renderer failure never takes the cell down
            self.log("[cell] %s frame %d not rendered: %s: %s" % (name, k, type(e).__name__, str(e)[:200]))
            self.render_failures += 1
            self.render_wall_s += time.time() - t
            self.frames_used = k + 1
            return {"error": "camera_unavailable"}
        self.frames_used = k + 1
        v = r.agent_view()
        out = {"ok": True, "camera": name, "frame_index": int(k), "width": int(v["rgb"][0].shape[1]), "height": int(v["rgb"][0].shape[0]),
               "format": "png", "rgb_png_b64": base64.b64encode(v["encoded"][0]).decode("ascii")}
        if want_depth:
            out["sensor_depth_u16_mm_b64"] = base64.b64encode(v["sensor_depth_encoded"][0]).decode("ascii")
            out["depth_unit_m"] = 0.001
            out["depth_invalid_value"] = 0
        self.render_wall_s += time.time() - t
        return out

    # ------------------------------------------------------------------ privileged (token only)
    def describe_privileged(self):
        out = {"layout": self.layout, "layout_digest": self.layout_digest, "entry": self.entry, "slots": self.kit["slots"],
               "capture_table": self.sm.capture_table(), "states": dict(self.sm.state), "score": self.sm.score()}
        return json.loads(json.dumps(_jsonable(out), default=str))

    def get_state_privileged(self):
        d = self.data
        parts = {n: {"pos": [float(x) for x in d.xpos[self.model.body(n).id]], "quat": [float(x) for x in d.xquat[self.model.body(n).id]]}
                 for n in self.cell.loose}
        veh = self.model.body(self.kit["base_body"]).id
        return {"sim_time": float(d.time), "parts": parts,
                "vehicle": {"pos": [float(x) for x in d.xpos[veh]], "quat": [float(x) for x in d.xquat[veh]]},
                "fixture_exact_N": [float(x) for x in self.cell.P.reading()], "states": dict(self.sm.state),
                "records": _jsonable(self.sm.record()), "scrapped": _jsonable(self.sm.scrapped), "score": float(self.sm.score())}

    def reference_assemble(self):
        """The privileged reference program on this episode (`reference_ee.reference_assemble(cell, server=..., log=...)`,
        the whole episode: R1 ballast and turn, R2 / R3 guards and thrusters, R4 the enclosure; imported here on first
        use), through the cell's routed ticks and its watch."""
        t0, s0 = time.time(), float(self.data.time)
        self.reference_calls += 1
        try:
            from . import reference_ee as reference
        except ImportError as e:
            self.log("[cell] the reference program is not available: %r" % (e,))
            return {"ok": False, "error": "reference_unavailable: %s" % type(e).__name__}
        entry = getattr(reference, "reference_assemble", None)
        if entry is None:
            return {"ok": False, "error": "reference_unavailable: no entry point"}
        try:
            rep = entry(self.cell, server=self, log=self.log) or {}
        except Exception as e:  # noqa: BLE001 - the episode is still finalized with what was fastened
            self.log("[cell] the reference program failed:\n%s" % traceback.format_exc())
            return {"ok": False, "error": "reference_failed: %s" % type(e).__name__, "score": float(self.sm.score())}
        self.reference_report = rep
        return {"ok": True, "score": float(self.sm.score()), "states": dict(self.sm.state),
                "steps": int(len(rep.get("steps") or [])), "unplanned_contacts": int(len(rep.get("unplanned_contacts") or [])),
                "sim_s": float(self.data.time) - s0, "wall_s": float(time.time() - t0)}

    # ------------------------------------------------------------------ dispatch
    def handle(self, method, params):
        method = str(method)
        self.calls[method] = self.calls.get(method, 0) + 1
        if method in PRIVILEGED_METHODS:
            if not self.privileged:
                self.illegal_calls += 1
                self.cell.budget.charge_illegal(method)
                return {"error": "forbidden"}
            return getattr(self, method)()
        if method not in PUBLIC_METHODS:
            self.illegal_calls += 1
            if self.cell is not None:
                self.cell.budget.charge_illegal(method[:40])
            return {"error": "forbidden"}
        try:
            return getattr(self, method)(**(params or {}))
        except (TypeError, ValueError, KeyError, IndexError) as e:     # malformed parameters: the caller's error, never a wire crash
            return {"error": "bad_params: %s" % str(e)[:200]}

    # ------------------------------------------------------------------ end of episode
    def finalize(self, functional_rng=None, run_functional_check=False):
        self._close_session()
        if not self.done:
            self._end("solve_returned")
        sm, cell = self.sm, self.cell
        rec = sm.record()
        ws = cell.watch_summary()
        hard = [v for v in ws["violations"] if v["what"].split(":")[0] in ("robot_contact", "part_force")]
        hard_ok = not hard
        sc = score_episode(rec, self.table, self.entry["deviation"], self.entry["work_order"], float(self.cfg["score"]["trim_weight"]), hard_ok)
        trim_sites = [e["name_03"] for e in sm.trim_sites()]
        if sorted(trim_sites) != sorted(sc["holes"]):
            raise RuntimeError("the coupling's trim sites %s differ from its record's %s" % (trim_sites, sc["holes"]))
        scored = [r for r in rec if r.get("scored", True)]
        sites = [r for r in rec if not r.get("scored", True)]
        d = self.data
        cm = cell.metrics()
        return {
            "cell_version": CELL_VERSION, "seed": self.seed, "stratum": self.entry["stratum"], "dome_end_deg": self.entry["dome_end_deg"],
            "layout_digest": self.layout_digest, "nominal_layout": bool(self.layout.get("nominal")), "done_reason": self.done_reason,
            "sim_time_s": float(d.time), "wall_time_s": float(self.wall_used), "ticks": int(self.tick_count),
            "mates": _jsonable(scored), "sites": _jsonable(sites), "states": dict(sm.state),
            "weights": {r["id"]: float(r["weight"]) for r in scored}, "weight_total": sc["total_w"], "weight_counted": sc["counted_w"],
            "stage_score": sc["stage_score"], "stage_score_gated": sc["stage_score_gated"], "score": sc,
            "trim": sc["trim"], "trim_eval": sc["trim_eval"], "ballast_holes": sc["holes"],
            "n_achieved": sc["n_counted"], "n_fastened": int(sum(1 for r in scored if r.get("fastened"))),
            "all_mates": bool(scored) and all(r.get("achieved") for r in scored),
            "locked_off": [r["id"] for r in scored if r.get("fastened") and r.get("design_ok") is False],
            "scrapped": _jsonable(sm.scrapped),
            "violations": _jsonable(ws["violations"]), "hard_ok": hard_ok,
            "robot_contact_violations": int(sum(1 for v in hard if v["what"].startswith("robot_contact"))),
            "part_force_violations": int(sum(1 for v in hard if v["what"].startswith("part_force"))),
            "watch": _jsonable({k: v for k, v in ws.items() if k != "violations"}),
            "cell_metrics": _jsonable({k: v for k, v in cm.items() if k != "watch"}),
            "illegal_call_count": int(self.illegal_calls), "calls": dict(self.calls), "step_calls": int(self.step_calls),
            "reference_calls": int(self.reference_calls),
            "steps_clamped": int(self.steps_clamped), "fixture_calls": dict(self.fixture_calls),
            "turns_used": int(cell.P.turns_used), "load_cell_samples": int(self.lc_samples),
            "events": [[t, e] for t, e in self.events_log][-400:], "coupling_events": [list(e) for e in sm.events][-400:],
            "camera_frames": dict(self.frames_by_camera), "camera_frames_total": int(self.frames_used),
            "render_wall_s": float(self.render_wall_s), "render_failures": int(self.render_failures), "renderer": self.renderer,
            "privileged_mode": bool(self.privileged), "work_order": self.entry["work_order"], "deviation": self.entry["deviation"],
            "reference": None if self.reference_report is None else _jsonable(
                {k: self.reference_report.get(k) for k in ("ok", "steps", "unplanned_contacts") if k in self.reference_report}),
        }


def rov_datums(fx):
    """The documented datums (the cell manual's tables; exact where the manual says so, nominal otherwise): the boards, the
    vehicle frame at the two index positions, the ballast box's exact place, the nominal places of the tray and the racks."""
    boards = fx.boards()
    out = {"frame": "cell frame: origin on the bench top at the robot arm's base axis, z up, x towards the turnover positioner; the "
                    "bench top is z = 0 and spans x %.2f..%.2f m, y %.2f..%.2f m" % (FX.TABLE["x_m"][0], FX.TABLE["x_m"][1],
                                                                                     FX.TABLE["y_m"][0], FX.TABLE["y_m"][1])}
    b = boards["bench"]
    T = np.asarray(b["T_rest"], float)
    out["bench_board"] = {"squares": [b["cols"], b["rows"]], "square_mm": b["square_mm"], "margin_mm": b["margin_mm"],
                          "plate_mm": [_r(x, 3) for x in b["plate_mm"]], "surveyed": True,
                          "origin_cell_m": [_r(T[0, 3], 4), _r(T[1, 3], 4), _r(b["face_height_mm"] / 1000.0, 5)],
                          "inner_corner": "(i - C / 2) x s, (R / 2 - j) x s in the board's frame (x along the C squares), i = 1..C-1, "
                                          "j = 1..R-1; the square at the board's (-x, +y) corner is black"}
    t = FX.TOOL_PLATE
    out["tool_plate"] = {"squares": [t["cols"], t["rows"]], "square_mm": t["square_mm"], "margin_mm": t["margin_mm"],
                         "thickness_mm": t["thickness_mm"], "mounting": "on the camera housing's outer face, facing the tool's +x",
                         "mounting_tolerance": "about 1 mm / 0.5 deg"}
    for key in sorted(boards):
        if key == "bench":
            continue
        bd = boards[key]
        out[key + "_board"] = {"squares": [bd["cols"], bd["rows"]], "square_mm": bd["square_mm"], "margin_mm": bd["margin_mm"],
                               "plate_mm": [_r(x, 3) for x in bd["plate_mm"]], "printed_face_above_bench_mm": _r(bd["face_height_mm"], 3),
                               "moves_with": bd["fixture"], "surveyed": False, "where": bd["where"]}
    veh = {}
    for idx in INDEXES:
        T = fx.T_cell_B(idx)
        veh[idx] = {"origin_cell_m": [_r(x, 4) for x in T[:3, 3]], "axes_in_cell": [[_r(x, 6) for x in T[:3, k]] for k in range(3)]}
    out["vehicle"] = {"frame": "the frame of base_vehicle.stl: x forward, y to port, z up (mm in the files)", "at_index": veh,
                      "exact_within": "the locating pins and the indexing repeatability (cell_manual.md)"}
    s, nom, _j = fx.station_spec("ballast_box")
    out["ballast_box"] = {"origin_cell_m": [_r(nom[0], 4), _r(nom[1], 4)], "turn_deg": _r(nom[2], 3), "exact": not fx.lmag,
                          "pockets": [int(s["pockets_x"]), int(s["pockets_y"])], "pitch_mm": [_r(s["pitch_x_mm"], 2), _r(s["pitch_y_mm"], 2)],
                          "blocks": int(s["blocks"])}
    if fx.lmag:
        out["ballast_box"]["tolerance_mm"] = _r(1e3 * float(FX.LMAG["jitter_xy_m"]), 2)
        out["ballast_box"]["turn_up_to_deg"] = _r(FX.LMAG["jitter_yaw_deg"], 2)
    for name, key in (("thruster_tray", "thruster_tray"), ("enclosure_rack", "enclosure_rack"), ("guard_rack_port", "guard_rack_port"),
                      ("guard_rack_stbd", "guard_rack_stbd")):
        s, nom, jit = fx.station_spec(name)
        out[key] = {"nominal_origin_cell_m": [_r(nom[0], 4), _r(nom[1], 4)], "nominal_turn_deg": _r(nom[2], 3),
                    "tolerance_mm": _r(1e3 * float(jit[0]), 2), "turn_up_to_deg": _r(jit[1], 2)}
    out["enclosure_rack"]["tube_axis_above_bench_mm"] = _r(fx.st["enclosure_rack"]["tube_axis_z_mm"], 2)
    out["level_datums"] = "the bench top (z = 0) and the top face of the thruster tray are level"
    return out


__all__ = ["RovInfoServer", "CELL_VERSION", "DEFAULT_ROV_CONFIG", "PUBLIC_METHODS", "PRIVILEGED_METHODS", "EXPECTED_KIT",
           "HIDDEN_STRATA", "DOME_ENDS", "merge_config", "check_rov_kit", "parse_deviation", "vehicle_kit", "ballast_holes",
           "stage_score", "score_episode", "check_trim_config", "rov_datums"]
