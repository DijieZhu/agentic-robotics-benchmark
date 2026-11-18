"""The race-car assembly cell behind the camera-only API (`CarInfoServer`).

Physics: `cell.CarCell` (the rolling chassis clamped on the work stand, the deck assembly on its shelf, the compute
module and the LiDAR in the parts tray, the four wheels and the stick pack on the bench, the UR with its 2F-85 driven by
`ur_controller.URController`, the coupling state machine of `coupling.py`, and the cell's own watch: scrapping and the
two hard safety rules, run inside every control tick). The observation side is the information tier's
(`arm.assembly_info.cell_info.InfoCellServer`, inherited for get_state / set_control / command / step / get_camera /
finish and the dispatch): two uncalibrated RGB-D cameras (top, wrist) through the render layer (`agent_render.py`:
Filament in the factory hall; `render_twin.py`: MuJoCo's classic renderer), a noisy flange F/T sensor, the joint
state; no unit poses, no hub angles, no fastening or scrapping events (`step` reports only `safety:<what>` and
`episode_over`).

`CarInfoServer.__init__` calls no parent constructor (the classes above it load the six-servo arm's kit); the inherited
methods read the attributes set here and in `reset` / `_init_state`.

Per episode (`reset(seed, stratum)`): the information tier's draws (`layout_info.draw_episode`: the cameras'
intrinsics and mounts, the lights, the F/T bias, the tool plate's mount jitter) and the physical layout
(`layout.episode_layout(seed, plan, kit, uv, cfg["layout"], stratum)`: the stand and the chassis clamped on it, the
shelf with the deck, the tray with its two units, the five bench units around the hidden bench plan, the hubs' start
angles); `layout.layout_digest` of it is returned (the grader compares it with the fixed hidden set's). The layout is
a pure function of its inputs: there is no reach check and no redraw (the hidden set's episodes are selected so that
the reference program completes them). With the oracle token (privileged mode) `reference_assemble()` runs the
reference program (`reference_shop.reference_assemble(cell, plan=..., log=...)`, imported on first use) inside the
cell, through the same routed ticks and the same watch; `oracle_nominal_layout` (default false) gives the privileged
mode the plan's nominal layout instead.

Score of an episode (`finalize`): stage_score = the counted slots' weight / the scored weight (`coupling.record()`:
fastened in a counting orientation, neither its unit nor a loose unit on its carrier chain scrapped; premounted slots
leave both sums; 11.5 with none); hard_ok = no robot_contact (a robot link other than the four pads above
safety.robot_force_N on one body) and no part_force (a unit pressed above damage.part_force_N for damage.ticks
consecutive ticks) violation (verify.py: an episode that is not hard_ok keeps scoring.safety_violation_factor of its
stage score, which describe() states). grader_config `scope.premounted` and `scoring.count_subassembly` are the coupling's
`premounted` and `count_subassembly`; `safety` / `damage` / `scrap` are the cell's watch thresholds.
"""

import copy
import json
import math
import time
import traceback

import numpy as np

from ..assembly_cell.server import _deep_update
from ..assembly_info import layout_info as LI
from ..assembly_info.cell_info import DEFAULT_INFO_CONFIG, InfoCellServer
from . import fixtures as FX
from . import layout as LY
from . import render_twin as RT
from .cell import CELL_CONFIG, CarCell, check_kit, load_kit

CELL_VERSION = "racecar-assembly-cell/1.0.0"
PUBLIC_METHODS = ("describe", "get_state", "set_control", "command", "step", "get_camera", "finish")
PRIVILEGED_METHODS = ("get_state_privileged", "describe_privileged", "reference_assemble")
# what this cell is built for (_check_kit): 8 slots of total weight 11.5, 8 loose units, 6 keyed slots, 4 hubs
EXPECTED_KIT = {"slots": 8, "weight": 11.5, "loose": 8, "keyed": 6, "rotors": 4}
# the given kit.json's unit ids per kit unit type
GIVEN_UNIT_IDS = {"WHL": "wheel", "BAT": "battery_pack", "DCK": "deck_assembly", "CMP": "compute_module", "LDR": "lidar"}
# describe()["fastening"]["places"]: (what it mounts on, its home) in words, per slot (the wheels' four by their corner)
PLACE_WORDS = {
    "battery": ("the chassis", "lying on the ribs of the battery bay (battery_bay.png)"),
    "deck": ("the chassis", "resting on the tops of the four deck standoffs (assembly_layout.png)"),
    "compute": ("the deck plate", "standing with its eight standoffs on the plate's top face (assembly_layout.png)"),
    "lidar": ("the deck plate", "standing with its base on the plate's top face (assembly_layout.png)"),
    "wheel": ("that corner's hub", "the floor of the rim's hex pocket against the outer face of the wheel hex (hub_and_wheel.png)"),
}
CORNER_WORDS = {"FL": "front left", "FR": "front right", "RL": "rear left", "RR": "rear right"}

# ------------------------------------------------------------------ the agent-visible strings (describe() shows them)
TASK_TEXT = ("build the car on its work stand as the picture manual shows: the stick pack into the battery bay, the deck assembly "
             "onto the four deck standoffs, the compute module and the LiDAR onto the deck plate, and a wheel onto every hub")
ITEMS_TEXT = (
    "the rolling chassis clamped on the work stand, the point midway between its axles at about ({stand_x}, {stand_y}) m "
    "(+-{stand_xy} mm, turned by up to +-{stand_yaw} deg), its front towards the robot arm, its wheel axles {axle} m above the "
    "bench; its four wheel hubs are unpowered, turn freely and start at random angles",
    "the deck assembly on the deck shelf, top face up, the middle of its plate at about ({shelf_x}, {shelf_y}) m "
    "(+-{shelf_xy} mm, turned by up to +-{shelf_yaw} deg)",
    "the compute module and the LiDAR in the two nests of the parts tray, the tray's middle at about ({tray_x}, {tray_y}) m "
    "(+-{tray_xy} mm, turned by up to +-{tray_yaw} deg); their storage orientation is not the mounting orientation",
    "four wheels and the stick pack loose on the bench within the robot's reach: each wheel on its inner face, the stick pack flat "
    "on its underside with its lead end either way; their places and their turns about the vertical vary between episodes, the "
    "face each rests on does not; the wheels are interchangeable",
)
FASTENING_SUMMARY = ("a unit inside the capture window of a place it can take is captured: held there and pulled home at up to "
                     "{pull_mm_s} mm/s and {pull_deg_s} deg/s, and fastened once it has rested at home within the lock tolerance for "
                     "{hold_ticks} consecutive ticks; a captured unit moved beyond the release window is released; nothing is unfastened; "
                     "no event reports any of this")
FASTENING_RULES = (
    "the stick pack, the deck assembly and the compute module are captured one way round only",
    "the LiDAR is captured and fastened in four orientations {lidar_pitch} deg apart; one of them counts, the other three are "
    "fastened for good and count nothing",
    "a wheel is captured in each of the six orientations of the wheel hex, and all of them count; any wheel fits any hub",
    "a wheel that is not captured rests on its stop {stop_mm} mm short of home; the stop is switched off while the wheel is captured",
    "the compute module and the LiDAR mount on the deck plate wherever the deck assembly is, and count there unless the deck "
    "assembly is scrapped",
    "no place needs another one fastened first",
)
MOUNT_TEXT = {
    "top": "overhead, roughly {top_h} m above the bench near the middle of the working area, looking straight down; image width "
           "roughly along the cell's x",
    "wrist": "on the gripper's coupling, about {wrist_cm} cm off the tool axis on the tool's +x side, tilted about {wrist_tilt} deg "
             "towards the fingertips; the fingers and a held unit are in the lower part of the image",
}
SAFETY_RULES = (
    "a robot link other than the gripper's four pad boxes pressing on the chassis, a fixture, the bench or a unit above "
    "robot_force_N: a safety violation",
    "a unit pressed on the environment or on another unit above part_force_N for part_force_window_s (the normal forces summed "
    "per unit and body it presses on; the gripper's pads on a unit do not count): a safety violation",
    "a safety violation does not end the episode, but an episode with one keeps only {keep_pct} % of its score "
    "(violation_score_factor), however many violations it has",
    "a loose unit whose centre of mass is faster than part_speed_m_s, or that falls off the bench, is scrapped: it is released, "
    "can no longer be fastened and no longer counts; the episode goes on; a scrapped deck assembly takes the compute module and "
    "the LiDAR fastened on it with it",
)
FIXTURE_BOARD_NOTE = {"stand": "its place on the stand and the chassis's place relative to it are in stand_and_shelf.png",
                      "shelf": "its place on the shelf and the deck's place relative to it are in stand_and_shelf.png",
                      "tray": "its place on the tray and the nests' places relative to it are in tray.png"}

DEFAULT_CAR_CONFIG = {
    "sim_cap_s": 650.0,
    "wall_cap_s": 3100.0,
    "privileged_wall_cap_s": 3100.0,
    "silence_s": 120.0,
    "max_ticks_per_step": 500,
    "camera_budget_frames": 450,
    "settle_s": 1.0,
    # the cell's watch (cell.py "watch"): the two hard rules and the scrap rule; the robot-contact limit is the cell's own
    # default (grader_config.json safety.robot_force_N is the same number)
    "safety": {"robot_force_N": CELL_CONFIG["watch"]["robot_force_N"], "part_speed_m_s": 3.0},
    "damage": {"part_force_N": 150.0, "ticks": 20},
    "scrap": {"below_bench_m": 0.05},
    "photo": json.loads(json.dumps(DEFAULT_INFO_CONFIG["photo"])),
    "info_layout": json.loads(json.dumps(RT.INFO_LAYOUT)),
    "agent_cameras": ["top", "wrist"],
    # the agent's cameras: "filament" (agent_render.py: physically based, the baked factory hall) or "mujoco" (render_twin.py)
    "agent_renderer": "filament",
    "robot": {"model": "ur10e", "pos": [0.0, 0.0, 0.0], "yaw_deg": 0.0},
    "layout": json.loads(json.dumps(LY.DEFAULT_LAYOUT)),
    # the gates' fallback (slots the cell fastens before the episode; they leave the scored weight) and the sub-assembly rule (count_subassembly)
    "scope": {"premounted": []},
    "scoring": {"count_subassembly": True},
    "oracle_nominal_layout": False,
}


def _f(v, decimals):
    """a number as the documents print it (placeholders.json: exactly `decimals` digits after the point)"""
    s = format(float(v), ".%df" % int(decimals))
    return "0" if s in ("-0", "-0.0") and decimals == 0 else s


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
    return x


def _check_kit(kit):
    """the kit is the one this cell is built for (8 slots of total weight 11.5, 8 loose units, 6 keyed slots with their
    key heights, 4 hubs on hinges), else ValueError"""
    slots = kit["slots"]
    if len(slots) != EXPECTED_KIT["slots"]:
        raise ValueError("kit: %d slots, the cell expects %d" % (len(slots), EXPECTED_KIT["slots"]))
    w = sum(float(s["weight"]) for s in slots)
    if abs(w - EXPECTED_KIT["weight"]) > 1e-9:
        raise ValueError("kit: total slot weight %.3f, the cell expects %.1f" % (w, EXPECTED_KIT["weight"]))
    if len(kit["loose"]) != EXPECTED_KIT["loose"]:
        raise ValueError("kit: %d loose units, the cell expects %d" % (len(kit["loose"]), EXPECTED_KIT["loose"]))
    keyed = [s["id"] for s in slots if s["kind"] == "keyed"]
    keys = (kit.get("carve") or {}).get("keys") or {}
    if len(keyed) != EXPECTED_KIT["keyed"] or any(k not in keys for k in keyed):
        raise ValueError("kit: %d keyed slots with %d key heights (carve.keys), the cell expects %d" % (
            len(keyed), sum(1 for k in keyed if k in keys), EXPECTED_KIT["keyed"]))
    rotors = [bn for bn, bb in kit["bodies"].items() if isinstance(bb.get("joint"), dict)]
    if len(rotors) != EXPECTED_KIT["rotors"]:
        raise ValueError("kit: %d hubs on hinges, the cell expects %d" % (len(rotors), EXPECTED_KIT["rotors"]))
    missing = [r for r in rotors if r not in (kit.get("hinges") or {})]
    if missing:
        raise ValueError("kit: hub(s) %s without a kit['hinges'] entry" % ", ".join(missing))


class CarInfoServer(InfoCellServer):
    """`kit_dir` = the cell's hidden kit (tests/gt: kit.json + meshes/, unpacked at verify start; any kit directory in the
    racecar-asm-kit/1 format), `robot_dir` = the given robot models (universal_robots_ur10e/, robotiq_2f85/), `plan` = the
    hidden bench plan (tests/hidden/bench_plan.json: {"rest": {unit: ...}, "oracle": ...}). `kit` / `uv`: the loaded kit
    and its `layout.unit_geometry` when the caller has them (one load for several episodes)."""

    def __init__(self, kit_dir, robot_dir, plan, config=None, mode="hidden", privileged=False, render=True, log=None, kit=None,
                 uv=None):
        import mujoco
        if mode != "hidden":
            raise ValueError("this cell has no practice mode")
        self.mujoco = mujoco
        self.cfg = _deep_update(json.loads(json.dumps(DEFAULT_CAR_CONFIG)), config or {})
        self.kit = kit if kit is not None else load_kit(kit_dir)
        check_kit(self.kit)
        _check_kit(self.kit)
        self.kit_dir = self.kit["dir"]
        self.robot_dir = robot_dir
        self.plan = plan
        self.uv = uv if uv is not None else LY.unit_geometry(self.kit)
        self.mode, self.allow_reset = mode, False
        self.privileged, self.render_enabled = bool(privileged), bool(render)
        self.log = log or (lambda s: None)
        self.kit_info = {}
        self.camera_names = tuple(self.cfg.get("agent_cameras") or ("top", "wrist"))
        self.model = self.data = self.seed = self.cell = None
        self.layout = self.layout_digest = None
        self.done = False
        self.session = None
        self.renderer = None
        self.token_mismatch = False
        self.calls = {}
        self.illegal_calls = 0
        self.reference_report = None

    # ------------------------------------------------------------------ episode setup
    def cell_config(self):
        """the CarCell config of this server's grader configuration: the robot, the watch, the coupling"""
        c = self.cfg
        return {"robot": dict(c["robot"], models_dir=self.robot_dir),
                "robot_home_q": list(CELL_CONFIG["robot_home_q"]),
                "watch": {"enabled": True, "robot_force_N": float(c["safety"]["robot_force_N"]),
                          "part_force_N": float(c["damage"]["part_force_N"]), "part_force_ticks": int(c["damage"]["ticks"]),
                          "scrap_speed_m_s": float(c["safety"]["part_speed_m_s"]),
                          "scrap_below_bench_m": float(c["scrap"]["below_bench_m"]), "scrap": True},
                "coupling": dict(CELL_CONFIG["coupling"], count_subassembly=bool(c["scoring"]["count_subassembly"]),
                                 premounted=list(c["scope"]["premounted"] or []))}

    def reset(self, seed, stratum=None):
        """build the episode's cell; returns {"ok", "layout_digest", "stratum", "nominal"}"""
        self.episode = LI.draw_episode(int(seed), self.cfg.get("info_layout"), strict=False)
        missing = [n for n in self.camera_names if self.episode["cameras"].get(n) is None]
        if missing and self.render_enabled:
            raise LI.MissingMount("camera(s) %s have no mount in the cell's config" % ", ".join(missing))
        nominal = bool(self.privileged and self.cfg.get("oracle_nominal_layout", False))
        t0 = time.time()
        lay = LY.episode_layout(int(seed), self.plan, self.kit, uv=self.uv, cfg=self.cfg["layout"], stratum=stratum, nominal=nominal)
        self.layout = lay
        self.layout_digest = LY.layout_digest(lay)
        cell = self._build_cell(lay)
        self.log("[cell] race-car cell for seed %d built in %.1f s (stratum %s%s)" % (
            int(seed) & 0xFFFFFFFF, time.time() - t0, lay["stratum"], ", nominal" if nominal else ""))
        cell.recording = False
        self.cell = cell
        self.model, self.data = cell.m, cell.d
        self.ctrl, self.sm = cell.ctrl, cell.sm
        self.tracker = None
        self._init_state(seed)
        self._close_session()
        self.frames_used = 0
        self.frames_by_camera = {n: 0 for n in self.camera_names}
        self.render_wall_s = 0.0
        self.ep_seed = int(seed)
        hi = (int(seed) >> 32) & 0xFFFFFFFF
        self._wrench_rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x3E17C4] + ([hi] if hi else []))
        # every control tick goes through the server (its event log, the sim cap), whoever drives it: an agent's step() or
        # the reference program; the cell's own tick runs the coupling and the watch
        self._raw_tick = cell.tick_one

        def routed():
            self.physics_tick(track=True)
        cell.tick_one = routed
        return {"ok": True, "layout_digest": self.layout_digest, "stratum": lay["stratum"], "nominal": nominal}

    def _build_cell(self, lay):
        hook = RT.visual_spec_hook if any(bb.get("visual") for bb in self.kit["bodies"].values()) else None
        return CarCell(self.kit_dir, lay, config=self.cell_config(), log=self.log, spec_hook=hook,
                       settle_s=float(self.cfg["settle_s"]), kit=self.kit)

    def _init_state(self, seed):
        m = self.model
        cell = self.cell
        self.seed = int(seed)
        self.done = False
        self.done_reason = None
        self.tick_count = 0
        self.events_log = []
        self.illegal_calls = 0
        self.calls = {}
        self.step_calls = 0
        self.wall_start = None
        self.wall_used = 0.0
        self.renderer = None
        self.reference_report = None
        self.tick_s = cell.tick_s
        self.tick_steps = cell.tick_steps
        self._obs_noise = (0.0, 0.0)
        self._obs_rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x0B5E])
        self.part_body = {n: m.body(n).id for n in cell.loose}
        self.sensor_force = m.sensor(cell.robot["force_sensor"]).id
        self.sensor_torque = m.sensor(cell.robot["torque_sensor"]).id
        self.wrist_parent = cell.robot["gripper_prefix"] + "base"
        self.meta = {"loose": list(cell.loose), "pad_geoms": list(cell.robot["pad_geoms"]), "root": "chassis"}

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
        if self.data.time >= self.cfg["sim_cap_s"] - 1e-9:
            self._end("sim_cap")
        return events

    # ------------------------------------------------------------------ the restricted API
    def _values(self):
        """the numbers the agent-visible strings print (placeholders.json's sources and decimals)"""
        c, L = self.cfg, self.cfg["layout"]
        ct = {e["slot"]: e for e in self.sm.capture_table()}
        cams = c["info_layout"]["cameras"]
        wh = [e for s, e in ct.items() if e.get("stop")]
        return {
            "stand_x": _f(L["stand"]["nominal_xy"][0], 2), "stand_y": _f(L["stand"]["nominal_xy"][1], 2),
            "stand_xy": _f(1e3 * float(L["stand"]["jitter_m"]), 0), "stand_yaw": _f(L["stand"]["jitter_yaw_deg"], 0),
            "axle": _f(L["stand"]["axle_height_m"], 2),
            "shelf_x": _f(L["shelf"]["nominal_xy"][0], 2), "shelf_y": _f(L["shelf"]["nominal_xy"][1], 2),
            "shelf_xy": _f(1e3 * float(L["shelf"]["jitter_m"]), 0), "shelf_yaw": _f(L["shelf"]["jitter_yaw_deg"], 0),
            "tray_x": _f(L["tray"]["nominal_xy"][0], 2), "tray_y": _f(L["tray"]["nominal_xy"][1], 2),
            "tray_xy": _f(1e3 * float(L["tray"]["jitter_m"]), 0), "tray_yaw": _f(L["tray"]["jitter_yaw_deg"], 0),
            "pull_mm_s": _f(ct["battery"]["pull"]["speed_mm_s"], 0), "pull_deg_s": _f(ct["battery"]["pull"]["rate_deg_s"], 0),
            "hold_ticks": _f(ct["battery"]["lock"]["hold_ticks"], 0),
            "lidar_pitch": _f(ct["lidar"]["orientations"]["pitch_deg"], 0),
            "stop_mm": _f(wh[0]["stop"]["stop_from_home_mm"], 2) if wh else "-",
            "top_h": _f(cams["top"]["eye"][2], 1), "wrist_cm": _f(100.0 * float(cams["wrist"]["offset_m"]), 0),
            "wrist_tilt": _f(cams["wrist"]["tilt_deg"], 0),
        }

    def _places(self):
        """describe()["fastening"]["places"]: one entry per scored place from the coupling's capture table (no frame, no pose,
        no direction as a vector)"""
        out = []
        units = self.kit["units"]
        for e in self.sm.capture_table():
            if e["premounted"]:
                continue
            sid = e["slot"]
            slot = self.sm.by_id[sid]
            utype = units[slot["cands"][0]["unit"]]["type"]
            key = "wheel" if sid.startswith("whl_") else sid
            mounts, home = PLACE_WORDS[key]
            if key == "wheel":
                mounts = "the %s hub (%s)" % (CORNER_WORDS.get(sid.split("_", 1)[1], sid), mounts)
            cap, rel, lock, pull = e["capture"], e["release"], e["lock"], e["pull"]
            p = {"place": sid, "unit": GIVEN_UNIT_IDS.get(utype, utype), "any_of_identical_units": len(slot["cands"]) > 1,
                 "mounts_on": mounts, "home": home,
                 "capture": {"across_mm": _r(cap["lateral_mm"]), "short_of_home_mm": _r(cap["from_home_mm"]),
                             "past_home_mm": _r(cap["past_home_mm"]), "tilt_deg": _r(cap["tilt_deg"])},
                 "release": {"across_mm": _r(rel["lateral_mm"]), "short_of_home_mm": _r(rel["from_home_mm"]),
                             "tilt_deg": _r(rel["tilt_deg"])},
                 "lock": {"pos_mm": _r(lock["pos_mm"]), "ang_deg": _r(lock["ang_deg"]), "hold_ticks": int(lock["hold_ticks"]),
                          "rest_speed_mm_s": _r(lock["rel_speed_mm_s"]), "rest_turn_deg_s": _r(lock["rel_ang_speed_deg_s"])},
                 "pull": {"speed_mm_s": _r(pull["speed_mm_s"]), "rate_deg_s": _r(pull["rate_deg_s"])},
                 "weight": float(e["weight"])}
            if e["kind"] == "seat":
                p["capture"]["tilt_is"] = "the total rotation from the home pose about any axis (tilt and turn in one)"
                p["pull"]["starts_below_mm_s"] = _r(pull["starts_below_mm_s"])
                p["pull"]["starts_below_deg_s"] = _r(pull["starts_below_deg_s"])
                p["orientations"] = {"captured": 1, "counting": 1}
            else:
                p["capture"]["turn_deg"] = _r(cap["clock_deg"])
                p["capture"]["inserted_fraction"] = _r(cap["key_frac"])
                p["release"]["turn_deg"] = _r(rel["clock_deg"])
                p["lock"]["along_mm"] = _r(lock["axial_mm"])
                p["key_height_mm"] = _r(e["key_height_mm"])
                o = e["orientations"]
                n = max(1, int(round(360.0 / float(o["pitch_deg"]))))
                p["orientations"] = {"captured": n, "counting": n if o["any_key"] else 1}
                if n > 1:
                    p["orientations"]["apart_deg"] = _r(o["pitch_deg"])
            if e.get("stop"):
                st = e["stop"]
                p["stop"] = {"short_of_home_mm": _r(st["stop_from_home_mm"]), "seat_depth_mm": _r(st["seat_depth_mm"]),
                             "rule": "a wheel that is not captured rests on its stop; the stop is switched off while the wheel is "
                                     "captured and back on only once the wheel is clear of every hub"}
            out.append(p)
        return out

    def _fastening(self):
        v = self._values()
        places = self._places()
        return {"summary": FASTENING_SUMMARY.format(**v), "places": places,
                "rules": [r.format(lidar_pitch=v["lidar_pitch"], stop_mm=v["stop_mm"]) for r in FASTENING_RULES],
                "weights": {p["place"]: p["weight"] for p in places}, "total_weight": float(self.sm.total_w),
                "premounted": list(self.sm.premounted)}

    def describe(self):
        m = self.model
        v = self._values()
        cams = {}
        for n in self.camera_names:
            c = self.episode["config"]["cameras"][n]
            cams[n] = {"resolution": list(c["size"]), "rgb": True, "depth": bool(c.get("depth", True)),
                       "nominal_hfov_deg": round(float(c["hfov_deg"])), "mount": MOUNT_TEXT.get(n, "").format(**v)}
        rb, ct = self.cell.robot, self.ctrl
        from ..assembly_quad.gripper_2f85 import FORCE_MODEL_TEXT
        dmg = self.cfg["damage"]
        keep = float(self.cfg["scoring"]["safety_violation_factor"])      # the grader's factor (verify.safety_factor)
        return {
            "cell_version": CELL_VERSION, "units": {"length": "m", "angle": "rad", "quaternion": "wxyz"},
            "task": TASK_TEXT,
            "robot": {"model": "6-DoF UR10e-class arm (given model: robot/universal_robots_ur10e/) with a 2F-85-class gripper "
                               "(robot/robotiq_2f85/)",
                      "base_pos": [float(x) for x in self.cfg["robot"]["pos"]],
                      "base_axes": "the arm's base frame is the cell frame turned 180 deg about z: q = 0 on the first joint points the arm "
                                   "along the cell's -x",
                      "joints": list(rb["joints"]),
                      "joint_range": [[float(m.jnt_range[m.joint(j).id][0]), float(m.jnt_range[m.joint(j).id][1])] for j in rb["joints"]],
                      "home_q": [float(x) for x in ct.q_cmd],
                      "torque_limits_Nm": [float(x) for x in rb["torque_limits_Nm"]],
                      "tcp_site": rb["tcp_site"], "tcp_offset_from_flange_m": [0.0, 0.0, round(float(rb["tcp_from_flange_m"]), 4)],
                      "tcp_axes": "z = the approach (the flange axis, out of the flange), y = the closing axis, x across the pads",
                      "gripper": {"max_width_m": float(ct.max_width), "rate_m_s": 0.10,
                                  "width": "the gap between the pad faces at the pads' middle",
                                  "pad_size_m": [0.022, 0.0375], "force_model": FORCE_MODEL_TEXT,
                                  "kinematics": "parallel pinch only: the finger linkage is rigid; along the approach the pads' face centre "
                                                "lies about 14 mm short of the tcp fully open and 1-4 mm short for 10-45 mm grasps, the "
                                                "fingertips 5-18 mm beyond it, depending on the width; the finger linkage spans +-62-74 mm "
                                                "along the closing axis"},
                      "wrist_housing": "a 36 x 24 x 35 mm camera housing on the gripper's coupling, on the tool's +x side (part of the robot)",
                      "gravity_compensated": True,
                      "ft_sensor": ("6-axis at the tool flange's centre in the flange frame (z along the tool axis outwards, x along the "
                                    "closing axis = the tcp's y, y along the tcp's -x); what the wrist applies to the tool, the gripper's "
                                    "own weight included (about 10.3 N with the tool pointing down): tare it; noise ~0.3 N / 0.02 N m rms "
                                    "per reading, an unknown bias up to 1 N / 0.05 N m per episode")},
            "control": {"tick_s": self.tick_s, "joint_rate_limit_rad_s": float(min(1.5, min(rb["speed_limits_rad_s"]))),
                        "cartesian_speed_limit_m_s": 0.5, "cartesian_angular_limit_rad_s": 1.5, "stiffness_scale_range": [0.2, 1.0],
                        "modes": ["joint", "cartesian"]},
            "fastening": self._fastening(),
            "cameras": cams, "camera_budget_frames": int(self.cfg["camera_budget_frames"]),
            "datums": car_datums(self.cfg),
            "items": [t.format(**v) for t in ITEMS_TEXT],
            "budgets": {"sim_time_cap_s": float(self.cfg["sim_cap_s"]), "wall_clock_cap_s": float(self.cfg["wall_cap_s"]),
                        "silence_abort_s": float(self.cfg["silence_s"]), "max_ticks_per_step": int(self.cfg["max_ticks_per_step"]),
                        "camera_frames": int(self.cfg["camera_budget_frames"])},
            "safety": {"robot_force_N": float(self.cfg["safety"]["robot_force_N"]), "part_speed_m_s": float(self.cfg["safety"]["part_speed_m_s"]),
                       "part_force_N": float(dmg["part_force_N"]), "part_force_window_s": _r(self.tick_s * int(dmg["ticks"])),
                       "off_bench_m": float(self.cfg["scrap"]["below_bench_m"]), "violation_score_factor": keep,
                       "rules": [t.format(keep_pct=format(100.0 * keep, ".0f")) for t in SAFETY_RULES]},
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

    def camera_spec(self, name):
        return RT.camera_spec(self, name)

    def photo_spec(self, name):
        return RT.photo_spec(self, name)

    def get_camera(self, name="top", depth=True):
        if str(name) not in self.camera_names:
            return {"error": "unknown camera (cameras: %s)" % ", ".join(self.camera_names)}
        return super().get_camera(name, depth=depth)

    # ------------------------------------------------------------------ privileged (oracle token only)
    def describe_privileged(self):
        out = {"layout": self.layout, "layout_digest": self.layout_digest, "slots": self.kit["slots"],
               "capture_table": self.sm.capture_table(), "states": dict(self.sm.state), "score": self.sm.score()}
        return json.loads(json.dumps(_jsonable(out), default=str))

    def get_state_privileged(self):
        d = self.data
        parts = {n: {"pos": [float(x) for x in d.xpos[b]], "quat": [float(x) for x in d.xquat[b]]} for n, b in self.part_body.items()}
        hubs = {hb: float(self.cell.hub_deg(hb)) for hb in self.cell.hinge_of}
        return {"sim_time": float(d.time), "parts": parts, "hub_deg": hubs, "states": dict(self.sm.state),
                "records": _jsonable(self.sm.record()), "scrapped": _jsonable(self.sm.scrapped), "score": float(self.sm.score())}

    def reference_assemble(self):
        """The privileged reference program on this episode (`reference_shop.reference_assemble`, imported here on first use),
        through the cell's routed ticks and its watch."""
        t0, s0 = time.time(), float(self.data.time)
        try:
            from . import reference_shop
        except ImportError as e:
            self.log("[cell] the reference program is not available: %r" % (e,))
            return {"ok": False, "error": "reference_unavailable: %s" % type(e).__name__}
        entry = getattr(reference_shop, "reference_assemble", None) or getattr(reference_shop, "program", None)
        if entry is None:
            return {"ok": False, "error": "reference_unavailable: no entry point"}
        try:
            rep = entry(self.cell, plan=self.plan, log=self.log) or {}
        except Exception as e:  # noqa: BLE001 - the episode is still finalized with what was fastened
            self.log("[cell] the reference program failed:\n%s" % traceback.format_exc())
            return {"ok": False, "error": "reference_failed: %s" % type(e).__name__, "score": float(self.sm.score())}
        self.reference_report = rep
        return {"ok": True, "score": float(self.sm.score()), "states": dict(self.sm.state),
                "steps": int(len(rep.get("steps") or [])), "unplanned_contacts": int(len(rep.get("unplanned_contacts") or [])),
                "sim_s": float(self.data.time) - s0, "wall_s": float(time.time() - t0)}

    def handle(self, method, params):
        if method in PRIVILEGED_METHODS:
            self.calls[method] = self.calls.get(method, 0) + 1
            if not self.privileged:
                self.illegal_calls += 1
                return {"error": "forbidden"}
            return getattr(self, method)()
        return super().handle(method, params)

    # ------------------------------------------------------------------ end of episode
    def finalize(self, functional_rng=None, run_functional_check=False):
        self._close_session()
        if not self.done:
            self._end("solve_returned")
        sm, cell = self.sm, self.cell
        rec = sm.record()
        weights = {r["id"]: float(r["weight"]) for r in rec}
        W = sum(weights.values())
        achieved_w = sum(weights[r["id"]] for r in rec if r["achieved"])
        ws = cell.watch_summary()
        hard = [v for v in ws["violations"] if v["what"].split(":")[0] in ("robot_contact", "part_force")]
        d = self.data
        return {
            "cell_version": CELL_VERSION, "seed": self.seed, "stratum": self.layout["stratum"], "layout_digest": self.layout_digest,
            "nominal_layout": bool(self.layout.get("nominal")), "done_reason": self.done_reason,
            "sim_time_s": float(d.time), "wall_time_s": float(self.wall_used), "ticks": int(self.tick_count),
            "mates": _jsonable(rec), "states": dict(sm.state), "weights": weights, "weight_total": W,
            "stage_score": achieved_w / W if W else 0.0, "weight_counted": achieved_w,
            "n_achieved": int(sum(1 for r in rec if r["achieved"])), "n_fastened": int(sum(1 for r in rec if r.get("fastened"))),
            "all_mates": bool(rec) and all(r["achieved"] for r in rec),
            "locked_off": [r["id"] for r in rec if r.get("fastened") and r.get("design_ok") is False],
            "coupling_metrics": _jsonable(sm.metrics()), "premounted": list(sm.premounted),
            "scrapped": _jsonable(sm.scrapped),
            "violations": _jsonable(ws["violations"]), "hard_ok": not hard, "safety_ok": not hard,
            "robot_contact_violations": int(sum(1 for v in hard if v["what"].startswith("robot_contact"))),
            "part_force_violations": int(sum(1 for v in hard if v["what"].startswith("part_force"))),
            "watch": _jsonable({k: v for k, v in ws.items() if k != "violations"}),
            "illegal_call_count": int(self.illegal_calls), "calls": dict(self.calls), "step_calls": int(self.step_calls),
            "events": [[t, e] for t, e in self.events_log][-400:], "coupling_events": [list(e) for e in sm.events][-400:],
            "hub_deg": {hb: float(cell.hub_deg(hb)) for hb in cell.hinge_of},
            "layout": _jsonable({k: self.layout.get(k) for k in ("stratum", "stand", "clamp", "fixtures", "plays", "hub_deg",
                                                                  "lead_end_deg", "jitter")}),
            "camera_frames": dict(self.frames_by_camera), "camera_frames_total": int(self.frames_used),
            "render_wall_s": float(self.render_wall_s), "renderer": self.renderer, "privileged_mode": bool(self.privileged),
            "reference": None if self.reference_report is None else _jsonable(
                {k: self.reference_report.get(k) for k in ("ok", "steps", "unplanned_contacts") if k in self.reference_report}),
        }


def car_datums(cfg):
    """The documented datums (the cell manual's survey table; exact, public): the five boards and the nominal places of the
    stand, the shelf and the tray. Inner corner (i, j) of a board of C x R squares of side s at ((i - C / 2) s, (R / 2 - j) s)
    in the board frame (render/scale_refs.py)."""
    b, t = FX.BENCH_BOARD, FX.TOOL_PLATE
    bench = CELL_CONFIG["bench"]
    cx, cy = (float(x) for x in bench["centre_xy"])
    hx, hy = float(bench["size_xy"][0]) / 2, float(bench["size_xy"][1]) / 2
    face = (b["thickness_mm"] + FX.PRINT_ABOVE_PLATE_MM) * 1e-3
    s = b["square_mm"] * 1e-3
    out = {
        "frame": "cell frame: origin on the bench top at the robot arm's base axis, z up (the bench top is z = 0, %.3f m above the "
                 "floor), x towards the work stand; the bench top spans x %.2f..%.2f m, y %.2f..%.2f m" % (
                     -float(bench["floor_z"]), cx - hx, cx + hx, cy - hy, cy + hy),
        "bench_board": {"squares": [b["cols"], b["rows"]], "square_mm": b["square_mm"], "margin_mm": b["margin_mm"],
                        "thickness_mm": b["thickness_mm"],
                        "inner_corner_cell_m": "x = %.3f + (i - %.1f) x %.3f, y = %.3f + (%.1f - j) x %.3f, z = %.5f (i = 1..%d, j = 1..%d)" % (
                            b["pos"][0], b["cols"] / 2.0, s, b["pos"][1], b["rows"] / 2.0, s, face, b["cols"] - 1, b["rows"] - 1),
                        "black_corner_square": "at the board's -x, +y corner", "lies_flat_on_bench": True},
        "tool_plate": {"squares": [t["cols"], t["rows"]], "square_mm": t["square_mm"], "margin_mm": t["margin_mm"],
                       "thickness_mm": t["thickness_mm"],
                       "mounting": "on the camera housing's outer face (the tool's +x side), facing +x",
                       "mounting_tolerance": "about 1 mm / 0.5 deg"},
    }
    for key, bd in FX.boards().items():
        out[key + "_board"] = {"squares": [bd["cols"], bd["rows"]], "square_mm": bd["square_mm"], "margin_mm": bd["margin_mm"],
                               "thickness_mm": bd["thickness_mm"], "plate_mm": [_r(x, 3) for x in bd["plate_mm"]],
                               "printed_face_above_bench_mm": _r(bd["face_height_mm"], 3), "mounting": bd["where"] + ", flat, printed face up",
                               "moves_with": FX.FIXTURES[bd["fixture"]]["label"], "note": FIXTURE_BOARD_NOTE[key]}
    L = LY._merge(cfg.get("layout"))
    st, sh, tr = L["stand"], L["shelf"], L["tray"]
    out["work_stand"] = {"nominal_axle_midpoint_m": [float(x) for x in st["nominal_xy"]], "axle_height_m": float(st["axle_height_m"]),
                         "tolerance_mm": _r(1e3 * float(st["jitter_m"]), 3), "turn_up_to_deg": float(st["jitter_yaw_deg"]),
                         "facing": "the car's front towards the robot arm (the car's forward axis along the cell's -x at the nominal placement)"}
    out["deck_shelf"] = {"nominal_deck_middle_m": [float(x) for x in sh["nominal_xy"]], "tolerance_mm": _r(1e3 * float(sh["jitter_m"]), 3),
                         "turn_up_to_deg": float(sh["jitter_yaw_deg"]),
                         "facing": "the deck the same way round as on the car (parallel to its place on the car at the nominal placements)"}
    out["parts_tray"] = {"nominal_middle_m": [float(x) for x in tr["nominal_xy"]], "tolerance_mm": _r(1e3 * float(tr["jitter_m"]), 3),
                         "turn_up_to_deg": float(tr["jitter_yaw_deg"])}
    return out


__all__ = ["CarInfoServer", "CELL_VERSION", "DEFAULT_CAR_CONFIG", "PUBLIC_METHODS", "PRIVILEGED_METHODS", "EXPECTED_KIT",
           "GIVEN_UNIT_IDS", "car_datums"]
