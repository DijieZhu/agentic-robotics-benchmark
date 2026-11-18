"""The information-tier assembly cell `assembly_info/cell_v1` (assembly.md 2.6, 5, 7.1; lane W1-ASM).

PUBLIC CODE (shipped with the practice cell). `InfoCellServer` subclasses the six-servo cell server
(`arm.assembly_six.server.CellServer`): the physics, the predicates, the latch, the welds, the safety and damage watch and
the functional check are the validated six-servo cell's; what changes is what the program may SEE and CALL:

  describe()              robot, control limits, budgets, safety and fastening rules, camera names / resolutions / nominal
                          intrinsics CLASS, the documented cell datums, the loose items present (text); no parts list with
                          poses, no mates, no pre-assembly joint angles, no fixture pose, no tray poses
  get_state()             sim_time, done, robot {q, dq, cmd_q, gripper_width, tcp_pos, tcp_quat, wrench, mode, stiffness_scale};
                          the wrench carries white noise (0.3 N / 0.02 N m) and a per-episode bias (+-1 N / +-0.05 N m); no
                          parts, no mates, no held
  set_control, command    unchanged
  step(n_ticks)           events limited to `safety:<what>` and `episode_over`
  get_camera(name, depth) one frame of a cell camera through the render layer (`shared/render`, MuJoCo backend, one session per
                          episode): PNG rgb (+ 16-bit PNG sensor depth in mm, 0 = invalid); no intrinsics, no pose, no exact
                          depth; at most `camera_budget_frames` frames per episode (any camera), then {"error": "camera_budget"}
  finish()                ends the episode
  get_state_privileged(), describe_privileged()
                          the six-servo cell's get_state / describe (exact part poses, mate status, held, the mating graph):
                          ONLY in privileged mode (a valid oracle token, T11) and in the practice cell; otherwise
                          {"error": "forbidden"}, counted in illegal_call_count

The cell adds to the six-servo cell: the resized workbench (1.6 x 1.0 m top, 0.994 m above a factory floor, white frame,
backdrop walls), the horn-less spare servo and an empty slot in the servo tray, the per-episode proximal posture, the
datum boards (a bench board at a documented pose, a fixture board on the fixture's front face, a hand plate on the Panda
hand), five RGB-D cameras (overview, top, oblique, wrist re-mounted on the hand's flank, inspect looking up) with
per-episode drawn mounts / intrinsics / distortion, lights +-20 %, and the part-force damage cap as a safety rule.
"""

import base64
import copy
import json
import math
import os
import time

import numpy as np

from ..assembly_cell.controller import PandaController, mat_to_quat
from ..assembly_cell.predicates import MateTracker
from ..assembly_six import cell_builder as CB
from ..assembly_six.layout import quat_from_rotmat, rotz
from ..assembly_six.server import CellServer as SixCellServer
from ..assembly_six.server import DEFAULT_SERVER_CONFIG as SIX_DEFAULTS
from ..assembly_cell.server import _deep_update
from ..chain_model import Chain
from ..frames import inv, pose_from_dict
from . import CELL_VERSION, layout_info as LI

__all__ = ["InfoCellServer", "DEFAULT_INFO_CONFIG", "PUBLIC_METHODS", "PRIVILEGED_METHODS", "load_kit", "nominal_proximal", "SPARE", "episode_model"]

SPARE = "SPARE"
RETRACT_Q = [0.0, -1.3, 0.0, -2.9, 0.0, 1.7, 0.785]      # the functional check's retract: folded behind the base (x < -0.13 m)
PUBLIC_METHODS = ("describe", "get_state", "set_control", "command", "step", "get_camera", "finish")
PRIVILEGED_METHODS = ("get_state_privileged", "describe_privileged")
SKU = "BUS-SERVO-12V-A"
SKU_HORNLESS = "BUS-SERVO-12V-A-NH"

DEFAULT_INFO_CONFIG = json.loads(json.dumps(SIX_DEFAULTS))
DEFAULT_INFO_CONFIG.update({
    "sim_cap_s": 600.0,
    "wall_cap_s": 720.0,
    "silence_s": 120.0,
    "camera_budget_frames": 300,
    "info_layout": {},
    "damage": {"pad_force_N": None, "part_force_N": 150.0, "ticks": 20, "report_N": {"pad_force_N": 120.0, "part_force_N": 150.0}},
})
DEFAULT_INFO_CONFIG["cell"] = {
    "table": {"size_xy": [1.6, 1.0], "thickness": 0.03, "floor_z": -0.994},
    "horn_body": {"mass_kg": 0.005, "radius_m": 0.010, "height_m": 0.0025},
    # the unpowered servo's back-drive friction: at the information tier's bus-zero design the fork is mounted up to ~30 deg
    # off the vertical, and with the six-servo cell's 0.1 N m the released forearm-to-gripper half toppled about the free horn
    # (0.1-0.3 N m of gravity torque) into the trays; 0.4 N m holds it (a geared hobby servo back-drives at a few 0.1 N m)
    "horn_hinge": {"frictionloss": 0.4, "damping": 0.05, "armature": 0.005},
}
# T7 physical-capture knob (assembly.md 7.6 "raise the physical capture first"; PLAN.md T7): the elbow pocket's mouth
# can carry a lead-in chamfer (chamfer_mm x 45 deg on the four walls) in the collision model; the design pose, the play
# inside the pocket and the latch tolerance are unchanged. 0 = off (the validated six-servo pocket; the task ships with it off).
DEFAULT_INFO_CONFIG["lead_in"] = {"chamfer_mm": 0.0, "mates": ["seat_S3"]}
# the fingertip pads' contact (see InfoCellServer._pad_contacts): the kit's contact stiffness, torsional friction
DEFAULT_INFO_CONFIG["pads"] = {"solref": [0.002, 1.0], "condim": 4}
DEFAULT_INFO_CONFIG["layout"] = {}
# horn fastening (task version 3, AP-17 / maintainer decision D): the cell fastens a fork wherever it RESTS on its horn (axis
# and axial offset within tolerance, moving < 5 mm/s and turning < 2 deg/s relative to the servo for 50 ticks), at the
# rotation it rests at; the mate counts only when that rotation is within 3 deg of the design (graded completion). A latch
# that fastened only at the design rotation was a physical probe of the hidden zero (a search with a tug test found it
# without the manual: x14 1.0 in version 2). "keyed" = the version-2 rule (fastens only within 3 deg of the design).
DEFAULT_INFO_CONFIG["latch"] = dict(DEFAULT_INFO_CONFIG["latch"], horn_fastening="at_rest", rest_ang_speed_max_deg_s=2.0)
# sensor models (assembly.md 7.1; T7 knobs): the depth units carry frame-constant errors as well (render/README RA1), so
# averaging frames of a static scene does not reach the exact depth
DEFAULT_INFO_CONFIG["photo"] = {
    "noise": {"shot": 0.002, "read": 0.004}, "defocus_px": 0.4, "vignetting": 0.1, "jitter": {"brightness": 0.05, "tint": 0.03},
    "depth_fixed": {"sigma_a_m": 0.0005, "sigma_b_per_m": 0.004, "wiggle_amp_m": 0.001, "fixed_pattern_a_m": 0.0005, "fixed_pattern_b_per_m": 0.0015,
                    "pixel_offset_sigma_m": 0.0004},
    "depth_near": {"sigma_a_m": 0.0002, "sigma_b_per_m": 0.003, "wiggle_amp_m": 0.0002, "fixed_pattern_a_m": 0.0001, "fixed_pattern_b_per_m": 0.0015,
                   "pixel_offset_sigma_m": 0.0001},
    "depth_common": {"min_range_m": 0.1, "max_range_m": 4.0, "quantisation_m": 0.001, "edge_threshold_rel": 0.03, "edge_invalid_px": 1,
                     "edge_ragged_px": 1, "dropout_rate": 0.03, "dropout_blob_px": 0},
    "lod": "full", "shadowsize": 2048, "png_compress_level": 1,
}
# the documented cell datums (cell manual; exact by construction: the integrator's survey)
BENCH_BOARD = {"cols": 9, "rows": 6, "square_mm": 20.0, "pos": [0.335, 0.005, 0.0], "yaw_deg": 0.0}
FIXTURE_BOARD = {"cols": 5, "rows": 4, "square_mm": 10.0, "margin_mm": 4.0, "thickness_mm": 2.0}
HAND_PLATE = {"cols": 5, "rows": 4, "square_mm": 7.0, "margin_mm": 3.0, "thickness_mm": 1.5,
              "pos_in_hand_m": [-0.0335, 0.050, 0.020], "normal_in_hand": [-1.0, 0.0, 0.0], "cols_along_in_hand": [0.0, 1.0, 0.0]}


# ------------------------------------------------------------------------------------------------ kits
def load_kit(kit_dir):
    """(a2_dir, design_path, design dict, manifest, graph) of a kit GT folder (`variant.build_variant` layout)."""
    a2 = os.path.join(kit_dir, "derived", "a2_assembly")
    design_path = os.path.join(kit_dir, "hardware-design", "assembly.json")
    with open(design_path) as f:
        design = json.load(f)
    manifest, graph = CB.load_givens(a2)
    return a2, design_path, design, manifest, graph


def carve_graph(graph):
    """The mating graph with every horn mate's frame_b_in_b at its slide-in pose (the collision carve input)."""
    g = copy.deepcopy(graph)
    for m in g["mates"]:
        if m["kind"] == "horn_mate" and m.get("slide_in"):
            m["frame_b_in_b"] = copy.deepcopy(m["slide_in"]["frame_b_in_b"])
    return g


def _seat_mate(graph, sid="S3"):
    return [m for m in graph["mates"] if m["id"] == "seat_" + sid][0]


def nominal_proximal(design, graph):
    """The nominal proximal posture (cell angles, rad) and the fixture yaw: the elbow pocket's insertion axis straight
    down (mouth up), the upper arm pointing along +y of the cell from the shoulder; the base-yaw joint at its tier zero.
    Returns {"q": {base_yaw, shoulder_pitch}, "fixture_yaw_rad"}."""
    chain = Chain(design)
    ref = {j["name"]: float(j.get("tier_angle_at_zero_rad", 0.0)) for j in design["joints"]}
    m = _seat_mate(graph)
    ins_a = np.asarray(m["insertion_axis_in_a"], float)          # in the upper arm frame
    qb = -ref["base_yaw"]
    best = None
    for qs_deg in np.arange(-180.0, 180.0, 0.25):
        q = np.zeros(len(chain.joints))
        q[0], q[1] = qb, math.radians(qs_deg) - ref["shoulder_pitch"]
        T = chain.fk(q)
        d = T["upper_arm"][:3, :3] @ ins_a
        if best is None or d[2] < best[0]:
            best = (float(d[2]), q[1], T)
    _, qs, T = best
    # refine
    for step in (0.05, 0.01, 0.002):
        for dq in np.arange(-5 * step, 5 * step + 1e-12, step):
            q = np.zeros(len(chain.joints))
            q[0], q[1] = qb, qs + math.radians(dq)
            T2 = chain.fk(q)
            d = T2["upper_arm"][:3, :3] @ ins_a
            if d[2] < best[0]:
                best = (float(d[2]), q[1], T2)
        _, qs, T = best
    j_el = chain.by_name["elbow_pitch"]
    p_sh = T["upper_arm"][:3, 3]
    p_el = (T["upper_arm"] @ j_el.origin)[:3, 3]
    v = p_el - p_sh
    yaw = math.pi / 2 - math.atan2(v[1], v[0])                   # rotate so the upper arm points +y
    return {"q": {"base_yaw": float(qb), "shoulder_pitch": float(qs)}, "fixture_yaw_rad": float((yaw + math.pi) % (2 * math.pi) - math.pi),
            "insertion_axis_down": float(best[0])}


def distal_q(design):
    """The distal compound's internal posture (cell angles): the source's calibrated zero of the wrist and the jaw (the
    shape the six-servo cell's grasp stations were validated on), recorded per joint in the cell design
    (`slide_tier_angle_rad`)."""
    out = {}
    for j in design["joints"]:
        if j["name"] in ("wrist_pitch", "wrist_roll", "jaw"):
            out[j["name"]] = float(j.get("slide_tier_angle_rad", j.get("tier_angle_at_zero_rad", 0.0))) - float(j.get("tier_angle_at_zero_rad", 0.0))
    return out


# ------------------------------------------------------------------------------------------------ the server
class InfoCellServer(SixCellServer):
    """The information-tier cell. `mode` = "hidden" (verify) | "practice" (the practice cell: privileged calls, reset and
    truth allowed); `privileged` = the oracle token matched (verify, T11)."""

    def __init__(self, kit_dir, panda_dir, config=None, scope_ids=None, mode="hidden", privileged=False, render=True, log=None,
                 kit_info=None):
        import mujoco
        self.mujoco = mujoco
        self.kit_dir = kit_dir
        self.a2_dir, self.design_path, self.design, self.manifest, self.graph = load_kit(kit_dir)
        self.panda_dir = panda_dir
        self.cfg = _deep_update(json.loads(json.dumps(DEFAULT_INFO_CONFIG)), config)
        self.parts = {p["name"]: p for p in self.manifest["parts"]}
        s3 = self.parts["S3"]
        self.parts[SPARE] = dict(copy.deepcopy(s3), name=SPARE, catalogue_id=SKU_HORNLESS, mesh_visual=s3["mesh_visual"].replace("servo.stl", "servo_no_horn.stl"))
        self.scope_ids = list(scope_ids or self.graph["scopes"]["elbow"])
        self.mode = mode
        self.allow_reset = mode == "practice"
        self.privileged = bool(privileged or mode == "practice")
        self.render_enabled = bool(render)
        self.log = log or (lambda s: None)
        self.assembly_json = self.design_path
        self.chain = Chain(self.design)
        self.joint_axis_in_child = {j.name: [float(v) for v in j.axis] for j in self.chain.joints}
        self.kit_info = kit_info or {}
        self.prox_nominal = nominal_proximal(self.design, self.graph)
        self.distal = distal_q(self.design)
        self.renderer = None
        self.model = None
        self.data = None
        self.seed = None
        self.done = False
        self.collision_cache = {}
        self.session = None
        self.token_mismatch = False

    # ------------------------------------------------------------------ episode setup
    def _preassembly(self, episode):
        pj = episode["proximal_jitter"]
        q_prox = {"base_yaw": self.prox_nominal["q"]["base_yaw"],
                  "shoulder_pitch": self.prox_nominal["q"]["shoulder_pitch"] + math.radians(pj["shoulder_pitch_deg"])}
        prox = ["base_clamp", "S1", "shoulder", "shoulder_clamp", "S2", "upper_arm"]
        dist = ["wrist_clamp", "S4", "wrist", "S5", "gripper", "S6", "jaw"]
        return {"groups": [{"name": "proximal", "root": "base", "fixed": True, "parts": prox, "q": q_prox},
                           {"name": "distal", "root": "forearm", "fixed": False, "parts": dist, "q": dict(self.distal)}]}

    def _slide_design(self):
        """The design with the elbow's horn mate at its slide-in pose (for the pre-oriented rest of the compound)."""
        d = copy.deepcopy(self.design)
        return d

    def _rests(self, preassembly, loose, fixture_yaw):
        """Rest descriptions of the loose bodies: every servo stands on its connector end (its insertion axis down) with
        its horn axis along the horn axis at the seat; the compound stands fork-down in the orientation it has at the
        SLIDE-IN pose of its horn mate (the six-servo pre-orientation, assembly.md 7.5)."""
        gp = CB.group_poses(self.design, preassembly)
        R_base = np.eye(4)
        R_base[:3, :3] = rotz(float(fixture_yaw))
        fixed_w = {"base": R_base}
        for g in gp.values():
            if g["fixed"]:
                for n, T in g["parts"].items():
                    fixed_w[n] = R_base @ T
        seat = _seat_mate(self.graph)
        horn = [m for m in self.graph["mates"] if m["id"] == "horn_elbow_pitch"][0]
        F_a, F_b = CB.pose_mm(seat["frame_a_in_a"]), CB.pose_mm(seat["frame_b_in_b"])
        T_servo = fixed_w[seat["part_a"]] @ F_a @ inv(F_b)
        Fh_a = CB.pose_mm(horn["frame_a_in_a"])
        Fh_b = CB.pose_mm(horn["slide_in"]["frame_b_in_b"]) if horn.get("slide_in") else CB.pose_mm(horn["frame_b_in_b"])
        T_link = T_servo @ Fh_a @ inv(Fh_b)
        out = {}
        MM = 1e-3
        ins_b = np.asarray(seat.get("insertion_axis_in_b", [-1.0, 0.0, 0.0]), float)
        for n in loose:
            p = self.parts[n]
            if p["kind"] == "servo":
                lo, hi = np.asarray(p["bbox_mm"][0], float) * MM, np.asarray(p["bbox_mm"][1], float) * MM
                corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2] + 0.0025)])
                out[n] = LI.rest_of("servo_on_connector_end", corners, down_part=ins_b, side_part=[0.0, 0.0, 1.0], side_world=T_servo[:3, 2])
                continue
            group = [g for g in gp.values() if g["root"] == n]
            members = {n: np.eye(4)}
            if group:
                members.update(group[0]["parts"])
            pts = []
            for name, T in members.items():
                q = self.parts[name]
                lo, hi = np.asarray(q["bbox_mm"][0], float), np.asarray(q["bbox_mm"][1], float)
                c = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                pts.append((T @ c.T).T[:, :3] * MM)
            P = np.vstack(pts)
            down = T_link[:3, :3].T @ np.array([0.0, 0.0, -1.0])        # world down in the link frame at the slide-in pose
            elbow_axis = np.asarray(self.joint_axis_in_child["elbow_pitch"], float)
            side = elbow_axis - down * float(elbow_axis @ down)
            side /= np.linalg.norm(side)
            rest = LI.rest_of("compound_fork_down", P, down_part=down, side_part=side, side_world=T_link[:3, :3] @ side)
            root_only = LI.rest_of("compound_fork_down", pts[0], down_part=down, side_part=side, side_world=T_link[:3, :3] @ side)
            rest["footprint_half"] = root_only["footprint_half"]
            rest["centre_part"] = root_only["centre_part"]
            out[n] = rest
        return out

    def _lead_in_pieces(self, cfg_cell):
        """The carrier's carved collision pieces with the lead-in chamfer of the configured pocket mouths (pre-filling the
        builder's collision cache: the six-servo builder then uses them as they are)."""
        li = self.cfg.get("lead_in") or {}
        c = float(li.get("chamfer_mm", 0.0) or 0.0)
        if c <= 0.0:
            return
        cfg6 = CB._cfg(cfg_cell)
        graph = carve_graph(self.graph)
        for mid in li.get("mates", []):
            m = [x for x in self.graph["mates"] if x["id"] == mid]
            if not m:
                continue
            m = m[0]
            car = m["part_a"]
            if car in self.collision_cache:
                continue
            part = self.parts[car]
            pieces = CB.carved_collision(self.a2_dir, part, self.parts, graph, cfg6)
            self.collision_cache[car] = chamfer_pocket(pieces, part, self.parts[m["part_b"]], m, self.a2_dir, c, float(cfg6["carve_margin_mm"]))

    def _build(self, layout, preassembly):
        cfg_cell = dict(self.cfg["cell"])
        self._lead_in_pieces(cfg_cell)
        cfg_cell["materials"] = {"printed": list(self.kit_info.get("rgba", [0.93, 0.42, 0.10, 1.0])), "servo": [0.18, 0.18, 0.20, 1.0], "horn": [0.85, 0.72, 0.30, 1.0]}
        # the collision model is carved with every horn mate at its SLIDE-IN pose (the entry sweep of the six-servo cell, whose
        # design pose was the slide-in pose); the predicates, the welds and the functional check use the bus-zero design
        build = CB.build_cell(self.a2_dir, self.panda_dir, layout, cfg_cell, self.scope_ids, self.manifest, carve_graph(self.graph), self.design,
                              preassembly, collision_cache=self.collision_cache)
        scoped = [m for m in self.graph["mates"] if m["id"] in set(self.scope_ids)]
        build.meta["mates"] = CB._mates_metric(scoped, build.meta["parts"])
        build.meta["graph"] = self.graph
        spec, old = build.spec, build.model
        if SPARE in layout["parts"]:
            old_q = {}
            for j in range(old.njnt):
                a = old.jnt_qposadr[j]
                n = {0: 7, 1: 4, 2: 1, 3: 1}[int(old.jnt_type[j])]
                old_q[old.joint(j).name] = old.key_qpos[old.key("home").id][a:a + n].copy()
            old_ctrl = old.key_ctrl[old.key("home").id].copy()
            for k in list(spec.keys):
                spec.delete(k)
            lp = layout["parts"][SPARE]
            body = spec.worldbody.add_body(name=SPARE, pos=lp["pos"], quat=lp["quat"])
            body.add_freejoint(name="free_" + SPARE)
            p = self.parts[SPARE]
            for bname, half, centre in CB.servo_boxes(p, build.meta["config"]["shrink_mm"]):
                body.add_geom(name="%s_%s" % (SPARE, bname), type=self.mujoco.mjtGeom.mjGEOM_BOX, size=half.tolist(), pos=centre.tolist(), group=3, material="mat_hull")
            spec.add_mesh(name="servo_visual_hornless_spare", file=os.path.join(self.a2_dir, p["mesh_visual"]), scale=[1e-3] * 3)
            body.add_geom(name=SPARE + "_visual", type=self.mujoco.mjtGeom.mjGEOM_MESH, meshname="servo_visual_hornless_spare", contype=0, conaffinity=0,
                          group=2, material="mat_servo")
            body.mass = float(p["mass_kg"])
            body.ipos = (np.asarray(p["com_mm"], float) * 1e-3).tolist()
            I = np.asarray(p["inertia_kg_mm2"], float) * 1e-6
            body.explicitinertial = True
            body.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
            model = spec.compile()
            qpos = np.array(model.qpos0, float).copy()
            for j in range(model.njnt):
                name = model.joint(j).name
                a = model.jnt_qposadr[j]
                if name in old_q:
                    qpos[a:a + len(old_q[name])] = old_q[name]
            a = model.jnt_qposadr[model.joint("free_" + SPARE).id]
            qpos[a:a + 3] = lp["pos"]
            qpos[a + 3:a + 7] = lp["quat"]
            spec.add_key(name="home", qpos=qpos.tolist(), ctrl=old_ctrl.tolist())
            model = spec.compile()
            build.model = model
            build.meta["loose"] = list(build.meta["loose"]) + [SPARE]
            build.meta["parts"] = dict(build.meta["parts"])
            build.meta["parts"][SPARE] = self.parts[SPARE]
        self._pad_contacts(build)
        return build

    def _pad_contacts(self, build):
        """The fingertip pads' contact parameters (`cfg["pads"]`, applied to the compiled model): the robot model's pad
        boxes carry MuJoCo's default soft contact (solref 0.02 s, condim 3), under which a servo held at ~30 N crept about
        the approach axis by ~9 % of every hand rotation (13 deg / 8 mm in one transport; measured 2026-09-28); the cell uses
        the kit's own contact stiffness for the pads (a rubber-pad grip does not creep)."""
        pc = self.cfg.get("pads") or {}
        if not pc:
            return
        m = build.model
        for name in build.meta.get("pad_geoms", []):
            g = m.geom(name).id
            if "solref" in pc:
                m.geom_solref[g] = [float(v) for v in pc["solref"]]
            if "solimp" in pc:
                m.geom_solimp[g] = [float(v) for v in pc["solimp"]]
            if "condim" in pc:
                m.geom_condim[g] = int(pc["condim"])
            if "friction" in pc:
                m.geom_friction[g] = [float(v) for v in pc["friction"]]

    def practice_rig(self, seed):
        """A practice kit's nominal camera mounts for `seed` (its `cell_rigs.json`, E-ASM-15); None in the hidden cell (its
        mounts are part of the grader's config)."""
        p = os.path.join(self.kit_dir, "cell_rigs.json")
        if self.mode != "practice" or not os.path.isfile(p):
            return None
        with open(p) as f:
            doc = json.load(f)
        rig = (doc.get("seeds") or {}).get(str(int(seed)))
        if rig is None:
            raise ValueError("the practice kit's cell_rigs.json has no rig for seed %s (practice seeds 0-19)" % seed)
        return {"cameras": rig}

    def reset(self, seed):
        # the hidden cell must have every camera (its rig is the grader's config); any other cell (a practice kit without a rig
        # file, an agent's own twin) may run without cameras: get_camera then answers camera_unavailable
        self.episode = LI.draw_episode(seed, self.cfg.get("info_layout"), rig=self.practice_rig(seed),
                                       strict=(self.mode == "hidden" and self.render_enabled))
        pre = self._preassembly(self.episode)
        root, loose, mates, member_of = CB.scope_parts(self.graph, self.scope_ids, self.manifest, pre)
        fixture_yaw = self.prox_nominal["fixture_yaw_rad"] + math.radians(self.episode["proximal_jitter"]["base_yaw_deg"])
        servo_loose = [n for n in loose if self.parts[n]["kind"] == "servo"]
        rest_names = list(loose) + ([SPARE] if self.cfg["info_layout"].get("spare", True) else [])
        # the trays pre-orient the loose parts for the NOMINAL proximal posture (the jig of the stand / sockets does not know the
        # episode's jitter): the compound stands vertical, the servo upright
        nominal = self._preassembly({"proximal_jitter": {"base_yaw_deg": 0.0, "shoulder_pitch_deg": 0.0}})
        rests = self._rests(nominal, rest_names, self.prox_nominal["fixture_yaw_rad"])
        layout = LI.place_parts(seed, self.cfg["layout"], loose, rests, servo_loose + ([SPARE] if SPARE in rests else []), self.episode)
        layout["base_yaw_rad"] = float(fixture_yaw)
        self.layout = layout
        self.preassembly = pre
        build = self._build(layout, pre)
        self.build = build
        self.model, self.meta = build.model, build.meta
        self.meta["joint_axis_in_child"] = self.joint_axis_in_child
        self._reset_state(seed)
        self._close_session()
        self.frames_used = 0
        self.frames_by_camera = {n: 0 for n in LI.CAMERA_NAMES}
        self.render_wall_s = 0.0
        self.ep_seed = int(seed)
        hi = (int(seed) >> 32) & 0xFFFFFFFF
        self._wrench_rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x3E17C4] + ([hi] if hi else []))
        return {"ok": True}

    def _reset_state(self, seed):
        """The six-servo server's reset body after the build (controller, tracker, geom classes, damage watch)."""
        mujoco = self.mujoco
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.tick_s = self.model.opt.timestep * self.meta["config"]["tick_steps"]
        self.tick_steps = int(self.meta["config"]["tick_steps"])
        self.seed = int(seed)
        self.done = False
        self.done_reason = None
        self.tick_count = 0
        self.events_log = []
        self.violations = []
        self.illegal_calls = 0
        self.calls = {}
        self.step_calls = 0
        self.regrasps = 0
        self.held = None
        self._held_cand = None
        self._held_ticks = 0
        self._drop_watch = {}
        self._lost = set()
        self._dropped = set()
        self.wall_start = None
        self.wall_used = 0.0
        self.renderer = None
        m = self.model
        self.ctrl = PandaController(m, self.data, tick_s=self.tick_s)
        for _ in range(int(self.cfg["settle_s"] / m.opt.timestep)):
            mujoco.mj_step(m, self.data)
        self.data.time = 0.0
        self.ctrl.q_cmd = self.ctrl.q().copy()
        self.ctrl.q_goal = self.ctrl.q_cmd.copy()
        self.ctrl.width_cmd = self.ctrl.width()
        self.ctrl.width_goal = self.ctrl.width_cmd
        self.ctrl.write_ctrl()
        lt = self.cfg["latch"]
        self.tracker = MateTracker(m, self.meta, hold_ticks=lt["hold_ticks"], rel_speed_max=lt["rel_speed_max_m_s"], tick_s=self.tick_s,
                                   tolerance_scale=lt.get("tolerance_scale", 1.0), horn_at_rest=(lt.get("horn_fastening") == "at_rest"),
                                   rest_ang_speed_max=math.radians(float(lt.get("rest_ang_speed_max_deg_s", 2.0))))
        for name in self.meta["preassembled"]:
            self.tracker.body[name] = m.body(name).id
        self._obs_noise = (0.0, 0.0)
        self._obs_rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x0B5E])
        self.geom_class = np.zeros(m.ngeom, dtype=int)
        self.geom_part = {}
        panda_bodies = set(b for b in range(m.nbody) if m.body(b).name.startswith("panda/"))
        loose_bodies = {m.body(n).id: n for n in self.meta["loose"]}
        for s in self.meta["servos"]:
            loose_bodies[m.body("horn_" + s).id] = s
        for gname, g in self.meta["group_poses"].items():
            if not g["fixed"]:
                for mem in g["parts"]:
                    loose_bodies[m.body(mem).id] = g["root"]
        self.loose_body_ids = loose_bodies
        self.static_env_bodies = set([0, m.body("fixture").id, m.body(self.meta["root"]).id])
        for gname, g in self.meta["group_poses"].items():
            if g["fixed"]:
                for mem in g["parts"]:
                    self.static_env_bodies.add(m.body(mem).id)
        pads = set(self.meta["pad_geoms"])
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            name = m.geom(g).name
            if name in pads:
                self.geom_class[g] = 2 if "left" in name else 3
            elif b in panda_bodies:
                self.geom_class[g] = 1
            elif b in loose_bodies:
                self.geom_class[g] = 4
                self.geom_part[g] = loose_bodies[b]
            elif b in self.static_env_bodies:
                self.geom_class[g] = 5
        self.part_body = {n: m.body(n).id for n in self.meta["loose"]}
        self.part_qadr = {n: m.jnt_qposadr[m.joint("free_" + n).id] for n in self.meta["loose"]}
        self.part_vadr = {n: m.jnt_dofadr[m.joint("free_" + n).id] for n in self.meta["loose"]}
        self.sensor_force = m.sensor("hand_force").id
        self.sensor_torque = m.sensor("hand_torque").id
        self._f6 = np.zeros(6)
        self._dmg_peak = {"pad": 0.0, "part": 0.0}
        self._dmg_over = {"pad": 0, "part": 0}
        self._dmg_sustained = {"part": 0.0}
        self._part_hist = []

    # ------------------------------------------------------------------ damage (the six-servo watch + the sustained part force)
    def _watch(self, d):
        """The six-servo watch (30 N non-pad link rule, part speed, held / dropped / lost, pad / part damage peaks and the
        part-force cap as a safety rule over `damage.ticks` consecutive ticks) plus the stat `max_part_force_sustained_N`:
        the largest force a part kept on the environment or another part over a whole cap window (the cap's calibration
        input, assembly.md 7.2)."""
        before = self._dmg_peak["part"]
        self._tick_part_max = 0.0
        events = super()._watch(d)
        ticks = int((self.cfg.get("damage") or {}).get("ticks", 20))
        # recompute this tick's part-on-environment maximum (the parent keeps only the running peak)
        m = self.model
        part_sum = {}
        for i in range(d.ncon):
            con = d.contact[i]
            c1, c2 = self.geom_class[con.geom1], self.geom_class[con.geom2]
            if (c1 == 4 and c2 in (4, 5)) or (c2 == 4 and c1 in (4, 5)):
                self.mujoco.mj_contactForce(m, d, i, self._f6)
                g_part = con.geom1 if c1 == 4 else con.geom2
                g_other = con.geom2 if c1 == 4 else con.geom1
                key = (self.geom_part[g_part], int(m.geom_bodyid[g_other]))
                part_sum[key] = part_sum.get(key, 0.0) + abs(float(self._f6[0]))
        pm = max(part_sum.values()) if part_sum else 0.0
        hist = self._part_hist
        hist.append(pm)
        if len(hist) > ticks:
            del hist[0]
        if len(hist) == ticks:
            self._dmg_sustained["part"] = max(self._dmg_sustained["part"], min(hist))
        return events

    def damage_summary(self):
        out = super().damage_summary()
        out["max_part_force_sustained_N"] = float(self._dmg_sustained["part"])
        return out

    # ------------------------------------------------------------------ functional range in this cell
    def _elbow_upper_clear(self, min_z_m=0.065, step_deg=1.0):
        """The largest elbow angle (cell design angle) up to which the distal compound, mounted at the design and swung about
        the seated elbow servo in THIS episode's proximal posture, keeps every point >= min_z_m above the bench (the trays'
        sockets and the compound's empty stand reach 0.05 m): the functional check's upper bound in the cell (the arm's
        own collision-free box knows no bench)."""
        m, d = self.model, self.data
        T_base = np.eye(4)
        b = m.body(self.meta["root"]).id
        T_base[:3, :3] = d.xmat[b].reshape(3, 3)
        T_base[:3, 3] = d.xpos[b]
        g = [v for v in self.meta["group_poses"].values() if not v["fixed"]][0]
        pts = _compound_points(self)
        gp = CB.group_poses(self.design, self.preassembly)
        members = {g["root"]: np.eye(4)}
        members.update(gp["distal"]["parts"])
        roles = {j.name: 0.0 for j in self.chain.joints}
        for gg in self.preassembly["groups"]:
            roles.update(gg.get("q", {}))
        j_el = self.chain.by_name["elbow_pitch"]
        q = 0.0
        best = 0.0
        while q <= j_el.upper + 1e-9:
            qv = np.array([float(roles.get(j.name, 0.0)) if j.name != "elbow_pitch" else q for j in self.chain.joints])
            T = self.chain.fk(qv)
            Tf = T_base @ _mm_T(T["forearm"])
            low = min(float(np.min(((Tf @ _mm_T(Tm))[:3, :3] @ pts[n].T).T[:, 2] + (Tf @ _mm_T(Tm))[2, 3])) for n, Tm in members.items())
            if low < min_z_m:
                break
            best = q
            q += math.radians(step_deg)
        return best

    # ------------------------------------------------------------------ the restricted API
    def describe(self):
        cams = {}
        for n in LI.CAMERA_NAMES:
            c = self.episode["config"]["cameras"][n]
            cams[n] = {"resolution": list(c["size"]), "rgb": True, "depth": bool(c.get("depth", True)),
                       "nominal_hfov_deg": round(float(c["hfov_deg"])), "mount": _MOUNT_TEXT[n]}
        rng = self.model.jnt_range
        return {
            "cell_version": CELL_VERSION, "units": {"length": "m", "angle": "rad", "quaternion": "wxyz"},
            "task": "elbow stage: seat the elbow servo in the upper arm's pocket, then mount the forearm's fork on its horn in the design orientation",
            "robot": {"model": "7-DoF Panda-class arm (given model: panda/)", "base_pos": list(self.meta["config"]["panda_pos"]), "base_axes": "aligned with the cell frame",
                      "joints": ["panda/joint%d" % (i + 1) for i in range(7)],
                      "joint_range": [[float(rng[j][0]), float(rng[j][1])] for j in self.ctrl.joints],
                      "home_q": list(self.meta["panda_home"]), "tcp_site": "tcp", "tcp_offset_in_hand_m": [0.0, 0.0, self.meta["config"]["tcp_offset_m"]],
                      "gripper": {"max_width_m": self.ctrl.max_width, "rate_m_s": 0.05, "pad_size_m": [0.017, 0.0165],
                                  "force_model": "2500 N/m x (object width - target width) / 2 per finger, capped at 60 N"},
                      "gravity_compensated": True, "ft_sensor": "6-axis at the hand flange (hand frame); noise ~0.3 N / 0.02 N m rms, "
                                                               "an unknown per-episode bias up to 1 N / 0.05 N m; it includes the hand's own weight"},
            "control": {"tick_s": self.tick_s, "joint_rate_limit_rad_s": 1.5, "cartesian_speed_limit_m_s": 0.5, "cartesian_angular_limit_rad_s": 1.5,
                        "stiffness_scale_range": [0.2, 1.0], "modes": ["joint", "cartesian"]},
            "fastening": self._fastening_text(),
            "cameras": cams, "camera_budget_frames": int(self.cfg["camera_budget_frames"]),
            "datums": cell_datums(),
            "items": ["the base-to-upper-arm half of the arm, clamped on the fixture (upper arm roughly horizontal, elbow pocket up; its "
                      "exact posture varies)", "one bus servo WITH horn and one horn-less bus servo standing on their connector ends in "
                      "the servo tray (plus one empty slot)", "the forearm-to-gripper half standing fork-down in a stand on the other tray"],
            "trays": "two trays on the bench at about x 0.08 m, y +-0.32 m (which holds which varies); the fixture near x 0.10 m, y 0 (+-10 mm)",
            "budgets": {"sim_time_cap_s": self.cfg["sim_cap_s"], "wall_clock_cap_s": self.cfg["wall_cap_s"], "silence_abort_s": self.cfg["silence_s"],
                        "max_ticks_per_step": self.cfg["max_ticks_per_step"], "camera_frames": int(self.cfg["camera_budget_frames"])},
            "safety": dict(self.cfg["safety"], part_force_N=(self.cfg.get("damage") or {}).get("part_force_N"),
                           part_force_window_s=0.01 * int((self.cfg.get("damage") or {}).get("ticks", 20))),
        }

    def _fastening_text(self):
        lt = self.cfg["latch"]
        out = {"rule": "a servo seat is fastened by the cell when the servo sits at its design pose within the tolerance for 50 consecutive "
                       "10 ms ticks while it moves slower than 5 mm/s (the part may be held); nothing is reported",
               "seat_tolerance": {"pos_mm": 1.5, "ang_deg": 3.0}, "hold_ticks": 50, "rel_speed_max_m_s": 0.005}
        if lt.get("horn_fastening") == "at_rest":
            out["horn_rule"] = ("a fork is fastened wherever it RESTS on its horn: axis within 3 deg and 1.5 mm, axial offset <= 1 mm, moving "
                                "slower than 5 mm/s and turning slower than %g deg/s relative to the servo for 50 consecutive ticks (it may be "
                                "held); it is bolted at the rotation it rests at and cannot be mounted again; the mate counts only when that "
                                "rotation is within 3 deg of the design (the calibration pose); nothing is reported" % float(lt.get("rest_ang_speed_max_deg_s", 2.0)))
            out["horn_tolerance"] = {"axis_ang_deg": 3.0, "axis_line_mm": 1.5, "axial_mm": 1.0, "counts_within_yaw_deg": 3.0,
                                     "rest_ang_speed_max_deg_s": float(lt.get("rest_ang_speed_max_deg_s", 2.0))}
        else:
            out["horn_rule"] = "as the seat, at the design rotation about the horn axis within 3 deg (keyed: one mounting orientation)"
            out["horn_tolerance"] = {"axis_ang_deg": 3.0, "axis_line_mm": 1.5, "axial_mm": 1.0, "yaw_deg": 3.0, "yaw_modulo_deg": 360.0}
        return out

    def get_state(self):
        d = self.data
        p, R = self.ctrl.tcp_actual()
        adr = self.model.sensor_adr
        w = np.array(list(d.sensordata[adr[self.sensor_force]:adr[self.sensor_force] + 3]) + list(d.sensordata[adr[self.sensor_torque]:adr[self.sensor_torque] + 3]))
        wn = self.episode["config"]["wrench_noise"]
        w = w + np.asarray(self.episode["wrench_bias"], float) + self._wrench_rng.normal(0.0, 1.0, 6) * np.array([wn["sigma_f_N"]] * 3 + [wn["sigma_t_Nm"]] * 3)
        return {"sim_time": float(d.time), "done": bool(self.done),
                "robot": {"q": [float(v) for v in self.ctrl.q()], "dq": [float(v) for v in self.ctrl.dq()], "cmd_q": [float(v) for v in self.ctrl.q_cmd],
                          "gripper_width": self.ctrl.width(), "tcp_pos": [float(v) for v in p], "tcp_quat": [float(v) for v in mat_to_quat(R)],
                          "wrench": [float(v) for v in w], "mode": self.ctrl.mode, "stiffness_scale": self.ctrl.stiffness_scale}}

    def get_state_privileged(self):
        st = SixCellServer.get_state(self)
        return st

    def describe_privileged(self):
        out = SixCellServer.describe(self)
        out["mates"] = [dict(g) for g in self.graph["mates"] if g["id"] in set(self.scope_ids)]
        out["layout"] = _jsonable(self.layout)
        out["preassembly"] = _jsonable(self.preassembly)
        out["cell_config"] = {"cell": self.cfg["cell"], "rgba": self.kit_info.get("rgba")}
        out["horn_fastening"] = str(self.cfg["latch"].get("horn_fastening", "keyed"))
        return out

    def step(self, n_ticks=1):
        r = SixCellServer.step(self, n_ticks)
        if isinstance(r, dict) and "events" in r:
            ev = [e for e in r["events"] if e.startswith("safety:")]
            if r.get("done"):
                ev.append("episode_over")
            r["events"] = ev
        return r

    # ------------------------------------------------------------------ cameras
    def _close_session(self):
        if getattr(self, "session", None) is not None:
            try:
                self.session.close()
            except Exception:  # noqa: BLE001
                pass
        self.session = None

    def _session(self):
        if self.session is None:
            from . import cell_render
            t = time.time()
            self.session = cell_render.open_session(self)
            self.render_wall_s += time.time() - t
        return self.session

    def camera_spec(self, name):
        from render.spec import CameraSpec
        c = self.episode["cameras"][name]
        intr = c["intrinsics"]
        if name == "wrist":
            b = self.model.body("panda/hand").id
            T = np.eye(4)
            T[:3, :3] = self.data.xmat[b].reshape(3, 3)
            T[:3, 3] = self.data.xpos[b]
            W = T @ np.asarray(c["hand_T_cam"], float)
        else:
            W = np.asarray(c["world_T_cam"], float)
        return CameraSpec(name=name, width=intr["width"], height=intr["height"], fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"],
                          world_T_cam=W.tolist(), k1=intr["k1"], k2=intr["k2"], p1=intr["p1"], p2=intr["p2"])

    def photo_spec(self, name):
        from render.spec import BlurSpec, ColourJitterSpec, DepthSensorSpec, NoiseSpec, PhotoSpec, derive_seed
        ph = self.cfg["photo"]
        dk = dict(ph["depth_common"])
        dk.update(ph["depth_near"] if name in ("wrist", "inspect") else ph["depth_fixed"])
        sensor = DepthSensorSpec(unit_seed=derive_seed(self.ep_seed, "depth_unit"), **dk)
        return PhotoSpec(noise=NoiseSpec(**ph["noise"]), blur=BlurSpec(defocus_px=float(ph["defocus_px"])), vignetting=float(ph["vignetting"]),
                         colour_jitter=ColourJitterSpec(**ph["jitter"]), image_format="png", depth_sensor=sensor,
                         png_compress_level=int(ph.get("png_compress_level", 1)))

    def get_camera(self, name="overview", depth=True):
        from render.spec import derive_seed
        name = str(name)
        if name not in LI.CAMERA_NAMES:
            return {"error": "unknown camera (cameras: %s)" % ", ".join(LI.CAMERA_NAMES)}
        if not self.render_enabled or self.episode["cameras"].get(name) is None:
            return {"error": "camera_unavailable"}
        if self.frames_used >= int(self.cfg["camera_budget_frames"]):
            return {"error": "camera_budget"}
        k = self.frames_used
        self.frames_used += 1
        self.frames_by_camera[name] += 1
        t = time.time()
        s = self._session()
        s.sync_from(self.model, self.data)
        want_depth = bool(depth) and bool(self.episode["cameras"][name]["depth"])
        outputs = ("rgb", "sensor_depth") if want_depth else ("rgb",)
        try:
            r = s.render([self.camera_spec(name)], photo=self.photo_spec(name), seed=derive_seed(self.ep_seed, "frame", k), outputs=outputs)
        except Exception as e:  # noqa: BLE001 - a renderer failure never takes the cell down (version 2 defect fix)
            self.log("[cell] %s frame %d not rendered: %s" % (name, k, type(e).__name__))
            self.render_failures = getattr(self, "render_failures", 0) + 1
            self.render_wall_s += time.time() - t
            return {"error": "camera_unavailable"}
        v = r.agent_view()
        out = {"ok": True, "camera": name, "frame_index": int(k), "width": int(v["rgb"][0].shape[1]), "height": int(v["rgb"][0].shape[0]),
               "format": "png", "rgb_png_b64": base64.b64encode(v["encoded"][0]).decode("ascii")}
        if want_depth:
            out["sensor_depth_u16_mm_b64"] = base64.b64encode(v["sensor_depth_encoded"][0]).decode("ascii")
            out["depth_unit_m"] = 0.001
            out["depth_invalid_value"] = 0
        self.render_wall_s += time.time() - t
        return out

    # ------------------------------------------------------------------ dispatch
    def handle(self, method, params):
        self.calls[method] = self.calls.get(method, 0) + 1
        if method == "reset":
            if not self.allow_reset:
                self.illegal_calls += 1
                return {"error": "forbidden"}
            seed = int((params or {}).get("seed", 0))
            self.reset(seed)
            return {"ok": True, "seed": seed}
        if method in PRIVILEGED_METHODS:
            if not self.privileged:
                self.illegal_calls += 1
                return {"error": "forbidden"}
            return getattr(self, method)()
        if method == "truth" and self.mode == "practice":
            return self.truth()
        if method not in PUBLIC_METHODS:
            self.illegal_calls += 1
            return {"error": "forbidden"}
        try:
            return getattr(self, method)(**(params or {}))
        except (TypeError, ValueError, KeyError, IndexError) as e:
            return {"error": "bad_params: %s" % str(e)[:200]}

    # ------------------------------------------------------------------ practice only
    def truth(self):
        """The practice cell's disclosed state: exact part poses and mate status, every camera's exact intrinsics and pose, the
        episode's draws (practice kits carry their own GT)."""
        st = SixCellServer.get_state(self)
        cams = {n: self.camera_spec(n).to_dict() for n in LI.CAMERA_NAMES if self.episode["cameras"].get(n) is not None}
        # the episode's own draws and exact cameras only: never the cell's configuration (its rig constants, spreads)
        ep = {k: v for k, v in self.episode.items() if k != "config"}
        return {"parts": st["parts"], "mates": st["mates"], "held": st["held"], "cameras": cams, "episode": _jsonable(ep),
                "layout": _jsonable(self.layout), "preassembly": _jsonable(self.preassembly)}

    # ------------------------------------------------------------------ end of episode
    def finalize(self, functional_rng, run_functional_check=True):
        """The six-servo summary + functional check, with the Panda retracted to a FOLDED posture behind its base (the
        six-servo check retracts to the home posture, whose hand hangs above the fixture: the compound standing on the
        seated servo swings into it), plus the information-tier stats."""
        self._close_session()
        home = self.meta["panda_home"]
        self.meta["panda_home"] = list(RETRACT_Q)
        j_el = self.chain.by_name["elbow_pitch"]
        upper0 = j_el.upper
        clear = self._elbow_upper_clear()
        j_el.upper = min(upper0, clear)
        try:
            s = SixCellServer.finalize(self, functional_rng, run_functional_check=run_functional_check)
        finally:
            self.meta["panda_home"] = home
            j_el.upper = upper0
        s["functional_elbow_range_rad"] = [float(j_el.lower), float(min(upper0, clear))]
        s["cell_version"] = CELL_VERSION
        s["camera_frames"] = dict(self.frames_by_camera)
        s["camera_frames_total"] = int(self.frames_used)
        s["render_wall_s"] = float(self.render_wall_s)
        s["privileged_mode"] = bool(self.privileged)
        s["episode_draws"] = {"proximal_jitter": self.episode["proximal_jitter"], "horn_flip": self.episode["horn_flip"],
                              "slot_order": self.episode["slot_order"], "fixture_yaw_rad": self.layout["base_yaw_rad"]}
        return s


def _mm_T(T):
    T = np.asarray(T, float).copy()
    T[:3, 3] *= 1e-3
    return T


def chamfer_pocket(pieces, carrier, servo, mate, a2_dir, chamfer_mm, margin_mm):
    """Carve a lead-in chamfer (chamfer_mm x 45 deg) into the mouth of the pocket of `mate` (a sleeve seat): the convex
    frustum over the pocket cross-section, widening by chamfer_mm from chamfer_mm below the mouth to the mouth (and on
    beyond it), is removed from every collision piece by half-space clipping. The mouth is where the carrier's material
    begins along the insertion path (rays along the insertion axis through the pocket outline). mm, carrier frame."""
    import trimesh
    from ..assembly_cell.cell_builder import _clip_convex
    from ..frames import inv, pose_from_dict
    F_a = pose_from_dict(mate["frame_a_in_a"])
    F_b = pose_from_dict(mate["frame_b_in_b"])
    T_s = F_a @ inv(F_b)                                            # the servo at its seat, in the carrier frame (mm)
    a = np.asarray(mate["insertion_axis_in_a"], float)
    a /= np.linalg.norm(a)
    body = [pr for pr in servo.get("collision_primitives", []) if pr["name"] == "body"][0]
    lo, hi = np.asarray(body["min"], float), np.asarray(body["max"], float)
    ctr = (lo + hi) / 2.0
    a_s = T_s[:3, :3].T @ a                                         # the insertion direction in the servo frame
    k = int(np.argmax(np.abs(a_s)))                                 # the case axis along the insertion
    others = [i for i in range(3) if i != k]
    # the mouth: rays along +a through the outline of the pocket cross-section, started well behind the trailing face
    mesh = trimesh.load(os.path.join(a2_dir, carrier["mesh_visual"]), force="mesh")
    trail = ctr.copy()
    trail[k] = lo[k] if a_s[k] > 0 else hi[k]                        # the trailing face (it enters last)
    origins = []
    for s1 in (-1.0, 1.0):
        for s2 in np.linspace(-0.8, 0.8, 5):
            for swap in (False, True):
                p = trail.copy()
                i, j = (others if not swap else others[::-1])
                p[i] = ctr[i] + s1 * ((hi[i] - lo[i]) / 2.0 + margin_mm + 0.6)
                p[j] = ctr[j] + s2 * (hi[j] - lo[j]) / 2.0
                origins.append(T_s[:3, :3] @ p + T_s[:3, 3] - a * 40.0)
    origins = np.asarray(origins)
    locs, idx, _tri = mesh.ray.intersects_location(origins, np.tile(a, (len(origins), 1)), multiple_hits=False)
    if len(locs) == 0:
        return pieces
    u0 = ((T_s[:3, :3] @ trail + T_s[:3, 3]) @ a)                   # the trailing face's depth at the seat
    u_m = float(np.median(locs @ a))                                # the mouth's depth along a
    # the frustum in the servo frame: along k from the mouth - 3 mm (outside) to the mouth + chamfer (inside)
    sgn = 1.0 if a_s[k] > 0 else -1.0
    def to_servo_u(u):
        return (u - u0)                                             # depth measured from the trailing face along a
    planes = []                                                     # (normal, origin) in the carrier frame; keep n.(x - o) >= 0
    e_k = T_s[:3, :3][:, k] * sgn                                   # = a
    base_c = T_s[:3, :3] @ ctr + T_s[:3, 3]
    planes.append((a, base_c + a * ((u_m + chamfer_mm) - (base_c @ a))))                 # deeper than the chamfer: keep
    planes.append((-a, base_c + a * ((u_m - 1.0) - (base_c @ a))))                      # above the mouth region: keep
    for i in others:
        e = T_s[:3, :3][:, i]
        half = (hi[i] - lo[i]) / 2.0 + margin_mm
        for sg in (-1.0, 1.0):
            # the flared wall: |(x - c).e| <= half + (u_m + chamfer - u)  ->  outward normal (sg e + a) / sqrt2
            n = (sg * e + a) / math.sqrt(2.0)
            o = base_c + sg * e * half + a * ((u_m + chamfer_mm) - (base_c @ a))
            planes.append((n, o))
    out = []
    for h in pieces:
        rest = h
        for n, o in planes:
            if rest is None:
                break
            outside = _clip_convex(rest, n, o)
            if outside is not None and outside.volume > 0.5:
                out.append(outside)
            rest = _clip_convex(rest, -n, o)
        # `rest` (inside the frustum) is the chamfer's material: dropped
    return out


def episode_model(kit_dir, panda_dir, layout, preassembly, config=None, rgba=None):
    """The physics model of an episode for a known layout / pre-assembly (the privileged anchor's private collision model)."""
    srv = InfoCellServer(kit_dir, panda_dir, config=config, mode="practice", render=False, kit_info={"rgba": rgba} if rgba else None)
    srv.layout, srv.preassembly = layout, preassembly
    return srv._build(layout, preassembly).model


def _compound_points(server):
    """Surface samples (m, each member's own frame) of the distal compound's printed parts and servos (cached per server)."""
    pts = getattr(server, "_cmp_pts", None)
    if pts is None:
        import trimesh
        pts = {}
        g = [v for v in server.meta["group_poses"].values() if not v["fixed"]][0]
        for name in [g["root"]] + list(g["parts"]):
            p = server.parts[name]
            m = trimesh.load(os.path.join(server.a2_dir, p["mesh_visual"]), force="mesh")
            v = np.asarray(m.vertices, float) * 1e-3
            pts[name] = v[:: max(1, len(v) // 1500)]
        server._cmp_pts = pts
    return pts


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


_MOUNT_TEXT = {
    "overview": "room camera on a tripod beyond the bench's +x / +y corner, roughly 1-1.5 m from the fixture, looking down at the bench",
    "top": "overhead, roughly 1 m above the fixture, looking down (image width roughly along the cell's y)",
    "oblique": "on a post beyond the bench's +x / -y corner, roughly 1 m from the fixture, looking down at it",
    "wrist": "on the Panda hand's +x flank, a few centimetres off the flange axis, tilted towards the fingertips",
    "inspect": "on the bench between the fixture and the robot's -y tray, looking up (for checking a part held above it)",
}


def cell_datums():
    """The documented datums (the cell manual's survey table; exact, public). Corner formulas follow the render layer's
    checkerboard (render/scale_refs.py): inner corner (i, j) of a board of C x R squares of side s at
    ((i - C / 2) s, (R / 2 - j) s) in the board frame, i = 1 .. C - 1, j = 1 .. R - 1; the square at the board's (-x, +y) corner
    is black."""
    b, f, h = BENCH_BOARD, FIXTURE_BOARD, HAND_PLATE
    return {
        "frame": "cell frame: origin at the bench-top centre, z up (the bench top is z = 0, 0.994 m above the floor); the robot base at "
                 "(-0.45, 0, 0) with its axes aligned with the cell frame",
        "bench_board": {"squares": [b["cols"], b["rows"]], "square_mm": b["square_mm"], "margin_mm": b["square_mm"], "thickness_mm": 3.0,
                        "inner_corner_cell_m": "x = 0.335 + (i - 4.5) x 0.020, y = 0.005 + (3 - j) x 0.020, z = 0.00305 (i = 1..8, j = 1..5)",
                        "black_corner_square": "at the board's -x, +y corner", "lies_flat_on_bench": True},
        "fixture_board": {"squares": [f["cols"], f["rows"]], "square_mm": f["square_mm"], "margin_mm": f["margin_mm"], "thickness_mm": f["thickness_mm"],
                          "mounting": "vertical on the fixture plate's +x face, facing +x, its lower edge on the bench, centred on the plate",
                          "inner_corner_fixture_m": "relative to the fixture plate's centre on the bench (fx, fy, 0): x = fx + 0.07255, "
                                                    "y = fy + (i - 2.5) x 0.010, z = 0.024 + (2 - j) x 0.010 (i = 1..4, j = 1..3)",
                          "black_corner_square": "the top square at the -y end",
                          "fixture": "the fixture plate (0.14 x 0.14 x 0.02 m) sits at a nominal centre (0.10, 0.00) m, +-10 mm, not rotated"},
        "hand_plate": {"squares": [h["cols"], h["rows"]], "square_mm": h["square_mm"], "margin_mm": h["margin_mm"], "thickness_mm": h["thickness_mm"],
                       "mounting": "flat on the Panda hand's -x face (hand frame: z = approach along the fingers, y = the closing direction), "
                                   "facing -x",
                       "inner_corner_hand_m": "x = -0.03505, y = 0.050 + (i - 2.5) x 0.007, z = 0.020 + (j - 2) x 0.007 (i = 1..4, j = 1..3)",
                       "black_corner_square": "at the plate's -y, -z corner", "mounting_tolerance": "about 1 mm / 0.5 deg"},
        "fixture": {"nominal_centre_m": [0.10, 0.0], "tolerance_mm": 10.0, "plate_m": [0.14, 0.14, 0.02]},
    }
