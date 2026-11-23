"""Every number the agent's documents state, computed from the cell itself.

``collect(cfg_path, measured_path)`` returns a flat dict {key: text}. Its sources:

* the grader's cell configuration (``tests/grader_config.json`` over the package defaults);
* a cell built from that configuration on an empty shift: ``describe()`` (the cameras' drawing poses
  and data-sheet optics, station hardware, joint ranges, torque limits, the tool centre point), and
  the start posture's clearance above the belt (rays cast up from the belt's
  footprint against the robot's collision shapes);
* the shift generator's configuration (the population's ranges, the fixed shift durations; nothing
  is drawn from the hidden or the development candidates);
* ``make_models.py`` (the data sheet's friction ranges) and ``nominal_dynamics.json`` (only its
  tolerances: the factors themselves never reach a document);
* ``measured.json`` beside this file: values only a measurement gives (the gripper's sheet,
  what each camera resolves, the sandbox). A missing measurement renders as a visible marker
  "[TO BE MEASURED: ...]", which the publication check refuses.

The templates (``templates/*.md``) contain no number of their own except the literals that
``LITERALS`` in ``make_docs.py`` lists with a reason; ``test_docs.py`` holds the binding.
"""

from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.path.dirname(HERE)
TESTS = os.path.dirname(REVIEW)
if TESTS not in sys.path:
    sys.path.insert(0, TESTS)
sys.dont_write_bytecode = True

MEASURED = os.path.join(HERE, "measured.json")
MARK_MEASURE = "[TO BE MEASURED: %s]"
MARK_LOOK = "[TO BE WRITTEN ON THE FINAL LOOK: %s]"

# Runner and commissioning numbers that live outside the cell's configuration (the
# packaging's verify / commissioning service). Taken from tests/grader_config.json when it
# carries them, else these defaults (the grader's runner defaults).
RUNNER_DEFAULTS = {
    "shifts_graded": 8,
    "cpu_cap_factor": 4.0,
    "import_check_wall_s": 120.0,
    "import_check_cpu_s": 120.0,
    "route_gate": {"max_submission_mb": 60,
                   "forbidden_extensions": [".pt", ".pth", ".onnx", ".safetensors", ".ckpt", ".h5", ".pb",
                                            ".pkl", ".npz"],
                   "forbidden_calls": ["torch.optim", ".backward(", "jax.grad"]},
    "blas_threads": 2,
    "session_hours": 4,
    "sandbox": {"cpus": 8, "memory_gb": 16, "gpus": 0},
    "min_W_per_shift": 4,
    "python": "3.12",
    "harness_input": "25-30k",
    "harness_output": "64k",
    "session_margin_min": 15,
    "commissioning": {"development_shifts": 8, "bench_shifts": 3, "full_shifts": 5,
                      "shifts_per_session": 12, "open_at_a_time": 2},
}


def g(x, nd=4):
    """A number as the documents write it: up to ``nd`` decimals, no trailing zeros."""
    if isinstance(x, bool):
        raise TypeError("a boolean is not a number")
    if isinstance(x, int):
        return str(x)
    x = float(x)
    if abs(x - round(x)) < 1e-12 and abs(x) < 1e12:
        return str(int(round(x)))
    s = ("%." + str(nd) + "f") % x
    s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def mm(x_m):
    return g(round(float(x_m) * 1000.0, 1), 1)


def pct(x):
    return g(round(float(x) * 100.0, 2), 2)


def deg(rad):
    return g(round(math.degrees(float(rad)), 1), 1)


def _runner(raw):
    """RUNNER_DEFAULTS, overridden by tests/grader_config.json's runner keys and by task.toml (the agent's session
    time and the sandbox's CPUs and memory) when the task folder has one."""
    out = json.loads(json.dumps(RUNNER_DEFAULTS))
    for k in out:
        if k in raw:
            if isinstance(out[k], dict) and isinstance(raw[k], dict):
                out[k].update(raw[k])
            else:
                out[k] = raw[k]
    toml_path = os.path.join(os.path.dirname(TESTS), "task.toml")
    if os.path.exists(toml_path):
        import tomllib
        with open(toml_path, "rb") as fh:
            t = tomllib.load(fh)
        env, agent = t.get("environment", {}), t.get("agent", {})
        if "timeout_sec" in agent:
            out["session_hours"] = float(agent["timeout_sec"]) / 3600.0
        if "cpus" in env:
            out["sandbox"]["cpus"] = int(env["cpus"])
        if "memory_mb" in env:
            out["sandbox"]["memory_gb"] = float(env["memory_mb"]) / 1024.0
        if "gpus" in env:
            out["sandbox"]["gpus"] = int(env["gpus"])
    return out


def _empty_cell(cfg):
    """A cell on an empty shift of the line setting (the station's job: its describe() is the agent's)."""
    from parcelcell.server import ParcelCellServer
    from parcelcell.shiftgen import line_setting
    from parcelcell.stream import Shift
    srv = ParcelCellServer(cfg, render=False)
    info = srv.reset(Shift(belt_speed=((0.0, 0.2),), arrivals=(), duration=float(cfg["line"]["shifts"]["duration_s"]),
                           seed=0, setting=line_setting(cfg, False)))
    return srv, info


def _clearance(world):
    """Lowest point of the robot (collision shapes) above the belt's footprint at the
    start posture: rays cast straight up from the belt surface on a 5 mm grid."""
    import mujoco
    import numpy as np
    m, d = world.model, world.data
    st = world.station
    grp = np.zeros(6, np.uint8)
    grp[3] = 1
    gid = np.zeros(1, np.int32)
    robot = set(int(x) for x in world.robot_geoms)
    best = 1e9
    for x in np.arange(st.belt_center_x - st.half_width, st.belt_center_x + st.half_width + 1e-9, 0.005):
        for y in np.arange(-st.visible_half_len, st.visible_half_len + 1e-9, 0.005):
            dist = mujoco.mj_ray(m, d, np.array([x, y, st.belt_height + 0.0005]), np.array([0.0, 0.0, 1.0]),
                                 grp, 1, -1, gid)
            if dist >= 0 and int(gid[0]) in robot:
                best = min(best, dist + 0.0005)
    return best


def _robot_extent_x(world):
    """The largest x of any point of the robot's moving collision shapes at the start
    posture (mesh vertices and box corners), and the lowest z of the hands."""
    import mujoco
    import numpy as np
    m, d = world.model, world.data
    xmax, zmin = -1e9, 1e9
    for g in world.robot_moving_geoms:
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        else:
            sz = m.geom_size[g]
            v = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * sz[:3]
        w = (R @ np.asarray(v, float).T).T + p
        xmax = max(xmax, float(w[:, 0].max()))
        zmin = min(zmin, float(w[:, 2].min()))
    return xmax, zmin


def _mirror_rule(world, joints):
    """Which joints take the same angle and which the opposite one in a mirror-symmetric posture (the right arm's
    axis against the left's mirrored as a rotation axis), checked by forward kinematics on random postures."""
    import mujoco
    import numpy as np
    m = world.model
    signs = []
    for jl, jr in zip(joints[:7], joints[7:]):
        a = np.asarray(m.jnt_axis[m.joint(jl).id], float)
        b = np.asarray(m.jnt_axis[m.joint(jr).id], float)
        mir = np.array([-a[0], a[1], -a[2]])
        signs.append(1 if np.allclose(b, mir) else (-1 if np.allclose(b, -mir) else 0))
    if 0 in signs:
        raise RuntimeError("the right arm's axes are not a mirror image of the left's")
    d = mujoco.MjData(m)
    qa = np.array([m.jnt_qposadr[m.joint(n).id] for n in joints])
    rng = np.random.default_rng(0)
    for _ in range(10):
        ql = rng.uniform(-0.3, 0.3, 7)
        ql[3] = rng.uniform(0.2, 1.5)
        d.qpos[:] = m.qpos0
        d.qpos[qa[:7]] = ql
        d.qpos[qa[7:]] = np.asarray(signs) * ql
        mujoco.mj_kinematics(m, d)
        pl = d.site_xpos[m.site("Left_tcp").id]
        pr = d.site_xpos[m.site("Right_tcp").id]
        if np.linalg.norm(pl - pr * np.array([1.0, -1.0, 1.0])) > 1e-9:
            raise RuntimeError("the mirror rule does not hold on the model")
    same = [joints[i].replace("Left_", "").replace("_Joint", "") for i, sg in enumerate(signs) if sg > 0]
    opp = [joints[i].replace("Left_", "").replace("_Joint", "") for i, sg in enumerate(signs) if sg < 0]
    return ("a mirror-symmetric posture has the same angle at %s and the opposite angle at %s (the start posture is "
            "such a pair)" % (", ".join("`%s`" % x for x in same), ", ".join("`%s`" % x for x in opp)))


def _md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _mat(M, nd=4):
    return "[" + ", ".join("[" + ", ".join(g(v, nd) for v in row) + "]" for row in M) + "]"


def collect(cfg_path=None, measured_path=None):
    from parcelcell.config import load_config
    from parcelcell import robot as R
    from parcelcell import stream as S
    from parcelcell import _geom
    from parcelcell.world import AFTER_OVER_STEPS
    from parcelcell.station import FRAME_SOLIDS
    import numpy as np

    cfg_path = cfg_path or os.path.join(TESTS, "grader_config.json")
    with open(cfg_path) as fh:
        raw = json.load(fh)
    cfg = load_config(cfg_path)
    run = _runner(raw)
    measured = {}
    mp = measured_path or MEASURED
    if os.path.exists(mp):
        with open(mp) as fh:
            measured = json.load(fh)

    srv, info = _empty_cell(cfg)
    world = srv.world
    m = world.model
    F = {}
    w, st_cfg = cfg["world"], cfg["world"]["station"]
    stn = info["station"]
    belt = stn["belt"]
    top, cx, width, L = belt["top_z"], belt["centre_x"], belt["width"], belt["frame_half_length"]
    W = belt["work_window_half_length"]
    hw = stn["hardware"]

    # ---------------------------------------------------------------- station
    F.update(belt_top=g(top), belt_cx=g(cx), belt_w=g(width), belt_x0=g(cx - width / 2), belt_x1=g(cx + width / 2),
             frame_L=g(L), frame_len=g(2 * L), window_W=g(W), window_len=g(2 * W),
             belt_friction=g(st_cfg["belt_friction"], 2))
    # the data sheet's friction ranges (the true values lie inside; the robot model carries the pads' middle)
    from make_models import BELT_FRICTION_RANGE, PAD_FRICTION_RANGE, nominal_factors
    F.update(belt_fr_lo=g(BELT_FRICTION_RANGE[0], 2), belt_fr_hi=g(BELT_FRICTION_RANGE[1], 2),
             pad_fr_lo=g(PAD_FRICTION_RANGE[0], 2), pad_fr_hi=g(PAD_FRICTION_RANGE[1], 2))
    _nf = nominal_factors()
    F["mass_tol_pct"] = g(100 * float(_nf["mass_tolerance"]))
    F["joint_tol_pct"] = g(100 * float(_nf["joint_tolerance"]))
    _sn = cfg["sensors"]
    F["tau_noise_pct"] = g(100 * float(_sn["tau_noise_rel"]), 2)
    F["tau_offset_pct"] = g(100 * float(_sn["tau_offset_rel"]), 2)
    _it = cfg["cameras"]["installation_tolerance"]
    for _k in ("infeed", "head", "wrist"):
        F["tol_%s_mm" % _k] = g(1000 * float(_it[_k]["position_m"]), 1)
        _deg = float(_it[_k]["angle_deg"])
        F["tol_%s_deg" % _k] = g(_deg, 2) + (" degree" if _deg == 1.0 else " degrees")
    F["tol_focal_pct"] = g(100 * float(_it["focal_rel"]), 2)
    sh = hw["belt_shell_col"]
    F["shell_gap_mm"] = mm(top - (sh["centre"][2] + sh["half_size"][2]))
    F["shell_over_mm"] = mm(sh["half_size"][0] - width / 2)
    F["shell_depth"] = g(2 * sh["half_size"][2], 3)
    legs = [hw[n] for n in FRAME_SOLIDS if n.startswith("belt_leg_col_") and not n.endswith("_far")]
    F["leg_ys"] = ", ".join(g(x["centre"][1], 3) for x in legs)
    F["leg_n"] = str(len(legs))
    F["leg_wx"] = g(2 * legs[0]["half_size"][0], 3)
    F["leg_ty"] = g(2 * legs[0]["half_size"][1], 3)
    F["leg_gap"] = g(round(top - (legs[0]["centre"][2] + legs[0]["half_size"][2]), 3), 3)
    far0 = hw.get("belt_leg_col_0_far")
    F["leg_posts_text"] = ("two square posts (%s m wide, their centres %s m apart across the belt) with a cap plate "
                           "under the frame and a brace between them" % (F["leg_wx"], g(round(far0["centre"][0] - legs[0]["centre"][0], 3), 3))
                           if far0 is not None else "one block")
    rf, rnu, lip = hw["belt_rail_far"], hw["belt_rail_near_up"], hw["belt_lip_near_up"]
    lip_dn = hw.get("belt_lip_near_down")
    F["rail_far_x"] = g(rf["centre"][0], 3)
    F["rail_far_above"] = g(rf["centre"][2] + rf["half_size"][2] - top, 3)
    rnd = hw["belt_rail_near_down"]
    F["rail_near_x"] = g(rnu["centre"][0], 3)
    F["rail_near_above"] = g(rnu["centre"][2] + rnu["half_size"][2] - top, 3)
    F["guide_up_above"] = g(rnu["centre"][2] + rnu["half_size"][2] - top, 3)
    F["guide_up_above_mm"] = mm(rnu["centre"][2] + rnu["half_size"][2] - top)
    F["guide_up_y0"] = g(rnu["centre"][1] - rnu["half_size"][1], 3)
    F["rail_down_above"] = g(rnd["centre"][2] + rnd["half_size"][2] - top, 3)
    F["guide_inner_x"] = g(rnu["centre"][0] + rnu["half_size"][0], 3)
    F["lip_above_mm"] = mm(lip["centre"][2] + lip["half_size"][2] - top)
    F["lip_y0"] = g(lip["centre"][1] - lip["half_size"][1], 3)
    F["lip_y1"] = g((lip_dn or lip)["centre"][1] + (lip_dn or lip)["half_size"][1], 3)
    F["lip_extent"] = ("through the whole work window" if lip_dn is not None else "in the work window's upstream half")
    F["lip_x"] = g(lip["centre"][0], 3)
    rows = [["belt surface (moving)", "[%s, y, %s]" % (g(cx), g(top - 0.02)), "[%s, runs on, 0.02]" % g(width / 2),
             "the moving belt (contact with it: the belt rule)"]]
    names = {"belt_rail_far": "far rail", "belt_rail_near_up": "near-side guide, upstream of the window",
             "belt_rail_near_down": "near rail, downstream of the window",
             "belt_lip_near_up": "near lip, upstream half of the window",
             "belt_lip_near_down": "near lip, downstream half of the window",
             "belt_shell_col": "frame (shell)", "infeed_camera_body": "infeed camera housing",
             "infeed_camera_bracket": "infeed camera mounting plate",
             "belt_drive_motor_cover": "belt drive motor guard (under the belt)",
             "belt_drive_motor_terminal": "belt drive motor terminal box",
             "infeed_camera_rod": "infeed camera drop rod", "belt_drive_unit": "belt drive unit (under the belt)",
             "belt_drive_motor": "belt drive motor (under the belt)",
             "head_camera_housing": "head camera housing (on the robot's head)"}
    for i in range(3):
        names["belt_leg_col_%d" % i] = "leg %d, near post" % (i + 1)
        names["belt_leg_col_%d_far" % i] = "leg %d, far post" % (i + 1)
        names["belt_leg_cap_%d" % i] = "leg %d, cap plate" % (i + 1)
        names["belt_leg_brace_%d" % i] = "leg %d, brace" % (i + 1)
    for n in hw:
        if n not in names:                               # further hardware (camera posts and their brackets, ...)
            names[n] = n.replace("_", " ")
    for n in names:
        if n in hw:
            b = hw[n]
            cls = "the robot's own head (arm to arm)" if n == "head_camera_housing" else "station structure"
            rows.append([names[n] + " `%s`" % n, "[" + ", ".join(g(v, 3) for v in b["centre"]) + "]",
                         "[" + ", ".join(g(v, 3) for v in b["half_size"]) + "]", cls])
    fl = m.geom_size[world._gid("floor")]
    rows.append(["floor `floor`", "[0, 0, %s]" % g(-fl[2], 3), "[" + ", ".join(g(v, 3) for v in fl) + "]",
                 "station structure"])
    F["hardware_table"] = _md_table(["part (model name)", "centre (m)", "half sizes (m)", "contact class"], rows)
    # the belt surface row: the slab's half thickness from the model
    slab = m.geom_size[world._belt_surface]
    F["hardware_table"] = F["hardware_table"].replace("0.02]", g(slab[2], 3) + "]", 1)
    F["hardware_table"] = F["hardware_table"].replace("%s]" % g(top - 0.02), "%s]" % g(top - slab[2], 3), 1)

    lines = stn["lines"]
    F.update(entry_y=g(lines["entry_y"]), scan_y=g(lines["scan_y"]),
             exit_y=g(lines["exit_y"]),
             transit_m=g(lines["scan_y"] - lines["entry_y"], 3),
             window_to_scan_m=g(lines["scan_y"] - W, 3))
    F["fall_drop"] = g(world.station.fall_drop)
    F["fall_xy"] = g(world.station.fall_xy)
    F["settle_s"] = g(w["settle_s"])

    # ---------------------------------------------------------------- belt and signals
    LS = cfg["line"]["shifts"]
    classes = LS["belt_speed_classes"]
    vmin = min(v[0] for v in classes.values())
    vmax = max(v[1] for v in classes.values())
    F.update(v_min=g(vmin), v_max=g(vmax))
    zone = [float(v) for v in cfg["line"]["indexing_zone_y"]]
    F.update(zone_y0=g(zone[0]), zone_y1=g(zone[1]), set_y0=g(LS["set_down_y"][0]), set_y1=g(LS["set_down_y"][1]))
    F["damage_look_text"] = (measured.get("damage_look") or {}).get("text") or \
        MARK_MEASURE % "the crushed look's measure in the infeed and head cameras"
    F["halt_text"] = (measured.get("indexing") or {}).get("text") or \
        MARK_MEASURE % "how the belt stops and starts again on an indexing line, and where a riding parcel stops"

    # ---------------------------------------------------------------- robot
    rob = info["robot"]
    rng = rob["joint_range"]
    tq = rob["torque_limit_Nm"]
    jrows = []
    for i, n in enumerate(rob["joints"]):
        jrows.append([str(i), "`%s`" % n, g(rng[i][0], 3), g(rng[i][1], 3), g(tq[i])])
    F["joint_table"] = _md_table(["index", "joint", "lower (rad)", "upper (rad)", "torque limit (N m)"], jrows)
    F["n_joints"] = str(len(rob["joints"]))
    F["n_arm"] = str(len(rob["joints"]) // 2)
    F["cmd_speed"] = g(rob["command_speed_limit_rad_s"])
    F["tick_s"] = g(info["tick_s"])
    F["tick_ms"] = g(info["tick_s"] * 1000)
    F["phys_ms"] = g(info["physics_step_s"] * 1000)
    F["substeps"] = str(int(round(info["tick_s"] / info["physics_step_s"])))
    F["cmd_step"] = g(rob["command_speed_limit_rad_s"] * info["tick_s"], 3)
    F["ticks_per_s"] = g(1.0 / info["tick_s"])
    F["start_posture"] = "[" + ", ".join(g(x, 4) for x in rob["start_posture"]) + "]"
    clear = _clearance(world)
    F["start_clear"] = g(round(clear, 3), 3) if clear < 1e8 else "none"
    xmax, zmin = _robot_extent_x(world)
    F["start_behind_mm"] = mm(round((cx - width / 2) - xmax, 4))
    F["start_xmax"] = g(round(xmax, 3), 3)
    F["mirror_rule"] = _mirror_rule(world, rob["joints"])
    sag = np.abs(world.joint_state().q - np.asarray(rob["start_posture"], float))
    F["start_sag_mrad"] = g(round(float(sag.max()) * 1000, 1), 1)
    tcp = np.asarray(rob["T_flange_tcp"]["left"])
    F["tcp_off"] = g(tcp[2, 3], 4)
    F["tcp_off_mm"] = mm(tcp[2, 3])
    for s in ("Left", "Right"):
        p, Rm = world.tcp_pose(s)
        F["tcp_start_%s" % s.lower()] = "(" + ", ".join(g(v, 2) for v in p) + ")"
    import mujoco
    shp = world.data.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Left_Shoulder_Pitch_Link")]
    F["shoulder_z"] = g(shp[2], 3)
    F["shoulder_y"] = g(abs(shp[1]), 3)
    big = [n for n in rob["joints"] if R.JOINT_DRIVE[n] == "drive_80nm"]
    F["big_drive_joints"] = ", ".join("`%s`" % n for n in big[:2])
    F["torque_big"] = g(R.DRIVE_LIMITS["drive_80nm"]["torque_nm"])
    F["torque_small"] = g(R.DRIVE_LIMITS["drive_36nm"]["torque_nm"])
    F["vendor_speed"] = g(R.DRIVE_LIMITS["drive_36nm"]["speed_rad_s"])
    hb = m.geom_size[world._gid("base_col_head")]
    hp = m.geom_pos[world._gid("base_col_head")]
    F["head_top"] = g(hp[2] + hb[2], 3)
    pd = m.geom_size[world._gid("base_col_pedestal")]
    pp = m.geom_pos[world._gid("base_col_pedestal")]
    F["pedestal_half"] = g(pd[0], 3)
    F["pedestal_top"] = g(pp[2] + pd[2], 3)

    # ---------------------------------------------------------------- the tool-pose level
    tpc = w["tool_pose"]
    F["tp_iters"] = str(int(tpc["iterations"]))
    F["tp_margin"] = g(tpc["joint_margin_rad"], 3)
    F["tp_reached_mm"] = g(1000 * float(tpc["reached_m"]), 2)
    F["tp_reached_deg"] = g(tpc["reached_deg"], 2)
    F["tp_speed_pct"] = g(round(100 * float(tpc["speed_share"])), 0)
    F["tp_stall_pct"] = g(100 * float(tpc["stall_rel"]), 2)
    F["tp_rot_m"] = g(tpc["rot_length_m"], 3)
    F["tp_unlatch_mm"] = g(1000 * float(tpc["unlatch_m"]), 1)
    F["tp_unlatch_pct"] = g(100 * float(tpc["unlatch_rel"]), 1)
    tpm = measured.get("tool_pose") or {}
    F["tp_beyond_text"] = (measured.get("tool_pose_beyond") or {}).get("text") or \
        MARK_MEASURE % "the hands' displacement with a two-hand target beyond reach"
    F["tp_measured_text"] = tpm.get("text") or MARK_MEASURE % "the tool-pose level's settling, tracking and out-of-reach behaviour"

    # ---------------------------------------------------------------- gripper
    grip = info["gripper"]
    F["grip_max_mm"] = mm(grip["max_opening_m"])
    go = w["gripper_object"]
    F["obj_hold_ms"] = g(float(go["hold_s"]) * 1000)
    F["obj_short_mm"] = mm(go["short_m"])
    F["obj_still_mm"] = mm(go["still_m"])
    gm = measured.get("gripper", {})

    def meas(block, key, what, fmt=lambda v: g(v, 3)):
        v = (measured.get(block) or {}).get(key)
        return fmt(v) if v is not None else MARK_MEASURE % what

    F["grip_close_s"] = meas("gripper", "close_s", "closing time from open to closed")
    F["grip_open_s"] = meas("gripper", "open_s", "opening time from closed to open")
    F["grip_open_m"] = meas("gripper", "opening_open_m", "opening reading fully open", lambda v: g(v, 4))
    st0 = (measured.get("gripper") or {}).get("opening_start_m")
    op0 = (measured.get("gripper") or {}).get("opening_open_m")
    F["grip_open_text"] = ("%s m fully open at rest" % F["grip_open_m"] if st0 is None or op0 is None or
                           abs(float(st0) - float(op0)) < 5e-5 else
                           "%s m fully open at rest (%s m at the shift's start, before the fingers have come to rest)"
                           % (F["grip_open_m"], g(st0, 4)))
    F["grip_closed_m"] = meas("gripper", "opening_closed_m", "opening reading closed on nothing")
    F["grip_force_N"] = meas("gripper", "pad_force_N", "pad force on a held carton", lambda v: g(round(v), 0))
    F["grip_block_mm"] = meas("gripper", "block_mm", "test block width", lambda v: g(v, 0))
    F["grip_force_reading"] = meas("gripper", "force_reading", "force reading while holding", lambda v: g(v, 2))
    curve = gm.get("opening_over_command")
    F["grip_curve"] = ("; ".join("%s: %s mm" % (g(c, 2), mm(o)) for c, o in curve) if curve else
                       MARK_MEASURE % "the opening over the command")
    F["grip_hold_kg"] = meas("gripper", "hold_kg", "heaviest carton held by a pinch", lambda v: g(v, 2))
    if gm.get("hold_kg") is not None:
        mmax = float(cfg["line"]["shifts"]["mass_kg"][1])
        F["grip_hold_text"] = ("holds the heaviest parcel of the population (%s kg) against gravity" % g(mmax)
                               if float(gm["hold_kg"]) >= mmax - 1e-9 else
                               "holds a carton of up to about %s kg against gravity" % g(gm["hold_kg"], 2))
    else:
        F["grip_hold_text"] = MARK_MEASURE % "heaviest carton held by a pinch"
    F["grip_detect_text"] = ("`object_detected` turns true about %s ms after the fingers stop on it" % g(gm["detect_ms"])
                             if gm.get("detect_ms") is not None else MARK_MEASURE % "object detection delay on a block")

    # ---------------------------------------------------------------- cameras
    cams = info["cameras"]
    rows = []
    role = {"infeed": "arriving parcels' top faces, the entry",
            "head": "the work window",
            "wrist_left": "close views from the left hand", "wrist_right": "close views from the right hand"}
    CAM_ORDER = ("infeed", "head", "wrist_left", "wrist_right")
    F["n_cameras"] = str(len(cams))
    F["n_cameras_word"] = {4: "four", 5: "five", 6: "six", 7: "seven"}.get(len(cams), str(len(cams)))
    F["n_cameras_cap"] = F["n_cameras_word"].capitalize()
    F["camera_names"] = ", ".join("`%s`" % n for n in CAM_ORDER if n in cams)
    for n in CAM_ORDER:
        c = cams[n]
        hfov = 2 * math.degrees(math.atan((c["width"] / 2) / c["fx"]))
        rows.append(["`%s`" % n, c["mount"], "%d x %d" % (c["width"], c["height"]), g(c["fovy_deg"], 1),
                     g(round(hfov, 1), 1), "%s-%s" % (g(c["depth_range_m"][0]), g(c["depth_range_m"][1])),
                     g(round(c["fx"], 2), 2), role[n]])
    F["camera_table"] = _md_table(["name", "mount", "size (px)", "vertical field (deg)", "horizontal field (deg)",
                                   "depth range (m)", "fx = fy (px)", "sees"], rows)
    Ti = np.asarray(cams["infeed"]["T_world_cam"])
    F["infeed_pos"] = "(" + ", ".join(g(v, 3) for v in Ti[:3, 3]) + ")"
    F["infeed_above"] = g(Ti[2, 3] - top, 3)
    vfov = math.radians(cams["infeed"]["fovy_deg"])
    hfov = 2 * math.atan((cams["infeed"]["width"] / 2) / cams["infeed"]["fx"])
    dist = Ti[2, 3] - top
    F["infeed_y0"] = g(Ti[1, 3] - dist * math.tan(vfov / 2), 3)
    F["infeed_y1"] = g(Ti[1, 3] + dist * math.tan(vfov / 2), 3)
    F["infeed_xhalf"] = g(dist * math.tan(hfov / 2), 2)
    Th = np.asarray(cams["head"]["T_world_cam"])
    F["head_pos"] = "(" + ", ".join(g(v, 3) for v in Th[:3, 3]) + ")"
    zc = Th[:3, 2]
    F["head_pitch"] = g(round(math.degrees(math.asin(-zc[2])), 1), 1)
    t_hit = (Th[2, 3] - top) / -zc[2]
    F["head_aim"] = "(" + ", ".join(g(v, 2) for v in (Th[:3, 3] + t_hit * zc)[:2]) + ", " + g(top) + ")"
    F["head_aim_dist"] = g(round(t_hit, 2), 2)
    Tw = np.asarray(cams["wrist_left"]["T_flange_cam"])
    Twr = np.asarray(cams["wrist_right"]["T_flange_cam"])
    F["wrist_ahead_mm"] = mm(Tw[2, 3])
    F["wrist_side_mm"] = g(round(1000 * math.hypot(Tw[0, 3], Tw[1, 3]), 1), 1)
    F["wrist_axis_deg"] = g(round(math.degrees(math.acos(min(1.0, abs(Tw[2, 2])))), 1), 1)
    F["T_flange_cam"] = _mat(Tw, 4)
    F["T_flange_cam_right"] = _mat(Twr, 4)
    cw = cams["wrist_left"]
    v_cam = np.linalg.inv(Tw) @ np.r_[tcp[:3, 3], 1.0]          # the tool centre point in the camera's frame
    F["wrist_tcp_mm"] = mm(v_cam[2])
    F["wrist_tcp_px"] = g(round(math.hypot(cw["fx"] * v_cam[0] / v_cam[2], cw["fy"] * v_cam[1] / v_cam[2])), 0)
    F["wrist_tcp_text"] = ("the tool centre point lies %s mm ahead of the camera, %s px from the image centre"
                           % (F["wrist_tcp_mm"], F["wrist_tcp_px"]))
    F["wrist_depth_min_mm"] = mm(cw["depth_range_m"][0])
    # the palm cameras' depth inside its minimum range: the fingertip plane lies this far ahead of the camera
    F["wrist_tip_mm"] = mm(v_cam[2])
    gm_ = measured.get("palm_views") or {}
    F["wrist_views_text"] = gm_.get("text") or MARK_MEASURE % "what the palm cameras show (fingers open, a face ahead, a palm at a face)"
    F["T_world_infeed"] = _mat(Ti, 4)
    # arrivals through the infeed camera (no line signal is on the wire): how long the tallest, longest parcel's
    # whole top face stays inside the camera's view at the fastest belt, from its landing at the entry
    cs_, ms_ = LS["rigid_size_m"], LS["soft_size_m"]
    h_top = max(cs_[2][1], ms_[2][1])
    # the longest footprint along the belt: a parcel may lie at any yaw, so the largest footprint's diagonal
    L_long = max(math.hypot(cs_[0][1], cs_[1][1]), math.hypot(ms_[0][1], ms_[1][1]))
    d_top = Ti[2, 3] - (top + h_top)
    yv0, yv1 = Ti[1, 3] - d_top * math.tan(vfov / 2), Ti[1, 3] + d_top * math.tan(vfov / 2)
    y_in = max(float(LS["set_down_y"][1]), yv0 + L_long / 2)     # the set-down range's downstream end: the shortest view
    y_out = yv1 - L_long / 2
    whole_s = (y_out - y_in) / vmax
    exp_s_ = cfg["cameras"]["exposure_ticks"] * info["tick_s"]
    F["infeed_top_h"] = g(h_top, 3)
    F["infeed_long"] = g(round(L_long, 3), 3)
    F["infeed_ytop0"], F["infeed_ytop1"] = g(round(yv0, 2), 2), g(round(yv1, 2), 2)
    F["infeed_whole_y0"], F["infeed_whole_y1"] = g(round(y_in, 2), 2), g(round(y_out, 2), 2)
    F["infeed_whole_s"] = g(round(whole_s, 2), 2)
    F["infeed_poll_s"] = g(math.floor((whole_s - exp_s_) * 100) / 100, 2)
    F["infeed_part_s"] = g(round((yv1 + L_long / 2 - float(LS["set_down_y"][1])) / vmax, 1), 1)
    F["T_world_head"] = _mat(Th, 4)
    c0 = cfg["cameras"]
    F["exp_s"] = g(c0["exposure_ticks"] * info["tick_s"])
    F["exp_ms"] = g(c0["exposure_ticks"] * info["tick_s"] * 1000)
    F["fps"] = g(1.0 / (c0["exposure_ticks"] * info["tick_s"]))
    F["age_ms"] = g(c0["delay_ticks"] * info["tick_s"] * 1000)
    F["age_max_ms"] = g((c0["delay_ticks"] + c0["exposure_ticks"] - 1) * info["tick_s"] * 1000)
    F["exp_ticks"] = str(int(c0["exposure_ticks"]))
    F["delay_ticks"] = str(int(c0["delay_ticks"]))
    sn = cfg["sensors"]
    F["rgb_sigma"] = g(sn["rgb_noise_sigma"])
    F["depth_rel_pct"] = pct(sn["depth_noise_rel"])
    F["dropout_pct"] = pct(sn["depth_dropout"])
    F["fixed_depth"] = "%s-%s" % (g(cams["infeed"]["depth_range_m"][0]), g(cams["infeed"]["depth_range_m"][1]))
    F["wrist_depth"] = "%s-%s" % (g(cams["wrist_left"]["depth_range_m"][0]), g(cams["wrist_left"]["depth_range_m"][1]))
    # what depth shows: the depth twin follows the colour backend (server._depth_meshes)
    _lk = cfg["look"]
    _twin = _lk.get("depth_twin", "auto")
    _twin = (_lk.get("backend") == "filament") if _twin == "auto" else bool(_twin)
    F["depth_surfaces_text"] = ("the station's surroundings near the cameras (the fence, posts, the gantry, the "
                                "conveyor's frame members, the photo-eye) as well as the belt, the robot and the parcels"
                                if _twin else
                                "the station as section 2 gives it (the belt, its frame, legs, rails, guide and lip), the "
                                "robot and the parcels")
    F["look_text"] = measured.get("look_text") or MARK_LOOK % (
        "what the colour images show (the hall, its light, the parcels' and the waybill's printed look)")
    # what each camera resolves: the waybill's papers (the parcel kit's sizes) facing the camera, from the published
    # calibration (pinhole: pixels = f x size / distance), at the distances a parcel's face can have
    from parcelcell.parcelkit.waybill import PAPERS as _P
    papers = sorted(_P.values(), key=lambda p: -p[0] * p[1])
    cs, ms = LS["rigid_size_m"], LS["soft_size_m"]
    hmin, hmax = min(cs[2][0], ms[2][0]), max(cs[2][1], ms[2][1])
    Ti_ = np.asarray(cams["infeed"]["T_world_cam"])
    Th_ = np.asarray(cams["head"]["T_world_cam"])
    rows = []

    def px(cam, dist):
        f = cams[cam]["fx"]
        return ["%d x %d" % (round(f * p[0] / 1000.0 / dist), round(f * p[1] / 1000.0 / dist)) for p in papers]
    for h in (hmax, hmin):
        dist = Ti_[2, 3] - (top + h)
        rows.append(["`infeed`", "top face of a parcel %s m tall, below the camera" % g(h, 3), g(round(dist, 2), 2)]
                    + px("infeed", dist))
    zc_h = Th_[:3, 2]
    for h in (hmax, hmin):
        # along the head camera's optical axis, at the height of a top face
        t_h = (Th_[2, 3] - (top + h)) / -zc_h[2]
        rows.append(["`head`", "top face of a parcel %s m tall where the optical axis meets it (seen obliquely: the "
                     "along-belt side shrinks with the view's slant)" % g(h, 3), g(round(t_h, 2), 2)] + px("head", t_h))
    for dist in (0.15, 0.25, 0.40):
        rows.append(["`wrist_left`, `wrist_right`", "a face straight ahead", g(dist, 2)] + px("wrist_left", dist))
    F["resolve_table"] = _md_table(["camera", "where the waybill is", "distance (m)"]
                                   + ["paper %s x %s mm (px)" % (g(p[0]), g(p[1])) for p in papers], rows)
    F["resolve_note"] = ("Sizes from the published calibration for a waybill facing the camera (short side x long "
                         "side, in pixels), for a parcel on the belt's centre line (where the line sets it down); its "
                         "yaw turns the waybill in the image but does not change these sizes; a face seen at a slant "
                         "covers fewer. A label on a parcel's bottom face is in view of no camera while the parcel "
                         "lies on the belt.")
    ob = measured.get("observability_faces") or {}
    F["observe_table"] = ob.get("table") or MARK_MEASURE % "which camera shows a waybill on each face before the work window"
    F["observe_text"] = ob.get("text") or MARK_MEASURE % "the observability of each face before the work window"
    from parcelcell.parcelkit.waybill import design as _wb_design
    F["waybill_design_text"] = _waybill_words(_wb_design("100x180"), _wb_design("76x130"))
    F["waybill_look"] = F["waybill_design_text"] + (". " + measured["waybill_picture"] if measured.get("waybill_picture")
                                                    else "")

    # ---------------------------------------------------------------- parcels (the line setting)
    from parcelcell.shiftgen import line_list_length
    sc = LS
    shares = sc["kind_shares"]
    F["kind_text"] = ("Each parcel of a list is rigid or soft with an equal chance, independently of the others"
                      if abs(float(shares["rigid"]) - float(shares["soft"])) < 1e-12 else
                      "Each parcel of a list is soft with a probability of %s, independently of the others"
                      % g(shares["soft"], 2))
    cs, ms = sc["rigid_size_m"], sc["soft_size_m"]
    F["carton_x"] = "%s-%s" % (g(cs[0][0], 3), g(cs[0][1], 3))
    F["carton_y"] = "%s-%s" % (g(cs[1][0], 3), g(cs[1][1], 3))
    F["carton_z"] = "%s-%s" % (g(cs[2][0], 3), g(cs[2][1], 3))
    F["mailer_x"] = "%s-%s" % (g(ms[0][0], 3), g(ms[0][1], 3))
    F["mailer_y"] = "%s-%s" % (g(ms[1][0], 3), g(ms[1][1], 3))
    F["mailer_z"] = "%s-%s" % (g(ms[2][0], 3), g(ms[2][1], 3))
    F["max_h"] = g(max(cs[2][1], ms[2][1]), 3)
    F["min_h"] = g(min(cs[2][0], ms[2][0]), 3)
    F["mass"] = "%s-%s" % (g(sc["mass_kg"][0]), g(sc["mass_kg"][1]))
    F["friction"] = "%s-%s" % (g(sc["friction"][0], 2), g(sc["friction"][1], 2))
    F["com_text"] = ("at or near the centre of its box" if float(sc["com_offset_max_m"]) <= 0 else
                     "off the centre of its box (its contents shift): horizontally by up to %s m along each side "
                     "(at most %s of the half side), vertically by up to %s of the half height"
                     % (g(sc["com_offset_max_m"]), g(0.8), g(0.3)))
    F["label_min_mm"] = mm(S.LABEL_MIN_FACE)
    F["uv_pct"] = pct(sc["label_uv_max"])
    from parcelcell.parcelkit.waybill import PAPERS
    from parcelcell.parcelkit.geometry import Carton as _Carton
    papers = sorted(PAPERS.items(), key=lambda kv: -kv[1][0] * kv[1][1]) if isinstance(PAPERS, dict) else []
    F["papers"] = (" or ".join("%s x %s mm" % (g(p[0] * 1000 if p[0] < 1 else p[0]), g(p[1] * 1000 if p[1] < 1 else p[1]))
                               for _, p in papers) if papers else MARK_MEASURE % "the waybill's paper sizes")
    _small = min(PAPERS.values(), key=lambda p: p[0] * p[1])
    _mnom, _mmin = _Carton((0.2, 0.2, 0.2)).label_margin()
    F["small_face_mm"] = mm((_small[0] / 1000.0 if _small[0] > 1 else _small[0]) + 2 * _mnom)
    F["min_margin_mm"] = mm(_mmin)
    F["drop_mm"] = mm(0.002)
    F["landing_drift_text"] = (measured.get("landing_drift") or {}).get("text") or \
        MARK_MEASURE % "how far a riding parcel's pose drifts from its set-down"
    F["rated_ride_m"] = g(sc["rated_ride_m"], 3)
    F["rated_handling_s"] = g(sc["rated_handling_s"])
    F["due_margin_s"] = g(cfg["line"]["score"]["due_margin_s"])
    dmin = dmax = float(sc["duration_s"])                 # every full shift lasts at most the same fixed time
    F["shift_s"] = g(dmin)
    F["bench_shift_s"] = g(float(sc["bench_duration_s"]))
    rows = []
    for name, (lo, hi) in sorted(classes.items(), key=lambda kv: kv[1][0]):
        rows.append("%s to %s parcels at %s-%s m/s" % (line_list_length(lo, sc), line_list_length(hi, sc), g(lo), g(hi)))
    F["list_text"] = "; ".join(rows)
    F["n_min"], F["n_max"] = str(line_list_length(vmin, sc)), str(line_list_length(vmax, sc))
    # mailers, as a data sheet states them (ranges, not the model's constants): checked against the model
    pf = _geom.parcel_fields("soft", (0.2, 0.2, 0.2), 1.0, 0.5)
    body_frac = pf["box_size"][2] / 0.1
    mu_plus = pf["box_friction"][0] - 0.5
    if not (0.75 <= body_frac <= 0.95 and 0.05 <= mu_plus <= 0.25):
        raise RuntimeError("the mailer model left the ranges the manual states")
    F["mailer_body_text"] = ("its firm core 75-95 % of its full height, the cover reaching the full height; its film "
                             "grips rails, pads and other parcels somewhat more than a carton's board, by a friction "
                             "coefficient about 0.05-0.25 higher")

    # ---------------------------------------------------------------- score
    sco = cfg["score"]
    duty = (cfg["line"].get("score") or {}).get("duty") or sco.get("duty") or {}
    from parcelcell.config import LABEL_POSITIONS
    words = {"top": "the top face", "underneath": "the bottom face (underneath)",
             "downstream": "the downstream face", "upstream": "the upstream face",
             "near_side": "the near side face (towards the robot)", "far_side": "the far side face (away from the robot)"}
    kinds = {"rigid": "rigid parcels", "soft": "soft parcels"}
    F["duty_kinds"] = " and ".join(kinds[k] for k in ("rigid", "soft") if k in duty.get("kinds", []))
    F["duty_not_kinds"] = " and ".join(kinds[k] for k in ("rigid", "soft") if k not in duty.get("kinds", [])) or "none"
    inp = [p for p in LABEL_POSITIONS if p in duty.get("label_positions", [])]
    outp = [p for p in LABEL_POSITIONS if p != "top" and p not in duty.get("label_positions", [])]
    F["duty_positions"] = _join([words[p] for p in inp])
    F["duty_not_positions"] = _join([words[p] for p in outp]) if outp else "none"
    F["duty_positions_codes"] = ", ".join("`%s`" % p for p in inp)
    F["label_position_codes"] = ", ".join("`%s`" % p for p in LABEL_POSITIONS)
    F["tilt"] = g(sco["label_tilt_deg"])
    F["max_w"] = g(sco["max_angular_speed_rad_s"])
    F["max_v"] = g(sco["max_rel_speed_m_s"], 3)
    F["rest_window"] = g(sco["rest_window_s"])
    F["rest_clear_mm"] = g(1000.0 * float(sco["rest_clearance_max_m"]))
    F["shifts_graded"] = str(run["shifts_graded"])

    # ---------------------------------------------------------------- safety
    sa = cfg["safety"]
    F.update(structure_N=g(sa["structure_N"]), self_N=g(sa["self_N"]), belt_N=g(sa["belt_N"]),
             press_ticks=str(int(sa["press_ticks"])), press_ms=g(int(sa["press_ticks"]) * info["tick_s"] * 1000),
             stop_lock_s=g(sa["stop_lock_s"]), damage_N=g(sa["damage_N"]), max_stops=str(int(sa["max_stops"])),
             lockout_stop=_ordinal(int(sa["max_stops"]) + 1),
             squeeze_N=g(sa["squeeze_N"]), squeeze_ticks=str(int(sa["squeeze_ticks"])),
             squeeze_ms=g(int(sa["squeeze_ticks"]) * info["tick_s"] * 1000))

    # ---------------------------------------------------------------- budgets
    b = cfg["budgets"]
    F.update(pt_factor=g(b["program_time_factor"]), pt_extra=g(b["program_time_extra_s"]),
             silence_s=g(b["silence_s"]), frames_per_s=g(b["frames_per_shift_s"]),
             max_ticks=str(int(b["max_ticks_per_step"])),
             max_ticks_s=g(int(b["max_ticks_per_step"]) * info["tick_s"]),
             after_steps=str(int(b["after_over_steps"])), after_requests=str(int(b["after_over_requests"])),
             after_s=g(b["after_over_s"]), line_mib=g(b["request_line_bytes"] / (1024 * 1024)),
             world_after_steps=str(AFTER_OVER_STEPS))
    F["pt_rule"] = ("the shift's duration + %s s" % g(b["program_time_extra_s"]) if float(b["program_time_factor"]) == 1.0
                    else "%s x the shift's duration + %s s" % (g(b["program_time_factor"]), g(b["program_time_extra_s"])))
    F["pt_rule_cap"] = F["pt_rule"][0].upper() + F["pt_rule"][1:]
    F["pt_shift"] = g(float(b["program_time_factor"]) * dmin + float(b["program_time_extra_s"]))
    F["frames_shift"] = g(int(math.floor(float(b["frames_per_shift_s"]) * dmin)))
    F["cpu_factor"] = g(run["cpu_cap_factor"])
    start_s = b.get("start_s")
    if start_s is None:
        try:
            import importlib.util as _ilu
            spec = _ilu.spec_from_file_location("review_wire_consts", os.path.join(TESTS, "cellkit", "wire.py"))
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            start_s = mod.START_S
        except (OSError, ImportError, AttributeError):
            start_s = None
    F["start_s"] = g(start_s) if start_s is not None else MARK_MEASURE % "the wire's start window"
    F["priv_factor"] = g(b.get("privileged_program_time_factor", 10))
    F["reply_mb"] = meas("timing", "reply_mb_max", "the largest camera reply", lambda v: g(v, 1))
    F["import_wall"] = g(run["import_check_wall_s"])
    F["import_cpu"] = g(run["import_check_cpu_s"])
    rg = run["route_gate"]
    F["route_mb"] = g(rg["max_submission_mb"])
    F["route_ext"] = ", ".join("`%s`" % e for e in rg["forbidden_extensions"])
    F["route_calls"] = ", ".join("`%s`" % e for e in rg["forbidden_calls"])
    F["blas_threads"] = g(run["blas_threads"])
    F["session_h"] = g(run["session_hours"])
    F["cpus"] = g(run["sandbox"]["cpus"])
    F["memory_gb"] = g(run["sandbox"]["memory_gb"])
    F["nproc"] = meas("sandbox", "nproc", "what `nproc` prints in your shell", lambda v: str(v))
    F["os_cpus"] = meas("sandbox", "os_cpu_count", "`os.cpu_count()` in your shell", lambda v: str(v))
    F["nproc_prog"] = meas("sandbox", "nproc_program", "what `nproc` prints in a program's environment", lambda v: str(v))
    F["os_cpus_prog"] = meas("sandbox", "os_cpu_count_program", "`os.cpu_count()` in a program's environment",
                             lambda v: str(v))
    F["sandbox_note"] = (measured.get("sandbox") or {}).get("note") or ""
    cm = dict(run["commissioning"])
    sys.path.insert(0, REVIEW)
    import gen_shifts as _gs
    import select_hidden as _ss
    cm["bench_shifts"], cm["full_shifts"] = _gs.DEV_BENCH, _gs.DEV_FULL
    cm["development_shifts"] = _gs.DEV_BENCH + _gs.DEV_FULL
    sites_path = os.path.join(os.path.dirname(TESTS), "environment", "commission_service", "sites.json")
    if os.path.exists(sites_path):
        with open(sites_path) as fh:
            sj = json.load(fh)
        cm["shifts_per_session"] = int(sj.get("shifts", cm["shifts_per_session"]))
        cm["open_at_a_time"] = int(sj.get("open", cm["open_at_a_time"]))
        cm["service_sites"] = len(sj.get("sites", []))
    run["commissioning"] = cm
    F["min_W"] = str(_ss.MIN_W)
    run["min_W_per_shift"] = _ss.MIN_W
    F.update(dev_shifts=str(cm["development_shifts"]), bench_shifts=str(cm["bench_shifts"]),
             full_shifts=str(cm["full_shifts"]), comm_shifts=str(cm["shifts_per_session"]),
             comm_open=str(cm["open_at_a_time"]))
    _bn = [len(c[2]) for c in _gs.LINE_BENCH]
    _nw = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    F["bench_n"] = "%s to %s" % (_nw[min(_bn)], _nw[max(_bn)])
    F["bench_contents"] = _gs.LINE_BENCH_TEXT
    F["full_contents"] = "; ".join("%s belt, %s" % (bc, "indexing" if idx else "running") for bc, idx in _gs.LINE_FULL)
    _cls_w = {bc: "%s (%s-%s m/s)" % (bc, g(v[0]), g(v[1])) for bc, v in classes.items()}
    _slots = list(_gs.LINE_SLOTS)
    _n_idx = sum(1 for _bc, idx in _slots if idx)
    F["graded_text"] = ("%s on a running line and %s on an indexing line; %s" % (
        _nw.get(len(_slots) - _n_idx, str(len(_slots) - _n_idx)), _nw.get(_n_idx, str(_n_idx)),
        ", ".join("%s %s" % (_nw.get(sum(1 for b, _ in _slots if b == bc), "?"), _cls_w[bc])
                  for bc in sorted(classes, key=lambda k: classes[k][0]))))
    F["cell_version"] = info["cell_version"]
    import mujoco as _mj
    F["mujoco"] = _mj.__version__.rsplit(".", 1)[0] if _mj.__version__.count(".") >= 2 else _mj.__version__
    F["python"] = str(run["python"])
    F["harness_input"] = str(run["harness_input"])
    F["harness_output"] = str(run["harness_output"])
    F["session_margin_min"] = g(run["session_margin_min"])
    from parcelcell import server as _sv
    F["num_max"] = "%g" % _sv.NUM_MAX
    F["quat_tol"] = "%g" % _sv.QUAT_TOL
    F["word_max"] = str(_sv.WORD_MAX)
    F["infeed_wh"] = "(%d, %d" % (cams["infeed"]["height"], cams["infeed"]["width"])
    F["frame_cost"] = meas("timing", "frame_cost_text", "the wall time of one frame in the task image",
                           lambda v: str(v))
    F["shift_start_text"] = meas("timing", "shift_start_text", "the wall time of a shift's build in the task image",
                                 lambda v: str(v))
    srv.close()
    return F, {"cfg": cfg, "raw": raw, "runner": run, "describe": info, "measured": measured,
               "fall": (float(world.station.fall_drop), float(world.station.fall_xy))}


INFEED_YAW_DEG = 12.0      # the yaw tolerance of the landing (8 degrees) plus the riding drift (section 7: at most about 12), rounded up


def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " or " + items[-1]


def _waybill_words(d, d2):
    """The waybill in words, from the parcel kit's design record (both paper sizes)."""
    e = d["elements"]
    z = d["zones"]
    W, H = d["size_mm"]
    w_b, h_b = d2["size_mm"]
    bc, qr = e["linear_barcode"], e["qr_code"]
    return ("a white thermal-paper sheet of %s x %s mm (or the same design scaled to %s x %s mm), printed in black only. "
            "From its top edge: a header band (%s-%s mm) with the carrier's emblem (a %s), its word mark %s with its "
            "tagline and, at the right, a service box (%s); the sender's block (%s lines); the recipient's block (%s "
            "lines); the routing band (%s-%s mm) with the routing code in %s, cap height %s mm, and a square %s of the "
            "tracking number (about %s mm wide); the barcode band (%s-%s mm) with a %s linear barcode of the tracking "
            "number, %s mm tall, the number printed below it; and a footer. Black rules separate the bands. No other "
            "mark on a parcel carries its print: stickers and stamps have no routing code, no barcode and no 2-D "
            "code, and plain paper stickers, some about the waybill's size and tone, carry no print at all."
            % (g(W), g(H), g(w_b), g(h_b), g(z["header"]["y"][0], 1), g(z["header"]["y"][1], 1),
               e["emblem"]["shape"], e["wordmark"]["text"], e["service_box"]["style"],
               e["sender_lines"], e["recipient_lines"], g(z["routing"]["y"][0], 1), g(z["routing"]["y"][1], 1),
               e["routing_code"]["style"], g(e["routing_code"]["cap_height_mm"], 1),
               "QR code (" + qr["symbol"] + ")", g(qr["box_mm"][2] - qr["box_mm"][0], 1),
               g(z["barcode"]["y"][0], 1), g(z["barcode"]["y"][1], 1), bc["symbology"].split(" (")[0],
               g(bc["bar_height_mm"], 1)))


def _ordinal(n):
    words = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
             9: "ninth", 10: "tenth"}
    return words.get(n, "%dth" % n)
