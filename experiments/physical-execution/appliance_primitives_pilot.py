"""Appliance-primitive feasibility pilot for the multi-appliance tending task.

Each primitive runs in its own fresh RoboCasa episode and is judged by a
predicate on the simulator's joint and contact state:

  stove_knob        pinch one stove knob and turn it until RoboCasa reports the
                    burner on
  microwave_start   press the microwave start button, then withdraw
  oven_door         open the oven door from fully closed: drag the handle out
                    with the fingertips, then the door's weight opens it
  oven_rack         hook the oven rack's front lip and slide the rack out (the
                    scene starts with the door open, as RoboCasa's own
                    SlideOvenRack task does)

The controller is a PRIVILEGED scripted reference: it reads fixture geometry,
joint values and contacts from the simulator. It answers one question only:
can a Panda on the Omron base physically operate these controls. It is not a
sensor-limited controller and not a benchmark task. The robot acts through
ordinary actions only; no joint is set and nothing is teleported after reset.

Seeds vary RoboCasa's default base-placement deviation and arm-joint noise
(`--fixed-spawn` removes both). The default layout is 2 because the oven
primitives need a layout with an Oven fixture; RoboCasa lists the layouts
without one in Kitchen.OVEN_EXCLUDED_LAYOUTS.

Run (the lab's conda environment):
    python -I appliance_primitives_pilot.py --primitive stove_knob --seed 0
    python -I appliance_primitives_pilot.py --primitive all --seeds 0 1 2 3 4

Every trial writes outputs/appliance_<primitive>_seed<k>_<time>/ with
result.json, trace.jsonl and a snapshot of this script; failed trials are kept.
With rendering it also writes the initial camera images and a video.

`--no-render` runs physics only (no cameras, no video). On Linux it also keeps
MuJoCo from loading an OpenGL backend, so it works in a container without
EGL/OSMesa libraries.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path

RENDER = "--no-render" not in sys.argv
if not RENDER and platform.system() == "Linux":
    # Import MuJoCo with its GL layer disabled, then stop robosuite from
    # forcing EGL. No GL context is ever created in this mode.
    os.environ["MUJOCO_GL"] = "disable"
    import mujoco  # noqa: F401,E402

    os.environ["MUJOCO_GL"] = "glx"

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import robosuite  # noqa: E402
import robosuite.macros as macros  # noqa: E402
from robosuite.controllers import load_composite_controller_config  # noqa: E402
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen  # noqa: E402

macros.IMAGE_CONVENTION = "opencv"

HERE = Path(__file__).resolve().parent
CAMERAS = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]
PRIMITIVES = ["stove_knob", "microwave_start", "oven_door", "oven_rack"]
BASE_REF = {
    "stove_knob": "stove",
    "microwave_start": "microwave",
    "oven_door": "oven",
    "oven_rack": "oven",
}

# Acceptance thresholds. The joint thresholds are RoboCasa's own predicates
# (Stove.update_state, Fixture.is_open, SlideOvenRack._check_success).
CRITERIA = {
    "stove_knob_on_rad": 0.35,          # burner flame shown from this knob angle
    "stove_knob_target_rad": 0.70,      # commanded angle, for margin
    "button_clearance_m": 0.15,         # Microwave.gripper_button_far
    "door_open_fraction": 0.90,         # Fixture.is_open
    "rack_out_fraction": 0.95,          # SlideOvenRack: fully out
    "stable_control_steps": 20,         # predicate must hold with the hand away
    "hand_clearance_m": 0.10,
}


class AppliancePilotKitchen(Kitchen):
    """A kitchen with no task objects; the robot starts at one appliance."""

    def __init__(self, pilot_primitive="stove_knob", enable_fixtures=None, *args, **kwargs):
        self.pilot_primitive = pilot_primitive
        super().__init__(enable_fixtures=list(enable_fixtures or []), *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.get_fixture(FixtureType.STOVE)
        self.microwave = self.get_fixture(FixtureType.MICROWAVE)
        self.oven = self.get_fixture(FixtureType.OVEN)
        ref = getattr(self, BASE_REF[self.pilot_primitive])
        if ref is None:
            raise RuntimeError(
                f"Layout {self.layout_id} has no fixture for '{self.pilot_primitive}' "
                "(RoboCasa lists oven-less layouts in Kitchen.OVEN_EXCLUDED_LAYOUTS)"
            )
        self.init_robot_base_ref = ref

    def _setup_scene(self):
        super()._setup_scene()
        if self.pilot_primitive == "oven_rack":
            self.oven.open_door(self)  # initial condition, before any action

    def _get_obj_cfgs(self):
        return []

    def _check_success(self):
        return False


def limit_norm(vector, maximum):
    length = np.linalg.norm(vector)
    return vector * min(1.0, maximum / max(length, 1e-12))


def unit(vector):
    return np.asarray(vector, dtype=float) / max(np.linalg.norm(vector), 1e-12)


class Pilot:
    """World-frame Cartesian control of the Panda, as in auto_pick_place.py."""

    MAX_STEPS = 5000

    def __init__(self, env, video, trace):
        self.env = env
        self.sim = env.sim
        self.robot = env.robots[0]
        self.video = video
        self.trace = trace
        self.site = self.robot.eef_site_id["right"]
        self.stage = "initializing"
        self.steps = 0
        self.watch = {}          # name -> callable returning a float for the trace
        self.measure = {}        # result fields filled by the primitive
        arm = self.robot.composite_controller.part_controllers["right"]
        # Kitchen moves the base after initializing controller references.
        arm.update(force=True)
        arm.input_ref_frame = "world"
        arm.reset_goal()
        self.has_torso = "torso" in self.robot.composite_controller.part_controllers
        self.finger_axis_local, self.tip_reach = self._gripper_geometry()

    # ---- simulator readouts (privileged) -------------------------------------

    def eef_position(self):
        return self.sim.data.site_xpos[self.site].copy()

    def eef_rotation(self):
        return self.sim.data.site_xmat[self.site].reshape(3, 3).copy()

    def geom(self, name):
        gid = self.sim.model.geom_name2id(name)
        return (
            self.sim.data.geom_xpos[gid].copy(),
            self.sim.data.geom_xmat[gid].reshape(3, 3).copy(),
            self.sim.model.geom_size[gid].copy(),
        )

    def joint_value(self, name):
        return float(self.sim.data.qpos[self.sim.model.get_joint_qpos_addr(name)])

    def joint_axis_world(self, name):
        jid = self.sim.model.joint_name2id(name)
        body = self.sim.model.jnt_bodyid[jid]
        rotation = self.sim.data.xmat[body].reshape(3, 3)
        return unit(rotation @ self.sim.model.jnt_axis[jid]), self.sim.data.xanchor[jid].copy()

    def joint_range(self, name):
        return self.sim.model.jnt_range[self.sim.model.joint_name2id(name)].copy()

    def touching(self, geom_names):
        if isinstance(geom_names, str):
            geom_names = [geom_names]
        return bool(self.env.check_contact(self.robot.gripper["right"], geom_names))

    def robot_contacts(self):
        """Names of non-robot geoms the robot touches, with the robot geom. For diagnostics."""
        model, data = self.sim.model, self.sim.data
        own = ("robot0_", "gripper0_", "mobilebase0_")
        pairs = set()
        for index in range(data.ncon):
            contact = data.contact[index]
            first = model.geom_id2name(contact.geom1) or f"geom{contact.geom1}"
            second = model.geom_id2name(contact.geom2) or f"geom{contact.geom2}"
            mine = [name.startswith(own) for name in (first, second)]
            if mine[0] != mine[1]:
                pairs.add((first, second) if mine[0] else (second, first))
        return sorted(pairs)

    def joints_at_limit(self, margin=0.03):
        indexes = self.robot._ref_joints_indexes_dict["right"]
        positions = self.sim.data.qpos[self.robot.composite_controller.part_controllers["right"].qpos_index]
        ranges = self.sim.model.jnt_range[indexes]
        return [f"joint{i + 1}={q:.2f} (range {lo:.2f}..{hi:.2f})"
                for i, (q, (lo, hi)) in enumerate(zip(positions, ranges))
                if q - lo < margin or hi - q < margin]

    def _gripper_geometry(self):
        """Finger closing axis in the grip-site frame, and how far the finger
        collision geometry reaches beyond the grip site along its z axis."""
        model, data = self.sim.model, self.sim.data
        gripper = self.robot.gripper["right"]
        prefix = gripper.naming_prefix
        position, rotation = self.eef_position(), self.eef_rotation()
        local = []
        reach = 0.0
        for gid in range(model.ngeom):
            body = model.body_id2name(model.geom_bodyid[gid]) or ""
            if not body.startswith(prefix) or "finger" not in body:
                continue
            if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                continue
            local.append((body, rotation.T @ (data.geom_xpos[gid] - position)))
            # Extent of the geom's bounding box along the grip site's z axis.
            aabb = np.asarray(model.geom_aabb[gid]).reshape(6)
            geom_rotation = data.geom_xmat[gid].reshape(3, 3)
            for signs in np.ndindex(2, 2, 2):
                corner = aabb[:3] + aabb[3:] * (2.0 * np.array(signs) - 1.0)
                world = data.geom_xpos[gid] + geom_rotation @ corner
                reach = max(reach, float((rotation.T @ (world - position))[2]))
        if not local:
            return np.array([0.0, 1.0, 0.0]), 0.0
        bodies = sorted({item[0] for item in local})
        centre = {b: np.mean([i[1] for i in local if i[0] == b], axis=0) for b in bodies}
        spread = centre[bodies[0]] - centre[bodies[-1]]
        spread[2] = 0.0
        axis = np.zeros(3)
        axis[int(np.argmax(np.abs(spread)))] = 1.0
        return axis, reach

    def facing(self, approach, finger_dir):
        """Rotation with the grip site's z axis along `approach` and the finger
        closing axis along `finger_dir` (made orthogonal to the approach)."""
        z = unit(approach)
        f = unit(np.asarray(finger_dir, dtype=float) - np.dot(finger_dir, z) * z)
        if abs(self.finger_axis_local[1]) > 0.5:      # fingers close along site y
            y = f
            x = np.cross(y, z)
        else:                                          # fingers close along site x
            x = f
            y = np.cross(z, x)
        return np.column_stack([x, y, z])

    def nearest_facing(self, approach, finger_dir):
        """`facing`, with the finger axis sign chosen to minimise wrist rotation."""
        current = self.eef_rotation()
        options = [self.facing(approach, finger_dir), self.facing(approach, -np.asarray(finger_dir))]
        return min(options, key=lambda r: Rotation.from_matrix(r @ current.T).magnitude())

    # ---- actuation -------------------------------------------------------------

    def tick(self, target, rotation, gripper, base=None, torso=0.0, speed=0.025, turn=0.04):
        if self.steps >= self.MAX_STEPS:
            raise RuntimeError(f"Episode exceeded {self.MAX_STEPS} control steps")
        position_error = np.asarray(target, dtype=float) - self.eef_position()
        rotation_error = Rotation.from_matrix(rotation @ self.eef_rotation().T).as_rotvec()
        # OSC deltas are scaled by 0.05 m and 0.5 rad per unit action.
        translation = limit_norm(2.0 * position_error, speed)
        angular = limit_norm(0.4 * rotation_error, turn)
        parts = {
            "right": np.clip(np.concatenate([translation / 0.05, angular / 0.5]), -1.0, 1.0),
            "right_gripper": np.array([gripper]),
            "base": np.zeros(3) if base is None else np.asarray(base, dtype=float),
            "base_mode": -1,
        }
        if self.has_torso:
            parts["torso"] = np.array([torso])
        action = self.robot.create_action_vector(parts)
        obs, _, _, _ = self.env.step(action)
        self.steps += 1
        record = {
            "step": self.steps,
            "stage": self.stage,
            "eef_target": np.asarray(target, dtype=float).tolist(),
            "eef_position": self.eef_position().tolist(),
            "gripper_command": float(gripper),
        }
        for name, read in self.watch.items():
            record[name] = float(read())
        self.trace.write(json.dumps(record) + "\n")
        if self.video is not None and self.steps % 2 == 0:
            frame = np.concatenate([obs[f"{camera}_image"] for camera in CAMERAS], axis=1)
            self.video.append_data(np.ascontiguousarray(frame))

    def hold(self, target, rotation, gripper, steps, **kwargs):
        for _ in range(steps):
            self.tick(target, rotation, gripper, **kwargs)

    def move(self, name, target, rotation, gripper, max_steps=400, tolerance=0.01,
             angle_tolerance=0.10, stop=None, **kwargs):
        """Servo to a pose. `stop` is an optional early-exit predicate."""
        self.stage = name
        target = np.asarray(target, dtype=float)
        print(f"{name}: target={np.round(target, 3)}", flush=True)
        reached = 0
        distance = angle = float("nan")
        for _ in range(max_steps):
            self.tick(target, rotation, gripper, **kwargs)
            if stop is not None and stop():
                return True
            distance = np.linalg.norm(target - self.eef_position())
            angle = Rotation.from_matrix(rotation @ self.eef_rotation().T).magnitude()
            reached = reached + 1 if distance < tolerance and angle < angle_tolerance else 0
            if reached >= 5:
                return stop is None
        if stop is not None:
            return False
        contacts = "; ".join(f"{a} on {b}" for a, b in self.robot_contacts()) or "none"
        raise RuntimeError(
            f"{name}: motion timeout; position error={distance:.3f} m, rotation error={angle:.3f} rad; "
            f"robot contacts: {contacts}; arm joints at a limit: {self.joints_at_limit() or 'none'}"
        )

    def settle(self):
        self.stage = "settle"
        self.hold(self.eef_position(), self.eef_rotation(), -1.0, 30)

    def confirm(self, name, predicate, away_from, rotation):
        """The predicate must hold for consecutive steps with the hand withdrawn."""
        self.stage = f"verify_{name}"
        target = self.eef_position()
        stable = 0
        clearance = 0.0
        for _ in range(60):
            self.tick(target, rotation, -1.0)
            clearance = float(np.linalg.norm(self.eef_position() - np.asarray(away_from)))
            stable = stable + 1 if predicate() else 0
        self.measure["stable_success_steps"] = stable
        self.measure["hand_clearance_m"] = clearance
        if stable < CRITERIA["stable_control_steps"]:
            raise RuntimeError(f"{name}: outcome did not hold for {CRITERIA['stable_control_steps']} steps")
        if clearance < CRITERIA["hand_clearance_m"]:
            raise RuntimeError(f"{name}: hand only {clearance:.3f} m from the control")

    # ---- primitives --------------------------------------------------------------

    def stove_knob(self, burner=None):
        stove = self.env.stove
        valid = sorted(k for k, v in stove.knob_joints.items() if v is not None)
        if not valid:
            raise RuntimeError("Stove has no knob joints")
        if burner is None:
            # Nearest knob to the hand: the pilot tests the control, not navigation.
            eef = self.eef_position()
            burner = min(valid, key=lambda k: np.linalg.norm(
                self.joint_axis_world(f"{stove.naming_prefix}knob_{k}_joint")[1] - eef))
        joint = f"{stove.naming_prefix}knob_{burner}_joint"
        bar = f"{stove.naming_prefix}knob_{burner}_main"
        self.measure.update(burner=burner, knob_joint=joint)
        self.watch["knob_angle_rad"] = lambda: self.joint_value(joint)
        self.settle()
        if stove.is_burner_on(env=self.env, burner_loc=burner):
            raise RuntimeError("Burner already on at reset")

        axis, anchor = self.joint_axis_world(joint)
        centre, bar_rotation, bar_size = self.geom(bar)
        outward = axis if np.dot(centre - anchor, axis) >= 0 else -axis
        if np.linalg.norm(centre - anchor) < 1e-4:
            outward = unit(centre - np.asarray(stove.pos))
            outward = axis * np.sign(np.dot(outward, axis) or 1.0)
        # The bar's thin in-plane dimension is where the fingers close.
        in_plane = [i for i in range(3) if abs(np.dot(bar_rotation[:, i], axis)) < 0.5]
        thin = min(in_plane, key=lambda i: bar_size[i])
        finger_dir = bar_rotation[:, thin]
        protrusion = float(max(bar_size[i] for i in range(3) if i not in in_plane) if len(in_plane) < 3 else bar_size[2])
        grasp_rotation = self.nearest_facing(-outward, finger_dir)
        # Finger pads reach `tip_reach` past the grip site; keep the tips just
        # short of the surface the bar stands on.
        depth = max(self.tip_reach - 2.0 * protrusion + 0.004, 0.0)
        grasp = centre + outward * depth
        above = centre + outward * (depth + 0.10)
        self.measure.update(bar_half_size_m=bar_size.tolist(), knob_outward=outward.tolist(),
                            finger_tip_reach_m=self.tip_reach)

        self.move("above_knob", above, grasp_rotation, -1.0)
        self.move("descend_to_knob", grasp, grasp_rotation, -1.0, speed=0.012, tolerance=0.004)
        self.stage = "close_on_knob"
        self.hold(grasp, grasp_rotation, 1.0, 30)
        if not self.touching(bar):
            raise RuntimeError("Fingers did not contact the knob bar")

        # Positive joint motion is a right-handed rotation about the joint axis.
        low, high = self.joint_range(joint)
        sign = 1.0 if high > abs(low) else -1.0
        self.stage = "twist_knob"
        angle = 0.0
        for _ in range(500):
            if abs(self.joint_value(joint)) >= CRITERIA["stove_knob_target_rad"]:
                break
            angle = min(angle + 0.01, abs(self.joint_value(joint)) + 0.35, 2.4)
            twisted = Rotation.from_rotvec(sign * angle * axis).as_matrix() @ grasp_rotation
            self.tick(grasp, twisted, 1.0, turn=0.03)
        else:
            raise RuntimeError(f"Knob reached only {self.joint_value(joint):.3f} rad")
        twisted = self.eef_rotation()
        self.stage = "release_knob"
        self.hold(grasp, twisted, -1.0, 25)
        self.move("retreat_from_knob", above, twisted, -1.0)
        self.measure["knob_angle_rad"] = self.joint_value(joint)
        self.confirm("stove_knob",
                     lambda: bool(stove.is_burner_on(env=self.env, burner_loc=burner))
                     and abs(self.joint_value(joint)) >= CRITERIA["stove_knob_on_rad"],
                     centre, twisted)
        self.measure["knob_angle_rad"] = self.joint_value(joint)

    def microwave_start(self):
        microwave = self.env.microwave
        button = f"{microwave.naming_prefix}start_button"
        self.watch["microwave_on"] = lambda: float(microwave.get_state()["turned_on"])
        self.settle()
        if microwave.get_state()["turned_on"]:
            raise RuntimeError("Microwave already on at reset")
        if not microwave.is_closed(self.env):
            raise RuntimeError("Microwave door not closed at reset")
        centre, rotation, size = self.geom(button)
        normal = rotation[:, int(np.argmin(size))]
        if np.dot(self.eef_position() - centre, normal) < 0:
            normal = -normal                           # points out of the panel, toward the robot
        start_rotation = self.eef_rotation()
        # Press with closed fingertips, hand pointing into the panel.
        press_rotation = self.nearest_facing(-normal, np.array([0.0, 0.0, 1.0]))
        standoff = centre + normal * (self.tip_reach + 0.10)
        through = centre + normal * (self.tip_reach - 0.012)
        lift = self.eef_position()
        lift[2] = max(lift[2], centre[2] - 0.05)
        self.move("raise_hand", lift, start_rotation, 1.0, tolerance=0.02)
        self.move("face_panel", standoff, press_rotation, 1.0, max_steps=600, tolerance=0.015)
        pressed = self.move("press_start", through, press_rotation, 1.0, speed=0.01,
                            stop=lambda: bool(microwave.get_state()["turned_on"]))
        self.measure["button_contact"] = self.touching(button)
        if not pressed:
            raise RuntimeError("Start button was not actuated")
        retreat = centre + normal * (self.tip_reach + CRITERIA["button_clearance_m"] + 0.06)
        self.move("withdraw", retreat, press_rotation, 1.0, tolerance=0.02)
        self.confirm("microwave_start",
                     lambda: bool(microwave.get_state()["turned_on"])
                     and bool(microwave.gripper_button_far(self.env, button="start_button")),
                     centre, press_rotation)
        self.measure["microwave_on"] = bool(microwave.get_state()["turned_on"])

    def drive_base(self, name, target_xy, rotation, gripper=-1.0, max_steps=500):
        """Translate the mobile base through base actions (no state edits). The
        velocity mapping and friction feed-forward follow auto_pick_place.align_base."""
        self.stage = name
        composite = self.robot.composite_controller
        base = composite.part_controllers["base"]
        start_base, _ = composite.get_controller_base_pose("right")
        start_eef = self.eef_position()
        target_xy = np.asarray(target_xy, dtype=float)
        site_name = self.robot.robot_model.naming_prefix + "right_center"
        print(f"{name}: base target={np.round(target_xy, 3)}", flush=True)
        reached = 0
        for _ in range(max_steps):
            position, _ = composite.get_controller_base_pose("right")
            error = target_xy - position[:2]
            velocity = limit_norm(1.2 * error, 0.06)
            jac = self.sim.data.get_site_jacp(site_name).reshape(3, -1)
            joint_velocity = np.linalg.solve(jac[:2, base.qvel_index[:2]], velocity)
            actuator_ids = self.robot._ref_actuators_indexes_dict["base"][:2]
            friction = self.sim.model.dof_frictionloss[base.qvel_index[:2]]
            gains = self.sim.model.actuator_gainprm[actuator_ids, 0]
            servo_velocity = joint_velocity + friction / gains * np.tanh(joint_velocity / 0.005)
            _, orientation = base.get_base_pose()
            theta = (np.arctan2(orientation[1, 0], orientation[0, 0])
                     - np.arctan2(base.init_ori[1, 0], base.init_ori[0, 0]))
            c, s = np.cos(theta), np.sin(theta)
            if base._check_forward_joint_reversed():
                mapping = np.array([[c, s], [-s, c]]) @ np.array([[0, 1], [1, 0]])
            else:
                mapping = np.array([[c, -s], [s, c]])
            command = np.r_[mapping.T @ servo_velocity, 0.0]
            eef_target = start_eef + position - start_base      # carry the arm with the base
            eef_target[:2] += velocity / self.env.control_freq
            self.tick(eef_target, rotation, gripper, base=command)
            reached = reached + 1 if np.linalg.norm(error) < 0.01 else 0
            if reached >= 5:
                self.hold(self.eef_position(), rotation, gripper, 25)
                return
        contacts = "; ".join(f"{a} on {b}" for a, b in self.robot_contacts()) or "none"
        position, _ = composite.get_controller_base_pose("right")
        raise RuntimeError(f"{name}: base motion timed out at {np.round(position[:2], 3)}, "
                           f"error={np.linalg.norm(error):.3f} m; robot contacts: {contacts}")

    def _oven_frame(self):
        """Hinge axis, anchor and the outward (toward the room) direction of the oven door."""
        oven = self.env.oven
        joint = oven.door_joint_names[0]
        axis, anchor = self.joint_axis_world(joint)
        handle, handle_rotation, handle_size = self.geom(f"{oven.naming_prefix}door_handle_main")
        door, _, _ = self.geom(f"{oven.naming_prefix}door_main")
        outward = handle - door
        outward[2] = 0.0
        outward = unit(outward - np.dot(outward, axis) * axis)
        return joint, axis, anchor, handle, handle_size, outward

    def oven_door(self):
        oven = self.env.oven
        joint, axis, anchor, handle, handle_size, outward = self._oven_frame()
        low, high = self.joint_range(joint)
        self.watch["door_fraction"] = lambda: (self.joint_value(joint) - low) / (high - low)
        self.settle()
        if not oven.is_closed(self.env):
            raise RuntimeError("Oven door not closed at reset")
        start_rotation = self.eef_rotation()
        half_width = float(max(handle_size))                 # the handle runs along the hinge axis
        radius = float(np.linalg.norm((handle - anchor) - np.dot(handle - anchor, axis) * axis))
        # Stand beside the door's sweep: the opened door would land on the arm
        # column. One fixed side keeps the arm in the same branch of its kinematics.
        side = 1.0
        stand = anchor + axis * side * (half_width + 0.16) + outward * (radius + 0.02)
        self.drive_base("step_aside", stand[:2], start_rotation, max_steps=700)
        self.measure["base_stand_xy"] = stand[:2].tolist()

        # Drag the bar outward from above with closed fingertips resting on its
        # upper rear quarter. Nothing is ever under the bar, so when the door's
        # weight takes over (past a few degrees) it falls clear of the hand.
        # The hand leans back from the door so its body clears the control
        # panel and knobs above the handle.
        grasp = handle + axis * side * (half_width - 0.16)
        bar_radius = float(min(handle_size[handle_size > 0]))
        lean = np.deg2rad(25.0)
        approach = unit(np.array([0.0, 0.0, -1.0]) * np.cos(lean) - outward * np.sin(lean))
        hand_rotation = self.nearest_facing(approach, axis)
        tip = grasp - outward * (0.65 * bar_radius) + np.array([0.0, 0.0, 0.55 * bar_radius])
        site = tip - approach * self.tip_reach
        handle_geom = f"{oven.naming_prefix}door_handle_main"
        self.move("over_handle", site - approach * 0.10, hand_rotation, 1.0, max_steps=600,
                  tolerance=0.015)
        self.move("touch_handle", site, hand_rotation, 1.0, speed=0.01, tolerance=0.004,
                  stop=lambda: self.touching(handle_geom))
        if not self.touching(handle_geom):
            raise RuntimeError("Fingertips did not contact the door handle")

        sign = 1.0 if high > abs(low) else -1.0
        span = high - low
        release_angle = 0.15
        self.stage = "drag_door"
        angle = 0.0
        for _ in range(400):
            current = abs(self.joint_value(joint) - low)
            if current >= release_angle:
                break
            angle = min(angle + 0.003, current + 0.05)
            swing = Rotation.from_rotvec(sign * angle * axis).as_matrix()
            self.tick(anchor + swing @ (site - anchor), hand_rotation, 1.0, speed=0.02)
        else:
            raise RuntimeError(f"Door reached only {abs(self.joint_value(joint) - low):.3f} rad")
        self.measure["release_angle_rad"] = abs(self.joint_value(joint) - low)
        clear = self.eef_position() + np.array([0.0, 0.0, 0.20])
        self.move("clear_door", clear, hand_rotation, 1.0, tolerance=0.03, angle_tolerance=0.5, speed=0.04)
        self.stage = "door_falls_open"
        for _ in range(300):
            self.tick(clear, hand_rotation, 1.0)
            if oven.is_open(self.env, th=0.985):
                break
        self.measure["door_fraction"] = self.watch["door_fraction"]()
        self.confirm("oven_door", lambda: bool(oven.is_open(self.env, th=CRITERIA["door_open_fraction"])),
                     anchor + Rotation.from_rotvec(sign * span * axis).as_matrix() @ (grasp - anchor),
                     hand_rotation)
        self.measure["door_fraction"] = self.watch["door_fraction"]()

    def oven_rack(self, level=0):
        oven = self.env.oven
        _, axis, anchor, handle, _, outward = self._oven_frame()
        radius = float(np.linalg.norm((handle - anchor) - np.dot(handle - anchor, axis) * axis))
        rack_joint = next(j for j in self.sim.model.joint_names
                          if j.startswith(oven.naming_prefix) and f"rack{level}" in j)
        low, high = self.joint_range(rack_joint)
        fraction = lambda: (self.joint_value(rack_joint) - low) / (high - low)  # noqa: E731
        self.watch["rack_fraction"] = fraction
        self.settle()
        if not oven.is_open(self.env):
            raise RuntimeError("Oven door not open at reset")
        if fraction() > 0.02:
            raise RuntimeError("Rack not pushed in at reset")

        # The rack's front lip: the collision box of the rack body that sits
        # farthest out and is thin along the slide direction.
        model, data = self.sim.model, self.sim.data
        body = model.jnt_bodyid[model.joint_name2id(rack_joint)]
        lips = []
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != body or not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                continue
            rotation = data.geom_xmat[gid].reshape(3, 3)
            extent = np.abs(rotation @ np.diag(model.geom_size[gid])).sum(axis=1)   # world half-extents
            if abs(np.dot(extent, np.abs(outward))) < 0.01 and extent[2] > 0.008:
                lips.append((float(np.dot(data.geom_xpos[gid], outward)), gid, extent))
        if not lips:
            raise RuntimeError("No front lip found on the rack")
        _, lip_gid, lip_extent = max(lips)
        lip = data.geom_xpos[lip_gid].copy()
        # The rack floor: the thin horizontal collision box with the largest footprint.
        floors = []
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != body or not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                continue
            extent = np.abs(data.geom_xmat[gid].reshape(3, 3) @ np.diag(model.geom_size[gid])).sum(axis=1)
            if extent[2] < 0.006:
                floors.append((float(extent[0] * extent[1]), float(data.geom_xpos[gid][2] + extent[2])))
        if not floors:
            raise RuntimeError("No floor plate found on the rack")
        floor_top = max(floors)[1]
        side = 1.0      # one fixed side, as for the door
        half_width = float(np.dot(lip_extent, np.abs(axis)))
        # Reach into the cavity is sensitive to where the base stands, so first
        # drive to one standing pose beside the open door.
        stand = lip + axis * side * (half_width + 0.14) + outward * 0.62
        # RoboCasa may spawn the robot on either side of the open door. Go round
        # the door's tip rather than through it: back out, across, then in.
        base, _ = self.robot.composite_controller.get_controller_base_pose("right")
        if np.dot(base - anchor, axis) * side < half_width + 0.10:
            hold_rotation = self.eef_rotation()
            back = float(np.dot(anchor, outward)) + radius + 0.11
            first = base + outward * (back - float(np.dot(base, outward)))
            second = stand + outward * (back - float(np.dot(stand, outward)))
            self.drive_base("back_out_past_door", first[:2], hold_rotation, gripper=1.0, max_steps=700)
            self.drive_base("cross_behind_door", second[:2], hold_rotation, gripper=1.0, max_steps=700)
        self.drive_base("stand_at_oven", stand[:2], self.eef_rotation(), gripper=1.0, max_steps=700)
        self.measure["base_stand_xy"] = stand[:2].tolist()
        base, _ = self.robot.composite_controller.get_controller_base_pose("right")
        # The open hand is about 0.21 m wide; keep it inside the cavity's side walls.
        hook_along = min(max(half_width - 0.13, 0.0), abs(float(np.dot(base - lip, axis))))
        hook = lip + axis * side * hook_along
        thickness = float(np.dot(lip_extent, np.abs(outward)))
        self.measure.update(lip_centre=lip.tolist(), lip_half_extent=lip_extent.tolist(), rack_floor_top_z=floor_top)

        # Closed fingertips go behind the lip, hand pitched down into the cavity.
        pitch = np.deg2rad(38.0)
        approach = unit(-outward * np.cos(pitch) + np.array([0.0, 0.0, -1.0]) * np.sin(pitch))
        hand_rotation = self.nearest_facing(approach, axis)
        tip = hook - outward * (thickness + 0.035)
        tip[2] = floor_top + 0.004
        site = tip - approach * self.tip_reach
        over = site + np.array([0.0, 0.0, 0.07])
        staging = over + outward * 0.22 + np.array([0.0, 0.0, 0.12])
        self.move("stage_over_door", staging, hand_rotation, 1.0, max_steps=700, tolerance=0.02)
        self.move("over_lip", over, hand_rotation, 1.0, max_steps=500, tolerance=0.012)
        self.move("behind_lip", site, hand_rotation, 1.0, speed=0.012, tolerance=0.006)
        travel = (high - low) + 0.03
        pulled = self.move("pull_rack", site + outward * travel, hand_rotation, 1.0, speed=0.015,
                           max_steps=700, stop=lambda: fraction() >= 0.985)
        self.measure["rack_fraction_after_pull"] = fraction()
        if not pulled and fraction() < CRITERIA["rack_out_fraction"]:
            raise RuntimeError(f"Rack reached only {fraction():.3f} of its travel")
        here = self.eef_position()
        self.move("lift_off_lip", here - outward * 0.012 + np.array([0.0, 0.0, 0.12]), hand_rotation, 1.0,
                  tolerance=0.02)
        self.move("withdraw", self.eef_position() + outward * 0.10 + np.array([0.0, 0.0, 0.08]), hand_rotation, 1.0,
                  tolerance=0.03)
        self.confirm("oven_rack", lambda: fraction() >= CRITERIA["rack_out_fraction"],
                     hook + outward * (high - low), hand_rotation)
        self.measure["rack_fraction"] = fraction()


def source_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def run_trial(primitive, seed, layout, style, render, out_root, fixed_spawn=False):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = out_root / f"appliance_{primitive}_seed{seed}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "success": False,
        "primitive": primitive,
        "seed": seed,
        "layout": layout,
        "style": style,
        "controller": "privileged_scripted_reference",
        "rendered": render,
        "spawn": "fixed" if fixed_spawn else "robocasa_default_deviation",
        "runtime_joint_edits": False,
        "runtime_object_teleportation": False,
        "criteria": CRITERIA,
        "script_sha256": source_digest(),
        "platform": f"{platform.system()} {platform.machine()}",
    }
    env = None
    pilot = None
    # Without spawn deviation every seed replays the same episode, so the default
    # keeps RoboCasa's own base-placement and arm-joint noise.
    spawn = dict(initialization_noise=None, robot_spawn_deviation_pos_x=0.0,
                 robot_spawn_deviation_pos_y=0.0, robot_spawn_deviation_rot=0.0) if fixed_spawn else {}
    try:
        env = robosuite.make(
            env_name="AppliancePilotKitchen",
            pilot_primitive=primitive,
            robots="PandaOmron",
            controller_configs=load_composite_controller_config(robot="PandaOmron"),
            layout_ids=[layout],
            style_ids=[style],
            seed=seed,
            has_renderer=False,
            has_offscreen_renderer=render,
            use_camera_obs=render,
            camera_names=CAMERAS if render else None,
            camera_widths=192,
            camera_heights=192,
            camera_depths=False,
            control_freq=20,
            ignore_done=True,
            **spawn,
        )
        obs = env.reset()
        base_position, _ = env.robots[0].composite_controller.get_controller_base_pose("right")
        result["spawn_base_xy"] = np.asarray(base_position)[:2].tolist()
        video = None
        if render:
            import imageio.v2 as imageio

            for camera in CAMERAS:
                imageio.imwrite(out / f"{camera}_initial.png", obs[f"{camera}_image"])
            video = imageio.get_writer(str(out / f"{primitive}.mp4"), fps=10)
        try:
            with (out / "trace.jsonl").open("w") as trace:
                pilot = Pilot(env, video, trace)
                getattr(pilot, primitive)()
        finally:
            if video is not None:
                video.close()
        result["success"] = True
    except Exception as exc:  # every failure is kept as evidence
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        if pilot is not None:
            result.update(last_stage=pilot.stage, steps=pilot.steps, **pilot.measure)
        if env is not None:
            env.close()
        (out / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        (out / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(f"{'PASS' if result['success'] else 'FAIL'} {primitive} seed={seed} "
          f"stage={result.get('last_stage')} steps={result.get('steps')} "
          f"{result.get('error', '')}", flush=True)
    return result, out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--primitive", choices=PRIMITIVES + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--layout", type=int, default=2)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--fixed-spawn", action="store_true",
                        help="no base or arm spawn noise (every seed is then the same episode)")
    parser.add_argument("--out", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else [0 if args.seed is None else args.seed]
    primitives = PRIMITIVES if args.primitive == "all" else [args.primitive]
    rows = []
    for primitive in primitives:
        for seed in seeds:
            result, out = run_trial(primitive, seed, args.layout, args.style, RENDER, args.out, args.fixed_spawn)
            rows.append({"primitive": primitive, "seed": seed, "success": result["success"],
                         "last_stage": result.get("last_stage"), "steps": result.get("steps"),
                         "error": result.get("error", ""), "folder": out.name})
    if len(rows) > 1:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        summary = args.out / f"appliance_pilot_summary_{stamp}.json"
        summary.write_text(json.dumps({"layout": args.layout, "style": args.style,
                                       "script_sha256": source_digest(), "trials": rows}, indent=2) + "\n")
        print(f"Summary: {summary}", flush=True)
    passed = sum(row["success"] for row in rows)
    print(f"Passed {passed}/{len(rows)}", flush=True)
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
