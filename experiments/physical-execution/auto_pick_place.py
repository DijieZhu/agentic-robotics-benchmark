import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

import robosuite
import robosuite.macros as macros
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import get_real_depth_map
from robocasa.environments.kitchen.atomic.kitchen_pick_place import (
    PickPlaceCounterToSink,
)

macros.IMAGE_CONVENTION = "opencv"

CAMERAS = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


class ReferenceCounterToSink(PickPlaceCounterToSink):
    """Simplified development scene, not the final benchmark."""

    def _get_obj_cfgs(self):
        target = super()._get_obj_cfgs()[0]
        target["obj_groups"] = "apple"
        target["object_scale"] = 0.6
        target["max_size"] = (0.06, 0.06, 0.08)
        # Counter regions are built from a set upstream, so their enumeration
        # can change between Python processes. Canonicalize before sampling.
        regions = self.counter.get_reset_regions(
            env=self, ref=self.sink, loc="left_right"
        )
        regions = sorted(regions.values(), key=lambda r: tuple(r["offset"]))
        target["reset_region"] = regions[int(self.rng.integers(len(regions)))]
        return [target]


def limit_norm(vector, maximum):
    length = np.linalg.norm(vector)
    return vector * min(1.0, maximum / max(length, 1e-12))


class ReferenceController:
    def __init__(self, env, video, trace):
        self.env = env
        self.robot = env.robots[0]
        self.video = video
        self.trace = trace
        self.site = self.robot.eef_site_id["right"]
        self.body = env.obj_body_id["obj"]
        self.stage = "initializing"
        self.steps = 0
        self.stable_success = 0
        self.previous_object = self.object_position()
        self.initial_object_z = self.previous_object[2]
        self.max_lift = 0.0
        self.lift_confirmed = False

    def object_position(self):
        return self.env.sim.data.body_xpos[self.body].copy()

    def eef_position(self):
        return self.env.sim.data.site_xpos[self.site].copy()

    def eef_rotation(self):
        return self.env.sim.data.site_xmat[self.site].reshape(3, 3).copy()

    def object_bounds(self):
        quaternion = self.env.sim.data.body_xquat[self.body].copy()
        points = np.asarray(
            self.env.objects["obj"].get_bbox_points(
                trans=self.object_position(),
                rot=np.roll(quaternion, -1),  # wxyz -> xyzw
            )
        )
        return points.min(axis=0), points.max(axis=0)

    def grasped(self):
        return bool(
            self.env._check_grasp(
                self.robot.gripper["right"],
                self.env.objects["obj"],
            )
        )

    def tick(self, target, rotation, gripper, base_velocity=None):
        if self.steps >= 3500:
            raise RuntimeError("Episode exceeded 3500 control steps")

        position_error = target - self.eef_position()
        rotation_error = Rotation.from_matrix(
            rotation @ self.eef_rotation().T
        ).as_rotvec()

        # World-frame OSC deltas. The configured scales are 0.05 m
        # and 0.5 rad per unit action.
        # A stronger outer position loop corrects payload-induced steady error;
        # the increment cap still bounds approach and transport speed.
        translation = limit_norm(2.0 * position_error, 0.025)
        angular = limit_norm(0.4 * rotation_error, 0.04)
        arm_action = np.concatenate(
            [translation / 0.05, angular / 0.5]
        )

        action = self.robot.create_action_vector({
            "right": np.clip(arm_action, -1.0, 1.0),
            "right_gripper": np.array([gripper]),
            "base": np.zeros(3) if base_velocity is None else base_velocity,
            "base_mode": -1,
        })

        obs, _, _, _ = self.env.step(action)
        self.steps += 1

        position = self.object_position()
        speed = (
            np.linalg.norm(position - self.previous_object)
            * self.env.control_freq
        )
        self.previous_object = position
        self.max_lift = max(
            self.max_lift,
            float(position[2] - self.initial_object_z),
        )

        success = bool(self.env._check_success())
        if success and speed < 0.05:
            self.stable_success += 1
        else:
            self.stable_success = 0

        self.trace.write(json.dumps({
            "step": self.steps,
            "stage": self.stage,
            "action": action.tolist(),
            "eef_target": target.tolist(),
            "eef_position": self.eef_position().tolist(),
            "object_position": position.tolist(),
            "gripper_joint_positions": self.env.sim.data.qpos[
                self.robot._ref_gripper_joint_pos_indexes["right"]
            ].tolist(),
            "arm_base_position": self.robot.composite_controller.get_controller_base_pose(
                "right"
            )[0].tolist(),
            "object_speed_m_s": float(speed),
            "grasp_contact": self.grasped(),
            "native_success": success,
        }) + "\n")

        # Record at 10 FPS while controlling at 20 Hz.
        if self.steps % 2 == 0:
            frame = np.concatenate(
                [obs[f"{camera}_image"] for camera in CAMERAS],
                axis=1,
            )
            self.video.append_data(np.ascontiguousarray(frame))

    def hold(self, target, rotation, gripper, steps):
        for _ in range(steps):
            self.tick(target, rotation, gripper)

    def align_base(self, object_center, destination, rotation):
        """Move along the counter front using base actions, never state edits."""
        self.stage = "align_base"
        composite = self.robot.composite_controller
        base = composite.part_controllers["base"]
        start_base, _ = composite.get_controller_base_pose("right")
        start_eef = self.eef_position()
        tangent = np.array([np.cos(self.env.sink.rot), np.sin(self.env.sink.rot), 0.0])
        midpoint = (object_center + destination) / 2
        target = start_base + tangent * np.dot(midpoint - start_base, tangent)
        site_id = self.env.sim.model.site_name2id(
            self.robot.robot_model.naming_prefix + "right_center"
        )
        print(f"align_base: target={np.round(target, 3)}", flush=True)
        reached = 0
        for _ in range(450):
            position, _ = composite.get_controller_base_pose("right")
            error = target[:2] - position[:2]
            velocity = limit_norm(1.2 * error, 0.06)
            jac = self.env.sim.data.get_site_jacp(
                self.env.sim.model.site_id2name(site_id)
            ).reshape(3, -1)
            joint_velocity = np.linalg.solve(
                jac[:2, base.qvel_index[:2]], velocity
            )
            # The velocity servos must overcome the mobile joints' Coulomb
            # friction. Feed it forward through actions, preserving physics.
            actuator_ids = self.robot._ref_actuators_indexes_dict["base"][:2]
            friction = self.env.sim.model.dof_frictionloss[base.qvel_index[:2]]
            gains = self.env.sim.model.actuator_gainprm[actuator_ids, 0]
            servo_velocity = joint_velocity + friction / gains * np.tanh(
                joint_velocity / 0.005
            )
            _, orientation = base.get_base_pose()
            theta = (
                np.arctan2(orientation[1, 0], orientation[0, 0])
                - np.arctan2(base.init_ori[1, 0], base.init_ori[0, 0])
            )
            c, s = np.cos(theta), np.sin(theta)
            if base._check_forward_joint_reversed():
                mapping = np.array([[c, s], [-s, c]]) @ np.array([[0, 1], [1, 0]])
            else:
                mapping = np.array([[c, -s], [s, c]])
            command = np.r_[mapping.T @ servo_velocity, 0.0]
            # Follow the moving base with the tucked arm in world coordinates.
            eef_target = start_eef + position - start_base
            eef_target[:2] += velocity / self.env.control_freq
            self.tick(eef_target, rotation, -1.0, command)
            reached = reached + 1 if np.linalg.norm(error) < 0.01 else 0
            if reached >= 5:
                self.hold(self.eef_position(), rotation, -1.0, 25)
                return
        raise RuntimeError("Base alignment timed out")

    def move(self, name, target, rotation, gripper, max_steps=300):
        self.stage = name
        target = np.asarray(target, dtype=float)
        print(f"{name}: target={np.round(target, 3)}", flush=True)
        reached = 0

        for _ in range(max_steps):
            self.tick(target, rotation, gripper)
            distance = np.linalg.norm(target - self.eef_position())
            angle = Rotation.from_matrix(
                rotation @ self.eef_rotation().T
            ).magnitude()
            reached = reached + 1 if distance < 0.01 and angle < 0.10 else 0
            if reached >= 5:
                return

        print(
            "Motion diagnostic:",
            {
                "target": target.tolist(),
                "actual_eef": self.eef_position().tolist(),
                "object": self.object_position().tolist(),
                "actual_rotation": self.eef_rotation().tolist(),
                "target_rotation": rotation.tolist(),
                "joint_positions": self.env.sim.data.qpos[
                    self.robot.composite_controller.part_controllers["right"].qpos_index
                ].tolist(),
                "joint_ranges": self.env.sim.model.jnt_range[
                    self.robot._ref_joints_indexes_dict["right"]
                ].tolist(),
            },
            flush=True,
        )
        raise RuntimeError(
            f"{name}: motion timeout; "
            f"position error={distance:.3f} m, "
            f"rotation error={angle:.3f} rad"
        )

    def run(self):
        # Open the gripper and let the scene settle.
        start = self.eef_position()
        initial_rotation = self.eef_rotation()
        self.stage = "settle"
        self.hold(start, initial_rotation, -1.0, 30)

        self.initial_object_z = self.object_position()[2]
        self.max_lift = 0.0

        low, high = self.object_bounds()
        center = (low + high) / 2
        if min((high - low)[:2]) > 0.07:
            raise RuntimeError("Object is too wide for this simple grasp")

        # Keep the initial yaw, with the fingers pointing downward.
        yaw = np.arctan2(initial_rotation[1, 0], initial_rotation[0, 0])
        downward = Rotation.from_euler(
            "xyz", [np.pi, 0.0, yaw]
        ).as_matrix()

        # Use a real basin region, including on double-basin sinks.
        candidates = []
        for name, points in self.env.sink.get_int_sites(
            relative=False
        ).items():
            p0, px, py, pz = map(np.asarray, points)
            destination = p0 + 0.5 * (px - p0) + 0.35 * (py - p0)
            rim_z = max(p[2] for p in points)
            distance = np.linalg.norm(destination[:2] - center[:2])
            candidates.append((distance, name, destination, rim_z))

        if not candidates:
            raise RuntimeError("No sink basin region found")

        _, basin, destination, rim_z = min(candidates, key=lambda x: x[0])
        print(f"Selected sink region: {basin}", flush=True)
        self.align_base(center, destination, initial_rotation)

        grasp = center.copy()
        # The finger pads sit slightly above the grip site when pointing down.
        # Aim below the object's centre so they contact its sides, not its cap.
        grasp[2] -= 0.006
        clear_z = max(high[2] + 0.16, rim_z + 0.20)

        # Raise with the initial orientation before turning above the object.
        raised = self.eef_position()
        raised[2] = max(raised[2], clear_z)
        self.move("raise_only", raised, initial_rotation, -1.0)

        above = grasp.copy()
        above[2] = clear_z
        self.move("above_object", above, downward, -1.0)
        self.move("descend_to_grasp", grasp, downward, -1.0)
        self.stage = "settle_at_grasp"
        self.hold(grasp, downward, -1.0, 20)

        self.stage = "close_gripper"
        print("close_gripper", flush=True)
        self.hold(grasp, downward, 1.0, 40)

        self.move("lift", above, downward, 1.0)
        self.hold(above, downward, 1.0, 15)

        lift = self.object_position()[2] - self.initial_object_z
        if lift < 0.05 or not self.grasped():
            raise RuntimeError(
                f"Grasp failed: lift={lift:.3f} m, "
                f"finger contacts={self.grasped()}"
            )
        print(f"Lift confirmed: {lift:.3f} m", flush=True)
        self.lift_confirmed = True

        over_sink = destination.copy()
        over_sink[2] = clear_z
        self.move("carry_to_sink", over_sink, downward, 1.0)

        if not self.grasped():
            raise RuntimeError("Object slipped during transport")

        # Keep the object's bottom just above the rim before release.
        # This first baseline releases into the basin rather than
        # inserting the whole gripper deep into it.
        low, _ = self.object_bounds()
        bottom_offset = self.eef_position()[2] - low[2]
        release = over_sink.copy()
        release[2] = rim_z + 0.03 + bottom_offset
        self.move("lower_over_basin", release, downward, 1.0)

        if not self.grasped():
            raise RuntimeError("Object slipped before release")

        self.stage = "release"
        print("release", flush=True)
        self.hold(release, downward, -1.0, 35)
        self.move("retreat", over_sink, downward, -1.0)

        self.stage = "verify"
        self.hold(over_sink, downward, -1.0, 50)

        if self.stable_success < 10:
            raise RuntimeError(
                "Final placement did not remain successful and settled "
                "for 10 consecutive steps"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = (
        Path(__file__).resolve().parent
        / "outputs"
        / f"reference_seed{args.seed}_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=True)

    result = {
        "success": False,
        "seed": args.seed,
        "layout": args.layout,
        "style": args.style,
        "controller": "privileged_scripted_reference",
        "scene": "small_apple_without_distractors",
        "base_control": "automatic_alignment_along_counter",
        "arm_control": "world_frame_after_native_reset",
    }
    env = None
    controller = None

    try:
        config = load_composite_controller_config(robot="PandaOmron")

        print("Creating environment...", flush=True)
        env = robosuite.make(
            env_name="ReferenceCounterToSink",
            robots="PandaOmron",
            controller_configs=config,
            layout_ids=[args.layout],
            style_ids=[args.style],
            seed=args.seed,
            initialization_noise=None,
            robot_spawn_deviation_pos_x=0.0,
            robot_spawn_deviation_pos_y=0.0,
            robot_spawn_deviation_rot=0.0,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=CAMERAS,
            camera_widths=192,
            camera_heights=192,
            camera_depths=True,
            control_freq=20,
            ignore_done=True,
        )
        obs = env.reset()
        # Kitchen moves the base after initializing controller references.
        # Keep its native base-frame controller during reset, then refresh
        # references before switching to the world-frame control used here.
        arm_controller = env.robots[0].composite_controller.part_controllers["right"]
        arm_controller.update(force=True)
        arm_controller.input_ref_frame = "world"
        arm_controller.reset_goal()
        print(
            "Initial EEF:",
            env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]],
            flush=True,
        )

        for camera in CAMERAS:
            imageio.imwrite(
                out / f"{camera}_initial_rgb.png",
                obs[f"{camera}_image"],
            )
            np.save(
                out / f"{camera}_initial_depth_m.npy",
                get_real_depth_map(
                    env.sim, obs[f"{camera}_depth"]
                ).astype(np.float32),
            )

        with (
            imageio.get_writer(str(out / "reference.mp4"), fps=10) as video,
            (out / "trace.jsonl").open("w") as trace,
        ):
            controller = ReferenceController(env, video, trace)
            controller.run()

        result["success"] = True

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    finally:
        if controller is not None:
            result.update({
                "last_stage": controller.stage,
                "steps": controller.steps,
                "max_lift_m": controller.max_lift,
                "stable_success_steps": controller.stable_success,
                "lift_confirmed": controller.lift_confirmed,
                "native_success": bool(env._check_success()),
                "final_grasp_contact": controller.grasped(),
                "final_object_position": controller.object_position().tolist(),
                "final_eef_position": controller.eef_position().tolist(),
                "final_gripper_distance_m": float(np.linalg.norm(
                    controller.eef_position() - controller.object_position()
                )),
            })
        if env is not None:
            env.close()
        (out / "result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print("\n" + ("PASS" if result["success"] else "FAIL"))
        print(json.dumps(result, indent=2))
        print(f"Outputs: {out}", flush=True)

    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
