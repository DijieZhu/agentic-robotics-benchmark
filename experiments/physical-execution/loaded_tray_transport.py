"""Privileged physical feasibility test for loading and transporting a tray.

Uses an unmodified RoboCasa tray asset and normal robot actions. No weld,
attachment, runtime object teleportation, or changes to contact parameters.
"""

import argparse
import hashlib
import importlib.util
import json
import traceback
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

import robocasa
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import get_real_depth_map

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pick_place_reference", HERE / "auto_pick_place.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
CAMERAS = reference.CAMERAS
TRAY_XML = (
    Path(robocasa.__file__).resolve().parent
    / "models/assets/objects/objaverse/tray/tray_2/model.xml"
)

CRITERIA = {
    "min_tray_lift_m": 0.08,
    "min_loaded_translation_m": 0.30,
    "delivery_xy_tolerance_m": 0.035,
    "resting_speed_m_s": 0.025,
    "resting_angular_speed_rad_s": 0.10,
    "gripper_clearance_m": 0.25,
    "stable_control_steps": 20,
}


def configure_payload_mass(env, requested_kg):
    """Configure a ballast mass before the first controller action.

    Scale inertia with mass at fixed geometry. Recompute model constants using
    scratch data, so the live scene's positions and velocities are untouched.
    """
    root = env.obj_body_id["obj"]
    model = env.sim.model._model
    ids = [root]
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in ids:
            ids.append(body)
    native_kg = float(model.body_mass[ids].sum())
    if requested_kg is not None:
        ratio = requested_kg / native_kg
        model.body_mass[ids] *= ratio
        model.body_inertia[ids] *= ratio
        mujoco.mj_setConst(model, mujoco.MjData(model))
        env.sim.forward()
    return native_kg, float(model.body_mass[ids].sum())


class LoadedTrayKitchen(reference.ReferenceCounterToSink):
    monitor = None

    def _get_obj_cfgs(self):
        regions = self.counter.get_reset_regions(env=self, ref=self.sink, loc="left_right")
        # The fixed test needs a continuous surface wide enough for both zones.
        region = max(regions.values(), key=lambda r: r["size"][0])
        if region["size"][0] < 0.95:
            raise RuntimeError("This layout has no sufficiently wide counter region")
        placement = dict(
            fixture=self.counter, size=(0.004, 0.004), pos=(0, 0),
            rotation=np.pi / 2, ensure_object_boundary_in_range=False,
        )
        return [
            dict(
                name="tray", obj_groups=str(TRAY_XML), object_scale=0.8,
                reset_region=region,
                placement={**placement, "offset": (-0.22, -0.20)},
            ),
            dict(
                name="obj", obj_groups="apple", object_scale=0.6,
                graspable=True, max_size=(0.06, 0.06, 0.08),
                reset_region=region,
                placement={**placement, "rotation": 0, "offset": (0.10, -0.23)},
            ),
        ]

    def _pre_action(self, action, policy_step=False):
        # Called after mj_step1 at every physics step, not just video frames.
        if self.monitor is not None:
            self.monitor.sample()
        super()._pre_action(action, policy_step)

    def _check_success(self):
        return bool(self.monitor is not None and self.monitor.final_conditions())


class TransportMonitor:
    require_payload = True

    def __init__(self, env):
        self.env = env
        self.robot = env.robots[0]
        self.site = self.robot.eef_site_id["right"]
        self.ids = {name: env.obj_body_id[name] for name in ("obj", "tray")}
        self.initial_tray = self.position("tray")
        self.destination = self.initial_tray + np.array([0.42, 0.0, 0.0])
        self.armed = False
        self.loaded_verified = False
        self.lift_verified = False
        self.loaded_transport_verified = False
        self.retention_violations = 0
        self.physics_samples = 0
        self.max_lift = 0.0
        self.max_loaded_translation = 0.0
        self.max_tilt_deg = 0.0
        self.tray_speed = 0.0
        self.tray_local_points = np.asarray(env.objects["tray"].get_bbox_points())
        self.local_low = self.tray_local_points.min(axis=0)
        self.local_high = self.tray_local_points.max(axis=0)
        # A conservative central rectangle excludes the rim and end handles.
        center = (self.local_low + self.local_high) / 2
        half = (self.local_high - self.local_low) / 2
        self.interior_low = center[:2] - half[:2] * 0.76
        self.interior_high = center[:2] + half[:2] * 0.76

    def position(self, name):
        return self.env.sim.data.body_xpos[self.ids[name]].copy()

    def rotation(self, name):
        return self.env.sim.data.body_xmat[self.ids[name]].reshape(3, 3).copy()

    def bounds(self, name):
        quat = np.roll(self.env.sim.data.body_xquat[self.ids[name]], -1)
        return np.asarray(self.env.objects[name].get_bbox_points(
            trans=self.position(name), rot=quat,
        ))

    def grasped(self, name):
        return bool(self.env._check_grasp(self.robot.gripper["right"], self.env.objects[name]))

    def contained(self):
        points = (self.bounds("obj") - self.position("tray")) @ self.rotation("tray")
        return bool(
            np.all(points[:, :2] >= self.interior_low)
            and np.all(points[:, :2] <= self.interior_high)
            and points[:, 2].min() >= self.local_low[2] - 0.005
            and points[:, 2].max() <= self.local_high[2] + 0.08
        )

    def tray_on_counter(self):
        return bool(self.env.check_contact(self.env.objects["tray"], self.env.counter))

    def sample(self, physics_step=True):
        position = self.position("tray")
        self.tray_speed = float(np.linalg.norm(self.env.sim.data.get_body_xvelp(
            self.env.objects["tray"].root_body
        )))
        if physics_step:
            self.physics_samples += 1
        if not self.armed:
            return
        self.max_lift = max(self.max_lift, float(position[2] - self.initial_tray[2]))
        tilt = np.degrees(np.arccos(np.clip(self.rotation("tray")[2, 2], -1, 1)))
        self.max_tilt_deg = max(self.max_tilt_deg, float(tilt))
        contained = not self.require_payload or self.contained()
        if not contained:
            self.retention_violations += 1
        held = self.grasped("tray")
        airborne = not self.tray_on_counter()
        if held and airborne and position[2] - self.initial_tray[2] >= CRITERIA["min_tray_lift_m"]:
            self.lift_verified = True
        if self.lift_verified and held and airborne and contained:
            distance = float(np.linalg.norm(position[:2] - self.initial_tray[:2]))
            self.max_loaded_translation = max(self.max_loaded_translation, distance)
            if distance >= CRITERIA["min_loaded_translation_m"]:
                self.loaded_transport_verified = True

    def final_conditions(self):
        eef = self.env.sim.data.site_xpos[self.site]
        tray = self.position("tray")
        return bool(
            (not self.require_payload or self.loaded_verified)
            and self.lift_verified and self.loaded_transport_verified
            and self.retention_violations == 0
            and (not self.require_payload or self.contained())
            and np.linalg.norm(tray[:2] - self.destination[:2]) < CRITERIA["delivery_xy_tolerance_m"]
            and abs(tray[2] - self.initial_tray[2]) < 0.012
            and self.tray_on_counter()
            and (not self.require_payload or self.env.check_contact(self.env.objects["obj"], self.env.objects["tray"]))
            and not self.grasped("tray") and not self.grasped("obj")
            and np.linalg.norm(eef - tray) > CRITERIA["gripper_clearance_m"]
            and self.tray_speed < CRITERIA["resting_speed_m_s"]
            and (not self.require_payload or np.linalg.norm(self.env.sim.data.get_body_xvelp(
                self.env.objects["obj"].root_body
            )) < CRITERIA["resting_speed_m_s"])
            and np.linalg.norm(self.env.sim.data.get_body_xvelr(
                self.env.objects["tray"].root_body
            )) < CRITERIA["resting_angular_speed_rad_s"]
        )

    def metrics(self):
        return {
            "payload_required": self.require_payload,
            "loaded_verified": self.loaded_verified,
            "lift_verified": self.lift_verified,
            "loaded_transport_verified": self.loaded_transport_verified,
            "max_tray_lift_m": self.max_lift,
            "max_loaded_translation_m": self.max_loaded_translation,
            "retention_violations": self.retention_violations,
            "physics_samples": self.physics_samples,
            "max_tray_tilt_deg": self.max_tilt_deg,
            "payload_contained": self.contained(),
            "tray_on_counter": self.tray_on_counter(),
            "tray_grasp_contact": self.grasped("tray"),
            "payload_tray_contact": bool(self.env.check_contact(self.env.objects["obj"], self.env.objects["tray"])),
            "final_tray_position": self.position("tray").tolist(),
            "destination": self.destination.tolist(),
            "gripper_distance_m": float(np.linalg.norm(self.env.sim.data.site_xpos[self.site] - self.position("tray"))),
            "final_delivery_error_m": float(np.linalg.norm(self.position("tray")[:2] - self.destination[:2])),
            "final_tray_speed_m_s": self.tray_speed,
            "final_payload_speed_m_s": float(np.linalg.norm(self.env.sim.data.get_body_xvelp(
                self.env.objects["obj"].root_body
            ))),
            "final_tray_angular_speed_rad_s": float(np.linalg.norm(self.env.sim.data.get_body_xvelr(
                self.env.objects["tray"].root_body
            ))),
            "tray_gripper_contacts": [
                [self.env.sim.model.geom_id2name(int(c.geom1)), self.env.sim.model.geom_id2name(int(c.geom2))]
                for c in self.env.sim.data.contact[:self.env.sim.data.ncon]
                if (self.env.sim.model.geom_id2name(int(c.geom1)) in self.env.objects["tray"].contact_geoms
                    or self.env.sim.model.geom_id2name(int(c.geom2)) in self.env.objects["tray"].contact_geoms)
                and ("gripper" in str(self.env.sim.model.geom_id2name(int(c.geom1)))
                     or "gripper" in str(self.env.sim.model.geom_id2name(int(c.geom2))))
            ],
        }


class LoadedTrayController(reference.ReferenceController):
    def __init__(self, env, video, trace):
        super().__init__(env, video, trace)
        self.monitor = env.monitor
        self.level_tray_rotation = self.monitor.rotation("tray")
        self.carrying_tray = False
        self.last_obs = None

    def tick(self, target, rotation, gripper, base_velocity=None):
        if self.steps >= 5000:
            raise RuntimeError("Episode exceeded 5000 control steps")
        error = target - self.eef_position()
        angular = Rotation.from_matrix(rotation @ self.eef_rotation().T).as_rotvec()
        if self.carrying_tray:
            # Regulate the tray's attitude rather than insisting on a vertical
            # wrist. A loaded rim grasp can deflect relative to the fingers.
            angular = Rotation.from_matrix(
                self.level_tray_rotation @ self.monitor.rotation("tray").T
            ).as_rotvec()
        speed_cap = 0.006 if self.carrying_tray else 0.020
        translation = reference.limit_norm(2.0 * error, speed_cap)
        if self.carrying_tray:
            # Maintain height independently of the long horizontal waypoint.
            # A single 3-D norm cap otherwise weakens vertical load correction.
            translation = np.r_[reference.limit_norm(2.0 * error[:2], speed_cap),
                                np.clip(2.0 * error[2], -speed_cap, speed_cap)]
        arm = np.r_[translation / 0.05,
                    reference.limit_norm(0.8 * angular, 0.06) / 0.5]
        action = self.robot.create_action_vector({
            "right": np.clip(arm, -1, 1), "right_gripper": np.array([gripper]),
            "base": np.zeros(3) if base_velocity is None else base_velocity,
            "base_mode": -1,
        })
        self.last_obs, _, _, _ = self.env.step(action)
        self.steps += 1
        self.monitor.sample(physics_step=False)
        valid = self.monitor.final_conditions()
        self.stable_success = self.stable_success + 1 if valid else 0
        self.trace.write(json.dumps({
            "step": self.steps, "stage": self.stage, "action": action.tolist(),
            "eef": self.eef_position().tolist(), "target": target.tolist(),
            "tray": self.monitor.position("tray").tolist(),
            "payload": self.monitor.position("obj").tolist(),
            "tray_grasp": self.monitor.grasped("tray"),
            "payload_grasp": self.monitor.grasped("obj"),
            "payload_contained": self.monitor.contained(),
            "tray_on_counter": self.monitor.tray_on_counter(),
            "retention_violations": self.monitor.retention_violations,
            "stable_success_steps": self.stable_success,
        }) + "\n")
        if self.steps % 2 == 0:
            self.video.append_data(np.ascontiguousarray(np.concatenate(
                [self.last_obs[f"{c}_image"] for c in CAMERAS], axis=1,
            )))
        if self.monitor.retention_violations:
            raise RuntimeError("Payload left the conservative tray interior during transport")

    def move(self, name, target, rotation, gripper, max_steps=300):
        self.stage = name
        target = np.asarray(target, dtype=float)
        print(f"{name}: target={np.round(target, 3)}", flush=True)
        reached = 0
        for _ in range(max_steps):
            self.tick(target, rotation, gripper)
            distance = np.linalg.norm(target - self.eef_position())
            if self.carrying_tray:
                error_rotation = self.level_tray_rotation @ self.monitor.rotation("tray").T
            else:
                error_rotation = rotation @ self.eef_rotation().T
            angle = Rotation.from_matrix(error_rotation).magnitude()
            reached = reached + 1 if distance < 0.01 and angle < 0.10 else 0
            if reached >= 5:
                return
        raise RuntimeError(f"{name}: motion timeout; position error={distance:.3f} m, attitude error={angle:.3f} rad")

    def loading_offset(self, payload_half):
        m = self.monitor
        return np.array([
            max(-0.040, m.interior_low[0] + payload_half[0] + 0.012),
            min(0.055, m.interior_high[1] - payload_half[1] - 0.012), 0.,
        ])

    def carrier_grasp(self):
        m = self.monitor
        local = (m.local_low + m.local_high) / 2
        local[0] = m.local_low[0] + 0.008
        local[1] = 0
        local[2] = m.local_high[2] - 0.028
        return (
            m.position("tray") + m.rotation("tray") @ local,
            Rotation.from_euler("xyz", [np.pi, 0, -np.pi / 2]).as_matrix(),
            6,
        )

    def payload_release_clearance(self):
        return 0.045

    def release_into_carrier(self, release, over_carrier, down):
        self.stage = "release_payload"
        self.hold(release, down, -1, 35)
        self.move("clear_payload", over_carrier, down, -1)
        self.hold(over_carrier, down, -1, 25)

    def retreat_offset(self):
        return np.array([0, -0.15, 0.20])

    def load_payload(self, orientation):
        m = self.monitor
        yaw = np.arctan2(orientation[1, 0], orientation[0, 0])
        down = Rotation.from_euler("xyz", [np.pi, 0, yaw]).as_matrix()

        # Load the apple using the already validated physical grasp strategy.
        points = m.bounds("obj")
        grasp = (points.min(axis=0) + points.max(axis=0)) / 2
        grasp[2] -= 0.006
        above = grasp.copy(); above[2] = 1.18
        self.move("above_payload", above, down, -1)
        self.move("grasp_payload", grasp, down, -1)
        self.stage = "close_payload"
        self.hold(grasp, down, 1, 35)
        self.move("lift_payload", above, down, 1)
        if not m.grasped("obj"):
            raise RuntimeError("Payload grasp failed")
        self.lift_confirmed = True
        # Put the ballast towards the grasped rim to reduce its moment arm.
        # Keep the full payload bbox and a 12 mm reserve inside the same
        # conservative tray interior used by the independent monitor.
        relative_points = (m.bounds("obj") - m.position("obj")) @ m.rotation("tray")
        payload_half = (relative_points.max(axis=0) - relative_points.min(axis=0)) / 2
        over_tray = m.position("tray") + m.rotation("tray") @ self.loading_offset(payload_half)
        over_tray[2] = above[2]
        self.move("payload_over_tray", over_tray, down, 1)
        offset = self.eef_position()[2] - m.bounds("obj")[:, 2].min()
        release = over_tray.copy()
        release[2] = m.bounds("tray")[:, 2].min() + self.payload_release_clearance() + offset
        self.move("lower_payload", release, down, 1)
        self.release_into_carrier(release, over_tray, down)
        if not m.contained() or not self.env.check_contact(self.env.objects["obj"], self.env.objects["tray"]):
            raise RuntimeError("Payload did not settle inside the tray")
        m.loaded_verified = True

    def run(self):
        m = self.monitor
        orientation = self.eef_rotation()
        self.stage = "settle"
        self.hold(self.eef_position(), orientation, -1, 30)
        self.align_base(m.position("tray"), m.destination, orientation)
        self.load_payload(orientation)

        # Pinch the raised front end rim. The original tray orientation is pi/2.
        rim, rim_down, preset_steps = self.carrier_grasp()
        above_rim = rim.copy(); above_rim[2] = 1.18
        self.move("above_tray_rim", above_rim, rim_down, -1)
        # Narrow the empty gripper above the scene, then preserve its opening
        # while descending; a fully open inner finger can hit a near-rim load.
        self.stage = "preset_rim_opening"
        self.hold(above_rim, rim_down, 1, preset_steps)
        self.move("grasp_tray_rim", rim, rim_down, 0)
        self.stage = "close_tray"
        self.hold(rim, rim_down, 1, 40)
        if not m.grasped("tray"):
            raise RuntimeError("Both finger pads did not grasp the tray rim")
        m.initial_tray = m.position("tray")
        m.armed = True
        self.carrying_tray = True
        lifted = rim + np.array([0, 0, 0.12])
        self.move("lift_loaded_tray", lifted, rim_down, 1, max_steps=450)
        self.hold(lifted, rim_down, 1, 15)
        if not m.lift_verified or not m.grasped("tray"):
            raise RuntimeError("Loaded tray was not physically lifted and held")
        destination_eef = lifted + (m.destination - m.position("tray")) * np.array([1, 1, 0])
        self.move("transport_loaded_tray", destination_eef, rim_down, 1, max_steps=600)
        if not m.loaded_transport_verified or not m.grasped("tray"):
            raise RuntimeError("Loaded transport was not confirmed")
        # Place at the original support height using the measured grasp offset.
        lower = destination_eef.copy()
        lower[2] = m.initial_tray[2] + self.eef_position()[2] - m.position("tray")[2]
        self.move("place_loaded_tray", lower, rim_down, 1, max_steps=450)
        self.stage = "release_tray"
        self.hold(lower, rim_down, -1, 40)
        self.carrying_tray = False
        retreat = lower + self.retreat_offset()
        self.move("retreat", retreat, rim_down, -1)
        self.stage = "verify"
        self.hold(retreat, rim_down, -1, 60)
        if self.stable_success < CRITERIA["stable_control_steps"]:
            raise RuntimeError("Final loaded delivery did not remain stable")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--payload-mass-g", type=float, default=None,
                        help="Initial ballast mass at unchanged payload geometry")
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    if args.payload_mass_g is not None and (not np.isfinite(args.payload_mass_g) or args.payload_mass_g <= 0):
        parser.error("--payload-mass-g must be finite and positive")
    return args


def run_trial(args, env_class=LoadedTrayKitchen, controller_class=LoadedTrayController,
              carrier_xml=TRAY_XML, carrier_scale=0.8, experiment="loaded_tray", entry_script=None,
              monitor_class=TransportMonitor, extra_sources=()):
    entry_script = Path(__file__) if entry_script is None else Path(entry_script)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    mass_label = "native" if args.payload_mass_g is None else f"{args.payload_mass_g:g}g"
    out = args.output_root.resolve() / f"{experiment}_{mass_label}_seed{args.seed}_{stamp}"
    out.mkdir(parents=True)
    result = {
        "success": False, "seed": args.seed, "layout": 1, "style": 1,
        "controller": "privileged_scripted_reference", "criteria": CRITERIA,
        "tray_asset": str(carrier_xml), "tray_object_scale": carrier_scale,
        "carrier_type": experiment,
        "script_sha256": hashlib.sha256(entry_script.read_bytes()).hexdigest(),
        "transport_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_script_sha256": hashlib.sha256((HERE / "auto_pick_place.py").read_bytes()).hexdigest(),
        "dependency_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in extra_sources},
        "runtime_attachments": False, "runtime_object_teleportation": False,
        "requested_payload_mass_g": args.payload_mass_g,
        "mass_configuration": "initial_mass_and_inertia_scaled_at_fixed_geometry",
    }
    (out / "script_snapshot.py").write_text(entry_script.read_text())
    (out / "transport_helper_snapshot.py").write_text(Path(__file__).read_text())
    (out / "reference_snapshot.py").write_text((HERE / "auto_pick_place.py").read_text())
    for source in extra_sources:
        (out / f"dependency_{Path(source).name}").write_text(Path(source).read_text())
    env = None; controller = None
    try:
        config = load_composite_controller_config(robot="PandaOmron")
        print(f"Creating {experiment} environment...", flush=True)
        env = robosuite.make(
            env_name=env_class.__name__, robots="PandaOmron", controller_configs=config,
            layout_ids=[1], style_ids=[1], seed=args.seed, initialization_noise=None,
            robot_spawn_deviation_pos_x=0., robot_spawn_deviation_pos_y=0.,
            robot_spawn_deviation_rot=0., has_renderer=False, has_offscreen_renderer=True,
            use_camera_obs=True, camera_names=CAMERAS, camera_widths=256,
            camera_heights=256, camera_depths=True, control_freq=20, ignore_done=True,
        )
        obs = env.reset()
        native_kg, actual_kg = configure_payload_mass(
            env, None if args.payload_mass_g is None else args.payload_mass_g / 1000
        )
        result["native_payload_mass_kg"] = native_kg
        result["payload_mass_kg"] = actual_kg
        print(f"Payload mass: {actual_kg * 1000:.3f} g (native {native_kg * 1000:.3f} g)", flush=True)
        arm = env.robots[0].composite_controller.part_controllers["right"]
        arm.update(force=True); arm.input_ref_frame = "world"; arm.reset_goal()
        env.monitor = monitor_class(env)
        print("Initial tray:", env.monitor.initial_tray, "delivery:", env.monitor.destination, flush=True)
        for camera in CAMERAS:
            imageio.imwrite(out / f"{camera}_initial.png", obs[f"{camera}_image"])
            np.save(out / f"{camera}_depth_m.npy", get_real_depth_map(env.sim, obs[f"{camera}_depth"]))
        result["tray_mass_kg"] = float(env.sim.model.body_mass[env.obj_body_id["tray"]])
        result["tray_size_m"] = env.objects["tray"].size
        result["video_file"] = f"{experiment}.mp4"
        with imageio.get_writer(str(out / result["video_file"]), fps=10) as video, (out / "trace.jsonl").open("w") as trace:
            controller = controller_class(env, video, trace)
            controller.run()
        result["success"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        if controller is not None:
            result.update(controller.monitor.metrics())
            result.update(last_stage=controller.stage, steps=controller.steps,
                          stable_success_steps=controller.stable_success)
            if controller.last_obs is not None:
                imageio.imwrite(out / "final.png", np.concatenate([
                    controller.last_obs[f"{c}_image"] for c in CAMERAS
                ], axis=1))
        if env is not None:
            env.close()
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("PASS" if result["success"] else "FAIL", flush=True)
        print(json.dumps(result, indent=2), flush=True)
        print("Outputs:", out, flush=True)
    return 0 if result["success"] else 1


def main():
    return run_trial(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
