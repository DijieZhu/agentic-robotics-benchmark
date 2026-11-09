"""messy_desk_env — the task's simulator (NVIDIA Isaac Sim 4.5) and its stdio server.

This file is the environment the grader runs; the agent never gets it. Its home has the desk
lab instead (environment/setup/payload/messy_desk_lab.py), which tests/review/make_lab.py derives
from this file: the same scene, robot, cameras, physics and action rule, without the episode
(no arrivals, requests, disturbances or container service, no detectors, values or privileged
state). It runs under the Isaac interpreter (/opt/isaac/bin/python3, Python 3.10), never under
the task's py3.12 `python3`.

Run as a script it serves one JSON object per line on stdin/stdout:

    {"cmd": "info"}                          -> {"ok": true, "info": {...}}      task card data
    {"cmd": "reset", "seed": <int>}          -> {"ok": true, "obs": {...}, "state": {...}}
    {"cmd": "step", "action": [8 floats, optional 9th: hi-res bitmask]}    -> {"ok": true, "obs": {...}, "state": {...}}
    {"cmd": "close"}                         -> {"ok": true}, then exit

`obs` is exactly what the tested agent receives over the ALE wire. `state` is privileged:
object classes, poses, container membership, the truth behind the open request. The graders use
it to score; the ALE wire never carries it. Everything an episode does is a deterministic
function of its 64-bit seed (arrivals, requests, container service, disturbances, scales).
"""

import base64
import io
import json
import math
import os
import sys
import time

# ---------------------------------------------------------------------------------------
# Task constants (the public contract; the task card is generated from these)
# ---------------------------------------------------------------------------------------
ASSET_ROOT = os.environ.get("MESSY_DESK_ASSETS", "/opt/isaac_assets")
PHYSICS_HZ = 60
CONTROL_HZ = 10
SUBSTEPS = PHYSICS_HZ // CONTROL_HZ
IMAGE_SIZE = 320
# Optional high-resolution frames: an agent that asks for them (ninth action element, a bitmask
# of HIRES_CAMERAS) gets 1920x1080 RGB-D of those cameras in the same observation, at the cost
# of the PNG encoding and transfer (about 0.1 s wall clock per camera): the agent chooses
# between picture quality and wall-clock time step by step. The hi-res render products render
# every frame (switching a product off and on again stops it for good in this Isaac build); the
# three idle 1920x1080 products cost about 9 ms per step.
HIRES_SIZE = (1920, 1080)
HIRES_CAMERAS = {"agentview": 1, "left_view": 2, "eye_in_hand": 4, "oblique": 8}
# The cameras are real sensors (2026-11-07). The three fixed ones stand at their nominal poses (position, look-at
# target; the task card's) only to within a mounting error that differs from desk to desk and stays put through
# an episode: up to CAMERA_MOUNT_ERROR[0] along each world axis and CAMERA_MOUNT_ERROR[1] about each, drawn from a
# stream of the episode's seed of its own (the world's draws are untouched). The hand camera is fixed to the hand
# as documented. Depth is an active-stereo sensor's: Gaussian noise of DEPTH_NOISE_1M at 1 m growing with the square
# of the distance, and a pixel beside a depth discontinuity missing (0) with probability DEPTH_EDGE_DROP, drawn per
# step and camera from the episode's sensor stream. A discontinuity is a jump of inverse depth (what a stereo pair's
# disparity measures) of more than DEPTH_EDGE_INV_JUMP between neighbouring pixels: an object's outline against what
# lies behind it (a 1 cm step at 1 m, 2 cm at 1.4 m), never a smooth surface: a plane at distance c from a camera
# changes inverse depth by at most 1 / (focal * c) a pixel, under 0.007 for every surface of the scene at the fixed
# cameras' range (c > 0.6 m). The hand camera, which looks at surfaces a few centimetres away, has the noise only.
# Neither reaches the physics.
CAMERA_NOMINAL = {"agentview": ([1.35, 0.0, 0.95], [0.42, 0.0, 0.05]),
                  "left_view": ([0.45, -1.15, 0.75], [0.45, 0.0, 0.05]),
                  "oblique": ([1.25, -0.95, 0.85], [0.42, 0.0, 0.08])}
CAMERA_MOUNT_ERROR = (0.01, math.radians(1.0))
DEPTH_NOISE_1M = 0.0015
DEPTH_EDGE_INV_JUMP = 0.01          # 1/m
DEPTH_EDGE_DROP = 0.5
SENSOR_STREAM = 0x53454E53          # keys the sensor stream with the episode's seed ("SENS")

CLASSES = {
    "A": ["003_cracker_box", "004_sugar_box", "008_pudding_box", "009_gelatin_box",
          "036_wood_block", "061_foam_brick"],
    "B": ["002_master_chef_can", "005_tomato_soup_can", "007_tuna_fish_can",
          "010_potted_meat_can", "006_mustard_bottle", "021_bleach_cleanser"],
    "C": ["011_banana", "024_bowl", "025_mug", "035_power_drill", "037_scissors",
          "040_large_marker", "051_large_clamp", "052_extra_large_clamp", "019_pitcher_base"],
}
CLASS_OF = {m: c for c, ms in CLASSES.items() for m in ms}
# Per-model base scale: in every resting pose (standing, on its side, flat) an object offers a
# pinch of at most ~6.8 cm, under the Franka hand's 8 cm opening: for a box that is its middle
# dimension (cracker box 16.4 cm -> 6.9 cm, sugar box 9.3 -> 6.8, mustard bottle 9.6 -> 6.8), for a
# can its diameter (master chef can 10.2 -> 6.3, tuna can 8.6 -> 6.2). The big ones fit a container.
BASE_SCALE = {"003_cracker_box": 0.42, "004_sugar_box": 0.73, "008_pudding_box": 0.63,
              "009_gelatin_box": 0.80, "010_potted_meat_can": 0.70, "006_mustard_bottle": 0.71, "061_foam_brick": 0.92,
              "036_wood_block": 0.6, "021_bleach_cleanser": 0.55,
              "024_bowl": 0.6, "035_power_drill": 0.55, "019_pitcher_base": 0.5,
              "052_extra_large_clamp": 1.0, "002_master_chef_can": 0.62, "007_tuna_fish_can": 0.72,
              "005_tomato_soup_can": 0.92}
INSTANCE_SCALE = (0.92, 1.00)       # the two instances of a model differ in size
# The stock YCB "Axis_Aligned" meshes are authored with their height along local Y (a bowl's
# opening, a bottle's cap and a can's lid face local -Y): a quarter turn about X stands them up.
# The flat tools are the exception: local Y is their length and local Z their thickness, so they
# rest as authored, lying flat. Arrivals are set down in these resting poses.
UPRIGHT_QUAT = (math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0)    # Rx(-90 deg), w x y z
FLAT_MODELS = ("037_scissors", "040_large_marker", "051_large_clamp", "052_extra_large_clamp")
# In the initial piles the cylinders stand (lying, they roll to the container's ends); everything
# else lies on its largest face, or stands on its long edge when lying flat it is too wide to pinch.
PILE_UPRIGHT = ("002_master_chef_can", "005_tomato_soup_can", "007_tuna_fish_can")
# Objects set down on the desk lie with their long side pointing away from the nearest container
# wall, within this many radians: the desk strips between the containers and the desk's edges are
# narrow, and a hand straddling an object lying along a wall would hit that wall.
DESK_YAW_JITTER = 0.6
DISPLAY_NAME = {m: " ".join(m.split("_")[1:]) for m in CLASS_OF}
CONTAINER_OF_CLASS = {"A": "gray", "B": "lightblue", "C": "blue"}
CONTAINER_COLORS = {"gray": (0.45, 0.45, 0.45), "lightblue": (0.62, 0.82, 1.0),
              "blue": (0.08, 0.20, 0.85), "red": (0.85, 0.08, 0.08)}
# containers: the red delivery container right in front of the robot base, the three class
# containers behind it, long sides roughly along x. The stock small_KLT asset is 0.297 x 0.198 x
# 0.146 with its origin at mid-height; it is stretched per container (CONTAINER_SCALES: long side,
# short side, height) and set down so that it stands on the desk. Every episode jitters each
# container's pose (CONTAINER_JITTER: x, y in metres, yaw in radians) around its nominal centre.
KLT_HALF = (0.1483, 0.0989, 0.0732)                    # stock half extents (long, short, height)
CONTAINER_SCALE = (1.10, 1.375, 0.82)     # class containers: short side widened to 24 cm inside (the open hand is 20.5 cm), walls 12 cm
RED_SCALE = (0.80, 1.375, 0.82)           # the delivery container, in front of the base: shorter along x so the class containers stay in reach
CONTAINER_SCALES = {"gray": CONTAINER_SCALE, "lightblue": CONTAINER_SCALE, "blue": CONTAINER_SCALE, "red": RED_SCALE}
# nominal centres: red in front of the base, the three class containers behind it; the middle one
# sits back behind red, the outer two forward so that all of their floor stays within the arm's reach
CONTAINER_NOMINAL = {"gray": (0.44, -0.31), "lightblue": (0.575, 0.0), "blue": (0.44, 0.31), "red": (0.25, 0.0)}
CONTAINER_JITTER = (0.02, 0.008, math.radians(5.0))   # x jitter +-2 cm: the middle container's far wall sits near the arm's reach
CONTAINER_OUTERS = {n: tuple(2.0 * h * k for h, k in zip(KLT_HALF, sc)) for n, sc in CONTAINER_SCALES.items()}   # 0.326 x 0.272 x 0.120; red 0.238 x 0.272 x 0.120
CONTAINER_INNER_HALVES = {n: (0.127 * sc[0], 0.0875 * sc[1]) for n, sc in CONTAINER_SCALES.items()}           # interior half extents (x, y)
CONTAINER_OUTER = CONTAINER_OUTERS["gray"]                                               # the class containers' (the height is shared by all)
CONTAINER_Z0 = KLT_HALF[2] * CONTAINER_SCALE[2] + 0.0005                             # origin height: bottom on the desk
CONTAINER_FLOOR_Z = 0.0005 + 0.0023 * CONTAINER_SCALE[2]                             # inner floor top (ribs along the long walls reach 0.012 * scale)
CONTAINER_RIM_Z = CONTAINER_OUTER[2] + 0.0005
DESK_X, DESK_Y = (0.15, 0.78), (-0.62, 0.62)
ROOM_OFFSET_X = 0.10       # the stock room (table included) is moved this far along +x, see __init__
REACH, BASE_CLEAR = 0.80, 0.36   # arrivals and nudges land no nearer than 0.36 m to the base: nearer, a straight-down hand stops above the desk (measured)
BASE_RESCUE = 0.26         # nearer than this nothing reaches or sees an object: the world sets it back on the desk
BASE_ZONE_REST_STEPS = 20    # steps at rest beside the base before the world sets an object back on the desk
HORIZON_DEFAULT = 20000

# world state machine parameters (per control step)
ARRIVAL_MEAN_STEPS = 500
LOOSE_CAP = 6
PRESENT_CAP = 22
CONTAINER_CAP = 9
REQUEST_SLOT_STEPS = 5000    # one request opens at a random time in each 5000-step window (4 per 20000-step episode)
REQUEST_DEADLINE_STEPS = 1200
REQUEST_ABSENT_P = 0.2
DISTURB_MEAN_STEPS = 2500
MAX_INSTANCES_PER_MODEL = 2

ACTION_DIM = 8
POS_STEP_MAX, ROT_STEP_MAX = 0.05, 0.25
GRIPPER_OPEN, GRIPPER_CLOSED = 0.04, 0.0
REST_SPEED = 0.08          # piles of convex-decomposed meshes jitter at a few cm/s
HOLD_LIFT = 0.03           # "held": both fingers on the object and its centre this far above where it last rested
ON_TABLE_Z_MAX = 0.16      # centre height of anything resting on the table top (tallest YCB upright ~0.25)
LOST_VALUE = -1            # an object knocked off the table counts as misfiled for the rest of the episode
REBASE_TIMEOUT = 100       # steps a nudged object may keep moving before its baseline is re-read anyway
PARK_Z = -3.0
RESET_WARMUP_FRAMES = 20
READY_Q = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]   # Franka 'ready': tool straight down
HOME_POS = (0.46, 0.0, 0.39)
IK_JUMP_MAX = 0.8                                     # rad: larger joint jumps are branch flips
IK_UNWIND_STEP = 0.3                                  # rad per step toward a solution beyond that jump
TCP_Z_MIN = 0.028                                     # open fingertips touch the desk at TCP z ~ 0.028 (measured)
FINGER_KP, FINGER_KD, FINGER_EFFORT = 2.0e4, 1.0e3, 100.0   # both finger drives (N/m, N s/m, N)
FREECAM = os.environ.get("MESSY_DESK_FREECAM", "")   # e.g. "800x500": extra free-flying camera for the dev portal


def _park_slot(k):
    return (-1.2 + 0.3 * (k % 9), -1.2 + 0.3 * (k // 9), PARK_Z + 0.1)


def _log(msg):
    print("[env] " + msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------
# Isaac side
# ---------------------------------------------------------------------------------------
class MessyDeskEnv:
    def __init__(self, headless=True):
        sys.argv += ["--/rtx/verifyDriverVersion/enabled=false",
                     "--/persistent/isaac/asset_root/default=" + ASSET_ROOT]
        from isaacsim import SimulationApp
        # anti_aliasing 0: the default (DLSS, temporal) accumulates previous frames, which leaves
        # ghost depth under anything that moved (the fingers, a swinging object): floating points
        # 1-2 cm above the desk and a 'skirt' around every object. Depth must be geometry.
        self.app = SimulationApp({"headless": headless, "renderer": "RayTracedLighting",
                                  "width": IMAGE_SIZE, "height": IMAGE_SIZE, "anti_aliasing": 0})
        import numpy as np
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.prims import get_prim_at_path
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot.manipulators.examples.franka import Franka, KinematicsSolver
        from isaacsim.sensors.camera import Camera
        self.np, self.Gf, self.Usd, self.UsdGeom, self.UsdPhysics, self.PhysxSchema = np, Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
        self.ArticulationAction, self.SingleRigidPrim, self.Camera = ArticulationAction, SingleRigidPrim, Camera
        self.get_prim_at_path, self.add_reference_to_stage = get_prim_at_path, add_reference_to_stage

        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / PHYSICS_HZ)
        self.stage = omni.usd.get_context().get_stage()
        # The robot stands on the stock table, which is flat for |x| < 0.709 around its centre and
        # then rises into a rounded 1.2 cm rim. The room is shifted by ROOM_OFFSET_X so that the
        # whole reachable desk (x <= 0.80) is flat; the rim now starts at x = 0.809.
        room = UsdGeom.Xform.Define(self.stage, "/World/RoomOffset")
        room.AddTranslateOp().Set(Gf.Vec3d(ROOM_OFFSET_X, 0.0, 0.0))
        add_reference_to_stage(ASSET_ROOT + "/Isaac/Environments/Simple_Room/simple_room.usd", "/World/RoomOffset/Room")
        self.franka = self.world.scene.add(Franka(prim_path="/World/Franka", name="franka",
                                                  usd_path=ASSET_ROOT + "/Isaac/Robots/Franka/franka.usd"))
        self._build_containers()
        self._build_parking()
        self._build_objects()
        self._build_cameras()
        # One World.reset() for the life of the process: it starts the simulation and
        # initialises every scene object (robot, rigid prims, contact view). Resetting the
        # World again later would invalidate PhysX's tensor views, so an episode reset only
        # teleports objects and re-homes the arm while the simulation keeps playing.
        self.world.reset()
        self._probe_views("after world.reset")
        self.ik = KinematicsSolver(self.franka)      # Lula kinematics of the `right_gripper` frame
        self._home_q = np.array(self.franka.get_joints_default_state().positions, dtype=float)
        # Isaac's default Franka posture points the tool 45 degrees forward-down and folds the
        # elbow (q4 ~ -2.8, q6 ~ 3.0): descending from it runs q6 into its limit near z = 0.2
        # and the IK flips branch. Home is instead the classic "ready" orientation (tool straight
        # down, fingers opening along y) at HOME_POS, solved from the ready posture; from there
        # the whole desk is reachable with small joint motions, and every later IK is
        # warm-started from the current joints so it stays on this branch.
        _, hr = self.ik._kinematics_solver.compute_forward_kinematics(self.ik._ee_frame, np.array(READY_Q, dtype=float))
        hq = np.array(_mat_to_wxyz(hr), dtype=float)
        q7, ok = self.ik._kinematics_solver.compute_inverse_kinematics(
            self.ik._ee_frame, np.array(HOME_POS, dtype=float), hq, warm_start=np.array(READY_Q, dtype=float))
        if ok:
            self._home_q[:7] = np.asarray(q7, dtype=float)
        _log("home posture ok=%s q=%s" % (ok, np.round(self._home_q[:7], 3).tolist()))
        self.franka.set_joint_positions(self._home_q)
        self.franka.set_joint_velocities(np.zeros_like(self._home_q))
        self.franka.apply_action(self.ArticulationAction(joint_positions=self._home_q))
        # The stock asset drives only one finger (7 N, the other finger has no drive at all):
        # objects get pushed instead of squeezed. Drive both fingers like the real hand.
        ctrl = self.franka.get_articulation_controller()
        kps, kds = ctrl.get_gains()
        kps, kds = np.array(kps, dtype=float), np.array(kds, dtype=float)
        kps[-2:], kds[-2:] = FINGER_KP, FINGER_KD
        ctrl.set_gains(kps=kps, kds=kds)
        eff = np.array(ctrl.get_max_efforts(), dtype=float)
        eff[-2:] = FINGER_EFFORT
        ctrl.set_max_efforts(eff)
        self._probe_views("after KinematicsSolver")
        for c in self.cams.values():
            c.initialize()
            c.add_distance_to_image_plane_to_frame()
        for c in self.hcams.values():
            c.initialize()
            c.add_distance_to_image_plane_to_frame()
        if self.freecam is not None:
            self.freecam.initialize()
        self._probe_views("after cameras")
        self.world.step(render=True)
        self._probe_views("after first step")
        self.horizon = HORIZON_DEFAULT
        self.rng = None
        self._sensor_key = 0
        self.t = 0
        _log("ready")

    def _probe_views(self, label):
        try:
            p, _ = self.objects[0]["rp"].get_world_pose()
            _log("views ok %s (obj0 at %s)" % (label, [round(float(x), 2) for x in p]))
        except Exception as exc:  # noqa: BLE001
            _log("views BROKEN %s: %s" % (label, exc))

    # -- scene construction ------------------------------------------------------------
    def _build_containers(self):
        Gf, UsdGeom, Usd, UsdPhysics = self.Gf, self.UsdGeom, self.Usd, self.UsdPhysics
        self.containers, self.container_rp = {}, {}
        for name, (x0, y0) in CONTAINER_NOMINAL.items():
            path = "/World/Containers/" + name
            self.add_reference_to_stage(ASSET_ROOT + "/Isaac/Props/KLT_Bin/small_KLT.usd", path)
            prim = self.get_prim_at_path(path)
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(x0, y0, CONTAINER_Z0))
            xf.AddRotateZOp().Set(90.0)   # long side along x
            # scale is applied in the asset's own frame (short side x, long side y)
            sc = CONTAINER_SCALES[name]
            xf.AddScaleOp().Set(Gf.Vec3f(float(sc[1]), float(sc[0]), float(sc[2])))
            rb = UsdPhysics.RigidBodyAPI.Apply(prim)
            rb.CreateKinematicEnabledAttr(True)
            self.container_rp[name] = self.SingleRigidPrim(prim_path=path, name="container_" + name)
            # The KLT asset ships one shared textured material (magenta); the RTX renderer
            # ignores displayColor, so each container gets its own flat-coloured material bound
            # with the strongest binding strength over every mesh.
            from pxr import Sdf, UsdShade
            mat = UsdShade.Material.Define(self.stage, "/World/Looks/bin_" + name)
            shader = UsdShade.Shader.Define(self.stage, "/World/Looks/bin_" + name + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*CONTAINER_COLORS[name]))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            for m in Usd.PrimRange(prim):
                if m.IsA(UsdGeom.Mesh):
                    binding = UsdShade.MaterialBindingAPI.Apply(m)
                    binding.Bind(mat, UsdShade.Tokens.strongerThanDescendants)
            self.containers[name] = (x0, y0, 0.0)

    def _place_containers(self):
        """Per-episode container poses: nominal layout + seeded jitter (position and yaw)."""
        np, rng = self.np, self.rng
        for name, (x0, y0) in CONTAINER_NOMINAL.items():
            x = x0 + rng.uniform(-CONTAINER_JITTER[0], CONTAINER_JITTER[0])
            y = y0 + rng.uniform(-CONTAINER_JITTER[1], CONTAINER_JITTER[1])
            yaw = rng.uniform(-CONTAINER_JITTER[2], CONTAINER_JITTER[2])
            a = math.pi / 2 + yaw               # the asset's long side is its local y
            self.container_rp[name].set_world_pose(position=np.array([x, y, CONTAINER_Z0]), orientation=np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]))
            self.containers[name] = (x, y, yaw)

    def _container_local(self, name, x, y):
        """Desk point -> coordinates in the container's own frame (x along its long side)."""
        bx, by, yaw = self.containers[name]
        dx, dy = x - bx, y - by
        c, s = math.cos(yaw), math.sin(yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def _container_point(self, name, lx, ly):
        bx, by, yaw = self.containers[name]
        c, s = math.cos(yaw), math.sin(yaw)
        return bx + c * lx - s * ly, by + s * lx + c * ly

    def _build_parking(self):
        # Parked objects are ordinary dynamic bodies resting on a hidden floor 3 m below the
        # room, out of every camera's view. Toggling bodies between kinematic and dynamic at
        # run time invalidates PhysX's tensor views, so nothing here ever changes body type.
        from isaacsim.core.api.objects import GroundPlane
        self.world.scene.add(GroundPlane(prim_path="/World/ParkingFloor", name="parking_floor",
                                         z_position=PARK_Z - 0.05, visible=False))

    def _build_objects(self):
        """Every instance that can ever appear exists from the start, parked far below the
        desk; spawning and removal are teleports. Convex decomposition is cooked once."""
        Gf, UsdGeom, Usd, UsdPhysics, PhysxSchema = self.Gf, self.UsdGeom, self.Usd, self.UsdPhysics, self.PhysxSchema
        self.objects = []          # list of dicts, index = object id
        k = 0
        for model in CLASS_OF:
            for inst in range(MAX_INSTANCES_PER_MODEL):
                path = "/World/Objects/o%02d" % k
                self.add_reference_to_stage(ASSET_ROOT + "/Isaac/Props/YCB/Axis_Aligned/%s.usd" % model, path)
                prim = self.get_prim_at_path(path)
                UsdPhysics.RigidBodyAPI.Apply(prim)
                mass = UsdPhysics.MassAPI.Apply(prim)
                mass.CreateMassAttr(0.2)
                # contact reporting is declared here, before the simulation starts: applying
                # physics APIs once PhysX is playing re-parses the scene and invalidates every
                # tensor view already created.
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
                for m in Usd.PrimRange(prim):
                    if m.IsA(UsdGeom.Mesh):
                        UsdPhysics.CollisionAPI.Apply(m)
                        UsdPhysics.MeshCollisionAPI.Apply(m).CreateApproximationAttr().Set("convexDecomposition")
                        PhysxSchema.PhysxCollisionAPI.Apply(m)
                xf = UsdGeom.Xformable(prim)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(*_park_slot(k)))
                xf.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
                # Scale is fixed per instance (the two instances of a model differ by 20 %):
                # editing a rigid body's scale while PhysX plays re-parses the scene and
                # invalidates every tensor view, so it is never touched after construction.
                scale = BASE_SCALE.get(model, 1.0) * INSTANCE_SCALE[inst]
                xf.AddScaleOp().Set(Gf.Vec3f(float(scale)))
                # half height in the resting pose: local Y for the standing models, Z for the flat ones
                rng3 = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeUntransformedBound(prim).ComputeAlignedRange()
                lo_b, hi_b = rng3.GetMin(), rng3.GetMax()
                axis = 2 if model in FLAT_MODELS else 1
                half_h = float(max(abs(lo_b[axis]), abs(hi_b[axis]))) * scale
                # footprint in the resting pose, before the yaw: (along world x, along world y)
                foot = (float(hi_b[0] - lo_b[0]) * scale, float(hi_b[1 if model in FLAT_MODELS else 2] - lo_b[1 if model in FLAT_MODELS else 2]) * scale)
                # resting pose in a pile: lying on its largest face when the hand can still pinch
                # it across its narrower side from above (at most 6.4 cm in the 8 cm opening), else
                # standing on its long edge like a book (the larger cracker box: 6.9 cm wide lying
                # flat, 3 cm on edge). Edge-standing boxes tip and slide, so only when needed.
                halves = [float(max(abs(lo_b[i]), abs(hi_b[i]))) * scale for i in range(3)]
                ext = [float(hi_b[i] - lo_b[i]) * scale for i in range(3)]
                if model in PILE_UPRIGHT:
                    pile_rest, pile_half = list(UPRIGHT_QUAT), halves[1]
                else:
                    order = sorted(range(3), key=lambda i: ext[i])
                    ax = order[0] if ext[order[1]] <= 0.064 else order[1]      # 1.6 cm of clearance or more
                    pile_rest = ([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0], list(UPRIGHT_QUAT), [1.0, 0.0, 0.0, 0.0])[ax]
                    pile_half = halves[ax]
                rp = self.SingleRigidPrim(prim_path=path, name="obj%02d" % k)
                self.world.scene.add(rp)
                self.objects.append(dict(id=k, model=model, cls=CLASS_OF[model], prim=prim, rp=rp,
                                         park=_park_slot(k), present=False, removed=False, scale=scale, half_h=half_h, foot=foot,
                                         pile_rest=pile_rest, pile_half=pile_half))
                k += 1
        # contact tracking between every object and the two finger links (first-contact bonus)
        self._contacts = None
        try:
            from isaacsim.core.prims import RigidPrim
            for f in ("/World/Franka/panda_leftfinger", "/World/Franka/panda_rightfinger"):
                PhysxSchema.PhysxContactReportAPI.Apply(self.get_prim_at_path(f)).CreateThresholdAttr(0.0)
            self._contacts = RigidPrim(prim_paths_expr="/World/Objects/o*", name="objects_view",
                                       contact_filter_prim_paths_expr=["/World/Franka/panda_leftfinger",
                                                                       "/World/Franka/panda_rightfinger"],
                                       prepare_contact_sensors=False,
                                       max_contact_count=len(self.objects) * 2 * 8)
            self.world.scene.add(self._contacts)
        except Exception as exc:  # noqa: BLE001
            _log("contact view unavailable (%s); using the distance proxy" % exc)
            self._contacts = None

    def _build_cameras(self):
        """Four RGB-D cameras: three fixed (front, left, oblique front-left) and one in the hand.
        Poses use Isaac's "world" camera axes (+X forward, +Z up)."""
        np, Camera = self.np, self.Camera
        self.cams = {}
        fixed = CAMERA_NOMINAL
        for name, (pos, target) in fixed.items():
            cam = Camera(prim_path="/World/Cameras/" + name, frequency=CONTROL_HZ, resolution=(IMAGE_SIZE, IMAGE_SIZE))
            cam.set_world_pose(position=np.array(pos), orientation=np.array(_look_at_wxyz(pos, target)), camera_axes="world")
            self.cams[name] = cam
        eih = Camera(prim_path="/World/Franka/panda_hand/eye_in_hand", frequency=CONTROL_HZ, resolution=(IMAGE_SIZE, IMAGE_SIZE))
        # forward (+X in world camera axes) along the hand's +Z, i.e. out between the fingers
        eih.set_local_pose(translation=np.array([0.05, 0.0, 0.03]), orientation=np.array([0.7071068, 0.0, -0.7071068, 0.0]), camera_axes="world")
        self.cams["eye_in_hand"] = eih
        # the same four views again at HIRES_SIZE (encoded and sent only when an action asks for them)
        self.hcams = {}
        for name, (pos, target) in fixed.items():
            # acquisition at the render rate: a camera at the control rate only takes a frame once the
            # simulated clock has advanced 0.1 s, and the extra render after switching on does not advance it
            cam = Camera(prim_path="/World/Cameras/" + name + "_hires", frequency=PHYSICS_HZ, resolution=HIRES_SIZE)
            cam.set_world_pose(position=np.array(pos), orientation=np.array(_look_at_wxyz(pos, target)), camera_axes="world")
            self.hcams[name] = cam
        heih = Camera(prim_path="/World/Franka/panda_hand/eye_in_hand_hires", frequency=PHYSICS_HZ, resolution=HIRES_SIZE)
        heih.set_local_pose(translation=np.array([0.05, 0.0, 0.03]), orientation=np.array([0.7071068, 0.0, -0.7071068, 0.0]), camera_axes="world")
        self.hcams["eye_in_hand"] = heih
        for c in list(self.cams.values()) + list(self.hcams.values()):
            c.set_clipping_range(0.05, 6.0)
            c.set_focal_length(1.6)          # wider field of view than the 2.4 cm default
        self._hires_pending = {}
        self.freecam = None
        if FREECAM:
            w, h = (int(v) for v in FREECAM.lower().split("x"))
            self.freecam = Camera(prim_path="/World/Cameras/free", frequency=CONTROL_HZ, resolution=(w, h))
            self.freecam.set_clipping_range(0.05, 10.0)
            self.freecam.set_focal_length(1.8)
            self.set_freecam([1.3, -0.9, 0.9], math.atan2(0.9, -0.85), -0.5)

    def _mount_cameras(self, seed):
        """This desk's fixed cameras: each nominal pose moved by its mounting error (CAMERA_MOUNT_ERROR), from the
        sensor stream of `seed`; the hi-res twin of a camera gets the same pose. Never touches the world's stream."""
        np = self.np
        self._sensor_key = int(seed) & ((1 << 64) - 1)
        g = np.random.default_rng([self._sensor_key, SENSOR_STREAM, 0])
        for name, (pos, target) in CAMERA_NOMINAL.items():
            dp = g.uniform(-CAMERA_MOUNT_ERROR[0], CAMERA_MOUNT_ERROR[0], 3)
            dr = g.uniform(-CAMERA_MOUNT_ERROR[1], CAMERA_MOUNT_ERROR[1], 3)
            q = _quat_mul(_rpy_to_quat(float(dr[0]), float(dr[1]), float(dr[2])), _look_at_wxyz(pos, target))
            p = np.array(pos, dtype=float) + dp
            for cam in (self.cams[name], self.hcams[name]):
                cam.set_world_pose(position=p, orientation=np.array(q, dtype=float), camera_axes="world")

    def _depth_sensor(self, z, k, edges=True):
        """What the depth sensor reports for the true depth `z` (metres, 0 where nothing is hit): Gaussian noise
        of DEPTH_NOISE_1M * z^2, and pixels beside a jump of inverse depth of more than DEPTH_EDGE_INV_JUMP missing
        (0) with probability DEPTH_EDGE_DROP (`edges`: the fixed cameras). Drawn from the episode's sensor stream,
        keyed by the step and camera `k`."""
        np = self.np
        z = np.asarray(z, dtype=np.float32)
        g = np.random.default_rng([self._sensor_key, SENSOR_STREAM, 1, int(self.t), int(k)])
        w = np.where(z > 0, 1.0 / np.maximum(z, np.float32(1e-3)), np.float32(0.0))
        edge = np.zeros(z.shape, dtype=bool)
        jx = np.abs(np.diff(w, axis=1)) > DEPTH_EDGE_INV_JUMP
        jy = np.abs(np.diff(w, axis=0)) > DEPTH_EDGE_INV_JUMP
        edge[:, 1:] |= jx
        edge[:, :-1] |= jx
        edge[1:, :] |= jy
        edge[:-1, :] |= jy
        out = z + g.standard_normal(z.shape, dtype=np.float32) * (np.float32(DEPTH_NOISE_1M) * z * z)
        drop = edge & (g.random(z.shape, dtype=np.float32) < DEPTH_EDGE_DROP) & bool(edges)
        return np.where((z > 0) & ~drop & (out > 0), out, np.float32(0.0))

    def _hires_capture(self, names):
        """This step's frames of the requested hi-res cameras (they render every frame, acquired
        at the render rate). -> {name: (rgb uint8 HxWx3, depth_mm uint16 HxW)}"""
        np = self.np
        out = {}
        for name in names:
            cam = self.hcams[name]
            img, d = cam.get_rgba(), cam.get_depth()
            if img is None or img.size == 0:
                img = np.zeros((HIRES_SIZE[1], HIRES_SIZE[0], 4), np.uint8)
            if d is None or d.size == 0:
                d = np.zeros((HIRES_SIZE[1], HIRES_SIZE[0]), np.float32)
            dm = self._depth_sensor(np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0), 10 + list(HIRES_CAMERAS).index(name),
                                    edges=name in CAMERA_NOMINAL) * 1000.0
            out[name] = (np.ascontiguousarray(img[..., :3]).astype(np.uint8), np.clip(dm, 0, 65535).astype(np.uint16))
        return out

    def set_freecam(self, pos, yaw, pitch):
        """Free camera pose: position + heading (yaw about z) and pitch (positive = up)."""
        np = self.np
        fwd = [math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)]
        target = [pos[i] + fwd[i] for i in range(3)]
        self.freecam.set_world_pose(position=np.array(pos, dtype=float), orientation=np.array(_look_at_wxyz(pos, target)), camera_axes="world")

    def freecam_jpeg(self, quality=80):
        np = self.np
        img = self.freecam.get_rgba()
        if img is None or img.size == 0:
            return None
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(img[..., :3]).astype(np.uint8), "RGB").save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # -- geometry helpers ------------------------------------------------------------------
    def _container_of_point(self, p):
        for name in self.containers:
            lx, ly = self._container_local(name, float(p[0]), float(p[1]))
            ix, iy = CONTAINER_INNER_HALVES[name]
            if abs(lx) <= ix and abs(ly) <= iy and CONTAINER_FLOOR_Z - 0.02 <= p[2] <= CONTAINER_RIM_Z + 0.12:
                return name
        return None

    def _footprint_dist(self, name, x, y):
        """Distance from (x, y) to container `name`'s outer footprint (0 inside it)."""
        lx, ly = self._container_local(name, x, y)
        ox, oy = CONTAINER_OUTERS[name][0] / 2, CONTAINER_OUTERS[name][1] / 2
        return math.hypot(max(abs(lx) - ox, 0.0), max(abs(ly) - oy, 0.0))

    def _random_desk_spot(self, margin=0.06, gap_clear=0.12, tries=200):
        """A free desk spot within reach, `margin` clear of every container and never in or at the
        mouth of the narrow gap between two containers (`gap_clear` from the second-nearest one):
        an object there cannot be reached by a hand that has to fit between two walls."""
        rng = self.rng
        occupied = [tuple(o["rp"].get_world_pose()[0][:2]) for o in self.objects if o["present"]]
        for _ in range(tries):
            x = rng.uniform(*DESK_X)
            y = rng.uniform(*DESK_Y)
            if math.hypot(x, y) > REACH - 0.05 or math.hypot(x, y) < BASE_CLEAR:
                continue
            d = sorted(self._footprint_dist(n, x, y) for n in self.containers)
            if d[0] < margin or (len(d) > 1 and d[1] < gap_clear):
                continue
            if any(math.hypot(x - ox, y - oy) < 0.09 for ox, oy in occupied):
                continue
            return x, y
        return None

    # -- object lifecycle --------------------------------------------------------------------
    def _teleport(self, obj, xyz, quat):
        np = self.np
        obj["rp"].set_world_pose(position=np.array(xyz, dtype=float), orientation=np.array(quat, dtype=float))
        obj["rp"].set_linear_velocity(np.zeros(3))
        obj["rp"].set_angular_velocity(np.zeros(3))

    def _desk_yaw(self, obj, x, y):
        """Yaw for an object set down at (x, y): its long side along the outward normal of the
        nearest container wall, +- DESK_YAW_JITTER, either way round."""
        best = None
        for name in self.containers:
            ox, oy = CONTAINER_OUTERS[name][0] / 2, CONTAINER_OUTERS[name][1] / 2
            lx, ly = self._container_local(name, x, y)
            dx, dy = max(abs(lx) - ox, 0.0), max(abs(ly) - oy, 0.0)
            d = math.hypot(dx, dy)
            if best is None or d < best[0]:
                best = (d, math.atan2(math.copysign(dy, ly), math.copysign(dx, lx)) + self.containers[name][2])
        normal = best[1] if best is not None else 0.0
        long_is_x = obj["foot"][0] >= obj["foot"][1]
        return normal - (0.0 if long_is_x else math.pi / 2) + self.rng.uniform(-DESK_YAW_JITTER, DESK_YAW_JITTER) \
            + (math.pi if self.rng.random() < 0.5 else 0.0)

    def _spawn(self, obj, xyz, yaw=None, on_desk=False, rest=None):
        """Place an object in its resting pose (upright, or flat for the flat tools) turned by
        `yaw` about z. `on_desk`: set it down on the desk top, its bottom 4 mm above it, oriented
        by `_desk_yaw` unless a yaw is given."""
        if yaw is None:
            yaw = self._desk_yaw(obj, xyz[0], xyz[1]) if on_desk else self.rng.uniform(-math.pi, math.pi)
        if rest is None:
            rest = [1.0, 0.0, 0.0, 0.0] if obj["model"] in FLAT_MODELS else list(UPRIGHT_QUAT)
        if on_desk:
            xyz = (xyz[0], xyz[1], obj["half_h"] + 0.004)
        self._teleport(obj, xyz, _quat_mul([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], rest))
        obj["present"], obj["removed"], obj["frozen"], obj["rebase_t"] = True, False, 0, None
        obj["rest_z"] = float(xyz[2])      # re-read once it rests (see HOLD_LIFT)
        if getattr(self, "baseline", None) is not None:
            self.baseline[obj["id"]] = 0

    def _park(self, obj):
        self._teleport(obj, obj["park"], [1.0, 0.0, 0.0, 0.0])
        obj["present"] = False

    def _free_instance(self, model):
        for o in self.objects:
            if o["model"] == model and not o["present"]:
                return o
        return None

    def _present_count(self, model):
        return sum(1 for o in self.objects if o["model"] == model and o["present"])

    # -- episode -------------------------------------------------------------------------------
    def reset(self, seed, horizon=None):
        np = self.np
        self.rng = np.random.default_rng(int(seed) & ((1 << 64) - 1))
        self.horizon = int(horizon or HORIZON_DEFAULT)
        self.t = 0
        self.events = []
        self.request = None
        self.requests_done = []
        self.touched = set()
        self.franka.set_joint_positions(self._home_q)
        self.franka.set_joint_velocities(np.zeros_like(self._home_q))
        self.franka.apply_action(self.ArticulationAction(joint_positions=self._home_q))
        self.franka.gripper.open()
        for o in self.objects:
            o["removed"], o["frozen"], o["state"], o["on_table"], o["rebase_t"] = False, 0, "absent", False, None
            self._park(o)
        self._place_containers()
        # initial piles: A in gray, B in lightblue; a few loose C and A on the desk
        for cls, name in (("A", "gray"), ("B", "lightblue")):
            models = list(CLASSES[cls])
            self.rng.shuffle(models)
            n = int(self.rng.integers(4, 7))         # five to seven objects per initial pile
            pile_top = CONTAINER_FLOOR_Z
            for i in range(n + 1):
                o = self._free_instance(models[i % len(models)])
                if o is None:
                    continue
                # each object is set down on top of the pile near the container's centre line, the
                # way a person fills a box; thrown in from above the rim they rolled to the ends,
                # where no top-down hand (0.205 m wide in a 0.28 m container) can reach them
                px, py = self._container_point(name, self.rng.uniform(-0.015, 0.015), self.rng.uniform(-0.03, 0.03))
                rest = o["pile_rest"]
                if i == n:
                    # the last one is tossed on top, tilted: it comes to rest leaning on its
                    # neighbours (a box does not stay flat-packed; the pile has to be dug)
                    tilt = self.rng.uniform(math.radians(35), math.radians(60))
                    rest = _quat_mul(_rpy_to_quat(tilt, 0.0, 0.0), list(rest))
                    px, py = self._container_point(name, self.rng.uniform(-0.03, 0.03), self.rng.uniform(-0.04, 0.04))
                self._spawn(o, (px, py, pile_top + o["pile_half"] + (0.03 if i == n else 0.01)), rest=rest)
                self._settle_until_rest(max_steps=150, quiet=6)
                z = float(o["rp"].get_world_pose()[0][2])
                pile_top = max(pile_top, z + o["pile_half"])
        loose = [m for m in self.rng.permutation(CLASSES["C"])[: int(self.rng.integers(2, 5))]]
        loose += [m for m in self.rng.permutation(CLASSES["A"])[: int(self.rng.integers(0, 3))]]
        for m in loose:
            o = self._free_instance(m)
            spot = self._random_desk_spot()
            if o is None or spot is None or self._present_count(m) >= MAX_INSTANCES_PER_MODEL:
                continue
            self._spawn(o, (spot[0], spot[1], 0.0), on_desk=True)
        self._settle_until_rest()
        self.next_arrival = int(self.rng.exponential(ARRIVAL_MEAN_STEPS))
        self.request_slot = 0
        self.next_request = self._request_time(0)
        self.next_disturb = int(self.rng.exponential(DISTURB_MEAN_STEPS)) + 500
        self._mount_cameras(seed)
        for _ in range(RESET_WARMUP_FRAMES):   # the RTX renderer needs frames to converge
            self.world.step(render=True)
        self._update_states()
        # v(0): container membership only. Nothing is held at reset, so "resting" is not required
        # here; demanding it would let a pile that is still jittering drift upward later.
        self.baseline = {o["id"]: self._value(o, require_rest=False) for o in self.objects}
        _log("reset seed=%d settled_steps=%d present=%d max_speed=%.3f" % (
            int(seed) & 0xFFFF, self._settled_steps, sum(o["present"] for o in self.objects),
            max([o.get("speed", 0.0) for o in self.objects if o["present"]] or [0.0])))
        return self.observe(), self.privileged()

    def _settle(self, n):
        for _ in range(n):
            self.world.step(render=False)

    def _settle_until_rest(self, max_steps=600, quiet=12):
        """Run physics until every present object has been at rest for `quiet` consecutive
        steps (or `max_steps`), so that the baseline v(0) is taken on a settled scene and an
        untouched episode cannot drift upward on its own."""
        np = self.np
        still = 0
        self._settled_steps = max_steps
        for i in range(max_steps):
            self.world.step(render=False)
            moving = False
            for o in self.objects:
                if o["present"] and float(np.linalg.norm(o["rp"].get_linear_velocity())) >= REST_SPEED:
                    moving = True
                    break
            still = 0 if moving else still + 1
            if still >= quiet:
                self._settled_steps = i + 1
                break

    # -- stepping ------------------------------------------------------------------------------
    def step(self, action):
        np = self.np
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.shape[0] not in (ACTION_DIM, ACTION_DIM + 1) or not np.all(np.isfinite(a)):
            raise ValueError("action must be %d finite numbers, plus an optional hi-res bitmask" % ACTION_DIM)
        hires_mask = int(round(a[ACTION_DIM])) if a.shape[0] > ACTION_DIM else 0
        hires = [n for n, bit in HIRES_CAMERAS.items() if hires_mask & bit]
        dpos = np.clip(a[:3], -POS_STEP_MAX, POS_STEP_MAX)
        drot = np.clip(a[3:6], -ROT_STEP_MAX, ROT_STEP_MAX)
        grip = float(np.clip(a[6], 0.0, 1.0))
        mark = int(round(a[7]))          # 0 none, 1 "done: delivered", 2 "done: no such object"
        self._apply_arm(dpos, drot, grip)
        for i in range(SUBSTEPS):
            if self._arm_plan is not None:
                # the step's joint targets are spread over its physics substeps, like a real
                # arm controller interpolating a set point: applied all at once, the stiff joint
                # drives cover the whole delta within a substep or two and the jerk flings a
                # held object out of the fingers
                q0, q1, idx = self._arm_plan
                self.franka.apply_action(self.ArticulationAction(joint_positions=q0 + (q1 - q0) * ((i + 1.0) / SUBSTEPS),
                                                                 joint_indices=idx))
            self.world.step(render=(i == SUBSTEPS - 1))
        self.t += 1
        self._update_states()
        self._world_events(mark)
        return self.observe(hires=self._hires_capture(hires)), self.privileged()

    def _ee_pose(self):
        """Tool-centre-point pose in the base frame, from the kinematics solver's own
        end-effector frame (`right_gripper`, between the fingertips): the same frame the IK
        targets refer to, so a delta applied to it is consistent."""
        np = self.np
        pos, rot = self.ik.compute_end_effector_pose()
        return np.asarray(pos, dtype=float), np.asarray(_mat_to_wxyz(rot), dtype=float)

    def _apply_arm(self, dpos, drot, grip):
        np = self.np
        pos, quat = self._ee_pose()
        target = pos + dpos
        target[0] = float(np.clip(target[0], -0.1, 0.85)); target[1] = float(np.clip(target[1], -0.75, 0.75)); target[2] = float(np.clip(target[2], TCP_Z_MIN, 0.9))
        tq = np.array(_quat_mul(_rpy_to_quat(*drot), [float(x) for x in quat]), dtype=float)
        tq /= max(float(np.linalg.norm(tq)), 1e-9)
        q_cur = np.asarray(self.franka.get_joint_positions(), dtype=float)[:7]
        q_new, idx, ok = self._solve_ik(np.asarray(target, dtype=float), tq, q_cur)
        self._arm_plan = None
        if q_new is not None:
            self._arm_plan = (q_cur, q_new, idx)
            self._ik_q = [float(x) for x in q_new]
        w = GRIPPER_OPEN * (1.0 - grip) + GRIPPER_CLOSED * grip
        self.franka.gripper.apply_action(self.ArticulationAction(joint_positions=np.array([w, w])))
        self._ik_ok = bool(ok)

    def _solve_ik(self, target, tq, q_cur):
        """Joint targets for a tool pose. One candidate is warm-started from the current joints,
        the way a tracking controller solves; three more are seeded from the 'ready' posture turned
        toward the target (base joint at the target's azimuth, forearm and wrist roll at zero, the
        hand twist as it is now / as the target yaw asks for, either sign). With seven joints and
        no posture term, a numerical IK followed over a long chain of small deltas lets the base
        and the forearm wind up (the base 150 degrees from the target's azimuth, the forearm roll
        at its limit) until every further delta is a branch flip and the arm is stuck. Of the
        candidates within the per-step jump limit (a solution on another branch would whip the arm
        across the desk), the one nearest the unwound posture is taken, so the arm unwinds whenever
        it can. When every solution is beyond that limit the arm steps toward the best one at
        IK_UNWIND_STEP per step instead of standing still (an arm wound to a joint limit could
        otherwise refuse every move for the rest of the episode); ik_ok is False until the pose is
        reached. A pose with no solution at all is refused and the arm stays put. -> (q, joint indices, ok)"""
        np = self.np
        act, ok = self.ik.compute_inverse_kinematics(target_position=target, target_orientation=tq)
        if ok and getattr(self, "_arm_idx", None) is None:
            self._arm_idx = act.joint_indices
        idx = getattr(self, "_arm_idx", None)
        if idx is None:
            idx = np.arange(7)
        cands, sols = [], []
        if ok:
            q_a = np.asarray(act.joint_positions, dtype=float).reshape(-1)[:7]
            sols.append(q_a)
            if float(np.max(np.abs(q_a - q_cur))) <= IK_JUMP_MAX:
                cands.append(q_a)
        phi = math.atan2(float(target[1]), float(target[0]))
        R = _quat_to_mat(tq)
        psi = math.atan2(R[1][1], R[0][1]) - math.pi / 2      # hand yaw: the fingers open along +y at yaw 0
        d = _wrap(psi - phi)
        seeds = []
        for q7 in (float(q_cur[6]), math.pi / 4 + d, math.pi / 4 - d):
            seed = np.array(READY_Q, dtype=float)
            seed[0] = float(np.clip(phi, -2.85, 2.85))
            seed[6] = float(np.clip(q7, -2.85, 2.85))
            seeds.append(seed)
        for seed in seeds:
            try:
                q_b, okb = self.ik._kinematics_solver.compute_inverse_kinematics(self.ik._ee_frame, target, tq, warm_start=seed)
            except Exception:  # noqa: BLE001
                okb = False
            if okb:
                q_b = np.asarray(q_b, dtype=float).reshape(-1)[:7]
                sols.append(q_b)
                if float(np.max(np.abs(q_b - q_cur))) <= IK_JUMP_MAX:
                    cands.append(q_b)
        partial = not cands
        if partial:
            if not sols:
                return None, idx, False
            cands = sols
        Rt = np.asarray(R, dtype=float)

        def accurate(q):
            """Does the candidate reach the pose within 4 mm / 0.03 rad? At the rim of the
            workspace the unwound posture may fall short where a wound one still reaches: the
            solver's closest-feasible answer is then not good enough and the other candidate wins."""
            try:
                fp, fr = self.ik._kinematics_solver.compute_forward_kinematics(self.ik._ee_frame, np.asarray(q, dtype=float))
            except Exception:  # noqa: BLE001
                return True
            fr = np.asarray(fr, dtype=float)
            ang = math.acos(max(-1.0, min(1.0, (float(np.trace(fr.T @ Rt)) - 1.0) / 2.0)))
            return float(np.linalg.norm(np.asarray(fp, dtype=float) - target)) < 0.004 and ang < 0.03

        def wound(q):
            return abs(_wrap(float(q[0]) - phi)) + abs(float(q[2])) + 0.5 * abs(float(q[4]))

        best = min(cands, key=lambda q: (not accurate(q), wound(q)))
        if partial:
            delta = best - q_cur
            return q_cur + delta * min(1.0, IK_UNWIND_STEP / max(float(np.max(np.abs(delta))), 1e-9)), idx, False
        return best, idx, True

    # -- state ---------------------------------------------------------------------------------
    def _update_states(self):
        np = self.np
        forces = None
        if self._contacts is not None:
            try:
                forces = self._contacts.get_contact_force_matrix()   # (N, 2, 3)
            except Exception:  # noqa: BLE001
                forces = None
        ee_pos = self._ee_pose()[0]
        for o in self.objects:
            if not o["present"]:
                o.update(pos=None, in_container=None, resting=False, contact=False, held=False, on_table=False,
                         state=("lost" if o.get("state") == "lost" else ("removed" if o["removed"] else "absent")))
                continue
            p, q = o["rp"].get_world_pose()
            v = o["rp"].get_linear_velocity()
            o["pos"], o["quat"] = [float(x) for x in p], [float(x) for x in q]
            o["speed"] = float(np.linalg.norm(v))
            o["resting"] = o["speed"] < REST_SPEED
            o["in_container"] = self._container_of_point(p)
            if p[2] < -0.2:            # fell off the table: lost for good
                self._park(o)
                o["removed"], o["frozen"], o["state"] = True, LOST_VALUE, "lost"
                o.update(in_container=None, on_table=False, contact=False, held=False)
                self.events.append({"t": self.t, "event": "fell_off_desk", "object": o["id"]})
                continue
            # the three detectors: finger contact (PhysX contact report), inside a container
            # (geometric, container frame), resting on the table top
            if forces is not None:
                per_finger = np.abs(forces[o["id"]]).reshape(2, -1).sum(axis=1)     # left, right finger link
                o["contact"] = bool(per_finger.sum() > 1e-4)
                pinched = bool(per_finger[0] > 1e-4 and per_finger[1] > 1e-4)
            else:
                o["contact"] = bool(np.linalg.norm(p - ee_pos) < 0.06)
                pinched = o["contact"]
            if o["contact"]:
                self.touched.add(o["id"])
            placed = o["resting"] and not o["contact"]          # released and still
            if placed:
                o["rest_z"] = float(p[2])
            # held: a real grasp, not a touch. Both fingers on the object and the object lifted
            # clear of its support, its centre at least HOLD_LIFT above where it last rested; a
            # finger brushing or pushing an object leaves it "transit"
            o["held"] = bool(pinched and o.get("rest_z") is not None and float(p[2]) - o["rest_z"] >= HOLD_LIFT)
            o["on_table"] = bool(placed and o["in_container"] is None and p[2] < ON_TABLE_Z_MAX)
            if o["in_container"] is not None and placed:
                o["state"] = "in_container"
            elif o["on_table"]:
                o["state"] = "on_table"
            elif o["held"]:
                o["state"] = "held"
            else:
                o["state"] = "transit"
            if o["on_table"] and (math.hypot(p[0], p[1]) < BASE_RESCUE or p[0] < DESK_X[0] - 0.02):
                # knocked to where no camera sees it and no hand reaches (beside the robot's base
                # or past the front edge of the reachable area): after a moment at rest the world
                # sets it back down on a free spot, like a nudge; its baseline is re-read once it rests
                o["base_zone_steps"] = o.get("base_zone_steps", 0) + 1
                if o["base_zone_steps"] >= BASE_ZONE_REST_STEPS:
                    spot = self._random_desk_spot()
                    if spot is not None:
                        self._spawn(o, (spot[0], spot[1], 0.0), on_desk=True)
                        o["rebase_t"] = self.t
                        o["base_zone_steps"] = 0
                        self.events.append({"t": self.t, "event": "disturb_nudged", "object": o["id"], "reason": "beside the base"})
                        continue
            else:
                o["base_zone_steps"] = 0
            if o.get("rebase_t") is not None and (placed or self.t - o["rebase_t"] >= REBASE_TIMEOUT):
                self.baseline[o["id"]] = self._value(o, require_rest=False)
                o["rebase_t"] = None

    def _is_target(self, o):
        """The requested object, from the step its request opens through the step it closes."""
        r = self.request
        return bool(r and r.get("object") == o["id"] and (r["status"] == "open" or r.get("closed_t") == self.t))

    def _value(self, o, require_rest=True):
        """v_o(t): +1 placed in its own container, -1 placed in a wrong one, 0 on the table, held
        or in flight. The requested object belongs to the red container only: +1 placed in red or
        held (both fingers on it, lifted clear of its support), -1 placed in any other container, 0
        on the table or in flight. "Placed" = resting and not touched by the fingers. Removed
        objects keep their last value; a lost object (off the table) is LOST_VALUE for the rest of
        the episode."""
        if o["removed"]:
            return o.get("frozen", 0)
        target = self._is_target(o)
        if target and o["present"] and o.get("held"):
            return 1
        if not o["present"] or o.get("in_container") is None:
            return 0
        if require_rest and (not o.get("resting") or o.get("contact")):
            return 0
        home = "red" if target else CONTAINER_OF_CLASS[o["cls"]]
        return 1 if o["in_container"] == home else -1

    def _world_events(self, mark):
        rng = self.rng
        ev = self.events
        # requests: the agent closes one by marking it (1 = delivered, 2 = no such object);
        # the grader then checks the physical truth. Nothing closes a request automatically
        # except its deadline.
        r = self.request
        if r and r["status"] != "open" and not r.get("restored") and r.get("closed_t", self.t) < self.t:
            # the step after a request closed, its object is an ordinary object again: back to the
            # baseline it had before the request named it
            r["restored"] = True
            if r["object"] is not None and "b_pre" in r:
                self.baseline[r["object"]] = r["b_pre"]
        if self.request and self.request["status"] == "open":
            r = self.request
            tgt = self.objects[r["object"]] if r["object"] is not None else None
            # Award the retrieval milestone only for a current hold, never for a
            # contact remembered from before this request. Each request gets it once.
            if tgt is not None and tgt["present"] and tgt["state"] == "held" and "held_t" not in r:
                r["held_t"] = self.t
                ev.append({"t": self.t, "event": "request_first_hold", "request": r["id"]})
            if mark == 1:
                delivered = tgt is not None and tgt["present"] and tgt["state"] == "in_container" and tgt["in_container"] == "red"
                r["status"], r["closed_t"] = ("delivered" if delivered else "claim_wrong"), self.t
                if delivered:
                    tgt["frozen"] = 1                 # keeps its value for the rest of the episode
                    self._park(tgt); tgt["removed"] = True
                ev.append({"t": self.t, "event": "request_" + r["status"], "request": r["id"]})
            elif mark == 2:
                r["status"], r["closed_t"] = ("absent_correct" if tgt is None else "absent_wrong"), self.t
                ev.append({"t": self.t, "event": "request_" + r["status"], "request": r["id"]})
            elif self.t >= r["deadline_t"]:
                r["status"], r["closed_t"] = "expired", self.t
                ev.append({"t": self.t, "event": "request_expired", "request": r["id"]})
            if r["status"] != "open":
                self.requests_done.append(r)
        if (self.request is None or self.request["status"] != "open") and self.t >= self.next_request:
            self._open_request()
            self.request_slot += 1
            self.next_request = self._request_time(self.request_slot)
        # arrivals
        if self.t >= self.next_arrival:
            self.next_arrival = self.t + int(rng.exponential(ARRIVAL_MEAN_STEPS)) + 50
            loose = sum(1 for o in self.objects if o["present"] and o["in_container"] is None)
            present = sum(1 for o in self.objects if o["present"])
            if loose < LOOSE_CAP and present < PRESENT_CAP:
                # never the model of the open request: a second instance would make "the X"
                # ambiguous, and an arrival of a model requested as absent would turn that
                # request wrong on its own
                req = self.request["model"] if (self.request and self.request["status"] == "open") else None
                models = [m for m in CLASS_OF if self._present_count(m) < MAX_INSTANCES_PER_MODEL and m != req]
                m = rng.choice(models)
                o, spot = self._free_instance(m), self._random_desk_spot()
                if o is not None and spot is not None:
                    self._spawn(o, (spot[0], spot[1], 0.0), on_desk=True)
                    ev.append({"t": self.t, "event": "arrival", "object": o["id"], "model": m})
        # container service: a full class container is emptied by the worker, but not while it holds
        # the open request's object (that would leave the request without a right answer: the object
        # is gone, yet "no such object" is wrong for a request that named a present object)
        req_obj = self.request["object"] if (self.request and self.request["status"] == "open") else None
        for name in ("gray", "lightblue", "blue"):
            inside = [o for o in self.objects if o["present"] and o["state"] == "in_container" and o["in_container"] == name]
            if req_obj is not None and any(o["id"] == req_obj for o in inside):
                continue
            if len(inside) >= CONTAINER_CAP:
                for o in inside:
                    o["frozen"] = self._value(o)
                    self._park(o); o["removed"] = True
                ev.append({"t": self.t, "event": "container_cleared", "container": name, "count": len(inside)})
        # disturbances
        if self.t >= self.next_disturb:
            self.next_disturb = self.t + int(rng.exponential(DISTURB_MEAN_STEPS)) + 300
            cands = [o for o in self.objects if o["present"] and o["resting"] and (self.request is None or self.request.get("object") != o["id"])]
            if cands:
                o = cands[int(rng.integers(len(cands)))]
                if rng.random() < 0.5:
                    wrong = [b for b in ("gray", "lightblue", "blue") if b != CONTAINER_OF_CLASS[o["cls"]]]
                    b = wrong[int(rng.integers(len(wrong)))]
                    px, py = self._container_point(b, rng.uniform(-0.03, 0.03), rng.uniform(-0.02, 0.02))
                    self._spawn(o, (px, py, CONTAINER_RIM_Z + 0.12))
                    self.baseline[o["id"]] = -1       # it will rest misfiled; fixing it is the gain available
                    ev.append({"t": self.t, "event": "disturb_misplaced", "object": o["id"], "container": b})
                else:
                    # moved across the desk to another free spot (never into a gap between two
                    # containers, where no hand fits), set down the way arrivals are
                    loose = [c for c in cands if c["in_container"] is None]
                    o = loose[int(rng.integers(len(loose)))] if loose else None
                    spot = self._random_desk_spot() if o is not None else None
                    if spot is not None:
                        self._spawn(o, (spot[0], spot[1], 0.0), on_desk=True)
                        o["rebase_t"] = self.t        # baseline re-read once it comes to rest
                        ev.append({"t": self.t, "event": "disturb_nudged", "object": o["id"]})

    def _request_time(self, k):
        """Opening step of the k-th request: a random time in its window, early enough that its
        deadline passes before the next window starts."""
        return k * REQUEST_SLOT_STEPS + 200 + int(self.rng.integers(0, REQUEST_SLOT_STEPS - REQUEST_DEADLINE_STEPS - 400 + 1))

    def _open_request(self):
        rng = self.rng
        present = [o for o in self.objects if o["present"]]
        unique = [o for o in present if self._present_count(o["model"]) == 1]
        absent_models = [m for m in CLASS_OF if self._present_count(m) == 0]
        if (rng.random() < REQUEST_ABSENT_P or not unique) and absent_models:
            m, obj = absent_models[int(rng.integers(len(absent_models)))], None
        else:
            o = unique[int(rng.integers(len(unique)))]
            m, obj = o["model"], o["id"]
        self.request = {"id": len(self.requests_done) + 1, "text": "the " + DISPLAY_NAME[m], "model": m,
                        "object": obj, "opened_t": self.t, "deadline_t": self.t + REQUEST_DEADLINE_STEPS, "status": "open"}
        self.events.append({"t": self.t, "event": "request_opened", "request": self.request["id"], "absent": obj is None})
        self._rebase_target()

    def _rebase_target(self):
        """A request changes its object's home to the red container: its baseline becomes its value
        under that rule (e.g. -1 resting in its class pile), so an untouched scene still scores 0;
        the previous baseline returns the step after the request closes."""
        r = self.request
        if r and r["object"] is not None:
            o = self.objects[r["object"]]
            r["b_pre"] = self.baseline.get(o["id"], 0)
            self.baseline[o["id"]] = self._value(o, require_rest=False)

    def force_request(self, model=None, absent=False):
        """Development aid: open a request now. `model` names the YCB model (default: a random
        present unique one, or a random absent one when `absent`). Marks and the deadline then
        behave exactly as in a real episode; the next random request waits its normal time."""
        if self.rng is None:
            self.reset(0)
        if self.request is not None and self.request["status"] == "open":
            raise ValueError("a request is already open")
        present_models = {o["model"] for o in self.objects if o["present"]}
        if model is None:
            pool = [m for m in CLASS_OF if (m not in present_models) == bool(absent)]
            if absent is False:
                pool = [m for m in pool if self._present_count(m) == 1]
            if not pool:
                raise ValueError("no suitable model")
            model = pool[int(self.rng.integers(len(pool)))]
        if model not in CLASS_OF:
            raise ValueError("unknown model %r" % model)
        obj = next((o["id"] for o in self.objects if o["present"] and o["model"] == model), None)
        self.request = {"id": len(self.requests_done) + 1, "text": "the " + DISPLAY_NAME[model], "model": model,
                        "object": obj, "opened_t": self.t, "deadline_t": self.t + REQUEST_DEADLINE_STEPS, "status": "open"}
        ev = {"t": self.t, "event": "request_opened", "request": self.request["id"], "absent": obj is None, "forced": True}
        self.events.append(ev)
        self._update_states()
        self._rebase_target()
        obs, state = self.observe(), self.privileged()
        self.events.append(ev)        # the next `step` carries it too: that is where scorers read events
        return obs, state

    def lab(self, model, instance=0, pose=None, clear=True):
        """Development aid, available to everyone who runs this file: optionally clear the
        desk, then place one instance of `model` at `pose` (x, y, z[, yaw]) and let it settle; a
        yaw of null sets it down the way an arrival is (resting pose and yaw rule of `_desk_yaw`).
        The world state machine is paused (no arrivals, requests or disturbances) until the
        next `reset`. Returns an observation and the privileged state like `step`."""
        if self.rng is None:
            self.reset(0)
        if clear:
            for o in self.objects:
                if o["present"]:
                    self._park(o)
            self.request = None           # a request stays open when only one object is placed
        self.next_arrival = self.next_request = self.next_disturb = 10 ** 12
        if model is not None:
            cands = [o for o in self.objects if o["model"] == model]
            if not cands:
                raise ValueError("unknown model %r" % model)
            o = cands[min(int(instance), len(cands) - 1)]
            pose = pose or [0.5, 0.0, 0.06, 0.0]
            # never below the resting height: a centre lower than the half height starts the
            # object inside the desk, and PhysX throws it out
            yaw = pose[3] if len(pose) > 3 else 0.0
            self._spawn(o, (pose[0], pose[1], max(float(pose[2]), o["half_h"] + 0.004)), yaw=yaw, on_desk=yaw is None)
        self._settle_until_rest()
        for _ in range(RESET_WARMUP_FRAMES):   # RTX needs frames to converge after a teleport
            self.world.step(render=True)
        self._update_states()
        return self.observe(), self.privileged()

    def gains(self, new=None):
        """Development aid: read (and optionally set) the finger drive gains and max efforts.
        `new` = {"finger_kp": .., "finger_kd": .., "finger_effort": ..}."""
        np = self.np
        ctrl = self.franka.get_articulation_controller()
        n = self.franka.num_dof
        if new:
            kps, kds = ctrl.get_gains()
            kps, kds = np.array(kps, dtype=float), np.array(kds, dtype=float)
            for j in (n - 2, n - 1):
                if "finger_kp" in new: kps[j] = float(new["finger_kp"])
                if "finger_kd" in new: kds[j] = float(new["finger_kd"])
            ctrl.set_gains(kps=kps, kds=kds)
            if "finger_effort" in new:
                eff = np.array(ctrl.get_max_efforts(), dtype=float)
                eff[n - 2:] = float(new["finger_effort"])
                ctrl.set_max_efforts(eff)
        kps, kds = ctrl.get_gains()
        return {"kps": [float(x) for x in kps], "kds": [float(x) for x in kds], "max_efforts": [float(x) for x in ctrl.get_max_efforts()],
                "dof_names": list(self.franka.dof_names)}

    # -- outputs -------------------------------------------------------------------------------
    def observe(self, hires=None):
        np = self.np
        rgb, depth = {}, {}
        for name, c in self.cams.items():
            img = c.get_rgba()
            d = c.get_depth()
            if img is None or img.size == 0:
                img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 4), np.uint8)
            if d is None or d.size == 0:
                d = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32)
            rgb[name] = np.ascontiguousarray(img[..., :3]).astype(np.uint8)
            dm = self._depth_sensor(np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0), list(HIRES_CAMERAS).index(name),
                                    edges=name in CAMERA_NOMINAL) * 1000.0
            depth[name] = np.clip(dm, 0, 65535).astype(np.uint16)
        # the six PNG encodes run in parallel (PIL releases the GIL while compressing)
        jobs = [(name, "rgb", rgb[name], "RGB") for name in rgb] + [(name, "depth", depth[name], "I;16") for name in depth]
        hrgb, hdepth = {}, {}
        for name, (img, dm) in (hires or {}).items():
            jobs += [(name, "hrgb", img, "RGB"), (name, "hdepth", dm, "I;16")]
        for (name, kind, _, _), b64 in zip(jobs, _ENCODER.map(lambda j: _png_b64(j[2], j[3]), jobs)):
            {"rgb": rgb, "depth": depth, "hrgb": hrgb, "hdepth": hdepth}[kind][name] = b64
        jp = self.franka.get_joint_positions()
        jv = self.franka.get_joint_velocities()
        pos, quat = self._ee_pose()
        req = None
        if self.request and self.request["status"] == "open":
            req = {"text": self.request["text"], "deadline_t": self.request["deadline_t"]}
        obs = {"t": self.t, "request": req, "rgb": rgb, "depth": depth,
               "joint_pos": [float(x) for x in jp[:7]], "joint_vel": [float(x) for x in jv[:7]],
               "gripper_width": float(jp[7] + jp[8]),
               "ee_pose": [float(x) for x in pos] + [float(x) for x in quat]}
        if hrgb:
            obs["rgb_hires"], obs["depth_hires"] = hrgb, hdepth
        return obs

    def privileged(self):
        objs = []
        for o in self.objects:
            if not o["present"] and not o["removed"]:
                continue
            target = bool(self.request and self.request["status"] == "open" and self.request.get("object") == o["id"])
            objs.append({"id": o["id"], "model": o["model"], "cls": o["cls"], "present": o["present"],
                         "removed": o["removed"], "pos": o.get("pos"), "quat": o.get("quat"), "scale": o["scale"],
                         "role": "target" if target else "non_target", "state": o.get("state", "absent"),
                         "contact": bool(o.get("contact")), "held": bool(o.get("held")), "in_container": o.get("in_container"), "on_table": bool(o.get("on_table")),
                         "resting": bool(o.get("resting")),
                         "value": self._value(o), "baseline": self.baseline.get(o["id"], 0)})
        ev, self.events = self.events, []
        return {"t": self.t, "horizon": self.horizon, "objects": objs, "request": self.request,
                "requests_done": self.requests_done[-3:], "events": ev, "ik_ok": getattr(self, "_ik_ok", True), "ik_q": getattr(self, "_ik_q", None)}

    def info(self):
        np = self.np
        cams = {}
        for name, c in self.cams.items():
            if name in CAMERA_NOMINAL:          # the documented (nominal) pose, never this desk's mounting error
                pos, target = CAMERA_NOMINAL[name]
                p, q = pos, _look_at_wxyz(pos, target)
            else:
                p, q = c.get_world_pose()
            cams[name] = {"intrinsics": c.get_intrinsics_matrix().tolist(), "resolution": [IMAGE_SIZE, IMAGE_SIZE],
                          "world_pose_xyz_wxyz": [float(x) for x in p] + [float(x) for x in q],
                          "attached_to": "/World/Franka/panda_hand" if name == "eye_in_hand" else "world"}
        hcams = {name: {"intrinsics": c.get_intrinsics_matrix().tolist(), "resolution": list(HIRES_SIZE), "action_bit": HIRES_CAMERAS[name]}
                 for name, c in self.hcams.items()}
        return {"control_hz": CONTROL_HZ, "image_size": IMAGE_SIZE, "action_dim": ACTION_DIM,
                "hires": {"resolution": list(HIRES_SIZE), "action_index": ACTION_DIM, "cameras": hcams},
                "pos_step_max": POS_STEP_MAX, "rot_step_max": ROT_STEP_MAX, "horizon": self.horizon,
                "containers": {n: {"center_xy": [c[0], c[1]], "yaw": c[2], "nominal_center_xy": list(CONTAINER_NOMINAL[n]),
                             "inner_half_xy": list(CONTAINER_INNER_HALVES[n]), "floor_z": CONTAINER_FLOOR_Z,
                             "rim_z": CONTAINER_RIM_Z, "outer_xyz": list(CONTAINER_OUTERS[n]), "scale_xyz": list(CONTAINER_SCALES[n]),
                             "class": next((k for k, v in CONTAINER_OF_CLASS.items() if v == n), "requested")}
                         for n, c in self.containers.items()},
                "container_jitter_xy_yaw": list(CONTAINER_JITTER), "container_scale_xyz": list(CONTAINER_SCALE),
                "desk": {"x": list(DESK_X), "y": list(DESK_Y), "surface_z": 0.0, "robot_base_xyz": [0, 0, 0]},
                "classes": {k: [DISPLAY_NAME[m] for m in v] for k, v in CLASSES.items()},
                "cameras": cams}

    def close(self):
        self.app.close()


# ---------------------------------------------------------------------------------------
# helpers (no Isaac imports)
# ---------------------------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor
_ENCODER = ThreadPoolExecutor(max_workers=6)


def _png_b64(arr, mode):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _look_at_wxyz(pos, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (w, x, y, z) that points Isaac's world-axes camera (+X forward, +Y left,
    +Z up) from `pos` at `target`."""
    f = [t - p for t, p in zip(target, pos)]
    n = math.sqrt(sum(v * v for v in f)) or 1.0
    f = [v / n for v in f]
    l = [up[1] * f[2] - up[2] * f[1], up[2] * f[0] - up[0] * f[2], up[0] * f[1] - up[1] * f[0]]
    n = math.sqrt(sum(v * v for v in l)) or 1.0
    l = [v / n for v in l]
    u = [f[1] * l[2] - f[2] * l[1], f[2] * l[0] - f[0] * l[2], f[0] * l[1] - f[1] * l[0]]
    m00, m01, m02 = f[0], l[0], u[0]
    m10, m11, m12 = f[1], l[1], u[1]
    m20, m21, m22 = f[2], l[2], u[2]
    tr = m00 + m11 + m22
    if tr > 0:
        sq = math.sqrt(tr + 1.0) * 2
        return [0.25 * sq, (m21 - m12) / sq, (m02 - m20) / sq, (m10 - m01) / sq]
    if m00 > m11 and m00 > m22:
        sq = math.sqrt(1.0 + m00 - m11 - m22) * 2
        return [(m21 - m12) / sq, 0.25 * sq, (m01 + m10) / sq, (m02 + m20) / sq]
    if m11 > m22:
        sq = math.sqrt(1.0 + m11 - m00 - m22) * 2
        return [(m02 - m20) / sq, (m01 + m10) / sq, 0.25 * sq, (m12 + m21) / sq]
    sq = math.sqrt(1.0 + m22 - m00 - m11) * 2
    return [(m10 - m01) / sq, (m02 + m20) / sq, (m12 + m21) / sq, 0.25 * sq]


def _mat_to_wxyz(m):
    """3x3 rotation matrix (row-major, indexable) to a unit quaternion (w, x, y, z)."""
    m00, m01, m02 = float(m[0][0]), float(m[0][1]), float(m[0][2])
    m10, m11, m12 = float(m[1][0]), float(m[1][1]), float(m[1][2])
    m20, m21, m22 = float(m[2][0]), float(m[2][1]), float(m[2][2])
    tr = m00 + m11 + m22
    if tr > 0:
        sq = math.sqrt(tr + 1.0) * 2
        return [0.25 * sq, (m21 - m12) / sq, (m02 - m20) / sq, (m10 - m01) / sq]
    if m00 > m11 and m00 > m22:
        sq = math.sqrt(1.0 + m00 - m11 - m22) * 2
        return [(m21 - m12) / sq, 0.25 * sq, (m01 + m10) / sq, (m02 + m20) / sq]
    if m11 > m22:
        sq = math.sqrt(1.0 + m11 - m00 - m22) * 2
        return [(m02 - m20) / sq, (m01 + m10) / sq, 0.25 * sq, (m12 + m21) / sq]
    sq = math.sqrt(1.0 + m22 - m00 - m11) * 2
    return [(m10 - m01) / sq, (m02 + m20) / sq, (m12 + m21) / sq, 0.25 * sq]


def _rpy_to_quat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _quat_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


# ---------------------------------------------------------------------------------------
# stdio server
# ---------------------------------------------------------------------------------------
def serve():
    # Kit writes its own log lines to file descriptor 1 at the C level, so redirecting
    # sys.stdout is not enough: move the wire to a private descriptor and point fd 1 at
    # stderr before Isaac starts. Clients still skip any line that is not a JSON object.
    wire_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    out = os.fdopen(wire_fd, "w", buffering=1)
    env = MessyDeskEnv(headless=True)
    # Live score for the practice tools (the same EpisodeScorer the graders use); every `step`
    # and `reset` reply carries `score`, and `{"cmd": "score"}` adds the per-object values.
    # The agent's observation never carries it: this is the privileged side of the wire.
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from messy_desk_score import EpisodeScorer
    except Exception:  # noqa: BLE001 - the simulator works without the scorer next to it
        EpisodeScorer = None
    scorer = [None]

    def live():
        return scorer[0].live() if scorer[0] is not None else None
    out.write(json.dumps({"ok": True, "ready": True}) + "\n"); out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            cmd = msg.get("cmd")
            if cmd == "info":
                reply = {"ok": True, "info": env.info()}
            elif cmd == "reset":
                obs, state = env.reset(msg.get("seed", 0), msg.get("horizon"))
                scorer[0] = EpisodeScorer(env.horizon) if EpisodeScorer else None
                reply = {"ok": True, "obs": obs, "state": state, "score": live()}
            elif cmd == "step":
                obs, state = env.step(msg["action"])
                if scorer[0] is not None:
                    scorer[0].add_step(state)
                reply = {"ok": True, "obs": obs, "state": state, "score": live()}
            elif cmd == "lab":
                obs, state = env.lab(msg.get("model"), msg.get("instance", 0), msg.get("pose"), msg.get("clear", True))
                reply = {"ok": True, "obs": obs, "state": state, "score": live()}
            elif cmd == "request":
                obs, state = env.force_request(msg.get("model"), bool(msg.get("absent", False)))
                reply = {"ok": True, "obs": obs, "state": state, "score": live()}
            elif cmd == "score":
                objs = [{"id": o["id"], "model": o["model"], "role": o.get("role"), "state": o.get("state"), "in_container": o.get("in_container"),
                         "value": o.get("value"), "baseline": o.get("baseline")} for o in env.privileged()["objects"] if o.get("present") or o.get("removed")]
                reply = {"ok": True, "score": live(), "objects": objs, "request": env.request}
            elif cmd == "gains":
                reply = {"ok": True, "gains": env.gains(msg.get("set"))}
            elif cmd == "freecam":
                if env.freecam is None:
                    reply = {"ok": False, "error": "start with MESSY_DESK_FREECAM=WxH"}
                else:
                    if msg.get("pos") is not None:
                        env.set_freecam(msg["pos"], float(msg.get("yaw", 0.0)), float(msg.get("pitch", 0.0)))
                        if msg.get("frame", True):
                            # the camera renders on the next frame: render now (no physics step) so
                            # the picture is taken from the new pose, not the previous one
                            for _ in range(3):
                                env.world.render()
                    reply = {"ok": True, "jpeg": env.freecam_jpeg() if msg.get("frame", True) else None}
            elif cmd == "close":
                out.write(json.dumps({"ok": True}) + "\n"); out.flush()
                break
            else:
                reply = {"ok": False, "error": "unknown cmd %r" % cmd}
        except Exception as exc:  # noqa: BLE001
            reply = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        out.write(json.dumps(reply) + "\n"); out.flush()
    env.close()


if __name__ == "__main__":
    serve()
