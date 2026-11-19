"""The kitchen simulator of the recovery task: a RoboCasa PandaOmron kitchen rebuilt from (task, seed, split),
an action-prefix replay that restores a failure state, the observation/action format of the wire, cameras on request,
and the success check. No learned model, no grading logic: the practice environment and the grader use this same file.

Determinism: RoboCasa is run exactly as in the original rollouts (gym wrapper, seeded construction with the
order-preserving Counter.get_reset_regions shim), BLAS/numba/threads are pinned by the task's run scripts and cameras
are only rendered on request, off the physics path.
"""
import gc
import json
import math
import os
import random
import time

# Pinned numerics, set before numpy loads (they only take effect then; the image and the grader also set them): one thread,
# OpenBLAS kernel fixed, numpy's AVX-512 loops off, numba's target fixed. A different BLAS kernel changes the replayed
# rescues (measured), so every host must run the same kernels.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
os.environ.setdefault("NUMBA_CPU_NAME", "haswell")
os.environ.setdefault("NUMBA_CPU_FEATURES", "+avx2,+fma,+sse4.2")
os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR AVX512FP16")

import numpy as np  # noqa: E402

CAMERA_KEYS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
# Jobs whose object must also stand upright to count: RoboCasa's own check_obj_upright (default threshold, 15 degrees)
# on top of the job's success check. A tipped mug under a dispenser is not a finished coffee setup.
UPRIGHT_JOBS = ("CoffeeSetupMug",)
CAM_BITS = {"robot0_agentview_left": 1, "robot0_agentview_right": 2, "robot0_eye_in_hand": 4}
DEPTH_BIT = 8
IMG_SIZE = 256


def _quat_xyzw_to_axis_angle(q):
    q = np.asarray(q, float).reshape(-1)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.zeros(3, np.float32)
    q = q / n
    if q[3] < 0:
        q = -q
    xyz = q[:3]
    s = np.linalg.norm(xyz)
    if s < 1e-12:
        return np.zeros(3, np.float32)
    return (xyz / s * (2.0 * np.arctan2(s, np.clip(q[3], -1.0, 1.0)))).astype(np.float32)


def build_env(task, seed, split):
    """Seeded construction exactly as the recording collector's make_env, EGL-only tweaks left out."""
    import gymnasium as gym
    import robocasa  # noqa: F401  (registers the environments)
    import robocasa.models.fixtures.counter as counter
    if "set" not in vars(counter):
        counter.set = lambda iterable=(): dict.fromkeys(iterable)
    gc.collect()
    gc.disable()
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # not in the image; keep parity if ever present
        torch.manual_seed(seed)
    except Exception:
        pass
    env = gym.make("robocasa/" + task, split=split, seed=seed)
    obs, _info = env.reset(seed=seed)
    return env, obs


class Kitchen:
    """One restored kitchen run. `step` applies an action in the XR-1 12-D layout through the same conversion as the
    gym wrapper's step, with no image rendering in between; `observation` builds the wire dict with images on request."""

    def __init__(self, task, seed, split="target", prefix=None):
        self.task, self.seed, self.split = task, int(seed), split
        t0 = time.monotonic()
        self.env, gym_obs = build_env(task, self.seed, split)
        self.raw = self.env.unwrapped.env
        self.build_s = time.monotonic() - t0
        self.instruction = gym_obs["annotation.human.task_description"]
        for name in CAMERA_KEYS:                     # cameras off the physics path: rendered only on request
            key = name + "_image"
            if key in self.raw._observables:
                self.raw._observables[key].set_enabled(False)
                self.raw._observables[key].set_active(False)
        self.t_since_start = 0
        self.prefix_len = 0
        self.frames_rendered = 0
        self.sim_s = 0.0
        self.render_s = 0.0
        ctx = self.raw.sim._render_context_offscreen
        try:
            ctx.gl_ctx.make_current()
            self.raw.sim.model.vis.quality.offsamples = 0
            ctx.con.free()
            ctx._set_mujoco_context_and_buffers()
        except Exception:
            pass
        self.arm_qadr = [self.raw.sim.model.get_joint_qpos_addr(j) for j in self.raw.robots[0].robot_joints]
        self.obs = None
        if prefix is not None:
            self.replay(prefix)
        self.obs = self.raw._get_observations(force_update=True)

    # ------------------------------------------------------------------ stepping
    def _env_action(self, action):
        """The 12-D XR-1 action as the gym wrapper turns it into the controller action vector."""
        a = np.asarray(action, np.float32).copy()
        action_dict = {"action.end_effector_position": a[0:3], "action.end_effector_rotation": a[3:6],
                       "action.gripper_close": a[6:7], "action.base_motion": a[7:11], "action.control_mode": a[11:12]}
        action_dict = self.env.unwrapped.key_converter.unmap_action(action_dict)
        out = []
        for robot in self.raw.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            act = np.zeros(cc.action_limits[0].shape)
            for part_name in cc.part_controllers:
                start, end = cc._action_split_indexes[part_name]
                act[start:end] = action_dict.pop(f"{pf}{part_name}")
            from robosuite.controllers.composite.composite_controller import HybridMobileBase
            if isinstance(cc, HybridMobileBase):
                act[-1] = action_dict.pop(f"{pf}base_mode")
            out.append(act)
        return np.concatenate(out)

    def step(self, action):
        t0 = time.monotonic()
        raw_obs, _reward, _done, _info = self.raw.step(self._env_action(action))
        self.obs = raw_obs
        self.sim_s += time.monotonic() - t0
        self.t_since_start += 1
        return raw_obs

    def integration_state(self):
        """The MuJoCo integration state (mjSTATE_INTEGRATION) of the current simulation."""
        import mujoco
        m, d = self.raw.sim.model._model, self.raw.sim.data._data
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        out = np.zeros(mujoco.mj_stateSize(m, spec), np.float64)
        mujoco.mj_getState(m, d, out, spec)
        return out

    def set_integration_state(self, state):
        """Overwrite the physical state with a recorded one, so a restored state does not depend on the host's
        round-off over the prefix replay (the prefix still leaves the controllers and fixtures in their recorded mode)."""
        import mujoco
        m, d = self.raw.sim.model._model, self.raw.sim.data._data
        mujoco.mj_setState(m, d, np.asarray(state, np.float64), mujoco.mjtState.mjSTATE_INTEGRATION)
        self.obs = self.raw._get_observations(force_update=True)   # no mj_forward: an extra one changes the next steps

    def replay(self, prefix):
        """Restore the failure state: replay the recorded action prefix (float32, Nx12), bit-exact against the recording."""
        for a in np.asarray(prefix, np.float32):
            self.step(a)
        self.prefix_len = len(prefix)
        self.t_since_start = 0

    # ------------------------------------------------------------------ cameras
    def render(self, cam_mask):
        """{camera: rgb uint8 HxWx3}, plus {camera: depth float32 metres} when DEPTH_BIT is set in cam_mask."""
        t0 = time.monotonic()
        out, sim = {}, self.raw.sim
        for name in CAMERA_KEYS:
            if not (cam_mask & CAM_BITS[name]):
                continue
            if cam_mask & DEPTH_BIT:
                rgb, depth = sim.render(camera_name=name, width=IMG_SIZE, height=IMG_SIZE, depth=True)
                out[name] = rgb[::-1].copy()
                near, far = self.depth_range()
                d = depth[::-1].copy()
                out[name + "_depth_m"] = (near / (1.0 - d * (1.0 - near / far))).astype(np.float32)
            else:
                out[name] = sim.render(camera_name=name, width=IMG_SIZE, height=IMG_SIZE, depth=False)[::-1].copy()
            self.frames_rendered += 1
        self.render_s += time.monotonic() - t0
        return out

    def depth_range(self):
        m = self.raw.sim.model
        extent = m.stat.extent
        return float(m.vis.map.znear * extent), float(m.vis.map.zfar * extent)

    # ------------------------------------------------------------------ observation
    def proprio(self):
        o = self.obs
        state = {
            "ee_pos_base": [float(x) for x in o["robot0_base_to_eef_pos"]],
            "ee_rotvec_base": [float(x) for x in _quat_xyzw_to_axis_angle(o["robot0_base_to_eef_quat"])],
            "gripper_qpos": [float(x) for x in o["robot0_gripper_qpos"]],
            "base_pos_world": [float(x) for x in o["robot0_base_pos"]],
            "base_rotvec_world": [float(x) for x in _quat_xyzw_to_axis_angle(o["robot0_base_quat"])],
            "arm_q": [float(self.raw.sim.data.qpos[adr]) for adr in self.arm_qadr[:7]],
        }
        return state

    def wrist_ft(self):
        raw = self.raw
        model = raw.sim.model._model
        data = raw.sim.data._data
        values = {}
        for i in range(model.nsensor):
            name = model.sensor(i).name
            if name in ("gripper0_right_force_ee", "gripper0_right_torque_ee"):
                adr, dim = int(model.sensor_adr[i]), int(model.sensor_dim[i])
                values[name] = data.sensordata[adr:adr + dim].tolist()
        return [*values["gripper0_right_force_ee"], *values["gripper0_right_torque_ee"]]

    def success(self):
        """The job's own RoboCasa success check; for the jobs in UPRIGHT_JOBS the object must also be upright
        (robocasa.utils.object_utils.check_obj_upright: the x and y Euler angles of its body below 15 degrees)."""
        if not self.raw._check_success():
            return False
        if self.task in UPRIGHT_JOBS:
            from robocasa.utils import object_utils
            return bool(object_utils.check_obj_upright(self.raw, "obj"))
        return True

    def physical_digest(self):
        import hashlib
        qpos = np.asarray(self.raw.sim.data.qpos, np.float64)
        qvel = np.asarray(self.raw.sim.data.qvel, np.float64)
        return hashlib.sha256(qpos.tobytes()).hexdigest(), hashlib.sha256(qvel.tobytes()).hexdigest()

    def close(self):
        try:
            self.env.close()
        finally:
            self.obs = None


# ---------------------------------------------------------------------- wire observation helpers
import base64 as _b64


def _encode_array(a, dtype):
    a = np.asarray(a)
    return {"dtype": dtype, "shape": list(a.shape), "b64": _b64.b64encode(a.tobytes()).decode("ascii")}


def camera_intrinsics(raw, name, size=IMG_SIZE):
    from robosuite.utils.camera_utils import get_camera_intrinsic_matrix
    return np.asarray(get_camera_intrinsic_matrix(raw.sim, name, size, size)).tolist()


def camera_pose_world(raw, name):
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    return np.asarray(get_camera_extrinsic_matrix(raw.sim, name)).tolist()


def wire_observation(kitchen, t, cam_mask):
    """The JSON observation for step t: proprioception, wrist wrench, robot-mounted camera poses and the images asked
    for by the previous reply (camera bit mask, bit 8 adds depth)."""
    obs = {"t": int(t), "proprio": kitchen.proprio(), "wrist_ft": [float(x) for x in kitchen.wrist_ft()],
           "camera_pose_world": {c: camera_pose_world(kitchen.raw, c) for c in CAMERA_KEYS}, "images": {}}
    if cam_mask & 7:
        imgs = kitchen.render(cam_mask)
        for name in CAMERA_KEYS:
            if name in imgs:
                entry = {"rgb": _encode_array(imgs[name], "uint8")}
                d = imgs.get(name + "_depth_m")
                if d is not None:
                    mm = np.clip(np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0, 0, 65000).astype(np.uint16)
                    entry["depth_mm"] = _encode_array(mm, "uint16")
                obs["images"][name] = entry
    return obs


def kitchen_map(raw, roles=None):
    """Coarse fixture map in the world frame, one row per fixture: type; `center`, the fixture body's origin (RoboCasa's
    `pos`); `box_center`, the centre of its bounding box (the origin plus RoboCasa's `origin_offset` turned by the
    yaw; equal to `center` for a fixture without one); `size_xyz`, the box's full extents along the fixture's own x, y
    and z (RoboCasa's `size`: width, depth, height); `yaw_rad`. No parts (handles, knobs, buttons, racks), no object
    positions, no model names. `roles` marks the fixture a job names."""
    out = []
    names = roles or {}
    for fx_name, fx in sorted(raw.fixtures.items()):
        try:
            pos = [float(x) for x in np.asarray(fx.pos)]
            size = [float(x) for x in np.asarray(fx.size)]
            yaw = float(getattr(fx, "rot", 0.0))
            off = np.asarray(getattr(fx, "origin_offset", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
        except Exception:
            continue
        c, s_ = math.cos(yaw), math.sin(yaw)
        box = [pos[0] + c * off[0] - s_ * off[1], pos[1] + s_ * off[0] + c * off[1], pos[2] + off[2]]
        row = {"type": type(fx).__name__, "center": pos, "box_center": [float(x) for x in box], "size_xyz": size,
               "yaw_rad": yaw}
        if fx_name in names:
            row["role"] = names[fx_name]
        out.append(row)
    return out
