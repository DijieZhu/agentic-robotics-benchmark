"""Strict camera-only RPC boundary; no simulation object is passed to user code."""
import base64
import io
import math
import time

import numpy as np
from PIL import Image

from cell import DT


class RequestError(Exception):
    """Bad solver input, distinct from an environment/renderer failure."""


class Server:
    def __init__(self, cell, config, privileged=False):
        self.cell, self.cfg, self.privileged = cell, config, privileged
        # The physical tick also enforces this limit, so the privileged planner
        # cannot bypass the submitted program's simulation budget.
        cell.sim_cap_s = config["sim_cap_s"]
        self.target = cell.d.qpos[cell.arm_qadr].copy()
        self.command_q = self.target.copy()
        self.width = .085
        self.integral = np.zeros(7)
        self.frames = 0
        self.done = False
        self.session = None
        self.start = time.monotonic()
        self.calls = {}

    def state(self):
        c = self.cell
        wrench = c.d.sensordata.copy() + c.ft_bias + c.rng.normal(0, .025, 6)
        return dict(time_s=float(c.d.time), q=c.d.qpos[c.arm_qadr].tolist(),
                    dq=c.d.qvel[c.arm_dadr].tolist(), wrench_N_Nm=wrench.tolist(),
                    gripper_width_m=self.measured_width(), frames_remaining=self.cfg["camera_frames"]-self.frames,
                    done=self.done)

    def measured_width(self):
        c = self.cell
        a, b = (c.m.geom(x).id for x in ("left_finger_pad_2", "right_finger_pad_2"))
        return float(np.linalg.norm(c.d.geom_xpos[a]-c.d.geom_xpos[b])-.004)

    def tick(self):
        c = self.cell
        self.command_q += np.clip(self.target-self.command_q, -.008, .008)
        self.integral = np.clip(self.integral + (self.command_q-c.d.qpos[c.arm_qadr])*DT, -.03, .03)
        gain = c.m.actuator_gainprm[:7, 0]
        damping = -c.m.actuator_biasprm[:7, 2]
        ff = (c.d.qfrc_bias[c.arm_dadr] + damping*c.d.qvel[c.arm_dadr]) / gain
        c.d.ctrl[:7] = np.clip(self.command_q + ff + self.integral*.15,
                               c.m.actuator_ctrlrange[:7, 0], c.m.actuator_ctrlrange[:7, 1])
        r = math.hypot(.035465, .042039)
        theta = np.clip(math.acos(np.clip((self.width/2-(.035-.024003-.002))/r, -1, 1)) -
                        math.atan2(.042039, .035465), 0, .85)
        c.d.ctrl[c.m.actuator("gripper").id] = float(np.clip(theta * 100 / .333, 0, 255))
        c.tick()
        if c.tick_count * DT >= self.cfg["sim_cap_s"] - 1e-8 or c.violations:
            self.done = True

    def handle(self, request):
        if not isinstance(request, dict):
            raise RequestError("Request must be an object")
        op = request.get("op")
        allowed = {"describe", "state", "control", "step", "camera", "finish"}
        if self.privileged:
            allowed.add("reference_assemble")
        if not isinstance(op,str) or op not in allowed:
            raise RequestError("Unsupported operation")
        self.calls[op] = self.calls.get(op, 0) + 1
        if op == "describe":
            return dict(robot="xArm 7", tick_s=DT, cameras=["top", "wrist"],
                        max_step_ticks=self.cfg["max_ticks_per_step"], sim_budget_s=self.cfg["sim_cap_s"],
                        camera_budget=self.cfg["camera_frames"], gripper_range_m=[0, .085])
        if op == "state":
            return self.state()
        if op == "finish":
            self.done = True
            return self.state()
        if self.done:
            raise RequestError("Episode has finished")
        if op == "control":
            try:
                q = np.asarray(request.get("q"), dtype=float)
                w = float(request.get("gripper_width_m"))
            except (TypeError, ValueError, OverflowError):
                raise RequestError("Numeric joint targets and width required") from None
            if q.shape != (7,) or not np.isfinite(q).all() or not np.isfinite(w):
                raise RequestError("Finite seven-joint target and width required")
            ranges = self.cell.m.jnt_range[[self.cell.m.joint(f"joint{i}").id for i in range(1,8)]]
            if np.any(q < ranges[:, 0]) or np.any(q > ranges[:, 1]) or not 0 <= w <= .085:
                raise RequestError("Command exceeds joint or gripper limits")
            self.target, self.width = q, w
            return self.state()
        if op == "step":
            n = request.get("ticks")
            if type(n) is not int or not 1 <= n <= self.cfg["max_ticks_per_step"]:
                raise RequestError("Invalid step count")
            for _ in range(n):
                self.tick()
                if self.done:
                    break
            return self.state()
        if op == "camera":
            name = request.get("name")
            if name not in ("top", "wrist") or self.frames >= self.cfg["camera_frames"]:
                raise RequestError("Invalid camera or camera budget exhausted")
            if self.session is None:
                from rendering import CameraSession
                self.session = CameraSession(self.cell)
            raw = self.session.capture(name)
            self.frames += 1
            rgb = raw["rgb"]
            depth = raw["depth_m"].copy()
            good = np.isfinite(depth) & (depth >= .1) & (depth <= 4)
            good &= self.cell.rng.random(depth.shape) > .03
            # The depth encoding is registered millimetres; zero = invalid.
            encoded_depth = np.where(good, np.rint(depth*1000), 0).astype(np.uint16)
            def encode(a):
                f = io.BytesIO(); Image.fromarray(a).save(f, format="PNG")
                return base64.b64encode(f.getvalue()).decode("ascii")
            return dict(rgb_png=encode(rgb), depth_png=encode(encoded_depth), camera=name,
                        time_s=float(self.cell.d.time), depth_unit="m", invalid_depth=0)
        if op == "reference_assemble":
            from reference import assemble
            # Hidden state aids planning only. The reference uses the same c.tick
            # and safety watch; no success setter or object teleport exists here.
            assemble(self.cell, wall_cap_s=max(0., self.cfg["reference_wall_cap_s"] - (time.monotonic()-self.start)))
            return self.state()

    def close(self):
        if self.session is not None:
            self.session.close()
