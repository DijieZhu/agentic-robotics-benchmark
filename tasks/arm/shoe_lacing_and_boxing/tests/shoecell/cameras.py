"""Cameras of the cell and an offscreen rig with calibration.

Fixed cameras (world frame):
* ``overhead``: at (0, 0, 1.25) looking straight down.  Its MuJoCo frame is the identity rotation, so the
  image's right edge is world +x and the image's up edge is world +y (top rows of the image = +y side,
  the side the arms stand on).  fovy 58 deg, 640x480.
* ``front``: at (0, -1.05, 0.65) looking at (0, 0, 0.08) with world +z up in the image.  fovy 50, 640x480.
Wrist cameras (body frames):
* ``wrist_left`` on ``left/hand``: 0.05 behind the tcp along -z and 0.04 to the +x side (the side of
  the finger plates, so the gap between the fingers is seen edge-on), pitched to look at the point
  0.15 m ahead of the tcp along +z; image up = the finger closing axis (+y of the hand).  fovy 70, 480x360.
* ``wrist_right`` on ``right/hook``: 0.05 behind the shaft end along -z and 0.04 to the +y side (the
  hook's L lies in the xz plane and is seen in profile), looking 0.15 m ahead of the tcp; image up =
  tool +x (the foot direction).  fovy 70, 480x360.

MuJoCo camera frame: looks along its -z, +y up, +x right.  OpenCV frame (what :meth:`CameraRig.extrinsics`
returns): +x right, +y down, +z forward  ->  R_cv = R_mj diag(1, -1, -1).
Depth: mujoco.Renderer returns the linearised z-buffer, i.e. the PLANAR depth along the optical axis
(not the ray length), in metres.  :meth:`CameraRig.render` passes it through unchanged (float32 HxW).
The cell's rig (built with its scene) draws the colour with MuJoCo's Filament renderer on a render twin of the cell
in its factory hall, and the depth with the classic renderer on a depth twin with the same geometry plus the hall's
meshes (look.py); a rig without a scene draws both with the classic renderer on the model itself.
Intrinsics use integer pixel indices (u = column, v = row, row 0 at the top): a pixel's centre ray is
x/z = (u - cx) / fx with cx = W/2 - 0.5, cy = H/2 - 0.5, fx = fy = (H/2) / tan(fovy/2).

Runtime dependencies: mujoco + numpy (look.py: Pillow and MuJoCo's Filament module).
"""

import math

import mujoco
import numpy as np

from .control import mat_to_quat, quat_to_mat

__all__ = ["CAMERA_SPECS", "lookat_quat", "build_cameras", "CameraRig"]

CAMERA_SPECS = {
    "overhead": {"size": (640, 480), "fovy": 58.0, "pos": (0.0, 0.0, 1.30), "quat": (1.0, 0.0, 0.0, 0.0), "parent": "world"},
    "front": {"size": (640, 480), "fovy": 52.0, "pos": (0.10, -1.10, 0.70), "lookat": (-0.05, 0.0, 0.05), "up": (0.0, 0.0, 1.0),
              "parent": "world"},
    "side": {"size": (640, 480), "fovy": 52.0, "pos": (-1.25, 0.05, 0.65), "lookat": (-0.30, 0.0, 0.05), "up": (0.0, 0.0, 1.0),
             "parent": "world"},
    "wrist_left": {"size": (480, 360), "fovy": 70.0, "parent": "left/hand", "offset": (0.04, 0.0, -0.05), "ahead": 0.15,
                   "up": (0.0, 1.0, 0.0)},
    "wrist_right": {"size": (480, 360), "fovy": 70.0, "parent": "right/hook", "offset": (0.0, 0.04, -0.05), "ahead": 0.15,
                    "up": (1.0, 0.0, 0.0)},
}


def lookat_quat(pos, target, up=(0.0, 0.0, 1.0)):
    """wxyz quaternion of a MuJoCo camera at pos looking at target with `up` as the image-up hint."""
    pos = np.asarray(pos, float)
    f = np.asarray(target, float) - pos
    f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        raise ValueError("up hint parallel to the viewing direction")
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.column_stack([r, u, -f])
    return mat_to_quat(R)


def build_cameras(spec, info, specs=CAMERA_SPECS):
    """Add the four cameras to the cell spec.  Returns {name: {size, fovy, parent, pos, quat}}."""
    out = {}
    for name, c in specs.items():
        if c["parent"] == "world":
            body = spec.worldbody
            pos = np.asarray(c["pos"], float)
            quat = np.asarray(c["quat"], float) if "quat" in c else lookat_quat(pos, c["lookat"], c["up"])
        else:
            body = spec.body(c["parent"])
            side = "left" if c["parent"].startswith("left/") else "right"
            tcp = np.asarray(info["arms"][side]["tcp_local"], float)
            pos = tcp + np.asarray(c["offset"], float)
            target = tcp + np.array([0.0, 0.0, c["ahead"]])
            quat = lookat_quat(pos, target, c["up"])
        body.add_camera(name=name, pos=pos.tolist(), quat=quat.tolist(), fovy=float(c["fovy"]))
        out[name] = {"size": tuple(c["size"]), "fovy": float(c["fovy"]), "parent": c["parent"], "pos": tuple(pos.tolist()),
                     "quat": tuple(quat.tolist())}
    return out


class CameraRig(object):
    """Offscreen RGB + depth rendering of the named cameras with pinhole calibration.

    sizes: {camera name: (width, height)}; one mujoco.Renderer per distinct size.
    The model's visual.global_.offwidth/offheight must be >= every size.
    scene: the cell's physics.Scene: colour and depth come from its render twins (look.AgentLook).
    """

    def __init__(self, model, sizes=None, scene=None):
        self.model = model
        self.sizes = dict(sizes) if sizes is not None else {k: v["size"] for k, v in CAMERA_SPECS.items()}
        self._renderers = {}
        self._option = mujoco.MjvOption()
        self._option.geomgroup[4] = 0   # hide the marker sites (group 4)
        self.scene = scene
        self._look = None

    def _renderer(self, name):
        w, h = self.sizes[name]
        key = (int(w), int(h))
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.model, height=key[1], width=key[0])
            self._renderers[key] = r
        return r

    def render(self, data, name):
        """(rgb uint8 HxWx3, depth float32 HxW planar z-depth in metres) of camera `name` at the data state."""
        if self.scene is not None:
            if self._look is None:
                from .look import AgentLook
                self._look = AgentLook(self.scene, self.sizes)
            return self._look.render(data, name)
        r = self._renderer(name)
        r.disable_depth_rendering()
        r.update_scene(data, camera=name, scene_option=self._option)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        depth = r.render().astype(np.float32)
        r.disable_depth_rendering()
        return rgb, depth

    def intrinsics(self, name):
        """dict(fx, fy, cx, cy, width, height) for integer pixel indices (see module docstring)."""
        w, h = self.sizes[name]
        fovy = math.radians(float(self.model.cam_fovy[self.model.camera(name).id]))
        fy = 0.5 * h / math.tan(0.5 * fovy)
        return {"fx": fy, "fy": fy, "cx": 0.5 * w - 0.5, "cy": 0.5 * h - 0.5, "width": int(w), "height": int(h)}

    def extrinsics(self, data, name):
        """4x4 T_world_cam in the OpenCV convention (x right, y down, z forward) at the data state."""
        cid = self.model.camera(name).id
        R_mj = data.cam_xmat[cid].reshape(3, 3)
        R_cv = R_mj @ np.diag([1.0, -1.0, -1.0])
        T = np.eye(4)
        T[:3, :3] = R_cv
        T[:3, 3] = data.cam_xpos[cid]
        return T

    def backproject(self, name, u, v, z, T_world_cam):
        """World point of pixel (u, v) at planar depth z through T_world_cam."""
        k = self.intrinsics(name)
        x = (float(u) - k["cx"]) * z / k["fx"]
        y = (float(v) - k["cy"]) * z / k["fy"]
        p = np.array([x, y, z, 1.0])
        return (T_world_cam @ p)[:3]

    def project(self, name, data, point_world):
        """(u, v, z) of a world point in camera `name` (float pixel indices, planar depth)."""
        T = self.extrinsics(data, name)
        R = T[:3, :3]
        pc = R.T @ (np.asarray(point_world, float) - T[:3, 3])
        k = self.intrinsics(name)
        u = k["cx"] + k["fx"] * pc[0] / pc[2]
        v = k["cy"] + k["fy"] * pc[1] / pc[2]
        return float(u), float(v), float(pc[2])

    def describe(self, specs=None, data=None):
        """The public calibration: per camera the size, the exact intrinsics, and for the fixed cameras the exact
        pose (OpenCV convention, world frame); for the wrist cameras the body they ride on and their exact mount
        (OpenCV pose in that body's frame)."""
        specs = specs or CAMERA_SPECS
        out = {}
        for name in self.sizes:
            c = specs[name]
            k = self.intrinsics(name)
            cid = self.model.camera(name).id
            R_mj = quat_to_mat(np.asarray(self.model.cam_quat[cid], float))
            T = np.eye(4)
            T[:3, :3] = R_mj @ np.diag([1.0, -1.0, -1.0])
            T[:3, 3] = np.asarray(self.model.cam_pos[cid], float)
            e = {"width": k["width"], "height": k["height"], "fx": k["fx"], "fy": k["fy"], "cx": k["cx"], "cy": k["cy"],
                 "fovy_deg": float(self.model.cam_fovy[cid]), "depth": "planar z along the optical axis",
                 "convention": "OpenCV: x right, y down, z forward; pixel (u, v) = (column, row), row 0 at the top"}
            if c["parent"] == "world":
                e["mount"] = "fixed"
                e["T_world_cam"] = T.tolist()
            else:
                e["mount"] = c["parent"]
                e["T_body_cam"] = T.tolist()
                e["note"] = "T_world_cam = T_world_body(%s) @ T_body_cam; the body pose follows from the arm's joint angles" % c["parent"]
            out[name] = e
        return out

    def truth(self, data):
        """Every camera's current T_world_cam (development only)."""
        return {name: self.extrinsics(data, name).tolist() for name in self.sizes}

    def close(self):
        """Free the GL contexts (and the render twins)."""
        if self._look is not None:
            self._look.close()
            self._look = None
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:
                pass
        self._renderers = {}
