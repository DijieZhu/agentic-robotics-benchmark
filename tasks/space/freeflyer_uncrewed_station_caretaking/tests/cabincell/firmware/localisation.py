"""Localisation pieces of the task firmware: the lighting's effect on the map localisation, the fiducial plates'
visibility (occlusion) gate, and the rail-relative mode of capture.

* ``CabinRichness`` -- the richness handed to ``spacekit.sensors.Localizer`` = weighted count of mapped feature points
  (the station's ``map_features.json``) in the forward camera's 120 deg / 0.3-6 m cone, facing the camera, not
  occluded (by the scene, or by the robot as the frames show it: ``RobotOccluder``), weight = min(1, lighting factor
  of the point's module (firmware.lighting.feature_factor: general / night / failed circuit) + the robot's lamps);
  evaluated at the sparse-map rate (spec loc.rates) with a hold in between; the lamp positions come from spec
  dev.lamp.positions.
* ``RobotOccluder`` -- the robot's own occluder in every camera-visibility test of the firmware (since 1.1.0): the
  maker's public mesh as the agent's frames render it, in the pose of the robot's bodies, with the frames' near clip.
  The objects are the renderer's (``cabincell.sensors_render``: the asset ``robot_author`` through its loader, added by
  its ``add_author_robot``; the same arrays, no copy), on one mocap body per robot body that carries one, in a second
  MuJoCo model beside the physics model (which is never changed); before a test its bodies take the poses (xpos,
  xquat) of the physics bodies in the data the test reads, the data the camera pose comes from.  A surface occludes
  only where its depth along the camera's optical axis is at least the frame recipe's near clip
  (``render.near_clip_m``): what the frames clip (the maker's own patches 2-2.5 mm in front of the lenses) does not
  occlude.  So at arm tilt 0 nothing of the robot is in any camera's view, with the arm tilted the maker's arm and
  fingers are, where the frames show them.  The reconstruction's display meshes (group 1 of the physics model) and the
  robot's collision proxies are transparent to every camera-visibility test.
* ``RayTester``, ``fiducial_check`` -- a marker point of a plate is seen by the
  aft wide camera when it is inside the 40 deg cone, the range and the incidence limit (``dock.fiducial_visible``),
  inside the delivered pinhole frame, and its ray is blocked neither by the robot as the frames show it
  (``RobotOccluder``) nor by the scene (walls, handrails, parts and items -- the held item included).  At least
  ``min_visible_markers`` such points: the plate is visible.  Changed: the robot's occluder was the
  reconstruction's display meshes (group 1 from 4.5 mm along the ray, among them a shroud at the aft wide camera that
  the project had modelled and windows in front of the forward cameras), which the agent's frames do not show.
* ``GatedFiducialLocalizer`` -- the simulation package's ``dock.DockRelativeLocalizer`` (consistent error model)
  whose fixes are gated by ``fiducial_check`` at the fiducial rate (spec loc.rates fiducial_Hz, held in between): a
  tick on which the plate is not visible gives no fiducial fix (the localiser's own no-view branch: inertial hold, then
  the map); the gate is computed inside the localiser.
* ``RailLocalizer`` -- the top of the stack; after a capture lock the estimate is bar-relative: the rail-relative
  estimate keeps the absolute error the estimate had at the lock (frozen) plus a slow inertial drift (relative errors
  cancel in the approach either way; the true pose would make the absolute pose exact, which is not what a
  bar-relative fix gives).
numpy + mujoco + spacekit only.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from spacekit.dock import DockRelativeLocalizer, fiducial_visible
from spacekit.rotations import quat_conj, quat_from_rotvec, quat_mul, quat_to_mat, rotvec_from_quat
from spacekit.sensors import true_state

CAMERA = dict(axis_body=(1.0, 0.0, 0.0), offset_body=(0.20, 0.0, 0.0), fov_deg=120.0, range_m=(0.3, 6.0), n_good=40,
              min_cos_incidence=0.15, occlusion_tol_m=0.10)                                   # [G]
LAMP_MODEL = dict(flux_lm=216.0, beam_full_deg=45.0, taper_to_deg=30.0, general_lux=150.0)   # dev.lamp; lux [G]
LAMP_NAMES = {"forward": "front", "aft": "aft"}                                             # API name -> spec key
AWC = "cam_aft_wide"
# The lens skip of the reconstruction's light appearance mesh, which closes the aft wide lens 2.0-3.1 mm in front of
# the camera.  No file of this folder uses it.
LENS_SKIP = 0.0045


# ----------------------------------------------------------------------------------------------- lighting
class CabinRichness:
    """Richness callable for ``spacekit.sensors.Localizer`` (module doc).

    features: dict(P (N,3), normal (N,3), module (N,)) of the mapped points (``station.mapfeat.from_json``);
    lighting: {module: 'general' | 'night' | 'off'}; light_factor: {state: factor}; lamp_positions: {'front' | 'aft':
    position in the body frame} (spec dev.lamp.positions); rate_hz: evaluation rate."""

    def __init__(self, model, data, robot_body: int, features: dict, lighting: dict, light_factor: dict,
                 lamp_positions: dict, rate_hz: float = 2.0, camera: dict | None = None, occlusion: bool = True,
                 occluder: "RobotOccluder | None" = None):
        self.model, self.data, self.body = model, data, int(robot_body)
        self.occluder = occluder
        root = int(model.body_rootid[self.body])
        self._robot_geom = np.asarray(model.body_rootid)[np.asarray(model.geom_bodyid)] == root
        self._gid1 = np.zeros(1, np.int32)
        self.n_robot_recasts = 0
        self.F = features
        self.cam = dict(CAMERA, **(camera or {}))
        self.modules = sorted(set(np.asarray(features["module"]).tolist()))
        self.lighting = {m: "general" for m in self.modules}
        self.lighting.update(lighting or {})
        self.light_factor = dict(light_factor)
        self.lamps = dict(front=False, aft=False)
        self.lamp_pos = {k: np.asarray(v, float) for k, v in lamp_positions.items()}
        self.lamp_axis = dict(front=np.array([1.0, 0.0, 0.0]), aft=np.array([-1.0, 0.0, 0.0]))
        self.occlusion = occlusion
        self.period = 1.0 / float(rate_hz)
        self._t_last = -1e9
        self._value = 1.0
        self.last = {}
        self._mod_idx = {m: np.asarray(features["module"]) == m for m in self.modules}
        I = LAMP_MODEL["flux_lm"] / (2 * math.pi * (1 - math.cos(math.radians(LAMP_MODEL["beam_full_deg"] / 2))))
        self.k_lamp = I / LAMP_MODEL["general_lux"]          # m^2: lamp illuminance relative to the general lighting
        self.cos_beam = math.cos(math.radians(LAMP_MODEL["beam_full_deg"] / 2))
        self.cos_taper = math.cos(math.radians(LAMP_MODEL["taper_to_deg"]))
        self.n_evals = 0

    def set_lighting(self, module: str, state: str) -> None:
        if state not in self.light_factor:
            raise ValueError(f"lighting state {state!r} not in {sorted(self.light_factor)}")
        self.lighting[module] = state
        self._t_last = -1e9

    def set_lamp(self, which: str, on: bool) -> None:
        self.lamps[LAMP_NAMES.get(which, which)] = bool(on)
        self._t_last = -1e9

    def ambient(self) -> np.ndarray:
        a = np.zeros(len(self.F["P"]))
        for m, idx in self._mod_idx.items():
            a[idx] = self.light_factor[self.lighting.get(m, "general")]
        return a

    def evaluate(self, pos, R) -> float:
        """Weighted feature count / n_good (clipped to 1) seen from body pose (pos, R)."""
        c = self.cam
        pos, R = np.asarray(pos, float), np.asarray(R, float)
        axis = R @ np.asarray(c["axis_body"], float)
        origin = pos + R @ np.asarray(c["offset_body"], float)
        P = self.F["P"]
        d = P - origin
        r = np.linalg.norm(d, axis=1)
        n = self.F["normal"]
        with np.errstate(invalid="ignore", divide="ignore"):
            u = d / r[:, None]
            cosv = u @ axis
            cos_inc = -np.einsum("ij,ij->i", np.nan_to_num(n), u)
        has_n = ~np.isnan(n[:, 0])
        vis = (r >= c["range_m"][0]) & (r <= c["range_m"][1]) & (cosv >= math.cos(math.radians(c["fov_deg"] / 2)))
        vis &= (~has_n) | (cos_inc >= c["min_cos_incidence"])
        idx = np.nonzero(vis)[0]
        if self.occlusion and len(idx):
            dirs = np.ascontiguousarray(u[idx]).reshape(-1)
            gid = np.zeros(len(idx), np.int32)
            dist = np.zeros(len(idx))
            mujoco.mj_multiRay(self.model, self.data, origin, dirs, None, 1, self.body, gid, dist, None, len(idx),
                               c["range_m"][1] + 1.0)
            tol = c["occlusion_tol_m"]
            ok = (dist < 0) | (dist >= r[idx] - tol)
            # the robot's own geoms of the physics model are transparent: re-cast past them
            for k in np.nonzero((~ok) & (gid >= 0) & self._robot_geom[np.maximum(gid, 0)])[0]:
                ok[k] = self._clear_past_robot(origin, u[idx[k]], float(dist[k]), float(r[idx[k]]) - tol)
            # the robot as the frames show it (the maker's mesh, the near clip along the camera axis)
            occ = self.occluder
            if occ is not None and ok.any():
                occ.sync(self.data)
                if occ.max_depth(origin, axis) >= occ.near:
                    for k in np.nonzero(ok)[0]:
                        if occ.blocks(origin, u[idx[k]], float(r[idx[k]]) - tol, axis):
                            ok[k] = False
            idx = idx[ok]
        w = self.ambient()[idx]
        for name, on in self.lamps.items():
            if not on or not len(idx):
                continue
            lp = pos + R @ self.lamp_pos[name]
            la = R @ self.lamp_axis[name]
            dl = P[idx] - lp
            rl = np.linalg.norm(dl, axis=1)
            ul = dl / rl[:, None]
            cone = np.clip((ul @ la - self.cos_taper) / max(self.cos_beam - self.cos_taper, 1e-9), 0.0, 1.0)
            nn = n[idx]
            ci = np.where(np.isnan(nn[:, 0]), 1.0, np.clip(-np.einsum("ij,ij->i", np.nan_to_num(nn), ul), 0.0, 1.0))
            w = w + self.k_lamp * cone * ci / np.maximum(rl, 0.05) ** 2
        w = np.minimum(w, 1.0)
        count = float(np.sum(w))
        val = min(1.0, count / float(c["n_good"]))
        self.n_evals += 1
        self.last = dict(count_weighted=count, n_visible=int(len(idx)), value=val)
        return val

    def _clear_past_robot(self, origin, u, s: float, reach: float) -> bool:
        """The scene along a ray whose first hit (at s) was one of the robot's geoms: re-cast past the robot's geoms
        (any group, the core body excluded as in the multi-ray); clear when nothing else is hit before ``reach``."""
        self.n_robot_recasts += 1
        for _ in range(16):
            s += 1e-4
            h = mujoco.mj_ray(self.model, self.data, origin + u * s, u, None, 1, self.body, self._gid1)
            if h < 0 or s + h >= reach:
                return True
            g = int(self._gid1[0])
            if not self._robot_geom[g]:
                return False
            s += h
        return True

    def __call__(self, pos, R, occlusion=None) -> float:
        t = float(self.data.time)
        if t - self._t_last >= self.period - 1e-9:
            self._value = self.evaluate(pos, R)
            self._t_last = t
        return self._value


# ----------------------------------------------------------------------------------------------- plate visibility
def project(K, size, cam_p, cam_R, pts):
    """True where world points project inside the delivered frame (MuJoCo camera frame: looks along -z, y up)."""
    X = (np.atleast_2d(pts) - cam_p) @ cam_R
    z = -X[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 2] + K[0, 0] * X[:, 0] / z
        v = K[1, 2] - K[1, 1] * X[:, 1] / z
    return (z > 1e-6) & (u >= 0) & (u < size[0]) & (v >= 0) & (v < size[1])


_OCCLUDER_MODELS: dict = {}


def occluder_model(author: dict | None = None) -> tuple:
    """(MjModel, robot body names, geom names by id) of the maker's mesh on mocap bodies named as the robot's bodies: the
    renderer's asset (``sensors_render.load_author``) added by the renderer's ``add_author_robot``.  Built once per
    process and asset."""
    from cabincell import sensors_render as SR
    A = author if author is not None else SR.load_author()
    key = A.get("dir") or id(A)
    if key not in _OCCLUDER_MODELS:
        bodies = sorted({o["rec"]["body"] for o in A["objects"]})
        spec = mujoco.MjSpec()
        spec.modelname = "robot_occluder"
        for nm in bodies:
            spec.worldbody.add_body(name=nm, mocap=True)
        SR.add_author_robot(spec, A, set(bodies))
        m = spec.compile()
        names = tuple(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom))   # by geom id
        _OCCLUDER_MODELS[key] = (m, tuple(bodies), names)
    return _OCCLUDER_MODELS[key]


class RobotOccluder:
    """The robot as the frames show it, for the ray tests of the camera-visibility rules (module doc).  ``model`` /
    ``robot_root``: the physics model and the robot's root body; ``near_clip_m``: the frame recipe's near clip
    (``render.near_clip_m``); ``author``: the loaded asset (default ``sensors_render.load_author()``)."""

    def __init__(self, model, robot_root: int, near_clip_m: float, author: dict | None = None):
        from cabincell import sensors_render as SR
        self.m, self.bodies, self.geom_names = occluder_model(author)
        self.d = mujoco.MjData(self.m)
        phys = []
        for nm in self.bodies:
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
            if b < 0 or int(model.body_rootid[b]) != int(robot_root):
                raise ValueError(f"the maker's object body {nm!r} is not a body of the robot in the physics model")
            phys.append(b)
        self.phys = np.array(phys, int)
        self.mocap = np.array([int(self.m.body_mocapid[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, nm)])
                               for nm in self.bodies], int)
        self.group = np.zeros(6, np.uint8)
        self.group[SR.GROUP_OTHER_VISUAL] = 1         # the group add_author_robot gives the maker's objects
        self.near = float(near_clip_m)
        if not self.near > 0:
            raise ValueError(f"near clip {near_clip_m!r}")
        self.gid = np.zeros(1, np.int32)
        self._key = None
        # every object's bounding box in the frame of its body, from the compiled vertices the rays are cast against
        # (tight along the body axes; the geom's own box is aligned with the mesh's principal axes and much looser)
        m = self.m
        lo, hi = [], []
        for g in range(m.ngeom):
            k = int(m.geom_dataid[g])
            V = np.asarray(m.mesh_vert[int(m.mesh_vertadr[k]):int(m.mesh_vertadr[k]) + int(m.mesh_vertnum[k])], float)
            Rg = np.zeros(9)
            mujoco.mju_quat2Mat(Rg, np.asarray(m.geom_quat[g], float))
            Vb = V @ Rg.reshape(3, 3).T + np.asarray(m.geom_pos[g], float)
            lo.append(Vb.min(axis=0))
            hi.append(Vb.max(axis=0))
        lo, hi = np.array(lo), np.array(hi)
        self._box_c, self._box_h = 0.5 * (lo + hi), 0.5 * (hi - lo) + 1e-6
        self._box_body = np.asarray(m.geom_bodyid, int)
        self.n_sync = 0

    def sync(self, data) -> None:
        """The maker's objects in the pose of the physics bodies of ``data`` (no work when the poses are unchanged)."""
        xp = data.xpos[self.phys]
        xq = data.xquat[self.phys]
        k = self._key
        if k is not None and np.array_equal(k[0], xp) and np.array_equal(k[1], xq):
            return
        self.d.mocap_pos[self.mocap] = xp
        self.d.mocap_quat[self.mocap] = xq
        mujoco.mj_kinematics(self.m, self.d)
        self._key = (xp.copy(), xq.copy())
        self.n_sync += 1

    def hit(self, p0, u, axis) -> tuple:
        """(distance along the unit ray u from p0, geom of this model) of the first maker's surface whose depth along
        ``axis`` (the camera's optical axis, unit) is at least the near clip; (None, -1) when there is none."""
        c = float(np.dot(u, axis))
        if c <= 0.0:
            return None, -1
        s0 = self.near / c                            # the ray reaches the near plane here
        h = mujoco.mj_ray(self.m, self.d, np.asarray(p0, float) + u * s0, u, self.group, 1, -1, self.gid)
        if h < 0:
            return None, -1
        return s0 + h, int(self.gid[0])

    def blocks(self, p0, u, reach: float, axis) -> bool:
        """The ray p0 + t u is blocked by the robot before ``reach`` (call ``sync`` first)."""
        h, _ = self.hit(p0, u, axis)
        return h is not None and h < reach

    def max_depth(self, p0, axis) -> float:
        """Upper bound of the depth along ``axis`` from p0 of any maker's surface (the objects' boxes in their bodies'
        frames, at the synced poses): below the near clip no ray from p0 can be blocked by the robot."""
        a = np.asarray(axis, float)
        R = self.d.xmat[self._box_body].reshape(-1, 3, 3)          # world = R @ body
        c = self.d.xpos[self._box_body] + np.einsum("gij,gj->gi", R, self._box_c)
        ext = np.einsum("gj,gj->g", np.abs(np.einsum("gij,i->gj", R, a)), self._box_h)
        return float(np.max((c - np.asarray(p0, float)) @ a + ext))

    def info(self) -> dict:
        return dict(bodies=list(self.bodies), objects=len(self.geom_names), triangles=int(self.m.nmeshface), near_clip_m=self.near)


class RayTester:
    """Ray visibility from a robot camera (module doc): the robot as the frames show it (``occluder``, a
    ``RobotOccluder``: the maker's mesh, nothing nearer than the near clip along the camera's optical axis ``axis``)
    and the scene (groups 1-3: walls, handrails, parts and items; every geom of the robot's own bodies in the physics
    model -- the reconstruction's display meshes, the collision proxies -- is passed by re-casting past it)."""

    def __init__(self, model, robot_root: int, occluder: RobotOccluder):
        self.m = model
        self.robot_root = int(robot_root)
        self.body_root = np.asarray(model.body_rootid)
        self.geom_body = np.asarray(model.geom_bodyid)
        self.gid = np.zeros(1, np.int32)
        self.g_scene = np.zeros(6, np.uint8)
        self.g_scene[[1, 2, 3]] = 1
        self.occluder = occluder

    def _robot_geom(self, g):
        return int(self.body_root[self.geom_body[g]]) == self.robot_root

    def clear(self, d, p0, p1, skip_geoms=(), tol: float = 0.002, axis=None) -> bool:
        """The ray from the camera at p0 to p1 is blocked neither by the robot (``axis``: the camera's optical axis,
        unit) nor by the scene within ``tol`` of p1 (geoms in ``skip_geoms`` clear it)."""
        if axis is None:
            raise TypeError("RayTester.clear needs the camera's optical axis (the frames' near clip is along it)")
        m = self.m
        vec = np.asarray(p1, float) - np.asarray(p0, float)
        L = float(np.linalg.norm(vec))
        u = vec / L
        occ = self.occluder
        occ.sync(d)
        if occ.blocks(p0, u, L - tol, np.asarray(axis, float)):
            return False
        s = 0.0
        for _ in range(8):
            dist = mujoco.mj_ray(m, d, np.asarray(p0, float) + u * s, u, self.g_scene, 1, -1, self.gid)
            if dist < 0 or s + dist >= L - tol:
                return True
            g = int(self.gid[0])
            if g in skip_geoms:
                return True
            if self._robot_geom(g):
                s += dist + 1e-4
                continue
            return False
        return True


class PlateCamera:
    """The aft wide camera as the plate gate sees it: pose from the robot body (rigid mount), delivered pinhole K and
    frame size, the cone / range / incidence limits."""

    def __init__(self, model, robot_body: int, K, size, half_fov_deg: float, range_m=(0.05, 1.5), max_incidence_deg=75.0,
                 cam: str = AWC):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        if cid < 0:
            raise KeyError(cam)
        d = mujoco.MjData(model)
        mujoco.mj_kinematics(model, d)
        mujoco.mj_camlight(model, d)
        Rb = d.xmat[robot_body].reshape(3, 3)
        self.body = int(robot_body)
        self.off = Rb.T @ (d.cam_xpos[cid] - d.xpos[robot_body])
        self.R_body_cam = Rb.T @ d.cam_xmat[cid].reshape(3, 3)
        self.axis_body = -self.R_body_cam[:, 2]
        self.K = np.asarray(K, float)
        self.size = (int(size[0]), int(size[1]))
        self.half_fov = float(half_fov_deg)
        self.range = tuple(range_m)
        self.max_inc = float(max_incidence_deg)

    def pose(self, data):
        Rb = data.xmat[self.body].reshape(3, 3)
        return data.xpos[self.body] + Rb @ self.off, Rb @ self.R_body_cam


def fiducial_check(data, cam: PlateCamera, tester: RayTester, pts, normal, skip_geoms=(), min_points: int = 3) -> dict:
    """Visibility of a plate's marker points from the aft wide camera (module doc).  visible = >= min_points points
    pass every test; strict = all pass."""
    cp, cR = cam.pose(data)
    axis = -cR[:, 2]
    P = np.atleast_2d(np.asarray(pts, float))
    v = P - cp
    dist = np.linalg.norm(v, axis=1)
    cos = (v @ axis) / np.maximum(dist, 1e-9)
    inc = (-v @ np.asarray(normal, float)) / np.maximum(dist, 1e-9)
    cone = ((dist >= cam.range[0]) & (dist <= cam.range[1]) & (cos >= math.cos(math.radians(cam.half_fov)))
            & (inc >= math.cos(math.radians(cam.max_inc))))
    img = project(cam.K, cam.size, cp, cR, P)
    ok = cone & img
    clear = np.zeros(len(P), bool)
    for i in np.nonzero(ok)[0]:
        clear[i] = tester.clear(data, cp, P[i], skip_geoms, axis=axis)
    ok &= clear
    vis = False
    if int(ok.sum()) >= min_points:
        vis, _ = fiducial_visible(cp, axis, P[ok], normal, cam.half_fov, cam.range, cam.max_inc, min_points)
    return dict(visible=bool(vis), strict=bool(ok.all()), n_ok=int(ok.sum()), n_cone=int(cone.sum()), n_img=int(img.sum()),
                n_cone_img=int((cone & img).sum()), dist=float(dist.min()))


class GatedFiducialLocalizer(DockRelativeLocalizer):
    """``DockRelativeLocalizer`` whose fixes need the plate to be visible by ``fiducial_check`` (module doc).
    gate: dict(model, cam (PlateCamera), tester (RayTester), min_points, rate_hz); the plate's own geoms are named in
    the reference (``skip``).  ``visible`` is the last gate result; ``hidden_since`` the time the plate was last lost
    (None while it is seen)."""

    def __init__(self, rng: np.random.Generator, base=None, gate: dict | None = None, **kw):
        super().__init__(rng, base=base, **kw)
        self.gate = gate
        self.skip = ()
        self.visible = None
        self.hidden_since = None
        self._t_gate = -1e9
        self.gate_log = []
        self.n_gate_hidden = 0
        self.n_gate = 0

    def reference(self, name: str, points, normal, cam_offset_body, cam_axis_body, skip=()) -> None:
        super().reference(name, points, normal, cam_offset_body, cam_axis_body)
        self.skip = tuple(skip)
        self.visible = None
        self.hidden_since = None
        self._t_gate = -1e9

    def release(self) -> None:
        super().release()
        self.visible = None
        self.hidden_since = None

    def check(self, data) -> dict:
        G = self.gate
        return fiducial_check(data, G["cam"], G["tester"], self.ref["points"], self.ref["normal"], self.skip, G["min_points"])

    def update(self, t, model, data, body: int, occlusion=None) -> dict:
        if self.ref is None or self.gate is None:
            return super().update(t, model, data, body, occlusion)
        if t - self._t_gate >= 1.0 / self.gate["rate_hz"] - 1e-9:
            self._t_gate = t
            c = self.check(data)
            self.visible = bool(c["visible"])
            self.n_gate += 1
            if not self.visible:
                self.n_gate_hidden += 1
        if self.visible:
            self.hidden_since = None
            return super().update(t, model, data, body, occlusion)
        if self.hidden_since is None:
            self.hidden_since = float(t)
        saved = self.ref["points"]
        self.ref["points"] = saved[:0]                    # no marker point seen: the localiser's no-view branch
        try:
            return super().update(t, model, data, body, occlusion)
        finally:
            self.ref["points"] = saved

    @property
    def mode(self) -> str:
        """'FIDUCIAL' while the estimate is fiducial-relative (fix or inertial hold), else 'MAP'."""
        return "FIDUCIAL" if (self.ref is not None and str(self.last_source).startswith("fiducial")) else "MAP"


# ----------------------------------------------------------------------------------------------- rail mode
class RailLocalizer:
    """Top of the localiser stack (module doc).  ``base`` keeps running underneath; ``engage()`` (at a capture lock)
    switches to the rail-relative estimate, ``release_rail()`` back.  The rail estimate is the true pose with the
    estimate's error at the lock (frozen) plus an inertial drift ``hold_drift`` [m^2/s]; reported std ``rail_sigma``."""

    def __init__(self, base, rng: np.random.Generator, hold_drift: float = 1e-6, rail_sigma_m: float = 0.005,
                 rail_sigma_rad: float = 0.01):
        self.base, self.rng, self.q = base, rng, float(hold_drift)
        self.rail = False
        self.e0_pos, self.e0_att, self.e = np.zeros(3), np.zeros(3), np.zeros(3)
        self.dt = 1.0 / 62.5
        self.sig_p, self.sig_a = float(rail_sigma_m), float(rail_sigma_rad)
        self.last_base = None
        self.last = None
        self.engagements = 0

    def __getattr__(self, k):            # reference() / release() / ref / last_source ... of the base
        return getattr(self.__dict__["base"], k)

    def engage(self, model, data, body: int) -> None:
        pos, quat, _, _ = true_state(model, data, body)
        b = self.last_base
        if b is None:
            self.e0_pos, self.e0_att = np.zeros(3), np.zeros(3)
        else:
            self.e0_pos = np.asarray(b["pos"], float) - pos
            self.e0_att = rotvec_from_quat(quat_mul(quat_conj(quat), np.asarray(b["quat"], float)))
        self.e = np.zeros(3)
        self.rail = True
        self.engagements += 1

    def release_rail(self) -> None:
        self.rail = False

    @property
    def mode(self) -> str:
        if self.rail:
            return "RAIL"
        m = getattr(self.base, "mode", "MAP")
        return m if isinstance(m, str) else "MAP"

    def update(self, t, model, data, body, occlusion=None):
        b = self.base.update(t, model, data, body, occlusion)
        self.last_base = b
        if not self.rail:
            self.last = b
            return b
        pos, quat, vel, om = true_state(model, data, body)
        self.e = self.e + self.rng.normal(0.0, math.sqrt(self.q * self.dt), 3)
        out = dict(pos=pos + self.e0_pos + self.e, quat=quat_mul(quat, quat_from_rotvec(self.e0_att)),
                   vel=vel + self.rng.normal(0.0, 0.001, 3), omega=om + self.rng.normal(0.0, 2e-4, 3),
                   cov_pos=np.eye(3) * self.sig_p ** 2, cov_att=np.eye(3) * self.sig_a ** 2, confidence=0, richness=1.0,
                   pos_error=float(np.linalg.norm(self.e0_pos + self.e)), att_error=float(np.linalg.norm(self.e0_att)),
                   source="rail")
        self.last = out
        return out


def lamp_positions(robot_values: dict) -> dict:
    """{'front' | 'aft': body-frame position} from spec dev.lamp.positions."""
    return {k: list(v) for k, v in robot_values["dev.lamp.positions"]["value"].items()}
