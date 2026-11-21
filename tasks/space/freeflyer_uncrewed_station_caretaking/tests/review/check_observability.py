"""check_observability.py -- the pre-registered per-site observability check of the caretaking task (the observability
rules of the hidden set).  Maintainers' tool: the site selection drops a candidate that fails and records why.
Never part of the grade, never named to the agent.

The rule: no detector and no pilot decides anything here.  Every row is a geometric
predicate (viewpoints, pixel sizes, occlusion rays, segmentation of the rendered frame) or a statistic of the rendered
frame (brightness, clipping, colour contrast) at the site's own lighting with the robot's lamps and the cameras' auto
exposure, rendered by the cell's renderer (cabincell.sensors_render: the agent's frames).  The state is the cell's
state at mission second 0 (the site package's settled state after the cell's pre-roll); the robot is placed
kinematically in the renderer's twin only (the physics data is never written).

    python3 -I -B tests/review/check_observability.py --site S.site.tar.xz [--site ...] --out DIR [--frames DIR]
                                                       [--look-root DIR] [--jobs N]

Rows (THRESHOLDS; a site passes when every row passes):
  V2 / V1   the blocked inlet at the evidence distances 2.0 m and 1.0 m: some viewpoint whose robot is in the keep-in
            (the firmware's stowed zone map), >= 0.35 m from the station's collision geometry, the forward imaging
            camera (FIC) aimed at the inlet centre, incidence <= 60 deg, and in the FIC frame: the inlet frame inside
            the image, its opening's short side >= 40 px and long side >= 80 px, <= 20 % of 20 rays to the frame
            points blocked (the cell's evidence-ray rule itself, the firmware's RayTester with the camera's optical
            axis: the station, the robot as the frames show it, items not on this inlet); and in the
            rendered frame (site lighting, forward lamp on, auto exposure, noise): mean luma of the inlet + its items
            >= 40 DN, <= 40 % of those pixels clipped (a lamp-lit pale item in a dark cabin may blow out in part: the
            camera meters the whole frame), the blocker's mean colour >= 20 DN (Euclidean RGB) from the open
            (uncovered) part of the opening (the frame when less than 200 px of the opening are open)
  R2 / R1   ranking among the inlets of the blocked module: each inlet's covered share seen by the FIC (items' pixels /
            the projected opening's pixels, segmentation of the frame at that inlet's own best viewpoint); the
            blocked inlet's share >= 0.30 and >= every other inlet's + 0.20, at 2 m (first look) and 1 m (confirmation)
  I         every item (the blocker, each pouch) in the 1 m frame of its inlet: >= 2000 px of it, its mean luma >=
            40 DN, each of its grasp bars' silhouette >= 20 px long
  B         every grasp feature (the blocker's strap and loop, each pouch's handle): a pose with the aft depth camera
            (ADC) 0.45-0.60 m (contract world.observability.bar_view_range_m) from the bar centre -- looking at it from
            within 45 deg of the bar's face normal (8 azimuths, 4 rolls), or one of the site's own pre-grasp poses
            (palm aligned) -- with the arm deployed (tilt 0) and the robot free on the arm-deployed zone map, from which the whole
            bar is seen: 9 points along its face toward the camera inside the frame (2 px margin) and the depth range,
            each ray's first collidable hit on the bar, and >= 20 depth pixels on the bar in the ray-cast frame
Outputs: <out>/<site_id>.obs.json (every row's numbers, the viewpoints, frame sha256s), <out>/<site_id>.obs.out.

Versions: 1.0.0 kept its own copy of the evidence-ray rule (the reconstruction's display meshes of the
twin from 4.5 mm along the ray, then the twin's collidable scene).  1.1.0: the rule is the firmware's
own RayTester (firmware 1.1.0: the maker's mesh with the near clip along the optical axis, the physics model's scene
groups 1-3), called with the camera's axis on a copy of the physics state; nothing else changed (thresholds and
predicates as frozen in 1.0.0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
if TESTS not in sys.path:
    sys.path.insert(0, TESTS)

THRESHOLDS = dict(
    view_distances_m=(2.0, 1.0), max_incidence_deg=60.0, incidence_step_deg=5.0, min_clear_m=0.35,
    max_viewpoints_rendered=3, inlet_short_px_min=40.0, inlet_long_px_min=80.0, occluded_max=0.20,
    region_luma_min_dn=40.0, region_clip_max=0.40, clip_level=254, blocker_contrast_min_dn=20.0,
    rank_cover_min=0.30, rank_margin=0.20, item_px_min=2000, item_luma_min_dn=40.0, bar_px_min_fic=20.0,
    adc_distances_m=(0.45, 0.50, 0.55, 0.60), adc_incidences_deg=(0.0, 15.0, 30.0, 45.0),
    adc_rolls_deg=(0.0, 90.0, 180.0, 270.0), adc_max_evaluated=48, bar_samples=9,
    bar_margin_px=2.0, adc_bar_px_min=20, adc_ray_tol_m=0.004)
SIDE_POINTS = (-0.8, -0.4, 0.0, 0.4, 0.8)
CHECK_VERSION = "1.1.0"            # 1.1.0: the evidence rays are the firmware's RayTester


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def luma(px) -> np.ndarray:
    x = np.asarray(px, np.float32)
    return x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722


# ============================================================================================== geometry
class Scene:
    """The cell (state at mission second 0) and the renderer's twin of one site, with the robot placed kinematically
    in the twin."""

    def __init__(self, site_path: str, look_root: str | None = None):
        import mujoco

        from cabincell import sensors_render as SR
        from cabincell import server as SV
        from spacekit import gripper as GR
        from spacekit.robot import FreeFlyerAdapter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.srv = SV.CabinCellServer(limits=dict(settle_after_exit_s=1.0))
            self.srv.reset(site_path)
        c = self.srv.cell
        self.cell = c
        self.site = c.site
        self.fr = SR.FrameRenderer(c.site, look_root=look_root, seed=int(c.seed), values=c.runtime["values"])
        for mod, st in c.svc.lighting_effective().items():
            self.fr.set_lighting(mod, st)
        self.lighting = dict(self.fr.lighting)
        self.m, self.d = self.fr.model, self.fr.data
        self.fr.sync(c.model, c.data)
        self.q0 = self.d.qpos.copy()
        self.ad = FreeFlyerAdapter(self.m)
        self.core = self.ad.body
        self.robot_bodies = set(np.flatnonzero(np.asarray(self.m.body_rootid) == self.m.body_rootid[self.core]).tolist())
        jid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.arm_j = [jid(n) for n in GR.ARM_NAMES["joints"]]
        self.finger_j = [jid(n) for n in GR.ARM_NAMES["fingers"]]
        self.feats = self.fr.feats
        self.zones = c.tf.zones
        self.world = c.world
        gb = np.asarray(self.m.geom_bodyid)
        self.item_geoms = {k: set(np.flatnonzero(np.isin(gb, it["bodies"])).tolist()) for k, it in c.world.items.items()}
        self.item_inlet = {k: it.get("inlet") for k, it in c.world.items.items()}
        names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(self.m.ngeom)]
        self.names = names
        # the evidence rays (``evidence_clear``) are the firmware's RayTester on the physics model: its geom ids differ
        # from the twin's (the maker's objects are added to the robot's bodies), see ``physics_geoms``
        self.tester = c.tf.fid_loc.gate["tester"]
        self._pd, self._pd_key = None, None
        self.inlets = {}
        for inl in self.feats["v02"]["inlets"]:
            F = c.world.faces[inl["id"]]
            n = unit(F["n"])
            a = unit(F["t1"])
            b = np.cross(n, a)
            fh = (0.5 * float(inl["frame_m"][0]), 0.5 * float(inl["frame_m"][1]))
            oh = (0.5 * float(inl["opening_m"][0]), 0.5 * float(inl["opening_m"][1]))
            pts = []
            for s in SIDE_POINTS:
                for sb in (-1.0, 1.0):
                    pts.append(F["c"] + a * s * fh[0] + b * sb * fh[1])
                    pts.append(F["c"] + b * s * fh[1] * 0.999 + a * sb * fh[0])
            own = {g for g, nm in enumerate(names) if nm.startswith(inl["id"] + "_")}
            on_it = set()
            on_it_phys = set()
            for k, iid in self.item_inlet.items():
                if iid == inl["id"]:
                    on_it |= self.item_geoms[k]
                    on_it_phys |= set(c.api.item_geoms[k])
            skip = self.physics_geoms(own | on_it)
            # the same geoms as the cell API names them (its inlet part by name, the items by body)
            if skip != frozenset(set(c.api.inlet_geo[inl["id"]]["own"]) | on_it_phys):
                raise RuntimeError(f"{inl['id']}: the twin's and the physics model's geoms of the inlet do not match")
            self.inlets[inl["id"]] = dict(id=inl["id"], label=inl["label"], module=inl["module"], c=np.asarray(F["c"], float),
                                          n=n, a=a, b=b, frame_half=fh, open_half=oh, pts=np.array(pts), own=own, items=on_it,
                                          ray_skip=skip)
        from firmware.bars import item_bars
        self.bars = item_bars(self.m, c.info)
        self.truth = c.site.get("truth") or {}
        self.blocked = self.truth.get("blocked_inlet") or c.world.items["blocker"]["inlet"]
        self.rng = np.random.default_rng([int(c.seed) & 0xFFFFFFFF, 0x4F4253])

    # -- robot placement ---------------------------------------------------------------------------------------
    def reset_state(self):
        import mujoco
        self.d.qpos[:] = self.q0
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)

    def place(self, p, R, tilt_deg: float | None = None, fingers_open: bool = True):
        """Robot body at (p, R) in the twin (arm at the firmware tilt when given, else as in the state)."""
        import mujoco
        from spacekit import gripper as GR
        from spacekit.rotations import mat_to_quat
        self.d.qpos[:] = self.q0
        self.ad.set_pose(self.d, np.asarray(p, float), np.asarray(mat_to_quat(np.asarray(R, float)), float))
        if tilt_deg is not None:
            GR.set_arm(self.m, self.d, self.arm_j, float(tilt_deg), 0.0)
            GR.set_fingers(self.m, self.d, self.finger_j, open_=fingers_open)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)

    def cam_offset(self, cam_name: str):
        """(offset in the core frame, R_core_cam) of a camera (rigid on the core) at the current twin state."""
        import mujoco
        cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        Rb = self.d.xmat[self.core].reshape(3, 3)
        return Rb.T @ (self.d.cam_xpos[cid] - self.d.xpos[self.core]), Rb.T @ self.d.cam_xmat[cid].reshape(3, 3)

    def camera_pose_robot(self, cam_name: str, cam_pos, look_dir, up_hint=(0.0, 0.0, 1.0), roll_deg: float = 0.0):
        """Robot body pose that puts camera ``cam_name`` at cam_pos looking along look_dir (MuJoCo camera -z), its
        image 'up' (camera +y) toward up_hint rotated by roll_deg about the optical axis."""
        off, Rbc = self._offsets[cam_name]
        zc = -unit(look_dir)
        u = np.asarray(up_hint, float) - (np.asarray(up_hint, float) @ zc) * zc
        if np.linalg.norm(u) < 1e-6:
            u = np.array([1.0, 0.0, 0.0]) - zc[0] * zc
        yc = unit(u)
        xc = np.cross(yc, zc)
        if roll_deg:
            r = math.radians(roll_deg)
            xc, yc = math.cos(r) * xc + math.sin(r) * yc, -math.sin(r) * xc + math.cos(r) * yc
        Rwc = np.column_stack([xc, yc, zc])
        Rwb = Rwc @ Rbc.T
        return np.asarray(cam_pos, float) - Rwb @ off, Rwb

    def prepare_offsets(self):
        self.reset_state()
        self._offsets = {nm: self.cam_offset(nm) for nm in ("cam_fwd_img", "cam_fwd_wide", "cam_aft_wide", "cam_aft_depth")}

    # -- rays ---------------------------------------------------------------------------------------------------
    def ray_first(self, o, u, maxd: float, skip=frozenset(), skip_robot: bool = True):
        """(distance, geom) of the first collidable geom along a ray (groups of the depth rays), passing geoms in
        ``skip`` and (by default) the robot's own; (None, -1) when nothing within maxd."""
        import mujoco
        from cabincell.sensors_render import DEPTH_GROUPS
        gg = np.zeros(6, np.uint8)
        gg[list(DEPTH_GROUPS)] = 1
        gid = np.array([-1], np.int32)
        o = np.asarray(o, float).copy()
        u = unit(u)
        tr = 0.0
        for _ in range(24):
            h = mujoco.mj_ray(self.m, self.d, o, u, gg, 1, -1, gid)
            if h < 0 or tr + h > maxd:
                return None, -1
            g = int(gid[0])
            if g in skip or (skip_robot and int(self.m.geom_bodyid[g]) in self.robot_bodies):
                o = o + u * (h + 1e-4)
                tr += h + 1e-4
                continue
            return tr + h, g
        return None, -1

    def physics_geoms(self, twin_geoms) -> frozenset:
        """The physics model's ids of twin geoms of the station and the items (the twin regroups geoms and adds visual
        objects, so the ids differ): a named geom by its name (geom names are unique), an unnamed one by its place among
        its body's geoms when the body has the same geoms in both models (same count); a robot geom has none."""
        import mujoco
        pm, tm = self.cell.model, self.m
        out = set()
        for g in twin_geoms:
            g = int(g)
            b = int(tm.geom_bodyid[g])
            nm = self.names[g]
            p = -1
            if b not in self.robot_bodies:
                if nm:
                    p = mujoco.mj_name2id(pm, mujoco.mjtObj.mjOBJ_GEOM, nm)
                elif b != 0 and int(pm.body_geomnum[b]) == int(tm.body_geomnum[b]):
                    p = int(pm.body_geomadr[b]) + g - int(tm.body_geomadr[b])
                    if mujoco.mj_id2name(pm, mujoco.mjtObj.mjOBJ_GEOM, p):
                        p = -1
            if p < 0 or int(pm.geom_bodyid[p]) != b:
                raise ValueError(f"twin geom {g} ({nm!r}, body {b}) has no counterpart in the physics model")
            out.add(int(p))
        return frozenset(out)

    def physics_state(self):
        """A scratch MjData of the cell's physics model with the robot where the twin has it (qpos and mocap copied
        from the twin, kinematics run; the twin has the physics model's bodies, joints and cameras).  The cell's own
        data is never written."""
        import mujoco
        m = self.cell.model
        if self._pd is None:
            self._pd = mujoco.MjData(m)
        key = (self.d.qpos.tobytes(), self.d.mocap_pos.tobytes(), self.d.mocap_quat.tobytes())
        if key != self._pd_key:
            self._pd.qpos[:] = self.d.qpos
            if m.nmocap:
                self._pd.mocap_pos[:] = self.d.mocap_pos
                self._pd.mocap_quat[:] = self.d.mocap_quat
            mujoco.mj_kinematics(m, self._pd)
            mujoco.mj_camlight(m, self._pd)
            self._pd_key = key
        return self._pd

    def evidence_clear(self, p0, p1, skip_geoms=(), tol: float = 0.002, axis=None) -> bool:
        """The cell's evidence-ray rule itself (since CHECK_VERSION 1.1.0): the firmware's RayTester
        (firmware.localisation; the instance the cell's API evidence -- the grader's D predicate -- and the plate gate
        use), on ``physics_state()``: the robot as the frames show it (the maker's mesh in the pose of the robot's
        bodies, nothing nearer than the frames' near clip along the camera's optical axis ``axis``, unit), then the
        physics model's scene (groups 1-3; the robot's own geoms passed); ``skip_geoms`` (geom ids of the PHYSICS
        model) clear the ray."""
        if axis is None:
            raise TypeError("evidence_clear needs the camera's optical axis (the frames' near clip is along it)")
        return bool(self.tester.clear(self.physics_state(), p0, p1, skip_geoms=skip_geoms, tol=tol, axis=axis))

    def clearance(self, p) -> float:
        """Distance from p to the station's collidable geometry along 26 directions (the robot transparent)."""
        best = 1e9
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    if (x, y, z) == (0, 0, 0):
                        continue
                    h, _ = self.ray_first(p, (x, y, z), 2.0)
                    if h is not None:
                        best = min(best, h)
        return best

    # -- projection ---------------------------------------------------------------------------------------------
    def optical(self, cam: str):
        import mujoco
        fm = self.fr.frames[cam]
        cid = self.fr.cam_id[cam]
        Rm = self.d.cam_xmat[cid].reshape(3, 3)
        return self.d.cam_xpos[cid].copy(), Rm @ np.diag([1.0, -1.0, -1.0]), fm

    def project(self, cam: str, pts):
        t, Ro, fm = self.optical(cam)
        P = (np.atleast_2d(np.asarray(pts, float)) - t) @ Ro
        K = fm["K"]
        z = P[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack([K[0, 0] * P[:, 0] / z + K[0, 2], K[1, 1] * P[:, 1] / z + K[1, 2]], 1)
        return uv, z, fm

    def polygon_mask(self, cam: str, corners):
        import cv2
        uv, z, fm = self.project(cam, corners)
        W, H = fm["width"], fm["height"]
        m = np.zeros((H, W), np.uint8)
        if np.any(z <= 1e-3):
            return m.astype(bool), uv, z
        cv2.fillPoly(m, [np.round(uv * 8).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=3)
        return m.astype(bool), uv, z


# ============================================================================================== rows
def inlet_corners(I, half):
    c, a, b = I["c"], I["a"], I["b"]
    return np.array([c + a * sa * half[0] + b * sb * half[1] for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def candidate_viewpoints(S: Scene, I: dict, D: float):
    """Robot poses with the FIC at distance D from the inlet centre, looking at it (incidence 0-60 deg in 5 deg
    steps, 8 azimuths), robot in the keep-in (stowed map) and >= min_clear from the station; sorted by incidence."""
    T = THRESHOLDS
    c, n, a, b = I["c"], I["n"], I["a"], I["b"]
    out = []
    mod = S.feats["modules"][I["module"]]
    # the cabin's overhead is -w (the inlets sit on the deck side of the rack faces, +w): the image up toward the
    # overhead, so the station's labels read upright (no predicate depends on the roll)
    w = np.asarray(mod["lat_w"], float)
    up = -w if abs(float(w @ n)) < 0.9 else -np.asarray(mod["lat_u"], float)
    for inc in np.arange(0.0, T["max_incidence_deg"] + 1e-9, T["incidence_step_deg"]):
        dirs = [n] if inc == 0 else [math.cos(math.radians(inc)) * n + math.sin(math.radians(inc)) * t
                                     for t in (a, -a, b, -b, unit(a + b), unit(a - b), unit(-a + b), unit(-a - b))]
        for dv in dirs:
            cam_pos = c + D * unit(dv)
            pb, Rb = S.camera_pose_robot("cam_fwd_img", cam_pos, c - cam_pos, up_hint=up)
            if S.zones.point(pb, arm=False) != "ok":
                continue
            cl = S.clearance(pb)
            if cl < T["min_clear_m"]:
                continue
            out.append(dict(incidence_deg=float(inc), cam_pos=cam_pos.tolist(), body_pos=pb.tolist(), R=Rb.tolist(), clear_m=float(cl)))
    out.sort(key=lambda v: (v["incidence_deg"], -v["clear_m"]))
    return out


def inlet_geometry(S: Scene, I: dict, cam: str = "FIC") -> dict:
    """FIC geometry of an inlet at the current twin pose: opening sides in px, frame inside, occlusion of 20 rays."""
    oc = inlet_corners(I, I["open_half"])
    fc = inlet_corners(I, I["frame_half"])
    uvo, zo, fm = S.project(cam, oc)
    uvf, zf, _ = S.project(cam, fc)
    W, H = fm["width"], fm["height"]
    inside = bool(np.all(zf > 0.01) and np.all((uvf[:, 0] >= 0) & (uvf[:, 0] < W) & (uvf[:, 1] >= 0) & (uvf[:, 1] < H)))
    long_px = float(0.5 * (np.linalg.norm(uvo[1] - uvo[0]) + np.linalg.norm(uvo[2] - uvo[3]))) if np.all(zo > 0) else 0.0
    short_px = float(0.5 * (np.linalg.norm(uvo[3] - uvo[0]) + np.linalg.norm(uvo[2] - uvo[1]))) if np.all(zo > 0) else 0.0
    t, Ro, _ = S.optical(cam)
    axis = Ro[:, 2]                                   # the optical axis (the cell API's evidence passes the same)
    skip = I.get("ray_skip")                          # (an inlet-like record of a caller: its twin geoms, translated)
    if skip is None:
        skip = S.physics_geoms(set(I["own"]) | set(I["items"]))
    blocked = sum(0 if S.evidence_clear(t, p, skip, axis=axis) else 1 for p in I["pts"])
    v = I["c"] - t
    dist = float(np.linalg.norm(v))
    inc = math.degrees(math.acos(max(-1.0, min(1.0, float(I["n"] @ (-v / dist))))))
    return dict(dist_m=dist, incidence_deg=inc, frame_inside=inside, long_px=long_px, short_px=short_px,
                occluded=blocked / float(len(I["pts"])))


def geometry_ok(g: dict) -> bool:
    T = THRESHOLDS
    return bool(g["frame_inside"] and g["short_px"] >= T["inlet_short_px_min"] and g["long_px"] >= T["inlet_long_px_min"]
                and g["occluded"] <= T["occluded_max"] and g["incidence_deg"] <= T["max_incidence_deg"] + 1e-6)


def frame_stats(S: Scene, I: dict, img, seg) -> dict:
    """Statistics of a rendered FIC frame at an inlet: the region = the projected frame polygon's pixels that show the
    inlet or its items; mean luma, clipped share; covered share = item pixels / opening polygon pixels; colours."""
    T = THRESHOLDS
    fmask, _, _ = S.polygon_mask("FIC", inlet_corners(I, I["frame_half"]))
    omask, _, _ = S.polygon_mask("FIC", inlet_corners(I, I["open_half"]))
    own = np.isin(seg, list(I["own"])) if I["own"] else np.zeros(seg.shape, bool)
    items = np.isin(seg, list(I["items"])) if I["items"] else np.zeros(seg.shape, bool)
    region = fmask & (own | items)
    Y = luma(img)
    mx = img.max(-1)
    out = dict(region_px=int(region.sum()), opening_px=int(omask.sum()))
    out["region_luma_dn"] = float(Y[region].mean()) if region.any() else 0.0
    out["region_clip"] = float((mx[region] >= T["clip_level"]).mean()) if region.any() else 1.0
    cov = omask & items
    out["covered_share"] = float(cov.sum() / max(1, omask.sum()))
    blk = np.isin(seg, list(S.item_geoms.get("blocker", set()))) & fmask
    unc = omask & own & ~items                     # the open grille the blocker is seen against
    if unc.sum() < 200:
        unc = fmask & own                          # (almost fully covered: against the frame)
    out["blocker_px"] = int(blk.sum())
    if blk.any() and unc.any():
        out["blocker_rgb"] = img[blk].mean(0).tolist()
        out["inlet_rgb"] = img[unc].mean(0).tolist()
        out["blocker_contrast_dn"] = float(np.linalg.norm(img[blk].mean(0) - img[unc].mean(0)))
    else:
        out["blocker_contrast_dn"] = None
    return out


def save_frame(path: str | None, img) -> str | None:
    if path is None:
        return None
    import cv2
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(img[..., ::-1]), [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    b = buf.tobytes()
    with open(path, "wb") as fh:
        fh.write(b)
    return _sha(b)


def render_fic(S: Scene, vp: dict, lamp_forward: bool = True):
    S.place(vp["body_pos"], np.asarray(vp["R"]))
    S.fr.set_lamp("forward", lamp_forward)
    S.fr.reset_exposure("FIC")
    img = S.fr.color("FIC")
    meta = dict(exposure=S.fr.last["exposure"], clipped=S.fr.last["clipped"], lighting=S.fr.last["lighting"],
                module=S.fr.last["module"], passes=len(S.fr.last["passes"]))
    seg = S.fr.segmentation("FIC")
    return img, seg, meta


def row_inlet_views(S: Scene, frames_dir: str | None, sid: str) -> dict:
    """Rows V2 / V1 (the blocked inlet) and R2 / R1 (ranking in its module)."""
    T = THRESHOLDS
    I0 = S.inlets[S.blocked]
    mod = I0["module"]
    res = dict(blocked=S.blocked, module=mod, lighting=S.lighting.get(mod), distances={})
    for D in T["view_distances_m"]:
        tag = f"{D:.1f}m"
        per_inlet = {}
        for iid, I in S.inlets.items():
            if I["module"] != mod:
                continue
            cands = candidate_viewpoints(S, I, D)
            rows = []
            for vp in cands:
                S.place(vp["body_pos"], np.asarray(vp["R"]))
                g = inlet_geometry(S, I)
                if geometry_ok(g):
                    rows.append(dict(vp, geometry=g))
                if len(rows) >= T["max_viewpoints_rendered"]:
                    break
            rendered = []
            for k, vp in enumerate(rows):
                img, seg, meta = render_fic(S, vp)
                st = frame_stats(S, I, img, seg)
                fn = None
                if frames_dir:
                    fn = os.path.join(frames_dir, f"{sid}_{iid}_{tag}_{k}.png")
                sha = save_frame(fn, img)
                ok_frame = bool(st["region_luma_dn"] >= T["region_luma_min_dn"] and st["region_clip"] <= T["region_clip_max"])
                if iid == S.blocked:
                    ok_frame = ok_frame and st["blocker_contrast_dn"] is not None and st["blocker_contrast_dn"] >= T["blocker_contrast_min_dn"]
                rendered.append(dict(incidence_deg=vp["incidence_deg"], cam_pos=vp["cam_pos"], body_pos=vp["body_pos"],
                                     clear_m=vp["clear_m"], geometry=vp["geometry"], frame=st, render=meta, frame_ok=ok_frame,
                                     file=os.path.basename(fn) if fn else None, sha256=sha))
            best = next((r for r in rendered if r["frame_ok"]), rendered[0] if rendered else None)
            per_inlet[iid] = dict(n_candidates=len(cands), n_geometry_ok=len(rows), views=rendered,
                                  observable=bool(any(r["frame_ok"] for r in rendered)),
                                  covered_share=(best["frame"]["covered_share"] if best else None))
        blk = per_inlet[S.blocked]
        others = {k: v["covered_share"] for k, v in per_inlet.items() if k != S.blocked}
        bc = blk["covered_share"]
        omax = max([v for v in others.values() if v is not None], default=0.0)
        rank_ok = bool(bc is not None and bc >= T["rank_cover_min"] and bc >= omax + T["rank_margin"])
        res["distances"][tag] = dict(view_ok=bool(blk["observable"]), rank_ok=rank_ok, blocked_cover=bc,
                                     others_cover=others, others_unobservable=[k for k, v in per_inlet.items() if k != S.blocked and not v["observable"]],
                                     inlets=per_inlet)
    return res


def row_items(S: Scene, frames_dir: str | None, sid: str, views: dict) -> dict:
    """Row I: every item in the 1 m FIC frame of its inlet (its best rendered geometry-ok viewpoint)."""
    T = THRESHOLDS
    out = {}
    for k, geoms in S.item_geoms.items():
        iid = S.item_inlet[k]
        I = S.inlets[iid]
        cands = candidate_viewpoints(S, I, 1.0)
        vp = None
        for c in cands:
            S.place(c["body_pos"], np.asarray(c["R"]))
            if geometry_ok(inlet_geometry(S, I)):
                vp = c
                break
        if vp is None:
            out[k] = dict(inlet=iid, ok=False, why="no geometry-ok 1 m viewpoint")
            continue
        img, seg, meta = render_fic(S, vp)
        msk = np.isin(seg, list(geoms))
        bars = [b for b in S.bars if b.name.split(":")[0] == k]
        bl = {}
        for b in bars:
            bm = np.isin(seg, list(b.geoms))
            if bm.any():
                ys, xs = np.nonzero(bm)
                import cv2
                rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
                bl[b.name] = float(max(rect[1]))
            else:
                bl[b.name] = 0.0
        Y = luma(img)
        fn = os.path.join(frames_dir, f"{sid}_item_{k}_1.0m.png") if frames_dir else None
        sha = save_frame(fn, img)
        ok = bool(msk.sum() >= T["item_px_min"] and (Y[msk].mean() if msk.any() else 0) >= T["item_luma_min_dn"]
                  and all(v >= T["bar_px_min_fic"] for v in bl.values()))
        out[k] = dict(inlet=iid, ok=ok, px=int(msk.sum()), luma_dn=float(Y[msk].mean()) if msk.any() else 0.0,
                      bar_long_px=bl, render=meta, incidence_deg=vp["incidence_deg"], file=os.path.basename(fn) if fn else None,
                      sha256=sha)
    return out


def bar_candidates(S: Scene, bar, dists):
    """Candidate ADC poses for a bar: the camera at distance d from the bar centre on directions within
    adc_incidences_deg of the bar's face normal (8 azimuths), looking at the centre, four rolls about the optical
    axis; then the site generator's pre-grasp poses (palm aligned) at a distance in the range."""
    T = THRESHOLDS
    c, a, nrm = bar.pose(S.d)
    b2 = np.cross(nrm, a)
    for inc in T["adc_incidences_deg"]:
        dirs = [nrm] if inc == 0 else [math.cos(math.radians(inc)) * nrm + math.sin(math.radians(inc)) * unit(t)
                                       for t in (a, -a, b2, -b2, a + b2, a - b2, -a + b2, -a - b2)]
        for dv in dirs:
            for D in dists:
                for roll in T["adc_rolls_deg"]:
                    cam_pos = c + unit(dv) * D
                    pb, Rb = S.camera_pose_robot("cam_aft_depth", cam_pos, c - cam_pos, up_hint=a, roll_deg=roll)
                    yield dict(source="adc_centred", incidence_deg=float(inc), dist_m=float(D), roll_deg=float(roll), pos=pb, R=Rb)
    from spacekit.rotations import quat_to_mat
    for r in (S.site.get("pregrasp_records") or {}).get(bar.name, []):
        yield dict(source="site_pregrasp", incidence_deg=None, dist_m=float(r["dist_m"]), roll_deg=None,
                   pos=np.asarray(r["pos"], float), R=quat_to_mat(np.asarray(r["quat"], float)))


def row_bars(S: Scene) -> dict:
    """Row B: every grasp feature seen whole by the ADC from 0.45-0.60 m with the arm deployed (module doc)."""
    T = THRESHOLDS
    lo, hi = S.bar_range
    dists = [d for d in T["adc_distances_m"] if lo - 1e-9 <= d <= hi + 1e-9]
    out = {}
    from firmware.bars import RAIL_FACE
    for bar in S.bars:
        S.reset_state()
        c, a, nrm = bar.pose(S.d)
        res = dict(kind=bar.kind, tries=0, zone_ok=0, evaluated=0, ok=False, best=None)
        s = np.linspace(-0.5, 0.5, T["bar_samples"]) * bar.length * 0.98
        pts = c[None, :] + nrm[None, :] * (RAIL_FACE / 2) + s[:, None] * a[None, :]
        for cand in bar_candidates(S, bar, dists):
            res["tries"] += 1
            pb, Rb = cand["pos"], cand["R"]
            if S.zones.map.check_point(pb, arm=True) != "ok":
                continue
            res["zone_ok"] += 1
            S.place(pb, Rb, tilt_deg=0.0, fingers_open=True)
            t, _Ro, _ = S.optical("ADC")
            dcam = float(np.linalg.norm(c - t))
            if not (lo - 0.005 <= dcam <= hi + 0.005):
                continue
            if res["evaluated"] >= T["adc_max_evaluated"]:
                break
            res["evaluated"] += 1
            uv, z, fm = S.project("ADC", pts)
            W, H, m = fm["width"], fm["height"], T["bar_margin_px"]
            inframe = (z > 0) & (uv[:, 0] >= m) & (uv[:, 0] <= W - m) & (uv[:, 1] >= m) & (uv[:, 1] <= H - m)
            rng_ok = (z >= S.fr.recipe["depth_range_m"][0]) & (z <= S.fr.recipe["depth_range_m"][1])
            hit = []
            for p in pts:
                v = p - t
                dd = float(np.linalg.norm(v))
                h, g = S.ray_first(t, v, dd + 0.05)
                hit.append(bool(h is not None and g in bar.geoms))
            zimg, _amp, gid = S.fr.depth("ADC", noise=False)
            npx = int(np.isin(gid, list(bar.geoms)).sum())
            row = dict(source=cand["source"], dist_m=round(dcam, 4), incidence_deg=cand["incidence_deg"], roll_deg=cand["roll_deg"],
                       in_frame=int(inframe.sum()), in_range=int(rng_ok.sum()), rays_on_bar=int(sum(hit)), depth_px=npx,
                       body_pos=np.asarray(pb).tolist())
            ok = bool(inframe.all() and rng_ok.all() and all(hit) and npx >= T["adc_bar_px_min"])
            if res["best"] is None or (row["rays_on_bar"], row["in_frame"], row["depth_px"]) > \
                    (res["best"]["rays_on_bar"], res["best"]["in_frame"], res["best"]["depth_px"]):
                res["best"] = row
            if ok:
                res["ok"] = True
                res["pass_pose"] = row
                break
        recs = (S.site.get("pregrasp_records") or {}).get(bar.name, [])
        res["site_pregrasp_zone_ok"] = int(sum(1 for r in recs if r.get("zone_ok")))
        out[bar.name] = res
    S.reset_state()
    return out


def check_site(site_path: str, frames_dir: str | None = None, look_root: str | None = None) -> dict:
    from cabincell import sensors_render as SR
    t0 = time.time()
    S = Scene(site_path, look_root=look_root)
    vals = S.cell.runtime["values"]
    S.bar_range = tuple(float(x) for x in SR._val(vals, "world.observability.bar_view_range_m"))
    S.prepare_offsets()
    sid = str(S.site.get("site_id"))
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
    views = row_inlet_views(S, frames_dir, sid)
    items = row_items(S, frames_dir, sid, views)
    bars = row_bars(S)
    rows = {}
    for tag, R in views["distances"].items():
        rows[f"V{tag}"] = bool(R["view_ok"])
        rows[f"R{tag}"] = bool(R["rank_ok"])
    rows["I"] = bool(all(v["ok"] for v in items.values()))
    rows["B"] = bool(all(v["ok"] for v in bars.values()))
    fails = [k for k, v in rows.items() if not v]
    why = []
    for tag, R in views["distances"].items():
        if not R["view_ok"]:
            bi = R["inlets"][S.blocked]
            why.append(f"V{tag}: blocked inlet {S.blocked}: {bi['n_candidates']} candidate viewpoints, {bi['n_geometry_ok']} "
                       f"geometry-ok, none passed the frame statistics")
        if not R["rank_ok"]:
            why.append(f"R{tag}: blocked cover {R['blocked_cover']}, others {R['others_cover']}")
    for k, v in items.items():
        if not v["ok"]:
            why.append(f"I: {k}: {v.get('why') or dict(px=v.get('px'), luma=round(v.get('luma_dn', 0), 1), bars=v.get('bar_long_px'))}")
    for k, v in bars.items():
        if not v["ok"]:
            why.append(f"B: {k}: zone-ok poses {v['zone_ok']} / {v['tries']}, best {v['best']}")
    import firmware as FW
    out = dict(schema="caretaking.observability/1", check_version=CHECK_VERSION, site_id=sid,
               evidence_rule=dict(impl="firmware.localisation.RayTester", firmware_version=FW.VERSION,
                                  near_clip_m=float(S.tester.occluder.near)),
               site_sha256=S.cell.site_sha, look_root=S.fr.look_root, renderer=S.fr.report["lamp_calibration"],
               thresholds=THRESHOLDS, bar_view_range_m=list(S.bar_range), lighting=S.lighting, blocked=S.blocked,
               tier=S.truth.get("tier"), rows=rows, passed=not fails, fails=fails, why=why,
               views=views, items=items, bars=bars, wall_s=round(time.time() - t0, 1))
    S.fr.close()
    S.srv.close()
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return round(v, 6) if math.isfinite(v) else None
    return x


def summary_line(r: dict) -> str:
    v = r["views"]["distances"]
    return (f"{r['site_id']}: {'PASS' if r['passed'] else 'FAIL'} rows={r['rows']} blocked={r['blocked']} "
            f"cover2={v['2.0m']['blocked_cover']} cover1={v['1.0m']['blocked_cover']} "
            f"others2={v['2.0m']['others_cover']} wall={r['wall_s']}s" + ("" if r["passed"] else f" why={r['why']}"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--look-root", default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rc = 0
    for sp in a.site:
        r = _jsonable(check_site(sp, a.frames, a.look_root))
        sid = r["site_id"]
        with open(os.path.join(a.out, f"{sid}.obs.json"), "w") as fh:
            json.dump(r, fh, indent=1, sort_keys=True)
            fh.write("\n")
        line = summary_line(r)
        with open(os.path.join(a.out, f"{sid}.obs.out"), "w") as fh:
            fh.write(line + "\n")
        print(line, flush=True)
        rc |= 0 if r["passed"] else 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
