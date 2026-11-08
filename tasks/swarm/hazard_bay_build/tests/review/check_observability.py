"""Maintainers' check that a candidate day's targets can be observed (round 4, spec section 4.1: the pre-registered
observer-rig check). It runs on every candidate of a batch before the reference does; a day that fails is dropped and
recorded (`make_fixed_set.py select` reads `<results>/<id>.obs.json`). Never part of the grade, never named to the
agent.

    # the grader's renderer (in the task image, one GPU; the real bay dressed as for the mission, no robots):
    /opt/venv/bin/python3 -I -s tests/review/check_observability.py --set FILE --out DIR [--ids a,b] [--bay-rows]
    # the pxr / numpy cross-check (the ray caster of the observability study, `obs_raycast/`):
    python3 -I tests/review/check_observability.py --raycast --set FILE --out DIR [--ids a,b] [--bay-rows]

The rows (THRESHOLDS, pre-registered; frozen with the reference: `make_fixed_set.py` hashes this file):

 1  each part on Table A: a CAM-RGB-M at 0.92 m, aimed, from any floor point within 2.5 m with a 0.35 m circle clear of
    the furniture and outside the burning pan's 2 m ring: best view >= 0.9 of its camera-facing surface and >= 10,000 px^2
 2  each part on Table B: the same without the ring (the value with the ring recorded)
 3  each part from above: a CAM-RGB-W looking down from 2.6 m over its table's centre: >= 0.9 of its top
 4  the lever: a CAM-RGB-M at 0.92 m over the white square, aimed: the grip >= 0.5 and >= 1,000 px^2 at the day's angle,
    85 and 90 degrees
 5  the dial's reading: the same camera aimed at the dial: the face >= 0.9, >= 100 px across, the needle >= 3 px wide
 6  the dial from the start: the reference's mast camera at the day's start pose, or after a turn on the spot of up to
    90 degrees: >= 0.5 of the face, >= 40 px across by R8's measure
 7  the pan: the reference's THERM-C at the approach hover: >= 0.9 of its top, >= 25 px across
 8  the vehicle's start pose: the reference's mast camera at the day's start pose: >= 3 of {dial, valve body, lever grip,
    supports 1-3}, each >= 0.5 and >= 1,000 px^2
 9-11 (once per bay, --bay-rows): the bucket from the reference formation's down cameras >= 0.9; a flying robot's own place
    (its forward CAM-RGB-M on its pad and at the approach hover: >= 3 fixed landmarks, each >= 0.5 and >= 1,000 px^2);
    the crate's rim from a CAM-RGB-M at 0.92 m within 2.5 m >= 0.9

The renderer's measure: the observer is a grader camera (the mission's: the catalog's intrinsics, the pinned render
settings) with instance segmentation, read after Replicator's orchestrator steps once two consecutive reads agree (the
mission's fresh read); a target's visible share is its pixels in the render over its pixels with every other prim
hidden (a second render; every visibility restored as it was), times the share of its pixels inside the observer's
frame (a guard camera at the same pose with three times the aperture: its middle third is the frame); its size across
is the longer side of the smallest rectangle round its pixels (a dial's diameter, however foreshortened). Its viewpoints
for rows 1-2 are the ray caster's best three per part (its sweep of floor points; the renderer then measures them),
for row 11 the ray caster's best point per crate wall, the fixed ones for the other rows; row 3 looks straight down;
the targets are prims (a part, the grip, the dial and the needle, the pan, the start's landmarks; row 10's: the five
floor pads, the mat, the tank, the crate, the four pipe supports). Rows 9-11 draw no parts, the lever open, the pan at
its nominal place; row 9 moves the bucket to the formation (its bottom at 1.0 m, at (0, -1.0)). The robots' bodies
are not drawn. The ray caster's numbers are recorded beside the renderer's.
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:  # (the ray caster's mode needs no OpenCV)
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(HERE, "obs_raycast"))

THRESHOLDS = dict(part_share=0.9, part_px2=10000.0, top_share=0.9, grip_share=0.5, grip_px2=1000.0, grip_angles=(85.0, 90.0),
                  dial_share=0.9, dial_px=100.0, needle_px=3.0, start_dial_share=0.5, start_dial_px=40.0, start_turns_deg=90.0,
                  pan_share=0.9, pan_px=25.0, landmarks=3, landmark_share=0.5, landmark_px2=1000.0, bucket_share=0.9,
                  rim_share=0.9, ring_m=2.0, reach_m=2.5, mast_z=0.92, square=(-1.24, 2.61), approach=(2.60, -0.40, 2.50))
LANDMARKS = ("dial", "valve_block", "lever_grip", "support_1", "support_2", "support_3")


def file_hash():
    """This check's code and thresholds (the freeze: make_fixed_set.py records it)."""
    h = hashlib.sha256(open(os.path.abspath(__file__), "rb").read())
    for f in ("extract_geom.py", "extract_geom_lib.py", "visibility.py", "views.py"):
        h.update(open(os.path.join(HERE, "obs_raycast", f), "rb").read())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- the ray caster's rows
def raycast_rows(day, bay, V, W):
    """Rows 1-8 by the ray caster (obs_raycast): per row the measure and pass."""
    th = THRESHOLDS
    occ, parts = V.day_scene(bay, day)
    rows = {}
    for n, b in parts.items():
        table = day["parts"][n]["table"]
        ring = V.best_ugv_view(b, occ, day, heights=(th["mast_z"],), fire=True, rmax=th["reach_m"])
        free = V.best_ugv_view(b, occ, day, heights=(th["mast_z"],), fire=False, rmax=th["reach_m"])
        best = ring if table == "Table_A" else free
        ok = bool(best and best["frac"] >= th["part_share"] and best["px2"] >= th["part_px2"])
        rows[f"{1 if table == 'Table_A' else 2}:{n}"] = dict(share=_r(best and best["frac"]), px2=_r(best and best["px2"]), at=best and [best["x"], best["y"]],
                                                             with_ring=dict(share=_r(ring and ring["frac"]), px2=_r(ring and ring["px2"])), ok=ok)
        t = V.TABLES[table]
        C = np.array([t["c"][0], t["c"][1], 2.6])
        top = V.view(C, np.array([0, 0, -1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), "CAM-RGB-W", b, occ)
        rows[f"3:{n}"] = dict(share=_r(top["frac"]), px2=_r(top["px2"]), ok=bool(top["frac"] >= th["top_share"]))
    sq = np.array([th["square"][0], th["square"][1], th["mast_z"]])
    for deg in (float(day["lever_deg"]),) + tuple(th["grip_angles"]):
        o2, _ = V.day_scene(bay, dict(day, lever_deg=deg))
        grip = W.find(o2, "Handle/grip")[0]
        d, u, r = V.look_cam(sq, grip.c)
        v = V.view(sq, d, u, r, "CAM-RGB-M", grip, o2)
        rows[f"4:grip@{deg:.1f}"] = dict(share=_r(v["frac"]), px2=_r(v["px2"]), ok=bool(v["frac"] >= th["grip_share"] and v["px2"] >= th["grip_px2"]))
    dial = W.find(occ, "Gauge/Dial")[0]
    d, u, r = V.look_cam(sq, dial.c)
    v = V.view(sq, d, u, r, "CAM-RGB-M", dial, occ)
    dp = W.dial_needle_px(sq)
    rows["5:dial"] = dict(share=_r(v["frac"]), dial_px=dp["dial_px"], needle_px=dp["needle_w_px"],
                          ok=bool(v["frac"] >= th["dial_share"] and dp["dial_px"] >= th["dial_px"] and dp["needle_w_px"] >= th["needle_px"]))
    dx, dy, dpsi = day["ugv"]
    root = W.UGV_ROOT + np.array([dx, dy, 0.0])
    best6 = None
    for turn in np.arange(-th["start_turns_deg"], th["start_turns_deg"] + 0.1, 15.0):
        C, d, u, r = W.cam_world(root, W.ugv_frame(root[:2], dpsi + turn), W.UGV_MAST)
        v = V.view(C, d, u, r, "CAM-RGB-M", dial, occ)
        dist = float(np.linalg.norm(dial.c - C))
        px = 1920 * math.degrees(2 * math.atan(0.05 / dist)) / 70.3 if v["frac"] > 0 else 0.0  # (R8's measure, cs:870)
        key = (v["frac"] >= th["start_dial_share"] and px >= th["start_dial_px"], v["frac"])
        if best6 is None or key > best6[0]:
            best6 = (key, dict(turn_deg=float(turn), share=_r(v["frac"]), px=round(px, 1)))
    rows["6:dial_from_start"] = dict(best6[1], ok=bool(best6[0][0]))
    pv = W.pan_views(day, occ)["approach_hover/Thermal"]
    rows["7:pan"] = dict(share=pv["frac"], px=pv["w"], ok=bool(pv["frac"] >= th["pan_share"] and (pv["w"] or 0) >= th["pan_px"]))
    sv = W.start_views(day, occ)["views"]
    good = [k for k in LANDMARKS if k in sv and sv[k]["frac"] >= th["landmark_share"] and sv[k]["px2"] >= th["landmark_px2"]]
    rows["8:start_landmarks"] = dict(seen=good, n=len(good), ok=len(good) >= th["landmarks"])
    return rows


def raycast_bay_rows(bay, V, W):
    th = THRESHOLDS
    rows = {}
    bv = W.bucket_views(bay)
    for n in ("UAV_1", "UAV_2", "UAV_3"):
        rows[f"9:bucket:{n}"] = dict(share=bv[n]["frac"], ok=bool(bv[n]["frac"] >= th["bucket_share"]))
    day0 = dict(lever_deg=0.0, pan=[0.0, 0.0], parts={})
    occ, _ = V.day_scene(bay, day0)
    marks = {k: s for k, s in W.landmarks(occ).items()}
    for k in ("Pad_1", "Pad_2", "Pad_3", "CalibMat"):
        f = [s for s in occ if f"/{k}/" in s.name]
        if f:
            marks[k] = f[0]
    for n, (px, py) in {"UAV_1": (0.0, -1.5), "UAV_2": (-0.95, -3.15), "UAV_3": (0.95, -3.15), "approach": th["approach"][:2]}.items():
        H = np.array([px, py, 0.479 if n != "approach" else th["approach"][2]])
        C, d, u, r = W.cam_world(H, np.eye(3), W.UAV_CAMS["Camera"])
        seen = []
        for k, s in marks.items():
            v = V.view(C, d, u, r, "CAM-RGB-M", s, occ)
            if v["frac"] >= th["landmark_share"] and v["px2"] >= th["landmark_px2"]:
                seen.append(k)
        rows[f"10:own_place:{n}"] = dict(seen=seen, n=len(seen), ok=len(seen) >= th["landmarks"])
    walls = [s for s in occ if "/Crate/" in s.name]
    best = 0.0
    for s in walls:
        b = V.best_ugv_view(s, occ, day0, heights=(th["mast_z"],), fire=False, rmax=th["reach_m"])
        best = max(best, b["frac"] if b else 0.0)
    rows["11:crate_rim"] = dict(share=_r(best), ok=bool(best >= th["rim_share"]))
    return rows


def _r(x):
    return None if x is None else round(float(x), 4)


# --------------------------------------------------------------------------------------------- the renderer
LEVER = "/World/Facility/SteamLine/Handle"
GRIP = "/World/Facility/SteamLine/Handle/grip"
DIAL = "/World/Facility/SteamLine/Gauge/Dial"
NEEDLE = "/World/Facility/SteamLine/Gauge/needle"
START_MARKS = {"dial": DIAL, "valve_block": "/World/Facility/SteamLine/ValveBody", "lever_grip": GRIP,
               "support_1": "/World/Facility/SteamLine/Pipe/support_1", "support_2": "/World/Facility/SteamLine/Pipe/support_2",
               "support_3": "/World/Facility/SteamLine/Pipe/support_3"}
# row 10's fixed landmarks: the pads (the five floor pads), the mat, the tank, the crate, the pipe's supports
OWN_MARKS = dict({k: f"/World/Facility/Pads/{k}" for k in ("Pad_1", "Pad_2", "Pad_3", "UGV_Bay", "Valve_Square")},
                 CalibMat="/World/Facility/CalibMat", Tank="/World/Facility/Tank", Crate="/World/Facility/Crate",
                 **{f"support_{k}": f"/World/Facility/SteamLine/Pipe/support_{k}" for k in range(4)})
GUARD = 3.0  # (the guard camera's aperture factor: its middle third is the observer's frame)


class Render:
    """The real bay of a day, dressed as for the mission (no robots), in Kit; observer cameras with instance
    segmentation through the mission's own render products (the pinned settings). A read is taken after Replicator's
    orchestrator steps, the mission's fresh read, and only once two consecutive reads agree (`fresh`). Each observer
    has a guard camera at the same pose with three times its aperture, so that a target's share counts what lies
    outside the observer's frame as not seen."""

    def __init__(self, cfg, work):
        from isaacsim import SimulationApp
        self.app = SimulationApp({"headless": True})
        import omni.kit.app
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
        for _ in range(2):  # (Replicator before the stage opens, as the probe's rendering groups)
            self.app.update()
        import carb
        import omni.replicator.core as rep
        self.rep, self.settings = rep, carb.settings.get_settings()
        self.cfg, self.work = cfg, work
        self.stats = dict(reads=0, steps=0, unsettled=0, auto_exposure=self.settings.get("/rtx/post/histogram/enabled"))

    def open_day(self, day, bay_layer, hide_items=False):
        import control_sim
        import mission_stage
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom
        self.opened = getattr(self, "opened", 0) + 1  # (a new layer per day: the open stage keeps the last one registered)
        path = os.path.join(self.work, f"obs_day_{self.opened}.usda")
        if os.path.exists(path):
            os.remove(path)
        layer = Sdf.Layer.CreateNew(path)
        layer.subLayerPaths.append(bay_layer)
        layer.defaultPrim = "World"
        layer.Save()
        self.dress = mission_stage.dress(path, {}, self.cfg["sensors"]["look"])
        st = Usd.Stage.Open(path)
        for n, p in day["parts"].items():  # the parts at the day's poses
            prim = st.GetPrimAtPath(f"/World/Facility/Items/{n}")
            if not prim.IsValid():
                continue
            m = Gf.Matrix4d(1.0)
            m.SetRotateOnly(Gf.Quatd(float(p["quat"][0]), Gf.Vec3d(*[float(v) for v in p["quat"][1:]])))
            m.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in p["pos"]]))
            xf = UsdGeom.Xformable(prim)
            parent = UsdGeom.XformCache().GetParentToWorldTransform(prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "arb_day").Set(m * parent.GetInverse())
        if hide_items:  # (the day-independent rows: no parts, as the ray caster's)
            UsdGeom.Imageable(st.GetPrimAtPath("/World/Facility/Items")).MakeInvisible()
        control_sim.move_in_world(st, LEVER, self._lever_row(float(day["lever_deg"])), "arb_day")
        control_sim.move_in_world(st, "/World/Facility/FirePan", Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(day["pan"][0], day["pan"][1], 0.0)), "arb_day")
        st.GetRootLayer().Save()
        ctx = omni.usd.get_context()
        ctx.open_stage(path)
        for _ in range(3):
            self.app.update()
        self.stage = ctx.get_stage()
        self.products = {}

    @staticmethod
    def _lever_row(deg):
        """The lever turned toward +x (closing) about its stem (+y): a world-space row-vector matrix."""
        from pxr import Gf
        stem = Gf.Vec3d(-1.2, 3.2, 0.95)
        return Gf.Matrix4d(1.0).SetTranslate(-stem) * Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), deg)) * \
            Gf.Matrix4d(1.0).SetTranslate(stem)

    def set_lever(self, deg):
        """The open stage's lever to another angle (its `arb_day` op)."""
        from pxr import UsdGeom
        prim = self.stage.GetPrimAtPath(LEVER)
        op = next(o for o in UsdGeom.Xformable(prim).GetOrderedXformOps() if o.GetOpName() == "xformOp:transform:arb_day")
        parent = UsdGeom.XformCache().GetParentToWorldTransform(prim)
        op.Set(parent * self._lever_row(deg) * parent.GetInverse())

    def move_bucket(self, bottom_centre):
        """The open stage's bucket moved so that the bottom centre of its bound is at `bottom_centre`."""
        import control_sim
        from pxr import Gf, Usd, UsdGeom
        b = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
        r = b.ComputeWorldBound(self.stage.GetPrimAtPath("/World/Facility/Bucket")).ComputeAlignedRange()
        lo, hi = r.GetMin(), r.GetMax()
        d = Gf.Vec3d(bottom_centre[0] - (lo[0] + hi[0]) / 2, bottom_centre[1] - (lo[1] + hi[1]) / 2, bottom_centre[2] - lo[2])
        control_sim.move_in_world(self.stage, "/World/Facility/Bucket", Gf.Matrix4d(1.0).SetTranslate(d), "arb_formation")
        return [float(lo[0] + d[0]), float(lo[1] + d[1]), float(lo[2] + d[2])], [float(hi[0] + d[0]), float(hi[1] + d[1]), float(hi[2] + d[2])]

    def product(self, model):
        import sensor_rig
        if model not in self.products:
            sm = sensor_rig.stream_of(model, self.cfg["catalog"]["sensors"][model])
            base = dict(kind="thermal", width=sm["width"], height=sm["height"], focal=sm["focal"], clip=self.cfg["sensors"]["clip_m"])
            specs = [dict(base, name=f"Obs_{model.replace('-', '_')}", hap=sm["hap"]),
                     dict(base, name=f"Obs_{model.replace('-', '_')}_guard", hap=sm["hap"] * GUARD)]
            self.products[model] = (sensor_rig.Products(self.stage, specs, self.cfg["sensors"]["render"], update=self.app.update), sm)
        return self.products[model]

    def _step(self):
        self.settings.set_bool("/app/player/playSimulations", False)
        try:
            self.rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
        finally:
            self.settings.set_bool("/app/player/playSimulations", True)
        self.stats["steps"] += 1

    def fresh(self, model, max_steps=10, tol=0.002):
        """Both products' segmentation after orchestrator steps, once two consecutive reads agree (at most `tol` of
        the pixels differing: the renderer's two silhouette variants); [(ids, labels)] for the observer and its guard."""
        P, sm = self.product(model)
        prev = None
        for k in range(max_steps):
            self._step()
            cur = []
            for arr, info in P.read():
                a = np.asarray(arr).reshape(sm["height"], sm["width"]) if arr is not None and getattr(arr, "size", 0) else None
                cur.append((a, (info or {}).get("idToLabels") or {}))
            if k >= 1 and prev is not None and all(c[0] is not None for c in cur) and \
                    all(p[0] is not None and np.mean(p[0] != c[0]) <= tol for p, c in zip(prev, cur)):
                self.stats["reads"] += 1
                return cur
            prev = cur
        self.stats["unsettled"] += 1
        return prev

    def _mask(self, ids, lab, prim):
        hit = [int(k) for k, v in lab.items() if isinstance(v, str) and (v == prim or v.startswith(prim + "/"))]
        return np.isin(ids, hit) if hit and ids is not None else np.zeros((1, 1), dtype=bool)

    def measure(self, model, eye, target_xyz, target_prim, q=None):
        """dict(share, px2, across, in_frame, occluded_share) of a prim's subtree from `eye` (aimed at `target_xyz`, or
        oriented by `q`): its visible pixels in the bay over its pixels with every other prim hidden (a second render),
        times the share of its pixels in the observer's frame (the guard camera's middle third)."""
        import sensor_rig
        from pxr import UsdGeom
        P, sm = self.product(model)
        q = sensor_rig.look_at(eye, target_xyz) if q is None else q
        for i in range(2):
            P.set_pose(i, eye, q)
        full = self.fresh(model)
        a = int(self._mask(*full[0], target_prim).sum())
        hidden = []  # every other prim hidden: the siblings of the target and of each of its ancestors (each one's own
        p = self.stage.GetPrimAtPath(target_prim)  # visibility restored as it was afterwards)
        while p and p.GetPath().pathString != "/":
            par = p.GetParent()
            for sib in par.GetChildren() if par else []:
                sp = sib.GetPath().pathString
                if sib.GetPath() != p.GetPath() and sib.IsA(UsdGeom.Imageable) and not sp.startswith("/World/Sensors") and not sp.startswith("/Render"):
                    attr = UsdGeom.Imageable(sib).GetVisibilityAttr()
                    hidden.append((attr, attr.Get() if attr.HasAuthoredValue() else None))
                    attr.Set(UsdGeom.Tokens.invisible)
            p = par
        alone = self.fresh(model)
        for attr, old in hidden:
            if old is None:
                attr.Clear()
            else:
                attr.Set(old)
        m = self._mask(*alone[0], target_prim)
        b = int(m.sum())
        g = self._mask(*alone[1], target_prim)
        H, W = g.shape if g.ndim == 2 else (0, 0)
        g_all = int(g.sum())
        h0, w0 = int(round(H * (1 - 1 / GUARD) / 2)), int(round(W * (1 - 1 / GUARD) / 2))
        g_in = int(g[h0:H - h0, w0:W - w0].sum()) if g_all else 0
        in_frame = (g_in / g_all) if g_all else (1.0 if b else 0.0)
        across = 0  # (the target's largest extent in the image: the longer side of its pixels' smallest rectangle,
        if m.ndim == 2 and m.any():  # e.g. a dial's diameter, however foreshortened its other axis)
            vs, us = np.nonzero(m)
            (_, _), (w_, h_), _ = cv2.minAreaRect(np.stack([us, vs], axis=1).astype(np.float32))
            across = int(round(max(w_, h_) + 1))
        occ = (a / b) if b else 0.0
        return dict(share=min(1.0, occ) * min(1.0, in_frame), px2=a, across=across, in_frame=round(in_frame, 4), occluded_share=round(occ, 4))


def _m(v):
    return {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()}


def render_rows(day, R, ray_rows, V, W, bay):
    """Rows 1-8 with the renderer: the viewpoints of the ray caster's sweep (rows 1-2: its best view per part and two
    runners-up), the fixed ones for the others."""
    th = THRESHOLDS
    occ, parts = V.day_scene(bay, day)
    rows = {}
    for n, b in parts.items():
        table = day["parts"][n]["table"]
        fire = table == "Table_A"
        rects, discs = V.footprints(day, fire)
        cands = []
        for x in np.arange(-3.5, 3.5001, 0.1):
            for y in np.arange(-3.6, 3.2501, 0.1):
                if math.hypot(x - b.c[0], y - b.c[1]) > th["reach_m"] or not V.free((x, y), rects, discs):
                    continue
                C = np.array([x, y, th["mast_z"]])
                d, u, r = V.look_cam(C, b.c)
                v = V.view(C, d, u, r, "CAM-RGB-M", b, occ)
                cands.append((v["frac"] >= 0.5, v["px2"], C))
        cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
        best = None
        for _, _, C in cands[:3]:
            v = R.measure("CAM-RGB-M", C, b.c, f"/World/Facility/Items/{n}")
            if best is None or (v["share"] >= th["part_share"], v["px2"]) > (best[0]["share"] >= th["part_share"], best[0]["px2"]):
                best = (v, C)
        rows[f"{1 if fire else 2}:{n}"] = dict(_m(best[0]) if best else {}, at=best and [round(float(v), 2) for v in best[1][:2]],
                                                ok=bool(best and best[0]["share"] >= th["part_share"] and best[0]["px2"] >= th["part_px2"]),
                                                raycast=ray_rows.get(f"{1 if fire else 2}:{n}"))
        t = V.TABLES[table]
        v = R.measure("CAM-RGB-W", np.array([t["c"][0], t["c"][1], 2.6]), np.array([t["c"][0], t["c"][1], 0.0]), f"/World/Facility/Items/{n}")
        rows[f"3:{n}"] = dict(_m(v), ok=bool(v["share"] >= th["top_share"]), raycast=ray_rows.get(f"3:{n}"))
    sq = np.array([th["square"][0], th["square"][1], th["mast_z"]])
    for deg in (float(day["lever_deg"]),) + tuple(th["grip_angles"]):
        R.set_lever(deg)
        o2, _ = V.day_scene(bay, dict(day, lever_deg=deg))
        grip = W.find(o2, "Handle/grip")[0]
        v = R.measure("CAM-RGB-M", sq, grip.c, GRIP)
        rows[f"4:grip@{deg:.1f}"] = dict(_m(v), ok=bool(v["share"] >= th["grip_share"] and v["px2"] >= th["grip_px2"]),
                                         raycast=ray_rows.get(f"4:grip@{deg:.1f}"))
    R.set_lever(float(day["lever_deg"]))
    dial = W.find(occ, "Gauge/Dial")[0]
    v = R.measure("CAM-RGB-M", sq, dial.c, DIAL)
    vn = R.measure("CAM-RGB-M", sq, dial.c, NEEDLE)
    needle_w = vn["px2"] / max(1.0, W.dial_needle_px(sq)["needle_len_px"])
    rows["5:dial"] = dict(_m(v), dial_px=v["across"], needle_px=round(needle_w, 2), needle_px2=vn["px2"],
                          ok=bool(v["share"] >= th["dial_share"] and v["across"] >= th["dial_px"] and needle_w >= th["needle_px"]),
                          raycast=ray_rows.get("5:dial"))
    dx, dy, dpsi = day["ugv"]
    root = W.UGV_ROOT + np.array([dx, dy, 0.0])
    best6, tried = None, []
    for turn in np.arange(-th["start_turns_deg"], th["start_turns_deg"] + 0.1, 15.0):
        C, d, u, r = W.cam_world(root, W.ugv_frame(root[:2], dpsi + turn), W.UGV_MAST)
        v = R.measure("CAM-RGB-M", C, C + d, DIAL)
        dist = float(np.linalg.norm(dial.c - C))
        r8 = 1920 * math.degrees(2 * math.atan(0.05 / dist)) / 70.3 if v["px2"] else 0.0  # (R8's measure, cs:870)
        tried.append(dict(turn_deg=float(turn), share=round(v["share"], 4), px2=v["px2"], r8_px=round(r8, 1)))
        key = (v["share"] >= th["start_dial_share"] and r8 >= th["start_dial_px"], v["share"])
        if best6 is None or key > best6[0]:
            best6 = (key, dict(turn_deg=float(turn), share=round(v["share"], 4), px2=v["px2"], px=round(r8, 1), in_frame=v["in_frame"]))
    rows["6:dial_from_start"] = dict(best6[1], ok=bool(best6[0][0]), turns=tried, raycast=ray_rows.get("6:dial_from_start"))
    H = np.array(th["approach"])
    C, d, u, r = W.cam_world(H, np.eye(3), W.UAV_CAMS["Thermal"])
    v = R.measure("THERM-C", C, C + d, "/World/Facility/FirePan")
    rows["7:pan"] = dict(_m(v), px=v["across"], ok=bool(v["share"] >= th["pan_share"] and v["across"] >= th["pan_px"]), raycast=ray_rows.get("7:pan"))
    C, d, u, r = W.cam_world(root, W.ugv_frame(root[:2], dpsi), W.UGV_MAST)
    seen, vals = [], {}
    for k, pr in START_MARKS.items():
        v = R.measure("CAM-RGB-M", C, C + d, pr)
        vals[k] = _m(v)
        if v["share"] >= th["landmark_share"] and v["px2"] >= th["landmark_px2"]:
            seen.append(k)
    rows["8:start_landmarks"] = dict(seen=seen, n=len(seen), values=vals, ok=len(seen) >= th["landmarks"], raycast=ray_rows.get("8:start_landmarks"))
    return rows


def render_bay_rows(R, V, W, bay, ray_rows):
    """Rows 9-11 with the renderer, once per bay (no parts drawn, the lever open, the pan at its nominal place)."""
    th = THRESHOLDS
    rows = {}
    for n, (px, py) in {"UAV_1": (0.0, -1.5), "UAV_2": (-0.95, -3.15), "UAV_3": (0.95, -3.15), "approach": tuple(th["approach"][:2])}.items():
        H = np.array([px, py, 0.479 if n != "approach" else th["approach"][2]])
        C, d, u, r = W.cam_world(H, np.eye(3), W.UAV_CAMS["Camera"])
        seen, vals = [], {}
        for k, pr in OWN_MARKS.items():
            v = R.measure("CAM-RGB-M", C, C + d, pr)
            vals[k] = _m(v)
            if v["share"] >= th["landmark_share"] and v["px2"] >= th["landmark_px2"]:
                seen.append(k)
        rows[f"10:own_place:{n}"] = dict(seen=seen, n=len(seen), values=vals, ok=len(seen) >= th["landmarks"], raycast=ray_rows.get(f"10:own_place:{n}"))
    day0 = dict(lever_deg=0.0, pan=[0.0, 0.0], parts={})
    occ, _ = V.day_scene(bay, day0)
    top = np.array([0.1, 1.5, 0.32])  # (the crate's rim: its top's centre, bay_facts.json crate)
    pts = []
    for s in [s for s in occ if "/Crate/" in s.name]:
        b = V.best_ugv_view(s, occ, day0, heights=(th["mast_z"],), fire=False, rmax=th["reach_m"])
        if b and (b["x"], b["y"]) not in pts:
            pts.append((b["x"], b["y"]))
    best = None
    for x, y in pts:
        v = R.measure("CAM-RGB-M", np.array([x, y, th["mast_z"]]), top, "/World/Facility/Crate")
        if best is None or v["share"] > best[0]["share"]:
            best = (v, (x, y))
    rows["11:crate_rim"] = dict(_m(best[0]) if best else {}, at=best and list(best[1]), ok=bool(best and best[0]["share"] >= th["rim_share"]),
                                raycast=ray_rows.get("11:crate_rim"))
    bc = np.array([0.0, -1.0, 1.0])  # the bucket in the carry formation (views.bucket_views): its bottom at 1.0 m
    lo, hi = R.move_bucket(bc)
    vert = math.sqrt(1.5 ** 2 - (0.78 - 0.14) ** 2)
    for n, bear in (("UAV_1", 90.0), ("UAV_2", 210.0), ("UAV_3", 330.0)):
        hook = bc + np.array([0.78 * math.cos(math.radians(bear)), 0.78 * math.sin(math.radians(bear)), 0.125 + vert])
        root = hook - np.array([0.0, -0.104, -0.08])
        C, d, u, r = W.cam_world(root, np.eye(3), W.UAV_CAMS["CameraDown"])
        v = R.measure("CAM-RGB-W", C, None, "/World/Facility/Bucket", q=_down_q())
        rows[f"9:bucket:{n}"] = dict(_m(v), root=[round(float(x), 3) for x in root], ok=bool(v["share"] >= th["bucket_share"]),
                                     raycast=ray_rows.get(f"9:bucket:{n}"))
    rows["9:bucket_pose"] = dict(lo=[round(x, 3) for x in lo], hi=[round(x, 3) for x in hi], ok=True)
    return rows


def _down_q():
    """A camera looking straight down, image up = +y (the flying robots' heading; views.cam_world's down cameras)."""
    import sensor_rig
    return sensor_rig.mat_quat(np.stack([np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])], axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, dest="set_file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--raycast", action="store_true")
    ap.add_argument("--bay-rows", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(os.path.join(TESTS, "grader_config.json")))
    missions = json.load(open(a.set_file))["missions"]
    ids = [x for x in a.ids.split(",") if x]
    R = None if a.raycast else Render(cfg, a.out)  # (Kit first: pxr imported before the app would not be Kit's own)
    import extract_geom_lib
    bay = extract_geom_lib.bay_shapes(os.path.join(TESTS, "hidden", "bay_true.usda"))
    import visibility as V
    import views as W
    h = file_hash()
    for m in missions:
        if ids and m["id"] not in ids:
            continue
        day = m["day"]
        rows = raycast_rows(day, bay, V, W)
        mode = "raycast"
        if R is not None:
            R.open_day(day, os.path.join(TESTS, "hidden", "bay_true.usda"))
            rows = render_rows(day, R, rows, V, W, bay)
            mode = "renderer"
        res = dict(id=m["id"], mode=mode, check_sha256=h, render_stats=R and dict(R.stats), check_file_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
                   thresholds=THRESHOLDS, rows=rows, ok=all(r["ok"] for r in rows.values()),
                   failed=[k for k, r in rows.items() if not r["ok"]])
        json.dump(res, open(os.path.join(a.out, f"{m['id']}.obs.json"), "w"), indent=1)
        print(m["id"], mode, "PASS" if res["ok"] else f"FAIL {res['failed']}", flush=True)
    if a.bay_rows:
        ray = raycast_bay_rows(bay, V, W)
        rows, mode = ray, "raycast"
        if R is not None:
            R.open_day(dict(lever_deg=0.0, pan=[0.0, 0.0], parts={}), os.path.join(TESTS, "hidden", "bay_true.usda"), hide_items=True)
            rows, mode = render_bay_rows(R, V, W, bay, ray), "renderer"
        res = dict(mode=mode, check_sha256=h, thresholds=THRESHOLDS, rows=rows, ok=all(r["ok"] for r in rows.values()),
                   failed=[k for k, r in rows.items() if not r["ok"]])
        json.dump(res, open(os.path.join(a.out, "bay.obs.json"), "w"), indent=1)
        print("bay rows", mode, "PASS" if res["ok"] else f"FAIL {res['failed']}", flush=True)
    if R is not None:
        R.app.close()


if __name__ == "__main__":
    main()
