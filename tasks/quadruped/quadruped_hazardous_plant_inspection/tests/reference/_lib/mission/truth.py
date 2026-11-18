"""The privileged Truth object: handed only to the reference program.

It gives the true pose, the true plan (site.json with every value removed), the traversability lookup, the
reference route, each instrument's pose, mount geometry and acceptance region, and the unlisted source's position
for planning the approach. It never returns an instrument's value, a lamp state, a temperature, or the dose field's
value at a point other than where the robot is. Version 2: the plan of the layout's fixed collision geometry
(static_outlines: walls, door leaves, equipment, ... as outlines in plan with their heights), for naming what stopped
a walk.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from . import paths

paths.setup()

import mujoco  # noqa: E402

# keys that carry a value (never handed out)
_VALUE_KEYS = {"truth", "value", "states", "angle_deg", "opening", "level", "temp_C", "hot_index", "hot_label",
               "strength", "peak_dose_rate_uSv_h", "dose_rate_uSv_h", "opening_pct", "level_pct", "t_hot",
               # field values away from the robot (review 2026-11-13): the background and the reference route's dose
               # integrals; the reference measures its own background
               "background_uSv_h", "dose_total_uSv", "dose_travel_uSv", "dose_dwell_uSv", "dose_budget_uSv"}


def look_at_optical(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """4x4 pose of an optical frame (x right, y down, z forward) at eye looking at target."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= max(np.linalg.norm(z), 1e-9)
    x = np.cross(z, np.asarray(up, float))
    if np.linalg.norm(x) < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = eye
    return T


def _hull2d(P):
    """Convex hull (counter-clockwise, Andrew's monotone chain) of 2-D points."""
    pts = sorted({(round(float(x), 6), round(float(y), 6)) for x, y in np.asarray(P, float)})
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def redact(obj):
    """Deep copy without any value-carrying key."""
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items() if k not in _VALUE_KEYS}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return copy.deepcopy(obj)


class Truth:
    def __init__(self, world):
        self.__w = world
        self.__site = redact(world.site)

    # ------------------------------------------------------------------ pose
    def pose(self) -> dict:
        w = self.__w
        p = w.true_pose()
        p["t"] = round(w.time, 4)
        return p

    def T_world_base(self) -> np.ndarray:
        return self.__w.T_world_base().copy()

    def head_pan_frame(self) -> np.ndarray:
        """World pose (4x4) of the head's pan body (the pan joint frame) at the current true pose."""
        w = self.__w
        bid = w.model.body("robot/head_pan").id
        T = np.eye(4)
        T[:3, :3] = w.data.xmat[bid].reshape(3, 3)
        T[:3, 3] = w.data.xpos[bid]
        return T

    def camera_offset(self, sensor: str) -> np.ndarray:
        """Camera position in the tilt body frame (the mount)."""
        w = self.__w
        return np.asarray(w.model.cam_pos[w.cam_ids[sensor]], float).copy()

    # ------------------------------------------------------------------ the plan
    def site(self) -> dict:
        return copy.deepcopy(self.__site)

    def evidence_rules(self) -> dict:
        """The evidence gate and tolerances the grader applies (config.evidence_from_acceptance)."""
        return copy.deepcopy(self.__w.cfg["evidence"])

    def route(self) -> dict:
        return copy.deepcopy(self.__site["route"])

    def elements(self) -> list:
        return copy.deepcopy(self.__site["elements"])

    def element(self, eid: str) -> dict:
        for e in self.__site["elements"]:
            if e["id"] == eid:
                return copy.deepcopy(e)
        raise KeyError(eid)

    def modes_at(self, x, y):
        return list(self.__w.site_obj.modes_at(x, y))

    def cell_info(self, x, y) -> dict:
        return self.__w.site_obj.cell_info(x, y)

    def plan(self, start, goal, modes=("walk", "high_step", "crouch_walk"), start_mode="walk"):
        return self.__w.site_obj.plan(start, goal, modes=tuple(modes), start_mode=start_mode)

    def trav_grid(self):
        """origin, res, shape and per mode the enterable / tight cells, plus the slow cells, of the truth map."""
        t = self.__w.site_obj.trav
        return {"origin": (t.g.x0, t.g.y0), "res": t.g.res, "shape": t.g.shape,
                "enter": {m: np.asarray(t.enter[m]).copy() for m in t.enter},
                "tight": {m: np.asarray(t.tight[m]).copy() for m in t.tight}, "slow": np.asarray(t.slow).copy()}

    def plan_avoiding(self, start, goal, modes=("walk", "high_step", "crouch_walk"), avoid_kinds=("puddle",),
                      margin=0.20, start_mode="walk", clearance=((0.10, 0.30), (0.20, 0.55), (0.30, 0.85)),
                      avoid_ids=(), avoid_discs=()):
        """Cheapest path on the truth map with the footprints of the given element kinds and element ids (dilated
        by `margin`) made non-enterable, and the cells close to the edge of the enterable area made slower
        (`clearance`: (distance to the nearest non-enterable cell in m, speed factor) pairs), so that paths keep off
        walls and door jambs where there is room; same output as plan(). avoid_discs: (x, y, r) discs made
        non-enterable as well (version 2: the places where the reference was blocked with no known element at
        hand)."""
        import copy as _copy
        from scipy import ndimage
        so = self.__w.site_obj
        avoid_ids = tuple(sorted(avoid_ids or ()))
        avoid_discs = tuple(sorted((round(float(x), 3), round(float(y), 3), round(float(r), 3))
                                   for x, y, r in (avoid_discs or ())))
        key = (tuple(modes), tuple(avoid_kinds), round(float(margin), 3)) + ((avoid_ids,) if avoid_ids else ()) + \
            ((("discs",) + avoid_discs,) if avoid_discs else ())
        cache = getattr(self, "_plan_cache", None)
        if cache is None:
            cache = self._plan_cache = {}
        if key not in cache:
            t = so.trav
            t2 = _copy.copy(t)
            # the avoided elements' real geoms (a puddle is a few discs, smaller than its footprint rectangle),
            # each disc grown by the margin; other geom types by their footprint
            m, d = self.__w.model, self.__w.data
            ys, xs = np.mgrid[0:t.g.ny, 0:t.g.nx]
            cx = t.g.x0 + (xs + 0.5) * t.g.res
            cy = t.g.y0 + (ys + 0.5) * t.g.res
            bad = np.zeros(t.g.shape, bool)
            for e in self.__w.site["elements"]:
                if e["kind"] not in avoid_kinds and e["id"] not in avoid_ids:
                    continue
                done = False
                for gname in e.get("geoms", []):
                    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
                    if gid >= 0 and int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                        gx, gy = d.geom_xpos[gid][:2]
                        rr = float(m.geom_size[gid][0]) + margin
                        bad |= (cx - gx) ** 2 + (cy - gy) ** 2 <= rr * rr
                        done = True
                if not done:
                    fp = np.zeros(t.g.shape, bool)
                    t.g.fill_poly(fp, e["footprint"], 1)
                    if margin > 0 and fp.any():
                        r = int(math.ceil(margin / t.g.res))
                        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
                        fp = ndimage.binary_dilation(fp, structure=(xx * xx + yy * yy) <= r * r)
                    bad |= fp
            for dx_, dy_, dr_ in avoid_discs:
                bad |= (cx - dx_) ** 2 + (cy - dy_) ** 2 <= dr_ * dr_
            t2.enter = {m: np.asarray(t.enter[m]) & ~bad for m in t.enter}
            t2.speed = {}
            for m in t.speed:
                dist = ndimage.distance_transform_edt(t2.enter[m]) * t.g.res
                f = np.ones(t.g.shape, np.float32)
                for d_lim, fac in sorted(clearance, reverse=True):
                    f = np.where(dist < d_lim, fac, f)
                t2.speed[m] = np.where(t2.enter[m], t.speed[m] * f, 0.0).astype(np.float32)
            cache[key] = (t2, t2.build_graph(tuple(modes)), bad)
        t2, G, bad = cache[key]
        sc = t2.nearest_enterable(start[0], start[1], start_mode)
        if sc is None:
            return None
        N = t2.g.nx * t2.g.ny
        from site_gen.loader import MODES as TM
        s = TM.index(start_mode) * N + sc[0] * t2.g.nx + sc[1]
        D, P = t2.dijkstra(G, [s])
        goals = []
        for m in modes:
            gc = t2.nearest_enterable(goal[0], goal[1], m)
            if gc is not None:
                n = TM.index(m) * N + gc[0] * t2.g.nx + gc[1]
                goals.append((D[0, n], n))
        goals = [gg for gg in goals if np.isfinite(gg[0])]
        if not goals:
            return None
        cost, n = min(goals)
        path = t2.path(P[0], s, n)
        segs = []
        for node in path:
            iy, ix, m = t2.unnode(node)
            xy = list(t2.g.xy(iy, ix))
            if not segs or segs[-1]["mode"] != m:
                segs.append({"mode": m, "waypoints": [xy] if not segs else [segs[-1]["waypoints"][-1], xy]})
            else:
                segs[-1]["waypoints"].append(xy)
        return {"time_s": float(cost), "segments": segs}

    def static_outlines(self) -> list:
        """The site's fixed collision geometry as a plan (version 2: what the map shows the body could meet):
        one record per collision geom of the layout that never moves (welded to the world; the robot, the floor
        plane and moving parts such as needles and levers left out): {"geom": name, "what": wall | door_leaf |
        door_frame | equipment | furnishing | station | repeater | element_part | other, "element": the traversal
        element's id or None, "class": its class or None, "zmin", "zmax" (m), "outline": the convex outline in the
        site's plan (N x 2 list), "c": the outline's centre, "r": its radius from c}. Built once (static geometry)."""
        cache = getattr(self, "_outlines", None)
        if cache is not None:
            return cache
        w = self.__w
        m, d = w.model, w.data
        out = []
        G = mujoco.mjtGeom
        for g in range(m.ngeom):
            if w.robot_geom[g] or (int(m.geom_contype[g]) == 0 and int(m.geom_conaffinity[g]) == 0):
                continue
            if int(m.body_weldid[int(m.geom_bodyid[g])]) != 0:
                continue
            typ = int(m.geom_type[g])
            if typ in (int(G.mjGEOM_PLANE), int(G.mjGEOM_HFIELD)):
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            c = np.asarray(d.geom_xpos[g], float)
            R = np.asarray(d.geom_xmat[g], float).reshape(3, 3)
            s = np.asarray(m.geom_size[g], float)
            if typ in (int(G.mjGEOM_SPHERE),) or (typ == int(G.mjGEOM_CYLINDER) and abs(R[2, 2]) > 0.95):
                # a disc in plan (a sphere; an upright cylinder)
                r = float(s[0])
                hz = r if typ == int(G.mjGEOM_SPHERE) else float(s[1])
                a = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
                xy = np.c_[c[0] + r / math.cos(math.pi / 12) * np.cos(a), c[1] + r / math.cos(math.pi / 12) * np.sin(a)]
                z0, z1 = c[2] - hz, c[2] + hz
            else:
                if typ == int(G.mjGEOM_BOX) or typ == int(G.mjGEOM_ELLIPSOID):
                    half, ctr = s[:3], np.zeros(3)
                elif typ == int(G.mjGEOM_CYLINDER):
                    half, ctr = np.array([s[0], s[0], s[1]]), np.zeros(3)
                elif typ == int(G.mjGEOM_CAPSULE):
                    half, ctr = np.array([s[0], s[0], s[1] + s[0]]), np.zeros(3)
                else:                                   # a mesh: its bounding box in the geom's frame
                    ab = np.asarray(m.geom_aabb[g], float)
                    ctr, half = ab[:3], ab[3:6]
                corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
                P = c + (ctr + corners * half) @ R.T
                xy = _hull2d(P[:, :2])
                z0, z1 = float(P[:, 2].min()), float(P[:, 2].max())
            ec = w.elem_class.get(g)
            if name.startswith("wall/"):
                what = "wall"
            elif name.startswith("door/") and "/leaf" in name:
                what = "door_leaf"
            elif name.startswith("door/"):
                what = "door_frame"
            elif name.startswith("eq/"):
                what = "equipment"
            elif name.startswith("room/"):
                what = "furnishing"
            elif name.startswith("station/"):
                what = "station"
            elif name.startswith("link/"):
                what = "repeater"
            elif name.startswith("elem/"):
                what = "element_part"
            else:
                what = "other"
            xy = np.asarray(xy, float)
            cc = xy.mean(0)
            out.append({"geom": name, "what": what, "element": ec[1] if ec else None, "class": ec[0] if ec else None,
                        "zmin": float(z0), "zmax": float(z1), "outline": xy.tolist(), "c": cc.tolist(),
                        "r": float(np.max(np.linalg.norm(xy - cc, axis=1)))})
        self._outlines = out
        return out

    def puddle_discs(self) -> list:
        """(element id, x, y, radius) of every puddle disc (the drawn wet patches; geometry, no friction value)."""
        w = self.__w
        m, d = w.model, w.data
        out = []
        for e in w.site["elements"]:
            if e["kind"] != "puddle":
                continue
            for gname in e.get("geoms", []):
                gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
                if gid >= 0 and int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                    out.append((e["id"], float(d.geom_xpos[gid][0]), float(d.geom_xpos[gid][1]),
                                float(m.geom_size[gid][0])))
        return out

    def enterable(self, x, y, mode="walk", avoid_kinds=("puddle",), margin=0.20) -> bool:
        so = self.__w.site_obj
        t = so.trav
        iy, ix = t.g.cell(x, y)
        if not (0 <= iy < t.g.ny and 0 <= ix < t.g.nx) or not t.enter[mode][iy, ix]:
            return False
        key = (("walk", "high_step", "crouch_walk"), tuple(avoid_kinds), round(float(margin), 3))
        cache = getattr(self, "_plan_cache", {})
        if key in cache:
            return not bool(cache[key][2][iy, ix])
        return True

    def predict_evidence(self, tag, cam_pos, sensor="inspect", aim=None, unit=None) -> dict:
        """The acceptance check (geometry only) of a frame taken from cam_pos looking at the instrument's face centre
        (or `aim`). For a thermal item the check is made for one named unit (`unit`: its label; an item with a single
        unit needs none), as if that unit were the one to report: it never tells which unit is hot."""
        w = self.__w
        ev = w.evidence
        rec = ev.items.get(tag)
        if rec is None:
            return {"ok": False, "failed": ["not predictable"]}
        hot = None
        if rec["type"] == "thermal":
            units = rec.get("units") or []
            if unit is None and len(units) == 1:
                hot = units[0]["geom"]
            else:
                hot = next((u["geom"] for u in units if u["label"] == unit), None)
            if hot is None:
                return {"ok": False, "failed": ["not predictable (name the unit)"]}
            sensor = "thermal"
        c, _ = ev.face_frame(rec, hot)
        tgt = np.asarray(c if aim is None else aim, float)
        T = look_at_optical(np.asarray(cam_pos, float), tgt)
        cm = w.cam_models[sensor]
        frame = {"T_world_cam": T, "K": cm.K, "width": cm.width, "height": cm.height, "sensor": sensor}
        r = ev.record(frame, tag, hot_geom=hot)
        return {k: r[k] for k in ("ok", "failed", "app_px", "off_axis_deg", "distance_m", "visible", "in_frame")}

    def frame_evidence(self, tag, T_world_cam, sensor="inspect", unit=None) -> dict:
        """The acceptance check (geometry only) of a frame taken with the camera pose T_world_cam (4 x 4, optical
        frame in the site frame): the same record the grader keeps for the frame. For a thermal item, as in
        predict_evidence, for one named unit; it never tells which unit is hot."""
        w = self.__w
        ev = w.evidence
        rec = ev.items.get(tag)
        if rec is None:
            return {"ok": False, "failed": ["not predictable"]}
        hot = None
        if rec["type"] == "thermal":
            units = rec.get("units") or []
            if unit is None and len(units) == 1:
                hot = units[0]["geom"]
            else:
                hot = next((u["geom"] for u in units if u["label"] == unit), None)
            if hot is None:
                return {"ok": False, "failed": ["not predictable (name the unit)"]}
            sensor = "thermal"
        cm = w.cam_models[sensor]
        frame = {"T_world_cam": np.asarray(T_world_cam, float), "K": cm.K, "width": cm.width, "height": cm.height,
                 "sensor": sensor}
        r = ev.record(frame, tag, hot_geom=hot)
        return {k: r[k] for k in ("ok", "failed", "app_px", "off_axis_deg", "distance_m", "visible", "in_frame")}

    def line_of_sight(self, p0, p1, groups=(1, 1, 0, 0, 0, 0)) -> bool:
        """True when the segment p0 -> p1 hits no geometry of the given groups (structure and decor by default)."""
        w = self.__w
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        v = p1 - p0
        d = float(np.linalg.norm(v))
        if d < 1e-6:
            return True
        gid = np.zeros(1, np.int32)
        hit = mujoco.mj_ray(w.model, w.data, p0, v / d, np.asarray(groups, np.uint8), 1, w.base_id, gid)
        return bool(hit < 0 or hit >= d - 0.05)

    # ------------------------------------------------------------------ instruments
    def instrument(self, tag: str) -> dict:
        """Geometry of a work-order item: type, face pose, mount, observation pose, model names (no value)."""
        w = self.__w
        ip = next(i for i in w.site["ips"] if i["tag"] == tag)
        red = next(i for i in self.__site["ips"] if i["tag"] == tag)
        out = {"tag": tag, "type": ip["type"], "room": ip["room"], "pose": copy.deepcopy(red["pose"]),
               "params": copy.deepcopy(red.get("params", {})), "observation_pose": copy.deepcopy(red["observation_pose"]),
               "marker_id": (ip.get("model_names") or {}).get("marker_id"), "body": ip.get("body")}
        mn = ip.get("model_names") or {}
        out["face_geoms"] = list(mn.get("face_geoms", []))
        out["geoms"] = list(mn.get("geoms", []))
        bid = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, ip.get("body", ""))
        if bid >= 0:
            out["face_center"] = w.data.xpos[bid].tolist()
            out["face_R"] = w.data.xmat[bid].reshape(3, 3).tolist()
        else:
            p = ip["pose"]["pos"]
            out["face_center"] = list(p)
            yaw = float(ip["pose"].get("yaw", 0.0))
            out["face_R"] = [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0, 0, 1.0]]
        # the tag plate's box (version 2: the reference aims at it and finds it in its frames; the text it reads from
        # its own pixels): centre, rotation (columns: the box's x = the plate's outward normal, y, z) and half sizes
        pg = next((g for g in out["geoms"] if g.endswith("/tag/plate")), None) or \
            ((ip.get("body") or "") + "/tag/plate")
        gid = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_GEOM, pg)
        if gid >= 0:
            out["plate"] = {"geom": pg, "center": w.data.geom_xpos[gid].tolist(),
                            "R": w.data.geom_xmat[gid].reshape(3, 3).tolist(),
                            "half": [float(v) for v in w.model.geom_size[gid][:3]]}
        if ip["type"] == "valve_lever":
            spec = ((mn.get("bricks") or {}).get("spec") or {})
            out["pipe_axis_local"] = spec.get("pipe_axis", "y")
        if ip["type"] == "thermal":
            units = []
            for u in (ip.get("truth") or {}).get("units", []):
                units.append({"geom": u["geom"], "label": u["label"], "pos": list(u["pos"])})
            out["units"] = units
            prm = ip.get("params") or {}
            out["has_face_normal"] = bool("face_w" in prm and "face_h" in prm)
        if ip["type"] == "hotspot":
            out["survey_rooms"] = copy.deepcopy(red.get("params", {}))
        return out

    def is_readable(self, tag, x, y, mode="walk") -> bool:
        return bool(self.__w.site_obj.is_readable(tag, x, y, mode))

    def readable_cells(self, tag, mode="walk") -> np.ndarray:
        """(N, 2) centres of the readable cells (0.05 m grid) of the instrument's acceptance region."""
        so = self.__w.site_obj
        rr = so.ip(tag)["readable_region"]
        g = rr["grid"]
        m = so.readable_mask(tag, mode)
        iy, ix = np.nonzero(m)
        x0, y0 = g["origin"]
        res = g["res"]
        return np.c_[x0 + (ix + 0.5) * res, y0 + (iy + 0.5) * res]

    def source_position(self, tag: str):
        """The unlisted source's position, for planning the survey approach only (never for the report)."""
        w = self.__w
        ip = next(i for i in w.site["ips"] if i["tag"] == tag)
        sid = (ip.get("truth") or {}).get("source")
        for s in w.site["sources"]:
            if s["id"] == sid:
                return [float(s["pos"][0]), float(s["pos"][1]), float(s["pos"][2])]
        return list(ip["pose"]["pos"])

    def dose_rate_here(self) -> float:
        """True dose rate (Gy/h) at the robot's dosimeter now (planning of the dose margin only)."""
        return float(self.__w.rate_now)
