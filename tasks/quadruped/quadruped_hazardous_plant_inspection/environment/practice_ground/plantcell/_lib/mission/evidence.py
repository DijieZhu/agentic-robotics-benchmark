"""Geometric evidence record of a frame for an instrument.

For a frame (camera pose in the site frame, intrinsics, image size) and a work-order instrument the record holds:
  distance_m     camera to the face centre
  off_axis_deg   angle between the line of sight (face centre -> camera) and the face normal
  app_px         apparent size of the type's face feature: f_px x nominal size / distance (the bricks study's
                 definition: dial = 0.9 x case, level window, lever length, gate stroke, lamp lens; thermal: the
                 smaller side of the component face's projected box x sqrt(visible), acceptance_v1's size rule)
  in_frame       share of the feature points inside the image (lever: also tag_in_frame for its tag plate; lamps:
                 the front face of the panel enclosure and the lenses, as the site generator tests it)
  visible        share of the feature's sample points that a ray from the camera reaches without another object in
                 front (ray casts against the drawn geometry, the robot's own body included); the points are a grid
                 over the feature's faces (EvidenceModel.sample_points), so the share is the share of the feature seen
  normal         (thermal) whether the site gives the component a face normal; only then is the off-axis limit
                 applied (acceptance_v1 off_axis_note)
  ok, failed     the acceptance conditions of acceptance_v1.yaml that pass / fail
No segmentation render at run time; a development test compares this record with one.

The world computes the record when a frame is taken (the robot's own legs can occlude) for every listed
instrument near the view, so that scoring needs only the stored records.
"""

from __future__ import annotations

import math

import numpy as np

from . import paths
from .scoring import evidence_failures, hottest_index

paths.setup()

import mujoco  # noqa: E402

G = mujoco.mjtGeom
SENSOR_OF = {"gauge": "inspect", "level": "inspect", "valve_lever": "inspect", "valve_handwheel": "inspect",
             "lamps": "inspect", "thermal": "thermal"}
GRID_AHEAD_M = 0.0015          # the visibility grid lies this far in front of a face (as the dial's samples)


def geom_points(model, data, gid, n_ring=12) -> np.ndarray:
    """Surface points of a geom in the world (box corners, cylinder rims, sphere poles, mesh AABB corners)."""
    t = int(model.geom_type[gid])
    s = np.asarray(model.geom_size[gid], float)
    p = np.asarray(data.geom_xpos[gid], float)
    R = np.asarray(data.geom_xmat[gid], float).reshape(3, 3)
    if t == G.mjGEOM_BOX:
        loc = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float) * s
    elif t in (G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE):
        a = np.linspace(0, 2 * math.pi, n_ring, endpoint=False)
        ring = np.column_stack([s[0] * np.cos(a), s[0] * np.sin(a), np.zeros_like(a)])
        loc = np.vstack([ring + [0, 0, s[1]], ring - [0, 0, s[1]]])
    elif t == G.mjGEOM_SPHERE:
        loc = np.vstack([np.eye(3), -np.eye(3)]) * s[0]
    else:
        c = np.asarray(model.geom_aabb[gid][:3], float)
        h = np.asarray(model.geom_aabb[gid][3:], float)
        loc = c + np.array([[a, b, cc] for a in (-1, 1) for b in (-1, 1) for cc in (-1, 1)], float) * h
    return p + loc @ R.T


def geom_samples(model, data, gid, shrink=0.8) -> np.ndarray:
    """Sample points for visibility: the geom centre and its surface points pulled toward the centre."""
    pts = geom_points(model, data, gid)
    c = np.asarray(data.geom_xpos[gid], float)
    return np.vstack([c[None, :], c + shrink * (pts - c)])


def project(K, T_world_cam, pts):
    Tinv = np.linalg.inv(np.asarray(T_world_cam, float))
    pc = pts @ Tinv[:3, :3].T + Tinv[:3, 3]
    z = pc[:, 2]
    u = K[0, 0] * pc[:, 0] / np.maximum(z, 1e-9) + K[0, 2]
    v = K[1, 1] * pc[:, 1] / np.maximum(z, 1e-9) + K[1, 2]
    return u, v, z


def first_hits(model, data, cam_pos, vec, geomgroup, exclude_body=-1):
    """(geom ids, distances) of the first geom each ray from cam_pos along the unit directions vec (n, 3) meets (-1,
    -1 when none): mujoco.mj_ray once per ray, static geoms included. Not mj_multiRay: its culling misses geoms of
    static bodies that hold several geoms (MuJoCo 3.14; on practice t1, 114 of 32,000 rays toward its instruments
    passed through the first geom, e.g. a frisker 0.26 m in front of the camera, and the points behind it counted
    as clear)."""
    vec = np.asarray(vec, np.float64).reshape(-1, 3)
    n = len(vec)
    gid = np.full(n, -1, np.int32)
    hit = np.full(n, -1.0, np.float64)
    p0 = np.asarray(cam_pos, np.float64).reshape(3)
    g1 = np.zeros(1, np.int32)
    gg = np.asarray(geomgroup, np.uint8)
    for i in range(n):
        hit[i] = mujoco.mj_ray(model, data, p0, vec[i], gg, 1, int(exclude_body), g1)
        gid[i] = g1[0]
    return gid, hit


def in_frame_share(K, T, pts, w, h, margin=1.0) -> float:
    if pts is None or len(pts) == 0:
        return float("nan")
    u, v, z = project(K, T, pts)
    ok = (z > 1e-3) & (u >= margin) & (u <= w - 1 - margin) & (v >= margin) & (v <= h - 1 - margin)
    return float(np.mean(ok))


class EvidenceModel:
    """Evidence records against the composed world's model (instrument geometry is static)."""

    def __init__(self, model, data, site: dict, cfg: dict, exclude_body: int = -1):
        self.m, self.d = model, data
        self.cfg = cfg["evidence"]
        self.exclude_body = int(exclude_body)
        self.items = {}
        par = np.asarray(model.body_parentid)
        for ip in site["ips"]:
            typ = ip["type"]
            if typ not in SENSOR_OF:
                continue
            mn = ip.get("model_names") or {}
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mn.get("body", ip.get("body", "")))
            sub = set()
            if bid >= 0:
                sub = {bid}
                ch = True
                while ch:
                    ch = False
                    for b in range(model.nbody):
                        if b not in sub and int(par[b]) in sub:
                            sub.add(b)
                            ch = True
            rec = {"tag": ip["tag"], "type": typ, "body": bid, "bodies": np.array(sorted(sub), int),
                   "names": mn, "ip": ip}
            rec.update(self._features(ip, bid))
            self.items[ip["tag"]] = rec

    # ------------------------------------------------------------------ static feature definitions
    def _gid(self, name):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, name)

    def _features(self, ip, bid):
        """Nominal feature size (m), face centre / normal, feature-point geoms, sample geoms."""
        typ = ip["type"]
        mn = ip.get("model_names") or {}
        br = (mn.get("bricks") or {}).get("spec") or {}
        face = [g for g in mn.get("face_geoms", []) if self._gid(g) >= 0]
        out = {"face_geoms": face}
        if typ == "gauge":
            D = float(br.get("size_mm", (ip.get("params") or {}).get("case_mm", 100))) / 1000.0
            out.update(size_m=0.9 * D, rim_r=0.45 * D)
        elif typ == "level":
            out.update(size_m=float(br.get("window_m", 0.44)))
        elif typ == "valve_lever":
            n = mn.get("body", "")
            out.update(size_m=float(br.get("lever_len", 0.18)),
                       feat_geoms=[g for g in (n + "/lever_grip", n + "/lever_bar", n + "/lever_hub") if self._gid(g) >= 0],
                       tag_geoms=[g for g in mn.get("geoms", []) if g.endswith("/tag/plate") and self._gid(g) >= 0],
                       pivot_body=mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n + "/lever"))
        elif typ == "valve_handwheel":
            n = mn.get("body", "")
            out.update(size_m=float(br.get("stroke", 0.10)),
                       feat_geoms=[g for g in (n + "/plate", n + "/pointer") if self._gid(g) >= 0])
        elif typ == "lamps":
            n = mn.get("body", "")
            enc = n + "/enclosure"
            # in frame: the enclosure's front face and the lenses (the generator's front-face rectangle,
            # a review); visibility: the lenses and the front face
            out.update(size_m=0.023, feat_geoms=list(face), enclosure=enc if self._gid(enc) >= 0 else None)
        elif typ == "thermal":
            units = (ip.get("truth") or {}).get("units", [])
            out.update(units=[{"geom": u["geom"], "label": u["label"], "gid": self._gid(u["geom"])} for u in units])
            hi = hottest_index(ip.get("truth") or {})        # hot_index may be None (no unit running hot)
            out["hot_geom"] = units[hi]["geom"] if units else None
            prm = ip.get("params") or {}
            # the site gives a face (ladder temperature point: face_w x face_h, normal = +x of the face frame)
            out["has_face"] = bool("face_w" in prm and "face_h" in prm)
        return out

    def face_frame(self, rec, hot_geom=None):
        """(c, R): the reference point of the distance, off-axis and size rules (c) and the face frame (R: x the face
        normal toward the viewer, y the viewer's right, z up). c is the centre of the feature the in-frame and size
        conditions are about:
          gauge            the dial's centre on the dial's surface (the instrument's position)
          level            the centre of the box around the glass tube and the scale plate, at mid-window (0.033 m
                           from the tube's axis toward the plate; the instrument's position is the tube's axis)
          valve_lever      the lever's pivot on the stem (lever_x = 0.62 OD + 0.035 in front of the pipe's axis, which
                           is the instrument's position; the tag plate's centre is 0.126 m below it): a fixed point,
                           the lever turns about it
          valve_handwheel  the indicator plate's centre on its front face, 0.42 m straight above the instrument's
                           position (at the height of the pipe's axis, which runs 0.035 m behind it)
          lamps            the centre of the panel's front face (the instrument's position is 5 mm above it, in the
                           face's plane)
          thermal          the hot face's centre (the unit's centre where the site gives no face)
        R is the instrument body's frame; it does not move with a lever or a stem."""
        if rec["type"] == "thermal":
            hg = hot_geom or rec.get("hot_geom")
            g = self._gid(hg) if hg else -1
            c = np.asarray(self.d.geom_xpos[g], float) if g >= 0 else np.asarray(rec["ip"]["pose"]["pos"], float)
            if not rec.get("has_face"):
                return c, None
            b = rec["body"]
            if b >= 0:
                R = np.asarray(self.d.xmat[b], float).reshape(3, 3)
            else:
                yaw = float(rec["ip"]["pose"].get("yaw", 0.0))
                R = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0],
                              [0.0, 0.0, 1.0]])
            return c, R
        b = rec["body"]
        c = np.asarray(self.d.xpos[b], float)
        R = np.asarray(self.d.xmat[b], float).reshape(3, 3)
        typ = rec["type"]
        if typ == "valve_lever" and rec.get("pivot_body", -1) >= 0:
            c = np.asarray(self.d.xpos[rec["pivot_body"]], float)
        elif typ == "level" and rec.get("face_geoms"):
            pts = np.vstack([geom_points(self.m, self.d, self._gid(g)) for g in rec["face_geoms"]])
            c = 0.5 * (pts.min(0) + pts.max(0))
        elif typ == "lamps" and rec.get("enclosure"):
            _, c = self._front_face(self._gid(rec["enclosure"]), R[:, 0])
        return c, R

    def _front_face(self, gid, n, shrink=1.0, ahead=0.0):
        """The four corners of a box geom's face whose outward normal is closest to n (optionally pulled toward
        the face centre and moved `ahead` along n)."""
        s = np.asarray(self.m.geom_size[gid], float)
        p = np.asarray(self.d.geom_xpos[gid], float)
        Rg = np.asarray(self.d.geom_xmat[gid], float).reshape(3, 3)
        best = max(((a, sg) for a in range(3) for sg in (-1.0, 1.0)), key=lambda k: k[1] * float(Rg[:, k[0]] @ n))
        a, sg = best
        o = [i for i in range(3) if i != a]
        fc = p + sg * s[a] * Rg[:, a]
        pts = [fc + shrink * (u * s[o[0]] * Rg[:, o[0]] + v * s[o[1]] * Rg[:, o[1]]) for u in (-1, 1) for v in (-1, 1)]
        return np.asarray(pts) + ahead * np.asarray(n, float), fc + ahead * np.asarray(n, float)

    def feature_points(self, rec, hot_geom=None):
        typ = rec["type"]
        c, R = self.face_frame(rec, hot_geom)
        if typ == "gauge":
            a = np.linspace(0, 2 * math.pi, 24, endpoint=False)
            r = rec["rim_r"]
            loc = np.column_stack([np.zeros_like(a), r * np.cos(a), r * np.sin(a)])
            return c + loc @ R.T, None
        if typ == "lamps":
            pts = [geom_points(self.m, self.d, self._gid(g)) for g in rec["feat_geoms"]]
            if rec.get("enclosure"):
                ff, _ = self._front_face(self._gid(rec["enclosure"]), R[:, 0])
                pts.append(ff)
            return (np.vstack(pts) if pts else np.zeros((0, 3))), None
        if typ in ("valve_lever", "valve_handwheel"):
            pts = [geom_points(self.m, self.d, self._gid(g)) for g in rec["feat_geoms"]]
            tag = None
            if typ == "valve_lever" and rec.get("tag_geoms"):
                tag = np.vstack([geom_points(self.m, self.d, self._gid(g)) for g in rec["tag_geoms"]])
            return (np.vstack(pts) if pts else np.zeros((0, 3))), tag
        if typ == "level":
            pts = [geom_points(self.m, self.d, self._gid(g)) for g in rec["face_geoms"]]
            return (np.vstack(pts) if pts else np.zeros((0, 3))), None
        if typ == "thermal":
            hg = hot_geom or rec.get("hot_geom")
            g = self._gid(hg) if hg else -1
            return (geom_points(self.m, self.d, g, n_ring=16) if g >= 0 else np.zeros((0, 3))), None
        return np.zeros((0, 3)), None

    def _face_grid(self, gid, n, nu, nv, ahead=GRID_AHEAD_M):
        """nu x nv points at the cell centres of a box geom's face whose outward normal is closest to n (nu along the
        face's longer side), `ahead` in front of the face. Any other geom type: geom_samples (no flat face)."""
        if gid < 0:
            return np.zeros((0, 3))
        if int(self.m.geom_type[gid]) != G.mjGEOM_BOX:
            return geom_samples(self.m, self.d, gid)
        s = np.asarray(self.m.geom_size[gid], float)
        p = np.asarray(self.d.geom_xpos[gid], float)
        Rg = np.asarray(self.d.geom_xmat[gid], float).reshape(3, 3)
        n = np.asarray(n, float)
        a, sg = max(((a, sg) for a in range(3) for sg in (-1.0, 1.0)), key=lambda k: k[1] * float(Rg[:, k[0]] @ n))
        o = sorted((i for i in range(3) if i != a), key=lambda i: -s[i])          # longer side first
        fc = p + sg * s[a] * Rg[:, a] + ahead * sg * Rg[:, a]
        uu = (2.0 * np.arange(nu) + 1.0) / nu - 1.0
        vv = (2.0 * np.arange(nv) + 1.0) / nv - 1.0
        pts = [fc + u * s[o[0]] * Rg[:, o[0]] + v * s[o[1]] * Rg[:, o[1]] for u in uu for v in vv]
        return np.asarray(pts, float)

    def _tube_line(self, gid, n, half_len, k, ahead=GRID_AHEAD_M):
        """k points along a cylinder's front line (its axis moved by the radius toward n), within +-half_len of the
        geom's centre along the axis: the side of a level gauge's glass tube that the camera sees."""
        if gid < 0:
            return np.zeros((0, 3))
        r = float(self.m.geom_size[gid][0])
        p = np.asarray(self.d.geom_xpos[gid], float)
        ax = np.asarray(self.d.geom_xmat[gid], float).reshape(3, 3)[:, 2]
        n = np.asarray(n, float)
        f = n - float(n @ ax) * ax
        f = f / max(float(np.linalg.norm(f)), 1e-9)
        zz = ((2.0 * np.arange(k) + 1.0) / k - 1.0) * float(half_len)
        return p[None, :] + zz[:, None] * ax[None, :] + (r + ahead) * f[None, :]

    def sample_points(self, rec, hot_geom=None):
        """The points of the visible share (`visible`): an area-proportional grid over the type's feature as the round
        procedure names it, so that "at least 90 % of the feature seen with nothing in front of it" is what is
        evaluated (a pipe in front of a third of a level's scale plate, or of a lever's
        tag plate, must count). Each part is sampled on its face toward the camera's side (the instrument's face
        normal), 1.5 mm in front of it:
          gauge            the dial: centre and four rings of 8 (33)
          level            the scale plate 3 x 12 and the glass tube's front line over the window 12 (48)
          valve_lever      the lever (bar 6, grip 4 x 2, hub 1) and the tag plate 6 x 3 (33)
          valve_handwheel  the indicator plate 2 x 6 and the pointer 3 (15)
          lamps            every lens (centre and both rims, 25 each) and the panel's front (centre and 4 corners)
          thermal          the hot face 5 x 4 where the site gives it a face (a box); else the unit's centre and corners
        Its interface and the record's fields are unchanged (the site generator calls it: site_gen/accept.py)."""
        typ = rec["type"]
        if typ == "lamps":
            pts = [geom_samples(self.m, self.d, self._gid(g)) for g in rec["feat_geoms"]]
            if rec.get("enclosure"):
                _, R = self.face_frame(rec)
                ff, fc = self._front_face(self._gid(rec["enclosure"]), R[:, 0], shrink=0.8, ahead=0.004)
                pts.append(np.vstack([fc[None, :], ff]))
            return np.vstack(pts) if pts else np.zeros((0, 3))
        if typ == "gauge":
            c, R = self.face_frame(rec)
            pts = [np.zeros(3)]
            for rr in (0.25, 0.5, 0.75, 0.92):
                for k in range(8):
                    a = 2 * math.pi * (k + 0.5 * (rr > 0.5)) / 8
                    pts.append([0.0015, rr * rec["rim_r"] * math.cos(a), rr * rec["rim_r"] * math.sin(a)])
            return c + np.asarray(pts) @ R.T
        if typ == "thermal":
            hg = hot_geom or rec.get("hot_geom")
            g = self._gid(hg) if hg else -1
            if g < 0:
                return np.zeros((0, 3))
            c, R = self.face_frame(rec, hot_geom)
            if R is not None and int(self.m.geom_type[g]) == G.mjGEOM_BOX:
                return self._face_grid(g, R[:, 0], 5, 4)
            return geom_samples(self.m, self.d, g)
        c, R = self.face_frame(rec)
        n = R[:, 0]
        name = (rec.get("names") or {}).get("body", "")
        pts = []
        if typ == "level":
            plate = [g for g in rec["face_geoms"] if g.endswith("/plate")]
            glass = [g for g in rec["face_geoms"] if g.endswith("/glass")]
            pts += [self._face_grid(self._gid(g), n, 12, 3) for g in plate]
            pts += [self._tube_line(self._gid(g), n, 0.5 * rec["size_m"], 12) for g in glass]
        elif typ == "valve_lever":
            for g, (nu, nv) in ((name + "/lever_bar", (6, 1)), (name + "/lever_grip", (4, 2)),
                                (name + "/lever_hub", (1, 1))):
                pts.append(self._face_grid(self._gid(g), n, nu, nv))
            pts += [self._face_grid(self._gid(g), n, 6, 3) for g in rec.get("tag_geoms") or []]
        elif typ == "valve_handwheel":
            pts.append(self._face_grid(self._gid(name + "/plate"), n, 6, 2))
            pts.append(self._face_grid(self._gid(name + "/pointer"), n, 3, 1))
        pts = [p for p in pts if len(p)]
        if not pts:                          # an instrument built otherwise: the parts' centres and corners, as before
            geoms = rec.get("feat_geoms") or rec["face_geoms"]
            pts = [geom_samples(self.m, self.d, self._gid(g)) for g in geoms]
        return np.vstack(pts) if pts else np.zeros((0, 3))

    # ------------------------------------------------------------------ visibility
    def visible_share(self, rec, cam_pos, pts, hot_geom=None):
        if len(pts) == 0:
            return float("nan")
        m, d = self.m, self.d
        gg = np.array([1, 1, 1, 1, 0, 0], np.uint8)
        vec = pts - cam_pos[None, :]
        dist = np.linalg.norm(vec, axis=1)
        vec = vec / np.maximum(dist[:, None], 1e-9)
        n = len(pts)
        gid, hit = first_hits(m, d, cam_pos, vec, gg, self.exclude_body)
        if rec["type"] == "thermal":
            # a thermal unit is one geom of a larger equipment body: only that geom counts as itself (the rest of
            # the pump or cabinet occludes it like anything else)
            hg = hot_geom or rec.get("hot_geom")
            own = (gid == self._gid(hg)) if hg else np.zeros(n, bool)
        else:
            own = np.isin(m.geom_bodyid[np.maximum(gid, 0)], rec["bodies"]) & (gid >= 0)
        clear = (gid < 0) | (hit < 0) | (hit >= dist - 0.012) | own
        return float(np.mean(clear))

    # ------------------------------------------------------------------ the record
    def record(self, frame: dict, tag: str, hot_geom: str | None = None) -> dict | None:
        """The evidence record of `frame` for instrument `tag`. hot_geom: (thermal, prediction only) evaluate this
        unit's geometry instead of the hot unit's (Truth.predict_evidence; it does not tell which unit is hot)."""
        rec = self.items.get(tag)
        if rec is None:
            return None
        typ = rec["type"]
        T = np.asarray(frame["T_world_cam"], float)
        K = np.asarray(frame["K"], float)
        w, h = int(frame["width"]), int(frame["height"])
        cam = T[:3, 3]
        c, R = self.face_frame(rec, hot_geom)
        to_cam = cam - c
        dist = float(np.linalg.norm(to_cam))
        out = {"tag": tag, "type": typ, "sensor": frame["sensor"], "distance_m": round(dist, 4)}
        if R is not None:
            n = R[:, 0]
            off = math.degrees(math.acos(float(np.clip(n @ to_cam / max(dist, 1e-9), -1.0, 1.0))))
        else:
            off = 0.0
        out["off_axis_deg"] = round(off, 2)
        if typ == "thermal":
            out["normal"] = R is not None
        fpts, tpts = self.feature_points(rec, hot_geom)
        out["in_frame"] = round(in_frame_share(K, T, fpts, w, h), 4)
        if tpts is not None:
            out["tag_in_frame"] = round(in_frame_share(K, T, tpts, w, h), 4)
        if typ == "thermal":
            u, v, z = project(K, T, fpts)
            okz = z > 1e-3
            if okz.any():
                bw = float(np.clip(u[okz], 0, w - 1).max() - np.clip(u[okz], 0, w - 1).min())
                bh = float(np.clip(v[okz], 0, h - 1).max() - np.clip(v[okz], 0, h - 1).min())
            else:
                bw = bh = 0.0
            out["box_px"] = [round(bw, 2), round(bh, 2)]
            out["app_px"] = round(min(bw, bh), 2)
        else:
            out["app_px"] = round(float(K[0, 0]) * rec["size_m"] / max(dist, 1e-9), 2)
        # the face must be in front of the camera
        u0, v0, z0 = project(K, T, c[None, :])
        out["center_px"] = [round(float(u0[0]), 1), round(float(v0[0]), 1)]
        out["in_front"] = bool(z0[0] > 0)
        out["visible"] = round(self.visible_share(rec, cam, self.sample_points(rec, hot_geom), hot_geom), 4)
        if typ == "thermal":
            # 6 x 6 px of the unit's face must be seen: the projected box shrunk by the visible share
            out["app_px"] = round(out["app_px"] * math.sqrt(max(out["visible"], 0.0)), 2)
        # conditions (one implementation, shared with the scoring of a stored record)
        failed = evidence_failures(out, self.cfg)
        out["failed"] = failed
        out["ok"] = not failed
        return out

    def near_view(self, frame: dict, max_dist: float = 7.0) -> list:
        """Tags of the listed instruments whose face centre is within max_dist and in front of the camera, with a
        projection inside the image enlarged by half its size (candidates for a record)."""
        T = np.asarray(frame["T_world_cam"], float)
        K = np.asarray(frame["K"], float)
        w, h = int(frame["width"]), int(frame["height"])
        cam = T[:3, 3]
        out = []
        for tag, rec in self.items.items():
            if SENSOR_OF[rec["type"]] != frame["sensor"]:
                continue
            c, _ = self.face_frame(rec)
            if np.linalg.norm(c - cam) > max_dist:
                continue
            u, v, z = project(K, T, c[None, :])
            if z[0] > 0 and -0.5 * w <= u[0] <= 1.5 * w and -0.5 * h <= v[0] <= 1.5 * h:
                out.append(tag)
        return out
