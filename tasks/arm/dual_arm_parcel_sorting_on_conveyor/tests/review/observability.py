"""Observability of the waybill (a maintainer's evidence; the grader never runs it).

    python3 -B tests/review/observability.py --out DIR [--write-measured]

For every camera, how large the waybill is in pixels and how it stands out from its parcel, over the places a parcel
can be. Fixed cameras (``infeed``, ``head``): a parcel is posed on the belt (resting, label on the named face, yaw 0
and 45 deg) at every place of a grid along the belt (entry to scan line) and across it (the lateral range), the robot
at its start posture; the smallest and the largest carton and a mailer of the population are posed in turn. Measured
per pose: the waybill's corners projected through the PUBLISHED calibration (describe()["cameras"]: its long and
short side in pixels and the dark square code block's side), the share of the waybill's pixels that are visible
(a segmentation render of the cell's own model: the robot and the parcel itself can hide it), and the contrast in
the colour render (mean luminance of the waybill's visible pixels against the parcel's own visible pixels, Weber
contrast, and the code block against the waybill). Wrist cameras: the waybill seen head-on at stand-off distances
inside their depth range (pinhole projection with the published intrinsics).

The renders are the backend in the cell's configuration (``look``; today the classic renderer); re-run on the final
look. ``--write-measured`` puts the summary table for the cell manual into ``docs/measured.json`` (block
"observability"). Output: ``DIR/observability.json`` (every pose) and ``DIR/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import common  # noqa: E402

MEASURED = os.path.join(HERE, "docs", "measured.json")


def _parcels(cfg):
    sc = cfg["shifts"]
    cs, ms = sc["carton_size_m"], sc["mailer_size_m"]
    small = (cs[0][0], cs[1][0], cs[2][0])
    large = (cs[0][1], cs[1][1], cs[2][1])
    mailer = (round((ms[0][0] + ms[0][1]) / 2, 3), round((ms[1][0] + ms[1][1]) / 2, 3), ms[2][1])
    return [("smallest carton", "rigid", small), ("largest carton", "rigid", large), ("mailer", "soft", mailer)]


def _lum(rgb):
    import numpy as np
    x = rgb.astype(np.float64)
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


class Bench(object):
    def __init__(self, cfg):
        import mujoco
        from parcelcell.cameras import CameraSystem
        from parcelcell.config import cell_config
        from parcelcell.stream import Shift
        from parcelcell.world import ShiftWorld
        self.mj = mujoco
        self.cfg = cfg
        self.world = ShiftWorld(Shift(belt_speed=((0.0, 0.2),), arrivals=(), duration=60.0, seed=0),
                                cell_config(cfg), spare_slots=1)
        self.cams = CameraSystem(self.world, cfg, 0, render=False)
        m = self.model = self.world.model
        self.d = self.world.data
        self.slot = 0
        g = self.world._slot_g[self.slot]
        self.label_ids = {g["label"], g["label_qr"], g["label_bar"]}
        self.sheet_id, self.qr_id = g["label"], g["label_qr"]
        self.body_ids = {g["geom_visual"], g["pad_visual"]}
        self.rend = {}
        for n, s in self.cams.specs.items():
            v = s["view"]
            self.rend[n] = (mujoco.Renderer(m, v.height, v.width), s["view"].model_camera)
        self.info = self.cams.describe(self.d)

    def pose(self, kind, size, face, x, y, yaw):
        import numpy as np
        from parcelcell.stream import Parcel
        from parcelcell.visuals import KRAFT_RGBA, MAILER_RGBA
        p = Parcel(kind=kind, size=size, mass=0.8, friction=0.55, label_face=face, label_uv=(0.0, 0.0))
        z = self.world.station.belt_height + size[2] / 2 + 0.0005
        q = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        w = self.world
        w.place_parcel(self.slot, p, np.array([x, y, z]), q)
        w.slots[self.slot].active = True
        rgba = KRAFT_RGBA if kind == "rigid" else MAILER_RGBA
        for gid in self.body_ids:
            self.model.geom_rgba[gid] = rgba
        self.mj.mj_forward(self.model, self.d)
        return p

    def label_corners(self):
        """World corners of the waybill sheet's outer face and of the square code block."""
        import numpy as np
        out = {}
        for key, gid in (("sheet", self.sheet_id), ("qr", self.qr_id)):
            c = self.d.geom_xpos[gid]
            R = self.d.geom_xmat[gid].reshape(3, 3)
            h = self.model.geom_size[gid]
            ax = int(np.argmin(h))
            t = [i for i in range(3) if i != ax]
            pts = []
            for s1, s2 in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                v = np.zeros(3)
                v[t[0]], v[t[1]], v[ax] = s1 * h[t[0]], s2 * h[t[1]], h[ax]
                pts.append(c + R @ v)
            out[key] = np.array(pts)
        return out

    def project(self, name, pts):
        import numpy as np
        c = self.info[name]
        T = np.asarray(c["T_world_cam"]) if "T_world_cam" in c else None
        Tinv = np.linalg.inv(T)
        P = (Tinv[:3, :3] @ pts.T).T + Tinv[:3, 3]
        if (P[:, 2] <= 1e-6).any():
            return None
        u = c["fx"] * P[:, 0] / P[:, 2] + c["cx"]
        v = c["fy"] * P[:, 1] / P[:, 2] + c["cy"]
        return np.stack([u, v], 1)

    def measure(self, name):
        import numpy as np
        c = self.info[name]
        corners = self.label_corners()
        uv = self.project(name, corners["sheet"])
        out = {"in_view": False}
        if uv is None:
            return out
        sides = [float(np.linalg.norm(uv[(i + 1) % 4] - uv[i])) for i in range(4)]
        qr = self.project(name, corners["qr"])
        inside = ((uv[:, 0] >= -0.5) & (uv[:, 0] <= c["width"] - 0.5) & (uv[:, 1] >= -0.5) &
                  (uv[:, 1] <= c["height"] - 0.5)).all()
        area = 0.5 * abs(sum(uv[i, 0] * uv[(i + 1) % 4, 1] - uv[(i + 1) % 4, 0] * uv[i, 1] for i in range(4)))
        out.update(in_view=bool(inside), long_px=round(max(sides), 1), short_px=round(min(sides), 1),
                   area_px=round(area, 0),
                   qr_px=None if qr is None else round(float(min(np.linalg.norm(qr[(i + 1) % 4] - qr[i])
                                                                 for i in range(4))), 1))
        r, cam = self.rend[name]
        r.update_scene(self.d, camera=cam)
        r.enable_segmentation_rendering()
        seg = r.render()[..., 0].copy()
        r.disable_segmentation_rendering()
        r.update_scene(self.d, camera=cam)
        rgb = r.render().copy()
        lab = np.isin(seg, list(self.label_ids))
        sheet = seg == self.sheet_id
        qr_m = seg == self.qr_id
        body = np.isin(seg, list(self.body_ids))
        out["visible_frac"] = round(min(1.0, float(lab.sum()) / max(1.0, area)), 3) if inside else None
        L = _lum(rgb)
        if sheet.sum() >= 4 and body.sum() >= 4:
            ls, lb = float(L[sheet].mean()), float(L[body].mean())
            out.update(label_lum=round(ls, 1), body_lum=round(lb, 1), weber=round((ls - lb) / max(lb, 1.0), 3))
        if qr_m.sum() >= 1 and sheet.sum() >= 4:
            out["code_contrast"] = round((float(L[sheet].mean()) - float(L[qr_m].mean())) / max(float(L[sheet].mean()), 1.0), 3)
        return out


def run(cfg):
    import numpy as np
    b = Bench(cfg)
    st = b.world.station
    lat = float(cfg["shifts"]["lateral_max_m"])
    ys = [-1.10, -0.90, -0.60, -0.42, -0.20, 0.0, 0.20, 0.42, 0.70, 0.95]
    rows = []
    # faces at yaw 0: top (0), far side (+x, 2), robot side (-x, 3), downstream (+y, 4), upstream (-y, 5); a bottom
    # label (1) faces the belt and no camera sees it while the parcel lies there
    for label, kind, size in _parcels(cfg):
        faces = (0, 1, 2, 3, 4, 5) if kind == "rigid" else (0, 1)
        for face in faces:
            from parcelcell.stream import valid_label_faces
            if face not in valid_label_faces(size, (face,)):
                continue
            for y in ys:
                for x_off in (-lat, 0.0, lat):
                    for yaw in (0.0, math.pi / 4):
                        b.pose(kind, size, face, st.belt_center_x + x_off, y, yaw)
                        for cam in ("infeed", "head"):
                            mres = b.measure(cam)
                            rows.append({"camera": cam, "parcel": label, "kind": kind, "size": list(size),
                                         "face": face, "y": y, "x_off": x_off, "yaw_deg": round(math.degrees(yaw)),
                                         **mres})
    # wrist cameras: head-on, analytic (published intrinsics)
    wr = []
    c = b.info["wrist_left"]
    from parcelcell import _geom
    for label, kind, size in _parcels(cfg):
        lay = _geom.label_layout(size, 0, (0.0, 0.0))
        lx, ly = 2 * lay["label"][1][0], 2 * lay["label"][1][1]
        q = 2 * lay["label_qr"][1][0]
        for d in (0.15, 0.25, 0.40, 0.55):
            wr.append({"camera": "wrist", "parcel": label, "distance_m": d,
                       "long_px": round(c["fx"] * max(lx, ly) / d, 1), "short_px": round(c["fx"] * min(lx, ly) / d, 1),
                       "qr_px": round(c["fx"] * q / d, 1)})
    for r, _ in b.rend.values():
        r.close()
    return rows, wr, b


def summarize(rows, wr):
    def pick(cam, parcel, face, ylo, yhi):
        xs = [r for r in rows if r["camera"] == cam and r["parcel"] == parcel and r["face"] == face
              and ylo <= r["y"] <= yhi and r.get("in_view")]
        return xs

    def fmt(xs, key, nd=0):
        v = [x[key] for x in xs if x.get(key) is not None]
        if not v:
            return "-"
        lo, hi = min(v), max(v)
        f = "%%.%df" % nd
        return (f % lo) if abs(hi - lo) < 0.5 * 10 ** (-nd) else "%s-%s" % (f % lo, f % hi)

    places = [("infeed", "just past the entry (y -1.10 to -0.90)", -1.10, -0.90, 0),
              ("infeed", "before the work window (y -0.60 to -0.42)", -0.60, -0.42, 0),
              ("head", "upstream window (y -0.42 to -0.20)", -0.42, -0.20, 0),
              ("head", "window centre (y -0.20 to 0.20)", -0.20, 0.20, 0),
              ("head", "downstream window (y 0.20 to 0.42)", 0.20, 0.42, 0),
              ("head", "towards the scan line (y 0.70 to 0.95)", 0.70, 0.95, 0),
              ("head", "window, label facing the robot", -0.42, 0.42, 3),
              ("head", "window, label facing upstream", -0.42, 0.42, 5),
              ("head", "window, label facing downstream", -0.42, 0.42, 4),
              ("head", "window, label facing the far side", -0.42, 0.42, 2),
              ("infeed", "label on the bottom (facing the belt)", -1.10, -0.42, 1),
              ("head", "label on the bottom (facing the belt)", -0.42, 0.42, 1)]
    head = ["camera", "where (label face)", "waybill long x short side (px): smallest carton",
            "largest carton", "mailer", "square code block side (px)", "share visible"]
    trows, contrast = [], []
    for cam, where, ylo, yhi, face in places:
        cells = []
        allx = []
        for parcel in ("smallest carton", "largest carton", "mailer"):
            xs = pick(cam, parcel, face, ylo, yhi)
            allx += xs
            posed = any(r["camera"] == cam and r["parcel"] == parcel and r["face"] == face for r in rows)
            seen = [x for x in xs if (x.get("visible_frac") or 0.0) >= 0.5]
            cells.append("%s x %s" % (fmt(seen, "long_px"), fmt(seen, "short_px")) if seen else
                         ("-" if posed else "n/a"))
        vis = [x["visible_frac"] for x in allx if x.get("visible_frac") is not None]
        seen_all = [x for x in allx if (x.get("visible_frac") or 0.0) >= 0.5]
        trows.append(["`%s`" % cam, where] + cells +
                     [fmt(seen_all, "qr_px"), ("%d %%" % round(100 * min(vis)) if round(100 * min(vis)) == round(100 * max(vis)) else "%d-%d %%" % (round(100 * min(vis)), round(100 * max(vis)))) if vis else "-"])
        contrast.append({"camera": cam, "where": where, "face": face, "waybill_vs_parcel_weber": fmt(allx, "weber", 2),
                         "code_vs_waybill": fmt(allx, "code_contrast", 2)})
    for d in (0.15, 0.25, 0.40):
        xs = [w for w in wr if w["distance_m"] == d]
        by = {w["parcel"]: w for w in xs}
        trows.append(["`wrist_*`", "head-on at %.2f m" % d] +
                     ["%.0f x %.0f" % (by[p]["long_px"], by[p]["short_px"]) for p in
                      ("smallest carton", "largest carton", "mailer")] +
                     ["%.0f-%.0f" % (min(w["qr_px"] for w in xs), max(w["qr_px"] for w in xs)), "-"])
    return head, trows, contrast


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-measured", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cfg = common.load_cfg()
    rows, wr, b = run(cfg)
    head, trows, contrast = summarize(rows, wr)
    look = cfg["look"]["backend"]
    note = ("Sizes for the waybill's projected outline, as ranges over the places named, the lateral range of the "
            "landing and two yaws, where the camera sees at least half of it; the share visible is the part of that "
            "outline the camera sees with the robot at its start posture, over all those poses (a side label is hidden "
            "where it faces away from the camera); a dash: not in view; n/a: that parcel never carries its label on that face. The wrist rows: the waybill seen head-on from that "
            "distance.")
    common.dump_json(os.path.join(a.out, "observability.json"), {"look": look, "fixed": rows, "wrist": wr})
    common.dump_json(os.path.join(a.out, "summary.json"), {"table_head": head, "table_rows": trows, "note": note,
                                                           "look": look, "contrast": contrast})
    print("| " + " | ".join(head) + " |")
    for r in trows:
        print("| " + " | ".join(r) + " |")
    print("contrast in the %s render:" % look)
    for c in contrast:
        print("  ", c)
    if a.write_measured:
        m = json.load(open(MEASURED)) if os.path.exists(MEASURED) else {}
        m["observability"] = {"table_head": head, "table_rows": trows, "note": note, "look": look,
                              "tool": "tests/review/observability.py"}
        with open(MEASURED, "w") as fh:
            json.dump(m, fh, indent=1)
            fh.write("\n")
        print("wrote", os.path.relpath(MEASURED, common.TASK))
    return 0


if __name__ == "__main__":
    sys.exit(main())
