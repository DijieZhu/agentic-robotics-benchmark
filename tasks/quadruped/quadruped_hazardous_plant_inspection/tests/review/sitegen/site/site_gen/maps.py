"""The maps of a site, all with a legend, a scale bar and the structural grid:
truth_topdown, prior_drawing (drawing.py), diff, traversability, dose_field, link_coverage, route."""

from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Polygon, Rectangle  # noqa: E402

from .trav import MODES, _hull  # noqa: E402

DPI = 150
FIG = (16.0, 10.4)
MAT_COLOR = {"hazard_stripe": (0.93, 0.75, 0.07), "cable_protector": (0.93, 0.75, 0.07), "zone_tape": (0.80, 0.25, 0.65),
             "grating": (0.30, 0.31, 0.33), "pad_paint": (0.93, 0.78, 0.15), "steel_plate": (0.5, 0.5, 0.52),
             "ramp_steel": (0.55, 0.56, 0.57), "floor_base": (0.62, 0.62, 0.6)}
MODE_COL = {"walk": "#2b7bba", "high_step": "#e08a1e", "crouch_walk": "#8e44ad"}
CLASS_COL = {"free": "#7fbf7f", "needs_high_step": "#e08a1e", "needs_crouch": "#8e44ad", "tight": "#e04f39",
             "slow": "#d4b106", "barrier": "#222222"}
TYPE_MARK = {"gauge": "o", "valve_lever": "s", "valve_handwheel": "D", "level": "^", "lamps": "P", "thermal": "*",
             "dose_point": "X", "hotspot": "h"}


# ============================================================================================ loading
class SiteData:
    def __init__(self, site_dir):
        self.dir = site_dir
        self.site = json.load(open(os.path.join(site_dir, "site.json")))
        self.prior = json.load(open(os.path.join(site_dir, "prior.json")))
        self._model = None
        self._polys = None

    @property
    def model(self):
        if self._model is None:
            self._model = mujoco.MjModel.from_xml_path(os.path.join(self.dir, "scene.xml"))
        return self._model

    def geom_polys(self, groups=(0, 1, 3)):
        """Footprint polygons of the compiled geoms: [(poly, rgba, ztop, group, name)] sorted by ztop."""
        if self._polys is None:
            m = self.model
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            out = []
            for i in range(m.ngeom):
                t = m.geom_type[i]
                if t == mujoco.mjtGeom.mjGEOM_PLANE:
                    continue
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                R = d.geom_xmat[i].reshape(3, 3)
                p = d.geom_xpos[i]
                s = m.geom_size[i]
                if t == mujoco.mjtGeom.mjGEOM_BOX:
                    c = np.array([[a * s[0], b * s[1], cc * s[2]] for a in (-1, 1) for b in (-1, 1) for cc in (-1, 1)])
                elif t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                    a = np.linspace(0, 2 * math.pi, 14, endpoint=False)
                    ring = np.stack([s[0] * np.cos(a), s[0] * np.sin(a), np.zeros(14)], 1)
                    c = np.concatenate([ring + [0, 0, s[1]], ring - [0, 0, s[1]]])
                    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                        c = np.concatenate([c, [[0, 0, s[1] + s[0]], [0, 0, -s[1] - s[0]]]])
                elif t in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                    a = np.linspace(0, 2 * math.pi, 12, endpoint=False)
                    rx = s[0]
                    c = np.stack([rx * np.cos(a), rx * np.sin(a), np.zeros(12)], 1)
                    c = np.concatenate([c, [[0, 0, s[0]]]])
                elif t == mujoco.mjtGeom.mjGEOM_MESH:
                    rb = m.geom_rbound[i]
                    a = np.linspace(0, 2 * math.pi, 10, endpoint=False)
                    c = np.stack([rb * 0.6 * np.cos(a), rb * 0.6 * np.sin(a), np.zeros(10)], 1)
                else:
                    continue
                w = p + c @ R.T
                mid = m.geom_matid[i]
                rgba = m.mat_rgba[mid].copy() if mid >= 0 else m.geom_rgba[i].copy()
                mname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mid) if mid >= 0 else ""
                if mname in MAT_COLOR:
                    rgba[:3] = MAT_COLOR[mname]
                elif mname and mname.startswith(("sign/", "marker", "tagplate", "dial", "floor/")):
                    rgba[:3] = (0.95, 0.95, 0.95)
                out.append((_hull(w[:, :2]), rgba, float(w[:, 2].max()), int(m.geom_group[i]), name))
            out.sort(key=lambda r: r[2])
            self._polys = out
        return [r for r in self._polys if r[3] in groups]

    def trav(self):
        return np.load(os.path.join(self.dir, "trav.npz"))

    def fields(self):
        return np.load(os.path.join(self.dir, "fields.npz"))


# ============================================================================================ plot helpers
def new_ax(sd, title, extent=None):
    fig, ax = plt.subplots(figsize=FIG, dpi=DPI)
    bb = extent or sd.site["building"]["bbox"]
    ax.set_xlim(bb[0] - 0.8, bb[2] + 3.6)
    ax.set_ylim(bb[1] - 1.6, bb[3] + 1.4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, loc="left", fontweight="bold")
    ax.set_xlabel("x (m, site frame: into the building)")
    ax.set_ylabel("y (m)")
    ax.tick_params(labelsize=8)
    return fig, ax


def grid_lines(ax, grid, bb, color="0.55", bubbles=True):
    for ln in grid["x_lines"]:
        ax.plot([ln["x"], ln["x"]], [bb[1] - 0.6, bb[3] + 0.6], ls=(0, (6, 3, 1, 3)), lw=0.6, color=color, zorder=1)
        if bubbles:
            ax.add_patch(Circle((ln["x"], bb[3] + 0.95), 0.36, fc="white", ec=color, lw=0.8, zorder=20))
            ax.text(ln["x"], bb[3] + 0.95, ln["label"], ha="center", va="center", fontsize=8, zorder=21)
    for ln in grid["y_lines"]:
        ax.plot([bb[0] - 0.6, bb[2] + 0.6], [ln["y"], ln["y"]], ls=(0, (6, 3, 1, 3)), lw=0.6, color=color, zorder=1)
        if bubbles:
            ax.add_patch(Circle((bb[0] - 0.95 + 0.0, ln["y"]), 0.36, fc="white", ec=color, lw=0.8, zorder=20))
            ax.text(bb[0] - 0.95, ln["y"], ln["label"], ha="center", va="center", fontsize=8, zorder=21)


def scale_bar(ax, bb, length=5.0):
    x0, y0 = bb[0] + 0.2, bb[1] - 1.25
    for k in range(5):
        ax.add_patch(Rectangle((x0 + k * length / 5, y0), length / 5, 0.18, fc="k" if k % 2 == 0 else "white",
                               ec="k", lw=0.6, zorder=30))
    ax.text(x0, y0 + 0.32, "0", fontsize=7, ha="center", zorder=30)
    ax.text(x0 + length, y0 + 0.32, "%g m" % length, fontsize=7, ha="center", zorder=30)


def north_arrow(ax, bb):
    x, y = bb[2] + 2.6, bb[3] - 1.2
    ax.annotate("", xy=(x, y + 0.9), xytext=(x, y - 0.4), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="k"))
    ax.text(x, y + 1.15, "N", ha="center", fontsize=11, fontweight="bold")


def draw_walls(ax, site, color="0.15", lw=0, alpha=1.0, z=10):
    """Wall segments between their openings; closed doors as red bars; partitions (temporary shielding)."""
    ops = {}
    for d in site["doors"]:
        if d.get("kind") == "fire" or not d.get("wall"):
            continue
        ops.setdefault(d["wall"], []).append(d)
    for w in site["walls"]:
        ext = 0.12
        cur = w["lo"] - ext
        segs = []
        for d in sorted(ops.get(w["id"], []), key=lambda d: d["center"]):
            a_, b_ = d["center"] - d["width"] / 2, d["center"] + d["width"] / 2
            segs.append((cur, a_))
            cur = b_
            if d["state"] == "closed" or d["kind"] in ("chase_mouth",):
                col = "#c0392b" if d["state"] == "closed" else "#8e44ad"
                if w["axis"] == "x":
                    r = (w["coord"] - 0.06, a_, 0.12, b_ - a_)
                else:
                    r = (a_, w["coord"] - 0.06, b_ - a_, 0.12)
                ax.add_patch(Rectangle(r[:2], r[2], r[3], fc=col, ec="none", zorder=z + 0.6, alpha=0.9))
        segs.append((cur, w["hi"] + ext))
        for a_, b_ in segs:
            if b_ - a_ < 1e-3:
                continue
            if w["axis"] == "x":
                r = (w["coord"] - w["t_neg"], a_, w["t_neg"] + w["t_pos"], b_ - a_)
            else:
                r = (a_, w["coord"] - w["t_neg"], b_ - a_, w["t_neg"] + w["t_pos"])
            ax.add_patch(Rectangle(r[:2], r[2], r[3], fc=color, ec="none", alpha=alpha, zorder=z))
    for p in site.get("partitions", []):
        legr = next(r for r in site["rooms"] if r["id"] == p["leg"])
        x = p["x"]
        ax.add_patch(Rectangle((x - p["thickness"] / 2, legr["rect"][1]), p["thickness"], legr["rect"][3] - legr["rect"][1],
                               fc="#1f4e79", ec="none", zorder=z + 0.7, alpha=0.95))


def legend(ax, handles, loc="upper right", title=None, ncol=1, fs=8):
    lg = ax.legend(handles=handles, loc=loc, fontsize=fs, title=title, title_fontsize=fs + 1, ncol=ncol,
                   framealpha=0.93, bbox_to_anchor=(1.0, 1.0) if loc == "upper right" else None)
    lg.set_zorder(50)
    return lg


def room_labels(ax, site, fs=7, color="0.15", only_drawn=False):
    for r in site["rooms"]:
        if r["type"] == "corridor":
            continue
        x0, y0, x1, y1 = r["rect"]
        txt = "%s\n%s" % (r["number"], r["name"])
        ax.text(0.5 * (x0 + x1), y1 - 0.45, txt, ha="center", va="top", fontsize=fs, color=color, zorder=40,
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.6))


def finish(fig, ax, sd, path, legend_handles=None, legend_title=None):
    bb = sd.site["building"]["bbox"]
    grid_lines(ax, sd.site["building"]["grid"], bb)
    scale_bar(ax, bb)
    north_arrow(ax, bb)
    if legend_handles:
        legend(ax, legend_handles, title=legend_title)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ============================================================================================ maps
def truth_topdown(sd, path):
    site = sd.site
    fig, ax = new_ax(sd, "Ground truth, top-down (seed %d)" % site["seed"])
    polys = sd.geom_polys((0, 1, 3))
    pc = PolyCollection([p[0] for p in polys], facecolors=[p[1][:3] for p in polys],
                        edgecolors=[tuple(np.array(p[1][:3]) * 0.6) for p in polys], linewidths=0.25, zorder=5)
    ax.add_collection(pc)
    # element outlines by class
    for e in site["elements"]:
        if e["kind"] in ("narrow_gap",) or e["class"] == "free":
            continue
        ax.add_patch(Polygon(e["footprint"], closed=True, fill=False, ec=CLASS_COL[e["class"]], lw=1.2, ls="-",
                             zorder=12))
    _ips(ax, site, numbers=True)
    for d in site["decoys"]:
        x, y = d["pose"]["pos"][:2]
        ax.plot(x, y, marker=TYPE_MARK.get(d["type"], "o"), ms=4, mfc="0.8", mec="0.3", zorder=13)
    for s in site["sources"]:
        x, y = s["pos"][:2]
        ax.plot(x, y, marker=(3, 2, 0), ms=10 if not s["documented"] else 7, color="#c2185b" if not s["documented"]
                else "#7b1fa2", mew=2, zorder=14)
    for m in site["grid_markers"]:
        if m["present"]:
            ax.plot(m["pos"][0], m["pos"][1], "s", ms=4, mfc="k", mec="w", zorder=14)
    bs = site["link"]["base_station"]["pos"]
    ax.plot(bs[0], bs[1], "^", ms=9, mfc="#27ae60", mec="k", zorder=15)
    for r in site["link"]["repeaters"]:
        ax.plot(r["pos"][0], r["pos"][1], "^", ms=8, mfc="#27ae60" if r["working"] else "#bbbbbb", mec="k", zorder=15)
    room_labels(ax, site)
    h = [Patch(fc=CLASS_COL[c], label="element: %s" % c) for c in ("needs_high_step", "needs_crouch", "slow",
                                                                  "barrier")]
    h += [Line2D([], [], marker=TYPE_MARK[t], ls="", color="#c0392b", mfc="#f1c40f", label="IP: %s" % t)
          for t in sorted({i["type"] for i in site["ips"]})]
    h += [Line2D([], [], marker="o", ls="", mfc="0.8", mec="0.3", label="decoy instrument"),
          Line2D([], [], marker=(3, 2, 0), ls="", color="#7b1fa2", label="documented source"),
          Line2D([], [], marker=(3, 2, 0), ls="", color="#c2185b", ms=10, label="unlisted source"),
          Line2D([], [], marker="s", ls="", mfc="k", mec="w", label="grid marker (present)"),
          Line2D([], [], marker="^", ls="", mfc="#27ae60", mec="k", label="base station / repeater")]
    finish(fig, ax, sd, path, h)


def _ips(ax, site, numbers=True, order=None):
    for k, ip in enumerate(site["ips"]):
        x, y = ip["pose"]["pos"][:2]
        ax.plot(x, y, marker=TYPE_MARK.get(ip["type"], "o"), ms=11, mfc="#f1c40f", mec="#c0392b", mew=1.6, zorder=16)
        lab = ip["tag"] if order is None else "%d  %s" % (order.index(ip["tag"]) + 1, ip["tag"])
        ax.text(x + 0.3, y + 0.3, lab, fontsize=7.5, fontweight="bold", color="#7b241c", zorder=17,
                bbox=dict(fc="white", ec="#c0392b", lw=0.5, alpha=0.85, pad=0.8))


def traversability(sd, path):
    site = sd.site
    z = sd.trav()
    res = float(z["res"])
    x0, y0 = [float(v) for v in z["origin"]]
    ew, eh, ec = z["enter_walk"], z["enter_high_step"], z["enter_crouch_walk"]
    slow, tw = z["slow"], z["tight_walk"]
    req = np.zeros(ew.shape, np.uint8)
    req[ec & ~ew & ~eh] = 5
    req[eh & ~ew] = 4
    req[ew] = 1
    req[ew & slow] = 2
    req[ew & tw] = 3
    req[~z["inside"]] = 6
    fp = sd.site.get("robot_footprint", [0.55, 0.33])
    fig, ax = new_ax(sd, "Traversability by required mode (0.05 m grid, footprint %.2f x %.2f m, r_in %.3f m)" % (
        fp[0], fp[1], sd.site["thresholds"]["r_in"]))
    cols = ["#3d3d3d", "#cfe8c8", "#f3dd6b", "#f08a6b", "#f2b25c", "#b991d6", "#ffffff"]
    cmap = ListedColormap(cols)
    ny, nx = req.shape
    ax.imshow(req, origin="lower", extent=[x0, x0 + nx * res, y0, y0 + ny * res], cmap=cmap,
              norm=BoundaryNorm(np.arange(-0.5, 7.5, 1), cmap.N), interpolation="nearest", zorder=2)
    draw_walls(ax, site, color="0.1", alpha=0.85)
    for e in site["elements"]:
        if e["class"] in ("needs_high_step", "needs_crouch", "tight", "slow"):
            ax.add_patch(Polygon(e["footprint"], closed=True, fill=False, ec=CLASS_COL[e["class"]], lw=0.9, zorder=12))
    _ips(ax, site)
    room_labels(ax, site, fs=6)
    h = [Patch(fc=cols[1], label="walk (free)"), Patch(fc=cols[2], label="walk, slow (ramp / puddle)"),
         Patch(fc=cols[3], label="walk, tight (no turning)"), Patch(fc=cols[4], label="high_step only"),
         Patch(fc=cols[5], label="crouch_walk only"), Patch(fc=cols[0], label="not enterable (footprint)")]
    finish(fig, ax, sd, path, h, "required mode")


def dose_field(sd, path):
    site = sd.site
    f = sd.fields()
    xs, ys, D = f["dose_xs"], f["dose_ys"], f["dose_truth"]
    fig, ax = new_ax(sd, "Dose-rate field at %.2f m (uSv/h, truth) with the zoning of the old survey" %
                     site["radiation"]["height"])
    inside = f["inside"]
    Dm = np.where(inside, D, np.nan)
    im = ax.imshow(Dm, origin="lower", extent=[xs[0] - 0.05, xs[-1] + 0.05, ys[0] - 0.05, ys[-1] + 0.05],
                   cmap="inferno", norm=LogNorm(vmin=0.1, vmax=max(1000.0, float(np.nanmax(Dm)))),
                   interpolation="bilinear", zorder=2, alpha=0.95)
    draw_walls(ax, site, color="0.75", alpha=0.9)
    hatch = {"supervised": "", "controlled": "//", "high_radiation": "xx"}
    zc = {"supervised": "#2ecc71", "controlled": "#f1c40f", "high_radiation": "#e74c3c"}
    for zn in sd.prior["zoning"]["zones"]:
        if zn["zone_class"] == "supervised":
            continue
        ax.add_patch(Polygon(zn["polygon"], closed=True, fill=False, hatch=hatch[zn["zone_class"]], ec=zc[zn["zone_class"]],
                             lw=1.5, zorder=12))
    for s in site["sources"]:
        x, y = s["pos"][:2]
        ax.plot(x, y, marker=(3, 2, 0), ms=12, color="#00e5ff" if not s["documented"] else "white", mew=2, zorder=14)
        if not s["documented"]:
            ax.text(x + 0.35, y + 0.35, "unlisted: %s\n%.0f uSv/h @ 1 m" % (s["kind"], s["strength"]), color="#00e5ff",
                    fontsize=7.5, zorder=15, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("uSv/h")
    h = [Patch(fc="none", ec=zc["controlled"], hatch="//", label="old zoning: controlled (> 2.5 uSv/h)"),
         Patch(fc="none", ec=zc["high_radiation"], hatch="xx", label="old zoning: high radiation (> 25 uSv/h)"),
         Line2D([], [], marker=(3, 2, 0), ls="", color="w", mec="k", label="documented source (drum)"),
         Line2D([], [], marker=(3, 2, 0), ls="", color="#00e5ff", label="unlisted source (zoning out of date)")]
    room_labels(ax, site, fs=6)
    finish(fig, ax, sd, path, h)


def link_coverage(sd, path):
    site = sd.site
    f = sd.fields()
    cov = f["link"]
    inside = f["inside"]
    x0, y0 = [float(v) for v in f["link_origin"]]
    res = float(f["link_res"])
    img = np.zeros(cov.shape, np.uint8)
    img[inside] = 1
    img[cov] = 2
    fig, ax = new_ax(sd, "Link coverage (line of sight to the base station within %.0f m, or to the working "
                     "repeater within %.0f m)" % (site["link"]["base_range"], site["link"]["repeater_range"]))
    ny, nx = img.shape
    cmap = ListedColormap(["#ffffff", "#ececec", "#9fd8a6"])
    ax.imshow(img, origin="lower", extent=[x0, x0 + nx * res, y0, y0 + ny * res], cmap=cmap,
              norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), interpolation="nearest", zorder=2)
    draw_walls(ax, site)
    bs = site["link"]["base_station"]["antenna"]
    ax.plot(bs[0], bs[1], "^", ms=12, mfc="#1e8449", mec="k", zorder=15)
    ax.text(bs[0] + 0.3, bs[1] - 0.6, "BASE", fontsize=8, fontweight="bold", zorder=15)
    for r in site["link"]["repeaters"]:
        ax.plot(r["pos"][0], r["pos"][1], "^", ms=11, mfc="#1e8449" if r["working"] else "#bdbdbd", mec="k", zorder=15)
        ax.text(r["pos"][0] + 0.3, r["pos"][1] + 0.3, "%s (%s)" % (r["id"], "working" if r["working"] else
                "DEAD, drawn in service"), fontsize=8, fontweight="bold", zorder=15,
                color="k" if r["working"] else "#c0392b")
    _ips(ax, site)
    room_labels(ax, site, fs=6)
    h = [Patch(fc="#9fd8a6", label="reports delivered here"), Patch(fc="#ececec", label="no link: reports queued"),
         Line2D([], [], marker="^", ls="", mfc="#1e8449", mec="k", label="base station / working repeater"),
         Line2D([], [], marker="^", ls="", mfc="#bdbdbd", mec="k", label="dead repeater")]
    finish(fig, ax, sd, path, h)


def route_map(sd, path):
    site = sd.site
    R = site["route"]
    fig, ax = new_ax(sd, "Reference route: %.0f m, %.0f s (incl. %.0f s dwell), %.1f uSv, %d rooms" %
                     (R["length_m"], R["total_time_s"], R["dwell_s"], R["dose_total_uSv"], R["n_rooms_visited"]))
    polys = sd.geom_polys((0, 3))
    pc = PolyCollection([p[0] for p in polys], facecolors=[tuple(0.55 + 0.45 * np.array(p[1][:3])) for p in polys],
                        edgecolors="none", zorder=4)
    ax.add_collection(pc)
    draw_walls(ax, site)
    for leg in R["legs"]:
        for s in leg["segments"]:
            w = np.array(s["waypoints"])
            if len(w) < 2:
                continue
            ax.plot(w[:, 0], w[:, 1], color=MODE_COL[s["mode"]], lw=2.6 if s["mode"] != "walk" else 1.8,
                    solid_capstyle="round", zorder=18)
            if "tight" in s["classes"] or "slow" in s["classes"]:
                pass
    # mark cells where the route meets tight / slow
    z = sd.trav()
    res = float(z["res"])
    x0, y0 = [float(v) for v in z["origin"]]
    for leg in R["legs"]:
        for s in leg["segments"]:
            for x, y in s["waypoints"]:
                iy, ix = int((y - y0) / res), int((x - x0) / res)
                if z["slow"][iy, ix]:
                    ax.plot(x, y, "o", ms=3, color=CLASS_COL["slow"], zorder=19)
    order = R["order"]
    _ips(ax, site, order=order)
    for ip in site["ips"]:
        p = ip["observation_pose"]["pos"]
        ax.plot(p[0], p[1], ".", ms=5, color="k", zorder=20)
        ax.plot([p[0], ip["pose"]["pos"][0]], [p[1], ip["pose"]["pos"][1]], ":", color="k", lw=0.7, zorder=19)
    ax.plot(0, 0, marker="s", ms=12, mfc="#f1c40f", mec="k", zorder=21)
    ax.text(0.4, -0.6, "PAD", fontsize=8, fontweight="bold", zorder=21)
    room_labels(ax, site, fs=6)
    h = [Line2D([], [], color=MODE_COL[m], lw=2.5, label="mode: %s" % m) for m in MODES]
    h += [Line2D([], [], marker="o", ls="", color=CLASS_COL["slow"], label="slow cells on the route"),
          Line2D([], [], marker=".", ls="", color="k", label="observation pose")]
    txt = "visiting order: " + " > ".join("%d %s" % (k + 1, t) for k, t in enumerate(order))
    txt += "\nclasses used: " + ", ".join(R["classes_used"])
    ax.text(0.01, 0.01, txt, transform=ax.transAxes, fontsize=8, va="bottom", zorder=40,
            bbox=dict(fc="white", ec="0.5", alpha=0.9))
    finish(fig, ax, sd, path, h)


def diff_map(sd, path):
    from .drawing import draw_prior
    site = sd.site
    fig, ax = new_ax(sd, "Drawing vs reality: the discrepancies (numbered)")
    polys = sd.geom_polys((0,))
    pc = PolyCollection([p[0] for p in polys if p[4].startswith(("wall/", "partition/", "block/", "door/"))],
                        facecolors="#1f4e79", edgecolors="none", alpha=0.75, zorder=4)
    ax.add_collection(pc)
    draw_prior(ax, sd.prior, style="overlay")
    for d in site["discrepancies"]:
        x, y = d["where"]
        ax.add_patch(Circle((x, y), 0.75, fc="none", ec="#e74c3c", lw=2.2, zorder=30))
        ax.text(x + 0.8, y + 0.55, str(d["id"]), fontsize=12, color="#e74c3c", fontweight="bold", zorder=31,
                bbox=dict(fc="white", ec="#e74c3c", lw=0.8, pad=1.0))
    for ip in site["ips"]:
        doc = ip.get("documented")
        if ip.get("discrepancy") in ("ip_moved", "ip_relocated") and isinstance(doc, dict) and doc.get("pos"):
            ax.annotate("", xy=tuple(ip["pose"]["pos"][:2]), xytext=tuple(doc["pos"][:2]), zorder=29,
                        arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=1.6, ls="--"))
            ax.text(doc["pos"][0], doc["pos"][1] - 0.6, "documented", color="#e74c3c", fontsize=7, zorder=29,
                    ha="center")
    if site.get("missing_sheet"):
        r = site["missing_sheet"]["rect"]
        ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], fc="#e74c3c", alpha=0.07, ec="#e74c3c",
                               ls="--", lw=1.2, zorder=3))
    lines = ["%d  %s: %s" % (d["id"], d["kind"], d["text"]) for d in site["discrepancies"]]
    ax.text(0.01, 0.01, "\n".join(lines), transform=ax.transAxes, fontsize=7.5, va="bottom", zorder=40,
            bbox=dict(fc="white", ec="#e74c3c", alpha=0.92))
    h = [Line2D([], [], color="k", lw=1.0, label="old drawing (line work)"),
         Patch(fc="#1f4e79", alpha=0.75, label="reality: walls, partitions, door leaves, stored material"),
         Line2D([], [], marker="o", ls="", mfc="none", mec="#e74c3c", ms=12, label="discrepancy")]
    _ips(ax, site)
    finish(fig, ax, sd, path, h)


def make_maps(site_dir):
    sd = SiteData(site_dir)
    md = os.path.join(site_dir, "maps")
    os.makedirs(md, exist_ok=True)
    from .drawing import prior_drawing
    truth_topdown(sd, os.path.join(md, "truth_topdown.png"))
    prior_drawing(sd.prior, os.path.join(md, "prior_drawing.png"), sd.site["building"]["bbox"])
    diff_map(sd, os.path.join(md, "diff.png"))
    traversability(sd, os.path.join(md, "traversability.png"))
    dose_field(sd, os.path.join(md, "dose_field.png"))
    link_coverage(sd, os.path.join(md, "link_coverage.png"))
    route_map(sd, os.path.join(md, "route.png"))
    if sd.site.get("layout") == "ladder_v1":
        from .sheets import make_ladder_maps
        make_ladder_maps(site_dir)
    return md
