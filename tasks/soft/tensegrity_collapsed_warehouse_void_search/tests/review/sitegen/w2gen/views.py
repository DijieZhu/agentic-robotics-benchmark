"""Figures and the short film of a W2 site for the owner (blockout look: plain boxes, MuJoCo classic renderer).

The robot, when the simulation base can be imported and its decimated visual copy of the authors' strut mesh is
present, is placed STATICALLY on the pad: ysim builds the model (three copies of the authors' strut mesh, the
authors' initial node positions), the model is lifted onto the floor and only mj_forward is called (no step).
Otherwise the figures have no robot. Nothing here models, redraws or completes any part of the robot.
"""
import math
import os
import subprocess
import sys

import numpy as np

from . import family as fam
from . import grid as gridmod
from . import mjcf
from . import predicates as prd

YALE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "yale"))
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
C_SP, C_PAD, C_ORANGE, C_PLUG = "#2a78d6", "#1baf7a", "#eb6834", "#c0392b"


# ------------------------------------------------------------------------------------------------ models
def _geom_zmin(m, d, g):
    import mujoco
    t = m.geom_type[g]
    p = d.geom_xpos[g]
    R = d.geom_xmat[g].reshape(3, 3)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return p[2] - m.geom_size[g][0]
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r, h = m.geom_size[g][0], m.geom_size[g][1]
        ax = R[:, 2]
        return p[2] - abs(ax[2]) * h - r * math.sqrt(max(0.0, 1 - ax[2] ** 2))
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        V = m.mesh_vert[v0:v0 + nv]
        return float((V @ R.T + p)[:, 2].min())
    return p[2]


def robot_world(site, offscreen=(1600, 1000), yaw_deg=35.0, markers=""):
    """(m, d, info): the site merged into the simulation base with the robot placed statically on the pad, or
    (m, d, info) of the bare site when the base or its decimated meshes are not available."""
    import mujoco
    info = dict(robot=False)
    dec = os.path.join("ysim_assets", "dec")
    try:
        if YALE_DIR not in sys.path:
            sys.path.insert(0, YALE_DIR)
        from ysim import model as mdl      # noqa: WPS433  (read-only import of the simulation base)
        if not all(os.path.exists(os.path.join(dec, f + ".stl")) for f in ("sock", "endcap", "housing", "rod", "other")):
            raise FileNotFoundError("decimated visual meshes missing in " + dec)
        frag = mjcf.world_fragment(site, include_floor=True, extra=markers)
        cfg = mdl.ModelConfig(world_xml=frag, asset_xml=mjcf.asset_fragment(site), floor=False, offscreen=offscreen,
                              mesh_res="dec")
        poses0 = mdl.initial_pose()
        c0 = np.mean([p["pos"] for p in poses0], axis=0)
        a = math.radians(yaw_deg)
        Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        pxy = site["pad"]["xy"]
        poses = []
        for p in poses0:
            q = dict(p)
            q["pos"] = Rz @ (p["pos"] - c0) + np.array([pxy[0], pxy[1], p["pos"][2]])
            q["R"] = Rz @ p["R"]
            poses.append(q)
        xml, _ = mdl.build_xml(cfg, poses)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        ids = mdl.Ids(m)
        zmin = min(_geom_zmin(m, d, g) for g in ids.col_geoms)
        for adr in ids.jnt_qpos:
            d.qpos[adr + 2] += 0.0005 - zmin
        mujoco.mj_forward(m, d)
        mjcf.apply_site_lighting(m, site)
        info = dict(robot=True, how="ysim.model.build_xml(ModelConfig(world_xml=site fragment, floor=False, mesh_res='dec')), "
                                    "authors' initial node positions, yaw %.0f deg about the pad, lifted onto the floor, "
                                    "mj_forward only (no step)" % yaw_deg, ngeom=int(m.ngeom))
        return m, d, info
    except Exception as exc:          # noqa: BLE001  (fallback: no robot in the figures)
        info = dict(robot=False, reason="%s: %s" % (type(exc).__name__, exc))
    xml = mjcf.site_xml(site, offscreen=offscreen)
    if markers:
        xml = xml.replace("  </worldbody>", markers + "\n  </worldbody>")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, info


def point_markers(site):
    """Development markers for the oblique views only (never in site.xml, never in the agent's camera)."""
    L = []
    px, py = site["pad"]["xy"]
    L.append('<geom name="devmark__pad" type="cylinder" size="0.60 0.0015" pos="%s" rgba="0.11 0.69 0.48 0.35" '
             'contype="0" conaffinity="0" group="4"/>' % mjcf._v([px, py, 0.0035]))
    for s in site["search_points"]:
        x, y = s["xy"]
        L.append('<geom name="devmark__sp%d" type="cylinder" size="%.3f 0.0015" pos="%s" rgba="0.16 0.47 0.84 0.45" '
                 'contype="0" conaffinity="0" group="4"/>' % (s["id"], fam.BL, mjcf._v([x, y, 0.0035])))
        L.append('<geom name="devmark__sp%d_pin" type="cylinder" size="0.012 0.30" pos="%s" rgba="0.16 0.47 0.84 1" '
                 'contype="0" conaffinity="0" group="4"/>' % (s["id"], mjcf._v([x, y, 0.30])))
    return "\n".join("    " + s for s in L)


# ------------------------------------------------------------------------------------------------ rendering
def _renderer(m, h, w):
    import mujoco
    return mujoco.Renderer(m, h, w)


def _project_free(m, lookat, az, el, dist, P, W, H):
    """Pixel position of world points in an image rendered with a MuJoCo free camera (vertical fovy of the model)."""
    ca, sa = math.cos(math.radians(az)), math.sin(math.radians(az))
    ce, se = math.cos(math.radians(el)), math.sin(math.radians(el))
    fwd = np.array([ce * ca, ce * sa, se])
    up = np.array([-se * ca, -se * sa, ce])
    right = np.cross(fwd, up)
    pos = np.asarray(lookat, float) - dist * fwd
    q = np.asarray(P, float) - pos
    z = q @ fwd
    f = (H / 2) / math.tan(math.radians(m.vis.global_.fovy) / 2)
    return W / 2 + (q @ right) / z * f, H / 2 - (q @ up) / z * f


def render_free(m, d, lookat, az, el, dist, W, H, groups=(1, 1, 1, 0, 1, 0), labels=None, hide=()):
    import mujoco
    r = _renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = list(groups)
    saved = []
    for g in hide:
        saved.append((g, m.geom_rgba[g].copy()))
        m.geom_rgba[g, 3] = 0.0
    r.update_scene(d, camera=cam, scene_option=opt)
    img = r.render().copy()
    uv = None
    if labels:
        uv = _project_free(m, lookat, az, el, dist, [p for _t, p in labels], W, H)
    for g, rgba in saved:
        m.geom_rgba[g] = rgba
    r.close()
    return img, uv


def render_search(m, d, W=640, H=480):
    import mujoco
    r = _renderer(m, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    r.update_scene(d, camera="search", scene_option=opt)
    img = r.render().copy()
    r.close()
    return img


def _annotate(img, labels, uv, path, scale=1.0, title=None):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(19 * scale))
        font2 = ImageFont.truetype("DejaVuSans.ttf", int(17 * scale))
    except OSError:
        font = font2 = ImageFont.load_default()
    if labels is not None and uv is not None:
        for (text, _p), u, v in zip(labels, uv[0], uv[1]):
            if not (0 <= u < im.width and 0 <= v < im.height):
                continue
            bb = dr.textbbox((u + 8, v - 26), text, font=font)
            dr.rectangle((bb[0] - 4, bb[1] - 3, bb[2] + 4, bb[3] + 3), fill=(252, 252, 251))
            dr.text((u + 8, v - 26), text, fill=(11, 11, 11), font=font)
            dr.line((u, v, u + 8, v - 12), fill=(11, 11, 11), width=2)
    if title:
        bb = dr.textbbox((14, 12), title, font=font2)
        dr.rectangle((bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4), fill=(252, 252, 251))
        dr.text((14, 12), title, fill=(11, 11, 11), font=font2)
    im.save(path)


def labels_for(site, with_void=True):
    lb = [("pad", (site["pad"]["xy"][0], site["pad"]["xy"][1], 0.0))]
    for s in site["search_points"]:
        lb.append(("SP%d" % s["id"] + (" (under the slab)" if s["id"] == 3 else ""), (s["xy"][0], s["xy"][1], 0.6 if s["id"] < 3 else 0.55)))
    if with_void:
        for g in site["r2"]["openings"]:
            name = {"M": "true mouth", "Bm": "blind mouth", "X": "decoy"}[g["id"]]
            lb.append((name, (0.5 * (g["x0"] + g["x1"]), g["y0"] - 0.06, 0.45)))
    return lb


# ------------------------------------------------------------------------------------------------ top view
def top_view(site, qc, g9, path_png, title_extra=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    lx, ly = site["size_m"]
    xml = mjcf.site_xml(site)
    m, d = prd.compile_world(xml)
    cpos, _R = prd.cam_pose(site)
    G = gridmod.Grid(site, 0.04)
    vis = prd.visible(m, d, cpos, np.c_[G.X.ravel(), G.Y.ravel(), np.full(G.X.size, 0.20)]).reshape(G.X.shape)
    fig = plt.figure(figsize=(16.0, 9.6), dpi=110, facecolor=SURF)
    ax = fig.add_axes([0.04, 0.06, 0.56, 0.86])
    ax.set_facecolor("#efeeea")
    ax.set_aspect("equal")
    hid = np.ma.masked_where(vis, np.ones_like(vis, float))
    ax.pcolormesh(np.r_[G.xs - 0.02, G.xs[-1] + 0.02], np.r_[G.ys - 0.02, G.ys[-1] + 0.02], hid, cmap="Greys", vmin=0,
                  vmax=2.4, shading="flat", zorder=1)
    for p in site["patches"]:
        ax.add_patch(Rectangle((p["x0"], p["y0"]), p["x1"] - p["x0"], p["y1"] - p["y0"], facecolor="none", edgecolor=INK2,
                               hatch="..." if p["name"] == "gravel" else "////", lw=0.8, zorder=2))
        ax.text(p["x1"] - 0.03, p["y0"] + 0.03, "%s patch, mu %.2f" % (p["name"], p["mu"]), fontsize=7.2, color=INK2, zorder=6,
                va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.12", fc="#efeeea", ec="none"))
    for b in site["boxes"]:
        if b["kind"] in ("roof", "wall"):
            continue
        a = b["aabb"]
        gv = 0.82 - 0.50 * min(a[5], 0.9) / 0.9
        fc = (gv, gv, gv * 0.985)
        ec, hatch = INK2, None
        if b["kind"] == "crate":
            fc = tuple(b["rgb"])
        if b["kind"] == "plug":
            fc, ec, hatch = "#f2c4bd", C_PLUG, "\\\\\\"
        ax.add_patch(Rectangle((a[0], a[2]), a[1] - a[0], a[3] - a[2], facecolor=fc, edgecolor=ec, lw=0.5, hatch=hatch, zorder=3))
        if b["kind"] == "rubble" and a[1] - a[0] > 0.33 and a[3] - a[2] > 0.2:
            ax.text(0.5 * (a[0] + a[1]), 0.5 * (a[2] + a[3]), "%.2f" % a[5], fontsize=6.5, color=SURF if gv < 0.55 else INK2,
                    ha="center", va="center", zorder=6)
    for c in site["chips"]:
        ax.plot([c["xy"][0]], [c["xy"][1]], marker="o" if c["shape"] == "disc" else "s", ms=3.6, color=tuple(c["rgb"]),
                mec=INK, mew=0.4, zorder=7)
    for b in site["boxes"]:
        if b["kind"] != "roof":
            continue
        a = b["aabb"]
        main = b["name"] == "roof__void_slab"
        ax.add_patch(Rectangle((a[0], a[2]), a[1] - a[0], a[3] - a[2], facecolor="none", edgecolor=INK, hatch="xx" if not main else None,
                               lw=2.0 if main else 1.0, zorder=5, alpha=0.9))
    gal = site["void"]["gallery"]
    ax.add_patch(Rectangle((gal["x0"], gal["y0"]), gal["x1"] - gal["x0"], gal["y1"] - gal["y0"], facecolor="#d9e7f7", alpha=0.55,
                           edgecolor=C_SP, lw=1.4, ls="--", zorder=4))
    ax.text(gal["x0"] + 0.04, gal["y1"] - 0.04, "void gallery under the slab (clear %.2f m, %.2f m deep)" % (gal["clear_h"], gal["y1"] - gal["y0"]),
            fontsize=7.6, color=INK, va="top", ha="left", zorder=8, bbox=dict(boxstyle="round,pad=0.12", fc=SURF, ec="none", alpha=0.85))
    slab = site["void"]["slab"]
    for row in ("r1", "r2"):
        for g in site[row]["openings"]:
            col = {"P": C_SP, "T": C_ORANGE, "XN": INK2, "XL": INK2, "B": C_PLUG}[g["cls"]]
            w = g["x1"] - g["x0"]
            if row == "r1":
                lab = {"gap_P": "open gap P\n%.2f w", "tunnel_P": "tunnel P (%s roof)\n%.2f w x %.2f h",
                       "gap_T": "tight gap T\n%.2f w", "gap_XN": "narrow decoy XN\n%.2f w",
                       "slot_XL": "low slot XL\n%.2f w x %.2f h"}[g["type"]]
                if g["type"] == "tunnel_P":
                    lab = lab % (g["roof"], w, g["clear_h"])
                elif g["type"] == "slot_XL":
                    lab = lab % (w, g["clear_h"])
                else:
                    lab = lab % w
            elif g["id"] == "M":
                lab = "TRUE mouth M\n%.2f w x %.2f h" % (w, g["clear_h"])
            elif g["id"] == "Bm":
                lab = "BLIND mouth Bm\nplug at %.2f m" % g["plug_depth_m"]
            elif g["cls"] == "XN":
                lab = "decoy XN %.2f w%s" % (w, " (under the slab)" if g["covered"] else "")
            else:
                lab = "decoy XL %.2f w x %.2f h" % (w, g["clear_h"])
            ax.plot([g["x0"], g["x1"]], [g["y0"], g["y0"]], color=col, lw=3.4, solid_capstyle="butt", zorder=7)
            ax.text(0.5 * (g["x0"] + g["x1"]), g["y0"] - 0.05, lab, fontsize=7.0, color=INK, ha="center", va="top", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.16", fc=SURF, ec=col, lw=1.0))
    if g9 is not None:
        for k, leg in enumerate(g9["a"]["legs"]):
            if leg["path"]:
                pa = np.asarray(leg["path"])
                ax.plot(pa[:, 0], pa[:, 1], color=INK, lw=1.2, ls=(0, (4, 2)), zorder=6)
        lv = qc.get("G8b", {}).get("last_visible_xy")
        leg = g9["a"]["legs"][2]
        if lv and leg["path"]:
            pa = np.asarray(leg["path"])
            i = int(np.argmin(np.hypot(pa[:, 0] - lv[0], pa[:, 1] - lv[1])))
            ax.plot(pa[i:, 0], pa[i:, 1], color=C_ORANGE, lw=2.6, zorder=6.5)
            ax.plot([lv[0]], [lv[1]], marker="D", ms=6, color=C_ORANGE, mec=SURF, zorder=9)
    pad = site["pad"]
    ax.add_patch(Circle(pad["xy"], pad["retrieval_radius_m"], facecolor="none", edgecolor=C_PAD, lw=1.6, ls="--", zorder=7))
    ax.plot([pad["xy"][0]], [pad["xy"][1]], marker="s", ms=9, color=C_PAD, mec=SURF, mew=1.5, zorder=9)
    ax.text(pad["xy"][0] + 0.09, pad["xy"][1] + 0.02, "pad", fontsize=8.5, color=INK, fontweight="bold", zorder=9,
            bbox=dict(boxstyle="round,pad=0.12", fc=SURF, ec="none", alpha=0.85))
    for s in site["search_points"]:
        x, y = s["xy"]
        ax.add_patch(Circle((x, y), fam.BL, facecolor="none", edgecolor=C_SP, lw=1.6, zorder=8))
        ax.add_patch(Circle((x, y), 2 * fam.BL, facecolor="none", edgecolor=C_SP, lw=0.8, ls=":", zorder=8))
        ax.plot([x], [y], marker="o", ms=8, color=C_SP, mec=SURF, mew=1.5, zorder=9)
        ax.text(x + 0.07, y + 0.07, "SP%d (w %.2f)" % (s["id"], s["weight"]), fontsize=8.5, color=INK, fontweight="bold", zorder=10,
                bbox=dict(boxstyle="round,pad=0.12", fc=SURF, ec="none", alpha=0.85))
    cp = site["camera"]["true"]["pos"]
    ax.plot([cp[0]], [cp[1]], marker="v", ms=10, color=INK, zorder=10)
    ax.text(cp[0], cp[1] - 0.09, "camera", fontsize=7.6, color=INK, va="top", ha="center", zorder=10)
    for name, (ya, yb) in site["bands"].items():
        ax.text(lx + 0.03, 0.5 * (ya + yb), {"A": "band A", "R1": "ridge R1", "B": "band B", "R2_front": "R2 front row",
                                             "gallery": "gallery"}[name], fontsize=8, color=INK2, va="center", rotation=90)
    ax.set_xlim(-0.05, lx + 0.18)
    ax.set_ylim(-0.05, ly + 0.05)
    ax.set_xlabel("x east [m]", color=INK2, fontsize=9)
    ax.set_ylabel("y north [m] (away from the camera side)", color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    for s_ in ax.spines.values():
        s_.set_color("#c9c8c2")
    ax.set_title("Site family W2, seed %d: top view of the blockout%s (numbers on blocks = height in m)" % (site["seed"], title_extra),
                 fontsize=11, color=INK, loc="left")
    ax3 = fig.add_axes([0.62, 0.06, 0.37, 0.86])
    ax3.axis("off")
    y = 0.985
    items = [("line", C_SP, "class P opening / true mouth: passable (%.2f m)" % site["classes"]["P"]["w"]),
             ("line", C_PLUG, "blind mouth: looks like the true one, solid plug inside"),
             ("line", C_ORANGE, "class T: tight, open top (%.2f m)" % site["classes"]["T"]["w"]),
             ("line", INK2, "decoys XN (%.2f m) / XL (clear %.2f m)" % (site["classes"]["XN"]["w"], site["classes"]["XL"]["clear_h"])),
             ("box", "#f2c4bd", "plug (never seen by the camera, predicate G7)"),
             ("gal", "#d9e7f7", "void gallery under the slab (bold outline = slab)"),
             ("hatch", INK, "R1 tunnel / slot roofs"),
             ("shade", "#9a9a98", "floor where the camera cannot see a point 0.20 m up"),
             ("dash", INK, "G9a grid route pad > SP1 > SP2 > SP3 > pad (disc r %.2f m)" % site["classes"]["plan_radius"]),
             ("thick", C_ORANGE, "blind part: last visible pose (diamond) to SP3"),
             ("circ", C_SP, "search point: full credit within 0.33 m, none at 0.66 m"),
             ("pad", C_PAD, "pad; dashed circle = retrieval zone r 0.60 m"),
             ("dot", "#cc2222", "colour chips (visual only, end-cap colours)")]
    for kind, col, txt in items:
        if kind in ("line", "thick"):
            ax3.plot([0.0, 0.06], [y, y], color=col, lw=3.2, transform=ax3.transAxes, clip_on=False)
        elif kind == "dash":
            ax3.plot([0.0, 0.06], [y, y], color=col, lw=1.3, ls=(0, (4, 2)), transform=ax3.transAxes, clip_on=False)
        elif kind == "hatch":
            ax3.add_patch(Rectangle((0.0, y - 0.012), 0.06, 0.024, facecolor="none", edgecolor=col, hatch="xxx", lw=0.9,
                                    transform=ax3.transAxes, clip_on=False))
        elif kind in ("shade", "box", "gal"):
            ax3.add_patch(Rectangle((0.0, y - 0.012), 0.06, 0.024, facecolor=col, edgecolor=C_PLUG if kind == "box" else "none",
                                    transform=ax3.transAxes, clip_on=False))
        elif kind == "pad":
            ax3.plot([0.03], [y], marker="s", ms=8, color=col, mec=SURF, transform=ax3.transAxes, clip_on=False)
        elif kind == "dot":
            ax3.plot([0.03], [y], marker="o", ms=5, color=col, mec=INK, mew=0.4, transform=ax3.transAxes, clip_on=False)
        else:
            ax3.plot([0.03], [y], marker="o", ms=11, mfc="none", mec=col, mew=1.6, transform=ax3.transAxes, clip_on=False)
        ax3.text(0.085, y, txt, fontsize=8.2, color=INK, va="center", transform=ax3.transAxes)
        y -= 0.034
    y -= 0.015
    v = site["void"]
    lines = ["site %.1f x %.1f m; %d boxes; R1 third opening %s; tunnel roof %s" % (lx, ly, len(site["boxes"]),
             site["r1"]["third"], site["r1"]["tunnel_roof"]),
             "SP3 mode %s; SP3.x - true mouth = %+.2f m" % (v["sp3_mode"], v["sp3_minus_true_mouth_x"]),
             "slab x %.2f-%.2f m, 0.12 thick, top %.2f m" % (slab["x0"], slab["x1"], slab["z1"]),
             "camera %.2f m up, tilt %.1f deg, far %.1f mm/px" % (site["camera"]["true"]["pos"][2], site["camera"]["true"]["tilt_deg"],
                                                                qc["G6"]["mm_per_px_far_centre"]),
             "G3 projection check: max %.2f px (flipped sign %.1f px)" % (qc["G3"]["as_built"]["max_px"], qc["G3"]["flipped_sign"]["max_px"])
             if qc.get("G3", {}).get("as_built") else "G3 not rendered",
             "open floor seen at 0.20 m: %.0f %% (G4)" % (100 * qc["G4"]["visible_fraction_open_floor_0p20"]),
             "G9a route %.1f m; blind into SP3 %.2f m (G8b)" % (qc["G9"]["a"]["total_m"] or 0, qc["G8b"]["blind_path_m"] or 0),
             "cheap stops: max SP3 credit %.3f (G8a, limit 0.30)" % qc["G8a"]["max_credit"],
             "free blocks kept %d (dropped %d); chips %d" % (len(site["free_blocks"]["kept"]), site["free_blocks"]["dropped_for_G9"],
                                                            len(site["chips"])),
             "floor mu %.3f; key light %.0f deg off nominal" % (site["physics"]["floor_mu"], site["lighting"]["key_off_nominal_deg"]),
             "all G predicates: %s" % ("pass" if qc["pass"] else "FAIL " + ",".join(qc["failed"])),
             "envelope: %s" % ", ".join(sorted(set(site["envelope"]["source"].values())))]
    for line in lines:
        ax3.text(0.0, y, line, fontsize=8.0, color=INK2, va="center", transform=ax3.transAxes, family="DejaVu Sans Mono")
        y -= 0.03
    fig.savefig(path_png, facecolor=SURF)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ owner set
def owner_set(site, qc, g9, out_dir, tag, robot=True):
    """Top view, search-camera view (x2 nearest), two obliques, one oblique without the slab."""
    from PIL import Image
    import mujoco
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    p = os.path.join(out_dir, "%s_top_view.png" % tag)
    top_view(site, qc, g9, p)
    files["top_view"] = p
    lx, ly = site["size_m"]
    m, d, info = robot_world(site, offscreen=(1600, 1000), markers="") if robot else (None, None, dict(robot=False))
    if m is None:
        xml = mjcf.site_xml(site, offscreen=(1600, 1000))
        m, d = prd.compile_world(xml)
    img = render_search(m, d)
    p = os.path.join(out_dir, "%s_search_camera_640x480_x2.png" % tag)
    Image.fromarray(img).resize((1280, 960), Image.NEAREST).save(p)
    files["search_camera"] = p
    m2, d2, _info2 = robot_world(site, offscreen=(1600, 1000), markers=point_markers(site)) if robot else (None, None, None)
    if m2 is None:
        xml = mjcf.site_xml(site, offscreen=(1600, 1000)).replace("  </worldbody>", point_markers(site) + "\n  </worldbody>")
        m2, d2 = prd.compile_world(xml)
    look = [lx / 2, ly / 2 - 0.1, 0.15]
    lb = labels_for(site)
    for name, az, el, dist, title in (("oblique_southwest", 55.0, -48.0, 6.6, "oblique from the south-west (camera side)"),
                                      ("oblique_southeast", 125.0, -46.0, 6.6, "oblique from the south-east")):
        img, uv = render_free(m2, d2, look, az, el, dist, 1600, 1000, labels=lb)
        p = os.path.join(out_dir, "%s_%s_1600x1000.png" % (tag, name))
        _annotate(img, lb, uv, p, title="seed %d, %s; markers: green = pad, blue = search points (dev view only)" % (site["seed"], title))
        files[name] = p
    slab_id = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_GEOM, "roof__void_slab")
    plug_id = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_GEOM, "plug__Bm")
    keep = m2.geom_rgba[plug_id].copy()
    m2.geom_rgba[plug_id] = [0.85, 0.25, 0.20, 1.0]
    gal = site["void"]["gallery"]
    look2 = [lx / 2, gal["y0"] - 0.35, 0.1]
    img, uv = render_free(m2, d2, look2, 90.0, -60.0, 4.9, 1600, 1000, labels=lb, hide=(slab_id,))
    m2.geom_rgba[plug_id] = keep
    p = os.path.join(out_dir, "%s_oblique_slab_hidden_1600x1000.png" % tag)
    _annotate(img, lb, uv, p, title="seed %d, the slab hidden: the gallery (void), both mouths, the plug (red) in the blind mouth" % site["seed"])
    files["oblique_slab_hidden"] = p
    return files, info


def film(site, out_mp4, frames_dir=None, fps=30, orbit_s=9.0, search_s=4.0, W=1280, H=720, robot=True):
    """Half orbit around the site, then a cut to the search camera's view (640 x 480 scaled to 960 x 720,
    pillar-boxed). H.264 via the system ffmpeg; without an encoder the frames are written as PNGs."""
    from PIL import Image
    lx, ly = site["size_m"]
    m, d, info = robot_world(site, offscreen=(max(W, 1600), max(H, 1000))) if robot else (None, None, dict(robot=False))
    if m is None:
        m, d = prd.compile_world(mjcf.site_xml(site, offscreen=(max(W, 1600), max(H, 1000))))
    n1, n2 = int(round(orbit_s * fps)), int(round(search_s * fps))
    look = [lx / 2, ly / 2 - 0.1, 0.15]
    frames = []
    for k in range(n1):
        s = k / max(1, n1 - 1)
        e = 0.5 - 0.5 * math.cos(math.pi * s)          # ease in / out
        az = 10.0 + 180.0 * e
        el = -38.0 + 6.0 * math.sin(math.pi * s)
        img, _ = render_free(m, d, look, az, el, 6.6, W, H, groups=(1, 1, 1, 0, 0, 0))
        frames.append(img)
    sv = render_search(m, d)
    big = np.asarray(Image.fromarray(sv).resize((960, 720), Image.NEAREST))
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:, (W - 960) // 2:(W - 960) // 2 + 960] = big
    from PIL import ImageDraw, ImageFont
    im = Image.fromarray(canvas)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    dr.text((W - 300, H - 34), "search camera, 640 x 480", fill=(230, 230, 230), font=font)
    canvas = np.asarray(im)
    frames += [canvas] * n2
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
    try:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(fps),
               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", "-threads", "2", out_mp4]
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in frames:
            pr.stdin.write(np.ascontiguousarray(f).tobytes())
        pr.stdin.close()
        rc = pr.wait()
        if rc != 0:
            raise RuntimeError("ffmpeg exit %d" % rc)
        enc = dict(encoder="ffmpeg libx264 crf 20", frames=len(frames), fps=fps, size=[W, H])
    except Exception as exc:          # noqa: BLE001
        enc = dict(encoder=None, error=str(exc))
        if frames_dir:
            for k, f in enumerate(frames):
                Image.fromarray(f).save(os.path.join(frames_dir, "f%04d.png" % k))
    return dict(robot=info, **enc)
