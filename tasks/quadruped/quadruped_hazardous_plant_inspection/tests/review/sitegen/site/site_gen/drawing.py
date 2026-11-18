"""The old drawing as an engineering drawing (maps/prior_drawing.png), from prior.json only: black line work on
off-white, wall poche hatching, door swings, room names and numbers, grid bubbles, a north arrow, a scale bar, a
title block with the drawing number and an old revision date, the zoning as hatched areas, the documented tag
positions, and a stamp over the missing sheet."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

PAPER = "#f6f1e3"
INK = "#161616"
FONT = "DejaVu Sans Mono"
ZHATCH = {"controlled": ("////", "#6d6d6d"), "high_radiation": ("xxxx", "#3a3a3a")}
TYPE_SYM = {"gauge": "o", "valve_lever": "s", "valve_handwheel": "D", "level": "^", "lamps": "P", "thermal": "*",
            "dose_point": "X", "hotspot": "h"}


def _wall_rect(w):
    (ax_, ay), (bx, by) = w["a"], w["b"]
    t = w["thickness"]
    off = w.get("offset", 0.0)
    if abs(ax_ - bx) < 1e-9:
        lo, hi = min(ay, by), max(ay, by)
        return (ax_ + off - t / 2, lo - 0.12, t, hi - lo + 0.24), "x"
    lo, hi = min(ax_, bx), max(ax_, bx)
    return (lo - 0.12, ay + off - t / 2, hi - lo + 0.24, t), "y"


class Labels:
    """Places drawing labels without overlaps (a review saw tag labels over equipment labels, a room name
    cut at the sheet edge). Text extents are estimated from the font size and the axes' metres per point."""

    def __init__(self, ax):
        fig = ax.figure
        pos = ax.get_position()
        x0, x1 = ax.get_xlim()
        self.k = (x1 - x0) / max(fig.get_figwidth() * pos.width * 72.0, 1.0)     # metres per point
        self.boxes = []

    def size(self, text, fs):
        lines = text.split("\n")
        return 0.62 * fs * max(len(t) for t in lines) * self.k, 1.25 * fs * len(lines) * self.k

    def box(self, x, y, text, fs, ha, va):
        w, h = self.size(text, fs)
        bx = x - w / 2 if ha == "center" else (x - w if ha == "right" else x)
        by = y - h / 2 if va == "center" else (y - h if va == "top" else y)
        return (bx, by, bx + w, by + h)

    def free(self, b, pad=0.05):
        """No reserved box within `pad` metres (the text extents are estimates: two labels the estimate keeps apart
        touched on the hall plan)."""
        return all(b[2] + pad <= o[0] or o[2] + pad <= b[0] or b[3] + pad <= o[1] or o[3] + pad <= b[1]
                   for o in self.boxes)

    def reserve(self, b):
        self.boxes.append(b)

    def place(self, ax, x, y, text, fs, offsets, ha="left", va="center", fallback_deg=None, **kw):
        """Try the offsets in order; the first free spot wins. When none is free and fallback_deg is given, the label
        is set along that direction (rotated, stepping outward until it is free: a row of instruments along a wall,
        a review's case); else the first offset."""
        best = None
        for dx, dy, h_, v_ in offsets:
            b = self.box(x + dx, y + dy, text, fs, h_ or ha, v_ or va)
            if best is None:
                best = (dx, dy, h_ or ha, v_ or va, b)
            if self.free(b):
                dx, dy, h_, v_, b = (dx, dy, h_ or ha, v_ or va, b)
                self.reserve(b)
                return ax.text(x + dx, y + dy, text, fontsize=fs, ha=h_, va=v_, **kw)
        if fallback_deg is not None:
            t = self.along(ax, x, y, text, fs, fallback_deg, **kw)
            if t is not None:
                return t
        dx, dy, h_, v_, b = best
        self.reserve(b)
        return ax.text(x + dx, y + dy, text, fontsize=fs, ha=h_, va=v_, **kw)

    def along(self, ax, x, y, text, fs, deg, start=0.22, step=0.25, tries=10, **kw):
        """A label rotated along direction `deg` (its reading direction kept upright), its near end `start` + k x step
        from (x, y): the first k whose box is free; None if none is."""
        w, h = self.size(text, fs)
        a = math.radians(deg)
        ux, uy = math.cos(a), math.sin(a)
        rot = deg if -90.0 < ((deg + 180.0) % 360.0 - 180.0) <= 90.0 else deg - 180.0
        flip = rot != deg
        for k in range(tries):
            d0 = start + k * step
            cx, cy = x + ux * (d0 + w / 2), y + uy * (d0 + w / 2)
            ex, ey = abs(ux) * w / 2 + abs(uy) * h / 2, abs(uy) * w / 2 + abs(ux) * h / 2
            b = (cx - ex, cy - ey, cx + ex, cy + ey)
            if self.free(b):
                self.reserve(b)
                ax.plot([x, x + ux * d0], [y, y + uy * d0], color=kw.get("color", "k"), lw=0.3, zorder=kw.get("zorder", 12))
                return ax.text(x + ux * d0, y + uy * d0, text, fontsize=fs, rotation=rot, rotation_mode="anchor",
                               ha="right" if flip else "left", va="center", **kw)
        return None


def _fit_name(L, name, width, fs):
    """A room name that fits its room: one line, else two lines, else a smaller font."""
    text = name
    w, _ = L.size(text, fs)
    if w > 0.9 * width and " " in name:
        words = name.split(" ")
        best = None
        for k in range(1, len(words)):
            t = " ".join(words[:k]) + "\n" + " ".join(words[k:])
            ww, _ = L.size(t, fs)
            if best is None or ww < best[0]:
                best = (ww, t)
        text = best[1]
        w = best[0]
    while w > 0.9 * width and fs > 4.4:
        fs -= 0.4
        w, _ = L.size(text, fs)
    return text, fs


def draw_prior(ax, prior, style="drawing"):
    """Draw the drawing's content into ax. style 'drawing' (filled poche) or 'overlay' (thin black lines)."""
    lw = 0.9 if style == "drawing" else 0.8
    fill = PAPER if style == "drawing" else "none"
    LB = Labels(ax)
    # the room names and number boxes first: other labels avoid them
    name_at = []
    for r in prior["rooms"]:
        if r["type"] == "corridor":
            continue
        x0, y0, x1, y1 = r["rect"]
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1) + 0.6
        text, fs = _fit_name(LB, r["name"], x1 - x0, 6.5)
        b = LB.box(cx, cy, text, fs, "center", "center")
        LB.reserve(b)
        LB.reserve((cx - 0.47, cy - 0.97, cx + 0.47, cy - 0.51))
        name_at.append((r, cx, cy, text, fs))
    # zoning first (under everything)
    if style == "drawing":
        for z in prior["zoning"]["zones"]:
            if z["zone_class"] in ZHATCH:
                h, c = ZHATCH[z["zone_class"]]
                ax.add_patch(Polygon(z["polygon"], closed=True, fill=False, hatch=h, ec=c, lw=0.0, zorder=2,
                                     alpha=0.55))
                ax.add_patch(Polygon(z["polygon"], closed=True, fill=False, ec=c, lw=1.0, ls=(0, (5, 2)), zorder=2))
    # equipment outlines
    for e in prior["equipment"]:
        if e["kind"] in ("deployment_pad",):
            continue
        if e["shape"] == "circle":
            ax.add_patch(Circle(e["center"], e["radius"], fill=False, ec=INK, lw=0.7, zorder=6))
            ax.add_patch(Circle(e["center"], e["radius"] * 0.12, fill=False, ec=INK, lw=0.5, zorder=6))
            rr_ = e["radius"]
            LB.place(ax, e["center"][0], e["center"][1], e["label"], 5.5,
                    [(0, -rr_ * 0.35, "center", "center"), (0, -rr_ - 0.2, "center", "top"),
                     (0, rr_ + 0.2, "center", "bottom"), (rr_ + 0.15, 0, "left", "center")],
                    family=FONT, color=INK, zorder=7)
        else:
            ax.add_patch(Polygon(e["outline"], closed=True, fill=False, ec=INK, lw=0.6,
                                 ls="-" if e["kind"] not in ("drum_store",) else (0, (3, 2)), zorder=6))
            xs = [p[0] for p in e["outline"]]
            ys = [p[1] for p in e["outline"]]
            if e["kind"] in ("drum_store", "pipe_rack", "pump_set", "switchgear", "mcc", "ahu", "shield_wall"):
                ex, ey = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
                room = next((r for r in prior["rooms"] if r["number"] == e.get("room")), None)
                if room is not None:
                    rx, ry = 0.5 * (room["rect"][0] + room["rect"][2]), 0.5 * (room["rect"][1] + room["rect"][3])
                    if abs(ex - rx) < 2.0 and abs(ey - (ry + 0.2)) < 1.0:
                        ey = min(ys) - 0.25 if ey > ry else max(ys) + 0.25
                LB.place(ax, ex, ey, e["label"], 5, [(0, 0, "center", "center"), (0, 0.3, "center", "bottom"),
                                                     (0, -0.3, "center", "top")], family=FONT, color=INK, zorder=7)
            if e["kind"] == "shield_wall":
                ax.add_patch(Polygon(e["outline"], closed=True, fill=False, hatch="xxxx", ec=INK, lw=0.0, zorder=6))
    # walls with poche
    for w in prior["walls"]:
        r, axis = _wall_rect(w)
        hatch = "xxxx" if w["kind"] == "shield" else ("////" if style == "drawing" else None)
        ax.add_patch(Rectangle(r[:2], r[2], r[3], fc=fill, ec=INK, lw=lw, hatch=hatch, zorder=8))
    # door openings, leaves and swings
    for d in prior["doors"]:
        x, y = d["pos"]
        wdt = d["width"]
        if d["kind"] == "fire_door":
            # double swing across the corridor (drawn in the corridor itself)
            for s in (-1, 1):
                hy = y + s * wdt / 2
                ax.add_patch(Arc((x, hy), 2 * wdt / 2, 2 * wdt / 2, theta1=180 if s > 0 else 90,
                                 theta2=270 if s > 0 else 180, ec=INK, lw=0.5, zorder=9))
                ax.plot([x, x - wdt / 2], [hy, hy], color=INK, lw=0.8, zorder=9)
            ax.text(x, y, "FD", fontsize=5, family=FONT, ha="center", va="center", zorder=9,
                    bbox=dict(fc=PAPER, ec="none", pad=0.3))
            continue
        # find the wall thickness at this door (from the nearest wall)
        t = 0.3
        best = None
        for w in prior["walls"]:
            r, axis = _wall_rect(w)
            if axis != d["axis"]:
                continue
            if axis == "x":
                if r[1] - 0.01 <= y <= r[1] + r[3] + 0.01 and abs(r[0] + r[2] / 2 - x) < 0.6:
                    best = r
            else:
                if r[0] - 0.01 <= x <= r[0] + r[2] + 0.01 and abs(r[1] + r[3] / 2 - y) < 0.6:
                    best = r
        if best is not None:
            t = best[2] if d["axis"] == "x" else best[3]
            cx = best[0] + best[2] / 2 if d["axis"] == "x" else x
            cy = y if d["axis"] == "x" else best[1] + best[3] / 2
        else:
            cx, cy = x, y
        if d["axis"] == "x":
            op = (cx - t / 2 - 0.005, y - wdt / 2, t + 0.01, wdt)
        else:
            op = (x - wdt / 2, cy - t / 2 - 0.005, wdt, t + 0.01)
        ax.add_patch(Rectangle(op[:2], op[2], op[3], fc=PAPER if style == "drawing" else "white", ec="none",
                               zorder=8.5))
        if d["kind"] == "hatch":
            ax.add_patch(Rectangle(op[:2], op[2], op[3], fc="none", ec=INK, lw=0.6, ls=(0, (2, 1.5)), zorder=9))
            ax.text(op[0] + op[2] / 2, op[1] + op[3] / 2, "H", fontsize=5, family=FONT, ha="center", va="center",
                    zorder=9)
            continue
        # jamb lines
        if d["axis"] == "x":
            ax.plot([cx - t / 2, cx + t / 2], [y - wdt / 2] * 2, color=INK, lw=0.6, zorder=9)
            ax.plot([cx - t / 2, cx + t / 2], [y + wdt / 2] * 2, color=INK, lw=0.6, zorder=9)
        else:
            ax.plot([x - wdt / 2] * 2, [cy - t / 2, cy + t / 2], color=INK, lw=0.6, zorder=9)
            ax.plot([x + wdt / 2] * 2, [cy - t / 2, cy + t / 2], color=INK, lw=0.6, zorder=9)
        # swing: hinge at one jamb on the swing-into side
        leaves = [(1, wdt)] if d["kind"] != "double_door" else [(1, wdt / 2), (-1, wdt / 2)]
        hinge = d.get("hinge", 1) or 1
        into_sign = _into_sign(prior, d, cx, cy)
        for k, (hs, L) in enumerate(leaves):
            hsg = hinge * hs
            if d["axis"] == "y":
                hx, hy = x + hsg * wdt / 2, cy + into_sign * t / 2
                tip = (hx, hy + into_sign * L)
                a0 = 90 if into_sign > 0 else 270
                ang_closed = 180 if hsg > 0 else 0
            else:
                hx, hy = cx + into_sign * t / 2, y + hsg * wdt / 2
                tip = (hx + into_sign * L, hy)
                a0 = 0 if into_sign > 0 else 180
                ang_closed = 270 if hsg > 0 else 90
            ax.plot([hx, tip[0]], [hy, tip[1]], color=INK, lw=0.9, zorder=9)
            t1, t2 = sorted([a0, ang_closed])
            if t2 - t1 > 180:
                t1, t2 = t2, t1 + 360
            ax.add_patch(Arc((hx, hy), 2 * L, 2 * L, theta1=t1, theta2=t2, ec=INK, lw=0.45, zorder=9))
        if d.get("note") and "STAIR" in d["note"]:
            ax.text(x - 0.9, y, "STAIR 2", fontsize=5.5, family=FONT, rotation=90, ha="center", va="center", zorder=9)
    # room names and numbers
    for r in prior["rooms"]:
        if r["type"] == "corridor":
            x0, y0, x1, y1 = r["rect"]
            if (x1 - x0) > (y1 - y0):
                LB.place(ax, x0 + 0.62 * (x1 - x0), 0.5 * (y0 + y1), "CORRIDOR %s" % r["number"], 6.5,
                        [(0, 0, "center", "center"), (-2.5, 0, "center", "center"), (2.5, 0, "center", "center")],
                        family=FONT, color=INK, zorder=10)
    for r, cx, cy, text, fs in name_at:
        ax.text(cx, cy, text, fontsize=fs, family=FONT, ha="center", va="center", color=INK, zorder=10,
                fontweight="bold")
        ax.add_patch(FancyBboxPatch((cx - 0.45, cy - 0.95), 0.9, 0.42, boxstyle="round,pad=0.02", fc=PAPER, ec=INK,
                                    lw=0.6, zorder=10))
        ax.text(cx, cy - 0.74, r["number"], fontsize=6, family=FONT, ha="center", va="center", zorder=11)
    # documented tag positions (the symbols reserve their place, then the labels go where they are free). The walls'
    # poche and the grid marker squares are reserved first: on the hall plan the wall row's labels sat on the hatched
    # wall and on the marker squares, unreadable
    for w in prior["walls"]:
        r, _ = _wall_rect(w)
        LB.reserve((r[0], r[1], r[0] + r[2], r[1] + r[3]))
    for m in prior["grid_markers"]:
        LB.reserve((m["pos"][0] - 0.14, m["pos"][1] - 0.14, m["pos"][0] + 0.14, m["pos"][1] + 0.14))
    pts = []
    for it in prior["work_order"]["items"]:
        loc = it["location"]
        if not loc.get("documented") or not loc.get("pos"):
            continue
        x, y = loc["pos"]
        ax.plot(x, y, marker=TYPE_SYM.get(it["type"], "o"), ms=5, mfc="none" if style == "drawing" else "#f1c40f",
                mec=INK, mew=0.9, zorder=12)
        LB.reserve((x - 0.13, y - 0.13, x + 0.13, y + 0.13))
        pts.append((x, y, it["tag"], float(loc.get("facing_deg") or 0.0)))
    for x, y, tag, fac in pts:
        LB.place(ax, x, y, tag, 5.5, [(0.22, -0.42, "left", "center"), (0.22, 0.4, "left", "center"),
                                     (-0.22, -0.42, "right", "center"), (-0.22, 0.4, "right", "center"),
                                     (0, -0.7, "center", "center"), (0, 0.7, "center", "center"),
                                     (0.25, 0, "left", "center"), (-0.25, 0, "right", "center")],
                fallback_deg=fac, family=FONT, color=INK, zorder=12, gid="tag_label")
    # grid markers (documented)
    for m in prior["grid_markers"]:
        x, y = m["pos"]
        ax.add_patch(Rectangle((x - 0.12, y - 0.12), 0.24, 0.24, fc=INK, ec=INK, lw=0.5, zorder=12))
        LB.place(ax, x, y, m["id"], 4.5, [(0.18, 0.18, "left", "bottom"), (0.18, -0.18, "left", "top"),
                                         (-0.18, 0.18, "right", "bottom"), (-0.18, -0.18, "right", "top")],
                family=FONT, zorder=12)
    # link infrastructure as drawn
    bs = prior["link"]["base_station"]["pos"]
    ax.plot(bs[0], bs[1], marker="v", ms=6, mfc="none", mec=INK, zorder=12)
    ax.text(bs[0] + 0.2, bs[1] + 0.25, "RADIO BASE", fontsize=5, family=FONT, zorder=12)
    for rp in prior["link"]["repeaters"]:
        ax.plot(rp["pos"][0], rp["pos"][1], marker="v", ms=6, mfc="none", mec=INK, zorder=12)
        ax.text(rp["pos"][0] + 0.2, rp["pos"][1] + 0.25, "RPT %s" % rp["id"][-1], fontsize=5, family=FONT, zorder=12)
    # the datum
    ax.add_patch(Circle((0, 0), 0.35, fill=False, ec=INK, lw=0.8, zorder=12))
    ax.plot([-0.55, 0.55], [0, 0], color=INK, lw=0.6, zorder=12)
    ax.plot([0, 0], [-0.55, 0.55], color=INK, lw=0.6, zorder=12)
    ax.text(0.2, -0.75, "DATUM 0,0", fontsize=5, family=FONT, zorder=12)


def _into_sign(prior, d, cx, cy):
    """+1 if the leaf swings toward +axis (from the wall line), from the room numbers."""
    tgt = d.get("swing_into")
    for r in prior["rooms"]:
        if r["number"] == tgt:
            x0, y0, x1, y1 = r["rect"]
            if d["axis"] == "y":
                return 1 if 0.5 * (y0 + y1) > cy else -1
            return 1 if 0.5 * (x0 + x1) > cx else -1
    return 1


def prior_drawing(prior, path, bbox=None):
    out = prior["building"]["outline"]
    al = prior["building"]["airlock"]
    bb = bbox or [min(out[0], al[0]) - 0.5, out[1] - 0.5, out[2] + 0.5, out[3] + 0.5]
    fig = plt.figure(figsize=(16.5, 11.0), dpi=150, facecolor=PAPER)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
    ax.set_facecolor(PAPER)
    ax.set_xlim(bb[0] - 2.5, bb[2] + 9.5)
    ax.set_ylim(bb[1] - 4.2, bb[3] + 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    # border frame
    X0, X1 = bb[0] - 2.3, bb[2] + 9.3
    Y0, Y1 = bb[1] - 4.0, bb[3] + 2.0
    ax.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False, ec=INK, lw=1.6, zorder=1))
    ax.add_patch(Rectangle((X0 + 0.25, Y0 + 0.25), X1 - X0 - 0.5, Y1 - Y0 - 0.5, fill=False, ec=INK, lw=0.6, zorder=1))
    # grid
    for ln in prior["grid"]["x_lines"]:
        ax.plot([ln["x"]] * 2, [bb[1] - 1.0, bb[3] + 0.9], color=INK, lw=0.35, ls=(0, (8, 3, 1, 3)), zorder=1)
        for yy in (bb[3] + 1.35, bb[1] - 1.45):
            ax.add_patch(Circle((ln["x"], yy), 0.42, fc=PAPER, ec=INK, lw=0.7, zorder=3))
            ax.text(ln["x"], yy, ln["label"], ha="center", va="center", fontsize=7, family=FONT, zorder=4)
    for ln in prior["grid"]["y_lines"]:
        ax.plot([bb[0] - 1.0, bb[2] + 0.9], [ln["y"]] * 2, color=INK, lw=0.35, ls=(0, (8, 3, 1, 3)), zorder=1)
        for xx in (bb[0] - 1.45, bb[2] + 1.35):
            ax.add_patch(Circle((xx, ln["y"]), 0.42, fc=PAPER, ec=INK, lw=0.7, zorder=3))
            ax.text(xx, ln["y"], ln["label"], ha="center", va="center", fontsize=7, family=FONT, zorder=4)
    draw_prior(ax, prior, "drawing")
    # missing sheet
    for s in prior["drawing"]["sheets"]:
        r = s["rect"]
        ax.plot([r[0], r[2], r[2], r[0], r[0]], [r[1], r[1], r[3], r[3], r[1]], color=INK, lw=0.3, ls=(0, (1, 3)),
                zorder=2)
        ax.text(r[0] + 0.2, r[1] + 0.2, "SHEET %d" % s["sheet"], fontsize=5.5, family=FONT, zorder=2, color="#555")
        if s["missing"]:
            ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], fc=PAPER, ec="none", zorder=13))
            cx, cy = 0.5 * (r[0] + r[2]), 0.5 * (r[1] + r[3])
            ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], fill=False, ec="#8b1a1a", lw=0.8,
                                   ls=(0, (6, 4)), zorder=14))
            ax.text(cx, cy, "SHEET %d OF %d\nMISSING" % (s["sheet"], len(prior["drawing"]["sheets"])), fontsize=17,
                    family=FONT, color="#8b1a1a", ha="center", va="center", rotation=12, zorder=15, alpha=0.85,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", fc="none", ec="#8b1a1a", lw=2.2))
    # north arrow and scale bar
    nx, ny = bb[2] + 5.5, bb[3] - 2.5
    ax.add_patch(Polygon([[nx, ny + 1.4], [nx - 0.45, ny], [nx, ny + 0.35]], closed=True, fc=INK, ec=INK, zorder=5))
    ax.add_patch(Polygon([[nx, ny + 1.4], [nx + 0.45, ny], [nx, ny + 0.35]], closed=True, fc=PAPER, ec=INK, zorder=5))
    ax.text(nx, ny + 1.75, "N", fontsize=12, family=FONT, ha="center", fontweight="bold")
    sx, sy = bb[2] + 2.5, bb[3] - 5.0
    for k in range(5):
        ax.add_patch(Rectangle((sx + k, sy), 1.0, 0.2, fc=INK if k % 2 == 0 else PAPER, ec=INK, lw=0.6, zorder=5))
    for k in (0, 1, 2, 5):
        ax.text(sx + k, sy - 0.35, str(k), fontsize=6, family=FONT, ha="center", va="top")
    ax.text(sx + 2.5, sy + 0.45, "METRES", fontsize=6, family=FONT, ha="center")
    # zoning legend
    lx, ly = bb[2] + 2.3, bb[3] - 8.0
    ax.text(lx, ly + 0.8, "RADIOLOGICAL ZONING", fontsize=6.5, family=FONT, fontweight="bold")
    ax.text(lx, ly + 0.35, "SURVEY %s" % prior["zoning"]["survey_date"], fontsize=6, family=FONT)
    for k, (c, (h, col)) in enumerate(ZHATCH.items()):
        ax.add_patch(Rectangle((lx, ly - 0.6 - 0.75 * k), 0.9, 0.5, fill=False, hatch=h, ec=col, lw=0.8))
        ax.text(lx + 1.1, ly - 0.35 - 0.75 * k, c.replace("_", " ").upper(), fontsize=6, family=FONT, va="center")
    # legend of symbols
    ly2 = ly - 2.6
    for k, (t, sym) in enumerate(TYPE_SYM.items()):
        ax.plot(lx + 0.3, ly2 - 0.5 * k, marker=sym, ms=5, mfc="none", mec=INK, mew=0.9)
        ax.text(lx + 0.8, ly2 - 0.5 * k, t.replace("_", " ").upper(), fontsize=5.5, family=FONT, va="center")
    ax.add_patch(Rectangle((lx + 0.18, ly2 - 0.5 * 8 - 0.12), 0.24, 0.24, fc=INK))
    ax.text(lx + 0.8, ly2 - 0.5 * 8, "GRID MARKER", fontsize=5.5, family=FONT, va="center")
    # title block
    D = prior["drawing"]
    tx0, ty0 = bb[2] + 1.6, bb[1] - 3.75
    tw, th = 7.4, 6.6
    ax.add_patch(Rectangle((tx0, ty0), tw, th, fc=PAPER, ec=INK, lw=1.2, zorder=20))
    rows = [("TITLE", D["title"]), ("WORK ORDER", prior["work_order"]["number"]), ("DRAWING No.", D["number"]),
            ("REV", "%s   %s" % (D["revision"], D["revision_date"])), ("SCALE", D["scale"]),
            ("SHEETS", "1-%d OF %d" % (len(D["sheets"]), len(D["sheets"]))),
            ("DRAWN / CHKD", "J.R.H. / P.M.")]
    y = ty0 + th
    for k, (a, b) in enumerate(rows):
        hh = 1.3 if k == 0 else 0.7
        y -= hh
        ax.plot([tx0, tx0 + tw], [y, y], color=INK, lw=0.6, zorder=21)
        ax.text(tx0 + 0.12, y + hh - 0.2, a, fontsize=5, family=FONT, va="top", zorder=21, color="#444")
        if k == 0:
            words = b.split(" - ")
            ax.text(tx0 + 0.12, y + 0.55, "\n".join(words), fontsize=6.3, family=FONT, va="center", zorder=21,
                    fontweight="bold")
        else:
            ax.text(tx0 + 2.2, y + hh / 2 - 0.05, b, fontsize=7, family=FONT, va="center", zorder=21,
                    fontweight="bold" if k == 2 else "normal")
    for k, n in enumerate(D["notes"]):
        ax.text(tx0 + 0.12, ty0 + 0.18 + 0.3 * (len(D["notes"]) - 1 - k), "NOTE %d: %s" % (k + 1, n), fontsize=4.6,
                family=FONT, zorder=21)
    fig.savefig(path, dpi=150, facecolor=PAPER)
    out = {"tag_label_overlaps": tag_label_overlaps(fig)}
    plt.close(fig)
    return out


def tag_label_overlaps(fig) -> int:
    """Pairs of instrument tag labels whose rendered text boxes overlap (the hall plan's wall row)."""
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer=r) for ax in fig.axes for t in ax.texts if t.get_gid() == "tag_label"]
    n = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1:
                n += 1
    return n
