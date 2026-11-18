"""Two maps of a ladder site:
  slots.png        the ten items by slot, the element each one is behind, and the acceptance region of each
                   instrument (margin region filled, generous region outlined)
  round_sheet.png  the round sheet (prior.json) rendered as the form a team would hold
"""

from __future__ import annotations

import json
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Polygon, Rectangle  # noqa: E402

from .util import mask_from_runs  # noqa: E402

SLOT_COL = {"A": "#1b9e77", "B-step": "#d95f02", "B-kerb": "#e6ab02", "B-low": "#7570b3", "C-sealed": "#e7298a",
            "C-moved": "#66a61e", "C-relocated": "#a6761d", "C-annex": "#1f78b4", "M": "#555555", "X": "#c2185b"}
PAPER = "#fbfaf4"
INK = "#151515"
FONT = "DejaVu Sans Mono"


def slots_map(site_dir, path=None):
    from .maps import SiteData, draw_walls, new_ax, room_labels
    sd = SiteData(site_dir)
    site = sd.site
    path = path or os.path.join(site_dir, "maps", "slots.png")
    fig, ax = new_ax(sd, "Ten-slot ladder (seed %d): items by slot, the element each one is behind, acceptance "
                         "regions" % site["seed"])
    polys = sd.geom_polys((0, 3))
    from matplotlib.collections import PolyCollection
    pc = PolyCollection([p[0] for p in polys], facecolors=[tuple(0.62 + 0.38 * np.array(p[1][:3])) for p in polys],
                        edgecolors="none", zorder=3)
    ax.add_collection(pc)
    draw_walls(ax, site, color="0.2")
    L = site["ladder"]
    els = {e["id"]: e for e in site["elements"]}
    doors = {d["id"]: d for d in site["doors"]}
    # the elements the slots are behind
    behind = {"A": [], "B-step": L.get("b_step_elements") or [], "B-kerb": L.get("b_kerb_elements") or [],
              "B-low": L.get("b_low_elements") or []}
    sill = els.get(L["corridor_sill"])
    sill_label = None
    if sill:
        ax.add_patch(Polygon(sill["footprint"], closed=True, fc="#ff7f00", ec="k", lw=1.2, zorder=14))
        cx = np.mean([p[0] for p in sill["footprint"]])
        cy = np.mean([p[1] for p in sill["footprint"]])
        sill_label = (cx, cy)
    for slot, ids in behind.items():
        for i in ids:
            e = els.get(i)
            if e:
                ax.add_patch(Polygon(e["footprint"], closed=True, fc=SLOT_COL[slot], ec="k", lw=0.8, alpha=0.85,
                                     zorder=14))
    for did, slot in ((L.get("as_built_opening"), "C-sealed"), (L.get("sealed_door"), "C-sealed"),
                      (L.get("annex_door"), "C-annex")):
        d = doors.get(did)
        if not d:
            continue
        x, y = d["pos"]
        mk = "X" if did == L.get("sealed_door") else "D"
        ax.plot(x, y, marker=mk, ms=12 if mk == "X" else 9, mfc=SLOT_COL[slot] if mk == "D" else "#c0392b",
                mec="k", zorder=26)
    # acceptance regions
    gi = None
    for ip in site["ips"]:
        slot = ip.get("slot")
        if slot in (None, "M", "X") or "acceptance_region" not in ip:
            continue
        ar = ip["acceptance_region"]
        rr = ip["readable_region"]
        gi = ar["grid"]
        shape = tuple(int(v) for v in gi["shape"])
        x0, y0 = gi["origin"]
        res = gi["res"]
        ext = [x0, x0 + shape[1] * res, y0, y0 + shape[0] * res]
        gen = mask_from_runs(ar["runs_generous"], shape)
        mar = mask_from_runs(rr["runs_stand"], shape) | mask_from_runs(rr["runs_crouch"], shape)
        col = matplotlib.colors.to_rgb(SLOT_COL[slot])
        img = np.zeros(shape + (4,))
        img[mar] = list(col) + [0.45]
        ax.imshow(img, origin="lower", extent=ext, interpolation="nearest", zorder=8)
        if gen.any():
            ax.contour(np.linspace(ext[0], ext[1], shape[1]), np.linspace(ext[2], ext[3], shape[0]),
                       gen.astype(float), levels=[0.5], colors=[col], linewidths=0.8, zorder=9)
    # the items; labels by the drawing's non-overlapping placer (room names reserved first)
    from .drawing import Labels
    LB = Labels(ax)
    for r in site["rooms"]:
        if r["type"] != "corridor":
            x0, y0, x1, y1 = r["rect"]
            LB.reserve(LB.box(0.5 * (x0 + x1), y1 - 0.45, "%s\n%s" % (r["number"], r["name"]), 6, "center", "top"))
    offs = [(0.35, 0.35, "left", "bottom"), (0.35, -0.35, "left", "top"), (-0.35, 0.35, "right", "bottom"),
            (-0.35, -0.35, "right", "top"), (0.0, 0.7, "center", "bottom"), (0.0, -0.7, "center", "top"),
            (0.9, 0.0, "left", "center"), (-0.9, 0.0, "right", "center"), (1.4, 1.0, "left", "bottom"),
            (-1.4, -1.0, "right", "top"), (1.4, -1.0, "left", "top"), (-1.4, 1.0, "right", "bottom")]
    for ip in site["ips"]:
        slot = ip.get("slot")
        if slot is None:
            continue
        x, y = ip["pose"]["pos"][:2]
        col = SLOT_COL[slot]
        if slot == "X":
            ax.plot(x, y, marker=(3, 2, 0), ms=15, color=col, mew=2.5, zorder=28)
        elif slot == "M":
            ax.add_patch(Circle((x, y), 0.3, fc="none", ec=col, lw=2.0, zorder=28))
        else:
            ax.plot(x, y, marker="o", ms=11, mfc=col, mec="k", mew=1.2, zorder=28)
        LB.reserve((x - 0.3, y - 0.3, x + 0.3, y + 0.3))
        op = ip.get("observation_pose")
        if op:
            ax.plot(op["pos"][0], op["pos"][1], ".", ms=7, color="k", zorder=27)
            ax.plot([op["pos"][0], x], [op["pos"][1], y], ":", color=col, lw=1.1, zorder=27)
        doc = ip.get("documented")
        if slot in ("C-moved", "C-relocated") and isinstance(doc, dict) and doc.get("pos"):
            ax.annotate("", xy=(x, y), xytext=tuple(doc["pos"][:2]), zorder=29,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, ls="--"))
            ax.plot(doc["pos"][0], doc["pos"][1], marker="s", ms=8, mfc="white", mec=col, mew=1.6, zorder=29)
    for ip in site["ips"]:
        slot = ip.get("slot")
        if slot is None:
            continue
        x, y = ip["pose"]["pos"][:2]
        if slot == "X":
            lab = "X  unlisted source\n%.0f Gy/h at 1 m" % float(L.get("dose", {}).get("x_strength_gyh_at_1m", 0.0))
        elif slot == "M":
            lab = "M  %s\nsurvey mark" % ip["tag"]
        else:
            lab = "%s  %s\n%s" % (slot, ip["tag"], (ip.get("ladder_type") or "").replace("_", " "))
        LB.place(ax, x, y, lab, 7.2, offs, fontweight="bold", color="k", zorder=31,
                 bbox=dict(fc="white", ec=SLOT_COL[slot], lw=1.2, alpha=0.92, pad=0.7))
    if sill_label:
        LB.place(ax, sill_label[0], sill_label[1], "corridor sill\n(site-wide)", 7,
                 [(0.0, -1.0, "center", "top"), (0.0, 1.0, "center", "bottom"), (0.6, -1.0, "left", "top")],
                 zorder=30, fontweight="bold", bbox=dict(fc="white", ec="#ff7f00", lw=0.8, pad=0.6))
    for d in site["decoys"]:
        ax.plot(d["pose"]["pos"][0], d["pose"]["pos"][1], marker="o", ms=4, mfc="0.75", mec="0.3", zorder=20)
    room_labels(ax, site, fs=6)
    h = [Line2D([], [], marker="o", ls="", mfc=SLOT_COL[s], mec="k", ms=9, label="slot %s" % s)
         for s in ("A", "B-step", "B-kerb", "B-low", "C-sealed", "C-moved", "C-relocated", "C-annex")]
    h += [Line2D([], [], marker="o", ls="", mfc="none", mec=SLOT_COL["M"], ms=10, label="M survey mark"),
          Line2D([], [], marker=(3, 2, 0), ls="", color=SLOT_COL["X"], ms=10, label="X unlisted source"),
          Patch(fc="#ff7f00", ec="k", label="corridor sill (site-wide)"),
          Patch(fc="0.5", alpha=0.45, label="margin acceptance region (fill)"),
          Line2D([], [], color="0.3", lw=0.8, label="generous region (outline)"),
          Line2D([], [], marker="X", ls="", mfc="#c0392b", mec="k", ms=9, label="sealed door"),
          Line2D([], [], marker="D", ls="", mfc="0.7", mec="k", ms=8,
                 label="undrawn way in (as-built opening, annex door)"),
          Line2D([], [], marker="s", ls="", mfc="white", mec="0.3", ms=8, label="drawn place (moved / relocated)"),
          Line2D([], [], marker=".", ls="", color="k", label="observation pose")]
    ttl = "assignment: " + ", ".join("%s %s" % (k, v) for k, v in L["assignment"].items())
    ax.text(0.99, 0.01, "\n".join(textwrap.wrap(ttl, 150)), transform=ax.transAxes, fontsize=7.5, va="bottom",
            ha="right", zorder=40, bbox=dict(fc="white", ec="0.5", alpha=0.9))
    # the legend outside the plan (inside it would hide the east rooms)
    from .maps import DPI, grid_lines, north_arrow, scale_bar
    bb = site["building"]["bbox"]
    grid_lines(ax, site["building"]["grid"], bb)
    scale_bar(ax, bb)
    north_arrow(ax, bb)
    lg = ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8, framealpha=0.93)
    lg.set_zorder(50)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def round_sheet(prior, path):
    """The round sheet as a printed form (A3 landscape proportions)."""
    rs = prior["round_sheet"]
    items = rs["items"]
    fig = plt.figure(figsize=(16.5, 11.7), dpi=130, facecolor=PAPER)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 100, 70, fill=False, ec=INK, lw=2.0))
    ax.add_patch(Rectangle((0.6, 0.6), 98.8, 68.8, fill=False, ec=INK, lw=0.6))
    ax.text(2.0, 66.6, rs["title"], fontsize=15, family=FONT, fontweight="bold", va="center")
    ax.text(2.0, 64.0, "ROUND SHEET %s    DRAWING %s REV %s    LAST ROUND %s" % (
        rs["number"], prior["drawing"]["number"], prior["drawing"]["revision"], rs["last_round"]), fontsize=9.5,
        family=FONT, va="center")
    ax.text(2.0, 62.0, rs["note"], fontsize=8.5, family=FONT, va="center", style="italic")
    ax.text(97.5, 66.6, "PAGE 1 / 1", fontsize=9, family=FONT, ha="right", va="center")
    # version 2: no label code column when the rows carry none (the tag is the only identifier)
    has_code = any("label_code" in it for it in items)
    cols = [("No", 3.0), ("TAG", 8.0)] + ([("LABEL CODE", 8.5)] if has_code else []) + [
        ("TYPE", 15.5 + (0.0 if has_code else 4.0)), ("RANGE / UNIT", 11.5), ("CASE / WINDOW / SURFACE", 10.5),
        ("LOCATION (DRAWING)", 21.0 + (0.0 if has_code else 4.5)), ("LAST 2009", 9.0), ("READING", 7.5),
        ("FRAME", 5.5)]
    tot = sum(w for _, w in cols)
    cols = [(n, w * 98.0 / tot) for n, w in cols]            # the table spans x 1..99, inside the frame
    # monospace width of one character at the cell font, in axes units (the axes is 100 units wide)
    fs_cell = 7.0
    ch = 0.60 * fs_cell / (0.94 * 16.5 * 72.0) * 100.0
    log = prior.get("modification_log", {}).get("entries", [])
    rows_log = (len(log) + 2) // 3
    n_rows = sum(1 for it in items if it.get("kind") != "unlisted_source")
    top = 59.5
    # bottom block: radiation line, modification record, signature and note (from the bottom up)
    bottom = 2.0 + 3.0 + 2.6 + 1.8 * rows_log + 2.2 + 3.6 + 1.0
    rh = min(4.2, (top - 2.4 - bottom) / max(n_rows, 1))
    xs = []
    x = 1.0
    for name, w in cols:
        xs.append((x, w))
        ax.add_patch(Rectangle((x, top - 2.4), w, 2.4, fc="#e6e3d6", ec=INK, lw=0.7))
        ax.text(x + 0.4, top - 1.2, "\n".join(textwrap.wrap(name, max(4, int((w - 0.6) / (ch * 1.03))))),
                fontsize=7.2 if len(name) * ch * 1.03 < w - 0.6 else 6.4, family=FONT, va="center",
                fontweight="bold")
        x += w
    y = top - 2.4
    k = 0
    for it in items:
        if it.get("kind") == "unlisted_source":
            continue
        k += 1
        y -= rh
        cells = [str(k), it["tag"]] + ([it.get("label_code", "-")] if has_code else []) + [
            it.get("type_text", it["type"])]
        sc = it.get("scale") or {}
        if sc.get("range"):
            cells.append("%s - %s %s" % (sc["range"][0], sc["range"][1], sc.get("unit", "")))
        elif it.get("kind") == "survey_mark":
            cells.append("Gy/h, dwell >= %gs" % it.get("evidence", {}).get("for_s", 10))
        else:
            cells.append(sc.get("unit", ""))
        if "case_mm" in it:
            cells.append("case %d mm" % it["case_mm"])
        elif "window_m" in it:
            cells.append("window %.2f m" % it["window_m"])
        elif "surface" in it:
            cells.append(it["surface"])
        elif "n_lamps" in it:
            cells.append("%d lamps: %s" % (it["n_lamps"], " ".join(c[0].upper() for c in it.get("colors", []))))
        else:
            cells.append("-")
        loc = it["location"]
        if loc.get("recorded") and it.get("kind") == "survey_mark":
            cells.append("Rm %s %s\nx %.1f y %.1f (floor)" % (loc["room"], loc.get("room_name", "")[:16],
                                                              loc["pos"][0], loc["pos"][1]))
        elif loc.get("recorded"):
            cells.append("Rm %s %s\nx %.1f y %.1f h %s f %s" % (loc["room"], loc.get("room_name", "")[:16],
                                                                loc["pos"][0], loc["pos"][1],
                                                                loc.get("height") if loc.get("height") is not None
                                                                else "-",
                                                                int(loc["facing_deg"]) if loc.get("facing_deg")
                                                                is not None else "-"))
        else:
            cells.append("NOT RECORDED")
        lr = it.get("last_2009") or {}
        rv = lr.get("reading")
        if isinstance(rv, list):
            rv = " ".join(str(v) for v in rv)
        cells.append("%s %s" % (rv if rv is not None else "-", lr.get("unit", "")))
        cells += ["", ""]
        for (cx, w), txt in zip(xs, cells):
            ax.add_patch(Rectangle((cx, y), w, rh, fc="none", ec=INK, lw=0.5))
            n_ch = max(4, int((w - 0.6) / ch))
            lines = []
            for part in txt.split("\n"):
                lines += textwrap.wrap(part, n_ch) or [""]
            ax.text(cx + 0.4, y + rh / 2, "\n".join(lines), fontsize=fs_cell if len(lines) <= 2 else 6.2,
                    family=FONT, va="center", linespacing=1.15,
                    color="#8b1a1a" if txt == "NOT RECORDED" else INK)
    # the bottom block, laid out from the bottom up so that nothing overlaps
    yb = 2.4
    ax.text(1.6, yb, "Report every reading with the frame that shows it; unreadable items: report the frame "
                     "without a value.", fontsize=7.6, family=FONT, style="italic", va="center")
    yb += 3.0
    ax.text(1.6, yb, "SIGNATURE ____________________    DATE ____________    ROBOT ID ____________", fontsize=8.5,
            family=FONT, va="center")
    yb += 2.6
    for j, e in enumerate(log):
        ax.text(3.0 + 32.0 * (j % 3), yb + 1.8 * (rows_log - 1 - j // 3),
                "%d  %-28s rm %s" % (e["year"], e["kind"][:28], e["room"]), fontsize=7.4, family=FONT, va="center")
    yb += 1.8 * rows_log + 0.6
    ax.text(1.6, yb, "MODIFICATION RECORD (kind and room only):", fontsize=8.2, family=FONT, fontweight="bold",
            va="center")
    yb += 2.4
    for it in items:
        if it.get("kind") == "unlisted_source":
            ax.add_patch(Rectangle((1.0, yb - 1.5), 98.0, 3.0, fc="#f4ecd2", ec=INK, lw=0.6))
            ax.text(1.6, yb, "RADIATION: %s   (weight %.2f)" % (it["text"], it["weight"]), fontsize=8.6,
                    family=FONT, va="center", fontweight="bold")
    fig.savefig(path, dpi=130, facecolor=PAPER)
    plt.close(fig)
    return path


def make_ladder_maps(site_dir):
    md = os.path.join(site_dir, "maps")
    os.makedirs(md, exist_ok=True)
    slots_map(site_dir, os.path.join(md, "slots.png"))
    prior = json.load(open(os.path.join(site_dir, "prior.json")))
    round_sheet(prior, os.path.join(md, "round_sheet.png"))
    return md
