"""Exports of a venue: the MJCF venue, the files, the greybox top view.

    spec = build_spec(world, variant="track", pushable=True)   # floor, zone slabs / decals, tape, timing line,
                                                                # clutter (the contract's greybox writer without its ducts)
                                                                # + the ducts through the duct model's add_ducts
    model_xml(world, variant)                                    # the whole model as the cell composes it
                                                                # (the cell's providers: venue + car (+ boxes and slow cars for avoid))
    top_view(world, "x.png")                                     # greybox top view (PIL; deterministic bytes)
    export_all(world, out_dir)                                   # <id>.json, <id>.venue_sheet.json, <id>.truth.json,
                                                                # <id>.digest, <id>.xml (track), <id>.png
"""

import copy
import json
import math
import os

from sketch.examples import check
from sketch.examples import constants as K
from sketch.examples import greybox

R = K.DUCT_RADIUS_M


def build_spec(world, variant="track", pushable=True, mode=None, params=None, debug_markers=False):
    """The venue of `world` in a fresh MjSpec without cars and boxes. The floor, the zone
    slabs and decals, the tape lines, the timing line and the clutter are the contract's greybox writer applied to a copy
    of the world without ducts; the ducts are the duct model's `add_ducts` (static when pushable is False, else the duct model's default
    mode, `mode` overrides it). `variant` is accepted for the contract's signature: the boxes and slow cars of the
    avoid variant are the obstacle model's (`add_boxes`, `add_slow_cars`), added by the caller."""
    import mujoco
    from sketch.ducts import add_ducts
    if variant not in ("track", "avoid"):
        raise ValueError("variant must be 'track' or 'avoid'")
    bare = copy.deepcopy(world)
    bare["truth"]["ducts"] = []
    spec = mujoco.MjSpec.from_string(greybox.build_xml(bare, variant="track", cars=None, debug_markers=debug_markers))
    add_ducts(spec, world, pushable=pushable, params=params, mode=mode)
    return spec


def model_xml(world, variant="track", pushable=True):
    """MJCF text of the model the cell builds for a session variant: the cell's composition with this
    package's build_spec, the drive model's car at the start pose and, for `avoid`, the obstacle model's boxes and slow cars."""
    from sketch.referee import resolve
    from sketch.referee.providers import build_spec as compose
    prov = resolve("auto", venue_spec=build_spec)
    prov.pushable = bool(pushable)
    spec = compose(world, variant, prov, world["truth"]["start_pose"]["pose"])
    spec.compile()
    return spec.to_xml()


def compiled(world, variant="track", pushable=True, mode=None, with_car=True):
    """(model, data) of the venue (and the ego car at the start pose, and in `avoid` the boxes and slow cars)."""
    import mujoco
    spec = build_spec(world, variant=variant, pushable=pushable, mode=mode)
    if with_car:
        from sketch.car import add_car
        add_car(spec, prefix=K.EGO_PREFIX, pose=tuple(world["truth"]["start_pose"]["pose"]), tail_box=False)
        if variant == "avoid":
            from sketch.others import add_boxes, add_slow_cars
            add_boxes(spec, world)
            add_slow_cars(spec, world)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def model_errors(world, variant="track", mode=None, with_car=True):
    """check.check_model on the compiled venue: pushable=True for every pushable duct mode (sleep, awake and, since
    contract 1.1, active_window), False for static ducts."""
    from sketch.ducts import load_params, resolve_mode
    model, _ = compiled(world, variant=variant, pushable=(mode != "static"),
                        mode=None if mode in (None, "static") else mode, with_car=with_car)
    md = "static" if mode == "static" else resolve_mode(load_params(), True, mode)
    pushable = md != "static"
    return check.check_model(model, world, variant=variant, pushable=pushable), md


# ------------------------------------------------------------------------------------------------ top view
COLOURS = {"orange_ribbed": (232, 112, 30), "yellow_ribbed": (236, 200, 40), "black_plain": (25, 25, 25),
           "painted_concrete": (96, 98, 104), "sealed_concrete": (128, 128, 124), "polished_concrete": (170, 172, 176),
           "exhibition_carpet": (128, 104, 112), "track": None, "dust": (176, 158, 120), "metal_cover": (150, 158, 172),
           "carpet_tile": (52, 82, 170), "clutter": (110, 112, 122), "clutter_dark": (45, 45, 52), "box": (245, 245, 238),
           "tape": (250, 215, 20), "checkpoint": (40, 170, 70), "car": (30, 60, 200), "slow": (200, 40, 160),
           "centre": (235, 235, 235)}


def top_view(world, path, px_per_m=40, margin_m=1.2, title=True):
    """A greybox top view of the venue as a PNG (PIL; the same world gives the same bytes)."""
    from PIL import Image, ImageDraw
    t = world["truth"]
    b = t["bounds"]
    x0, y1 = b["xmin"] - margin_m, b["ymax"] + margin_m
    W = int(round((b["xmax"] - b["xmin"] + 2 * margin_m) * px_per_m))
    H = int(round((b["ymax"] - b["ymin"] + 2 * margin_m) * px_per_m)) + (28 if title else 0)
    top = 28 if title else 0

    def P(x, y):
        return ((x - x0) * px_per_m, top + (y1 - y) * px_per_m)

    floor_rgb = COLOURS[t["floor"]["kind"]]
    img = Image.new("RGB", (W, H), floor_rgb)
    dr = ImageDraw.Draw(img, "RGBA")
    tr = check.Track(world)
    # corridor
    left = [tr.boundary(s, "left") for s in tr.s]
    right = [tr.boundary(s, "right") for s in tr.s]
    track_rgb = tuple(min(255, int(c * 1.12 + 12)) for c in floor_rgb)
    for i in range(tr.n):
        j = (i + 1) % tr.n
        dr.polygon([P(*left[i]), P(*left[j]), P(*right[j]), P(*right[i])], fill=track_rgb)
    # patches
    for p in t["floor"]["patches"]:
        c = COLOURS[p["kind"]]
        dr.polygon([P(x, y) for x, y in p["polygon"]], fill=c + (170,), outline=(0, 0, 0, 120))
    # centreline (dashed)
    for i in range(0, tr.n, 4):
        a, bb = tr.p[i], tr.p[(i + 2) % tr.n]
        dr.line([P(*a), P(*bb)], fill=COLOURS["centre"] + (160,), width=1)
    # clutter
    for c in t["clutter"]:
        col = COLOURS["clutter_dark"] if c["reflectivity"] == "dark" else COLOURS["clutter"]
        x, y, yaw = c["pose"]
        if c["shape"] == "cylinder":
            r = 0.5 * c["size_m"][0] * px_per_m
            cx, cy = P(x, y)
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        else:
            dr.polygon([P(*q) for q in check.box_corners(c["pose"], c["size_m"][0], c["size_m"][1])], fill=col)
    # timing line (checkered) and checkpoints
    for c in t["checkpoints"]:
        dr.line([P(*c["a"]), P(*c["b"])], fill=COLOURS["checkpoint"] + (200,), width=1)
    tl = t["timing_line"]
    n_sq = 8
    for k in range(n_sq):
        a = [tl["a"][0] + (tl["b"][0] - tl["a"][0]) * k / n_sq, tl["a"][1] + (tl["b"][1] - tl["a"][1]) * k / n_sq]
        bb = [tl["a"][0] + (tl["b"][0] - tl["a"][0]) * (k + 1) / n_sq, tl["a"][1] + (tl["b"][1] - tl["a"][1]) * (k + 1) / n_sq]
        dr.line([P(*a), P(*bb)], fill=(0, 0, 0) if k % 2 else (255, 255, 255), width=4)
    # ducts
    for d in t["ducts"]:
        col = COLOURS[d["wall_class"]]
        edge = (0, 0, 0) if d["wall_class"] != "black_plain" else (90, 90, 90)
        for piece in d["pieces"]:
            corners = check.box_corners(piece, d["piece_length_m"], K.DUCT_DIAMETER_M)
            dr.polygon([P(*q) for q in corners], fill=col, outline=edge)
    # open walls: tape
    for o in t["open_walls"]:
        (ax, ay), (bx, by) = o["tape"]
        dr.line([P(ax, ay), P(bx, by)], fill=COLOURS["tape"], width=3)
    # avoidance set
    for bx_ in t["avoidance"]["boxes"]:
        dr.polygon([P(*q) for q in check.box_corners(bx_["pose"], bx_["size_m"][0], bx_["size_m"][1])],
                   fill=COLOURS["box"], outline=(0, 0, 0))
    for sc in t["avoidance"]["slow_cars"]:
        pts = sc["path"]["samples"]
        for i in range(0, len(pts), 3):
            x, y = pts[i][0], pts[i][1]
            cx, cy = P(x, y)
            dr.point((cx, cy), fill=COLOURS["slow"])
    # the car at the start pose
    sp = t["start_pose"]["pose"]
    cx = 0.5 * (K.X_FRONT_M + K.X_REAR_M)
    c, s = math.cos(sp[2]), math.sin(sp[2])
    car = check.box_corners([sp[0] + c * cx, sp[1] + s * cx, sp[2]], K.X_FRONT_M - K.X_REAR_M, K.ENVELOPE_M[1])
    dr.polygon([P(*q) for q in car], fill=COLOURS["car"], outline=(255, 255, 255))
    if title:
        st = t["stats"]
        txt = "%s  %s / %s  %s  L %.1f m  footprint %.1f x %.1f m  width %.2f-%.2f m  ducts %d  grip %.3f" % (
            world["world_id"], t["cell"]["layout_class"], t["cell"]["grip_tier"], t["centreline"]["direction"],
            st["length_m"], st["footprint_m"][0], st["footprint_m"][1], st["min_width_m"],
            float((tr.left + tr.right).max()), st["n_ducts"], t["floor"]["base_grip"])
        dr.rectangle([0, 0, W, top], fill=(20, 20, 24))
        dr.text((8, 8), txt, fill=(240, 240, 240))
    img.save(path, format="PNG", optimize=False)
    return path


# ------------------------------------------------------------------------------------------------ files
def export_all(world, out_dir, png=True, mjcf=True):
    """The exports of one venue into out_dir. Returns {kind: path}."""
    from .gen_venue import write_world
    os.makedirs(out_dir, exist_ok=True)
    wid = world["world_id"]
    out = {"world": write_world(world, os.path.join(out_dir, wid + ".json"))}
    p = os.path.join(out_dir, wid + ".venue_sheet.json")
    with open(p, "w", encoding="ascii", newline="\n") as f:
        f.write(json.dumps(check.public_view(world), indent=1, sort_keys=False, ensure_ascii=True) + "\n")
    out["venue_sheet"] = p
    p = os.path.join(out_dir, wid + ".truth.json")
    with open(p, "w", encoding="ascii", newline="\n") as f:
        f.write(json.dumps({"world_id": wid, "layout_digest": world["layout_digest"], "generator": world["generator"],
                            "truth": world["truth"]}, separators=(",", ":"), ensure_ascii=True) + "\n")
    out["truth"] = p
    p = os.path.join(out_dir, wid + ".digest")
    with open(p, "w", encoding="ascii", newline="\n") as f:
        f.write(world["layout_digest"] + "  " + wid + "\n")
    out["digest"] = p
    if mjcf:
        p = os.path.join(out_dir, wid + ".xml")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(model_xml(world, "track"))
        out["mjcf"] = p
    if png:
        out["png"] = top_view(world, os.path.join(out_dir, wid + ".png"))
    return out
