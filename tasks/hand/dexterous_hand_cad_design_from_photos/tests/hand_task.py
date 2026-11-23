"""hand_task: the hand's mechanism specification (the hidden truth of the walk) and the vocabulary shared by the grader's
mechanism module (handtree.py) and the oracle builder. Pure stdlib + numpy; no FreeCAD, no geometry kernel.

The vocabulary is fixed: the digits index, middle, ring (fingers) and thumb; a finger's chain PALM - A - B - C - D with
the link bar E closing its loop, the thumb's chain PALM - TA - TB - TC - TD; the hinges of a finger h1, h2, h3 (servo
hinges), distal (C - D), base_pivot (B - E) and link_pin (E - D) (pin hinges), of the thumb h1 .. h4. Everything else is
read, never assumed: which digits the reference holds, each link's made parts (tags), given instances (placement ids),
structural parts and anchor, each hinge's centre, axis (its positive sense), parent and child link, driver (the servo
instance whose output axis it is) and the side of the servo's case, the true limits of the independent hinges, the loops,
the fingertip points (each a point of a given tactile module instance). Lengths mm, angles rad, in the reference frame
(configuration Z).

The block is `reference.json["mechanism"]` (schema hand-cad-mechanism/1); a reference folder may instead hold it as
`mechanism.json` beside reference.json.

    spec = load_spec(ref)          HandSpec of a loaded grade.Reference (raises SpecError: the task's own files are broken)
    cfg = mech_config(cfg)         the walk's numbers (grader_config.json: mechanism, mass, fit, interference blocks)
"""
import json
import math
import os

import numpy as np

SCHEMA = "hand-cad-mechanism/1"
PALM = "PALM"
FINGERS = ("index", "middle", "ring")
THUMB = "thumb"
DIGITS = FINGERS + (THUMB,)
FINGER_LINKS = ("A", "B", "C", "D")
THUMB_LINKS = ("TA", "TB", "TC", "TD")
FINGER_HINGES = ("h1", "h2", "h3", "distal", "base_pivot", "link_pin")
THUMB_HINGES = ("h1", "h2", "h3", "h4")
SERVO = "SERVO"
AXIS_ONLY_TYPES = ("PIN", "BRG")
TIP_TYPES = {"finger": "TIP-F", "thumb": "TIP-T"}

# every number of the walk and of the four mechanism criteria; grader_config.json must hold each of them
REQUIRED = {
    "mechanism": ("coaxial_mm", "coaxial_deg", "frames_apart_mm", "frames_apart_deg", "slot_max_mm", "slot_max_deg",
                  "slot_mm_per_deg", "slot_tiebreak_per_mm", "driver_deg", "driver_mm", "consistent_deg", "parallel_deg",
                  "consistent_mm", "loop_parallel_deg", "loop_step_deg", "toggle_margin_deg", "pin_along_slack_mm",
                  "full_turn_tol_deg", "range_tol_deg", "weights_finger", "weights_thumb", "axis_mm", "axis_deg",
                  "length_mm", "length_deg", "fingertip_median_mm", "fingertip_p90_mm", "fingertip_w_median",
                  "coupling_deg", "coupling_angles", "poses_per_digit", "unclosed_error_mm"),
    "mass": ("palm_weight", "link_weight", "w_mass", "w_com", "mass_pct", "com_mm", "extra_density_g_mm3",
             "excluded_types"),
    "fit": ("w_given", "w_made", "gap_mm", "overlap_pct", "contact_patch_mm"),
    "interference": ("w_within", "w_between", "adjacent_ratio", "other_ratio"),
}


class SpecError(RuntimeError):
    """The reference's mechanism block is missing or inconsistent: a task error, never the agent's."""


def mech_config(cfg):
    """The walk's numbers from grader_config.json, checked for presence (no default decides a score)."""
    out = {}
    for block, keys in REQUIRED.items():
        b = cfg.get(block)
        if not isinstance(b, dict):
            raise SpecError("grader_config.json has no %s block" % block)
        missing = [k for k in keys if k not in b]
        if missing:
            raise SpecError("grader_config.json %s lacks %s" % (block, ", ".join(missing)))
        out[block] = b
    fk = cfg.get("floors", {}).get("fk")
    if not (isinstance(fk, list) and len(fk) == 2):
        raise SpecError("grader_config.json floors.fk is not [full, zero]")
    out["fk"] = [float(fk[0]), float(fk[1])]
    return out


def _v3(a, what, unit=False):
    v = np.asarray(a, float).reshape(-1)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise SpecError("%s is not a finite 3-vector" % what)
    if unit:
        n = float(np.linalg.norm(v))
        if not n > 0.5:
            raise SpecError("%s is not a direction" % what)
        v = v / n
    return v


class HandSpec(object):
    """The reference hand's mechanism. Attributes (all ids as in reference.json):

    digits   {digit: {name, kind finger|thumb, chain [link, ...] (PALM excluded), bar link or None, hinges [id, ...],
             independent [id, ...], loop {driver, distal, base_pivot, link_pin} or None, tip {type, placement,
             point_mm np 3}, tip_type}}
    links    {link: {name, digit (None for PALM), key, made [tag], given [placement id], structural [tag], anchor}}
    hinges   {id: {id, digit, key, kind servo|pin, parent, child, centre np 3, axis np 3 (unit, the positive sense),
             independent, driver placement id or None, case_side parent|child|None, limits (lo, hi) or None,
             located (the reference's mark; absent = true)}}
    link_of_tag {tag: link}, link_of_placement {placement id: link}, digit_of_link {link: digit or None}
    order    the digits present, in DIGITS order
    """

    def __init__(self, block, tags, placements, placement_types):
        if not isinstance(block, dict) or block.get("schema") != SCHEMA:
            raise SpecError("the mechanism block's schema is %r (expected %s)" % (
                block.get("schema") if isinstance(block, dict) else None, SCHEMA))
        self.raw = block
        tags, placements = set(tags), set(placements)
        self.links = {}
        for name, L in sorted(block.get("links", {}).items()):
            rec = {"name": name, "digit": L.get("digit"), "key": L.get("key"), "made": list(L.get("made", [])),
                   "given": list(L.get("given", [])), "structural": list(L.get("structural", [])),
                   "anchor": L.get("anchor")}
            if name == PALM:
                rec["digit"] = None
            for t in rec["made"] + rec["structural"]:
                if t not in tags:
                    raise SpecError("link %s names the tag %s, which the reference does not hold" % (name, t))
            for p in rec["given"]:
                if p not in placements:
                    raise SpecError("link %s names the placement %s, which the reference does not hold" % (name, p))
            if rec["anchor"] not in rec["made"] + rec["given"]:
                raise SpecError("link %s: its anchor %r is not one of its members" % (name, rec["anchor"]))
            if not set(rec["structural"]) <= set(rec["made"]):
                raise SpecError("link %s: a structural part that is not one of its made parts" % name)
            self.links[name] = rec
        if PALM not in self.links:
            raise SpecError("the mechanism block has no PALM link")
        self.link_of_tag, self.link_of_placement = {}, {}
        for name, L in self.links.items():
            for t in L["made"]:
                if t in self.link_of_tag:
                    raise SpecError("tag %s is in two links" % t)
                self.link_of_tag[t] = name
            for p in L["given"]:
                if p in self.link_of_placement:
                    raise SpecError("placement %s is in two links" % p)
                self.link_of_placement[p] = name
        self.hinges = {}
        for h in block.get("hinges", []):
            hid = h["id"]
            if hid in self.hinges:
                raise SpecError("hinge %s listed twice" % hid)
            for k in ("parent", "child"):
                if h.get(k) not in self.links:
                    raise SpecError("hinge %s: unknown %s link %r" % (hid, k, h.get(k)))
            kind = h.get("kind")
            if kind not in ("servo", "pin"):
                raise SpecError("hinge %s: kind %r" % (hid, kind))
            drv = h.get("driver")
            if kind == "servo":
                if drv not in placements or placement_types.get(drv) != SERVO:
                    raise SpecError("hinge %s: its driver %r is not a servo placement" % (hid, drv))
                if h.get("case_side") not in ("parent", "child"):
                    raise SpecError("hinge %s: case_side %r" % (hid, h.get("case_side")))
            located = h.get("located", True)
            if not isinstance(located, bool):
                raise SpecError("hinge %s: located is %r, not true or false" % (hid, located))
            lim = h.get("limits_rad")
            if h.get("independent"):
                if not (isinstance(lim, list) and len(lim) == 2 and all(math.isfinite(float(x)) for x in lim)
                        and float(lim[0]) <= 0.0 <= float(lim[1])):
                    raise SpecError("independent hinge %s has no limits around Z" % hid)
            self.hinges[hid] = {"id": hid, "digit": h.get("digit"), "key": h.get("key"), "kind": kind,
                                "parent": h["parent"], "child": h["child"],
                                "centre": _v3(h["centre_mm"], "hinge %s centre_mm" % hid),
                                "axis": _v3(h["axis"], "hinge %s axis" % hid, unit=True),
                                "independent": bool(h.get("independent")), "driver": drv if kind == "servo" else None,
                                "case_side": h.get("case_side") if kind == "servo" else None,
                                "limits": (float(lim[0]), float(lim[1])) if h.get("independent") else None,
                                "located": located}
        self.digit_of_link = {n: L["digit"] for n, L in self.links.items()}
        loops = {lp["digit"]: lp for lp in block.get("loops", [])}
        tips = {t["digit"]: t for t in block.get("tips", [])}
        self.digits = {}
        for d, D in sorted(block.get("digits", {}).items()):
            if d not in DIGITS:
                raise SpecError("unknown digit %r" % d)
            kind = D.get("kind")
            want = FINGER_LINKS if kind == "finger" else THUMB_LINKS
            chain = list(D.get("chain", []))
            if [self.links.get(c, {}).get("key") for c in chain] != list(want):
                raise SpecError("digit %s: chain %s (keys %s expected)" % (d, chain, list(want)))
            hkeys = FINGER_HINGES if kind == "finger" else THUMB_HINGES
            hids = ["%s.%s" % (d, k) for k in hkeys]
            for hid in hids:
                if hid not in self.hinges:
                    raise SpecError("digit %s lacks hinge %s" % (d, hid))
            full = [PALM] + chain
            for i, k in enumerate(("h1", "h2", "h3") + (("h4",) if kind == "thumb" else ())):
                h = self.hinges["%s.%s" % (d, k)]
                if (h["parent"], h["child"]) != (full[i], full[i + 1]) or h["kind"] != "servo":
                    raise SpecError("digit %s: servo hinge %s is not between %s and %s" % (d, k, full[i], full[i + 1]))
            bar = D.get("bar")
            loop = None
            if kind == "finger":
                if bar not in self.links or self.links[bar].get("key") != "E":
                    raise SpecError("finger %s: no link bar" % d)
                A, B, C, Dl = chain
                need = {"distal": (C, Dl), "base_pivot": (B, bar), "link_pin": (bar, Dl)}
                for k, (pa, ch) in need.items():
                    h = self.hinges["%s.%s" % (d, k)]
                    if (h["parent"], h["child"]) != (pa, ch) or h["kind"] != "pin":
                        raise SpecError("finger %s: pin hinge %s is not between %s and %s" % (d, k, pa, ch))
                lp = loops.get(d)
                if lp is None:
                    raise SpecError("finger %s has no loop" % d)
                loop = {"driver": lp["driver_hinge"], "distal": lp["distal_hinge"], "base_pivot": lp["base_pivot_hinge"],
                        "link_pin": lp["link_pin_hinge"]}
                if loop != {"driver": d + ".h3", "distal": d + ".distal", "base_pivot": d + ".base_pivot",
                            "link_pin": d + ".link_pin"}:
                    raise SpecError("finger %s: the loop does not name the finger's hinges" % d)
            t = tips.get(d)
            if t is None or t.get("placement") not in placements or placement_types.get(t["placement"]) != TIP_TYPES[kind]:
                raise SpecError("digit %s: its tip is not a point of a %s placement" % (d, TIP_TYPES[kind]))
            indep = [hid for hid in hids if self.hinges[hid]["independent"]]
            self.digits[d] = {"name": d, "kind": kind, "chain": chain, "bar": bar if kind == "finger" else None,
                              "hinges": hids, "independent": indep, "loop": loop, "tip_type": TIP_TYPES[kind],
                              "tip": {"type": t["placement"] and placement_types[t["placement"]],
                                      "placement": t["placement"], "point_mm": _v3(t["point_mm"], "tip of %s" % d)}}
        self.order = [d for d in DIGITS if d in self.digits]
        if not self.order:
            raise SpecError("the mechanism block holds no digit")

    def hinge_between(self, parent, child):
        for h in self.hinges.values():
            if h["parent"] == parent and h["child"] == child:
                return h
        return None

    def chain_hinges(self, d):
        """The servo hinges and the distal hinge of digit d in chain order (h1, h2, h3, distal / h1 .. h4)."""
        D = self.digits[d]
        full = [PALM] + D["chain"]
        return [self.hinge_between(full[i], full[i + 1])["id"] for i in range(len(D["chain"]))]


def block_from_layout(spec):
    """The mechanism block from the reference build's own top-level layout of reference.json (links {name: anchor,
    digit, made, given, structural}; digits {d: chain with PALM, hinges, loop_bar, tip}; hinges [id, digit, key, kind,
    parent, child, centre_mm, axis, independent, driver {placement, case_side}]; limits {id: lower_rad, upper_rad};
    loops; tips [digit, module_type, placement, point_mm])."""
    try:
        links = {}
        for name, L in spec["links"].items():
            links[name] = {"digit": None if name == PALM else L.get("digit"),
                           "key": PALM if name == PALM else name.split(".", 1)[1],
                           "made": list(L.get("made", [])), "given": list(L.get("given", [])),
                           "structural": list(L.get("structural", [])), "anchor": L.get("anchor")}
        digits = {}
        for d, D in spec["digits"].items():
            chain = [c for c in D["chain"] if c != PALM]
            digits[d] = {"kind": "finger" if D.get("loop_bar") else "thumb", "chain": chain, "bar": D.get("loop_bar")}
        lim = spec.get("limits", {})
        hinges = []
        for h in spec["hinges"]:
            drv = h.get("driver") or {}
            rec = {"id": h["id"], "digit": h.get("digit"), "key": h.get("key"), "kind": h.get("kind"),
                   "parent": h["parent"], "child": h["child"], "centre_mm": h["centre_mm"], "axis": h["axis"],
                   "independent": bool(h.get("independent")), "driver": drv.get("placement"),
                   "case_side": drv.get("case_side"), "limits_rad": None}
            if rec["independent"]:
                rec["limits_rad"] = [lim[h["id"]]["lower_rad"], lim[h["id"]]["upper_rad"]]
            hinges.append(rec)
        loops = [{k: lp[k] for k in ("digit", "driver_hinge", "distal_hinge", "base_pivot_hinge", "link_pin_hinge")}
                 for lp in spec.get("loops", [])]
        tips = [{"digit": t["digit"], "type": t.get("module_type") or t.get("type"), "placement": t["placement"],
                 "point_mm": t["point_mm"]} for t in spec.get("tips", [])]
    except (KeyError, TypeError, AttributeError) as exc:
        raise SpecError("the reference's mechanism layout cannot be read: %s %s" % (type(exc).__name__, exc))
    return {"schema": SCHEMA, "links": links, "digits": digits, "hinges": hinges, "loops": loops, "tips": tips}


def load_spec(ref):
    """The HandSpec of a loaded grade.Reference (its spec dict and folder), cached on the reference: the block
    reference.json["mechanism"], else mechanism.json beside it, else the reference build's top-level layout."""
    got = getattr(ref, "_hand_spec", None)
    if got is not None:
        return got
    block = ref.spec.get("mechanism")
    if block is None:
        path = os.path.join(ref.dir, "mechanism.json")
        if os.path.isfile(path):
            with open(path) as fh:
                block = json.load(fh)
        elif all(k in ref.spec for k in ("links", "digits", "hinges", "loops", "tips")):
            block = block_from_layout(ref.spec)
        else:
            raise SpecError("the reference holds no mechanism (reference.json mechanism, mechanism.json, or links, "
                            "digits, hinges, loops and tips)")
    spec = HandSpec(block, ref.parts.keys(), ref.placements.keys(),
                    {p: pl["type"] for p, pl in ref.placements.items()})
    ref._hand_spec = spec
    return spec
