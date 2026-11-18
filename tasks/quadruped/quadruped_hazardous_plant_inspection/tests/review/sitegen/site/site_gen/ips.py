"""Inspection points: types (at least one of each of the first five), constructive placement for the guarantees
(one in the pipe chase -> needs_crouch, one in the high-step-gated room -> needs_high_step, one in the annex ->
behind a discrepancy), the rest spread over the rooms; ground-truth values; tags; decoys; untagged instruments."""

from __future__ import annotations

import math

import numpy as np

from .util import choice, uniform

PREFIX = {"gauge": "PI", "valve_lever": "HV", "valve_handwheel": "V", "level": "LG", "lamps": "LCP",
          "thermal": "TS", "dose_point": "DP", "hotspot": "HS"}
# a dial's tag letters follow its quantity: PI pressure, TI temperature
GAUGE_PREFIX = {"°C": "TI"}
REPORT = {"gauge": "reading in the dial's unit", "valve_lever": "open / closed / angle",
          "valve_handwheel": "opening in percent", "level": "level in percent",
          "lamps": "on/off state of each lamp (left to right)",
          "thermal": "which unit is hot and its surface temperature", "dose_point": "dose rate (uSv/h)",
          "hotspot": "position and peak dose rate of the unlisted source"}
LAMP_COLORS = ["green", "red", "amber", "white"]


def choose_types(cfg, g):
    I = cfg["ips"]
    types = list(I["mandatory"])
    extra = dict(I["extra_weights"])
    while len(types) < int(I["count"]):
        keys = sorted(extra.keys())
        w = np.array([extra[k] for k in keys], float)
        k = keys[int(g.choice(len(keys), p=w / w.sum()))]
        types.append(k)
        if k in ("thermal", "dose_point", "hotspot"):
            extra.pop(k)
    return types


def case_for_height(cfg, z, hint):
    """The smallest dial case (not below the hint) that the robot's head can read at mount height z from inside
    its window: the camera tilt at the window's largest stand-off must stay below the window's elevation limit."""
    h = float(cfg["modes"]["walk"]["head_height"])
    win = cfg["observability"]
    cases = sorted(int(c) for c in cfg["ips"]["gauge_cases_mm"])
    for c in cases:
        if c < hint:
            continue
        w = win.get("gauge_%d" % c)
        if w is None:
            continue
        w = w[0] if isinstance(w, (list, tuple)) else w
        if z - h <= 0.8 * float(w["d"][1]) * math.tan(math.radians(float(w["elev_deg"]))):
            return c
    return cases[-1]


def _truth_and_params(cfg, g, slot, typ):
    I = cfg["ips"]
    lo_f, hi_f = I["value_span"]
    prm, tr = {}, {}
    if typ == "gauge":
        lo, hi, unit = choice(g, I["gauge_scales"])
        case = int(slot.get("case_hint") or choice(g, I["gauge_cases_mm"]))
        case = case_for_height(cfg, slot["pos"][2], case)
        span = hi - lo
        v = lo + uniform(g, lo_f, hi_f) * span
        step = span / 200.0
        v = round(round(v / step) * step, 4)
        prm = {"lo": lo, "hi": hi, "unit": unit, "case_mm": case}
        tr = {"value": v, "unit": unit}
    elif typ == "valve_lever":
        r = g.random()
        ang = 0.0 if r < 0.4 else (90.0 if r < 0.75 else round(uniform(g, 20, 70)))
        state = "open" if ang == 0 else ("closed" if ang == 90 else "throttled")
        pp = slot.get("pipe", {})
        prm = {"pipe_axis": pp.get("axis", [1, 0, 0]), "pipe_r": pp.get("r", 0.04)}
        tr = {"angle_deg": ang, "state": state}
    elif typ == "valve_handwheel":
        pp = slot.get("pipe", {})
        prm = {"pipe_axis": pp.get("axis", [1, 0, 0]), "pipe_r": pp.get("r", 0.05), "stem": slot.get("stem", "up")}
        tr = {"opening_pct": float(round(uniform(g, 5, 95)))}
    elif typ == "level":
        prm = {"span": slot.get("span", [0.35, 1.75])}
        tr = {"level_pct": float(round(uniform(g, 8, 92)))}
    elif typ == "lamps":
        n = int(g.integers(I["lamps_count"][0], I["lamps_count"][1] + 1))
        cols = [choice(g, LAMP_COLORS) for _ in range(n)]
        st = [bool(g.random() < 0.5) for _ in range(n)]
        if all(st):
            st[int(g.integers(n))] = False
        if not any(st):
            st[int(g.integers(n))] = True
        prm = {"colors": cols}
        tr = {"states": st}
    elif typ == "thermal":
        units = slot.get("units", [])
        temps = [round(uniform(g, *I["thermal_normal_C"]), 1) for _ in units]
        hot = None
        if units and g.random() >= float(I["thermal_none_probability"]):
            hot = int(g.integers(len(units)))
            temps[hot] = round(uniform(g, *I["thermal_hot_C"]), 1)
        prm = {"n_units": len(units)}
        tr = {"units": [dict(u, temp_C=t) for u, t in zip(units, temps)], "hot_index": hot,
              "hot_label": units[hot]["label"] if hot is not None else None}
    elif typ == "dose_point":
        prm = {"label": ""}
        tr = {}
    return prm, tr


def tag_prefix(typ, unit=None):
    """The tag letters of an instrument: by type, a dial by its quantity (unit)."""
    if typ == "gauge" and unit in GAUGE_PREFIX:
        return GAUGE_PREFIX[unit]
    return PREFIX[typ]


class Tagger:
    def __init__(self, g):
        self.g = g
        self.used = set()

    def __call__(self, typ, room_number, unit=None):
        digits = "".join(ch for ch in room_number if ch.isdigit()) or "0"
        pre = tag_prefix(typ, unit)
        for k in range(400):
            if typ in ("dose_point", "hotspot"):
                t = "%s-%02d" % (pre, int(self.g.integers(1, 60)))
            elif k < 40:
                t = "%s-%s%d" % (pre, digits, int(self.g.integers(1, 10)))
            else:
                t = "%s-%s%02d" % (pre, digits, int(self.g.integers(10, 99)))
                if len(t) > 8:
                    continue          # version 2's tag grammar: at most 8 characters (bricks.lettering.TAG_MAX_CHARS)
            if t not in self.used:
                self.used.add(t)
                return t
        raise RuntimeError("tags exhausted")


def select(W, cfg, rng, hotspot_source=None):
    """Return (ips, decoys, plain) lists of instrument records (slot + role + tag + truth + params)."""
    g = rng("ips")
    lay = W.lay
    rooms = lay.rooms
    types = choose_types(cfg, g)
    slots = [s for s in W.slots if not s.get("decoy_only")]
    by_type = {}
    for s in slots:
        by_type.setdefault(s["type"], []).append(s)
    chosen = []          # (type, slot, reason)
    used_slots = set()
    room_count = {}
    remaining = list(types)

    def take(typ, s, reason):
        chosen.append((typ, s, reason))
        used_slots.add(s["id"])
        room_count[s["room"]] = room_count.get(s["room"], 0) + 1
        remaining.remove(typ)

    def pick(cands, typ, reason, prefer_new_room=True):
        cands = [s for s in cands if s["id"] not in used_slots and
                 room_count.get(s["room"], 0) < int(cfg["ips"]["max_per_room"])]
        if not cands:
            return False
        if prefer_new_room:
            fresh = [s for s in cands if s["room"] not in room_count]
            if fresh:
                cands = fresh
        cands.sort(key=lambda s: s["id"])
        take(typ, cands[int(g.integers(len(cands)))], reason)
        return True

    chase = lay.params["chase"]
    gated = lay.params.get("gated_room")
    annex = lay.params["annex"]
    # 1) needs_crouch: a slot in the chase station
    order = [t for t in ("gauge", "dose_point", "valve_lever") if t in remaining]
    for t in order:
        if pick([s for s in by_type.get(t, []) if s["room"] == chase], t, "needs_crouch"):
            break
    # 2) needs_high_step: a slot in the gated room
    if gated:
        rt = rooms[gated].type
        pref = {"pump_room": ["thermal", "valve_handwheel", "lamps", "gauge", "valve_lever"],
                "electrical_room": ["lamps", "thermal"], "tank_room": ["level", "valve_handwheel", "dose_point"],
                "waste_store": ["dose_point"], "valve_gallery": ["valve_handwheel", "gauge", "valve_lever"]}[rt]
        for t in pref:
            if t in remaining and pick([s for s in by_type.get(t, []) if s["room"] == gated], t, "needs_high_step"):
                break
    # 3) behind a discrepancy: a slot in the annex (not on the drawing)
    for t in ("lamps", "gauge"):
        if t in remaining and pick([s for s in by_type.get(t, []) if s["room"] == annex], t, "room_missing"):
            break
    # 4) the hotspot (an undocumented source)
    if "hotspot" in remaining and hotspot_source is not None:
        s = W.slot("hotspot", hotspot_source["pos"][:2] + [0.0], 0.0, hotspot_source["room"],
                   "unlisted source: survey for it", "radiation survey", hotspot_source["id"])
        take("hotspot", s, "survey")
    elif "hotspot" in remaining:
        remaining.remove("hotspot")
        remaining.append("dose_point")
    # 5) the rest, spread over the rooms
    for t in list(remaining):
        cands = [s for s in by_type.get(t, []) if s["room"] not in (annex,)]
        if not pick(cands, t, "spread"):
            if not pick(by_type.get(t, []), t, "spread", prefer_new_room=False):
                # no slot of this type left: fall back to a gauge
                remaining.remove(t)
                remaining.append("gauge")
                pick(by_type.get("gauge", []), "gauge", "fallback", prefer_new_room=False)
    tagger = Tagger(rng("tags"))
    ips = []
    dpn = 0
    for typ, s, reason in chosen:
        prm, tr = _truth_and_params(cfg, g, s, typ)
        rn = rooms[s["room"]].number
        tag = tagger(typ, rn, unit=prm.get("unit"))
        if typ == "dose_point":
            dpn += 1
            prm["label"] = tag
        ips.append({"tag": tag, "type": typ, "slot": s, "role": "ip", "params": prm, "truth": tr,
                    "room": s["room"], "reason": reason, "body": "ip/%s" % tag})
    # weights
    wt = cfg["ips"]["weights"]
    tot = sum(float(wt[i["type"]]) for i in ips)
    for i in ips:
        i["weight"] = round(float(wt[i["type"]]) / tot * float(cfg["ips"]["weight_total"]), 4)
    # decoys: instruments with their own tags, not on the work order
    nd = int(g.integers(cfg["ips"]["decoys"][0], cfg["ips"]["decoys"][1] + 1))
    pool = [s for s in W.slots if s["id"] not in used_slots and s["type"] in
            ("gauge", "valve_lever", "valve_handwheel", "lamps", "level")]
    pool.sort(key=lambda s: s["id"])
    perm = [pool[i] for i in g.permutation(len(pool))]
    decoys = []
    rooms_d = {}
    for s in perm:
        if len(decoys) >= nd:
            break
        if rooms_d.get(s["room"], 0) >= 2:
            continue
        prm, tr = _truth_and_params(cfg, g, s, s["type"])
        tag = tagger(s["type"], rooms[s["room"]].number, unit=prm.get("unit"))
        decoys.append({"tag": tag, "type": s["type"], "slot": s, "role": "decoy", "params": prm, "truth": tr,
                       "room": s["room"], "body": "ip/%s" % tag})
        used_slots.add(s["id"])
        rooms_d[s["room"]] = rooms_d.get(s["room"], 0) + 1
    # untagged instruments: every remaining valve, level gauge and lamp panel; some gauges
    plain = []
    ng = 0
    for s in perm:
        if s["id"] in used_slots:
            continue
        if s["type"] in ("valve_lever", "valve_handwheel", "level", "lamps") or (s["type"] == "gauge" and ng < 8):
            if s["type"] == "gauge":
                ng += 1
            prm, tr = _truth_and_params(cfg, g, s, s["type"])
            plain.append({"tag": "", "type": s["type"], "slot": s, "role": "plain", "params": prm, "truth": tr,
                          "room": s["room"], "body": "inst/%s" % s["id"]})
            used_slots.add(s["id"])
    return ips, decoys, plain, types


def retag_draft(W, rng, ips, decoys, discs):
    """An IP's tag must not tell where it really is. A moved IP is numbered from its documented room, an
    IP without a recorded location (and every annex IP) from a randomly drawn room on the drawing; there is no
    special series for the annex. Tags of IPs at their documented place are unchanged (their room is public)."""
    g = rng("tags_fix")
    lay = W.lay
    drawn = sorted(r.number for r in lay.rooms.values() if r.on_drawing and r.type not in ("corridor", "airlock"))
    tagger = Tagger(g)
    tagger.used = {i["tag"] for i in ips} | {d["tag"] for d in decoys}
    for ip in ips:
        if ip["type"] in ("dose_point", "hotspot"):
            continue
        doc = ip.get("documented", "same")
        if isinstance(doc, dict):
            num = lay.rooms[doc["room"]].number
        elif doc is None or not lay.rooms[ip["room"]].on_drawing:
            num = drawn[int(g.integers(len(drawn)))]
        else:
            continue
        old = ip["tag"]
        new = tagger(ip["type"], num, unit=(ip.get("params") or {}).get("unit"))
        ip["tag"] = new
        ip["body"] = "ip/%s" % new
        for d in discs:
            d["refs"] = [new if x == old else x for x in d["refs"]]
            d["text"] = d["text"].replace(old, new)
