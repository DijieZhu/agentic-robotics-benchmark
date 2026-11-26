import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import minimal as MR
import shift as SH
from openarm_unit import world as W

import numpy as np

FIELD_OF = {"M-load": "trays", "M-fric": "friction", "M-geom": "geometry", "M-geom-d": "geometry",
            "M-shelf": "shelf_pose", "M-int": "geometry", "M-grip": "squeeze", "M-k": "torque_scale",
            "M-loop": None}
SIZES = {
    "M-load": ("documents", "mass+0.1", "mass-0.1", "mass+0.35", "mass-0.35", "mass+0.7", "mass-0.7",
               "centre+0.02", "centre-0.02", "centre+0.04", "centre-0.04", "centre_nearer_mid0.02"),
    "M-fric": (0.25, 0.5, 1.0),
    "M-geom": (0.25, 0.5, 1.0, 2.0),
    "M-geom-d": (0.25, 0.5, 1.0, 2.0),
    "M-shelf": ("drawn",),
    "M-int": ("per_arm", "common", "left"),
    "M-grip": (0.6, 0.8, 1.2, 1.4, 1.7, "driver_default", "bus_default", "mit", "documents"),
    "M-k": (1.0,),
    "M-loop": ("host_wrist",),
}
DOCUMENTS_TRAY = (1.25, (0.0, 0.0, -0.0225))
MAKER_DEFAULT_ILIM = {"driver_default": 1.0 / 4.5, "bus_default": 0.5}
MAKER_MIT_GRIP = {"kp": 16.0, "kd": 0.2}
DOCUMENTS_WORST = {"pad_friction": 0.55, "finger_friction_Nm": 0.92, "torque_scale": 0.88}


def _lerp_pose(a, b, k):
    p = tuple(float(a[0][i] + k * (b[0][i] - a[0][i])) for i in range(3))
    q = W.qnormalize(tuple(float(a[1][i] + k * (b[1][i] - a[1][i])) for i in range(4)))
    return p, q


def drawn_mounting():
    from openarm_unit.state import public_params
    ys = public_params()["conventions"]["base_link_y_m"]["value"]
    return tuple(((0.0, float(ys[s]), 0.0), (1.0, 0.0, 0.0, 0.0)) for s in SH.SIDES)


def _geometry(B, k, fixture_drawn=False):
    g = B.geometry
    z = tuple(float(x) * (1.0 - k) for x in g["zero_residual"])
    dm = drawn_mounting()
    mt = tuple(_lerp_pose(g["mounting"][a], dm[a], k) for a in range(2))
    fx = SH.pose_of((list(SH.Cell.F_DRAWN[0]), list(SH.Cell.F_DRAWN[1]))) if fixture_drawn else g["fixture_pose"]
    return {"zero_residual": z, "mounting": mt, "fixture_pose": fx}


def _load_belief(size, m, c):
    if size == "documents":
        return DOCUMENTS_TRAY
    if size.startswith("mass"):
        return (max(0.62 + 1e-6, m + float(size[4:])), c)
    if size.startswith("centre_nearer_mid"):
        d = float(size[len("centre_nearer_mid"):])
        a = float(c[1])
        return (m, (c[0], math.copysign(max(0.0, abs(a) - d), a) if a != 0.0 else 0.0, c[2]))
    return (m, (c[0], c[1] + float(size[6:]), c[2]))


def mutant(name, B, size=None, slot=None, slots=None, squeeze_from_belief=False, precheck_form=False):
    if name not in FIELD_OF:
        raise KeyError(name)
    size = SIZES[name][0] if size is None else size
    if size not in SIZES[name]:
        raise ValueError(f"{name}: size {size!r} is not one of {SIZES[name]}")
    if (slot is not None or squeeze_from_belief) and name != "M-load":
        raise ValueError(f"{name}: slot and squeeze_from_belief are M-load's options")
    if precheck_form and name != "M-geom-d":
        raise ValueError(f"{name}: precheck_form is M-geom-d's option")
    opt = {}
    if name == "M-load":
        idx = None
        if slot is not None:
            if slots is None:
                raise ValueError("M-load for one slot needs the run's slot list (slots=...)")
            idx = [k for k, sl in enumerate(slots) if int(sl["slot"]) == int(slot)]
            if len(idx) != 1:
                raise ValueError(f"M-load: slot {slot} is not once in the slot list")
            idx = idx[0]
        trays = []
        for k, (m, c) in enumerate(B.trays):
            trays.append(_load_belief(size, m, c) if idx is None or k == idx else (m, c))
        Bm = MR.replace(B, trays=tuple(trays))
        if squeeze_from_belief:
            sq = dict(B.squeeze)
            if sq.get("kind") != "reference":
                raise ValueError("squeeze_from_belief needs the reference's kind of squeeze")
            sq["load"] = "belief"
            Bm = MR.replace(Bm, squeeze=sq)
        return Bm, opt
    if name == "M-fric":
        k = float(size)
        tw, base14, prof = B.friction["terms"][0]
        terms = ((1.0 - k, base14, prof), (k, tuple(MR.MAKER_MODEL_FRICTION14), None))
        damp = tuple(float(d) + k * (MR.MAKER_MODEL_DAMPING14[a] - float(d)) for a, d in
                     enumerate(B.friction["damping"]))
        return MR.replace(B, friction={"terms": terms, "damping": damp}), opt
    if name == "M-geom":
        return MR.replace(B, geometry=_geometry(B, float(size))), opt
    if name == "M-geom-d":
        opt = {"geom_mode": "displacement"}
        if precheck_form:
            opt["displacement_scope"] = "shift"
        return MR.replace(B, geometry=_geometry(B, float(size), fixture_drawn=True)), opt
    if name == "M-shelf":
        return MR.replace(B, shelf_pose=SH.pose_of((list(SH.Cell.S_DRAWN[0]), list(SH.Cell.S_DRAWN[1])))), opt
    if name == "M-int":
        return MR.replace(B, geometry=_geometry(B, 1.0)), {"integral": size}
    if name == "M-grip":
        sq = dict(B.squeeze)
        if isinstance(size, float):
            sq["factor"] = float(sq.get("factor", 1.0)) * size
        elif size == "mit":
            sq = {"kind": "mit", **MAKER_MIT_GRIP}
        elif size == "documents":
            if sq.get("kind") != "reference":
                raise ValueError("the documents' clamp needs the reference's kind of squeeze")
            sq["load"] = "belief"
            sq["printed_worst"] = dict(DOCUMENTS_WORST)
        else:
            sq = {"kind": "fixed_ilim", "ilim": MAKER_DEFAULT_ILIM[size]}
        return MR.replace(B, squeeze=sq), opt
    if name == "M-k":
        return MR.replace(B, torque_scale=(1.0,) * 16), opt
    if name == "M-loop":
        return B, {"loop_wrist": True, "loop_damping": MR.MAKER_MODEL_DAMPING}
    raise KeyError(name)


def compose(B, specs):
    Bm, opt, applied = B, {}, []
    for sp in specs:
        name, size = sp[0], sp[1]
        kw = dict(sp[2]) if len(sp) > 2 and sp[2] else {}
        B0 = Bm
        Bm, o = mutant(name, Bm, size, **kw)
        for k, v in o.items():
            if k in opt and opt[k] != v:
                raise ValueError(f"law option {k!r} set to {opt[k]!r} and to {v!r}")
            opt[k] = v
        applied.append({"name": name, "size": size, "options": {k: v for k, v in kw.items() if k != "slots"},
                        "law_options": dict(o), "differs": list(Bm.differs_from(B0))})
    return Bm, opt, applied


class FrameLog:

    def __init__(self, ref):
        self.ref = ref
        self.cycles, self.bus, self.ids, self.data, self.decline = [], [], [], [], []
        self.modes = None
        setup0, step0 = ref.setup, ref.step

        def setup(unit, site, slots):
            out = setup0(unit, site, slots)
            self.modes = dict(out["gripper_mode"])
            return out

        def step(c, sig, replies):
            frames, decline = step0(c, sig, replies)
            b, i, d = np.full(16, -1, np.int64), np.full(16, -1, np.int64), np.zeros((16, 8), np.uint8)
            for k, f in enumerate(frames):
                if f is not None:
                    b[k], i[k] = int(f.bus), int(f.can_id)
                    d[k, :len(f.data)] = np.frombuffer(bytes(f.data), dtype=np.uint8)
            self.cycles.append(int(c))
            self.bus.append(b)
            self.ids.append(i)
            self.data.append(d)
            self.decline.append(bool(decline))
            return frames, decline
        ref.setup, ref.step = setup, step

    def save(self, path):
        np.savez_compressed(path, cycle=np.asarray(self.cycles, np.int64), bus=np.asarray(self.bus, np.int64),
                            can_id=np.asarray(self.ids, np.int64), data=np.asarray(self.data, np.uint8),
                            decline=np.asarray(self.decline, bool),
                            modes=np.array([self.modes["left"], self.modes["right"]]))
        return {"path": path, "cycles": len(self.cycles)}


class Replay:

    def __init__(self, path, slots, jobs):
        from openarm_unit.frames import Frame
        z = np.load(path)
        self._Frame = Frame
        self.cycle = z["cycle"]
        self.bus, self.can_id, self.data, self.dec = z["bus"], z["can_id"], z["data"], z["decline"]
        self.modes = {"left": str(z["modes"][0]), "right": str(z["modes"][1])}
        self.slots, self.jobs = slots, jobs
        self.timeline = []
        self.sent = 0
        if len(self.cycle) and not np.array_equal(self.cycle, np.arange(len(self.cycle))):
            raise ValueError("the stored frames are not one per cycle from cycle 0")

    def setup(self, unit, site, slots):
        return {"gripper_mode": dict(self.modes)}

    def start(self, unit):
        pass

    def slot_plan_lengths(self):
        return None

    def step(self, c, site_signal, replies):
        k = min(int(c), len(self.cycle) - 1)
        frames = []
        for m in range(16):
            if int(self.can_id[k, m]) < 0:
                frames.append(None)
            else:
                frames.append(self._Frame(int(self.bus[k, m]), int(self.can_id[k, m]), bytes(self.data[k, m])))
        self.sent += 1
        return frames, bool(self.dec[k]) if int(c) < len(self.cycle) else False

    def on_trace(self, recs):
        pass

    def figures(self):
        return {"replay": {"stored_cycles": int(len(self.cycle)), "sent": int(self.sent)}}


def shelf_site(site, top_dz_m):
    out = dict(site)
    S = dict(site["S"])
    p, q = SH.Cell.S_DRAWN
    S["true"] = [[p[0], p[1], p[2] + float(top_dz_m)], list(q)]
    S["offset_m"] = [0.0, 0.0, float(top_dz_m)]
    S["offset_deg"] = [0.0, 0.0, 0.0]
    out["S"] = S
    return out


def run(name, size, unit, state, session, site, slots, jobs, beliefs=None, **kw):
    B = MR.beliefs_from_truth(unit, state, session, site, slots) if beliefs is None else beliefs
    Bm, opt = mutant(name, B, size)
    ref = MR.MinimalReference(Bm, slots, jobs, options=opt)
    res = SH.run_shift(unit, site, slots, ref, **kw)
    res["mutant"] = {"name": name, "size": size, "differs": list(Bm.differs_from(B)), "options": opt}
    return res
