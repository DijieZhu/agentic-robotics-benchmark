"""Station ventilation, per-module fans and a module CO2 balance, built from the station feature catalogue
(the station's feature file: modules, vents, hatches; frame = the flight frame, z toward the deck).

Flow model (deterministic, mass-balanced, no CFD):
  * one axis-aligned box per module (median rack-face clearances, end walls) and one per hatch vestibule
    (1.27 m square opening between a module's end and its neighbour's wall);
  * fans: one cabin fan per module (``cabin:<module>``) drawing through all the module's deck return grilles
    (published design 14.375 cfm per grille) and supplying the same flow through its overhead diffusers;
    one inter-module fan per hatch link (``imv:<hub>:<lab>``, published 120-150 cfm, default 130 cfm): ducted from
    an intake in the hub to the lab's diffusers, back through the open hatch (the measured-design pattern:
    diffusers carry the cabin loop plus the inter-module supply, the hatch carries the return);
  * per fan a potential-flow basis field on each box it touches (``cabin.FlowFieldBuilder``: exact DCT Poisson
    solve, wall fluxes as given, mass balance checked); corner vents put half their flux on each adjacent wall;
  * per module a two-cell cross-sectional circulation (down the rack faces, up the middle; zero normal velocity at
    the walls, divergence-free), driven by the supply diffusers, its strength calibrated so the published cabin
    band holds (2/3 of the velocities in 0.076-0.61 m/s) -- a [G] stand-in for the jets' momentum;
  * the field at a point = sum over fans of (speed fraction) x basis + sum over modules of (cabin-fan fraction) x
    circulation, evaluated in the box that contains the point (nearest box otherwise).
Fans spin up / down first order (tau 3 s, an estimate).  The cabin fans are ``suction.FanNetwork``
objects, so the same object sets the return-inlet suction on items.

CO2 (``CO2Balance``): per module a bulk zone and a port zone (fractions of the module volume).  The cabin fan mixes
bulk <-> port with its loop flow (residual mixing ``residual_mix`` x that flow when it is stopped); inter-module
fans exchange port zones; a scrubber in one module removes eta Q_scrub c from its port zone while that module's cabin
fan runs.  Sources: none by default (uncrewed station); ``add_source(module, kg_s)`` for whatever a design adds.
So "fan off costs CO2 exchange" is computed, not asserted.  Units: kg CO2 per m^3; ``pp_mmhg`` converts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .cabin import FlowFieldBuilder
from .suction import CFM, FanNetwork, Inlet

__all__ = ["CABIN_BAND", "INLET_VICINITY_MAX", "RETURN_GRILLE_CFM", "SUPPLY_DIFFUSER_CFM", "INTER_MODULE_CFM", "HATCH_OPENING",
           "DESIGN_RETURN_INLET_CFM", "FAN_STOP_RULES", "FlowBox", "StationVentilation", "SimpleFan", "CO2Balance",
           "StationModel", "pp_mmhg", "load_features"]

CABIN_BAND = (0.0762, 0.6096)       # m/s, 2/3 of cabin velocities  [R human integration handbook 6.2.3.3]
INLET_VICINITY_MAX = 0.2032         # m/s near the inlets  [R same]
RETURN_GRILLE_CFM = 14.375          # per deck return grille  [R measured-design module CFD]
SUPPLY_DIFFUSER_CFM = 26.88         # per overhead diffuser  [R same]
INTER_MODULE_CFM = 130.0                     # inter-module, per hatch  [R same; station 120-150 cfm, >= 100]
DESIGN_RETURN_INLET_CFM = 40.0      # the task station's design flow per return inlet (real modules ~14.4)  [C]
# Flight rules of a cabin-fan stop requested by the robot (fire detection relies on the ventilation carrying
# smoke to the detectors): at most 2 stops per cabin air unit per mission, the station restarts the fan after at most
# 60 s, and at least 120 s of running between the restart of one stop and the next stop.  [C]
FAN_STOP_RULES = dict(max_stops_per_unit=2, max_stop_s=60.0, min_gap_s=120.0)
HATCH_OPENING = 1.27                # m  [V]
CO2_PA_PER_KG_M3 = 8.314 * 295.0 / 0.04401   # ideal gas at 22 C
MMHG = 133.322


def pp_mmhg(c_kg_m3) -> np.ndarray:
    """CO2 partial pressure [mmHg] of a concentration [kg/m^3] at 22 C."""
    return np.asarray(c_kg_m3, float) * CO2_PA_PER_KG_M3 / MMHG


def load_features(path) -> dict:
    """Read the feature catalogue JSON."""
    import json
    with open(path) as fh:
        return json.load(fh)


@dataclass
class FlowBox:
    """Axis-aligned flow box (world): lo, hi; module id; kind 'module' | 'vestibule'; axis index along the module."""

    name: str
    lo: np.ndarray
    hi: np.ndarray
    module: str
    kind: str
    axis: int
    builders: dict = field(default_factory=dict)   # fan name -> FlowFieldBuilder (filled during build)
    fields: dict = field(default_factory=dict)     # fan name -> GridFlowField

    def contains(self, x, pad: float = 0.0) -> np.ndarray:
        x = np.atleast_2d(x)
        return np.all((x >= self.lo - pad) & (x <= self.hi + pad), axis=1)

    def outside_distance(self, x) -> np.ndarray:
        x = np.atleast_2d(x)
        d = np.maximum(self.lo - x, 0.0) + np.maximum(x - self.hi, 0.0)
        return np.linalg.norm(d, axis=1)


def _wall_of(box: FlowBox, normal_into_cabin):
    """Walls (name, axis, side) a vent faces: the components of -normal pick the wall(s) behind it."""
    n = -np.asarray(normal_into_cabin, float)
    out = []
    for ax in range(3):
        if abs(n[ax]) > 0.3:
            out.append(("%s%s" % ("+" if n[ax] > 0 else "-", "xyz"[ax]), ax, 1 if n[ax] > 0 else 0))
    return out


class SimpleFan:
    """First-order fan state (inter-module fans): ``omega`` fraction, ``set_fan(on)``, ``advance(dt)``."""

    def __init__(self, name: str, tau_fan: float = 3.0, flow: float = INTER_MODULE_CFM * CFM):
        self.name, self.tau, self.flow0 = name, float(tau_fan), float(flow)
        self.omega, self.omega_cmd = 1.0, 1.0
        self.history = []

    def set_fan(self, on: bool, t: float | None = None) -> None:
        self.omega_cmd = 1.0 if on else 0.0
        if t is not None:
            self.history.append((float(t), "on" if on else "off"))

    def advance(self, dt: float) -> None:
        self.omega += (self.omega_cmd - self.omega) * (1.0 - math.exp(-dt / self.tau))
        if self.omega_cmd == 0.0 and self.omega < 1e-3:
            self.omega = 0.0
        if self.omega_cmd == 1.0 and self.omega > 1.0 - 1e-6:
            self.omega = 1.0

    @property
    def flow(self) -> float:
        return self.flow0 * self.omega


class StationVentilation:
    """Ventilation field of the chosen modules (see module doc).

    features: catalogue dict; modules: neutral module ids; h: grid cell [m]; return_cfm / imv_cfm: per grille / per
    hatch link -- a number, or (v0.4) a dict per cabin air unit {module: cfm per inlet} / per link {(hub, lab) or
    'hub:lab': cfm} (missing keys keep the module defaults RETURN_GRILLE_CFM / INTER_MODULE_CFM); builds: inlet build
    per grille id (default 'screen': the catalogue's grille with a 20-mesh screen) or one name; circulation: None =
    calibrate per module to ``band_target`` of the cell volume inside the band, a float (same speed for all), or a dict
    module -> speed [m/s]; fan_scale: cabin-fan curve scaling (``FanNetwork``).  inlets (v0.4): the station's return
    inlets as a list -- entries of the station scene's inlet table (``id``, ``module``, ``wall_point``, ``normal``,
    ``quat_wxyz``, ``opening_m``, ``frame_m``) or vent dicts (``pos``, ``normal_into_cabin``, ``size``, ``face_offset``);
    they replace every return grille of the catalogue (the supply diffusers stay).
    Attributes: boxes, modules, links [(hub, lab)], grilles {id: Inlet}, cabin_fans {module: FanNetwork},
    imv_fans {(hub, lab): SimpleFan}, circulation {module: V}, report; return_q_m {module: m^3/s per inlet},
    imv_q_l {(hub, lab): m^3/s}.
    """

    def __init__(self, features: dict, modules=("l2", "h2", "l1"), h: float = 0.05, return_cfm: float = RETURN_GRILLE_CFM,
                 imv_cfm: float = INTER_MODULE_CFM, builds="screen", circulation=None, band_target: float = 0.69,
                 fan_scale="design_point", tau_fan: float = 3.0, imv_intake_half=(0.15, 0.15), vent_overrides=None,
                 inlets=None):
        self.features = features
        self.h = float(h)
        mods = {m["id"]: m for m in features["modules"].values()}
        self.modules = [m for m in modules if m in mods]
        self.mod = {m: mods[m] for m in self.modules}
        if isinstance(return_cfm, dict):
            self.return_q_m = {m: float(return_cfm.get(m, RETURN_GRILLE_CFM)) * CFM for m in self.modules}
            self.return_q = float(np.mean(list(self.return_q_m.values()))) if self.return_q_m else RETURN_GRILLE_CFM * CFM
        else:
            self.return_q = return_cfm * CFM
            self.return_q_m = {m: self.return_q for m in self.modules}
        self.boxes = {}
        self._module_boxes()
        self.links = self._links()
        if isinstance(imv_cfm, dict):
            def _q(lk):
                v = imv_cfm.get(lk, imv_cfm.get(f"{lk[0]}:{lk[1]}", INTER_MODULE_CFM))
                return float(v) * CFM
            self.imv_q_l = {lk: _q(lk) for lk in self.links}
            self.imv_q = float(np.mean(list(self.imv_q_l.values()))) if self.imv_q_l else INTER_MODULE_CFM * CFM
        else:
            self.imv_q = imv_cfm * CFM
            self.imv_q_l = {lk: self.imv_q for lk in self.links}
        cat = features.get("vents", [])
        if inlets is not None:
            cat = [v for v in cat if v.get("kind") != "return_grille"] + [self._inlet_vent(i) for i in inlets]
        self.vents = [dict(v) for v in cat if v.get("module") in self.modules
                      and v.get("kind") in ("return_grille", "supply_diffuser")]
        for v in self.vents:                      # e.g. a design's own inlet geometry replacing a catalogue grille
            v.update((vent_overrides or {}).get(v["id"], {}))
        self.grilles = {}
        for v in self.vents:
            if v["kind"] != "return_grille":
                continue
            R = _quat_to_mat(v["quat_wxyz"])
            n = np.asarray(v["normal_into_cabin"], float)
            c = np.asarray(v["pos"], float) + n * float(v.get("face_offset", 0.006))   # cabin face of the grille part
            self.grilles[v["id"]] = Inlet(v["id"], c, n, R[:, 0], (v["size"][0] / 2, v["size"][1] / 2),
                                          flow=self.return_q_m[v["module"]])
        self.cabin_fans = {}
        for m in self.modules:
            inl = [g for gid, g in self.grilles.items() if self._vent(gid)["module"] == m]
            if not inl:
                continue
            b = {g.name: (builds if isinstance(builds, str) else builds.get(g.name, "screen")) for g in inl}
            self.cabin_fans[m] = FanNetwork(inl, build=b, Q_design=self.return_q_m[m] * len(inl), tau_fan=tau_fan,
                                            fan_scale=fan_scale, name=f"cabin:{m}")
        self.imv_fans = {lk: SimpleFan(f"imv:{lk[0]}:{lk[1]}", tau_fan, self.imv_q_l[lk]) for lk in self.links}
        self.imv_intake_half = imv_intake_half
        self._build_bases()
        self.circulation = {}
        self._box_list = list(self.boxes.values())
        if circulation is None:
            for m in self.modules:
                self.circulation[m] = self._calibrate(m, band_target)
        elif isinstance(circulation, dict):
            self.circulation = {m: float(circulation.get(m, 0.0)) for m in self.modules}
        else:
            self.circulation = {m: float(circulation) for m in self.modules}
        self.report = self._report()

    # ------------------------------------------------------------------------------------------- geometry
    @staticmethod
    def _inlet_vent(i: dict) -> dict:
        """A return-grille vent dict from an inlet-table entry (the station scene's format) or a vent dict."""
        if "wall_point" in i:
            fr = i.get("frame_m")
            return dict(id=i["id"], module=i["module"], kind="return_grille", pos=list(i["wall_point"]), quat_wxyz=list(i["quat_wxyz"]),
                        normal_into_cabin=list(i["normal"]), size=list(i.get("opening_m", (0.40, 0.12))),
                        face_offset=float(fr[2]) if fr is not None else float(i.get("face_offset", 0.030)), part=i["id"])
        v = dict(i)
        v["kind"] = "return_grille"
        return v

    def _vent(self, vid):
        for v in self.vents:
            if v["id"] == vid:
                return v
        raise KeyError(vid)

    def _module_boxes(self):
        for mid in self.modules:
            m = self.mod[mid]
            o, d = np.asarray(m["axis_origin"], float), np.asarray(m["axis_dir"], float)
            u, w = np.asarray(m["lat_u"], float), np.asarray(m["lat_w"], float)
            e = m["end_walls_s"]
            cu, cw = m["clear_u_m"]["median"] / 2, m["clear_w_m"]["median"] / 2
            pts = [o + d * s + u * a + w * b for s in (e["start"], e["end"]) for a in (-cu, cu) for b in (-cw, cw)]
            P = np.array(pts)
            ax = int(np.argmax(np.abs(d)))
            self.boxes[mid] = FlowBox(mid, P.min(0), P.max(0), mid, "module", ax)

    def _links(self):
        """(hub, lab) pairs joined by an open hatch in the catalogue, with vestibule boxes."""
        links = []
        hatch = self.features.get("hatches", [])
        for hz in hatch:
            a, b = hz.get("module"), hz.get("leads_to_id")
            if a in self.modules and b in self.modules and hz.get("state_in_model") == "open":
                pair = (a, b) if a.startswith("h") else (b, a)
                if pair not in links:
                    links.append(pair)
        for hub, lab in links:
            mb = self.mod[lab]
            o, d = np.asarray(mb["axis_origin"], float), np.asarray(mb["axis_dir"], float)
            e = mb["end_walls_s"]
            ends = [o + d * e["start"], o + d * e["end"]]
            hb = self.boxes[hub]
            dist = [float(hb.outside_distance(p)[0]) for p in ends]
            k = int(np.argmin(dist))
            p_end = ends[k]
            ax = int(np.argmax(np.abs(d)))
            q = np.clip(p_end, hb.lo, hb.hi)               # the hub's face point toward the lab end
            lo, hi = np.minimum(p_end, q), np.maximum(p_end, q)
            half = HATCH_OPENING / 2
            for a2 in range(3):
                if a2 != ax:
                    c = o[a2]
                    lo[a2], hi[a2] = c - half, c + half
            name = f"vest:{hub}:{lab}"
            self.boxes[name] = FlowBox(name, lo, hi, lab, "vestibule", ax)
            self.boxes[name].link = (hub, lab)
            self.boxes[name].lab_end = k                   # which end of the lab is at the hatch
        return links

    # ------------------------------------------------------------------------------------------- bases
    def _builder(self, box: FlowBox, fan: str) -> FlowFieldBuilder:
        if fan not in box.builders:
            box.builders[fan] = FlowFieldBuilder(box.lo, box.hi, self.h)
        return box.builders[fan]

    def _patch(self, box, fan, wall, ax, side, center, half_inplane, flow, inflow):
        """Add one wall patch (inflow True = into the box) clamped inside the wall face."""
        B = self._builder(box, fan)
        others = [a for a in range(3) if a != ax]
        c = np.asarray(center, float).copy()
        hs = np.asarray(half_inplane, float).copy()
        for k, a in enumerate(others):
            span = (B.hi[a] - B.lo[a]) / 2
            hs[k] = max(min(hs[k], span), 0.5 * self.h + 1e-9)
            c[a] = min(max(c[a], B.lo[a] + hs[k]), B.hi[a] - hs[k])
        c[ax] = B.hi[ax] if side == 1 else B.lo[ax]
        (B.add_inlet if inflow else B.add_outlet)(wall, c, hs, flow)

    def _vent_patches(self, box, fan, vent, flow, inflow):
        R = _quat_to_mat(vent["quat_wxyz"])
        walls = _wall_of(box, vent["normal_into_cabin"])
        long_ax = int(np.argmax(np.abs(R[:, 0])))
        for wall, ax, side in walls:
            others = [a for a in range(3) if a != ax]
            half = []
            for a in others:
                if a == long_ax:
                    half.append(vent["size"][0] / 2)
                else:
                    half.append(vent["size"][1] / 2)
            self._patch(box, fan, wall, ax, side, vent["pos"], half, flow / len(walls), inflow)

    def _hatch_face(self, box: FlowBox, vest: FlowBox):
        """(wall name, axis, side, centre) of the face of ``box`` shared with the vestibule."""
        ax = vest.axis
        c = 0.5 * (vest.lo + vest.hi)
        if abs(box.hi[ax] - vest.lo[ax]) < abs(box.lo[ax] - vest.hi[ax]):
            return ("+" + "xyz"[ax], ax, 1, c)
        return ("-" + "xyz"[ax], ax, 0, c)

    def _build_bases(self):
        half = HATCH_OPENING / 2
        for m in self.modules:
            box = self.boxes[m]
            fan = f"cabin:{m}"
            rets = [v for v in self.vents if v["module"] == m and v["kind"] == "return_grille"]
            sups = [v for v in self.vents if v["module"] == m and v["kind"] == "supply_diffuser"]
            if not rets or not sups:
                continue
            q_tot = self.return_q_m[m] * len(rets)
            for v in rets:
                self._vent_patches(box, fan, v, self.return_q_m[m], inflow=False)
            for v in sups:
                self._vent_patches(box, fan, v, q_tot / len(sups), inflow=True)
        for hub, lab in self.links:
            fan = f"imv:{hub}:{lab}"
            vest = self.boxes[f"vest:{hub}:{lab}"]
            iq = self.imv_q_l[(hub, lab)]
            sups = [v for v in self.vents if v["module"] == lab and v["kind"] == "supply_diffuser"]
            for v in sups:
                self._vent_patches(self.boxes[lab], fan, v, iq / len(sups), inflow=True)
            wl, ax, side, c = self._hatch_face(self.boxes[lab], vest)
            self._patch(self.boxes[lab], fan, wl, ax, side, c, (half, half), iq, inflow=False)
            # vestibule: in at the lab side, out at the hub side
            lab_side = 1 if abs(vest.hi[ax] - (self.boxes[lab].lo[ax])) < abs(vest.lo[ax] - self.boxes[lab].hi[ax]) else 0
            self._patch(vest, fan, ("+" if lab_side else "-") + "xyz"[ax], ax, lab_side, c, (half, half), iq, inflow=True)
            self._patch(vest, fan, ("-" if lab_side else "+") + "xyz"[ax], ax, 1 - lab_side, c, (half, half), iq, inflow=False)
            wl, ax2, side2, c2 = self._hatch_face(self.boxes[hub], vest)
            self._patch(self.boxes[hub], fan, wl, ax2, side2, c2, (half, half), iq, inflow=True)
            # intake in the hub: the catalogue's duct-pair hint, on the wall it is nearest to
            hint = None
            for v in self.features.get("vents", []):
                if v.get("kind") == "imv_duct_pair" and v.get("module") == hub and lab in v["id"]:
                    hint = np.asarray(v["pos_hint"], float)
            hb = self.boxes[hub]
            if hint is None:
                hint = c2
            q = np.clip(hint, hb.lo, hb.hi)
            dl, dh = np.abs(q - hb.lo), np.abs(hb.hi - q)
            dl[ax2] = dh[ax2] = np.inf                     # not on the hatch wall itself
            a3 = int(np.argmin(np.minimum(dl, dh)))
            s3 = 1 if dh[a3] < dl[a3] else 0
            self._patch(hb, fan, ("+" if s3 else "-") + "xyz"[a3], a3, s3, q, self.imv_intake_half, iq, inflow=False)
        self.solve_report = {}
        for name, box in self.boxes.items():
            for fan, B in box.builders.items():
                fld, rep = B.build()
                box.fields[fan] = fld
                self.solve_report[f"{name}|{fan}"] = rep

    # ------------------------------------------------------------------------------------------- field
    def circulation_velocity(self, m: str, x) -> np.ndarray:
        """Two-cell cross-sectional circulation of module m (unit cabin-fan speed) at points x inside its box."""
        box = self.boxes[m]
        V = self.circulation.get(m, 0.0)
        out = np.zeros_like(x)
        if V == 0.0:
            return out
        M = self.mod[m]
        u_ax = int(np.argmax(np.abs(M["lat_u"])))
        w_ax = int(np.argmax(np.abs(M["lat_w"])))
        W = box.hi[u_ax] - box.lo[u_ax]
        H = box.hi[w_ax] - box.lo[w_ax]
        uh = np.clip((x[:, u_ax] - box.lo[u_ax]) / W, 0, 1)
        wh = np.clip((x[:, w_ax] - box.lo[w_ax]) / H, 0, 1)
        A = -V * W / (2 * math.pi)
        out[:, u_ax] = A * (math.pi / H) * np.sin(2 * math.pi * uh) * np.cos(math.pi * wh)
        out[:, w_ax] = -A * (2 * math.pi / W) * np.cos(2 * math.pi * uh) * np.sin(math.pi * wh)
        return out

    def fan_fractions(self) -> dict:
        """{fan name: speed fraction} of all fans."""
        f = {f"cabin:{m}": net.omega for m, net in self.cabin_fans.items()}
        f.update({fan.name: fan.omega for fan in self.imv_fans.values()})
        return f

    def __call__(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        out = np.zeros_like(x)
        if len(x) == 0:
            return out
        fr = self.fan_fractions()
        dist = np.stack([b.outside_distance(x) for b in self._box_list], 1)
        which = np.argmin(dist, axis=1)
        for k, box in enumerate(self._box_list):
            sel = np.nonzero(which == k)[0]
            if len(sel) == 0:
                continue
            xs = x[sel]
            u = np.zeros_like(xs)
            for fan, fld in box.fields.items():
                w = fr.get(fan, 0.0)
                if w > 0:
                    u += w * fld(xs)
            if box.kind == "module":
                w = fr.get(f"cabin:{box.module}", 0.0)
                if w > 0:
                    u += w * self.circulation_velocity(box.module, xs)
            out[sel] = u
        return out

    # ------------------------------------------------------------------------------------------- reports
    def _cells(self, m: str, h: float | None = None):
        box = self.boxes[m]
        hh = h or max(self.h, 0.1)
        axes = [np.arange(box.lo[a] + hh / 2, box.hi[a], hh) for a in range(3)]
        G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        return G

    def band_fraction(self, m: str, V: float | None = None) -> float:
        """Volume fraction of module m's box with all fans on whose speed is inside CABIN_BAND."""
        G = self._cells(m)
        old = self.circulation.get(m, 0.0)
        if V is not None:
            self.circulation[m] = V
        sp = np.linalg.norm(self(G), axis=1)
        self.circulation[m] = old
        return float(np.mean((sp >= CABIN_BAND[0]) & (sp <= CABIN_BAND[1])))

    def _calibrate(self, m: str, target: float) -> float:
        self.circulation[m] = 0.0
        lo, hi = 0.0, 0.6
        if self.band_fraction(m, hi) < target:
            return hi
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if self.band_fraction(m, mid) < target:
                lo = mid
            else:
                hi = mid
        self.circulation[m] = hi
        return hi

    def _report(self) -> dict:
        rep = dict(modules={}, solves=self.solve_report, links=[list(l) for l in self.links])
        for m in self.modules:
            G = self._cells(m)
            sp = np.linalg.norm(self(G), axis=1)
            near = []
            for gid, g in self.grilles.items():
                if self._vent(gid)["module"] == m:
                    near.append(float(np.linalg.norm(self(g.c + 0.1 * g.n)[0])))
            rep["modules"][m] = dict(
                circulation_ms=self.circulation.get(m, 0.0), band_fraction=float(np.mean((sp >= CABIN_BAND[0]) & (sp <= CABIN_BAND[1]))),
                speed_median=float(np.median(sp)), speed_p10=float(np.percentile(sp, 10)), speed_p90=float(np.percentile(sp, 90)),
                speed_max=float(sp.max()), near_inlet_0p1m_max=max(near) if near else None,
                cabin_loop_cfm=(self.cabin_fans[m].Qd / CFM) if m in self.cabin_fans else 0.0,
                volume_m3=float(np.prod(self.boxes[m].hi - self.boxes[m].lo)))
        for k, v in self.solve_report.items():
            v["balance_rel"] = abs(v["net_m3s"]) / max(v["inflow_m3s"], 1e-12)
        return rep


class CO2Balance:
    """Two-zone CO2 balance per module (see module doc).

    vent: StationVentilation (volumes, links, fans); bulk_fraction: share of a module's volume in its bulk zone;
    residual_mix: bulk <-> port mixing with the cabin fan stopped, as a fraction of its loop flow; scrubber:
    (module, Q_scrub [m^3/s], eta) or None; c0: initial concentration [kg/m^3] everywhere.
    ``step(dt)`` integrates (explicit sub-steps, stable for the flows of a station); ``c`` {(module, zone): kg/m^3}.
    """

    def __init__(self, vent: StationVentilation, bulk_fraction: float = 0.9, residual_mix: float = 0.02,
                 scrubber=("l1", 20 * CFM, 0.8), c0: float = 2.5 * MMHG / CO2_PA_PER_KG_M3):
        self.v = vent
        self.mods = [m for m in vent.modules]
        self.Vb = {m: bulk_fraction * vent.report["modules"][m]["volume_m3"] for m in self.mods}
        self.Vp = {m: (1 - bulk_fraction) * vent.report["modules"][m]["volume_m3"] for m in self.mods}
        self.res = float(residual_mix)
        self.scrubber = scrubber
        self.c = {(m, z): float(c0) for m in self.mods for z in ("bulk", "port")}
        self.sources = {m: 0.0 for m in self.mods}
        self.removed_kg = 0.0
        self.t = 0.0

    def add_source(self, module: str, kg_s: float) -> None:
        """CO2 source in the module's bulk zone [kg/s] (none by default: the station is uncrewed)."""
        self.sources[module] = self.sources.get(module, 0.0) + float(kg_s)

    def _rates(self, c):
        v = self.v
        r = {k: 0.0 for k in c}
        for m in self.mods:
            net = v.cabin_fans.get(m)
            q_loop = net.Qd if net is not None else 0.0
            w = net.omega if net is not None else 0.0
            qmix = q_loop * (w + self.res * (1.0 - w))
            ex = qmix * (c[(m, "bulk")] - c[(m, "port")])
            r[(m, "bulk")] += -ex + self.sources.get(m, 0.0)
            r[(m, "port")] += ex
        for (hub, lab), fan in v.imv_fans.items():
            q = fan.flow
            ex = q * (c[(hub, "port")] - c[(lab, "port")])
            r[(hub, "port")] -= ex
            r[(lab, "port")] += ex
        if self.scrubber is not None and self.scrubber[0] in self.mods:
            m, qs, eta = self.scrubber
            net = v.cabin_fans.get(m)
            w = net.omega if net is not None else 0.0
            rem = eta * qs * w * c[(m, "port")]
            r[(m, "port")] -= rem
            self._rem_rate = rem
        else:
            self._rem_rate = 0.0
        return {k: r[k] / (self.Vb[k[0]] if k[1] == "bulk" else self.Vp[k[0]]) for k in r}

    def step(self, dt: float, max_sub: float = 0.5) -> None:
        """Advance dt seconds (sub-steps of at most max_sub s, forward Euler)."""
        n = max(1, int(math.ceil(dt / max_sub)))
        h = dt / n
        for _ in range(n):
            r = self._rates(self.c)
            for k in self.c:
                self.c[k] += h * r[k]
            self.removed_kg += h * self._rem_rate
            self.t += h

    def module_mean(self, m: str) -> float:
        """Volume-weighted module concentration [kg/m^3]."""
        return (self.c[(m, "bulk")] * self.Vb[m] + self.c[(m, "port")] * self.Vp[m]) / (self.Vb[m] + self.Vp[m])

    def snapshot(self) -> dict:
        """{module: dict(bulk_mmhg, port_mmhg, mean_mmhg)}."""
        return {m: dict(bulk_mmhg=float(pp_mmhg(self.c[(m, "bulk")])), port_mmhg=float(pp_mmhg(self.c[(m, "port")])),
                        mean_mmhg=float(pp_mmhg(self.module_mean(m)))) for m in self.mods}


class StationModel:
    """Ventilation + fans + CO2 of the station, advanced in simulated time.

    ``request_fan(module, on, t, which='cabin')`` stops / restarts the module's cabin fan (``which='imv'`` the
    inter-module fans of its links, ``'all'`` both) without any rule (scenario setup, graders);
    ``request_fan_stop(module, t)`` / ``request_fan_restart(module, t)`` are the robot's requests under the flight
    rules ``rules`` (default ``FAN_STOP_RULES``): a stop is refused ('refused_inhibited', 'refused_max_stops',
    'refused_min_gap', 'refused_already_stopped', 'refused_unknown_unit') or accepted ('stopped'); the station
    restarts the fan by itself ``max_stop_s`` after an accepted stop ('auto_restart' in ``fan_events``).
    ``advance(dt)`` spins the fans, applies the automatic restarts and integrates CO2; ``field`` is the velocity field
    callable; ``telemetry()`` a snapshot (fan fractions, inlet pressures, flows, open fractions, CO2, fan-stop state).
    Every request is logged in ``requests``; rule decisions and automatic restarts in ``fan_events``.
    inhibited (v0.4): {module: reason} of cabin air units whose stop requests the station refuses
    ('refused_inhibited', checked first; restart requests are unaffected); ``set_inhibited(module, reason)`` changes it.
    """

    def __init__(self, vent: StationVentilation, co2: CO2Balance | None = None, rules: dict | None = None,
                 inhibited: dict | None = None):
        self.vent = vent
        self.co2 = co2 if co2 is not None else CO2Balance(vent)
        self.requests = []
        self.t = 0.0
        self.rules = dict(FAN_STOP_RULES, **(rules or {}))
        self.fan_events = []
        self._stops = {m: [] for m in vent.cabin_fans}       # per unit: [(t_stop, t_restart or None, how)]
        self.inhibited = dict(inhibited or {})

    def set_inhibited(self, module: str, reason: str | None) -> None:
        """Inhibit the stop requests of a unit (reason) or lift the inhibition (None)."""
        if reason is None:
            self.inhibited.pop(module, None)
        else:
            self.inhibited[module] = str(reason)

    # -- the robot's fan-stop requests under the flight rules ---------------------------------------------------
    def fan_stop_state(self, module: str) -> dict:
        """Stops used / left, whether stopped, the time the fan may next be stopped, the automatic restart time."""
        st = self._stops.get(module)
        if st is None:
            return dict(known=False)
        R = self.rules
        stopped = bool(st and st[-1][1] is None)
        last_restart = max((r for _, r, _ in st if r is not None), default=None)
        nxt = None if stopped else (0.0 if last_restart is None else last_restart + R["min_gap_s"])
        out = dict(known=True, stops_used=len(st), stops_left=max(0, R["max_stops_per_unit"] - len(st)), stopped=stopped,
                   auto_restart_at=(st[-1][0] + R["max_stop_s"]) if stopped else None, next_stop_allowed_at=nxt)
        if module in self.inhibited:
            out["inhibited"] = self.inhibited[module]          # v0.4 (key present only for an inhibited unit)
        return out

    def request_fan_stop(self, module: str, t: float) -> str:
        """Robot request to stop the cabin fan of ``module`` (flight rules); returns the decision."""
        t = float(t)
        st = self._stops.get(module)
        R = self.rules
        if st is None:
            res = "refused_unknown_unit"
        elif module in self.inhibited:
            res = "refused_inhibited"
        elif st and st[-1][1] is None:
            res = "refused_already_stopped"
        elif len(st) >= R["max_stops_per_unit"]:
            res = "refused_max_stops"
        elif st and t < st[-1][1] + R["min_gap_s"] - 1e-9:
            res = "refused_min_gap"
        else:
            res = "stopped"
            st.append([t, None, "request"])
            self.request_fan(module, False, t, "cabin")
        self.fan_events.append((t, module, "stop_request", res))
        return res

    def request_fan_restart(self, module: str, t: float) -> str:
        """Robot request to restart a fan it stopped (allowed at any time); returns 'restarted' or 'refused_not_stopped'."""
        t = float(t)
        st = self._stops.get(module)
        if not st or st[-1][1] is not None:
            res = "refused_not_stopped"
        else:
            st[-1][1], st[-1][2] = t, "request"
            self.request_fan(module, True, t, "cabin")
            res = "restarted"
        self.fan_events.append((t, module, "restart_request", res))
        return res

    def _auto_restart(self, t: float) -> None:
        for m, st in self._stops.items():
            if st and st[-1][1] is None and t >= st[-1][0] + self.rules["max_stop_s"] - 1e-9:
                st[-1][1], st[-1][2] = t, "auto"
                self.request_fan(m, True, t, "cabin")
                self.fan_events.append((t, m, "auto_restart", "restarted"))

    @property
    def field(self):
        return self.vent

    def request_fan(self, module: str, on: bool, t: float, which: str = "cabin") -> None:
        """Stop / restart fans of a module (logged)."""
        if which in ("cabin", "all") and module in self.vent.cabin_fans:
            self.vent.cabin_fans[module].set_fan(on, t)
        if which in ("imv", "all"):
            for lk, fan in self.vent.imv_fans.items():
                if module in lk:
                    fan.set_fan(on, t)
        self.requests.append((float(t), module, bool(on), which))

    def advance(self, dt: float, solve: bool = True) -> None:
        """Apply the automatic restarts, spin the fans by dt, re-solve the cabin-fan networks, integrate CO2."""
        self._auto_restart(self.t)
        for net in self.vent.cabin_fans.values():
            net.advance(dt)
            if solve:
                net.solve()
        for fan in self.vent.imv_fans.values():
            fan.advance(dt)
        self.co2.step(dt)
        self.t += dt

    def telemetry(self) -> dict:
        """Snapshot of fans, inlets and CO2 (mechanics level; what a design exposes to an agent is its choice)."""
        v = self.vent
        out = dict(t=self.t, fans=v.fan_fractions(), co2=self.co2.snapshot(), inlets={},
                   fan_stops={m: self.fan_stop_state(m) for m in self._stops})
        for m, net in v.cabin_fans.items():
            out[f"Pc:{m}"] = net.Pc
            out[f"Q:{m}"] = net.Q
            for inl in net.inlets:
                out["inlets"][inl.name] = dict(flow=inl.flow, open_fraction=net.open_fraction(inl.name))
        return out


def _quat_to_mat(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
