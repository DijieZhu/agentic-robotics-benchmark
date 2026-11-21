"""Station wiring of a built site (no robot): ventilation through the return inlets with the site's build and
resistance per cabin air unit, fan-stop flight rules with the site's inhibition, suction of the items on the return
inlets and on the stowage units' faces (one SuctionFaces, the cabin air units' and the units' networks), item drag in
the cabin air plus the units' own air, rotating-station loads, network updates at 100 Hz, suction and item aero at
50 Hz zero-order hold or suction every physics step.

Station library v0.3: spacekit 0.4.0's ``StationVentilation(inlets=..., return_cfm={module: cfm},
imv_cfm={link: cfm})`` and ``StationModel(inhibited=...)`` carry the ventilation and the inhibition; the stowage
units' networks join the same SuctionFaces (occlusion on).  The components (``station_components``) are what the cell wires around the robot.
"""
from __future__ import annotations

import time

import numpy as np

from . import data
from . import inlets as IN
from . import stowage as SW
from .util import state_hash

ITEM_GRID_SHEET = (2, 6)
ITEM_GRID_POUCH = (3, 2)


class NullSuction:
    """Stand-in for SuctionFaces when a site has no loose item (spacekit's SuctionFaces needs at least one): no wrench,
    the networks keep no sealed area."""

    def __init__(self, networks):
        self.nets = list(networks)
        self.covered, self.item_force, self.shadowed = {}, {}, {}

    def accumulate(self, model, data_, W) -> None:
        return None

    def wrench(self, model, data_):
        return np.zeros((model.nbody, 6))

    def update_networks(self) -> None:
        for net in self.nets:
            net.solve()

    def hold_force(self, bodies=None) -> float:
        return 0.0


class SiteRig:
    """Handles of a wired site (attributes set by ``make_site_rig`` / ``station_components``)."""

    def step(self, n: int) -> None:
        self.sim.step(n)

    def hold(self, group: str) -> float:
        return float(self.suction.hold_force(self.groups[group]))


def ventilation(site: dict, feats: dict, h: float = 0.05):
    """StationVentilation of a site: v0.2 inlets, 40 cfm per inlet (or the site's ``flow_cfm_per_inlet`` per unit),
    the site's build and r_face per unit, inter-module flow per link."""
    from spacekit import station as ST
    mods = list(feats["modules"])
    units = site["cabin_air_units"]
    builds, ret = {}, {}
    for i in feats["v02"]["inlets"]:
        u = units[i["cab_air_unit"]]
        B = IN.BUILDS[u["build"]]
        builds[i["id"]] = dict(K_face=B["K_face"], r_face=float(u.get("r_face", B["r_face"])))
        ret[i["module"]] = float(u.get("flow_cfm_per_inlet", IN.DESIGN_FLOW_CFM))
    imv = {}
    for k, v in (site.get("inter_module") or {}).items():
        _, hub, lab = k.split("_", 2)
        imv[f"{hub}:{lab}"] = float(v.get("flow_cfm", ST.INTER_MODULE_CFM))
    return ST.StationVentilation(feats, modules=tuple(mods), h=h, return_cfm=ret, imv_cfm=imv or ST.INTER_MODULE_CFM,
                                 builds=builds, inlets=feats["v02"]["inlets"])


def station_components(model, data_, info, h: float = 0.05) -> SiteRig:
    """Ventilation, station model (inhibited units refuse stops), unit networks, SuctionFaces on every item, the
    combined air field (cabin + units), item shapes, groups.  No simulator, no aero object (the caller adds its jets)."""
    import mujoco
    from spacekit import aero as AE
    from spacekit import station as ST
    from spacekit import suction as SU

    site = info["site"]
    feats = data.features(info["scene"])
    R = SiteRig()
    R.model, R.data, R.info, R.site, R.feats = model, data_, info, site, feats
    R.vent = ventilation(site, feats, h)
    R.co2 = ST.CO2Balance(R.vent)
    inh = {k[4:]: (u.get("inhibit_reason") or "inhibited") for k, u in site["cabin_air_units"].items() if u.get("fan_stop") == "inhibited"}
    R.station = ST.StationModel(R.vent, R.co2, inhibited=inh)
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    items, shapes, groups, g_inlet = [], [], {}, {}
    if info.get("blocker"):
        bb = [bid(n) for n in info["blocker"]["bodies"]]
        groups["blocker"], g_inlet["blocker"] = bb, info["blocker"]["inlet"]
        items += [(b, info["blocker"]["half"], ITEM_GRID_SHEET) for b in bb]
        shapes += [AE.ItemShape(b, "box", info["blocker"]["half"], grid=(1, 3, 1)) for b in bb]
    for s in info.get("small_items", []):
        b = bid(s["body"])
        groups[s["id"]], g_inlet[s["id"]] = [b], s["inlet"]
        items.append((b, s["half"], ITEM_GRID_POUCH))
        shapes.append(AE.ItemShape(b, "box", s["half"], grid=(2, 2, 1)))
    for s in info.get("free_items", []):
        bb = [bid(n) for n in s["bodies"]]
        groups[s["id"]] = bb
        for b in bb:
            items.append((b, s["half"], tuple(s.get("grid", ITEM_GRID_SHEET))))
            shapes.append(AE.ItemShape(b, "box", s["half"], grid=tuple(s.get("aero_grid", (1, 3, 1)))))
    R.groups, R.item_inlet = groups, g_inlet
    R.units = []
    for U in info.get("units", []):
        N = U["part"].network()
        R.units.append(dict(U, network=N, net=N.net))
    nets = list(R.vent.cabin_fans.values()) + [u["net"] for u in R.units]
    R.suction = SU.SuctionFaces(model, items, nets) if items else NullSuction(nets)
    R.flow = SW.UnitFlow(R.vent, [u["network"] for u in R.units])
    R.shapes = shapes
    R.shelters = AE.Shelters([])
    return R


def make_site_rig(model, data_, info, suction_hz: int = 50, h: float = 0.05, slow_every: int = 10, frame: bool = True,
                  units_on: bool = True) -> SiteRig:
    """Robot-free wiring: ``station_components`` + item aero (cabin + unit air) + rotating-station loads + a Simulator
    with the networks advanced every 5 physics steps.  suction_hz: 50 (slow source, zero-order hold) or 500."""
    import mujoco
    from spacekit import aero as AE
    from spacekit.frames import RotatingStationFrame
    from spacekit.stepping import Simulator

    if suction_hz not in (50, 500):
        raise ValueError("suction_hz must be 50 or 500")
    R = station_components(model, data_, info, h)
    if not units_on:
        for u in R.units:
            u["net"].set_fan(False)
            u["net"].omega = 0.0
            u["net"]._key = None
            u["net"].solve()
    R.item_aero = AE.ItemAero(model, R.shapes, flow=R.flow, jets=None, shelters=R.shelters)
    R.frame = RotatingStationFrame(model) if frame else None
    fast = [R.suction] if suction_hz == 500 else []
    slow = [R.item_aero] + ([R.frame] if R.frame is not None else []) + ([] if suction_hz == 500 else [R.suction])
    R.sim = Simulator(model, data_, sources=fast, slow_sources=slow, slow_every=slow_every)
    dt5 = 5 * model.opt.timestep

    def net(sim):
        R.station.advance(dt5, solve=False)
        for u in R.units:
            if u["net"].omega != u["net"].omega_cmd:
                u["net"].advance(dt5)
        R.suction.update_networks()
    R.sim.add_periodic(net, dt5, "network")
    mujoco.mj_forward(model, data_)
    R.suction.wrench(model, data_)
    R.suction.update_networks()
    R.suction.wrench(model, data_)
    return R


def unit_of(R: SiteRig, module: str) -> dict:
    return next(u for u in R.units if u["module"] == module)


def inlet_flow_report(R: SiteRig, distances=(0.0, 0.05, 0.10, 0.20, 0.30)) -> dict:
    """Per return inlet: network state and the cabin field's speed on the axis in front of its face."""
    out = {}
    for m, net in R.vent.cabin_fans.items():
        for inl in net.inlets:
            prof = [float(np.linalg.norm(R.vent(inl.c + inl.n * max(s, 1e-3))[0])) for s in distances]
            out[inl.name] = dict(module=m, flow_cfm=inl.flow / IN.CFM, open_fraction=net.open_fraction(inl.name), Pc_Pa=net.Pc,
                                 speed_on_axis_ms=dict(zip([f"{s:.2f}" for s in distances], prof)))
    return out


def run_free(R: SiteRig, seconds: float, sample_s: float = 1.0, sample=None) -> dict:
    """Step a robot-free rig for ``seconds``; ``sample(R)`` is called every ``sample_s``.  Returns timing."""
    n_per = int(round(sample_s / R.model.opt.timestep))
    n_tot = int(round(seconds / R.model.opt.timestep))
    c0, w0 = time.process_time(), time.time()
    k = 0
    while k < n_tot:
        n = min(n_per, n_tot - k)
        R.sim.step(n)
        k += n
        if sample is not None:
            sample(R)
    return dict(steps=n_tot, cpu_s=time.process_time() - c0, wall_s=time.time() - w0, sim_s=float(R.data.time),
                final_state_sha256=state_hash(R.data))
