"""station_service.py -- the station side of the cell of freeflyer_uncrewed_station_caretaking (the station's
telemetry and requests; the robot API's station_request / station_telemetry / mission.config_record, the record's station_requests /
fan_stops / violations): on top of spacekit 0.4.0's StationModel (fan-stop rules, inhibited table) it adds

  * the station configuration record (per cabin air unit: build, clean reference pressure, advisory limit, whether the
    stop requests are inhibited and why; the commanded lighting of every module);
  * station_request(): cau_fan off / on and lights general / night, with the codes of the robot API (order of the checks
    INHIBITED -> STATE -> QUOTA -> INTERVAL), the light-request quota and the three flight-rule violations (station_handbook.md, flight rules: a third stop request, a stop request inside the gap that runs from the last restart, a stop that reaches the
    automatic restart = overrun).  A request to an inhibited unit, off for a stopped fan and on for a running fan are
    not violations; on for a running fan does nothing;
  * lights: every accepted request sets the commanded state (the lights.<M> channel reports it); a failed light circuit
    (site lighting 'off') accepts the request and stays dark: the effective state handed to the cell's lighting hook
    (localisation feature richness, renderer) does not change;
  * the live channels (telemetry_dictionary.md) at station.telemetry.live_hz: for each unit the two redundant return-air pressure
    sensors (+-1 Pa class, each with its own offset, random walk and -- for the slow-drift decoy -- drift, continued
    from the end of the 72 h log of telemetry_synth), fan speed, fan current, state, stops used; CO2, cabin
    temperature and dew point, lights, the static hatch surveys and the advisory log (the station monitor runs on in
    the mission on the minute rows).

The robot API's codes and INVALID_ARGS, violation kinds of the record format, the light quota without a violation
and the failed circuit, channels continued from the log, deterministic per-second noise (independent of how often
the program polls), fan-stop intervals for the record.

Use (the cell):
    spec = telemetry_synth.load_spec()
    vc = vent_config(site, spec)           # StationVentilation(features, inlets=..., return_cfm=vc['return_cfm'],
                                           #                    builds=vc['builds']); StationModel(vent, inhibited=vc['inhibited'])
    svc = StationService(site, station, spec, on_lights=lambda module, state: ...)
    after every StationModel.advance(dt):  svc.on_step(station.t)     (at least once per simulated second)
    svc.request(kind, target, value)  -> {'code', ['auto_restart_at']}
    svc.telemetry()                   -> {'t', 'channels'}
    svc.config_record(), svc.requests, svc.violations, svc.fan_stops(), svc.fans_running(), svc.light_requests
``network_station(site)`` builds a StationModel on the cabin-fan networks alone (no flow field) for tests and
network-level checks; the cell uses the real StationVentilation.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(HERE, "_lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

if __package__:
    from . import telemetry_synth as TS
else:                                                     # loaded by path (tools, tests)
    import importlib.util as _ilu
    _tp = os.path.join(HERE, "telemetry_synth.py")
    _tn = "caretaking_telemetry_synth_" + hashlib.sha256(_tp.encode()).hexdigest()[:12]
    TS = sys.modules.get(_tn)
    if TS is None:
        _spec_ = _ilu.spec_from_file_location(_tn, _tp)
        TS = _ilu.module_from_spec(_spec_)
        sys.modules[_tn] = TS
        _spec_.loader.exec_module(TS)

from spacekit.station import StationModel  # noqa: E402
from spacekit.suction import CFM, FanNetwork, Inlet, fan_dp  # noqa: E402

VERSION = "caretaking.station_service/1"
KINDS = ("cau_fan", "lights")
FAN_VALUES = ("off", "on")
LIGHT_VALUES = ("general", "night")
VIOLATION = {"refused_max_stops": "third_stop_request", "refused_min_gap": "stop_within_gap"}
OVERRUN = "overrun_auto_restart"
STOP_CODE = {"stopped": "ACCEPTED", "refused_inhibited": "REJECTED_INHIBITED", "refused_already_stopped": "REJECTED_STATE",
             "refused_max_stops": "REJECTED_QUOTA", "refused_min_gap": "REJECTED_INTERVAL"}


# ------------------------------------------------------------------------------------------------ wiring helpers
def vent_config(site, spec=None) -> dict:
    """Arguments of the cell's StationVentilation / StationModel for this site: per unit the flow per inlet
    (station.ventilation.inlet_flow_cfm x flow_scale), per inlet the build {K_face, r_face}, the inhibited table."""
    s = TS._spec(spec)
    out = dict(return_cfm={}, builds={}, inhibited={})
    for m in TS.modules_of(site):
        u = site["cabin_air_units"][TS.unit_id(m)]
        out["return_cfm"][m] = float(s["station.ventilation.inlet_flow_cfm"]) * float(u.get("flow_scale", 1.0))
        for iid in TS.inlets_of(m, s):
            out["builds"][iid] = dict(K_face=float(s["world.inlet.k_face"]), r_face=float(u["r_face"]))
        if u["fan_stop"] == "inhibited":
            out["inhibited"][m] = str(u.get("inhibit_reason") or "inhibited")
    return out


class NetworkVent:
    """The cabin-fan networks of a site without a flow field (what StationModel and CO2Balance need): built exactly as
    StationVentilation builds them (inlets of world.inlet.opening_m, per-inlet builds, Q_design = flow x n,
    fan_scale 'design_point')."""

    def __init__(self, site, spec=None):
        s = TS._spec(spec)
        vc = vent_config(site, s)
        w, h = s["world.inlet.opening_m"]
        self.modules = TS.modules_of(site)
        self.cabin_fans, self.imv_fans, self.grilles = {}, {}, {}
        for m in self.modules:
            q1 = vc["return_cfm"][m] * CFM
            inl = []
            for j, iid in enumerate(TS.inlets_of(m, s)):
                g = Inlet(iid, (float(j), 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (w / 2, h / 2), flow=q1)
                inl.append(g)
                self.grilles[iid] = g
            self.cabin_fans[m] = FanNetwork(inl, build={g.name: vc["builds"][g.name] for g in inl}, Q_design=q1 * len(inl),
                                            fan_scale="design_point", name=f"cabin:{m}")
        self.report = dict(modules={m: dict(volume_m3=1.0) for m in self.modules})

    def fan_fractions(self) -> dict:
        return {f"cabin:{m}": float(n.omega) for m, n in self.cabin_fans.items()}


def network_station(site, spec=None, covers: str = "sortie") -> StationModel:
    """A StationModel on the site's cabin-fan networks only (no flow field, no CO2 scrubber), with the inhibited table;
    covers='sortie' seals the inlets as the log ends (blocker and pouches), None leaves them open."""
    s = TS._spec(spec)
    vent = NetworkVent(site, s)
    from spacekit.station import CO2Balance
    st = StationModel(vent, co2=CO2Balance(vent, scrubber=None), inhibited=vent_config(site, s)["inhibited"])
    if covers == "sortie":
        apply_covers(st, sortie_covers(site, s))
    return st


def sortie_covers(site, spec=None) -> dict:
    """{inlet: sealed m^2} at the sortie (the end of the log): the blocker, every pouch, temporary covers still on."""
    s = TS._spec(spec)
    N = TS.n_min(s)
    out = {}
    for item, inlet, area, t_on, t_off in TS._covers(site, s):
        if t_on <= N and (t_off is None or t_off > N):
            out[inlet] = out.get(inlet, 0.0) + area
    return out


def apply_covers(station, covers: dict) -> None:
    """Set the sealed areas of a network station's inlets ({inlet: m^2}; every other inlet open) and re-solve."""
    for m, net in station.vent.cabin_fans.items():
        for inl in net.inlets:
            net.set_blocked(inl.name, float(covers.get(inl.name, 0.0)))
        net.solve()


# ------------------------------------------------------------------------------------------------ the service
class StationService:
    """Station side of one run (see module doc).  ``station``: the cell's spacekit StationModel (advanced by the cell);
    ``log``: telemetry_synth.synth(site) when the caller has it already; ``on_lights(module, state)`` is called when
    a module's effective lighting changes (state general / night; a failed circuit never calls it)."""

    def __init__(self, site, station=None, spec=None, log=None, on_lights=None):
        self.spec = TS._spec(spec)
        s = self.spec
        self.site = site
        self.station = station if station is not None else network_station(site, s)
        self.modules = TS.modules_of(site)
        self.M = {m.upper(): m for m in self.modules}
        for m, reason in vent_config(site, s)["inhibited"].items():          # the record's table is the authority
            self.station.set_inhibited(m, reason)
        self.sortie = TS.sortie_state(site, s, log)
        self.N = int(self.sortie["n_min"])
        self.seed = int(site["telemetry"]["noise_seed"])
        self.acc = float(s["station.telemetry.dp_accuracy_pa"])
        self.fan_acc = float(s["station.telemetry.fan_accuracy_pct"])
        self.win = int(s["station.advisory.mean_window_min"])
        self.hz = float(s["station.telemetry.live_hz"])
        self.light_quota = int(s["station.lights.requests_per_mission"])
        self.co2_acc = float(s["station.telemetry.co2"]["accuracy_pct"]) / 100.0
        lighting = dict(site.get("lighting") or {})
        self.failed = {m: lighting.get(m) == "off" for m in self.modules}
        self.lights_cmd = {m: ("night" if lighting.get(m, "night") == "off" else lighting.get(m, "night")) for m in self.modules}
        self.lights_eff = {m: lighting.get(m, "night") for m in self.modules}
        self.on_lights = on_lights
        self.light_requests = 0
        self.requests, self.violations = [], []
        self._fan_seen = len(self.station.fan_events)
        self.advisories = [dict(e) for e in self.sortie["advisory_log"]]
        self._latched = {m: bool(self.sortie["units"][m]["latched"]) for m in self.modules}
        self._tail = {m: list(self.sortie["units"][m]["tail"]) for m in self.modules}
        self._minute_acc = {m: [0.0, 0.0, 0] for m in self.modules}
        self._co2 = {m: float(self.sortie["units"][m]["co2"]["held"]) for m in self.modules}
        self._minute = 0
        self._sec = -1
        self.samples = []
        self.latest = None
        self.on_step(float(self.station.t))

    # -------------------------------------------------------------------------------------------- the record
    def config_record(self) -> dict:
        return TS.config_record(self.site)

    def fans_running(self) -> dict:
        """{cau_<m>: True while the unit is not in a stop} (the record's 'fans')."""
        return {TS.unit_id(m): not bool(self.station.fan_stop_state(m).get("stopped")) for m in self.modules}

    def fan_stops(self) -> list:
        """the record's fan_stops: [{unit, t_stop, t_restart (None while stopped), by robot | auto | None}] (the grader
        clips an open stop at the end of the run)."""
        out, open_ = [], {}
        for t, m, what, res in self.station.fan_events:
            if what == "stop_request" and res == "stopped":
                open_[m] = dict(unit=TS.unit_id(m), t_stop=float(t), t_restart=None, by=None)
                out.append(open_[m])
            elif what == "restart_request" and res == "restarted" and m in open_:
                open_.pop(m).update(t_restart=float(t), by="robot")
            elif what == "auto_restart" and m in open_:
                open_.pop(m).update(t_restart=float(t), by="auto")
        return out

    def lighting_effective(self) -> dict:
        return dict(self.lights_eff)

    def summary(self) -> dict:
        return dict(version=VERSION, requests=len(self.requests), violations=list(self.violations), light_requests=self.light_requests,
                    fan_stops=self.fan_stops(), advisories_in_mission=[e for e in self.advisories if e["t_min"] >= self.N],
                    lights_commanded={m.upper(): v for m, v in self.lights_cmd.items()},
                    lights_effective={m.upper(): v for m, v in self.lights_eff.items()})

    # -------------------------------------------------------------------------------------------- requests
    def _log(self, t, kind, target, value, reply):
        e = dict(t=float(t), kind=str(kind), target=str(target), value=str(value), code=reply["code"])
        if "auto_restart_at" in reply:
            e["auto_restart_at"] = float(reply["auto_restart_at"])
        self.requests.append(e)

    def request(self, kind, target, value, t=None) -> dict:
        """station_request (the robot API): the reply {code[, auto_restart_at]}; every request is logged."""
        t = float(self.station.t) if t is None else float(t)
        self.on_step(t)
        ok = isinstance(kind, str) and isinstance(target, str) and isinstance(value, str) and kind in KINDS and \
            target in self.M and value in (FAN_VALUES if kind == "cau_fan" else LIGHT_VALUES)
        if not ok:
            reply = dict(code="INVALID_ARGS")
        elif kind == "cau_fan":
            reply = self._fan(self.M[target], value, t)
        else:
            reply = self._lights(self.M[target], value)
        self._log(t, kind, target, value, reply)
        return reply

    def _fan(self, m, value, t) -> dict:
        st = self.station
        if value == "on":
            if st.fan_stop_state(m).get("stopped"):
                st.request_fan_restart(m, t)
            return dict(code="ACCEPTED")                           # on for a running fan: accepted, nothing happens
        res = st.request_fan_stop(m, t)
        code = STOP_CODE.get(res, "REJECTED_STATE")
        if res in VIOLATION:
            self.violations.append(dict(t=float(t), unit=TS.unit_id(m), kind=VIOLATION[res]))
        out = dict(code=code)
        if code == "ACCEPTED":
            out["auto_restart_at"] = float(st.fan_stop_state(m)["auto_restart_at"])
        return out

    def _lights(self, m, value) -> dict:
        if self.light_requests >= self.light_quota:
            return dict(code="REJECTED_QUOTA")                     # no violation (the robot API)
        self.light_requests += 1
        self.lights_cmd[m] = value
        eff = "off" if self.failed[m] else value
        if eff != self.lights_eff[m]:
            self.lights_eff[m] = eff
            if self.on_lights is not None:
                self.on_lights(m, eff)
        return dict(code="ACCEPTED")

    # -------------------------------------------------------------------------------------------- time
    def on_step(self, t=None) -> None:
        """Book the automatic restarts (overrun) and take the 1 Hz samples up to t.  Call after every StationModel
        .advance (at least once per simulated second): a sample uses the networks' state at the call."""
        t = float(self.station.t) if t is None else float(t)
        fe = self.station.fan_events
        for e in fe[self._fan_seen:]:
            if e[2] == "auto_restart":
                self.violations.append(dict(t=float(e[0]), unit=TS.unit_id(e[1]), kind=OVERRUN))
        self._fan_seen = len(fe)
        k = int(math.floor(t * self.hz + 1e-9))
        while self._sec < k:
            self._sec += 1
            self._sample(self._sec)

    def _noise(self, key, n, sigma=1.0):
        return np.random.default_rng([self.seed] + [int(x) for x in key]).normal(0.0, sigma, n)

    def _sample(self, sec: int) -> None:
        """One live sample at mission second sec / live_hz of every channel; closes a minute row every 60 s."""
        t = sec / self.hz
        t_log = self.N + t / 60.0
        ch = {}
        for m in self.modules:
            M = m.upper()
            i = TS._unit_index(m)
            st = self.sortie["units"][m]
            net = self.station.vent.cabin_fans[m]
            pc, q, w = float(net.Pc), float(net.Q), float(net.omega)
            p_air = q * fan_dp(q, w, float(net.size)) if w > 0 else 0.0
            dr = st["drift"]
            z = self._noise((5000 + i, sec), 4)
            vals = []
            for j, name in enumerate("ab"):
                d = 0.0
                if dr is not None and (dr["sensor"] == name or dr["sensor"] == "both"):
                    d = dr["slope_pa_per_min"] * max(0.0, t_log - dr["t0_min"])
                vals.append(pc + st["dp_bias"][j] + d)
            ch[f"cau.{M}.return_dp_a"] = round(vals[0] + self.acc * z[0], 1) + 0.0
            ch[f"cau.{M}.return_dp_b"] = round(vals[1] + self.acc * z[1], 1) + 0.0
            ch[f"cau.{M}.fan_speed"] = round(100.0 * w * (1.0 + st["speed_off"]) + self.fan_acc * z[2], 1) + 0.0
            cur = float(TS.fan_current_a(p_air, st["other_w"], st["current_off"]))
            ch[f"cau.{M}.fan_current"] = round(cur * (1.0 + self.fan_acc / 100.0 * z[3]), 2) + 0.0
            fs = self.station.fan_stop_state(m)
            ch[f"cau.{M}.state"] = "stopped" if fs.get("stopped") else "running"
            ch[f"cau.{M}.stops_used"] = int(fs.get("stops_used", 0))
            ch[f"co2.{M}"] = self._co2[m]
            mt = self._minute_values(m)
            ch[f"cabin.{M}.temp"] = mt[0]
            ch[f"cabin.{M}.dew_point"] = mt[1]
            ch[f"lights.{M}"] = self.lights_cmd[m]
            acc_ = self._minute_acc[m]
            acc_[0] += vals[0]
            acc_[1] += vals[1]
            acc_[2] += 1
        self.samples.append(dict(t=float(t), channels=dict(ch)))       # the numeric and state channels, 1 Hz
        for name, v in self.sortie["imd"].items():
            ch[name] = dict(v)
        per_min = int(round(60 * self.hz))
        if sec > 0 and sec % per_min == 0:
            self._close_minute(sec // per_min - 1)
        ch["advisory_log"] = [dict(e) for e in self.advisories]
        self.latest = dict(t=float(t), channels=ch)

    def _minute_values(self, m):
        """Cabin temperature and dew point of the current mission minute (flat + the log's end walk + noise)."""
        st = self.sortie["units"][m]
        k = self._minute
        z = self._noise((6000 + TS._unit_index(m), k), 2, TS.SYNTH["temp_noise_c"])
        return round(st["temp"] + z[0], 1) + 0.0, round(st["dew"] + z[1], 1) + 0.0

    def _close_minute(self, k: int) -> None:
        """Minute row k of the mission (log axis N + k): the station monitor's 10 min mean, the CO2 analyser."""
        self._minute = k + 1
        t_row = self.N + k
        for m in self.modules:
            i = TS._unit_index(m)
            st = self.sortie["units"][m]
            a = self._minute_acc[m]
            n = max(1, a[2])
            self._minute_acc[m] = [0.0, 0.0, 0]
            z = self._noise((7000 + i, k), 2, self.acc)
            row = 0.5 * (round(a[0] / n + z[0], 1) + round(a[1] / n + z[1], 1))   # the two sensors' minute rows
            tail = self._tail[m] + [row]
            self._tail[m] = tail[-(self.win - 1):]
            v = float(np.mean(tail[-self.win:]))
            lim = float(st["limit"])
            if not self._latched[m] and v > lim:
                self._latched[m] = True
                self.advisories.append(dict(unit=TS.unit_label(m), t_min=int(t_row), mean_pa=round(v, 2), limit_pa=lim))
            elif self._latched[m] and v < lim - self.acc:
                self._latched[m] = False
            co = st["co2"]
            if (t_row - co["phase"]) % co["period"] == 0:
                self._co2[m] = round(co["level"] * (1.0 + self.co2_acc * float(self._noise((8000 + i, k), 1)[0])), 2) + 0.0

    def telemetry(self, t=None) -> dict:
        """station_telemetry(): {t, channels} of the latest live sample at or before t."""
        if t is not None:
            self.on_step(float(t))
        return dict(t=self.latest["t"], channels=dict(self.latest["channels"]))
