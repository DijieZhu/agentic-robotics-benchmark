"""The hidden world: one session flown by a node: the start hold, the check-out, the blocks of jobs and the time
between them, the supervisor and its hand-backs, the sensors, the safety ends and the records the grader reads.




























































































































"""
import copy
import hashlib
import json
import math
import os
import pickle
import re

import numpy as np

from . import rot, traj
from .config import V, params, torque_gains, force_gains, DATA, path, key_bytes, SIGNED, optional_path
from .plant import Plant, static_response
from .sensors import InertialUnit, FixProcess, AccelWander, stream, keyed_seed
from .stocklaw import BatchController
from . import budget as _budget

MODES = ("nominal", "quiet", "precision")
# the changes that Settings.compat may switch off
CHANGES = ("cap_reach", "gross_whole_error", "force_gain", "mass_as_prepared",
           "watchdog_from_control", "plan_verdict_keys", "block_end_at_budget",       # these three:
           "checkout_cap", "frame_time_once")                                          # these two:

VERDICTS = ("valid", "refused", "invalid", "protocol_error")
_TABLES = None
_ER_MEMO = {}          # effective_response(World) per digest of its inputs (one process)


def tables():
    global _TABLES
    if _TABLES is None:
        z = np.load(os.path.join(DATA, "sensor_tables.npz"), allow_pickle=False)
        _TABLES = {k: z[k] for k in z.files}
    return _TABLES


SCENARIO_FORMAT = "estimator-controller-scenario/1"
# the paper's key for each field of sessions.paper_fields that a scenario keeps apart, with the field's name
# as the scenario holds it (each name is the start of one entry of sessions.paper_fields)
PAPER_APART = (("release", "software release in force"), ("zone_file", "zone file in force"),
               ("battery_slots", "battery slots occupied"), ("equipment", "equipment"),
               ("time_window", "the session's time window"))


def load_session(name):
    """a development session's file when there is one; else the hidden set's scenario with the generated inputs."""
    p = os.path.join(path("sessions_dir"), name + ".json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    sdir = optional_path("scenarios_dir") or path("sessions_dir")
    sp = os.path.join(sdir, name + ".scenario")
    if not os.path.exists(sp):
        raise FileNotFoundError("no session file %s.json and no scenario %s.scenario" % (name, name))
    return scenario_session(sp, name)


def scenario_session(scenario_path, name):
    """a session in the world's form from a scenario file (read only, as generated) and the session's generated inputs."""
    with open(scenario_path, "rb") as f:
        raw = f.read()
    sc = json.loads(raw.decode("utf-8"))
    if sc.get("format") != SCENARIO_FORMAT:
        raise ValueError("%s: not a scenario of the format %s" % (scenario_path, SCENARIO_FORMAT))
    ws = copy.deepcopy(sc["world_session"])
    if sc.get("session") != name or ws.get("session") != name:
        raise ValueError("%s: the scenario is of session %r, not %r" % (scenario_path, ws.get("session"), name))
    for k in ("start_pose", "blocks_1_2", "block_3"):
        if k not in ws:
            raise ValueError("%s: world_session has no %r" % (scenario_path, k))
    if "inputs" in ws:
        raise ValueError("%s: a scenario holds no world inputs" % scenario_path)
    with open(path("session_inputs")) as f:
        rows = json.load(f)["sessions"]
    if name not in rows:
        raise KeyError("the acceptance sessions' inputs hold no row for %s" % name)
    inp = copy.deepcopy(rows[name])
    apart = sc.get("paper_fields_the_world_does_not_hand_out")
    if not isinstance(apart, dict) or sorted(apart) != sorted(n for _, n in PAPER_APART):
        raise ValueError("%s: the fields kept apart for the paper are %s" % (scenario_path, sorted(apart or {})))
    names = V("sessions.paper_fields")
    for _, n in PAPER_APART:
        assert sum(1 for f in names if f == n or f.startswith(n + ":") or f.startswith(n + " ")) == 1, n
    ws["day"] = inp["day"]
    ws["inputs"] = inp
    ws["paper_apart"] = {k: copy.deepcopy(apart[n]) for k, n in PAPER_APART}
    ws["scenario"] = {"file": os.path.basename(scenario_path), "sha256": hashlib.sha256(raw).hexdigest(),
                      "tight_mode": sc.get("tight_mode"), "list": sc.get("list")}
    return ws


def cap_reach_min_s():
    """the least reach of the cap before a scored block's start, read from the text of sessions.outage_cap_rule."""
    txt = V("sessions.outage_cap_rule")
    m = re.search(r"the cap is applied from one cap length, and at least ([0-9.]+) s, before a scored block's start", txt)
    assert m, "sessions.outage_cap_rule no longer states the reach before a block's start"
    assert "a cap of 0 s means no failed attempt: every attempt whose image time lies inside the block, and the last attempt before its start, succeeds" in txt
    return float(m.group(1))


def provisional_or_value(name):
    """(value, provisional?) of a world parameter: its value once set, else its named provisional value."""
    e = params()["parameters"][name]
    if e.get("value") is not None:
        return e["value"], False
    return e["provisional"], True


def wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def mode_limits(mode):
    sp = V("plan.speed_limit." + mode)
    ac = V("plan.acceleration_limit")[mode]
    return (sp["linear_m_s"], ac["linear_m_s2"], sp["angular_rad_s"], ac["angular_rad_s2"])


def plan_rules():
    st = V("plan.start_tolerance")
    et = V("plan.end_tolerance")["leg, slew"]
    txt = V("plan.consistency")
    assert "at most 1.0 s apart" in txt and "within 2 mm and 0.2 degrees" in txt and "2 mm/s and 0.2 deg/s" in txt
    return {"max_setpoints": int(V("plan.max_setpoints")), "max_gap_s": 1.0,
            "consistency": ((0.002, 0.002), (math.radians(0.2), math.radians(0.2))),
            "start_tol": (st["position_m"], math.radians(st["attitude_deg"])),
            "end_tol": (et["position_m"], math.radians(et["attitude_deg"]))}


def body_goal(job, mount):
    if job["kind"] == "view" or job.get("request_shape") == "view":
        g = job["goal"]
        a = np.asarray(g["axis"], float); u = np.asarray(g["up"], float)
        ez = -u; ey = np.cross(ez, a)
        R = np.stack([a, ey, ez], 1)
        q = rot.qfrom_mat(R)
        return np.asarray(g["centre"], float) - R @ np.asarray(mount, float), q
    return np.asarray(job["goal"]["position"], float), rot.qnormalize(job["goal"]["attitude"])


class Settings:
    """Part of the hidden world."""

    def __init__(self, **kw):
        self.k = float(V("jobs.budget_factor"))
        self.caps = dict(V("sessions.outage_cap_s"))      # the table's caps per mode
        # monitor.recovery_s (required to fly blocks): per mode of the block, {"nominal": .., "quiet": ..,
        # "precision":..}; a single number is taken for every mode (tests)
        self.recovery_s = None
        self.cap_outages = True
        self.handback_enabled = True
        self.compat = ()             # names of CHANGES switched off (the comparison only)
        self.budget_fn = None        # a runner's own budget function (block, k, mount) -> (B, ...); default world/budget.py
        if "drift_max_mm_s2_per_min" in kw:
            # a linear ramp of the accelerometer offset is withdrawn: the slow change is the
            # bounded wander of world.imu.accel_offset, read from the table (no setting)
            raise ValueError("drift_max_mm_s2_per_min is withdrawn; the wander is read from the table")
        self.__dict__.update(kw)
        self.compat = tuple(self.compat)
        for c in self.compat:
            if c not in CHANGES:
                raise ValueError("unknown change %r (CHANGES: %s)" % (c, CHANGES))

    def recovery(self, mode):
        R = self.recovery_s
        if R is None:
            return None
        if isinstance(R, dict):
            return float(R[mode])
        return float(R)


class World:
    def __init__(self, session_name, tight_mode, list_no, seed_label, settings=None):
        self.S = settings or Settings()
        self.sess = load_session(session_name)
        self.name = session_name
        self.tight = tight_mode
        self.list_no = int(list_no)
        self.seed_label = str(seed_label)
        self.key = key_bytes()
        inp = params()
        self.inp = inp
        self.provisional = []        # the named pending parameters this world runs with their provisional values
        compat = self.S.compat
        dev = self.sess["inputs"]
        self.dt = float(V("interface.step_period_ms")) / 1000.0
        assert abs(self.dt - float(V("world.actuator_tick_ms")) / 1000.0) < 1e-12
        assert abs(float(V("world.battery_voltage_v")) - 14.0) < 1e-12
        # ---- per-session hidden parameters (fixed by the session; the same in every noise seed)
        hp = stream(self.key, session_name, "hidden")
        mf = V("world.mass_file")
        self.mass_file = mf
        if "mass_as_prepared" in compat:
            # with this change switched off: the file's mass plus the arm's, the file's centre of mass and inertia
            mass = float(mf["mass_kg"]) + float(V("world.arm_mass_kg"))
            self.com = np.asarray(mf["com_m"], float)
            self.J = np.diag(mf["inertia_diag"])
        else:
            # world.mass_as_prepared (the profile on file plus the arm's links at their stowed pose)
            mp, prov = provisional_or_value("mass_as_prepared")
            if prov:
                self.provisional.append("mass_as_prepared")
            mass = float(mp["mass_kg"])
            self.com = np.asarray(mp["com_m"], float)
            self.J = np.asarray(mp["inertia_kgm2"], float)
        self.mass = mass
        self.com_alloc = np.asarray(mf["com_m"], float)          # the allocator's profile: the one on file
        if "force_gain" in compat:
            pf = inp["prototype_stand_ins"]["force_gain_rule_numbers"]
            fgain = lambda lv: [pf["pooled_level2"]] * 6 if lv in (1, 2) else [pf["level3"][s_] for s_ in SIGNED]  # noqa: E731
        else:
            fgain = lambda lv: force_gains(lv, inp)  # noqa: E731
        self.force_gains = fgain
        gains = {lv: {"torque": torque_gains(lv), "force": fgain(lv)} for lv in (1, 2, 3)}
        dly = V("world.pure_delay_ms")["sessions_2022"] if dev["pure_delay_key"] == "sessions_2022" else V("world.pure_delay_ms_2024")
        dly = float(dly)                  # 0 for every session
        assert math.isfinite(dly) and dly >= 0.0
        self.pure_delay_ms = dly
        sr = V("world.station_rate")
        self.Om = float(sr["rate_mrad_s"]) * 1e-3 * np.asarray(sr["axis_cabin"], float)
        self.plant = Plant(gains, mass, self.com, self.J, float(dly) / 1000.0, float(V("world.impeller_ramp_rpm_s")),
                           self.Om, self.dt, com_alloc=None if "mass_as_prepared" in compat else self.com_alloc)
        self.fan = V("mode.fan_level")
        # air: body-fixed minus the delivered hold wrench, plus a cabin-frame slow component
        g2t, g2f = torque_gains(2), fgain(2)
        assert abs(g2f[0] - 1.45) < 1e-12
        Fh = np.asarray(dev["hold_force_median_N"], float)
        Th = np.asarray(dev["hold_torque_median_Nm"], float)
        from .config import load_spec as _load_spec
        _ht = (_load_spec().get("world.hold_torque_median_Nm") or {}).get("value") or {}
        if session_name in _ht:
            Th = np.asarray(_ht[session_name], float)        # by key (the inputs are asserted equal to it)
        Td = np.array([(g2t[2 * i] if Th[i] >= 0 else g2t[2 * i + 1]) * Th[i] for i in range(3)])
        self.air_body = np.concatenate([-1.45 * Fh, -Td])
        # world.hold_wrench.rule: the slow component's time constant and standard deviation, its
        # one-minute change clipped to told.steady_force_change_mN_per_min (in _air), the caps per axis
        hw = V("world.hold_wrench.rule")
        m_slow = re.search(r"time constant of ([0-9.]+) s and a standard deviation of ([0-9.]+) mN per axis", hw)
        m_cap = re.search(r"each axis capped at ([0-9.]+) mN and ([0-9.]+) mN m", hw)
        assert m_slow and m_cap and "change over one minute clipped to told.steady_force_change_mN_per_min" in hw
        self.air_cap = (float(m_cap.group(1)) / 1000.0, float(m_cap.group(2)) / 1000.0)
        self.air_tau, self.air_sd = float(m_slow.group(1)), float(m_slow.group(2)) / 1000.0
        # inertial unit
        im = V("world.imu.mount")
        self.R_mount = rot.rotz(90.0)
        assert "+90 degrees about z" in im["quarter_turn"]
        self.imu_pos = np.asarray(im["position_m"], float)
        nz = V("world.imu.noise")
        lv2_g = [hp.uniform(*r) * 1e-3 for r in nz["level2"]["rate_mrad_s_xyz"]]
        lv2_a = [hp.uniform(*r) * 1e-3 for r in nz["level2"]["accel_mm_s2_xyz"]]
        lv3_a = [hp.uniform(*r) * 1e-3 for r in nz["level3"]["accel_mm_s2_xyz"]]
        idle_tab = nz["impellers_idle"]
        if "rate_mrad_s" in idle_tab:
            idle_g = hp.uniform(*idle_tab["rate_mrad_s"]) * 1e-3      # the superseded table: one range for the three axes
        else:
            _ = hp.uniform()        # the draw is still taken (not used) so that the stream keeps its order
            idle_g = None
        idle_a = [hp.uniform(*r) * 1e-3 for r in nz["impellers_idle"]["accel_mm_s2_xyz"]]
        g3, prov = provisional_or_value("gyro_white_noise_level3")
        tab3 = nz.get("level3")
        if isinstance(tab3, dict) and "rate_mrad_s_xyz" in tab3 and "rate_mrad_s_xyz" in idle_tab:
            # the table as the maintainers changed it carries the gyro's white noise per level and sensor axis; 's
            # parameters, where set, hold the same numbers; the draws are 's, in its order
            for nm, tv in (("gyro_white_noise_level3", tab3), ("gyro_white_noise_level2", nz["level2"]),
                           ("gyro_white_noise_idle", idle_tab)):
                pe = inp["parameters"].get(nm)
                if pe is not None and pe.get("value") is not None:
                    assert [list(map(float, r)) for r in pe["value"]["rate_mrad_s_xyz"]] == [list(map(float, r)) for r in tv["rate_mrad_s_xyz"]], nm
            hp2 = stream(self.key, session_name, "hidden_part2")
            lv3_g = [hp2.uniform(*r) * 1e-3 for r in tab3["rate_mrad_s_xyz"]]
            lv2_g = [hp2.uniform(*r) * 1e-3 for r in nz["level2"]["rate_mrad_s_xyz"]]
            idle_gx = [hp2.uniform(*r) * 1e-3 for r in idle_tab["rate_mrad_s_xyz"]]
        elif prov:
            # the stand-in, a named pending parameter of: the level-2 values
            assert g3 == "the level-2 values of world.imu.noise"
            lv3_g = list(lv2_g)
            self.provisional.append("gyro_white_noise_level3")
            idle_gx = [idle_g] * 3
        else:
            # the measured white noise per level and sensor axis, drawn from a stream of its own
            pp = inp["parameters"]
            hp2 = stream(self.key, session_name, "hidden_part2")
            lv3_g = [hp2.uniform(*r) * 1e-3 for r in g3["rate_mrad_s_xyz"]]
            g2p, gip = pp.get("gyro_white_noise_level2"), pp.get("gyro_white_noise_idle")
            if g2p is not None and g2p.get("value") is not None:
                lv2_g = [hp2.uniform(*r) * 1e-3 for r in g2p["value"]["rate_mrad_s_xyz"]]
            if gip is not None and gip.get("value") is not None:
                idle_gx = [hp2.uniform(*r) * 1e-3 for r in gip["value"]["rate_mrad_s_xyz"]]
            else:
                idle_gx = [idle_g] * 3
        lv1_g = [0.5 * (i + g) for i, g in zip(idle_gx, lv2_g)]
        lv1_a = [0.5 * (i + a) for i, a in zip(idle_a, lv2_a)]
        self.imu_noise = {1: (np.array(lv1_g), np.array(lv1_a)), 2: (np.array(lv2_g), np.array(lv2_a)),
                          3: (np.array(lv3_g), np.array(lv3_a))}
        self.accel_offset = np.asarray(dev["accel_offset_sensor_m_s2"], float)
        # the offset's slow change: the bounded wander of world.imu.accel_offset, generated per run in
        # run(); the three draws of the withdrawn ramp rate are still taken from the hidden stream (and not used) so that
        # the session's other hidden parameters keep their draws
        _ = [hp.uniform(-1.0, 1.0) for _ in range(3)]
        sw = V("world.imu.accel_offset")["slow_drift_max_mm_s2_per_min"]
        assert sw["shape"] == "per axis a first-order process about the session's constant offset"
        self.wander_tau = float(sw["time_constant_s"])
        self.wander_sd = float(sw["standard_deviation_mm_s2"]) / 1000.0
        self.wander_clip = float(sw["one_minute_change_clipped_at_mm_s2"]) / 1000.0
        assert self.wander_clip <= float(V("told.accel_offset")["change_per_minute_max_mm_s2"]) / 1000.0 + 1e-15
        go = V("world.imu.gyro_offset")["body_axes_mrad_s"]
        self.gyro_offset = np.array([hp.uniform(*go[a]) * 1e-3 for a in ("x", "y", "z")])
        fil = V("world.imu.device_filter")
        # a moving average of n taps at an internal rate r has a group delay (n - 1) / (2 r): r and the window follow
        rate_int = (int(fil["taps"]) - 1) / (2.0 * float(fil["delay_ms"]) / 1000.0)
        self.imu_window = int(fil["taps"]) / rate_int
        if "internal_rate_hz" in fil:
            # the entry gives the internal rate and the window beside the delay: they agree
            assert abs(rate_int - float(fil["internal_rate_hz"])) <= 1e-9 * float(fil["internal_rate_hz"])
            assert abs(self.imu_window - float(fil["window_ms"]) / 1000.0) <= 1e-12
        self.imu_start = int(hp.integers(0, len(tables()["imu_intervals_s"])))
        # fix process
        fx = V("world.fix.noise")
        gr = V("world.fix.gross")
        gs = inp["parameters"]["gross_share"]["value"]["share_fewer_than_20_by_day"]
        self.p_gross = float(gs[dev["day"]])
        ratio, prov = provisional_or_value("corrections_to_fixes_ratio")
        if prov:
            self.provisional.append("corrections_to_fixes_ratio")
        fpm = dev["corrections_per_min"] / ratio
        fsp, prov = provisional_or_value("fix_success_process")
        if prov:
            self.provisional.append("fix_success_process")
        tab = tables()
        self.fix_process_kind = fsp.get("process", "stand_in")
        if self.fix_process_kind == "four_statistics":
            # world.fix.success_process as sets it: per session the parameters fit_four_stat found for the day
            per = dict(fsp["per_session"][session_name])
            fsp = dict(fsp, p_track=per["p_track"])
            mean_on = mean_off = None
            four = {k: float(per[k]) for k in ("n_on", "off_lo_s", "off_hi_s", "longest_s", "attempt_max_s")}
            four.update({k: float(per[k]) for k in ("q_long", "long_lo_s", "long_hi_s") if per.get(k) is not None})
        else:
            att_mean = float(np.mean(np.concatenate([tab["fix_delay_success_s"], tab["fix_delay_fail_s"]])))
            rate_track = fsp["p_track"] / att_mean * 60.0
            f_on = min(0.98, fpm / rate_track)
            mean_off = dev["longest_s"] / float(fsp["mean_off_divisor_of_longest"])
            mean_on = mean_off * f_on / (1.0 - f_on)
            four = {}
        # the first fix after an outage (world.fix.after_outage), a world parameter once set
        ao = inp["parameters"].get("fix_after_outage")
        self.after_outage = dict(ao["value"]) if ao is not None and ao.get("value") is not None else None
        fr = V("world.fix.frame")["camera_mount"]
        self.cam_t = np.asarray(fr["position_m"], float)
        self.cam_q = rot.qnormalize(fr["quaternion_xyzw"])
        self.fix_params = dict(p_track=fsp["p_track"], mean_on_s=mean_on, mean_off_s=mean_off, gross_lt20=self.p_gross,
                               gross_ge20=gr["share_20_or_more"], gross_pos=tuple(gr["position_m"]),
                               gross_att_rad=tuple(math.radians(x) for x in gr["attitude_deg"]),
                               noise_ge20=(fx["landmarks_20_or_more"]["position_m_per_axis"], fx["landmarks_20_or_more"]["attitude_rad"]),
                               noise_lt20=(fx["fewer_than_20"]["position_m_per_axis"], fx["fewer_than_20"]["attitude_rad"]),
                               cam_t=self.cam_t, cam_q=self.cam_q, fixes_per_min=fpm, **four)
        # supervisor
        sn = V("supervisor.noise")
        self.sup_sd = (sn["position_sd_m"], math.radians(sn["attitude_sd_deg"]))
        self.window = V("monitor.window_s")["releases_from_2021_08_26"]
        self.tol = {m: V("monitor.tolerance." + m) for m in MODES}
        self.mount = V("jobs.camera_mount_m")
        # blocks
        self.blocks = copy.deepcopy(self.sess["blocks_1_2"]) + [copy.deepcopy(self.sess["block_3"][tight_mode][list_no - 1])]
        self.block_budget = self.S.budget_fn or _budget.block_budget
        for b in self.blocks:
            b["budget_s"] = self.block_budget(b, self.S.k, self.mount)[0]
        self.rules = plan_rules()
        rt = V("world.ready_thresholds_rad_s")
        self.ready_enter, self.ready_leave = float(rt["enter"]), float(rt["leave"])
        self.watchdog_s = float(V("interface.watchdog_s"))
        self.capture_s = float(V("jobs.capture_s"))
        self.overspeed = float(V("safety.overspeed_m_s"))
        # the zone file in force with the exact test; in force when the session's paper names a zone
        # file in force (the scenario's kept-apart field, which World.paper() hands out), else recorded only
        from .zone_rule import ZoneRule
        self.zone_rule = ZoneRule(session_name, self.sess.get("day", dev.get("day")),
                                  paper_zone_file=(self.sess.get("paper_apart") or {}).get("zone_file"))
        self.zone_in_force = bool(self.zone_rule.in_force)
        self.zone_on = False
        self.zone_record = None
        ew = V("score.end_window_s")
        self.end_window = (float(ew[0]), float(ew[1]))
        self.end_window_close = float(ew[1])
        self._table_agrees(session_name, inp)

    def _table_agrees(self, session_name, inp):
        """the table's numbers of the world's entries equal the parameters the world flies."""
        from .config import load_spec
        sp = load_spec()

        def val(k):
            e = sp.get(k)
            return None if e is None else e.get("value")
        P = inp["parameters"]
        m = val("world.mass_as_prepared")
        if isinstance(m, dict) and "mass_as_prepared" not in self.S.compat:
            assert m == P["mass_as_prepared"]["value"], "world.mass_as_prepared differs from PARAMETERS.json"
        f = val("world.force_gain.rule")
        if isinstance(f, dict) and "force_gain" not in self.S.compat:
            assert f == P["force_gain"]["value"], "world.force_gain.rule differs from PARAMETERS.json"
        g = val("world.fix.gross")
        if isinstance(g, dict) and session_name in g.get("per_session_share_fewer_than_20", {}):
            assert g["per_session_share_fewer_than_20"][session_name] == self.p_gross, "world.fix.gross per session"
        f2 = val("world.fix.success_process")
        if isinstance(f2, dict) and self.fix_process_kind == "four_statistics" and session_name in f2.get("per_session", {}):
            per = P["fix_success_process"]["value"]["per_session"][session_name]
            assert {k: per[k] for k in f2["per_session"][session_name]} == f2["per_session"][session_name], "world.fix.success_process"
        a = val("world.fix.after_outage")
        if isinstance(a, dict) and self.after_outage is not None:
            assert all(a[k] == self.after_outage[k] for k in ("share_lt20", "min_stretch_s", "landmarks_below")), "world.fix.after_outage"
        h = val("world.hold_force_median_N")
        if isinstance(h, dict) and session_name in h:
            # the session's hold force is the table's
            assert [float(x) for x in h[session_name]] == [float(x) for x in self.sess["inputs"]["hold_force_median_N"]], \
                "world.hold_force_median_N differs from the session's inputs"
        ht = val("world.hold_torque_median_Nm")
        if isinstance(ht, dict) and session_name in ht:
            # the session's hold torque is the table's
            assert [float(x) for x in ht[session_name]] == [float(x) for x in self.sess["inputs"]["hold_torque_median_Nm"]], \
                "world.hold_torque_median_Nm differs from the session's inputs"

    # ------------------------------------------------------------------ the largest world time
    def max_world_time(self, worst=False):
        """the largest world time this session (its block-3 list as built) can take, with its terms:
            T = start_hold + checkout + sum_b (B_b + R_b) + sum_{b > 1} (G_b + start_hold) + n_stretches * dt
        B_b the block's budget (jobs.block_budget, Settings.k); R_b the recovery time of the block's mode (monitor.recovery_s,
        Settings.recovery_s): at most one recovery's tail runs past a block's budget; G_b the reposition into
        block b by the built-in law (its pose move under the limits it flies, accel_factor 0.5, plus its 5 s), from the pose
        block b - 1 ends at; dt one step per stretch (a stretch ends at the first step at or after its end).
        worst=False: the block ends within the supervisor's reach of the poses of its job list (its start pose, every goal
        pose of a leg, slew or view, plus monitor.tolerance's position and safety.overspeed_m_s over monitor.window_s):
        distance and angle to a fixed pose are convex along a straight leg, so the farthest of these poses bounds it.
        worst=True: any valid plan (a plan may wander between its start and its goal): the block ends within the largest
        speed limit of its modes times its budget of its start pose, plus the same reach; the turn at most pi. Without the
        zone file in force (row 19) nothing else bounds the cabin's extent in the world."""
        dt = self.dt
        hold = float(V("sessions.start_hold_s"))
        ck = float(V("sessions.checkout_s"))
        win = float(self.window)
        terms = []
        total = hold + ck
        n_str = 2
        for i, b in enumerate(self.blocks):
            B = float(b["budget_s"])
            R = self.S.recovery(b["mode"])
            if R is None:
                raise RuntimeError("monitor.recovery_s is not set")
            G = 0.0
            if i > 0:
                pb = self.blocks[i - 1]
                rep_mode = b["mode"] if b["mode_before"] == b["mode"] else "nominal"
                lim = mode_limits(rep_mode)
                p1, q1 = np.asarray(b["start_pose"]["position"], float), rot.qnormalize(b["start_pose"]["attitude"])
                reach = float(self.tol[pb["mode"]]["position_m"]) + self.overspeed * win
                poses = [(np.asarray(pb["start_pose"]["position"], float), rot.qnormalize(pb["start_pose"]["attitude"]))]
                for j in pb["jobs"]:
                    if j["kind"] in ("leg", "slew", "view") or (j["kind"] == "request" and j.get("request_kind") != "mode_not_available"):
                        poses.append(body_goal(j, self.mount))
                d = max(float(np.linalg.norm(p - p1)) for p, _ in poses)
                th = max(rot.qangle(q, q1) for _, q in poses)
                if worst:
                    vmax = max(mode_limits(m)[0] for m in {pb["mode"]} | {j["mode"] for j in pb["jobs"] if j["mode"] in MODES})
                    d = float(np.linalg.norm(poses[0][0] - p1)) + vmax * float(pb["budget_s"])
                    th = math.pi
                d += reach
                v, a, w, al = lim
                G = max(traj.trap_profile(d, 0.5 * a, v)[0], traj.trap_profile(th, 0.5 * al, w)[0]) + 5.0
                total += G + hold
                n_str += 2
            total += B + R
            n_str += 2
            terms.append({"block": b["name"], "mode": b["mode"], "budget_s": B, "recovery_tail_s": R, "reposition_s": G,
                          "start_hold_s": hold if i > 0 else 0.0})
        total += n_str * dt
        return {"T_s": total, "start_hold_s": hold, "checkout_s": ck, "blocks": terms, "steps_margin_s": n_str * dt,
                "worst": bool(worst)}

    # ------------------------------------------------------------------ what the node is told
    def paper(self):
        """the session paper (sessions.paper_fields): functions of published quantities and the job lists only."""
        blocks = []
        for b in self.blocks:
            jobs = []
            for j in b["jobs"]:
                jj = {"id": j["id"], "kind": "leg" if j["kind"] == "request" and j["request_shape"] == "leg" else
                      ("view" if j["kind"] == "request" else j["kind"]), "mode": j["mode"], "goal": j["goal"]}
                jobs.append(jj)
            blocks.append({"name": b["name"], "mode": b["mode"], "start_pose": b["start_pose"], "jobs": jobs,
                           "budget_s": b["budget_s"]})
        out = {"session": self.name + "/" + self.tight + "/" + str(self.list_no), "release_modes": list(MODES),
               "checkout_mode": "nominal", "start_pose": self.sess["start_pose"], "blocks": blocks,
               "window_s": self.window}
        apart = self.sess.get("paper_apart")
        if apart is not None:
            # the five fields of sessions.paper_fields the scenario keeps apart, as the scenario holds them
            for k, _ in PAPER_APART:
                out[k] = copy.deepcopy(apart[k])
        return out

    REFERENCE_READS = ("effective_response", "noise", "mass", "p_gross")       # cert.reference_reads may_read, in its order

    def reference_reads(self):
        """what the sensor-only reference may read beyond the frame, the paper and the printed values: exactly the four
        kinds of cert.reference_reads: the response per
        signed axis and fan level by the registered function (world.effective_response: effective_response(World), its
        gains per level, kind and signed axis, nothing else of its result), the noise levels of the inertial unit (per
        fan level: the gyro's and the accelerometer's white noise per axis) and of the fixes (per landmark class), the
        mass (the unit as prepared: mass, centre of mass, inertia), the per-session probability of a grossly wrong fix
        (per landmark class). Never the offsets' values, the air's force, the true pose, the lag, the static response or
        the plant's gains."""
        fx = self.fix_params
        return {"effective_response": self._effective_response_gains(),
                "noise": {"imu": {lv: (v[0].tolist(), v[1].tolist()) for lv, v in self.imu_noise.items()},
                          "fix": {"ge20": fx["noise_ge20"], "lt20": fx["noise_lt20"]}},
                "mass": {"mass_kg": self.mass, "com_m": self.com.tolist(), "inertia": self.J.tolist()},
                "p_gross": {"lt20": fx["gross_lt20"], "ge20": fx["gross_ge20"]}}

    def _effective_response_inputs(self):
        """every input effective_response(World) reads (the plant as built, the session's steady wrench and hold gains, the
        mass and inertia, the step, the start pose, the table's sequence and limits, the function's own code)."""
        from . import effective_response as _er
        pl = {k: v for k, v in self.plant.__dict__.items()}
        lv = (1, 2, 3)
        return (pl, np.asarray(self.air_body, float), [list(self.force_gains(x)) for x in lv], [list(torque_gains(x)) for x in lv],
                float(self.mass), np.asarray(self.J, float), float(self.dt), json.dumps(self.sess["start_pose"], sort_keys=True),
                V("cert.checkout_sequence"), json.dumps(V("mode.fan_level"), sort_keys=True),
                json.dumps(V("plan.acceleration_limit"), sort_keys=True),
                json.dumps([V("plan.speed_limit." + m) for m in MODES], sort_keys=True),
                hashlib.sha256(open(_er.__file__, "rb").read()).hexdigest())

    def _effective_response_gains(self):
        """world.effective_response of this session by the registered function: {level: {"force": {signed axis: gain}, "torque": {signed axis: gain}}}; the function's best-aligning lag
        and its static column are not handed over. Computed once per process for identical inputs.
        """
        from . import effective_response as _er
        dig = hashlib.sha256(pickle.dumps(self._effective_response_inputs(), protocol=4)).hexdigest()
        if dig not in _ER_MEMO:
            res = _er.effective_response(self)
            _ER_MEMO[dig] = {lv: {"force": {s: float(g) for s, g in res[lv]["force"].items()},
                                  "torque": {s: float(g) for s, g in res[lv]["torque"].items()}} for lv in sorted(res)}
        return copy.deepcopy(_ER_MEMO[dig])

    # ------------------------------------------------------------------ the run
    def run(self, node, log_every=1, stop_after_block=None, builtin_only=False):
        S = self.S
        dt = self.dt
        rs = lambda name: stream(self.key, self.name, self.seed_label, name)  # noqa: E731
        self.rng_sup = rs("supervisor")
        self.rng_air = rs("air")
        self.rng_bi = rs("builtin")
        t = 0.0
        sp = self.sess["start_pose"]
        st = self.plant.initial_state(self.fan["nominal"], sp["position"], sp["attitude"], t)
        self.st = st
        self.wander = AccelWander(rs("accel_wander"), self.wander_tau, self.wander_sd, self.wander_clip, t)
        self.imu = InertialUnit(rs("imu"), tables()["imu_intervals_s"], self.imu_start, self.imu_pos, self.R_mount,
                                self.com, self.imu_window, self.accel_offset, self.wander, self.gyro_offset,
                                self.imu_noise, t)
        ao = None
        if self.after_outage is not None:
            ao = {"rng": rs("after_outage"), "share_lt20": float(self.after_outage["share_lt20"]),
                  "min_stretch_s": float(self.after_outage["min_stretch_s"])}
        self.fix = FixProcess(rs("fix"), tables(), self.fix_params, t,
                              gross_whole_error="gross_whole_error" not in self.S.compat, process=self.fix_process_kind,
                              after_outage=ao)
        self.capw = None            # the last cap window opened (see _open_cap_window)
        self.cap_windows = []
        self.camera_busy_until = -1.0
        self.air_slow = self.air_sd * self.rng_air.standard_normal(3)
        self.air_change_max = float(V("told.steady_force_change_mN_per_min")) * 1e-3
        self.air_ring = np.zeros((int(round(60.0 / self.dt)), 3))
        self.air_k = 0
        self.hist = []          # (t, p_origin, q) for truth_at
        self.log = {k: [] for k in ("t", "p", "q", "v", "w", "cmd", "phase", "job", "inctl")}
        self.jobs_out = []
        self.events = []
        self.cmd = np.zeros(6)
        self.last_applied_t = 0.0
        self.block_records = []        # one record per block flown (its end at the budget, a recovery's tail apart)
        self.step_frame_t = None       # the time of the last step frame handed to the node
        self.capw_checkout = None      # the check-out's cap window
        self._budget_end = None        # the world time at which the block in flight reaches its budget
        self._tail = None
        self.level_cmd = self.fan["nominal"]
        self.mode_now = "nominal"
        self.ready_state = [True, True]
        self.ended = None
        # which steps the zone and contact tests may end the session at (in force), what the
        # untested steps and, for a session recorded only, the tested steps did; the kit's validator beside
        self.zone_on = False
        self.zone_record = dict(self.zone_rule.describe(), untested_steps=0, untested_failing_steps=0, untested_first_failure=None,
                                tested_steps=0, tested_failing_steps=0, tested_first_failure=None, kit_disagreeing_steps=0,
                                kit_first_disagreement=None)
        self.stock_mass = (float(self.mass_file["mass_kg"]), np.diag(self.mass_file["inertia_diag"]))
        self.t = t
        paper = self.paper()
        reply = node.setup(paper, self.reference_reads()) if (node is not None and not builtin_only) else None
        if isinstance(reply, dict) and reply.get("mass_profile"):
            mp = reply["mass_profile"]
            self.stock_mass = (float(mp["mass_kg"]), np.asarray(mp["inertia"], float))
        self.bi = BatchController(1, self.stock_mass[0], self.stock_mass[1])
        self.node = node
        if builtin_only == "init":
            return self
        # the check-out is treated as a block of the nominal mode for the cap: its window is opened
        # now, from one cap length and at least 5 s before the check-out's start (the start hold's end)
        if "checkout_cap" not in self.S.compat:
            self.capw_checkout = self._open_cap_window(self.CHECKOUT_MODE, self.t + float(V("sessions.start_hold_s")), kind="checkout")
        # 1. start hold by the built-in law
        self._builtin_hold(sp, float(V("sessions.start_hold_s")), phase="start_hold")
        if builtin_only:
            return self
        # 2. check-out: node in control, no supervisor, box check; the cap of block 1 reaches into its end
        ck = float(V("sessions.checkout_s"))
        if self.blocks:
            self._open_cap_window(self.blocks[0]["mode"], self.t + ck)
        self._checkout(sp, ck)
        if self.capw_checkout is not None:
            self._close_cap_window(self.capw_checkout)  # the check-out's window closes at its end
        # 3. blocks
        for bi_, b in enumerate(self.blocks):
            if self.ended:
                break
            if bi_ > 0:
                hold = float(V("sessions.start_hold_s"))
                self._builtin_reposition(b["start_pose"], b["mode"] if b["mode_before"] == b["mode"] else "nominal",
                                         on_plan=lambda until, b_=b, h_=hold: self._open_cap_window(b_["mode"], until + h_))
                self._builtin_hold(b["start_pose"], hold, phase="between")
            self._block(b)
            if stop_after_block is not None and bi_ + 1 >= stop_after_block:
                break
        return self

    def fly_block3(self, tight_mode, list_no):
        """after run(node, stop_after_block=2): fly block 3 with one of its job lists (the world, the node and every
        random stream continue from the state after block 2; a copy of the world is taken per list)."""
        b = copy.deepcopy(self.sess["block_3"][tight_mode][list_no - 1])
        b["budget_s"] = self.block_budget(b, self.S.k, self.mount)[0]
        self.tight, self.list_no = tight_mode, int(list_no)
        self.blocks = self.blocks[:2] + [b]
        if hasattr(self.node, "update_paper"):
            self.node.update_paper(self.paper())
        if not self.ended:
            hold = float(V("sessions.start_hold_s"))
            self._builtin_reposition(b["start_pose"], "nominal",
                                     on_plan=lambda until: self._open_cap_window(b["mode"], until + hold))
            self._builtin_hold(b["start_pose"], hold, phase="between")
            self._block(b)
        return self

    # ------------------------------------------------------------------ one world step
    def truth_at(self, t):
        h = self.hist
        if t >= h[-1][0]:
            return h[-1][1], h[-1][2]
        lo, hi = 0, len(h) - 1
        while hi - lo > 1:
            m = (lo + hi) // 2
            if h[m][0] <= t:
                lo = m
            else:
                hi = m
        a, b = h[lo], h[hi]
        f = (t - a[0]) / (b[0] - a[0])
        p = a[1] + f * (b[1] - a[1])
        q = rot.qnormalize((1 - f) * a[2] + f * (b[2] if np.dot(a[2], b[2]) >= 0 else -b[2]))
        return p, q

    def _frame(self, in_control, job_id, budget, t_left, cap):
        t = self.t
        if not self.hist or self.hist[-1][0] < t:
            self.hist.append((t, self.plant.origin(self.st), self.st["q"].copy()))
            if len(self.hist) > 600:
                del self.hist[:200]
        if getattr(self, "sensors_off", False):
            return None
        ins = self.imu.samples_until(t, max(1, int(self.st["level"])))
        # the cap in force at each image's time
        regs, fixes = self.fix.step(t, self.truth_at, cap if "cap_reach" in self.S.compat else self._cap_at)
        a = self.st["act"]
        sc = self.st["buf"][-1][1]
        target = self.plant.C.levels[self.level_cmd - 1] if self.level_cmd > 0 else 0.0
        enter, leave = self.ready_enter, self.ready_leave
        for i in (0, 1):
            err = abs(a["isp"][i] - target)
            if self.ready_state[i] and err > leave:
                self.ready_state[i] = False
            elif not self.ready_state[i] and err < enter:
                self.ready_state[i] = True
        return {"t": t, "in_control": bool(in_control), "job_id": job_id, "mode": self.mode_now,
                "fan_level_commanded": int(self.level_cmd), "inertial": ins, "registrations": regs, "fixes": fixes,
                "propulsion": {"impeller_speed": a["isp"].tolist(), "module_state": ["ready" if r else "ramping" for r in self.ready_state],
                               "nozzle_command": sc.tolist()},
                "camera": {"busy": bool(t < self.camera_busy_until - 1e-12)},
                "block": {"budget_s": budget, "time_left_s": t_left}}

    def _air(self):
        a = math.exp(-self.dt / self.air_tau)
        x = a * self.air_slow + self.air_sd * math.sqrt(1 - a * a) * self.rng_air.standard_normal(3)
        # the printed bound told.steady_force_change_mN_per_min binds the generator: the slow component's change over
        # any minute is clipped to it (the first-order process of world.hold_wrench.rule alone exceeds it in a few
        # per cent of minutes)
        n = self.air_ring.shape[0]
        old = self.air_ring[self.air_k % n]
        if self.air_k >= n:
            x = np.clip(x, old - self.air_change_max, old + self.air_change_max)
        self.air_ring[self.air_k % n] = x
        self.air_k += 1
        self.air_slow = x
        R = rot.qmat(self.st["q"])
        F = self.air_body[:3] + R.T @ self.air_slow
        F = np.clip(F, -self.air_cap[0], self.air_cap[0])
        T = np.clip(self.air_body[3:], -self.air_cap[1], self.air_cap[1])
        return np.concatenate([F, T])

    def _advance(self, wrench, phase, job_idx, in_ctl):
        st = self.st
        t0 = self.t
        fs, alpha, w0, w1 = self.plant.step(st, wrench, self.level_cmd, self._air())
        self.imu.push(t0, t0 + self.dt, fs, alpha, w0, w1)
        self.t = st["t"]
        lg = self.log
        lg["t"].append(self.t); lg["p"].append(self.plant.origin(st)); lg["q"].append(st["q"].copy())
        lg["v"].append(st["v"].copy()); lg["w"].append(st["w"] - rot.qmat(st["q"]).T @ self.Om)
        lg["cmd"].append(np.asarray(wrench, float).copy()); lg["phase"].append(phase); lg["job"].append(job_idx)
        lg["inctl"].append(in_ctl)
        if np.linalg.norm(st["v"]) > self.overspeed:
            self.ended = ("overspeed", self.t)
        if getattr(self, "ended", None) is None and getattr(self, "zone_record", None) is not None:
            self._zone_test(phase)

    def _zone_test(self, phase):
        """the zone rule's and the contact rule's test of this step (after the over-speed test):
        the exact test decides; the kit's validator is evaluated beside for the record; a session whose paper names no zone file
        in force is recorded only."""
        zr, rec = self.zone_rule, self.zone_record
        p = self.plant.origin(self.st)
        why = zr.keep_in(p)
        kit = zr.kit_keep_in(p)
        if (kit != "ok") != (why is not None):
            rec["kit_disagreeing_steps"] += 1
            if rec["kit_first_disagreement"] is None:
                rec["kit_first_disagreement"] = {"t": self.t, "phase": phase, "position": p.tolist(), "exact": why, "kit": kit,
                                                 "clearance_m": zr.clearance(p)}
        if not self.zone_on:
            rec["untested_steps"] += 1
            if why is not None:
                rec["untested_failing_steps"] += 1
                if rec["untested_first_failure"] is None:
                    rec["untested_first_failure"] = {"t": self.t, "phase": phase, "position": p.tolist(), "exact": why}
            return
        rec["tested_steps"] += 1
        cause = "keep_in" if why is not None else None
        if cause is None:
            why = zr.contact(p)
            cause = "contact" if why is not None else None
        if cause is None:
            return
        rec["tested_failing_steps"] += 1
        if rec["tested_first_failure"] is None:
            rec["tested_first_failure"] = {"t": self.t, "phase": phase, "position": p.tolist(), "cause": cause, "exact": why,
                                           "kit": kit, "clearance_m": zr.clearance(p)}
        if not self.zone_in_force:
            return                               # recorded only (no paper names a zone file in force)
        self.ended = (cause, self.t)
        self.events.append(("session_end", cause, self.t, {"phase": phase, "position": p.tolist(), "exact": why, "kit": kit,
                                                            "clearance_m": zr.clearance(p), "edition": zr.edition,
                                                            "radius_m": zr.radius}))

    def _builtin_cmd(self, sps, t):
        p, q = self.plant.origin(self.st), self.st["q"]
        n = self.rng_bi.standard_normal(6)
        fx = self.fix_params["noise_ge20"]
        pe = p + fx[0] * n[:3]
        qe = rot.qmul(rot.qexp(fx[1] * n[3:]), q)
        ve = self.st["v"]
        we = self.st["w"] - rot.qmat(q).T @ self.Om
        k = 0
        for i, s in enumerate(sps):
            if s[0] <= t:
                k = i
        s = sps[k]
        dts = max(0.0, t - s[0])
        spd = {"pos": s[1], "vel": s[2], "acc": s[3], "quat": wxyz(s[4]), "omega": s[5], "alpha": s[6]}
        F, T, status, _, _ = self.bi.step(np.array([dts]), pe[None], wxyz(qe)[None], ve[None], we[None],
                                         {k_: np.asarray(v)[None] for k_, v in spd.items()}, np.array([2]))
        return np.concatenate([F[0], T[0]])

    def _builtin_fly(self, sps, until, phase, job_idx=-1, frames=True, cap=None):
        once = "frame_time_once" not in self.S.compat
        while self.t < until - 1e-9 and not self.ended:
            if once and frames and self.node is not None and self.step_frame_t is not None and self.t <= self.step_frame_t + 1e-9:
                # a step frame of this instant has been handed already (the block or the job ended at a job's
                # step): this step is flown without a frame; the next frame comes one step later and holds what arrived since
                pass
            else:
                fr = self._frame(False, None, None, None, cap)
                if frames and self.node is not None:
                    self.node.step(fr)
                    self.step_frame_t = self.t
            w = self._builtin_cmd(sps, self.t)
            self.cmd = w
            self._advance(w, phase, job_idx, False)

    def _builtin_hold(self, pose, dur, phase):
        p, q = np.asarray(pose["position"], float), rot.qnormalize(pose["attitude"])
        sps = [(self.t, p, np.zeros(3), np.zeros(3), q, np.zeros(3), np.zeros(3))]
        self.bi.reset()
        self._builtin_fly(sps, self.t + dur, phase)

    def _builtin_path(self, p1, q1, mode):
        p0, q0 = self.plant.origin(self.st), self.st["q"]
        return traj.pose_move(p0, q0, p1, q1, mode_limits(mode), self.t, accel_factor=0.5)

    def _builtin_reposition(self, pose, mode, on_plan=None):
        p1, q1 = np.asarray(pose["position"], float), rot.qnormalize(pose["attitude"])
        sps = self._builtin_path(p1, q1, mode)
        until = sps[-1][0] + 5.0
        if on_plan is not None:
            on_plan(until)
        self.bi.reset()
        self._builtin_fly(sps, until, "between")

    # ------------------------------------------------------------------ the outage cap
    CHECKOUT_MODE = "nominal"          # the check-out takes the nominal mode's cap (sessions.outage_cap_s, its unit)

    def _open_cap_window(self, mode, t_block_start, kind="block"):
        """sessions.outage_cap_rule: the cap of the block's mode is in force from one cap length, and at least the
        rule's least reach, before the block's (planned) start; the block's end closes it (_close_cap_window). A block
        starts at the first step at or after its planned start, so the window opens at least the reach before it.
        kind "checkout" is the check-out's window; it is returned so that its end closes it."""
        if "cap_reach" in self.S.compat or not self.S.cap_outages:
            return None
        c = float(self.S.caps[mode])
        reach = max(c, cap_reach_min_s())
        w = {"mode": mode, "cap": c, "reach_s": reach, "t_block_start": float(t_block_start),
             "t_from": float(t_block_start) - reach - 1e-6, "t_to": None}
        if kind != "block":
            w["kind"] = kind
        self.capw = w
        self.cap_windows = self.cap_windows + [w]
        return w

    def _close_cap_window(self, w=None):
        """close the last window opened (a block's end), or the given one (the check-out's at its end)."""
        w = self.capw if w is None else w
        if w is not None and w["t_to"] is None:
            w["t_to"] = float(self.t)

    def _cap_at(self, t_img):
        """the cap in force at an image time: inside a block's window; after the block's end, for a cap above 0 s,
        until the first successful image after the end (the stretch running at the end is cut as well); None
        elsewhere. Every window opened so far is read (the next block's window is opened while the last block's tail
        may still run); where two are in force, the smaller cap holds."""
        caps = []
        for w in reversed(getattr(self, "cap_windows", [])):
            if t_img < w["t_from"]:
                continue
            if w["t_to"] is None or t_img <= w["t_to"] + 1e-12:
                caps.append(w["cap"])
            elif w["cap"] > 0.0 and self.fix.last_success_img <= w["t_to"] + 1e-12:
                caps.append(w["cap"])
            if len(caps) == 0 and w["t_to"] is not None and self.fix.last_success_img > w["t_to"] + 1e-12:
                break            # this window and every older one are over
        return min(caps) if caps else None

    # ------------------------------------------------------------------ check-out
    def _checkout(self, sp, dur):
        box = V("sessions.checkout_box")
        p0, q0 = np.asarray(sp["position"], float), rot.qnormalize(sp["attitude"])
        t_end = self.t + dur
        out = False
        self.zone_on = True                  # the check-out's steps are tested (in control and after a box exit)
        while self.t < t_end - 1e-9 and not self.ended:
            fr = self._frame(not out, "checkout", None, None, None)
            rep = self.node.step(fr)
            self.step_frame_t = self.t
            if not out:
                w = self._reply(rep)
                if w is None:
                    out = True
                    w = self.cmd
            else:
                w = self._builtin_cmd([(self.t, p0, np.zeros(3), np.zeros(3), q0, np.zeros(3), np.zeros(3))], self.t)
            self.cmd = w
            self._advance(w, "checkout", -1, not out)
            if not out:
                pm = self.plant.origin(self.st) + self.sup_sd[0] * self.rng_sup.standard_normal(3)
                qm = rot.qmul(rot.qexp(self.sup_sd[1] * self.rng_sup.standard_normal(3)), self.st["q"])
                if np.linalg.norm(pm - p0) > box["position_m"] or rot.qangle(qm, q0) > math.radians(box["attitude_deg"]):
                    out = True
                    self.events.append(("checkout_box", self.t))
                    self.bi.reset()
        self.zone_on = False

    def _reply(self, rep):
        """the applied wrench of a reply, or None for a protocol error; a reply without a command keeps the last."""
        if rep is None or rep.get("force") is None:
            return self.cmd
        try:
            w = np.concatenate([np.asarray(rep["force"], float).reshape(3), np.asarray(rep["torque"], float).reshape(3)])
        except Exception:
            return None
        if not np.all(np.isfinite(w)):
            return None
        self.last_applied_t = self.t
        return w

    # ------------------------------------------------------------------ a block
    def _block(self, b):
        mode = b["mode"]
        tol = self.tol[mode]
        B = b["budget_s"]
        t_start = self.t
        cap = self.S.caps[mode] if self.S.cap_outages else None
        if b["mode_before"] != mode:
            self.level_cmd = self.fan[mode]
            self.mode_now = mode
        else:
            self.mode_now = mode
            self.level_cmd = self.fan[mode]
        handbacks = 0
        prev_goal = (np.asarray(b["start_pose"]["position"], float), rot.qnormalize(b["start_pose"]["attitude"]))
        prev_completed_hint = True
        block_over = False
        ended_by = None
        self.last_applied_t = self.t
        keyed = "plan_verdict_keys" not in self.S.compat
        self._budget_end = t_start + B
        self._tail = None
        self.zone_on = True                  # a block's steps are tested (jobs and recoveries, a tail included)
        for ji, job in enumerate(b["jobs"]):
            rec = {"block": b["name"], "mode": mode, "id": job["id"], "kind": job["kind"], "job_mode": job["mode"],
                   "goal": job["goal"], "t_start": self.t, "status": None, "events": [], "source": job.get("source")}
            if job["kind"] == "request":
                rec["request_kind"] = job["request_kind"]
            self.jobs_out.append(rec)
            if block_over or self.ended or self.t - t_start >= B:
                rec["status"] = "not_started"
                continue
            fr = self._frame(True, job["id"], B, B - (self.t - t_start), cap)
            pj = {"id": job["id"], "kind": "leg" if job["kind"] == "request" and job["request_shape"] == "leg" else
                  ("view" if job["kind"] == "request" else job["kind"]), "mode": job["mode"], "goal": job["goal"]}
            ans = self.node.plan(pj, fr)
            ans = ans if isinstance(ans, dict) else {}
            verdict = ans.get("verdict") if keyed else None
            if verdict is not None:
                # the node's verdict on the plan call
                vo = verdict.get("outcome") if isinstance(verdict, dict) else None
                if vo not in VERDICTS:
                    raise RuntimeError("task error: a plan verdict whose outcome is not one of %s: %r" % (VERDICTS, verdict))
                rec["verdict"] = {k: verdict.get(k) for k in ("outcome", "reason", "rule", "setpoint", "what") if verdict.get(k) is not None}
                if vo == "protocol_error":
                    rec["plan_answer"] = "protocol_error"
                    rec["status"] = "protocol"
                    rec["protocol_error_t"] = self.t
                    rec["protocol_error_call"] = "plan"
                    block_over = True
                    ended_by = "protocol"
                    continue
                if vo == "refused":
                    ans = dict(ans, refuse=verdict.get("reason") or ans.get("refuse") or True)
                elif ans.get("refuse"):
                    raise RuntimeError("task error: the plan verdict %r with a refusal in the reply" % vo)
            rec["plan_answer"] = "refuse" if ans.get("refuse") else "accept"
            if keyed and ans.get("refuse") and isinstance(ans.get("refuse"), str):
                rec["refusal_reason"] = ans["refuse"]          # the reason of a refusal in the record
            if job["kind"] == "request":
                if ans.get("refuse"):
                    rec["status"] = "refused"
                    continue
                if job["request_kind"] == "mode_not_available":
                    rec["status"] = "accepted_request_not_flown"
                    continue
            elif ans.get("refuse"):
                rec["status"] = "refused_feasible"
                continue
            gp, gq = prev_goal if job["kind"] == "hold" else body_goal(job, self.mount)
            bad_idx = None
            if job["kind"] == "hold":
                sps = [(self.t, prev_goal[0], np.zeros(3), np.zeros(3), prev_goal[1], np.zeros(3), np.zeros(3))]
                bad = None
                if verdict is not None and verdict["outcome"] == "invalid":
                    bad, bad_idx = verdict.get("rule"), verdict.get("setpoint")      # an invalid hold
            elif verdict is not None and verdict["outcome"] == "invalid":
                sps = []
                bad, bad_idx = verdict.get("rule"), verdict.get("setpoint")          # the node's rule
            else:
                sps = ans.get("setpoints") or []
                lim = mode_limits(job["mode"] if job["mode"] in MODES else mode)
                if keyed:
                    # the attitudes normalised before the checks (as the runner does); the keyed rules
                    nsps, what = traj.normalise_plan(sps)
                    if what is not None:
                        rec["plan_answer"] = "protocol_error"
                        rec["status"] = "protocol"
                        rec["protocol_error_t"] = self.t
                        rec["protocol_error_call"] = "plan"
                        rec["protocol_error_what"] = what
                        block_over = True
                        ended_by = "protocol"
                        continue
                    sps = nsps
                    kb = traj.check_plan_keyed(sps, lim, prev_goal, (gp, gq), job["kind"], self.rules)
                    bad, bad_idx = (kb[0], kb[1]) if kb else (None, None)
                    if verdict is not None and bad:
                        raise RuntimeError("task error: the node's verdict 'valid' and the world's own check (%s at setpoint %s) "
                                           "disagree on job %s" % (bad, bad_idx, job["id"]))
                else:
                    bad = traj.check_plan(sps, lim, prev_goal, (gp, gq), job["kind"], self.rules)
            if bad:
                rec["status"] = "invalid_plan"
                rec["invalid_rule"] = bad
                if keyed:
                    rec["invalid_setpoint"] = bad_idx
                self._recover(gp, gq, mode, ji, cap, rec)
                prev_goal = (gp, gq)
                continue
            rec["t_plan_end"] = sps[-1][0]
            rec["setpoints"] = [(s[0], s[1].tolist(), s[4].tolist()) for s in sps[::max(1, len(sps) // 50)]]
            res = self._fly_job(job, rec, sps, mode, tol, t_start, B, cap, ji, prev_goal, (gp, gq))
            if res == "handback":
                handbacks += 1
                rec["status"] = "handback"
                if handbacks >= 3:
                    block_over = True
                    ended_by = "third_handback"
                    rec["third_handback"] = True
                else:
                    self._recover(gp, gq, mode, ji, cap, rec)
            elif res in ("budget", "watchdog", "protocol", "ended"):
                rec["status"] = res
                block_over = True
                ended_by = res
            else:
                rec["status"] = res
            if job["kind"] != "hold":
                prev_goal = (gp, gq)
        # the block is over: the built-in law takes over (the rest of the budget is not flown by the node)
        tail = self._tail
        t_end = tail["t_budget"] if tail is not None else self.t          # the block's record ends at the budget
        if ended_by is None:
            ended_by = "budget" if (tail is not None or t_end - t_start >= B - 1e-9) else ("ended" if self.ended else "jobs_done")
        self.events.append(("block_end", b["name"], t_end, t_end - t_start, B))
        if tail is not None:
            self.events.append(("recovery_after_budget", b["name"], tail["t_budget"], tail["t_to"], tail["job_id"]))
        self.block_records.append({"name": b["name"], "mode": mode, "t_start": t_start, "t_end": t_end, "elapsed_s": t_end - t_start,
                                   "budget_s": B, "ended_by": ended_by, "recovery_after_budget": tail})
        self._close_cap_window()
        self._budget_end = None
        self._tail = None
        self.zone_on = False

    def _recover(self, gp, gq, mode, ji, cap=None, rec=None):
        """hand-back (or invalid plan): the built-in law brings the robot to rest at the job's goal pose; that takes
        monitor.recovery_s of world time (off the block's budget); then the node flies the next job. A
        recovery that reaches the block's budget is flown on to its end as part of the gap ("recovery_after_budget"; the
        block's record ends at the budget, the cap window closes there). The watchdog is reset when control
        returns to the controller (it does not run while the built-in controller flies)."""
        R = self.S.recovery(mode)          # per mode of the block
        if R is None:
            raise RuntimeError("monitor.recovery_s is not set (it is required before any block is flown)")
        sps = self._builtin_path(gp, gq, mode)
        self.bi.reset()
        t0 = self.t
        t_to = self.t + R
        tb = self._budget_end
        if "block_end_at_budget" not in self.S.compat and tb is not None and t_to > tb + 1e-9:
            self._builtin_fly(sps, tb, "recovery", job_idx=ji, cap=cap)
            t_budget = self.t
            self._close_cap_window()
            self._builtin_fly(sps, t_to, "recovery_after_budget", job_idx=ji, cap=None)
            self._tail = {"job_id": self.jobs_out[-1]["id"] if self.jobs_out else None, "t_from": t0, "t_budget": t_budget, "t_to": self.t,
                          "tail_s": self.t - t_budget, "recovery_s": R}
        else:
            self._builtin_fly(sps, t_to, "recovery", job_idx=ji, cap=cap)
        if rec is not None and "block_end_at_budget" not in self.S.compat:
            rec["recovery"] = {"t_from": t0, "t_to": self.t, "recovery_s": R,
                               "after_budget_s": (self._tail["tail_s"] if self._tail is not None and self._tail["t_from"] == t0 else 0.0)}
        if "watchdog_from_control" not in self.S.compat:
            self.last_applied_t = self.t           # control returns to the controller now

    def _fly_job(self, job, rec, sps, mode, tol, t_block, B, cap, ji, prev_goal, goal):
        kind = job["kind"]
        t_job = self.t
        viol_since = None
        accepted_event = None
        win_end = None
        hold_end = self.t + float(job["goal"]["duration_s"]) if kind == "hold" else None
        if kind == "hold":
            win_end = hold_end
        rec["t_motion_start"] = t_job
        while True:
            if self.ended:
                return "ended"
            el = self.t - t_block
            if el >= B - 1e-9:
                return "budget" if (win_end is None or self.t < win_end - 1e-9) else "finished"
            fr = self._frame(True, job["id"], B, B - el, cap)
            rep = self.node.step(fr)
            self.step_frame_t = self.t
            w = self._reply(rep)
            if w is None:
                rec["protocol_error_t"] = self.t
                return "protocol"
            if self.t - self.last_applied_t > self.watchdog_s + 1e-9:
                return "watchdog"
            ev = (rep or {}).get("event") or "none"
            if ev in ("done", "ready") and accepted_event is None:
                if self.t >= sps[-1][0] - 1e-9 and ((ev == "done" and kind in ("leg", "slew")) or (ev == "ready" and kind == "view")):
                    accepted_event = self.t
                    rec["t_event"] = self.t
                    if kind in ("leg", "slew"):
                        pm = self.plant.origin(self.st) + self.sup_sd[0] * self.rng_sup.standard_normal(3)
                        if np.linalg.norm(pm - goal[0]) > tol["end_point_m"]:
                            rec["handback_reason"] = "end_point"
                            rec["t_handback"] = self.t
                            return "handback"
                        win_end = self.t + self.end_window_close
                    else:
                        win_end = self.t + self.capture_s
                        self.camera_busy_until = win_end      # the camera takes the view's picture
                else:
                    rec["events"].append(("ignored_event", ev, self.t))
            self.cmd = w
            self._advance(w, "job", ji, True)
            # supervisor (true motion against the accepted plan, with its measurement noise)
            if self.S.handback_enabled:
                if kind == "hold":
                    rp, rq = prev_goal
                else:
                    rp, _, rq, _ = traj.ref_at(sps, self.t)
                pm = self.plant.origin(self.st) + self.sup_sd[0] * self.rng_sup.standard_normal(3)
                qm = rot.qmul(rot.qexp(self.sup_sd[1] * self.rng_sup.standard_normal(3)), self.st["q"])
                e_p = float(np.linalg.norm(rp - pm))
                e_a = rot.qangle(rq, qm)
                if e_p > tol["position_m"] or e_a > math.radians(tol["attitude_deg"]):
                    if viol_since is None:
                        viol_since = self.t
                    elif self.t - viol_since > self.window + 1e-9:
                        rec["handback_reason"] = "tolerance"
                        rec["t_handback"] = self.t
                        return "handback"
                else:
                    viol_since = None
            if win_end is not None and self.t >= win_end - 1e-9:
                rec["t_window_end"] = self.t
                return "finished"
