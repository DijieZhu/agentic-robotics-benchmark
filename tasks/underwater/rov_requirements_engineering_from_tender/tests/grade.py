"""grade -- the static grader of rov_requirements_engineering_from_tender (verify stage, hidden side).

The submission is three static JSON files.  Nothing in them is imported or executed: they are read as bytes and parsed
by a strict JSON loader (below), validated against schemas.json and the cross-file rules of gate G0, then scored on the
hidden world in world/ with evaluation_basis.py, the byte-identical copy of the published procedure.

    st  = load_stage(stage_dir)                      # config, basis, digest, schemas, world; truth table recomputed
    docs, problems = load_submission(folder, st)     # gate G0 (files, strict JSON, schemas, cross-file rules)
    value, metrics = grade(docs, st)                 # 0.45 V + 0.15 A + 0.15 P + 0.05 Q + 0.20 F, metrics whitelisted

Errors: G0Error = the submission breaks a structural rule (gate G0 = 0, score 0); TaskError = the task itself is
inconsistent (a world file differs from its recomputation, a pinned file changed, F would not be a total function);
verify.py turns a TaskError into a task error of the run, never into a zero.

Scoring (DESIGN v1.3 sec. 5, scoring v1.2 / ruling L8; weights J10):
  sets     on the truth table (N-01 by the Part C rule n01_trim_reachable): F_k = {c: m_k < -delta_k}, OK, Acc, Rej,
           c*_b, J*, valid(b), neutral(b)
  V        S_acc x S_rej; S_acc = mean over Acc of [accept with options = c*_b]; S_rej = mean over Rej of r_b,
           r_b = 1 when b is rejected, cites >= 1 non-neutral ground and every cited non-neutral ground is valid
  A        award_ok x ramp(1 - (price(c_a) - J*) / (B J*)) x G; award_ok = [c_a in OK] and [c_a is the cheapest
           configuration the submission accepts]; G = min r_b over the rejected bids whose base price < price(c_a)
  P        [V > 0] x S_rej x mean over Acc of ([accept with options = c*_b] x mean over the 8 site keys of
           ramp((5 tau - e) / (4 tau))), tau = rows.json tolerance of c*_b
  Q        [V > 0] x max(0, correct non-keep labels - wrong non-keep labels) / true non-keep labels
  F        award_ok x [A(S) non-empty and every condition inside the awarded bid's envelope] x |Phi+ & A(S)| / |Phi+|
           x mean over the live band rows k of [A(S) & F_k = empty], on the awarded configuration's own delivery band
           (N-01 by the Part D rule n01_as_delivered); facility quantities memoised by the band dimension they read.
The formulas and their floating-point evaluation order are those of the phase 1 prototype grader
(-dev/phase1/r3_world/grader_proto.py, scoring 'v12'); the world loader reproduces its world model
(-dev/phase1/r3_world/world_proto.py World) on the contract's world files.  Standard library + numpy + jsonschema.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import os
import stat
import sys
from pathlib import Path

import jsonschema

SUBMISSION_FILES = ("evaluation.json", "spec_review.json", "acceptance_schedule.json")
SCHEMA_KEY = {"evaluation.json": "evaluation", "spec_review.json": "spec_review",
              "acceptance_schedule.json": "acceptance_schedule"}
WORLD_FILES = {"site.json": "uw-req-site/1", "as_built.json": "uw-req-as-built/1", "catalogue.json": "uw-req-catalogue/1",
               "rows.json": "uw-req-rows/1", "truth_table.json": "uw-req-truth/1", "clause_map.json": "uw-req-clause-map/1",
               "delivery_ranges.json": "uw-req-delivery/1", "selection.json": "uw-req-selection/1"}
CONFIG_SCHEMA = "uw-req-grader-config/1"
SERIES_CODE = "series"            # the BOM's word for an external camera housing whose as-built variant is hidden
MAX_PROBLEMS = 8


class TaskError(RuntimeError):
    """The task is inconsistent (a world file, a pinned file or a precondition of the score): a task error."""


class G0Error(ValueError):
    """The submission breaks a structural rule of gate G0."""


def ramp(x):
    return max(0.0, min(1.0, x))


def _short(x, n=120):
    s = repr(x)
    return s if len(s) <= n else s[:n] + "..."


# ================================================================================================ strict JSON (G0-2)
def _check_strings(obj, what):
    """Every string (keys included) must be valid Unicode text: a JSON escape can produce a lone surrogate."""
    todo = [obj]
    while todo:
        x = todo.pop()
        if isinstance(x, str):
            try:
                x.encode("utf-8")
            except UnicodeEncodeError:
                raise G0Error(f"{what}: a string holds a lone surrogate") from None
        elif isinstance(x, dict):
            todo.extend(x.keys())
            todo.extend(x.values())
        elif isinstance(x, list):
            todo.extend(x)


def strict_json_text(text, what):
    """json.loads with the G0-2 rules: no duplicate key, no NaN / Infinity, no integer literal longer than
    sys.get_int_max_str_digits() characters, every number a finite float, no lone surrogate, an object at the top."""
    limit = sys.get_int_max_str_digits()

    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise G0Error(f"{what}: duplicate key {_short(k)}")
            out[k] = v
        return out

    def constant(name):
        raise G0Error(f"{what}: {name} is not a finite number")

    def p_int(s):
        digits = len(s) - (1 if s[:1] == "-" else 0)
        if limit and digits > limit:
            raise G0Error(f"{what}: an integer literal has more than {limit} digits")
        v = int(s)
        try:
            float(v)
        except OverflowError:
            raise G0Error(f"{what}: an integer is too large for a float") from None
        return v

    def p_float(s):
        v = float(s)
        if not math.isfinite(v):
            raise G0Error(f"{what}: a number is not finite once read")
        return v

    try:
        obj = json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_int=p_int, parse_float=p_float)
    except G0Error:
        raise
    except RecursionError:
        raise G0Error(f"{what}: nested too deeply") from None
    except ValueError as exc:
        raise G0Error(f"{what}: not JSON ({' '.join(str(exc).split())[:200]})") from None
    _check_strings(obj, what)
    if not isinstance(obj, dict):
        raise G0Error(f"{what}: the top level is not an object")
    return obj


def strict_json_bytes(data, what):
    """G0-2 on raw bytes: UTF-8 without a byte-order mark, then strict_json_text."""
    if data[:3] == b"\xef\xbb\xbf":
        raise G0Error(f"{what}: starts with a byte-order mark")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise G0Error(f"{what}: not UTF-8 ({exc.reason} at byte {exc.start})") from None
    return strict_json_text(text, what)


def read_regular_file(path, max_bytes):
    """G0-1: the bytes of a regular file (not a link, not special) of at most max_bytes bytes, read without following
    a link and never more than max_bytes + 1 bytes (a sparse or growing file cannot make it read more)."""
    path = os.fspath(path)
    name = os.path.basename(path)
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise G0Error(f"{name}: missing") from None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise G0Error(f"{name}: not a regular file")
    if st.st_size > max_bytes:
        raise G0Error(f"{name}: larger than {max_bytes} bytes")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        raise G0Error(f"{name}: cannot be opened ({type(exc).__name__})") from None
    with os.fdopen(fd, "rb") as fh:
        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
            raise G0Error(f"{name}: not a regular file")
        data = fh.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise G0Error(f"{name}: larger than {max_bytes} bytes")
    return data


# ================================================================================================ stage files
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_trusted_json(path):
    """A file of the task itself (config, schemas, world): the same strict loader; any defect is a task error."""
    try:
        with open(path, "rb") as fh:
            return strict_json_bytes(fh.read(), os.path.basename(path))
    except (OSError, G0Error) as exc:
        raise TaskError(f"cannot read {path}: {exc}") from None


def load_basis(path, sha256):
    """The published procedure by path (sha256-pinned in grader_config.json; never a module of the submission)."""
    got = sha256_file(path)
    if got != sha256:
        raise TaskError(f"evaluation_basis.py sha256 {got[:16]}... is not the pinned {sha256[:16]}...")
    spec = importlib.util.spec_from_file_location("uw_req_verify_evaluation_basis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def digest_facts(path, sha256, version, basis):
    """The reference-vehicle facts the world refers to, read from the byte copy of the frozen digest
    (spec_values.json; sha256-pinned).  Same extraction as -dev/gen/ref_config.py digest_facts (= the R3 prototype's
    digest.load); the one transcendental step (the thruster-axis cosines of the reference system coefficient) uses the
    basis' correctly rounded cosine, so the result does not depend on the machine's maths library."""
    got = sha256_file(path)
    if got != sha256:
        raise TaskError(f"digest sha256 {got[:16]}... is not the pinned {sha256[:16]}...")
    raw = load_trusted_json(path)
    if (raw.get("digest") or {}).get("version") != version:
        raise TaskError(f"digest version {(raw.get('digest') or {}).get('version')!r} is not {version!r}")
    v = raw["values"]

    def val(key):
        return v[key]["value"]

    trim = val("mission.static_trim_fresh_deg")
    angles = val("horizontal_thruster_angle_deg")
    wrench12 = val("mission.max_wrench")["12V"]
    curve = val("thruster_curve_limits")
    unit = val("uwkit.ballast_unit")
    sens = val("sensor_specs")
    f = {
        "m_nom_kg": float(val("mission.mass_kg")),
        "v_nom_l": float(val("mission.displacement_m3")) * 1000.0,
        "n_nom": int(val("mission.ballast")["n"]),
        "n_sea": int(val("uwkit.mission.ballast_sea_SP35")["n"]),
        "bg_m": float(val("mission.BG_m")),
        "theta0_deg": float(trim["pitch"]),
        "cg_x_m": float(val("mission.cg_B_m")[0]),
        "bal_m_kg": float(unit["mass_kg"]),
        "bal_v_l": float(unit["displacement_m3"]) * 1000.0,
        "ang_front_deg": float(angles["front"]),
        "ang_rear_deg": float(angles["rear"]),
        "fx12_astern_n": float(max(wrench12["Fx"])),
        "my12_nm": float(max(wrench12["My"])),
        "thr12_fwd_n": float(curve["12.0"]["fwd_N"]),
        "thr12_rev_n": float(curve["12.0"]["rev_N"]),
        "jaw_tip_x_m": float(sens["gripper_tcp"]["jaw_tip_m"][0]),
        "jaw_opening_mm": float(sens["gripper_tcp"]["jaw_opening_mm"]),
        "sonar_min_range_m": float(sens["sonar_head"]["range_m"][0]),
        "dvl_min_alt_m": float(sens["dvl"]["altitude_m"][0]),
        "depth_range_bar": float(sens["depth"]["range_bar"][1]),
    }
    f["arm_m"] = f["jaw_tip_x_m"] - f["cg_x_m"]
    a_f, a_r = math.radians(f["ang_front_deg"]), math.radians(f["ang_rear_deg"])
    f["axis_sum12_astern_n"] = 2.0 * f["thr12_fwd_n"] * basis._cos(a_f) + 2.0 * f["thr12_rev_n"] * basis._cos(a_r)
    return f


def config_key(ck):
    """CONTRACTS world_ids.config_key: the bid id, then '+' and each option id in ascending code-point order."""
    b, sel = ck
    return "+".join((b,) + tuple(sorted(sel)))


# ================================================================================================ the world
class World:
    """The hidden world (tests/world/*.json, CONTRACTS sec. 7) as the grader sees it: configurations, public basis
    records, the truth table recomputed and checked against truth_table.json, the sets, the Part B labels, the
    delivery bands of every fit configuration (checked: F is a total function), the P tolerances."""

    def __init__(self, files, cfg, basis, facts):
        self.cfg, self.eb, self.facts = cfg, basis, facts
        self.files = files
        for name, schema in WORLD_FILES.items():
            if files[name].get("schema") != schema:
                raise TaskError(f"world/{name}: schema {files[name].get('schema')!r} is not {schema!r}")
        self.rows_of_need = dict(cfg["rows"])                       # need -> row id (hidden)
        self.need_of_row = {r: n for n, r in self.rows_of_need.items()}
        self.row_ids = tuple(cfg["row_order"])
        self.site_key_rows = dict(cfg["site_key_rows"])
        rows = files["rows.json"]
        if dict(rows["policy"]) != dict(basis.POLICY):
            raise TaskError("rows.json policy (TS-01 Part A as printed) differs from evaluation_basis.POLICY")
        if [r["id"] for r in rows["rows"]] != list(self.row_ids):
            raise TaskError("rows.json rows are not the twelve rows in order")
        for r in rows["rows"]:
            if self.rows_of_need.get(r.get("need", self.need_of_row[r["id"]])) != r["id"]:
                raise TaskError(f"rows.json row {r['id']}: need {r.get('need')!r} differs from the grader's row table")
        scale = float(cfg["delta_scale"])
        self.delta = {r["id"]: float(r["delta"]) * scale for r in rows["rows"]}
        if not all(math.isfinite(d) and d > 0.0 for d in self.delta.values()):
            raise TaskError("a tolerance delta is not a positive finite number")
        sc = rows.get("scoring")
        if sc is not None:
            mine = {"weights": cfg["weights"], "award_ramp_B_frac_of_Jstar": cfg["award_ramp_B_frac_of_Jstar"],
                    "full_at": cfg["full_at"], "n01_rules": cfg["n01_rules"],
                    "phi_plus_delta_mult": cfg["phi_plus_delta_mult"], "fail_delta_mult": cfg["fail_delta_mult"]}
            for k, v in sc.items():
                if k == "n01_rules" and isinstance(v, list) and len(v) == 2:     # [Part C rule, Part D rule]
                    v = {"part_c": v[0], "part_d": v[1]}
                if k not in mine or mine[k] != v:
                    raise TaskError(f"rows.json scoring.{k} = {v!r} differs from grader_config.json")
        self.tolerance = rows.get("tolerance") or {}
        # ---------------------------------------------------------------- site, catalogue, bids
        sdoc = files["site.json"]
        self.site = dict(sdoc["site"])
        ab = files["as_built.json"]
        self.cat = files["catalogue.json"]
        self.dr = files["delivery_ranges.json"]
        self.tanks = ab["tanks"]
        self.tank_rho = {k: (float(t["rho"]) if "rho" in t else basis.rho(float(t["t_c"]), 0.0)) for k, t in self.tanks.items()}
        ref = ab["reference"]
        self.eta_ref = ref["eta_loss"] * facts["fx12_astern_n"] / facts["axis_sum12_astern_n"]
        nw = ref["n_works"]
        self.n_works = facts[nw["digest"]] if isinstance(nw, dict) else int(nw)
        bu = self.cat.get("ballast_unit")
        if bu != {"digest": "uwkit.ballast_unit"}:
            raise TaskError(f"catalogue ballast_unit {bu!r} is not the digest's ballast unit")
        self.bids_rec = ab["bids"]
        self.bids = sorted(self.bids_rec)
        if self.bids != sorted(cfg["bids"]):
            raise TaskError(f"world bids {self.bids} differ from the grader's bid list")
        self.cfgs = {}
        for b in self.bids:
            opts = sorted(self.bids_rec[b]["tendered"]["options"])
            if len(opts) != cfg["n_options_per_bid"]:
                raise TaskError(f"bid {b} has {len(opts)} options")
            for r in range(len(opts) + 1):
                for sel in itertools.combinations(opts, r):
                    self.cfgs[(b, sel)] = self._config(b, sel)
        self.key_of = {ck: config_key(ck) for ck in self.cfgs}
        self.ck_of = {k: ck for ck, k in self.key_of.items()}
        self.site_rec = self.site_record()
        pub = sdoc.get("public_site_record")
        if pub is not None and pub != self.site_rec:
            raise TaskError("site.json public_site_record differs from its recomputation")
        self.rho_site = basis.site_density(self.site_rec["water_temp_c"], self.site_rec["working_depth_m"])
        self._veh = {}
        # ---------------------------------------------------------------- truth, sets, labels, bands
        self.truth = self._truth_table()
        self._check_truth(files["truth_table.json"])
        self._sets()
        self._check_sets(files["truth_table.json"].get("sets"))
        self.clause_ids, self.labels = self._labels(files["clause_map.json"])
        self.site_truth = {b: basis.site_values(self.vehicle(ck), self.site_rec) for b, ck in self.cstar.items()}
        self._check_tolerance()
        self.bands = {ck: Band(self, ck) for ck in sorted(self.OK, key=config_key)}
        self._check_bands()

    # ---------------------------------------------------------------- catalogue scalars
    def scalar(self, k):
        sc = self.cat.get("scalars", {})
        return sc[k] if k in sc else self.cat[k]

    def is_reference(self, b):
        m = self.bids_rec[b]["hidden"]["m_air"]
        return isinstance(m, dict) and m.get("derive") == "reference_as_tendered_mass"

    def _eta(self, spec):
        if isinstance(spec, dict):
            if spec.get("derive") != "eta_reference":
                raise TaskError(f"unknown system-coefficient reference {spec!r}")
            return self.eta_ref * float(spec.get("factor", 1.0))
        return float(spec)

    def coefficients(self, b, state, camera_ext):
        """(system coefficient, xu, xuu) of bid b: 'tendered' (site rows, the band) or 'tested' (the test day).  The
        external-camera drag factor applies only while the configuration has the external camera."""
        hb = self.bids_rec[b]["hidden"]
        if state == "tested":
            eta, xu, xuu = self._eta(hb["eta_tested"]), hb["xu_tested"], hb["xuu_tested"]
        else:
            eta, xu, xuu = self._eta(hb["eta"]), hb["xu"], hb["xuu"]
        if not camera_ext and hb.get("k_ext", 1.0) != 1.0:
            xu, xuu = xu / hb["k_ext"], xuu / hb["k_ext"]
        return eta, xu, xuu

    def _config(self, b, sel):
        vb, hb = self.bids_rec[b]["tendered"], self.bids_rec[b]["hidden"]
        slots, variant = dict(vb["slots"]), dict(vb["variant"])
        price = vb["price"]
        for o in sel:
            op = vb["options"][o]
            slots.update(op.get("slots", {}))
            variant.update(op.get("variant", {}))
            price += op["price"]
        if not isinstance(price, int) or isinstance(price, bool):
            raise TaskError(f"price of {b}+{sel} is not an integer")
        built = dict(variant)
        if slots["camera"] == "flat-external" and built.get("cam_housing") == SERIES_CODE:
            built["cam_housing"] = hb["cam_built"]
        f = self.facts
        n_nom = f["n_nom"]
        rho_t = self.tank_rho[hb["tank"]]
        if self.is_reference(b):
            n_fit = self.n_works
            m_air = f["m_nom_kg"] + (n_fit - n_nom) * f["bal_m_kg"]
            vol = f["v_nom_l"] + (n_fit - n_nom) * f["bal_v_l"]
            b_ship = rho_t * vol / 1000.0 - m_air
        else:
            m = hb["m_air"]
            if not (isinstance(m, dict) and m.get("derive") == "digest_mass_plus"):
                raise TaskError(f"bid {b}: unknown air-mass reference {m!r}")
            n_fit = n_nom
            m_air = f["m_nom_kg"] + m["delta_kg"]
            b_ship = hb["b_ship"]
            vol = (m_air + b_ship) / rho_t * 1000.0
        return dict(bid=b, sel=sel, slots=slots, variant=variant, variant_built=built, price=price, m_air=m_air, vol=vol,
                    b_ship=b_ship, rho_tank=rho_t, n_fit=n_fit,
                    n_spare=hb["n_spare"] if slots["ballast"] == "removable" else 0)

    def configs(self):
        return list(self.cfgs)

    def options(self, b):
        return sorted(self.bids_rec[b]["tendered"]["options"])

    def subsets(self, b):
        return [ck for ck in self.cfgs if ck[0] == b]

    def price(self, ck):
        return self.cfgs[ck]["price"]

    def rating_msw(self, slots, variant):
        pr = self.cat["pressure"]
        r = [pr["e_tube"][slots["e_tube"]][variant["e_tube_len"]], pr["foam"][variant["foam"]]]
        if slots["camera"] == "flat-external" and variant["cam_housing"]:
            r.append(pr["cam_housing"][variant["cam_housing"]])
        return min(r)

    def camera(self, slots, variant):
        if slots["camera"] == "dome-internal":
            return "dome", self.cat["camera"]["dome-internal"]["hfov_air"]
        return "flat", float(variant["lens"])

    def thruster_record(self, sku):
        t = self.cat["thrusters"][sku]
        return {"supply_v": list(t["v"]), "thrust_kgf": list(t["fwd_kgf"]), "current_a": list(t["amps"]),
                "speed_rpm": list(t["rpm"]), "reverse_ratio": t["k_rev"], "thrust_power_exp": t["power_exp"],
                "prop_pitch_m": t["pitch_m"], "power_at_1kgf_w": t["w1"]}

    def pack_record(self, code):
        p = self.cat["packs"][code]
        ocv = self.cat["ocv_tables"][p["ocv"]]
        return {"capacity_ah": p["ah"], "capacity_factor": p["cap"], "retention_temp_c": list(p["ret_t"]),
                "retention_frac": list(p["ret_v"]), "r_ss_23c_ohm": p["r23_ohm"], "r_factor_temp_c": list(p["rf_t"]),
                "r_factor": list(p["rf_v"]), "continuous_current_a": p["i_cont_a"], "ocv_soc_frac": list(ocv["soc"]),
                "ocv_v": [x * ocv["series"] for x in ocv["cell_v"]], "ocv_ref_temp_c": ocv["t_ref_c"],
                "ocv_drift_v_per_c": ocv["tc_v_per_c"]}

    def vehicle(self, ck, pack=None, preset=None, n_ballast=None):
        """Public basis record of configuration ck as tendered (the site rows and the delivery band), optionally with
        the band's overrides: pack page code (grade), current preset, ballast pieces fitted at the works."""
        k = (ck, pack, preset, n_ballast)
        if k in self._veh:
            return self._veh[k]
        c = self.cfgs[ck]
        b = c["bid"]
        hb = self.bids_rec[b]["hidden"]
        f = self.facts
        s, vbuilt = c["slots"], c["variant_built"]
        m_air, vol = c["m_air"], c["vol"]
        if n_ballast is not None:
            m_air += (n_ballast - c["n_fit"]) * f["bal_m_kg"]
            vol += (n_ballast - c["n_fit"]) * f["bal_v_l"]
        port, hfov = self.camera(s, vbuilt)
        d_mm, length = s["tether"].split("/")
        eta, xu, xuu = self.coefficients(b, "tendered", s["camera"] == "flat-external")
        veh = {"thruster": self.thruster_record(s["thruster"]), "front_thruster_angle_deg": f["ang_front_deg"],
               "rear_thruster_angle_deg": f["ang_rear_deg"], "thrust_system_coeff": eta,
               "surge_drag_lin_n_per_mps": xu, "surge_drag_quad_n_per_mps2": xuu,
               "pack": self.pack_record(pack or s["pack"]),
               "current_preset_a": hb["i_preset"] if preset is None else float(preset),
               "hotel_load_w": self.scalar("hotel_base_w") + s["lights"] * self.scalar("light_w") + hb["cam_hotel_w"],
               "lights_load_w": self.scalar("light_w") * s["lights"],
               "camera_port": port, "camera_hfov_air_deg": hfov,
               "pressure_part_ratings_msw": [self.rating_msw(s, vbuilt)], "air_mass_kg": m_air, "displaced_volume_l": vol,
               "tank_water_density_kg_m3": c["rho_tank"], "ballast_piece_mass_kg": f["bal_m_kg"],
               "ballast_piece_volume_l": f["bal_v_l"],
               "removable_ballast_pieces": c["n_fit"] if s["ballast"] == "removable" else 0,
               "spare_ballast_pieces": c["n_spare"], "bg_m": f["bg_m"], "static_pitch_trim_deg": f["theta0_deg"],
               "active_pitch_moment_nm": f["my12_nm"], "jaw_reach_m": f["arm_m"], "tether_length_m": float(length),
               "tether_diameter_mm": float(d_mm), "wet_part_min_temps_c": [float(x) for x in self.scalar("wet_min_c").values()],
               "link_test_length_m": self.scalar("link_test_m"), "sonar_min_range_m": f["sonar_min_range_m"],
               "dvl_min_altitude_m": f["dvl_min_alt_m"], "gripper_opening_mm": f["jaw_opening_mm"],
               "depth_sensor_range_bar": f["depth_range_bar"]}
        self._veh[k] = veh
        return veh

    def site_record(self):
        s = self.site
        path = s["el_deck"] - s["el_low"]
        if "routing_m" in s:
            path = path + s["routing_m"]
        return {"water_temp_c": s["t_c"], "working_depth_m": s["el_fsl"] - s["el_low"], "launch_path_m": path,
                "face_reach_m": s["face_reach"], "design_flow_m3s": s["q_insp"], "rack_width_m": s["rack_w"],
                "rack_height_m": s["rack_h"], "open_height_m": s["h_open"], "secchi_m": s["secchi"],
                "min_water_temp_c": s["t_min_c"], "crane_swl_kg": s["swl_kg"], "rigging_kg": s["rigging_kg"],
                "survey_altitude_m": s["survey_alt_m"], "sample_max_mm": s["sample_max_mm"]}

    # ---------------------------------------------------------------- the two named N-01 rules (I-CONS)
    def n01(self, name, veh):
        """N-01 margin [kg] of a vehicle record by the rule's name: n01_trim_reachable (Part C, the unit's own kit)
        or n01_as_delivered (Part D, as ballasted at the works)."""
        f = getattr(self.eb, name)
        if name == "n01_trim_reachable":
            return float(f(veh, self.rho_site))
        if name == "n01_as_delivered":
            return float(f(veh["air_mass_kg"], veh["displaced_volume_l"], self.rho_site))
        raise TaskError(f"unknown N-01 rule {name!r}")

    # ---------------------------------------------------------------- truth table (recomputed; D16)
    def _truth_table(self):
        eb = self.eb
        out = {}
        for ck in self.cfgs:
            veh = self.vehicle(ck)
            m = eb.rows(veh, self.site_rec)
            d = eb.row_details(veh, self.site_rec)
            row = {self.rows_of_need[n]: float(v) for n, v in m.items()}
            row[self.rows_of_need["N-01"]] = self.n01(self.cfg["n01_rules"]["part_c"], veh)   # Part C rule, by name
            row["b_site_kg"] = float(eb.n01_net_buoyancy_kg(veh["air_mass_kg"], veh["displaced_volume_l"], self.rho_site))
            row["binding_w3"] = d["binding_limit"]
            row["strips"] = int(d["strips"])
            if not all(math.isfinite(row[k]) for k in self.row_ids + ("b_site_kg",)):
                raise TaskError(f"configuration {self.key_of[ck]}: a margin is not finite")
            out[ck] = row
        return out

    def _check_truth(self, tt):
        stored = tt.get("configs")
        if not isinstance(stored, dict) or set(stored) != set(self.ck_of):
            raise TaskError("truth_table.json configs are not the 32 configurations")
        bad = []
        for ck, row in self.truth.items():
            s = stored[self.key_of[ck]]
            if set(s) != set(row):
                raise TaskError(f"truth_table.json {self.key_of[ck]}: keys {sorted(s)} differ from {sorted(row)}")
            for k, v in row.items():
                sv = s[k]
                same = (type(sv) is type(v)) and (sv == v) if not isinstance(v, float) else (isinstance(sv, float) and sv == v)
                if not same:
                    bad.append(f"{self.key_of[ck]}.{k}")
        if bad:
            raise TaskError(f"truth_table.json differs from the recomputation in {len(bad)} cells (first: {bad[:4]})")
        canon = {self.key_of[ck]: row for ck, row in self.truth.items()}
        if "sha256" in tt and tt["sha256"] != canonical_digest(canon):
            raise TaskError("truth_table.json sha256 is not the canonical digest of the recomputed configs")

    # ---------------------------------------------------------------- sets (DESIGN 5.2)
    def _sets(self):
        t, dl = self.truth, self.delta
        fm = float(self.cfg["fail_delta_mult"])
        rows = self.row_ids
        self.F = {k: {ck for ck in t if t[ck][k] < -(fm * dl[k])} for k in rows}
        self.OK = {ck for ck in t if not any(ck in self.F[k] for k in rows)}
        self.acc = sorted({ck[0] for ck in self.OK})
        self.rej = [b for b in self.bids if b not in self.acc]
        self.cstar = {b: min((ck for ck in self.OK if ck[0] == b), key=self.price) for b in self.acc}
        self.jstar_cfg = min(self.OK, key=self.price) if self.OK else None
        self.jstar = self.price(self.jstar_cfg) if self.jstar_cfg else None
        self.valid = {b: {k for k in rows if all(t[ck][k] < -(fm * dl[k]) for ck in self.subsets(b))} for b in self.rej}
        self.neutral = {b: {k for k in rows if k not in self.valid[b] and all(t[ck][k] < dl[k] for ck in self.subsets(b))}
                        for b in self.rej}
        if not self.OK:
            raise TaskError("no configuration is fit: the award term has no reference price")

    def summary(self):
        """The truth_table.json `sets` block (the format of -dev/src/uw_req/sets.summary)."""
        order = {k: i for i, k in enumerate(self.row_ids)}
        return dict(ok=sorted(self.key_of[ck] for ck in self.OK), acc=list(self.acc), rej=list(self.rej),
                    cstar={b: self.key_of[ck] for b, ck in self.cstar.items()}, jstar=self.key_of[self.jstar_cfg],
                    valid={b: [self.need_of_row[k] for k in sorted(v, key=order.get)] for b, v in self.valid.items()},
                    neutral={b: [self.need_of_row[k] for k in sorted(v, key=order.get)] for b, v in self.neutral.items()})

    def _check_sets(self, stored):
        if stored is None:
            return
        mine = self.summary()
        for k in mine:
            if stored.get(k) != mine[k]:
                raise TaskError(f"truth_table.json sets.{k} differs from the recomputation")

    # ---------------------------------------------------------------- Part B labels (DESIGN 5.3)
    def _labels(self, cm):
        clauses = cm["clauses"]
        ids = sorted(clauses)
        dl = self.delta
        t = self.truth

        def fails(ck, row):
            return t[ck][row] < -dl[row]
        ok = {ck for ck in t if not any(fails(ck, k) for k in self.row_ids)}
        best = {}
        for ck in sorted(ok, key=config_key):
            if ck[0] not in best or self.price(ck) < self.price(best[ck[0]]):
                best[ck[0]] = ck
        labels = {}
        for cl in ids:
            d = clauses[cl]
            row = d["row"]
            if row is not None and row not in self.row_ids:
                raise TaskError(f"clause_map {cl}: unknown row {row!r}")
            comp = d["compliance"]
            if set(comp) != set(self.ck_of):
                raise TaskError(f"clause_map {cl}: compliance is not given for the 32 configurations")
            c = {ck: comp[self.key_of[ck]]["comply"] for ck in t}
            if not all(isinstance(v, bool) for v in c.values()):
                raise TaskError(f"clause_map {cl}: a compliance flag is not a boolean")
            lax = row is not None and any(c[ck] and fails(ck, row) for ck in t)
            strict = any(not c[ck] for ck in best.values())
            lab = {(False, False): "keep", (True, False): "lax", (False, True): "strict", (True, True): "lax_and_strict"}[(lax, strict)]
            if d.get("label") is not None and d["label"] != lab:
                raise TaskError(f"clause_map {cl}: stored label {d['label']!r} differs from the recomputed {lab!r}")
            labels[cl] = lab
        return ids, labels

    # ---------------------------------------------------------------- P tolerances
    def _check_tolerance(self):
        for b, ck in self.cstar.items():
            tk = self.tolerance.get(self.key_of[ck])
            if not isinstance(tk, dict) or set(tk) != set(self.site_key_rows):
                raise TaskError(f"rows.json tolerance of {self.key_of[ck]} is missing or not the eight site keys")
            for skey, tau in tk.items():
                if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not (math.isfinite(tau) and tau > 0.0):
                    raise TaskError(f"rows.json tolerance {self.key_of[ck]}.{skey} is not a positive number")

    def tau(self, ck, skey):
        return self.tolerance[self.key_of[ck]][skey]

    # ---------------------------------------------------------------- bands (F must be a total function)
    def _check_bands(self):
        audit = self.dr.get("audit") or {}
        for ck, bd in self.bands.items():
            if not bd.phi:
                raise TaskError(f"Phi+ is empty on the band of {self.key_of[ck]}: F would not be a total function")
            a = audit.get(self.key_of[ck])
            if a is not None:
                mine = {"n_units": len(bd.units), "n_phi_plus": len(bd.phi),
                        "n_fail": {k: len(bd.fk[k]) for k in self.cfg["band_rows"]}}
                if {k: a.get(k) for k in mine} != mine:
                    raise TaskError(f"delivery_ranges audit of {self.key_of[ck]} differs from the recomputation")


class Band:
    """The delivery band of one fit configuration (DESIGN 4.5): 11 x 3 x 4 units (n_ballast, grade, preset_a) on its
    own pack family; true W1 (Part D rule), W3, W4 of every unit; Phi+, F_k; facility quantities memoised by the band
    dimension they read (CONTRACTS band.dims_read)."""

    def __init__(self, W, ck):
        self.W, self.ck = W, ck
        b = ck[0]
        rec = W.dr["bids"][b]
        dims = rec["dims"]
        lo, hi = dims["ballast_pieces"]
        if list(dims["grades"]) != list(W.cfg["grades"]):
            raise TaskError(f"delivery_ranges {b}: grades {dims['grades']!r} are not {W.cfg['grades']!r}")
        presets = [float(p) for p in dims["presets_a"]]
        if presets != sorted(presets) or len(set(presets)) != len(presets):
            raise TaskError(f"delivery_ranges {b}: presets are not strictly increasing")
        self.units = [(n, g, p) for n in range(int(lo), int(hi) + 1) for g in dims["grades"] for p in presets]
        gbp = rec["grades_by_pack"].get(W.cfgs[ck]["slots"]["pack"])
        if gbp is None or set(gbp) != set(dims["grades"]):
            raise TaskError(f"no grade family for the pack of {W.key_of[ck]}: F would not be a total function")
        self.gmap = dict(gbp)
        self.envelope = {k: [float(x) for x in rec["envelope"][k]] for k in W.cfg["envelope_keys"]}
        self.n_works = W.cfgs[ck]["n_fit"]
        eb, site = W.eb, W.site_rec
        rule_d = W.cfg["n01_rules"]["part_d"]
        w1 = {}
        w34 = {}
        for n, g, p in self.units:
            if n not in w1:
                v = W.vehicle(ck, pack=self.gmap[g], preset=p, n_ballast=n)
                w1[n] = W.n01(rule_d, v)
            if (g, p) not in w34:
                v = W.vehicle(ck, pack=self.gmap[g], preset=p, n_ballast=n)
                w34[(g, p)] = (float(eb.n03_station_thrust(v, site)), float(eb.n04_energy_reserve(v, site)))
        self.true = {u: {"W1": w1[u[0]], "W3": w34[(u[1], u[2])][0], "W4": w34[(u[1], u[2])][1]} for u in self.units}
        dl = W.delta
        pm, fm = float(W.cfg["phi_plus_delta_mult"]), float(W.cfg["fail_delta_mult"])
        rows = tuple(W.cfg["band_rows"])
        self.phi = [u for u in self.units if all(self.true[u][k] >= pm * dl[k] for k in rows)]
        self.fk = {k: {u for u in self.units if self.true[u][k] < -(fm * dl[k])} for k in rows}
        self.live = [k for k in rows if self.fk[k]]
        self._memo = {}

    @staticmethod
    def dim_key(u, q, cond):
        if q in ("tank_net_buoyancy_n", "air_mass_kg", "displaced_volume_l"):
            return ("n", u[0])
        if q in ("bollard_thrust_n", "bollard_bus_current_a"):
            return ("p", u[2]) if cond["limiter"] == "on" else ("-",)
        if q in ("battery_delivered_energy_wh", "pack_dcir_mohm"):
            return ("g", u[1])
        return ("-",)

    def fac(self, u, q, cond):
        k = (q, tuple(sorted(cond.items())), self.dim_key(u, q, cond))
        if k not in self._memo:
            n, g, p = u
            veh = self.W.vehicle(self.ck, pack=self.gmap[g], preset=p, n_ballast=n)
            try:
                x = self.W.eb.facility_value(q, dict(cond), veh)
            except (ValueError, ArithmeticError, KeyError) as exc:
                raise TaskError(f"facility value {q} {cond!r} cannot be computed: {exc}") from None
            self._memo[k] = float(x)
        return self._memo[k]

    def in_envelope(self, cond):
        for k, v in cond.items():
            if k in self.envelope:
                lo, hi = self.envelope[k]
                if not (lo <= float(v) <= hi):
                    return False
        return True

    def ease_first(self, units):
        """Delivery statement sec. 3: ballast closest to the works practice (fewer pieces on ties), grade C before B
        before A, preset low to high."""
        order = list(reversed(self.W.cfg["grades"]))
        return min(units, key=lambda u: (abs(u[0] - self.n_works), u[0], order.index(u[1]), u[2]))

    def score_f(self, clauses):
        """F on this band (the caller applies award_ok): (F, detail)."""
        if any(not self.in_envelope(c["conditions"]) for c in clauses):
            return 0.0, {"in_envelope": False, "contractable": False, "BA": 0.0, "g": 0.0, "guard": {}, "delivered_fit": None}
        a = set()
        for u in self.units:
            okc = True
            for c in clauses:
                x = self.fac(u, c["quantity"], c["conditions"])
                if (c["comparator"] == ">=" and not x >= c["bound"]) or (c["comparator"] == "<=" and not x <= c["bound"]):
                    okc = False
                    break
            if okc:
                a.add(u)
        if not a:
            return 0.0, {"in_envelope": True, "contractable": False, "BA": 0.0, "g": 0.0, "guard": {}, "delivered_fit": None}
        ba = sum(u in a for u in self.phi) / len(self.phi)
        guard = {k: not (a & self.fk[k]) for k in self.live}
        g = (sum(guard.values()) / len(self.live)) if self.live else 1.0
        u0 = self.ease_first(a)
        fit = all(self.true[u0][k] >= 0.0 for k in self.W.cfg["band_rows"])
        return ba * g, {"in_envelope": True, "contractable": True, "BA": ba, "g": g, "guard": guard, "delivered_fit": fit}


# ================================================================================================ the stage
class Stage:
    """Everything the grader reads from tests/: config, basis, digest facts, schema validators, the world."""

    def __init__(self, stage_dir):
        self.dir = Path(stage_dir)
        cfg = load_trusted_json(self.dir / "grader_config.json")
        if cfg.get("schema") != CONFIG_SCHEMA:
            raise TaskError(f"grader_config.json schema {cfg.get('schema')!r} is not {CONFIG_SCHEMA!r}")
        self.cfg = cfg
        self.basis = load_basis(self.dir / cfg["basis"]["file"], cfg["basis"]["sha256"])
        dg = cfg["digest"]
        self.facts = digest_facts(self.dir / dg["file"], dg["sha256"], dg["version"], self.basis)
        self.schemas = load_trusted_json(self.dir / cfg["schemas_file"])
        wdir = self.dir / cfg["world_dir"]
        files = {}
        for name in WORLD_FILES:
            p = wdir / name
            if not p.is_file():
                raise TaskError(f"world/{name} is missing")
            files[name] = load_trusted_json(p)
        self._check_selection(files["selection.json"])
        self.world = World(files, cfg, self.basis, self.facts)
        self._check_schemas()
        self.validators = {}
        for name in SUBMISSION_FILES:
            sch = self.schemas[SCHEMA_KEY[name]]
            try:
                jsonschema.Draft202012Validator.check_schema(sch)
            except jsonschema.SchemaError as exc:
                raise TaskError(f"schemas.json {SCHEMA_KEY[name]} is not a valid schema: {exc.message}") from None
            self.validators[name] = jsonschema.Draft202012Validator(sch)
        # the array-length gate of G0-3 (ruling O12): every array of the three schemas is bounded, the longest by the
        # clause limit; a schema without a bound would switch the gate off, so that is a task error
        self.array_limit = _array_limit(self.schemas)
        if self.array_limit != int(cfg["max_clauses"]):
            raise TaskError(f"schemas.json: the longest array bound is {self.array_limit}, the clause limit {cfg['max_clauses']}")

    def _check_selection(self, sel):
        """selection.json inputs that name a file of this stage by its path relative to tests/ are checked."""
        for name, sha in (sel.get("inputs") or {}).items():
            p = self.dir / name
            if "/" in name or name.endswith((".json", ".py")):
                if p.is_file() and sha256_file(p) != sha:
                    raise TaskError(f"selection.json pins {name} at sha256 {sha[:16]}..., the file differs")

    def _check_schemas(self):
        """The public schemas must say exactly what the grader enforces: bids, site keys, needs, Part B ids, and per
        quantity the unit, the allowed comparators, the condition signature and enumerations (I-DIR)."""
        cfg, eb, W, S = self.cfg, self.basis, self.world, self.schemas
        ev = S["evaluation"]
        if sorted(ev["properties"]["bids"]["required"]) != W.bids or sorted(ev["properties"]["bids"]["properties"]) != W.bids:
            raise TaskError("schemas.json evaluation bids differ from the world's bids")
        if sorted(ev["$defs"]["accept"]["properties"]["site"]["required"]) != sorted(cfg["site_key_rows"]):
            raise TaskError("schemas.json site keys differ from the grader's site keys")
        if list(ev["$defs"]["reject"]["properties"]["grounds"]["items"]["enum"]) != list(eb.NEEDS):
            raise TaskError("schemas.json grounds differ from the needs of evaluation_basis")
        sr = S["spec_review"]["properties"]["clauses"]
        if sorted(sr["required"]) != W.clause_ids or sorted(sr["properties"]) != W.clause_ids:
            raise TaskError("schemas.json spec_review clause ids differ from the world's Part B")
        cl = S["acceptance_schedule"]["$defs"]["clause"]
        quantities = cl["properties"]["quantity"]["enum"]
        if sorted(quantities) != sorted(eb.FACILITY_QUANTITIES) or sorted(cfg["dir_allowed"]) != sorted(eb.FACILITY_QUANTITIES):
            raise TaskError("schemas.json / grader_config.json quantities differ from evaluation_basis.FACILITY_QUANTITIES")
        if S["acceptance_schedule"]["properties"]["clauses"]["maxItems"] != cfg["max_clauses"]:
            raise TaskError("schemas.json clause limit differs from grader_config.json")
        enums = {"axis": list(eb.AXES), "limiter": list(eb.LIMITER), "medium": list(eb.MEDIUM)}
        seen = set()
        for br in cl["allOf"]:
            q = br["if"]["properties"]["quantity"]["const"]
            seen.add(q)
            th = br["then"]["properties"]
            unit, sig = eb.FACILITY_QUANTITIES[q]
            if th["unit"] != {"const": unit}:
                raise TaskError(f"schemas.json {q}: unit {th['unit']!r} is not {unit!r}")
            if th["comparator"] != {"enum": list(cfg["dir_allowed"][q])}:
                raise TaskError(f"schemas.json {q}: comparators differ from grader_config.json dir_allowed")
            cd = th["conditions"]
            if sorted(cd["required"]) != sorted(sig) or sorted(cd["properties"]) != sorted(sig) or cd.get("additionalProperties") is not False:
                raise TaskError(f"schemas.json {q}: conditions differ from the signature {sig}")
            for name, p in cd["properties"].items():
                want = {"enum": enums[name]} if name in enums else {"type": "number"}
                if p != want:
                    raise TaskError(f"schemas.json {q}.{name}: {p!r} is not {want!r}")
        if seen != set(eb.FACILITY_QUANTITIES):
            raise TaskError("schemas.json has not one branch per quantity")


def load_stage(stage_dir):
    """The stage of tests/ (TaskError for any defect of a task file: a malformed world file is a task error too)."""
    try:
        return Stage(stage_dir)
    except TaskError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, ArithmeticError) as exc:
        raise TaskError(f"malformed task file: {type(exc).__name__}: {exc}") from None
    except OSError as exc:                         # a missing or unreadable stage file (e.g. digest/spec_values.json)
        raise TaskError(f"cannot read a task file: {type(exc).__name__}: {exc}") from None


# ================================================================================================ gate G0
def _array_limit(schemas):
    """The largest maxItems of the three submission schemas; every array-typed node must declare one (else None)."""
    best, todo = 0, [schemas[k] for k in SCHEMA_KEY.values()]
    while todo:
        x = todo.pop()
        if isinstance(x, dict):
            if x.get("type") == "array":
                if not isinstance(x.get("maxItems"), int):
                    return None
                best = max(best, x["maxItems"])
            todo.extend(x.values())
        elif isinstance(x, list):
            todo.extend(x)
    return best


def _shape_gate(doc, name, limit):
    """G0-3 before the schema validator: no array may be longer than the longest one any schema allows.  Such a
    document can never pass the schemas (every array a schema admits has maxItems <= limit; an array anywhere else
    breaks a type or additionalProperties rule), so the verdict is the one the validator would reach; it only spares
    the validator its pairwise uniqueItems fallback (quadratic on an unsortable array) and its per-item error flood."""
    if limit is None:
        return
    todo = [doc]
    while todo:
        x = todo.pop()
        if isinstance(x, list):
            if len(x) > limit:
                raise G0Error(f"{name}: an array has {len(x)} items; the schemas allow at most {limit}")
            todo.extend(x)
        elif isinstance(x, dict):
            todo.extend(x.values())


def load_submission(folder, st):
    """(docs, problems): G0-1 .. G0-3 per file (regular, size, strict JSON, schema; first errors per file), then the
    cross-file rules G0-4 .. G0-9.  docs = {'evaluation', 'spec_review', 'acceptance_schedule'} when problems = []."""
    docs, problems = {}, []
    max_bytes = int(st.cfg["submission"]["max_bytes"])
    for name in SUBMISSION_FILES:
        try:
            data = read_regular_file(os.path.join(folder, name), max_bytes)
            doc = strict_json_bytes(data, name)
            _shape_gate(doc, name, st.array_limit)
        except G0Error as exc:
            problems.append(str(exc))
            continue
        try:
            errs = sorted(st.validators[name].iter_errors(doc), key=lambda e: ([str(x) for x in e.absolute_path], e.message))
        except RecursionError:
            problems.append(f"{name}: nested too deeply")
            continue
        if errs:
            for e in errs[:MAX_PROBLEMS]:
                where = "/".join(str(x) for x in e.absolute_path) or "(top)"
                problems.append(f"{name}: {where}: {' '.join(e.message.split())[:200]}")
            continue
        docs[SCHEMA_KEY[name]] = doc
    if problems:
        return docs, problems
    return docs, g0_rules(docs, st)


def _cond_key(cond):
    return tuple(sorted((k, float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in cond.items()))


def g0_rules(docs, st):
    """G0-4 .. G0-9 on schema-valid documents (CONTRACTS g0_rules)."""
    W, cfg, eb = st.world, st.cfg, st.basis
    probs = []
    ev = docs["evaluation"]
    bids, award = ev["bids"], ev["award"]
    for b in sorted(bids):
        d = bids[b]
        if d["verdict"] == "accept":
            extra = sorted(set(d["options"]) - set(W.options(b)))
            if extra:
                probs.append(f"G0-4: {b}: {extra} are not options of {b}")
    if award is not None and bids[award]["verdict"] != "accept":
        probs.append(f"G0-5: the award {award} is not a bid the submission accepts")
    if sorted(docs["spec_review"]["clauses"]) != W.clause_ids:
        probs.append("G0-6: spec_review.clauses is not exactly the Part B clause ids")
    sc = docs["acceptance_schedule"]
    if award is None:
        if sc["for_bid"] is not None or sc["for_options"] or sc["clauses"]:
            probs.append("G0-7: no award, so for_bid must be null and for_options and clauses empty")
    elif sc["for_bid"] != award or (bids[award]["verdict"] == "accept" and set(sc["for_options"]) != set(bids[award]["options"])):
        probs.append("G0-7: for_bid / for_options do not match the award in evaluation.json")
    ids = [c["id"] for c in sc["clauses"]]
    if len(set(ids)) != len(ids):
        probs.append("G0-8: clause ids are not unique")
    enums = {"axis": eb.AXES, "limiter": eb.LIMITER, "medium": eb.MEDIUM}
    seen = set()
    for c in sc["clauses"]:
        q = c["quantity"]
        unit, sig = eb.FACILITY_QUANTITIES[q]
        cond = c["conditions"]
        if (c["unit"] != unit or c["comparator"] not in cfg["dir_allowed"][q] or set(cond) != set(sig)
                or any(k in enums and cond[k] not in enums[k] for k in cond)
                or any(k not in enums and (isinstance(cond[k], bool) or not isinstance(cond[k], (int, float))) for k in cond)
                or not math.isfinite(float(c["bound"]))):
            probs.append(f"G0-8: clause {c['id']} is not well formed")
            continue
        k = (q, _cond_key(cond), c["comparator"])
        if k in seen:
            probs.append(f"G0-9: clause {c['id']} repeats a (quantity, conditions, comparator)")
        seen.add(k)
    return probs[:MAX_PROBLEMS * 3]


# ================================================================================================ the score
TERM_METRICS = ("verdicts", "award", "site_prediction", "spec_review", "acceptance_schedule", "s_acc", "s_rej",
                "award_ok", "award_g", "acceptance_in_envelope", "acceptance_contractable", "acceptance_ba",
                "acceptance_g", "delivered_unit_fit", "n_accepted", "n_rejected", "n_clauses")


def zero_metrics():
    """The term metrics of a submission that did not pass G0."""
    return {k: 0.0 for k in TERM_METRICS}


def grade(docs, st):
    """(value, metrics) of a submission that passed G0; value = 0.45 V + 0.15 A + 0.15 P + 0.05 Q + 0.20 F."""
    W, cfg = st.world, st.cfg
    w = cfg["weights"]
    bids = docs["evaluation"]["bids"]
    award = docs["evaluation"]["award"]
    exact = {b: bids[b]["verdict"] == "accept" and set(bids[b]["options"]) == set(W.cstar[b][1]) for b in W.acc}
    rb = {}
    for b in W.rej:
        d = bids[b]
        if d["verdict"] != "reject":
            rb[b] = 0.0
            continue
        g = [W.rows_of_need[n] for n in d["grounds"] if W.rows_of_need[n] not in W.neutral[b]]
        rb[b] = 1.0 if g and all(k in W.valid[b] for k in g) else 0.0
    s_acc = sum(exact.values()) / len(exact) if exact else 0.0
    s_rej = sum(rb.values()) / len(rb) if rb else 1.0
    V = s_acc * s_rej
    A, award_ok, ca, G = 0.0, False, None, 0.0
    if award is not None:
        ca = (award, tuple(sorted(bids[award]["options"])))
        own = [W.price((b, tuple(sorted(d["options"])))) for b, d in bids.items() if d["verdict"] == "accept"]
        award_ok = (ca in W.OK) and (W.price(ca) <= min(own))
        if award_ok:
            cheaper = [rb[b] for b in W.rej if W.price((b, ())) < W.price(ca)]
            G = min(cheaper) if cheaper else 1.0
            A = ramp(1.0 - (W.price(ca) - W.jstar) / (cfg["award_ramp_B_frac_of_Jstar"] * W.jstar)) * G
    pl = []
    for b in W.acc:
        if not exact[b]:
            pl.append(0.0)
            continue
        ck = W.cstar[b]
        tv = W.site_truth[b]
        d = bids[b]
        sc = []
        for skey in cfg["site_key_order"]:
            tau = W.tau(ck, skey)
            e = abs(float(d["site"][skey]) - tv[skey])
            sc.append(ramp((5.0 * tau - e) / (4.0 * tau)))
        pl.append(sum(sc) / len(sc))
    P = (s_rej * (sum(pl) / len(pl) if pl else 0.0)) if V > 0 else 0.0
    Q = 0.0
    if V > 0:
        lab = docs["spec_review"]["clauses"]
        n_def = sum(v != "keep" for v in W.labels.values())
        tp = sum(lab[c] == v and v != "keep" for c, v in W.labels.items())
        fp = sum(lab[c] != "keep" and lab[c] != v for c, v in W.labels.items())
        Q = max(0, tp - fp) / n_def if n_def else 1.0
    F, fdet = 0.0, None
    if w["F"] > 0 and award_ok and cfg["f_gate"] == "c_a in OK":
        F, fdet = W.bands[ca].score_f(docs["acceptance_schedule"]["clauses"])
    total = w["V"] * V + w["A"] * A + w["P"] * P + w["Q"] * Q + w["F"] * F
    metrics = {"gate_g0_structure": 1.0, "verdicts": V, "award": A, "site_prediction": P, "spec_review": Q,
               "acceptance_schedule": F, "s_acc": s_acc, "s_rej": s_rej, "award_ok": float(award_ok), "award_g": float(G),
               "acceptance_in_envelope": float(bool(fdet and fdet["in_envelope"])),
               "acceptance_contractable": float(bool(fdet and fdet["contractable"])),
               "acceptance_ba": float(fdet["BA"]) if fdet else 0.0, "acceptance_g": float(fdet["g"]) if fdet else 0.0,
               "delivered_unit_fit": float(bool(fdet and fdet["delivered_fit"])),
               "n_accepted": float(sum(d["verdict"] == "accept" for d in bids.values())),
               "n_rejected": float(sum(d["verdict"] == "reject" for d in bids.values())),
               "n_clauses": float(len(docs["acceptance_schedule"]["clauses"]))}
    if fdet:
        for k, held in fdet["guard"].items():
            metrics["acceptance_guard:" + W.need_of_row[k]] = float(held)
    check_metrics(metrics, cfg)
    return total, metrics


def check_metrics(metrics, cfg):
    """Every metric key is whitelisted (no per-bid truth, J*, site condition or Phi+ in the record) and finite."""
    allowed = set(cfg["metrics_whitelist"])
    bad = sorted(k for k in metrics if k not in allowed)
    if bad:
        raise TaskError(f"metric keys outside the whitelist: {bad}")
    for k, v in metrics.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise TaskError(f"metric {k} is not a finite number")
