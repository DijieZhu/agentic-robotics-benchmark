"""Drive parameters of the owner's car: load, derive, check.

`drive_params.json` beside this file is the one parameter set of the car (no hidden variants). It has two kinds of
entries:

* chosen values (`suspension.wheel_rate_npm`, `tyre.slip_angle.cornering_stiffness_per_rad`, ...): set by hand, each
  with its source in `provenance`;
* derived values (blocks whose key starts with `derived`, and the frozen key `motor_nameplate`'s computed lines): a
  pure function of the chosen values and of the shared inputs (the car's shared specification values (v2),
  `car_parts_manifest.json`). `python -m sketch.car.params --check` recomputes them and fails on any difference;
  `--write` rewrites them in place. Measured values (`measured`, `top_speed_mps`) are copied from the gates' table
  `calibration_table.json` (`sketch.car.gates`); neither the derivation nor the gates write them automatically.

Standard library and numpy only; no MuJoCo here.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "drive_params.json")
DEV = os.path.dirname(os.path.dirname(HERE))                              # the folder that holds sketch/ (import root)
SHARED = os.path.abspath(os.path.join(DEV, "..", "..", "shared"))         # the car's shared inputs
SPEC_VALUES = os.path.join(SHARED, "spec_values.json")
MANIFEST = os.path.join(SHARED, "car_parts_manifest.json")
G = 9.81
WHEELS = ("fl", "fr", "rl", "rr")
RPM = 2.0 * math.pi / 60.0

# Parts of the 120-part table that are unsprung, with the fraction carried by the wheel side and the body of the car
# model that carries them (`wheel` turns with the wheel; `knuckle` steers; `corner` moves with the suspension only).
# A link between the chassis and a wheel (arm, damper, drive shaft, tie rod) is split half and half.
UNSPRUNG = {
    "fl": [("wheel_fl", 1.0, "wheel"), ("wheel_hex_fl", 1.0, "wheel"), ("wheel_nut_fl", 1.0, "wheel"),
           ("front_wheel_axle_l", 1.0, "wheel"), ("front_knuckle_l", 1.0, "knuckle"),
           ("front_lower_arm_l", 0.5, "corner"), ("front_upper_arm_l", 0.5, "corner"), ("damper_fl", 0.5, "corner"),
           ("front_drive_shaft_l", 0.5, "corner"), ("tie_rod_l", 0.5, "corner")],
    "fr": [("wheel_fr", 1.0, "wheel"), ("wheel_hex_fr", 1.0, "wheel"), ("wheel_nut_fr", 1.0, "wheel"),
           ("front_wheel_axle_r", 1.0, "wheel"), ("front_knuckle_r", 1.0, "knuckle"),
           ("front_lower_arm_r", 0.5, "corner"), ("front_upper_arm_r", 0.5, "corner"), ("damper_fr", 0.5, "corner"),
           ("front_drive_shaft_r", 0.5, "corner"), ("tie_rod_r", 0.5, "corner")],
    "rl": [("wheel_rl", 1.0, "wheel"), ("wheel_hex_rl", 1.0, "wheel"), ("wheel_nut_rl", 1.0, "wheel"),
           ("rear_wheel_axle_l", 1.0, "wheel"), ("rear_upright_l", 1.0, "corner"),
           ("rear_lower_arm_l", 0.5, "corner"), ("rear_upper_arm_l", 0.5, "corner"), ("damper_rl", 0.5, "corner"),
           ("rear_drive_shaft_l", 0.5, "corner")],
    "rr": [("wheel_rr", 1.0, "wheel"), ("wheel_hex_rr", 1.0, "wheel"), ("wheel_nut_rr", 1.0, "wheel"),
           ("rear_wheel_axle_r", 1.0, "wheel"), ("rear_upright_r", 1.0, "corner"),
           ("rear_lower_arm_r", 0.5, "corner"), ("rear_upper_arm_r", 0.5, "corner"), ("damper_rr", 0.5, "corner"),
           ("rear_drive_shaft_r", 0.5, "corner")],
}
# Parts whose bounding box (leads, plugs, mounting ears) is a poor stand-in for the body that carries the mass: the
# size of the body itself, mm, x y z.
OWN_BOX_MM = {"battery_pack": (131.4, 46.4, 23.9), "motor_540": (54.5, 35.7, 35.7), "esc": (39.5, 36.5, 16.2),
              "steering_servo": (38.1, 55.4, 20.0)}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_params(path=None):
    """The drive parameters as a dict (a fresh copy on every call)."""
    return load_json(path or PARAMS_PATH)


def params_digest(params):
    """sha256 of the canonical text of a parameter set; stored in every model built from it."""
    text = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


# ------------------------------------------------------------------------------------------------------- geometry
def geometry_from_spec(spec_values):
    """Car geometry in the car frame C, metres, from `spec_values.json` (v2) and the owner model's digest."""
    v = spec_values["values"]
    wb = v["wheelbase_mm"] * 1e-3
    tf, tr = v["track_mm"]["front"] * 1e-3, v["track_mm"]["rear"] * 1e-3
    r = 0.5 * v["tyre_od_mm"] * 1e-3
    return {
        "wheelbase_m": wb,
        "track_front_m": tf,
        "track_rear_m": tr,
        "tyre_radius_m": r,
        "steer_lock_rad": round(math.radians(v["steer_lock_deg"]), 4),
        "min_turn_radius_m": round(wb / math.tan(round(math.radians(v["steer_lock_deg"]), 4)), 4),
        "wheel_centres_c": {"fl": [wb, 0.5 * tf, r], "fr": [wb, -0.5 * tf, r],
                            "rl": [0.0, 0.5 * tr, r], "rr": [0.0, -0.5 * tr, r]},
        "mass_kg": v["mass_account"]["total_g"] * 1e-3,
        "com_c": [x * 1e-3 for x in v["com_mm"]],
        "final_drive_ratio": v["final_drive_ratio"],
    }


def _box_inertia(m, size):
    a, b, c = size
    return m / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])


def mass_properties(manifest, geometry):
    """Sprung chassis and the unsprung bodies of each corner from the 120-part table.

    Every unsprung body has its centre of mass at the wheel centre (the parts sit within 25 mm of it); the chassis takes
    the rest of the mass at the point that makes the whole car's centre of mass exactly `geometry.com_c`. Inertia of a
    part = a uniform box of its bounding box (or of `OWN_BOX_MM`) at its own centre of mass, summed with the parallel
    axis theorem."""
    parts = {c["id"]: c for c in manifest["components"]}
    total = sum(c["mass_g"] for c in manifest["components"]) * 1e-3
    frac = {}
    corners = {}
    for w in WHEELS:
        body = {"corner": 0.0, "knuckle": 0.0, "wheel": 0.0}
        for pid, f, where in UNSPRUNG[w]:
            if pid not in parts:
                raise KeyError("part %r of corner %s is not in the manifest" % (pid, w))
            frac[pid] = frac.get(pid, 0.0) + f
            body[where] += f * parts[pid]["mass_g"] * 1e-3
        corners[w] = body
    m_unsprung = sum(sum(b.values()) for b in corners.values())
    m_sprung = total - m_unsprung
    com = np.asarray(geometry["com_c"], dtype=float)
    first = total * com
    for w in WHEELS:
        first = first - sum(corners[w].values()) * np.asarray(geometry["wheel_centres_c"][w], dtype=float)
    c_s = first / m_sprung
    # inertia of the sprung parts about the sprung centre of mass
    inertia = np.zeros((3, 3))
    m_check = 0.0
    for c in manifest["components"]:
        f = 1.0 - frac.get(c["id"], 0.0)
        if f <= 1e-12:
            continue
        mm = f * c["mass_g"] * 1e-3
        m_check += mm
        lo, hi = np.asarray(c["bbox_mm"][0], dtype=float), np.asarray(c["bbox_mm"][1], dtype=float)
        size = (np.asarray(OWN_BOX_MM[c["id"]], dtype=float) if c["id"] in OWN_BOX_MM else hi - lo) * 1e-3
        dvec = np.asarray(c["com_mm"], dtype=float) * 1e-3 - c_s
        inertia += _box_inertia(mm, size) + mm * (float(dvec @ dvec) * np.eye(3) - np.outer(dvec, dvec))
    if abs(m_check - m_sprung) > 1e-9:
        raise ValueError("sprung mass mismatch %.9f / %.9f" % (m_check, m_sprung))
    full = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    return {
        "total_kg": round(total, 6),
        "sprung_kg": round(m_sprung, 6),
        "sprung_com_c": [round(float(x), 6) for x in c_s],
        "sprung_fullinertia_kgm2": [float("%.6e" % x) for x in full],
        "unsprung_kg": {w: {k: round(v, 6) for k, v in corners[w].items()} for w in WHEELS},
        "unsprung_total_kg": round(m_unsprung, 6),
    }


def static_loads(geometry, mass):
    """Normal force of each wheel at rest (N) and the spring force each corner's suspension carries."""
    m = mass["total_kg"]
    com = geometry["com_c"]
    wb = geometry["wheelbase_m"]
    front = m * G * com[0] / wb
    rear = m * G - front
    tf, tr = geometry["track_front_m"], geometry["track_rear_m"]
    n = {"fl": front * (0.5 + com[1] / tf), "fr": front * (0.5 - com[1] / tf),
         "rl": rear * (0.5 + com[1] / tr), "rr": rear * (0.5 - com[1] / tr)}
    spring = {w: n[w] - sum(mass["unsprung_kg"][w].values()) * G for w in WHEELS}
    return {"normal_n": {w: round(n[w], 6) for w in WHEELS},
            "spring_n": {w: round(spring[w], 6) for w in WHEELS},
            "front_share": round(front / (m * G), 6)}


# ------------------------------------------------------------------------------------------------------- steering
def ackermann_angles(delta, wheelbase, track):
    """Exact Ackermann angles (left wheel, right wheel) for the bicycle angle `delta` (rad, left positive); the
    turning centre lies on the rear axle line, the kingpins at the wheel centres."""
    delta = np.asarray(delta, dtype=float)
    t = np.tan(delta)
    left = np.arctan2(wheelbase * t, wheelbase - 0.5 * track * t)
    right = np.arctan2(wheelbase * t, wheelbase + 0.5 * track * t)
    # arctan2 keeps the sign of tan(delta) in the numerator; for negative delta the denominators stay positive
    return left, right


def ackermann_polycoef(wheelbase, track, fit_range_rad, lock_rad):
    """Quartic `polycoef` of the two kingpin equalities (MuJoCo: q_wheel = c0 + c1 q + ... + c4 q^4 about qpos0 = 0),
    least squares over +-fit_range; and the largest error over +-lock."""
    x = np.linspace(-fit_range_rad, fit_range_rad, 4001)
    left, right = ackermann_angles(x, wheelbase, track)
    out = {}
    for name, y in (("fl", left), ("fr", right)):
        a = np.vander(x, 5, increasing=True)
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        xs = np.linspace(-lock_rad, lock_rad, 2001)
        ys = ackermann_angles(xs, wheelbase, track)[0 if name == "fl" else 1]
        err = float(np.abs(np.vander(xs, 5, increasing=True) @ coef - ys).max())
        out[name] = {"polycoef": [float("%.10g" % c) for c in coef], "max_error_rad": float("%.3e" % err)}
    return out


# ------------------------------------------------------------------------------------------------------- motor, drive
def motor_derived(nameplate, battery, controller, wiring_ohm):
    kv = nameplate["kv_rpm_per_v"] * RPM                 # rad/s per volt
    ke = 1.0 / kv                                         # V per rad/s = N m per A
    r_total = nameplate["resistance_ohm"] + battery["cells"] * battery["cell_resistance_ohm"] \
        + controller["on_resistance_ohm"] + wiring_ohm
    v = battery["voltage_v"]
    i0 = nameplate["no_load_current_a"]
    w0 = (v - i0 * r_total) / ke
    return {"ke_v_per_radps": float("%.8g" % ke), "kt_nm_per_a": float("%.8g" % ke),
            "circuit_resistance_ohm": round(r_total, 6),
            "no_load_speed_radps": round(w0, 3), "no_load_speed_rpm": round(w0 / RPM, 1),
            "stall_current_a": round(v / r_total, 3),
            "stall_torque_nm": round(ke * (v / r_total - i0), 6),
            "current_limited_torque_nm": round(ke * (controller["current_limit_a"] - i0), 6),
            "loss_torque_nm": round(ke * i0, 6)}


def drivetrain_derived(geometry, mass, drive, motor, controller, loads=None):
    """Tendon armature, losses, the speed-loop gains (pole placement on the rigid-rolling plant) and the centre
    coupling.

    Centre: the propeller shaft locks the mean spin of the front axle to that of the rear. The friction cone has no
    longitudinal slip stiffness, so a hard lock would make the tyres slide whenever the two axles travel different
    distances (every corner). Real tyres take that mismatch by slip: binding force = C_k eps N_f N_r / (N_f + N_r)
    for a speed mismatch eps. On the coordinate `centre` (front mean minus rear mean spin angle) this is a damper
    c = C_k N_series r^2 / v, which CarRuntime schedules with the forward speed (v >= the floor of centre_schedule)."""
    gr = geometry["final_drive_ratio"]
    r = geometry["tyre_radius_m"]
    if loads is not None:
        n = loads["normal_n"]
        nf, nr = n["fl"] + n["fr"], n["rl"] + n["rr"]
        n_series = nf * nr / (nf + nr)
    else:
        n_series = 0.25 * mass["total_kg"] * G
    c_coef = drive["centre_slip_stiffness_per_unit"] * n_series * r * r      # N m s / rad times m/s
    # rotor and differential carriers, reflected to the mean spin of the four wheels; MuJoCo puts a fixed tendon's
    # armature only on the diagonal of the mass matrix for joints in different branches (measured), so the inertia is
    # given to the four spin joints as armature, a quarter each: the driving mode then carries exactly this inertia
    armature = drive["rotor_inertia_kgm2"] * gr * gr + drive["diff_carrier_inertia_kgm2"] * 2.0
    # inertia on the mean-wheel coordinate when all four wheels roll without slip
    i_spin = 4.0 * drive["wheel_spin_inertia_kgm2"]
    j_roll = armature + i_spin + mass["total_kg"] * r * r
    b = gr * gr * drive["efficiency"] * motor["kt_nm_per_a"] / j_roll           # (rad/s^2 at the motor) per A
    wc = 2.0 * math.pi * controller["speed_loop_bandwidth_hz"]
    kp = wc / b
    ki = kp * wc / controller["speed_loop_ti_ratio"]
    j_free = armature + i_spin
    return {"driveline_inertia_kgm2": float("%.6e" % armature), "spin_armature_kgm2": float("%.6e" % (0.25 * armature)),
            "tendon_frictionloss_nm": round(motor["loss_torque_nm"] * gr, 6),
            "mean_wheel_inertia_rolling_kgm2": float("%.6e" % j_roll),
            "mean_wheel_inertia_free_kgm2": float("%.6e" % j_free),
            "speed_kp_a_per_radps": float("%.6g" % kp), "speed_ki_a_per_rad": float("%.6g" % ki),
            "loop_gain_free_x_dt": None,
            "centre_n_series_n": round(n_series, 6), "centre_damping_coef_nm_per_rad": float("%.6g" % c_coef)}


def centre_schedule(drive_derived, drive, dt):
    """Speed floor of the centre damping schedule c = coef / max(|v|, v_lo): the largest damping keeps
    c dt / J_centre <= centre_stability_ratio, J_centre = the centre mode's inertia (a wheel's spin inertia plus its
    armature; measured: at c dt / J = 5 a launch on grip 0.15 diverged at 2.5 ms, at 1.0 it is stable at 2.5 and
    1.25 ms). Below v_lo the binding of the propeller shaft is weaker than the tyre-slip law."""
    j_centre = drive["wheel_spin_inertia_kgm2"] + drive_derived["spin_armature_kgm2"]
    c_max = drive["centre_stability_ratio"] * j_centre / dt
    v_lo = drive_derived["centre_damping_coef_nm_per_rad"] / c_max
    return {"centre_mode_inertia_kgm2": float("%.6e" % j_centre), "centre_low_speed_mps": round(v_lo, 4),
            "centre_damping_rest_nms": float("%.6g" % c_max)}


def tyre_derived(geometry, loads, slip, bodies, dt):
    """Stiffness and damping of the slip pivot of each corner: k = C_S x N0 x e (so that F_y = C_S N0 alpha in the
    linear range), damping for the ratio `pivot_damping_ratio` on the pivot's yaw inertia (the wheel about the pivot
    axis: its transverse inertia plus m e^2)."""
    e = slip["pivot_offset_m"]
    out = {}
    for w in WHEELS:
        axle = "front" if w[0] == "f" else "rear"
        cs = slip["cornering_stiffness_per_rad"][axle]
        n0 = loads["normal_n"][w]
        k = cs * n0 * e
        wb = bodies["corners"][w]["wheel"]
        i_p = wb["diaginertia_kgm2"][2] + wb["mass_kg"] * e * e
        c = 2.0 * slip["pivot_damping_ratio"] * math.sqrt(k * i_p)
        out[w] = {"stiffness_nm_per_rad": round(k, 6), "damping_nms_per_rad": float("%.6g" % c),
                  "yaw_inertia_kgm2": float("%.6e" % i_p), "omega_x_dt": round(math.sqrt(k / i_p) * dt, 4)}
    return out


def body_inertials(geometry, mass, params):
    """Mass, centre of mass and inertia of every body of the car model (car frame C, design pose).

    The rack (the bicycle steering joint) and, in the slip variant, the four pivot carriers take their mass out of
    the chassis and the knuckle / corner, so that the whole car keeps exactly the mass and centre of mass of the
    owner's model in both tyre variants (a carrier's centre of mass sits at its wheel centre)."""
    wb = geometry["wheelbase_m"]
    r = geometry["tyre_radius_m"]
    st, dr, sl = params["steering"], params["drivetrain"], params["tyre"]["slip_angle"]
    m_rack = st["rack_mass_kg"]
    p_rack = np.array([wb, 0.0, r])
    m_s = mass["sprung_kg"]
    c_s = np.asarray(mass["sprung_com_c"], dtype=float)
    ixx, iyy, izz, ixy, ixz, iyz = mass["sprung_fullinertia_kgm2"]
    i_s = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    m_c = m_s - m_rack
    c_c = (m_s * c_s - m_rack * p_rack) / m_c
    # inertia of the chassis about c_c: the sprung inertia moved to c_c minus the rack as a point mass
    d0 = c_s - c_c
    i_about_cc = i_s + m_s * (float(d0 @ d0) * np.eye(3) - np.outer(d0, d0))
    d1 = p_rack - c_c
    i_c = i_about_cc - m_rack * (float(d1 @ d1) * np.eye(3) - np.outer(d1, d1))
    out = {"chassis": {"mass_kg": round(m_c, 6), "com_c": [round(float(x), 7) for x in c_c],
                       "fullinertia_kgm2": [float("%.6e" % x) for x in (i_c[0, 0], i_c[1, 1], i_c[2, 2],
                                                                         i_c[0, 1], i_c[0, 2], i_c[1, 2])]},
           "rack": {"mass_kg": m_rack, "pos_c": [wb, 0.0, r], "diaginertia_kgm2": [1e-7, 1e-7, 1e-7]},
           "corners": {}}
    m_carrier = sl["carrier_mass_kg"]
    for w in WHEELS:
        u = mass["unsprung_kg"][w]
        front = w[0] == "f"
        steer_body = "knuckle" if front else "corner"
        m_steer = u[steer_body] - m_carrier            # the carrier's mass comes out of the knuckle (front) / corner
        corner = {"wheel": {"mass_kg": round(u["wheel"], 6),
                            "diaginertia_kgm2": [dr["wheel_transverse_inertia_kgm2"], dr["wheel_spin_inertia_kgm2"],
                                                 dr["wheel_transverse_inertia_kgm2"]]},
                  "carrier": {"mass_kg": m_carrier, "diaginertia_kgm2": [2e-7, 2e-7, 2e-7]}}
        if front:
            corner["corner"] = {"mass_kg": round(u["corner"], 6), "diaginertia_kgm2": [4e-6, 4e-6, 4e-6]}
            corner["knuckle"] = {"mass_kg": round(m_steer, 6), "diaginertia_kgm2": [2e-6, 2e-6, 2e-6]}
        else:
            corner["corner"] = {"mass_kg": round(m_steer, 6), "diaginertia_kgm2": [4e-6, 4e-6, 4e-6]}
        out["corners"][w] = corner
    return out


def suspension_derived(susp, loads):
    out = {}
    for w in WHEELS:
        k = susp["wheel_rate_npm"]
        out[w] = {"springref_m": round(-loads["spring_n"][w] / k, 7), "static_compression_m": round(loads["spring_n"][w] / k, 7)}
    return out


def derive(params, spec_values, manifest):
    """The derived blocks of `params` (a new dict; `params` is not changed)."""
    p = copy.deepcopy(params)
    geo = geometry_from_spec(spec_values)
    mass = mass_properties(manifest, geo)
    loads = static_loads(geo, mass)
    st = p["steering"]
    ack = ackermann_polycoef(geo["wheelbase_m"], geo["track_front_m"], st["fit_range_rad"], geo["steer_lock_rad"])
    np_ = p["motor_nameplate"]
    motor = motor_derived(np_, p["battery"], p["controller"], p["controller"]["wiring_resistance_ohm"])
    drive = drivetrain_derived(geo, mass, p["drivetrain"], motor, p["controller"], loads)
    drive.update(centre_schedule(drive, p["drivetrain"], p["physics_dt_s"]))
    drive["loop_gain_free_x_dt"] = round(
        drive["speed_kp_a_per_radps"] * geo["final_drive_ratio"] ** 2 * p["drivetrain"]["efficiency"]
        * motor["kt_nm_per_a"] / drive["mean_wheel_inertia_free_kgm2"] * p["physics_dt_s"], 4)
    bodies = body_inertials(geo, mass, p)
    p["derived_geometry"] = geo
    p["derived_mass"] = mass
    p["derived_bodies"] = bodies
    p["derived_static_loads"] = loads
    p["derived_steering"] = ack
    p["derived_motor"] = motor
    p["derived_drivetrain"] = drive
    p["derived_suspension"] = suspension_derived(p["suspension"], loads)
    p["derived_slip_pivot"] = tyre_derived(geo, loads, p["tyre"]["slip_angle"], bodies, p["physics_dt_s"])
    p["inputs"] = {"spec_values_sha256": sha256_file(SPEC_VALUES), "manifest_sha256": sha256_file(MANIFEST)}
    return p


def check(params=None, path=None):
    """Differences between the derived blocks in the file and a fresh derivation: a list of strings (empty = ok)."""
    params = params if params is not None else load_params(path)
    fresh = derive(params, load_json(SPEC_VALUES), load_json(MANIFEST))
    diffs = []
    for k in sorted(set(fresh) | set(params)):
        if k.startswith("derived") or k == "inputs":
            if json.dumps(fresh.get(k), sort_keys=True) != json.dumps(params.get(k), sort_keys=True):
                diffs.append(k)
    return diffs


def write(path=None):
    path = path or PARAMS_PATH
    params = load_params(path)
    fresh = derive(params, load_json(SPEC_VALUES), load_json(MANIFEST))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fresh, f, indent=1, sort_keys=False)
        f.write("\n")
    return fresh


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if a.write:
        write()
        print("written", PARAMS_PATH)
    diffs = check()
    print("derived blocks differing: %s" % (", ".join(diffs) if diffs else "none"))
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
