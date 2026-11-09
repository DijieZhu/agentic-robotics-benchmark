"""Contractual bench sizing procedure. All instance and vendor values are inputs.

SI units throughout. This is a reduced-order acceptance model, not a flight simulator.
"""
from itertools import product
import math
import numpy as np

STATIONS = ("FL", "FR", "RL", "RR")


def candidates(cat):
    for motors in product(sorted(cat["motors"]), repeat=4):
        for prop, esc, battery in product(*[sorted(cat[k]) for k in ("props", "escs", "batteries")]):
            yield {"motors": dict(zip(STATIONS, motors)), "propeller": prop,
                   "esc": esc, "battery": battery}


def price(bom, cat):
    return sum(cat["motors"][bom["motors"][s]]["usd"] for s in STATIONS) + sum(
        cat[k][bom[r]]["usd"] * n for k, r, n in
        (("props", "propeller", 4), ("escs", "esc", 1), ("batteries", "battery", 1)))


def line_changes(a, b):
    return sum(a["motors"][s] != b["motors"][s] for s in STATIONS) + sum(
        a[k] != b[k] for k in ("propeller", "esc", "battery"))


def evaluate(bom, geometry, cat, duty):
    """Evaluate the selected hardware on the supplied as-built geometry and mass ledger.

    geometry: dimensions_mm, fixed_components (mass_kg, position_mm, box_mm),
    payload_mass_kg. Battery, payload and ESC centres are geometry dimensions.
    All fixed and purchased components are uniform boxes; motors/props are point masses.
    """
    d = geometry["dimensions_mm"]
    positions = np.array([[d["motor_x_mm"], d["motor_y_mm"], 0],
                          [d["motor_x_mm"], -d["motor_y_mm"], 0],
                          [-d["motor_x_mm"], d["motor_y_mm"], 0],
                          [-d["motor_x_mm"], -d["motor_y_mm"], 0]]) / 1000
    motors = [cat["motors"][bom["motors"][s]] for s in STATIONS]
    prop, esc, battery = (cat[k][bom[r]] for k, r in
                         (("props", "propeller"), ("escs", "esc"), ("batteries", "battery")))
    components = [(c["mass_kg"], np.array(c["position_mm"]) / 1000,
                   np.array(c["box_mm"]) / 1000) for c in geometry["fixed_components"]]
    for component, mass, box in (("battery", battery["mass_kg"], battery["box_mm"]),
                                ("payload", geometry["payload_mass_kg"], [d[f"payload_box_{axis}_mm"] for axis in "xyz"]),
                                ("esc", esc["mass_kg"], esc["box_mm"])):
        components.append((mass, np.array([d[f"{component}_{axis}_mm"] for axis in "xyz"]) / 1000,
                           np.array(box) / 1000))
    components += [(m["mass_kg"] + prop["mass_kg"], p, np.zeros(3)) for m, p in zip(motors, positions)]
    mass = sum(m for m, _, _ in components)
    cg = sum(m * p for m, p, _ in components) / mass
    inertia = np.zeros((3, 3))
    for m, p, box in components:
        r = p - cg
        inertia += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        inertia += np.diag(m / 12 * (np.sum(box * box) - box * box))
    spins = np.array(duty["reaction_signs"])
    arm = prop["reaction_arm_m"]
    radii = np.linalg.norm(positions[:, :2], axis=1)
    tangent = np.column_stack((-positions[:, 1]/radii, positions[:, 0]/radii, np.zeros(4)))
    angles = np.radians([geometry['rotor_cant_deg'][station] for station in STATIONS])
    radial_angles = np.radians([geometry['rotor_radial_tilt_deg'][station] for station in STATIONS])
    radial = positions/radii[:,None]
    normals = -spins[:, None]*np.sin(angles)[:, None]*tangent
    normals[:, 2] = np.cos(angles)
    normals = np.cos(radial_angles)[:,None]*normals + np.sin(radial_angles)[:,None]*radial
    moment_map = (np.cross(positions-cg, normals) + spins[:, None]*arm*normals).T
    allocation = np.vstack((normals[:, 2], moment_map))
    scaled_allocation = allocation.copy()
    scaled_allocation[1:] /= radii.mean()
    allocation_condition = float(np.linalg.cond(scaled_allocation))
    unit_z_thrust = np.linalg.solve(allocation, [1., 0., 0., 0.])
    unit_force = normals.T @ unit_z_thrust
    states=duty["states"]
    times=np.array([state["seconds"] for state in states])
    gust=np.array([state["gust_n"] for state in states])
    ax=np.array([state["ax_m_s2"] for state in states])
    az=np.array([state["az_m_s2"] for state in states])
    force=np.sqrt((mass*(duty["g"]+az))**2+(mass*ax-gust[:,0])**2+gust[:,1]**2)
    omega=np.array([state["omega_rad_s"] for state in states])
    alpha=np.array([state["alpha_rad_s2"] for state in states])
    gust_point=np.array([d[f"payload_{axis}_mm"] for axis in "xyz"])/1000
    angular_demand=alpha@inertia.T+np.cross(omega,omega@inertia.T)-np.cross(gust_point-cg,gust)
    particular_thrust=np.linalg.solve(allocation,np.column_stack((np.zeros(len(states)),angular_demand)).T).T
    particular_force=particular_thrust@normals
    qa=float(unit_force@unit_force)
    qb=2*particular_force@unit_force
    qc=np.sum(particular_force*particular_force,axis=1)-force**2
    force_discriminant=qb*qb-4*qa*qc
    body_z_force=(-qb+np.sqrt(np.maximum(force_discriminant,0.)))/(2*qa)
    t=particular_thrust+body_z_force[:,None]*unit_z_thrust
    body_force=t@normals
    moments=angular_demand+np.cross(cg,body_force)
    feasible=bool(np.all(force_discriminant>=0) and np.all(body_z_force>0) and np.all(t>=0))
    # Apply the identical per-state bench equations as a batch; no interpolation.
    coefficients=np.array([m["power_coeff"]*cat["lots"][bom["motors"][s]]["power_factor"]
                           *prop["power_factor"] for m,s in zip(motors,STATIONS)])
    motor_power=coefficients*np.maximum(t,0)**1.5
    w=motor_power.sum(axis=1)/esc["efficiency"]+duty["aux_power_w"]
    voc=battery["test_ocv_v"]
    resistance=battery["resistance_ohm"]*cat["battery_lot"][bom["battery"]]["resistance_factor"]
    discriminant=voc*voc-4*resistance*w
    feasible &= bool(np.all(discriminant>0))
    v=(voc+np.sqrt(np.maximum(discriminant,0.)))/2
    a=motor_power/esc["efficiency"]/v[:,None]
    battery_current = w / v
    capacity = np.array([m["thrust_ref_n"] * cat["lots"][bom["motors"][s]]["thrust_factor"]
                         * prop["thrust_factor"] for m, s in zip(motors, STATIONS)])
    capacity = (v[:, None] / duty["reference_voltage_v"]) ** 2 * capacity
    minimum_capacity = np.array([m["minimum_thrust_ref_n"] * cat["lots"][bom["motors"][s]]["minimum_thrust_factor"]
                                 * prop["thrust_factor"] for m, s in zip(motors, STATIONS)])
    minimum_capacity = (v[:, None] / duty["reference_voltage_v"]) ** 2 * minimum_capacity
    energy = float(np.dot(w, times) / 3600)
    usable = battery["energy_wh"] * cat["battery_lot"][bom["battery"]]["capacity_factor"] * duty["usable_fraction"]
    # Conservative disk-cylinder fit: its plan projection is contained in
    # a radius-D/2 circle shifted from the station by the shaft-axis offset.
    # This explicit vendor datum is independent of the retired survey prop.
    disk_centre_shift_mm = prop["plane_offset_mm"] * float(np.linalg.norm(normals[:, :2], axis=1).max())
    fit = (prop["diameter_mm"] + 2 * (duty["tip_clearance_mm"] + disk_centre_shift_mm) <= d["guard_inner_mm"]
           and all(x <= d[f"battery_bay_{axis}_mm"] for x, axis in zip(battery["box_mm"], "xyz"))
           and all(m["mount_mm"] == d["motor_mount_mm"] for m in motors))
    checks = {
        "allocation": bool(feasible), "fit": bool(fit),
        "thrust": bool(np.all(t <= capacity)),
        "minimum_control_thrust": bool(np.all(t >= minimum_capacity)),
        "motor_current": bool(np.all(a <= np.array([m["current_limit_a"] for m in motors]))),
        "esc_peak": bool(a.max() <= esc["peak_channel_a"] * cat["esc_lot"][bom["esc"]]),
        "esc_thermal": bool(np.sqrt(np.average(a * a, axis=0, weights=times)).max() <= esc["continuous_channel_a"] * cat["esc_lot"][bom["esc"]]),
        "esc_board": bool(a.sum(axis=1).max() <= esc["board_a"] * cat["esc_lot"][bom["esc"]]),
        "battery_current": bool(battery_current.max() <= battery["current_a"]),
        "voltage": bool(v.min() >= max(battery["cutoff_v"], esc["minimum_v"])),
        "energy": bool(energy <= usable),
    }
    derived = {"all_up_mass_kg": mass, "cg_x_mm": float(cg[0] * 1000),
               "cg_y_mm": float(cg[1] * 1000), "cg_z_mm": float(cg[2]*1000),
               "battery_peak_a": float(battery_current.max()), "minimum_voltage_v": float(v.min()),
               "mission_energy_wh": energy, "usable_energy_wh": usable}
    for j, axis in enumerate("xyz"):
        for k in range(j,3):
            derived[f"inertia_{axis}{'xyz'[k]}_kg_m2"] = float(inertia[j,k])
        derived[f"moment_peak_{axis}_nm"] = float(np.abs(np.array(moments)[:,j]).max())
    hover_index = next(j for j,state in enumerate(duty["states"]) if state["name"] == "hover")
    for j, s in enumerate(STATIONS):
        derived[f"thrust_min_{s}_n"] = float(t[:, j].min())
        derived[f"thrust_hover_{s}_n"] = float(t[hover_index,j])
        derived[f"esc_peak_{s}_a"] = float(a[:,j].max())
        derived[f"thrust_peak_{s}_n"] = float(t[:, j].max())
        derived[f"esc_rms_{s}_a"] = float(np.sqrt(np.average(a[:, j] ** 2, weights=times)))
    return {"pass": all(checks.values()), "checks": checks, "derived": derived,
            "minimum_individual_thrust_n": float(t.min()), "allocation_condition": allocation_condition,
            "minimum_control_ratio": (t / minimum_capacity).min(axis=0).tolist(),
            "minimum_control_margin_n": (t - minimum_capacity).min(axis=0).tolist(),
            "thrust_utilization": (t / capacity).max(axis=0).tolist(),
            "esc_peak_utilization": float(a.max() / (esc["peak_channel_a"] * cat["esc_lot"][bom["esc"]])),
            "price": price(bom, cat)}


def passing_designs(geometry, cat, duty):
    return [(b, r) for b in candidates(cat) if (r := evaluate(b, geometry, cat, duty))["pass"]]
