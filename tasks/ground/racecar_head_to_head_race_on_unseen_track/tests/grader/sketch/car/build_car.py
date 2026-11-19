"""MJCF generator of the owner's car.

    import mujoco
    from sketch.car import add_car, CarRuntime
    spec = venue_spec                                        # an MjSpec that already holds the floor (the venue generator's build_spec)
    add_car(spec, prefix="car/", pose=(x, y, yaw))           # the ego car; slow cars: prefix "slow0/", tail_box=True
    model = spec.compile(); data = mujoco.MjData(model)
    car = CarRuntime(model, data, prefix="car/", grip_at=floor.grip_at)

The car, in the car frame C (origin on the floor under the rear-axle midpoint, x forward, y left, z up):

    chassis (free joint `root`; sprung mass; collision boxes `chassis/col0..6`; LiDAR targets; sites lidar, imu, ref)
      rack                     hinge `steer`: the bicycle steering angle, driven by the servo actuator `servo`
      <w>/corner               slide `<w>/susp` (z): spring with preload and damper, bump and droop stops
        <w>/knuckle            front only: hinge `<w>/kingpin` (z) tied to `steer` by a quartic polycoef (Ackermann)
          <w>/carrier          tyre model `slip_angle` only: hinge `<w>/slip` (z) 20 mm ahead of the contact
            <w>/wheel          hinge `<w>/spin` (y, forward rolling positive); geom `<w>/tyre` (cylinder)

Drive line: fixed tendon `drive` = mean of the four spin angles (the motor's no-load loss as its frictionloss; the
rotor and differential inertia reflected to the wheels as a quarter each on the spin joints' armature) driven by the actuator `motor` (ctrl = motor shaft torque in N m, gain =
final drive x efficiency); the two wheels of an axle are free against each other (open differentials); the propeller
shaft that locks the front axle's mean spin to the rear's is the damped tendon `centre` (front mean minus rear mean):
its damping C_k N_series r^2 / v, scheduled by CarRuntime, is the longitudinal slip stiffness of the tyres that
real tyres use to take the axles' path difference in a corner and the friction cone lacks (a hard lock made the
cone tyres slide in every corner). Grip: one explicit contact pair per tyre and
floor geom (`floor`, `zone/<P>/slab`) named `<w>/grip/<floor geom>`; `CarRuntime.pre_step` rewrites their friction
from `grip_at` under each wheel. No two geoms of one car collide (chassis-wheel excludes; other bodies carry no geom).

Command line (a standalone model: the car on a flat floor, for inspection):
    python -m sketch.car.build_car --out car.xml [--tyre-model slip_angle] [--tail-box]
"""

import argparse
import math
import re
import sys

import numpy as np

from sketch.examples import constants as K

from .params import load_params, params_digest

FLOOR_GEOM_RE = re.compile(r"^(?:floor|zone/[A-Za-z][A-Za-z0-9_]{0,23}/slab)$")
TYRE_MODELS = ("cone", "slip_angle")


def _f(*v):
    return " ".join("%.10g" % float(x) for x in v)


def _bits(cat):
    return 'contype="%d" conaffinity="%d"' % K.CONTACT[cat]


def _box(name, lo, hi, group, cat, rgba, friction=None):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    fr = (' friction="%s"' % _f(*friction)) if friction is not None else ""
    return ('<geom name="%s" type="box" size="%s" pos="%s" group="%d" %s mass="0"%s rgba="%s"/>'
            % (name, _f(*(0.5 * (hi - lo))), _f(*(0.5 * (hi + lo))), group, _bits(cat), fr, rgba))


def resolve_tyre_model(params, tyre_model=None):
    tm = tyre_model or params["tyre_model"]
    if tm not in TYRE_MODELS:
        raise ValueError("tyre_model must be one of %s, not %r" % (TYRE_MODELS, tm))
    return tm


def car_xml(params=None, tail_box=False, tyre_model=None, tail_box_centre=None, sleep="never"):
    """MJCF text of one car without prefix (the root body is `chassis` at the origin of the world, at rest on z = 0).
    `add_car` attaches it under a prefix; `standalone_xml` puts it on a floor."""
    p = params or load_params()
    tm = resolve_tyre_model(p, tyre_model)
    geo, bodies = p["derived_geometry"], p["derived_bodies"]
    st, dr, sus = p["steering"], p["drivetrain"], p["suspension"]
    r = geo["tyre_radius_m"]
    half_w = 0.5 * K.TYRE_WIDTH_M
    ch = bodies["chassis"]
    tf = ch["fullinertia_kgm2"]
    x = ['<mujoco model="racecar_owner_car">', '<compiler angle="radian" autolimits="true"/>', "<worldbody>"]
    x.append('<body name="chassis" pos="0 0 0" sleep="%s">' % sleep)
    x.append('<freejoint name="root"/>')
    x.append('<inertial pos="%s" mass="%s" fullinertia="%s"/>' % (_f(*ch["com_c"]), _f(ch["mass_kg"]), _f(*tf)))
    # collision hull (group 1, category car)
    hf = p["hull"]["friction"]
    for bx in p["hull"]["boxes"]:
        x.append(_box(bx["name"], bx["lo"], bx["hi"], K.GROUP_COLLISION, "car", "0.15 0.25 0.70 1", friction=hf))
    # LiDAR targets of the car (group 2, no contact), the contract
    lt = p["lidar_targets"]
    x.append(_box("hull/lidar", lt["hull_lo"], lt["hull_hi"], K.GROUP_LIDAR, "none", "0.2 0.2 0.2 0.15"))
    z0, z1 = lt["head_z_m"]
    x.append('<geom name="lidar_head/lidar" type="cylinder" size="%s" pos="%s" group="%d" %s mass="0" rgba="0.05 0.05 0.05 1"/>'
             % (_f(lt["head_radius_m"], 0.5 * (z1 - z0)), _f(K.LIDAR_POS_C[0], K.LIDAR_POS_C[1], 0.5 * (z0 + z1)),
                K.GROUP_LIDAR, _bits("none")))
    if tail_box:
        c = np.asarray(tail_box_centre if tail_box_centre is not None else K.TAIL_BOX_CENTRE_C, dtype=float)
        s = 0.5 * np.asarray(K.TAIL_BOX_SIZE_M, dtype=float)
        x.append(_box("tail_box/col", c - s, c + s, K.GROUP_COLLISION, "car", "0.92 0.92 0.90 1", friction=hf))
        x.append(_box("tail_box/lidar", c - s, c + s, K.GROUP_LIDAR, "none", "0.92 0.92 0.90 1"))
    for site, pos in (("lidar", K.LIDAR_POS_C), ("imu", K.IMU_POS_C), ("ref", K.REF_POINT_C)):
        x.append('<site name="%s" pos="%s"/>' % (site, _f(*pos)))
    # the bicycle steering joint, driven by the servo actuator
    rk = bodies["rack"]
    x.append('<body name="rack" pos="%s"><joint name="steer" type="hinge" axis="0 0 1" range="%s" armature="%s"/>'
             '<inertial pos="0 0 0" mass="%s" diaginertia="%s"/></body>'
             % (_f(*rk["pos_c"]), _f(-st["joint_range_rad"], st["joint_range_rad"]), _f(st["armature_kgm2"]),
                _f(rk["mass_kg"]), _f(*rk["diaginertia_kgm2"])))
    sl = p["tyre"]["slip_angle"]
    e = sl["pivot_offset_m"]
    pivots = p["derived_slip_pivot"]
    rr_torque = p["resistance"]["rolling_coefficient"] * r          # rolling resistance per newton of load
    for w in K.WHEELS:
        cw = geo["wheel_centres_c"][w]
        cb = bodies["corners"][w]
        front = w[0] == "f"
        ds = p["derived_suspension"][w]
        n0 = p["derived_static_loads"]["normal_n"][w]
        x.append('<body name="%s/corner" pos="%s">' % (w, _f(*cw)))
        x.append('<joint name="%s/susp" type="slide" axis="0 0 1" stiffness="%s" springref="%s" damping="%s" range="%s"/>'
                 % (w, _f(sus["wheel_rate_npm"]), _f(ds["springref_m"]), _f(sus["damping_nspm"]),
                    _f(-sus["droop_m"], sus["bump_m"])))
        corner_mass = cb["corner"]["mass_kg"] + (cb["carrier"]["mass_kg"] if (tm == "cone" and not front) else 0.0)
        x.append('<inertial pos="0 0 0" mass="%s" diaginertia="%s"/>' % (_f(corner_mass), _f(*cb["corner"]["diaginertia_kgm2"])))
        depth = 1
        if front:
            kn_mass = cb["knuckle"]["mass_kg"] + (cb["carrier"]["mass_kg"] if tm == "cone" else 0.0)
            x.append('<body name="%s/knuckle" pos="0 0 0"><joint name="%s/kingpin" type="hinge" axis="0 0 1"/>'
                     '<inertial pos="0 0 0" mass="%s" diaginertia="%s"/>'
                     % (w, w, _f(kn_mass), _f(*cb["knuckle"]["diaginertia_kgm2"])))
            depth += 1
        if tm == "slip_angle":
            pv = pivots[w]
            x.append('<body name="%s/carrier" pos="%s"><joint name="%s/slip" type="hinge" axis="0 0 1" stiffness="%s" '
                     'damping="%s" range="%s"/><inertial pos="%s" mass="%s" diaginertia="%s"/>'
                     % (w, _f(e, 0.0, 0.0), w, _f(pv["stiffness_nm_per_rad"]), _f(pv["damping_nms_per_rad"]),
                        _f(-sl["pivot_range_rad"], sl["pivot_range_rad"]), _f(-e, 0.0, 0.0), _f(cb["carrier"]["mass_kg"]),
                        _f(*cb["carrier"]["diaginertia_kgm2"])))
            depth += 1
            wheel_pos = (-e, 0.0, 0.0)
        else:
            wheel_pos = (0.0, 0.0, 0.0)
        wb = cb["wheel"]
        x.append('<body name="%s/wheel" pos="%s"><joint name="%s/spin" type="hinge" axis="0 1 0" frictionloss="%s" damping="%s" '
                 'armature="%s"/>'
                 '<inertial pos="0 0 0" mass="%s" diaginertia="%s"/>'
                 '<geom name="%s/tyre" type="ellipsoid" size="%s" group="%d" %s '
                 'mass="0" friction="%s" condim="%d" rgba="0.08 0.08 0.08 1"/></body>'
                 % (w, _f(*wheel_pos), w, _f(rr_torque * n0), _f(dr["spin_damping_nms_per_rad"]),
                    _f(p["derived_drivetrain"]["spin_armature_kgm2"]), _f(wb["mass_kg"]),
                    _f(*wb["diaginertia_kgm2"]), w, _f(r, half_w, r), K.GROUP_COLLISION, _bits("car"),
                    _f(*p["tyre"]["geom_friction"]), int(p["tyre"]["contact"]["condim"])))
        x.append("</body>" * depth)
    x.append("</body>")                                  # chassis
    x.append("</worldbody>")
    x.append("<contact>" + "".join('<exclude body1="chassis" body2="%s/wheel"/>' % w for w in K.WHEELS) + "</contact>")
    dd = p["derived_drivetrain"]
    # drive: the mean spin angle of the four wheels; centre: front mean minus rear mean (the propeller shaft locks it;
    # its damping, scheduled by CarRuntime, is the series longitudinal slip stiffness of the front and rear tyres)
    x.append('<tendon><fixed name="drive" frictionloss="%s">%s</fixed>'
             '<fixed name="centre" damping="%s"><joint joint="fl/spin" coef="0.5"/>'
             '<joint joint="fr/spin" coef="0.5"/><joint joint="rl/spin" coef="-0.5"/><joint joint="rr/spin" coef="-0.5"/>'
             '</fixed></tendon>'
             % (_f(dd["tendon_frictionloss_nm"]), "".join('<joint joint="%s/spin" coef="0.25"/>' % w for w in K.WHEELS),
                _f(dd["centre_damping_rest_nms"])))
    ack = p["derived_steering"]
    x.append("<equality>")
    for w in ("fl", "fr"):
        x.append('<joint name="%s/ackermann" joint1="%s/kingpin" joint2="steer" polycoef="%s" solref="%s"/>'
                 % (w, w, _f(*ack[w]["polycoef"]), _f(*st["kingpin_solref"])))
    x.append("</equality>")
    gr = geo["final_drive_ratio"]
    tmax = 2.0 * p["derived_motor"]["stall_torque_nm"]
    kp, kv = st["servo_kp_nm_per_rad"], st["servo_kv_nms_per_rad"]
    x.append('<actuator><general name="motor" tendon="drive" gainprm="%s" ctrlrange="%s"/>'
             '<general name="servo" joint="steer" dyntype="filter" dynprm="%s" gainprm="%s" biastype="affine" biasprm="%s" '
             'ctrlrange="%s" forcerange="%s"/></actuator>'
             % (_f(gr * dr["efficiency"]), _f(-tmax, tmax), _f(st["lag_s"]), _f(kp), _f(0.0, -kp, -kv),
                _f(-st["lock_rad"], st["lock_rad"]), _f(-st["servo_torque_max_nm"], st["servo_torque_max_nm"])))
    x.append("</mujoco>")
    return "\n".join(x) + "\n"


def _yaw_quat(yaw):
    return [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]


def floor_geom_names(spec):
    """Names of the floor geoms of a spec (the contract: `floor` and `zone/<P>/slab`) in spec order."""
    return [g.name for g in spec.geoms if g.name and FLOOR_GEOM_RE.match(g.name)]


def add_floor_pairs(spec, prefix="car/", params=None):
    """One explicit contact pair per tyre of the car `prefix` and floor geom of the spec, named
    `<prefix><w>/grip/<floor geom>`; pairs that exist already are kept. Returns the number of pairs added.
    `add_car` calls it; call it again when floor geoms are added to the spec after the car."""
    p = params or load_params()
    c = p["tyre"]["contact"]
    g = p["tyre"]["default_grip"]
    have = {pr.name for pr in spec.pairs}
    n = 0
    for fname in floor_geom_names(spec):
        for w in K.WHEELS:
            name = "%s%s/grip/%s" % (prefix, w, fname)
            if name in have:
                continue
            spec.add_pair(name=name, geomname1=fname, geomname2="%s%s/tyre" % (prefix, w), condim=int(c["condim"]),
                          friction=[g, g, 0.005, 0.0001, 0.0001], solref=list(c["solref"]),
                          solreffriction=list(c["solreffriction"]), solimp=list(c["solimp"]),
                          margin=float(c["margin_m"]), gap=0.0)
            n += 1
    return n


def apply_options(spec, params=None):
    """The options of the shared model that the drive model needs: elliptic cone, the integrator
    `implicitfast` (joint and actuator damping implicit), `impratio` of the parameters. The time step is the venue's
    (`sim.physics_dt_s`, the drive model's value `physics_dt_s`); this function does not change it."""
    import mujoco
    p = params or load_params()
    o = p["options"]
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.integrator = {"implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                              "euler": mujoco.mjtIntegrator.mjINT_EULER,
                              "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT}[o["integrator"]]
    spec.option.impratio = float(o["impratio"])


def add_car(spec, prefix="car/", pose=(0.0, 0.0, 0.0), tail_box=False, params=None, tyre_model=None,
            tail_box_centre=None, sleep=None, options=True):
    """Add one car to an `MjSpec` at the planar pose [x, y, yaw] of its frame C, every name prefixed.

    tail_box         the 0.12 x 0.12 x 0.20 m box of other teams' cars (slow cars); never on the ego car
    tyre_model       'cone' or 'slip_angle'; default: `drive_params.json` `tyre_model`
    tail_box_centre  centre of the tail box in C (default: the contract's TAIL_BOX_CENTRE_C, the obstacle model's starting value)
    sleep            sleep policy of the car's tree ('never' by default: a car is never put to sleep)
    options          also set the shared model's options (`apply_options`)
    The floor geoms must be in the spec already (their contact pairs are added here, `add_floor_pairs`)."""
    import mujoco
    p = params or load_params()
    if tail_box and prefix == K.EGO_PREFIX:
        raise ValueError("the ego car never carries a tail box")
    pol = sleep or p["options"]["car_sleep"]
    child = mujoco.MjSpec.from_string(car_xml(p, tail_box=tail_box, tyre_model=tyre_model,
                                              tail_box_centre=tail_box_centre, sleep=pol))
    frame = spec.worldbody.add_frame(pos=[float(pose[0]), float(pose[1]), 0.0], quat=_yaw_quat(float(pose[2])))
    # attach_body brings the body tree and the child's tendons, equalities, actuators and excludes (renamed)
    frame.attach_body(child.worldbody.first_body(), prefix, "")
    spec.add_text(name=prefix + "drive_params_sha256", data=params_digest(p))
    add_floor_pairs(spec, prefix, p)
    if options:
        apply_options(spec, p)


def standalone_xml(params=None, tyre_model=None, tail_box=False, prefix="car/", pose=(0.0, 0.0, 0.0), floor_size=50.0,
                   timestep=None):
    """A whole model as MJCF text: a flat floor (`floor`, group 3) and one car; for tests, tools and inspection."""
    import mujoco
    p = params or load_params()
    dt = timestep or p["physics_dt_s"]
    spec = mujoco.MjSpec.from_string(
        '<mujoco model="s1_car_on_floor"><compiler angle="radian" autolimits="true"/>'
        '<option timestep="%s" cone="elliptic"/><worldbody>'
        '<light pos="0 0 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>'
        '<geom name="floor" type="plane" size="%s %s 0.05" group="%d" %s rgba="0.55 0.55 0.57 1"/>'
        '</worldbody></mujoco>' % (_f(dt), _f(floor_size), _f(floor_size), K.GROUP_FLOOR, _bits("floor")))
    add_car(spec, prefix=prefix, pose=pose, tail_box=tail_box, params=p, tyre_model=tyre_model)
    return spec.to_xml()


def main(argv=None):
    ap = argparse.ArgumentParser(description="write a standalone MJCF of the owner's car on a flat floor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tyre-model", choices=TYRE_MODELS, default=None)
    ap.add_argument("--tail-box", action="store_true")
    a = ap.parse_args(argv)
    prefix = "slow0/" if a.tail_box else "car/"
    xml = standalone_xml(tyre_model=a.tyre_model, tail_box=a.tail_box, prefix=prefix)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(xml)
    print("written", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
