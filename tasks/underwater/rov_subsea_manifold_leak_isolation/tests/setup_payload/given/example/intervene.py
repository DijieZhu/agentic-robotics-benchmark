"""Example program for the intervention vehicle: a starting point that runs and earns nothing.

It shows how a dive is programmed, not how the intervention is done. The vehicle lifts off its seat in the deployment
cage, drives a few metres out of the entrance, keeps station there, takes one camera frame and one sonar sector, drives
back and ends the dive hovering over its seat. It never looks for the manifold or the leak, never puts the leak marker
down, takes no frame of the panel and touches no valve, no dropped object and no hot stab: no item of the assessment
is earned.

Things to notice:
  * simulated time passes only inside step() and sonar_scan(); every step reply says whether the dive is over;
  * state() has empty entries right after power-up, so a short step comes first;
  * the navigation estimate starts at zero: set it to the cage's charted position to work in the work order's frame;
  * station and velocity keeping need the DVL bottom lock; without it they degrade to depth and attitude keeping, and
    this example then takes its frame and its sonar sector where it is and ends the dive there;
  * a motion call that the firmware refuses is a reply (accepted false), not an error;
  * the dive ends with the vehicle hovering: a dive that is to end at rest in the cage has to seat the vehicle on
    the cage deck (vehicle manual, section 8.3).
The example logs to the standard error stream: you see that in a practice run; nothing of it is kept from a graded
dive.
"""

import math
import sys

LIFT_M = 0.30          # height above the seat at which the vehicle flies out of the cage and back in
OUT_M = 3.0            # distance in front of the seat at which it keeps station
CRUISE_M_S = 0.20      # speed over ground on the way out


def log(*words):
    print("[example]", *words, file=sys.stderr)


def wait(vehicle, seconds, until=None):
    """Let `seconds` of simulated time pass in half-second steps. Stops early when `until(state)` is true.
    Returns "over" when the dive has ended, "met" when `until` was met, "time" otherwise."""
    t_end = vehicle.clock()["t"] + seconds
    while True:
        reply = vehicle.step(0.5)
        if reply["done"]:
            log("the dive is over:", reply["end_reason"])
            return "over"
        if until is not None and until(vehicle.state()):
            return "met"
        if reply["t"] >= t_end:
            return "time"


def command(vehicle, call, *args, **kwargs):
    """A motion call; waits out a collision stop and repeats the call once. Returns True when accepted."""
    reply = call(*args, **kwargs)
    if not reply["accepted"] and reply.get("reason") == "protective_stop":
        log("collision stop in progress, waiting", reply.get("remaining_s"), "s")
        if wait(vehicle, float(reply.get("remaining_s") or 0.0) + 0.5) == "over":
            return False
        reply = call(*args, **kwargs)
    if not reply["accepted"]:
        log("motion call refused:", reply.get("reason"))
    return bool(reply["accepted"])


def has_lock(state):
    """False once the firmware has given up station or velocity keeping for want of the bottom lock, or the
    navigation estimate has gone stale. A dropout of a moment (nav_status HOLD) is bridged by the firmware."""
    return not state["degraded"] and state["nav_status"] != "STALE"


def look_and_listen(vehicle):
    """One camera frame and one sonar sector from where the vehicle is. Returns False when the dive is over."""
    vehicle.set_camera_tilt(-20.0)                             # degrees, optical axis up positive
    if wait(vehicle, 3.0) == "over":
        return False
    frame = vehicle.camera_frame((320, 240))                   # the small frame: fewer frame units than (640, 480)
    image = frame["image"]                                     # numpy uint8, height x width x 3, RGB
    log("camera frame %s, mean level %.1f" % (image.shape, float(image.mean())))
    scan = vehicle.sonar_scan(-45.0, 45.0, 10.0)               # a sector ahead; every ping takes simulated time
    echoes = scan["intensity"]                                 # numpy uint8, one row per ping
    log("sonar sector: %d pings x %d samples, strongest echo %d, took %.1f s"
        % (echoes.shape[0], echoes.shape[1], int(echoes.max()), scan["t"][-1] - scan["t"][0]))
    used = vehicle.state()["budget"]
    log("used so far: %.1f frame units, %d pings, %.0f s"
        % (used["frame_units_used"], used["sonar_pings_used"], used["sim_time_used_s"]))
    vehicle.set_camera_tilt(0.0)
    return True


def end_without_lock(vehicle, heading, depth, where):
    """No bottom lock: the navigation estimate no longer follows the vehicle, and this example does not navigate
    without it. It keeps depth and attitude (the vehicle drifts with the water), uses its sensors once and ends the
    dive."""
    log("bottom lock lost %s: ending the dive here" % where)
    if command(vehicle, vehicle.set_attitude, heading=heading, depth=depth) and look_and_listen(vehicle):
        vehicle.finish()


def solve(vehicle):
    order = vehicle.mission()                                  # this site's work order and the budgets of the dive
    cage = order["cage"]
    seat_x, seat_y = float(cage["position_m"][0]), float(cage["position_m"][1])
    log("work order blocks:", sorted(order), "| budgets:", order["budget"])

    vehicle.step(0.5)                                          # let the sensors deliver their first samples
    vehicle.set_nav_fix(seat_x, seat_y)                        # navigation frame = the frame of the work order
    state = vehicle.state()
    seat_depth = float(state["depth_m"])
    heading = float(state["heading_rad"])                      # the vehicle sits heading out of the entrance
    fly_depth = seat_depth - LIFT_M
    log("seated at depth %.2f m, heading %.2f rad, battery %.1f V, jaws: %s"
        % (seat_depth, heading, state["battery_V"], state["gripper"]))
    vehicle.set_lights(0.5)

    # 1. lift off the seat, keeping station over it (the DVL has the cage deck below it)
    if not command(vehicle, vehicle.set_hold, seat_x, seat_y, fly_depth, heading):
        return
    if wait(vehicle, 8.0, until=lambda s: abs(s["depth_m"] - fly_depth) < 0.03) == "over":
        return

    # 2. drive out of the entrance until OUT_M from the seat
    def far_enough(s):
        x, y = s["nav_position_m"][0], s["nav_position_m"][1]
        return math.hypot(x - seat_x, y - seat_y) >= OUT_M or not has_lock(s)

    if not command(vehicle, vehicle.set_velocity, CRUISE_M_S, 0.0, fly_depth, heading):
        return
    if wait(vehicle, OUT_M / CRUISE_M_S + 10.0, until=far_enough) == "over":
        return
    state = vehicle.state()
    if not has_lock(state):
        end_without_lock(vehicle, heading, fly_depth, "on the way out")
        return

    # 3. keep station, look and listen
    here_x, here_y = state["nav_position_m"][0], state["nav_position_m"][1]
    if not command(vehicle, vehicle.set_hold, here_x, here_y, fly_depth, heading):
        return
    if not look_and_listen(vehicle):
        return

    # 4. back over the seat
    def over_seat(s):
        x, y = s["nav_position_m"][0], s["nav_position_m"][1]
        u, v = (s["velocity_b_m_s"] or (0.0, 0.0, 0.0))[:2]
        return (math.hypot(x - seat_x, y - seat_y) < 0.05 and math.hypot(u, v) < 0.03) or not has_lock(s)

    if not command(vehicle, vehicle.set_hold, seat_x, seat_y, fly_depth, heading):
        return
    if wait(vehicle, 60.0, until=over_seat) == "over":
        return
    if not has_lock(vehicle.state()):
        end_without_lock(vehicle, heading, fly_depth, "on the way back")
        return

    # 5. hover over the seat for a moment and end the dive
    if wait(vehicle, 5.0) == "over":
        return
    vehicle.set_lights(0.0)
    state = vehicle.state()
    log("hovering %.2f m above the seat, %d collision stops, %.0f s of the dive used"
        % (seat_depth - state["depth_m"], state["protect_stops"], state["budget"]["sim_time_used_s"]))
    vehicle.finish()                                           # the launcher also ends the dive when solve returns
