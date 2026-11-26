"""A toy chain for the interface's self-test. It is NOT the robot and holds no value of any unit: two modules in a
planar chain under gravity and one finger pair, with made-up masses, lengths and ranges.

The toy speaks the controller interface's wire: sixteen module places in the interface's order, the same frames, the
same feedback, the 4 ms exchange with replies one cycle old. Its live places are 0 and 1 (the planar chain, in the
places of left J1 and left J2) and 7 (the finger pair, in the place of the left gripper); the other thirteen places
answer as locked joints (angle and speed 0, the torque their frame's law would command there).

    rec = toy_record()                  # the toy's delivery record (the record's format; made-up values)
    site = toy_site()                   # {"cell": a made-up cell sheet, "shift": a two-slot toy shift sheet}
    toy = ToyChain(gripper_mode, frames.record_ranges(rec))
                                        # {"left": "mit" | "posforce", "right": ...}; only the left one is live
    replies = toy.start()               # the module states before cycle 0 (each module's last reply)
    replies = toy.exchange(frames)      # frames: 16 entries (dict or None) of cycle k; returns the replies sampled
                                        # at the start of cycle k (None for a module that got no frame of its kind)

Physics: 1 ms steps, four per cycle; each module's law acts inside the step from the frame it holds (a module without
a new frame keeps its last one). Standard library only.
"""
import math

import frames as FR                                    # the frame codec, beside this file

N = 16
LIVE_CHAIN = (0, 1)
LIVE_FINGERS = 7
STEP_S = 0.001
STEPS_PER_CYCLE = 4
CYCLE_S = STEP_S * STEPS_PER_CYCLE
G = 9.81
# made-up toy values (not the robot's)
M1, M2 = 1.2, 0.8                  # kg, point masses at the ends of the two links
L1, L2 = 0.30, 0.25                # m
DAMP = 0.05                        # N m s/rad at each chain joint
FINGER_INERTIA = 2.0e-4            # kg m^2
FINGER_RANGE = (0.0, 0.8)          # rad; 0 = closed
FINGER_DAMP = 0.01                 # N m s/rad
FP_SPEED_GAIN = 0.5                # N m s/rad: the toy's force-position speed loop
FP_POS_TIME = 0.05                 # s: the toy's force-position position loop
TOY_LIMITS = (12.5, 30.0, 10.0)    # PMAX, VMAX, TMAX of every toy module (made up; printed in its register dump)
TEMPERATURE_C = 25
TOY_IMAX_A = 2.0                   # the toy's current base (A), printed at its power-up
TOY_RECORD_FORMAT = "openarm05/DeliveryRecord/1"
TOY_LABELS = tuple(f"{s} J{j}" for s in ("left", "right") for j in range(1, 9))
TOY_HOME = (0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0)
# the toy's registers, as its dump prints them (made up; the force-position torque per unit of i_lim is the toy's
# current base times its torque per ampere, 1.5 x pole pairs x flux x gear ratio x gear torque efficiency)
TOY_NPP, TOY_FLUX, TOY_GEAR, TOY_GREF = 10, 0.01, 10.0, 1.0
TOY_REGISTERS = (
    ("Under-Voltage Threshold", "RW", "float", "(10.0, fmax]", "10.0000"),
    ("KT Value (Torque Const)", "RW", "float", "[0.0, fmax]", "0.0000"),
    ("Over-Temp Threshold", "RW", "float", "[80.0, 200)", "100.0000"),
    ("Over-Current Threshold", "RW", "float", "(0.0, 1.0)", "0.5000"),
    ("CAN Timeout", "RW", "uint32", "[0, 2^32-1]", "0"),
    ("Control Mode", "RW", "uint32", "[0, 4]", "1 (MIT)"),
    ("Number of Pole Pairs", "RO", "uint32", "/", f"{TOY_NPP:d}"),
    ("Stator Resistance (Rs)", "RO", "float", "/", "1.000000e-01"),
    ("Stator Inductance (Ls)", "RO", "float", "/", "1.000000e-04"),
    ("Rotor Flux", "RO", "float", "/", f"{TOY_FLUX:.6e}"),
    ("Gear Ratio", "RO", "float", "/", f"{TOY_GEAR:.4f}"),
    ("Position Limit (PMAX)", "RW", "float", "(0.0, fmax]", f"{TOY_LIMITS[0]:.4f}"),
    ("Velocity Limit (VMAX)", "RW", "float", "(0.0, fmax]", f"{TOY_LIMITS[1]:.4f}"),
    ("Torque Limit (TMAX)", "RW", "float", "(0.0, fmax]", f"{TOY_LIMITS[2]:.4f}"),
    ("Gear Torque Efficiency", "RW", "float", "(0.0, 1.0]", f"{TOY_GREF:.4f}"),
)
FP_TORQUE_PER_UNIT = TOY_IMAX_A * 1.5 * TOY_NPP * TOY_FLUX * TOY_GEAR * TOY_GREF     # N m per unit of i_lim
# the toy's two-slot sheet (made-up times in s and targets in m, in the shift sheet's format)
TOY_SHEET = {"first_presentation_s": 1.0, "move_s": 1.16, "hold_s": 0.9, "lifted_point_above_seated_m": 0.01}
TOY_WINDOWS = {"closing": (0.0, 5.0), "trial_lift_and_decision": (5.0, 9.0), "decline_accepted": (0.0, 9.0),
               "carry": (9.0, 11.06), "set_down": (11.06, 17.06), "retreat": (17.06, 20.06), "next_tray_s": 23.06}
TOY_DECLINED = {"put_back_by_s": 4.0, "hands_clear_by_s": 7.0, "next_tray_after_s": 10.0}
TOY_MISSED = {"clear_by_s": 12.0, "next_tray_s": 15.0}
TOY_TARGETS = ((0.22, 0.0, 0.38), (0.36, 0.0, 0.40))
# the toy's cell sheet, in the cell sheet's format (made-up values)
TOY_CELL = {
    "format": "openarm05/CellSheet/1",
    "frame": "the toy's: z up (made up)",
    "gravity_m_s2": (0.0, 0.0, -G),
    "bench": {"top_z_m": 0.2, "top_tolerance_m": 0.001, "x_m": (0.1, 0.6), "half_width_y_m": 0.5},
    "F": {"what": "the toy's fixture (made up)", "reference_point_drawn_m": (0.2, 0.0, 0.3),
          "frame_drawn": ((0.2, 0.0, 0.2), (1.0, 0.0, 0.0, 0.0)), "frame_note": "made up (toy)",
          "survey_tolerance": {"per_axis_m": 0.001, "about_each_axis_deg": 0.1},
          "cups": {"centres_in_F_m": ((0.05, 0.2), (0.05, -0.2), (-0.05, 0.2), (-0.05, -0.2)),
                   "inner_size_m": (0.02, 0.02), "wall_height_m": 0.005, "wall_thickness_m": 0.005,
                   "lead_in": False},
          "presented": "made up (toy)"},
    "S": {"what": "the toy's shelf (made up)", "x_m": (0.3, 0.5), "half_width_y_m": 0.2, "top_z_drawn_m": 0.3,
          "seat_reference_point_drawn_m": (0.4, 0.0, 0.4), "frame_drawn": ((0.4, 0.0, 0.3), (1.0, 0.0, 0.0, 0.0)),
          "frame_note": "made up (toy)", "survey_tolerance": {"per_axis_m": 0.002, "about_each_axis_deg": 0.2}},
    "clear_volume": {"outside_boxes_grown_by_m": 0.05, "above_z_m": 0.2, "boxes": "made up (toy)"},
    "ready_pose": {"tool_points_above_grip_points_m": 0.05, "grippers": "open", "of": "made up (toy)"},
    "trays": {"band_kg_about": (0.5, 2.0), "centre_within_along_handle_line_m": 0.05,
              "centre_within_fore_aft_m": 0.01, "centre_below_grip_line_m": (0.01, 0.05),
              "heavier_trays_up_to_about_kg": 5.0, "heavier_trays": "made up (toy)"},
    "ambient_C": (20.0, 30.0),
}


def _dump(i):
    """A made-up register dump of toy module i, laid out as the maker's parameter tool prints one (the format of the
    delivery record's register_dump), followed by the power-up serial line with the toy's current base."""
    rows = ["", "=" * 50, f" MOTOR ID: 0x{(i % 8) + 1:x} (Response ID: 0x{(i % 8) + 0x11:x})", "=" * 50,
            f"{'Parameter':<30}{'R/W':<5}{'Type':<10}{'Range':<20}Value", "-" * 80]
    for name, rw, typ, rng, val in TOY_REGISTERS:
        rows.append(f"{name:<30}{rw:<5}{typ:<10}{rng:<20}{val}")
    rows += ["", "Power-up serial output:", f"Imax: {TOY_IMAX_A:.6f}"]
    return "\n".join(rows) + "\n"


def toy_record():
    """The toy's delivery record, in the delivery record's format (made-up values): per module its label, product,
    edition, firmware, serial, current base and register dump (whose PMAX, VMAX and TMAX lines are the toy's ranges)."""
    mods = [{"label": TOY_LABELS[i], "product": "toy module", "edition": "toy", "firmware": "toy",
             "serial": f"TOY{i:02d}", "Imax_A": TOY_IMAX_A, "register_dump": _dump(i)} for i in range(N)]
    return {"format": TOY_RECORD_FORMAT, "unit_id": "toy", "modules": mods, "zeroing": "made up (toy)",
            "mounting": "made up (toy)", "pads": "made up (toy)",
            "home_pose_rad": {"left": list(TOY_HOME), "right": list(TOY_HOME), "grippers": "open"},
            "not_identified": "nothing of the toy was identified"}


def toy_site():
    """The toy's site: a made-up cell sheet in the cell sheet's format, and a toy shift of two slots in the shift
    sheet's format (one made-up target each)."""
    w, d, m = TOY_WINDOWS, TOY_DECLINED, TOY_MISSED
    slots = []
    t = TOY_SHEET["first_presentation_s"]
    for i, tg in enumerate(TOY_TARGETS):
        slots.append({"tray": i, "planned_presentation_s": t, "job": "short", "pick_rest": "F",
                      "destination_rest": "S", "n_targets": 1,
                      "windows": {k: (list(v) if isinstance(v, tuple) else v) for k, v in w.items()},
                      "declined": dict(d), "missed_capture": dict(m),
                      "targets": [{"k": 0, "p": list(tg), "move_start_s": w["carry"][0],
                                   "arrive_s": w["carry"][0] + TOY_SHEET["move_s"], "hold_end_s": w["carry"][1]}]})
        t += w["next_tray_s"]
    shift = {"format_version": 1, "cycle_s": CYCLE_S, "time_origin": "toy", "frame": "toy"}
    shift.update(TOY_SHEET)
    shift["slots"] = slots
    return {"cell": json_copy(TOY_CELL), "shift": shift}


def json_copy(o):
    """A plain copy of a nested dict / tuple / list (tuples as lists), as the data reaches a controller."""
    if isinstance(o, dict):
        return {k: json_copy(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_copy(v) for v in o]
    return o


def toy_schedule(site):
    """The toy supervisor's signal: a function of the cycle k giving (tray index or None, cycles since that tray's
    presentation or None), the trays presented at the shift sheet's planned times (the toy declines nothing and
    carries nothing)."""
    starts = [int(round(sl["planned_presentation_s"] / CYCLE_S)) for sl in site["shift"]["slots"]]

    def sig(k):
        cur = None
        for i, s in enumerate(starts):
            if k >= s:
                cur = i
        return (None, None) if cur is None else (cur, k - starts[cur])
    return sig


def toy_cycles(site, after_s=2.0):
    """The toy shift's length in cycles: to `after_s` after the last tray's presentation."""
    return int(round((site["shift"]["slots"][-1]["planned_presentation_s"] + after_s) / CYCLE_S))


class ToyChain:
    def __init__(self, gripper_mode, limits=None):
        """limits: per module (PMAX, VMAX, TMAX), as the toy's record prints them (frames.record_ranges of
        toy_record()); by default read from that record."""
        self.lim = [tuple(x) for x in (limits if limits is not None else FR.record_ranges(toy_record()))]
        self.mode = ["mit"] * N
        self.mode[7] = gripper_mode["left"]
        self.mode[15] = gripper_mode["right"]
        self.q = [0.0] * N
        self.w = [0.0] * N
        self.q[1] = 0.3
        self.q[LIVE_FINGERS] = 0.6
        self.cmd = [None] * N
        self.tau = [0.0] * N

    # -- the module laws
    def _law(self, i, q, w):
        c = self.cmd[i]
        pm, vm, tm = self.lim[i]
        if c is None:
            return 0.0
        if self.mode[i] == "mit":
            e = FR.mit_effective(c, self.lim[i])
            t = e["kp"] * (e["p"] - q) + e["kd"] * (e["v"] - w) + e["t"]
            return max(-tm, min(tm, t))
        e = FR.posforce_effective(c)
        w_ref = max(-e["v_lim"], min(e["v_lim"], (e["p"] - q) / FP_POS_TIME))
        lim = e["i_lim"] * FP_TORQUE_PER_UNIT
        return max(-lim, min(lim, FP_SPEED_GAIN * (w_ref - w)))

    def _chain_acc(self, t1, t2):
        q1, q2 = self.q[0], self.q[1]
        w1, w2 = self.w[0], self.w[1]
        c2, s2 = math.cos(q2), math.sin(q2)
        m11 = (M1 + M2) * L1 * L1 + M2 * L2 * L2 + 2.0 * M2 * L1 * L2 * c2
        m12 = M2 * L2 * L2 + M2 * L1 * L2 * c2
        m22 = M2 * L2 * L2
        h = M2 * L1 * L2 * s2
        cor1 = -h * (2.0 * w1 * w2 + w2 * w2)
        cor2 = h * w1 * w1
        g1 = (M1 + M2) * G * L1 * math.cos(q1) + M2 * G * L2 * math.cos(q1 + q2)
        g2 = M2 * G * L2 * math.cos(q1 + q2)
        r1 = t1 - DAMP * w1 - cor1 - g1
        r2 = t2 - DAMP * w2 - cor2 - g2
        det = m11 * m22 - m12 * m12
        return (m22 * r1 - m12 * r2) / det, (m11 * r2 - m12 * r1) / det

    def _step(self):
        tau = [self._law(i, self.q[i], self.w[i]) for i in range(N)]
        a1, a2 = self._chain_acc(tau[0], tau[1])
        f = LIVE_FINGERS
        af = (tau[f] - FINGER_DAMP * self.w[f]) / FINGER_INERTIA
        for i, a in ((0, a1), (1, a2), (f, af)):
            self.w[i] += a * STEP_S
            self.q[i] += self.w[i] * STEP_S
        lo, hi = FINGER_RANGE
        if self.q[f] < lo or self.q[f] > hi:
            self.q[f] = min(hi, max(lo, self.q[f]))
            self.w[f] = 0.0
        return tau

    def _reply(self, i, tau):
        pm, vm, tm = self.lim[i]
        q = FR.from_code(FR.to_code(self.q[i], -pm, pm, 16), -pm, pm, 16)
        dq = FR.from_code(FR.to_code(self.w[i], -vm, vm, 12), -vm, vm, 12)
        t = FR.from_code(FR.to_code(tau, -tm, tm, 12), -tm, tm, 12)
        return {"q": q, "dq": dq, "tau": t, "t_mos": TEMPERATURE_C, "t_rotor": TEMPERATURE_C, "status": 1}

    def start(self):
        return [self._reply(i, 0.0) for i in range(N)]

    def exchange(self, frames):
        """frames: 16 entries (a checked frame dict of its kind, or None). A frame of the wrong kind for a module's
        mode is ignored and gets no reply. Returns 16 replies (None where no reply)."""
        got = [False] * N
        for i, f in enumerate(frames):
            if f is None:
                continue
            kind = "mit" if "kp" in f else "posforce"
            if kind != self.mode[i]:
                continue
            self.cmd[i] = f
            got[i] = True
        tau0 = [self._law(i, self.q[i], self.w[i]) for i in range(N)]
        replies = [self._reply(i, tau0[i]) if got[i] else None for i in range(N)]
        for _ in range(STEPS_PER_CYCLE):
            self._step()
        return replies
