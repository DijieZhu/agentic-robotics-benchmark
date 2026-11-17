"""Example for the DA-6 motion controller interface: it shows the calls and the conversions; it does NOT move the arm.

  setup  reads the unit's calibration record (servo order, directions, zero ticks, EEPROM limits) and the tool
  plan   accepts every job ("ok") and notes where the arm stands (its `state` argument: the present bus readings)
  step   re-commands that pose in the job's first cycle (the arm holds still) and sends no new goals afterwards; a
         reach job is reported done after HOLD_S seconds; a path job gets path_start in its first cycle and is
         reported done when the path's time has run out

Every job it runs therefore fails the acceptance: it is a starting point for the interface, not a controller. It uses
the standard library only and runs under the interface self-test:

    python3 check_controller.py example_controller.py
"""

import math

JOINTS = ("base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll", "jaw")      # servo ids 1 to 6
TICKS_PER_TURN = 4096
TICK_RAD = 2.0 * math.pi / TICKS_PER_TURN
CYCLE_S = 0.010
HOLD_S = 1.0


class Controller(object):
    def __init__(self, workdir):
        self.workdir = workdir              # empty and writable; it is empty again after a restart
        self.serial = None
        self.direction = self.zero = self.lo = self.hi = None
        self.tool = None
        self.job = None
        self.start_q = None
        self.held = False
        self.path_t0 = None

    # ---------------------------------------------------------------------------------------- unit, cell, tool
    def setup(self, unit, cell, tool):
        rec = unit["calibration"]
        if rec.get("format") != "calibration-record/1":
            raise ValueError("unknown calibration record format {!r}".format(rec.get("format")))
        by_id = sorted(rec["servos"].items(), key=lambda kv: kv[1]["id"])
        if [name for name, _ in by_id] != list(JOINTS):
            raise ValueError("unexpected servos {}".format([name for name, _ in by_id]))
        self.serial = unit["serial"]
        self.direction = [int(s["direction"]) for _, s in by_id]
        self.zero = [int(s["zero_ticks"]) for _, s in by_id]
        self.lo = [int(s["min_ticks"]) for _, s in by_id]
        self.hi = [int(s["max_ticks"]) for _, s in by_id]
        self.tool = dict(tool)
        print("setup: unit {}, tool {}, {} objects in the cell".format(self.serial, tool["kind"], len(cell["objects"])))

    # ---------------------------------------------------------------------------------------- conversions
    def ticks_to_q(self, ticks):
        """Design angles (rad) of six tick readings: q = direction x (ticks - zero_ticks) x 2 pi / 4096."""
        return [d * (t - z) * TICK_RAD for t, z, d in zip(ticks, self.zero, self.direction)]

    def q_to_ticks(self, q):
        """Six goal ticks (whole numbers, inside the EEPROM limits) of six design angles (rad)."""
        out = []
        for qi, z, d, lo, hi in zip(q, self.zero, self.direction, self.lo, self.hi):
            t = int(round(z + d * qi / TICK_RAD))
            out.append(min(max(t, lo), hi))
        return out

    # ---------------------------------------------------------------------------------------- jobs
    def plan(self, job, state):
        """job: the job (cell frame); state: the present bus readings (position, speed, load, moving), the arm holds."""
        self.job = job
        self.start_q = self.ticks_to_q(state["position"])      # design angles where the job starts
        self.held = False
        self.path_t0 = None
        return "ok"

    def step(self, state):
        cmd = {"goal": None, "speed": None, "acc": None, "path_start": False, "done": False}
        if not self.held:
            cmd["goal"] = self.q_to_ticks(self.start_q)     # the pose of the job's start: the arm holds still
            self.held = True
        if self.job["type"] == "path":
            if self.path_t0 is None:
                cmd["path_start"] = True            # the reference starts at the end of this cycle
                self.path_t0 = state["t"] + CYCLE_S
            cmd["done"] = state["t"] >= self.path_t0 + self.job["t"][-1]
        else:
            cmd["done"] = state["t"] >= HOLD_S
        return cmd
