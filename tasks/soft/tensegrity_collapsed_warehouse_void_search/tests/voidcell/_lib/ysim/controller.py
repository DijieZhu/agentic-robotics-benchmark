"""Cable-length control as on the real robot: the off-board PC's PID of the authors' control code
(ChemStud24/tensegrity rss25 @4b79620, src/run_tensegrity_Astar.py compute_command(), lines 401-428),
run at the control rate on the short-sensor lengths, sending a PWM duty to each motor (firmware onboard.ino
lines 392-393: duty = speed/99, speed = command * max_speed).

Per motor i, on every update:
  pos = (L_i - min_length) / RANGE_i           (L_i = short sensor i length, mm)
  tolerance = low_tol if target < 0.5 else tol
  if |pos - target| < tolerance: done_i = True (latched for this state), command 0
  else: error = pos - target; d = error - prev; cum += error; command = clip(P e + I cum + D d, -1, 1)
when all six are done the executor moves to the next state and resets prev/cum/done.

Optional slack rule (execution mode 'tracked' only; the authors' simulator, hybrid_mppi @d1e58a4
mujoco_physics_engine/pid.py lines 98-99): a motor whose command would pay out a slack line gets command 0 at that
tick; it is NOT latched as done (ysim 0.2.0 latched it: our rule, removed in 0.3.0). The authors' simulator ends a
state at the first tick at which every command is zero (tensegrity_mjc_simulation.py line 594):
``state_complete()``.
"""
from . import params as prm


class FirmwarePID:
    """variant "hardware" (default) = the robot's PC code above. variant "authors_sim" = the done rule of the
    authors' MuJoCo simulator (hybrid_mppi @d1e58a4 mujoco_physics_engine/pid.py update_control_by_target_gait:
    done if |error| < tol or |L - L_target| < 10 mm or (target 0 and pos < 0)); used only for diagnostics."""

    def __init__(self, variant="hardware", P=None):
        self.P, self.I, self.D = prm.value("pid_P"), prm.value("pid_I"), prm.value("pid_D")
        if P is not None:
            self.P = P
        self.variant = variant
        self.max_frac = prm.value("pwm_max_fraction")
        self.min_length = prm.value("min_length") * 1000.0
        self.reset()

    def reset(self):
        self.done = [False] * 6
        self.zeroed = [False] * 6
        self.prev = [0.0] * 6
        self.cum = [0.0] * 6
        self.command = [0.0] * 6

    def update(self, lengths_mm, targets, ranges_mm, tol, low_tol, slack=None):
        """Returns the six duties (positive = wind in). Mirrors compute_command() line by line.
        slack (optional, list of bool, the authors' simulator rule): a motor not done whose command is negative
        (pay out) while its line is slack gets command 0 for this tick only (``zeroed``)."""
        duty = [0.0] * 6
        self.zeroed = [False] * 6
        for i in range(6):
            pos = (lengths_mm[i] - self.min_length) / ranges_mm[i]
            t = targets[i]
            tolerance = low_tol if t < 0.5 else tol
            hit = pos + tolerance > t and pos - tolerance < t
            if self.variant == "authors_sim":
                hit = hit or abs(lengths_mm[i] - (self.min_length + ranges_mm[i] * t)) < 10.0 or (t == 0 and pos < 0)
            if hit:
                self.done[i] = True
                self.command[i] = 0.0
            if not self.done[i]:
                e = pos - t
                de = e - self.prev[i]
                self.cum[i] += e
                self.prev[i] = e
                self.command[i] = max(min(self.P * e + self.I * self.cum[i] + self.D * de, 1.0), -1.0)
                if slack is not None and slack[i] and self.command[i] < 0.0:
                    self.command[i] = 0.0
                    self.zeroed[i] = True
                duty[i] = self.command[i] * self.max_frac
        return duty

    def all_done(self):
        return all(self.done)

    def state_complete(self):
        """Every motor is done (latched) or zeroed by the slack rule at this tick: the state ends."""
        return all(d or z for d, z in zip(self.done, self.zeroed))
