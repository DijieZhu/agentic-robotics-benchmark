"""The runner's configuration (runner_config.json next to this file): every number the runner uses is read here by
its key. No number of the task's table is typed into the runner's code."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "runner_config.json")
MODES = ("nominal", "quiet", "precision")


class Config(object):
    def __init__(self, path=CONFIG):
        with open(path) as f:
            self.raw = json.load(f)
        self.E = self.raw["entries"]
        self.D = self.raw["derived"]
        self.N = self.raw["numerics"]

    def v(self, key):
        return self.E[key]["value"]

    def d(self, key):
        return self.D[key]["value"]

    def eps(self, key):
        return float(self.N[key]["value"])

    # ---------------------------------------------------------------- the call budgets (s)
    @property
    def step_period_s(self):
        return float(self.v("interface.step_period_ms")) / 1000.0

    @property
    def step_cpu_s(self):
        return float(self.v("interface.step_cpu_ms")) / 1000.0

    @property
    def setup_caps(self):
        return float(self.v("interface.setup_cpu_s")), float(self.v("interface.setup_wall_s"))

    @property
    def plan_caps(self):
        return float(self.v("interface.plan_cpu_s")), float(self.v("interface.plan_wall_s"))

    @property
    def line_cap(self):
        return int(self.v("interface.reply_line_cap_bytes"))

    @property
    def watchdog_s(self):
        return float(self.v("interface.watchdog_s"))

    @property
    def wall_slack_s(self):
        return float(self.d("block_wall_slack_s"))

    # ---------------------------------------------------------------- the plan rules
    def mode_limits(self, mode):
        sp = self.v("plan.speed_limit." + mode)
        ac = self.v("plan.acceleration_limit")[mode]
        return (float(sp["linear_m_s"]), float(ac["linear_m_s2"]), float(sp["angular_rad_s"]),
                float(ac["angular_rad_s2"]))

    def plan_rules(self):
        st = self.v("plan.start_tolerance")
        et = self.v("plan.end_tolerance")["leg, slew"]
        return {"max_setpoints": int(self.v("plan.max_setpoints")),
                "rel_tol": float(self.d("limit_relative_tolerance")),
                "max_gap_s": float(self.d("consistency_max_gap_s")),
                "cons_pos": float(self.d("consistency_position_m")),
                "cons_att": math.radians(float(self.d("consistency_attitude_deg"))),
                "cons_vel": float(self.d("consistency_velocity_m_s")),
                "cons_rate": math.radians(float(self.d("consistency_rate_deg_s"))),
                "start_pos": float(st["position_m"]), "start_att": math.radians(float(st["attitude_deg"])),
                "end_pos": float(et["position_m"]), "end_att": math.radians(float(et["attitude_deg"])),
                "gap_eps": self.eps("gap_eps_s"), "zero_eps": float(self.d("zero_at_most"))}

    @property
    def refusal_reasons(self):
        return list(self.v("refusal.reasons"))

    @property
    def kinds(self):
        return list(self.v("jobs.kinds"))

    # ---------------------------------------------------------------- the block budget
    @property
    def budget_factor(self):
        return float(self.v("jobs.budget_factor"))

    @property
    def camera_mount(self):
        return [float(x) for x in self.v("jobs.camera_mount_m")]

    def fixed_term(self, kind, duration_s=None):
        if kind in ("leg", "slew"):
            return float(self.d("fixed_term_leg_slew_s"))
        if kind == "view":
            return float(self.d("fixed_term_view_s"))
        return float(duration_s)

    # ---------------------------------------------------------------- the runner's frame
    @property
    def controller_account(self):
        """the account the controller's process runs under when the runner runs as root (frame.controller_account; read
        only then)"""
        return str(self.raw["frame"]["controller_account"]["value"])

    def spin_up_s(self, mode_from, mode_to):
        lv = self.v("mode.fan_level")
        rpm = self.d("impeller_rpm_by_level")
        return (abs(float(rpm[str(lv[mode_to])]) - float(rpm[str(lv[mode_from])])) / float(self.d("spin_up_rate_rpm_s"))
                + float(self.d("spin_up_add_s")))


_CFG = None


def get():
    global _CFG
    if _CFG is None:
        _CFG = Config()
    return _CFG
