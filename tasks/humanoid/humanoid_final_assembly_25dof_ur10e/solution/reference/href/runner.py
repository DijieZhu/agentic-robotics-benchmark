"""solve(cell): the reference program as the graded run calls it. Start (describe, the state, the arm's kinematics
calibrated against the cell, the privileged reads), one frame of each camera, then every module of ORDER in turn, then
a last privileged read of the states. A failure stops the program where it happened: the phase, the reason, every event
so far are logged (never hidden) and the failure goes up to the launcher (which ends the episode with finish()).

ORDER is the reference order of the mates the modules of this folder can assemble (leg_R, leg_L, arm_L,
arm_R, head); each module file of the folder assembles its mates (the legs, one file
module_leg.py for both sides; the arms, one file module_arm.py for both sides)."""
import os
import time

import numpy as np

from . import VERSION
from . import camera as CAM
from . import module_arm
from . import module_head
from . import module_leg
from .errors import ReferenceFailure
from .kin import ArmKin
from .log import PhaseLog
from .motion import Motion
from .session import Session
from .truth import Truth

MODULES = {"leg_R": module_leg.RIGHT, "leg_L": module_leg.LEFT, "arm_L": module_arm.arm_L, "arm_R": module_arm.arm_R,
           "head": module_head}
ORDER = ("leg_R", "leg_L", "arm_L", "arm_R", "head")


def context(cell, log=None):
    log = log or PhaseLog()
    s = Session(cell, log)
    log.sim_time = lambda: s.t
    return {"session": s, "log": log}


def start(ctx):
    s, log = ctx["session"], ctx["log"]
    with log.phase("start", version=VERSION):
        info = s.describe()
        st = s.state()
        kin = ArmKin(info)
        dp, da, ok = kin.calibrate(st)
        log.check("the arm model reproduces the cell's tcp", ok, dp_m=dp, da_rad=da)
        log.check("the arm starts at home", float(np.max(np.abs(np.array(st["robot"]["q"]) - np.array(info["robot"]["home_q"])))) < 5e-3)
        truth = Truth(s)
        truth.read()
        ctx.update({"info": info, "kin": kin, "truth": truth, "motion": Motion(s, kin, log, info)})
        log.write("truth", states=truth.last.get("states"), modules=sorted(truth.last.get("modules") or {}),
                  interfaces=sorted(truth.ifaces))
    return ctx


def solve(cell, order=None, cameras=None, k_offset=None):
    """the reference program (order / cameras / k_offset: None = the reference's own: ORDER, cameras required, every
    module at its design bolt position)"""
    ctx = context(cell)
    log = ctx["log"]
    t0 = time.time()
    reports = {}
    try:
        start(ctx)
        req = (os.environ.get("HREF_CAMERAS", "required") != "optional") if cameras is None else bool(cameras)
        with log.phase("cameras"):
            reports["cameras"] = CAM.first_frames(ctx["session"], log, required=req)
        for mid in (ORDER if order is None else order):
            reports[mid] = MODULES[mid].assemble(ctx, k_offset=0 if k_offset is None else int(k_offset))
        with log.phase("finish"):
            st = ctx["truth"].read()
            log.write("final", states=st.get("states"), score=st.get("score"), sim_time=st.get("sim_time"))
        log.write("reference_done", ok=True, wall_s=round(time.time() - t0, 3), reports=reports,
                  calls=ctx["session"].calls, events=ctx["session"].events)
    except ReferenceFailure as e:
        log.write("reference_failed", phase=e.phase, reason=e.reason, data=e.data, wall_s=round(time.time() - t0, 3),
                  calls=ctx["session"].calls, events=ctx["session"].events, reports=reports)
        raise
    return reports
