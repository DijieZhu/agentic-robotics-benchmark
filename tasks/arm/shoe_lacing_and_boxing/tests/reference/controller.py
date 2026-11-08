"""The reference controller of shoe_lacing_and_boxing (the oracle; privileged: runs inside the grader with the exact
state). `run(server)` works the cell's work order pair after pair: seat and lace each shoe, erect the blank, pack
the pair, put the lid on, push the box to the outfeed.

A failing skill never ends the episode: the lacing of one shoe is skipped after an exception (the box work of the
pair still earns its credits), a failed box phase moves on to the next pair when the box could not be shipped, and
only EpisodeOver (the server's done flag, a physics blow-up or the sim cap) stops the controller."""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from primitives import EpisodeOver, Rig  # noqa: E402

import box_skills  # noqa: E402
import lace_skills  # noqa: E402


def _try(rig, label, fn, *args):
    """Run one skill; an ordinary exception is logged and reported as a failure, EpisodeOver propagates."""
    try:
        return fn(*args)
    except EpisodeOver:
        raise
    except Exception:  # noqa: BLE001 - a skill failure ends this step, not the episode
        rig.log("[ref] %s failed:\n%s" % (label, traceback.format_exc()))
        return False


def run(server):
    rig = Rig(server)
    log = rig.log
    results = []
    try:
        while True:
            rec = server.work.current
            if rec is None or rec["shipped"]:
                break
            k = rec["index"]
            log("[ref] pair %d: start t=%.1f" % (k + 1, rig.time))
            laced = []
            for s in (0, 1):
                if not bool(server.cfg.get("reference_lacing", True)):
                    laced.append(False)            # the grader's reference_lacing knob: box the pair unlaced
                    continue
                ok = _try(rig, "pair %d seat_shoe(%d)" % (k + 1, s), lace_skills.seat_shoe, rig, s)
                if ok is not False:
                    ok = _try(rig, "pair %d lace_shoe(%d)" % (k + 1, s), lace_skills.lace_shoe, rig, s)
                _try(rig, "pair %d unseat_shoe(%d)" % (k + 1, s), lace_skills.unseat_shoe, rig, s)
                laced.append(ok is not False)
            _try(rig, "pair %d park_hook_arm" % (k + 1), box_skills.park_hook_arm, rig)
            erected = _try(rig, "pair %d erect_box" % (k + 1), box_skills.erect_box, rig)
            packed = _try(rig, "pair %d pack_pair" % (k + 1), box_skills.pack_pair, rig)
            closed = _try(rig, "pair %d put_lid" % (k + 1), box_skills.put_lid, rig)
            shipped = _try(rig, "pair %d ship_box" % (k + 1), box_skills.ship_box, rig)
            log("[ref] pair %d: laced %s erected %s packed %s closed %s shipped %s t=%.1f" % (
                k + 1, laced, erected, packed, closed, shipped, rig.time))
            results.append({"pair": k, "t": rig.time, "laced": laced, "erected": bool(erected), "packed": bool(packed),
                            "closed": bool(closed), "shipped": bool(shipped)})
            if server.work.current is rec and not rec["shipped"]:
                log("[ref] pair %d not shipped: the next pair cannot arrive, stopping" % (k + 1))
                break
    except EpisodeOver as e:
        log("[ref] episode over: %s" % e)
    return {"pairs": results, "t": rig.time}
