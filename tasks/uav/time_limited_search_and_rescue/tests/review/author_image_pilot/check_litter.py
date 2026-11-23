"""Private paired referee: same RGB pilot, with or without the second load turn."""
import argparse
import base64
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from check_rescue import Record, World, check_known_returns, run
from rescue import LIMIT, png


class Referee(Record):
    def __init__(self, world, capture):
        super().__init__(world, capture)
        self.grasp_frame = self.grasp_angle = None

    def save(self, obs):
        w = self.world
        if self.grasp_frame is None and w.held:
            axis = w.data.geom_xmat[w.board].reshape(3, 3)[:, 1]
            self.grasp_angle = math.remainder(math.atan2(-axis[0], axis[1])-w.data.qpos[3], math.pi)
            self.grasp_frame = w.steps
        return super().save(obs)


def impact_audit(w):
    """Keep contact XYZ and the jambs fixed; rotate only a separate copy of the board."""
    if w.failure != "throat_exit":
        return {}
    jambs = w.solids[np.array(w.kinds) == "throat_exit"]
    def gap(data, ids):
        return min(mujoco.mj_geomDistance(w.model, data, g, j, 1.5, None) for g in ids for j in jambs)
    alternate = mujoco.MjData(w.model)
    alternate.qpos[:] = w.data.qpos
    alternate.mocap_pos[:] = w.data.mocap_pos
    alternate.mocap_quat[:] = w.data.mocap_quat
    axis = w.data.geom_xmat[w.board].reshape(3, 3)[:, 1]
    turn = math.remainder(math.pi/2-math.atan2(-axis[0], axis[1]), math.pi)
    yaw = np.array([math.cos(turn/2), 0., 0., math.sin(turn/2)])
    mujoco.mju_mulQuat(alternate.qpos[7:11], yaw, w.data.qpos[7:11])
    mujoco.mj_forward(w.model, alternate)
    return {"board_at_impact_m": gap(w.data, [w.board]), "yaw_correction_deg": math.degrees(turn),
            "impact_lane_error_m": w.payload[1]-w.layout.pocket[1]-w.layout.loading[1]*w.layout.loading[2],
            "aircraft_and_tool_at_impact_m": gap(w.data, [*w.parts["uav"], *w.parts[f"collet{w.tool}"]]),
            "same_xyz_aligned_board_m": gap(alternate, [w.board])}


class OnDemandWorld(World):
    """Author smoke only: render consumed frames on demand at their original step."""
    def observe(self):
        if self._observation is None:
            world, step = self, self.steps

            class Pair:
                frames = None

                def __getitem__(self, camera):
                    if self.frames is None:
                        assert world.steps == step
                        self.frames = [png(f) for f in world.images()]
                    return self.frames[camera]

            self._observation = {"images": Pair(), "frame_id": step, "done": self.done,
                                 "steps_remaining": LIMIT-step}
        return self._observation


def flight(seed, hold, out, quick):
    w = (OnDemandWorld if quick else World)("rescue", seed)
    record = Referee(w, capture=not quick)
    mode = "fixed-heading" if hold else "rgb"
    try:
        error, p = None, None
        try:
            p = run(record, hold_heading=hold)
        except RuntimeError as exc:
            error = str(exc)
            tb = exc.__traceback__
            while tb and tb.tb_frame.f_code is not run.__code__:
                tb = tb.tb_next
            p = tb.tb_frame.f_locals["p"] if tb else None
        trace = p.trace if p else []
        stages = {e["event"]: e for e in trace if e["event"] in (
            "sled_found", "flange_survey", "depot_choice", "sled_reacquired", "litter_axis",
            "pickup_attempt", "entry_jambs", "pocket_fix", "exit_jambs")}
        result = {"seed": str(seed), "mode": mode, "quick": quick, "error": error,
                  "loading": list(w.layout.loading), "flange_radius": w.layout.flange,
                  "grasp_frame": record.grasp_frame,
                  "actual_attachment_deg": None if record.grasp_angle is None else math.degrees(record.grasp_angle),
                  "rgb_attachment_deg": None if not hasattr(p, "litter_relative") else math.degrees(p.litter_relative),
                  "failure": w.failure, "contact_geom": w.impact_geom, "contact_obstacle": w.impact_obstacle,
                  "stages": stages, **impact_audit(w), **w.metrics()}
        print(json.dumps(result), flush=True)
        (out/f"{mode}-trace.json").write_text(json.dumps(trace, indent=2))
        (out/f"{mode}-actions.json").write_text(json.dumps(record.actions))
        for camera in range(2):
            (out/f"{mode}-final-{camera}.png").write_bytes(base64.b64decode(w.observe()["images"][camera]))
        if not quick:
            for event in ("flange_survey", "litter_axis", "entry_jambs", "exit_jambs"):
                if event in stages:
                    step = stages[event]["frame"]
                    for camera in (0, 1):
                        record.export(step, out/f"{mode}-{event}-{camera}.png", camera)
        return record.actions, result
    finally:
        w.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--positive-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    actions, pos = flight(args.seed, False, args.output, args.quick)
    assert not pos["error"] and pos["success"] and not pos["failure"], pos
    assert pos["grasped"] and pos["cargo_return"] and 0 < pos["entry_clear"] < pos["exit_clear"]
    check_known_returns(json.loads((args.output/"rgb-trace.json").read_text()))
    if not args.positive_only:
        other, neg = flight(args.seed, True, args.output, args.quick)
        common = pos["stages"]["exit_jambs"]["frame"]
        assert actions[:common] == other[:common] and common > pos["steps_to_lift"] > 0
        assert not neg["success"] and neg["failure"] == "throat_exit" and neg["pod_impact"], neg
        assert neg["entry_clear"] and not neg["exit_clear"] and not neg["error"], neg
        assert neg["board_at_impact_m"] <= 0 < min(neg["same_xyz_aligned_board_m"],
                                                   neg["aircraft_and_tool_at_impact_m"])
        print(json.dumps({"identical_prefix_actions": common, "seed": str(args.seed)}), flush=True)


if __name__ == "__main__":
    main()
