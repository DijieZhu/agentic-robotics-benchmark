"""An example program (it starts nothing): it shows the calls a program makes. It reads the state, takes one frame,
moves the left tool 5 cm up and back with the tool-pose level, and then holds the start posture until the shift is
over, so it turns no parcel and scores 0. Copy it to ~/submission/sort.py and build from there."""

from parcelcell.client import decode_frame


def solve(cell):
    info = cell.describe()
    print("cell", info["cell_version"], "shift", info["shift"]["duration_s"], "s;", "cameras", sorted(info["cameras"]))
    st = cell.get_state()
    print("t", st["t"], "frames left", st["frames_left"], "program time left", st["program_time_left"], "s")

    f = cell.get_camera("infeed", depth=True)              # the newest delivered frame
    rgb, depth = decode_frame(f)                           # uint8 H x W x 3, float32 H x W (metres, NaN = none)
    print("infeed frame exposed at", f["stamp"], "s:", rgb.shape, "depth", depth.shape)

    # the tool-pose level: a target pose of the left tool centre point in the world frame (position, quaternion
    # w, x, y, z); the cell's controller solves the arm's joints every tick, the right arm stays at the joint level
    tool = st["tool"]["left"]
    up = {"pos": [tool["pos"][0], tool["pos"][1], tool["pos"][2] + 0.05], "quat": tool["quat"]}
    st = cell.step(tool_pose={"left": up}, n=25)           # 0.5 s
    tt = st["tool_target"]["left"]
    print("left tool at", st["tool"]["left"]["pos"], "reached", tt["reached"], "out of reach", tt["out_of_reach"],
          "error %.4f m %.2f deg" % (tt["error_m"], tt["error_deg"]))

    q0 = info["robot"]["start_posture"]                    # back to the start posture (joint level), grippers open
    st = cell.step(q_target=q0, grip=[0.0, 0.0], n=25)
    n = info["budgets"]["max_ticks_per_step"]
    while not st["shift_over"]:
        st = cell.step(n=n)                                # keep the targets, n ticks per call
    print("shift over at", st["t"], "s:", st["over_reason"])
