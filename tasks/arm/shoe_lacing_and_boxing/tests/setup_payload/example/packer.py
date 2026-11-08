"""A complete, policy-free program for the cell: it reads one frame per fixed camera, moves both arms to a pose above
the bench, and finishes. Copy it to ~/submission/packer.py and replace `solve` with your controller. As shipped it does
no work and scores 0."""

import sys

from shoecell.client import decode_frame


def solve(cell):
    info = cell.describe()
    print("cell", info["cell_version"], "cameras", list(info["cameras"]), file=sys.stderr)
    for name in ("overhead", "front"):
        rgb, depth = decode_frame(cell.get_camera(name))
        print(name, rgb.shape, "depth valid fraction", float((depth == depth).mean()), file=sys.stderr)
    for arm, x in (("left", 0.15), ("right", -0.40)):          # each arm toward its own side of the bench
        cell.set_control(arm, "cartesian", 0.6)
        cell.command(arm, tcp_pos=[x, 0.0, 0.30], tcp_quat=[0.0, 1.0, 0.0, 0.0])
    r = cell.step(200)
    print("stepped to sim time %.2f s, events %s" % (r["sim_time"], r["events"]), file=sys.stderr)
    cell.finish()
