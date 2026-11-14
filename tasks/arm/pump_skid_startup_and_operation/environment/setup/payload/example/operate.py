"""An example program (it starts nothing): it shows the calls a program makes. Copy it to ~/submission/operate.py and
build from there."""

from skidcell.client import decode_frame


def solve(cell):
    info = cell.describe()
    print("cell", info["cell_version"], "cameras", sorted(info["cameras"]))
    rgb, depth = decode_frame(cell.get_camera("overview"))         # the whole skid, too coarse to read a dial
    print("overview frame", rgb.shape)
    cell.command_rail(x=1.0, lift=0.3)                             # drive the carriage along the rail
    while True:
        r = cell.step(100)                                         # 1 s of simulated time
        st = cell.get_state()
        if abs(st["rail"]["x"] - 1.0) < 0.002 or r["done"]:
            break
    rgb, depth = decode_frame(cell.get_camera("wrist"))            # the wrist camera, close up
    print("wrist frame", rgb.shape, "horn", cell.get_state()["horn"])
    cell.finish()
