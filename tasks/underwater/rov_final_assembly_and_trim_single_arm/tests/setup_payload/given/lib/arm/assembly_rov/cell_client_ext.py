"""The station's cell proxy (`StationCell`) and the launcher that hands it to `solve(cell)` in assemble.py.

    def solve(cell):                                  # in /home/user/submission/assemble.py; cell is a StationCell
        info = cell.describe()                        # info["work_order"]: this run's work order
        st = cell.get_state()                         # st["fixture"]: {"index", "moving", "load_N": [A, B, C]}
        cell.fixture_index("deck_up")                 # {"ok": True} or {"refused": "<why>"}
        cell.set_control("cartesian", 0.5)
        cell.command(tcp_pos=[...], tcp_quat=[w, x, y, z], width=0.06, force=60.0)
        r = cell.step(50)                             # r["events"]: only "safety:..." and "episode_over"
        cell.finish()

`StationCell` is the general camera-cell proxy of this library (`arm.assembly_info.client.InfoCell`: describe,
get_state, set_control, command, step, get_camera, finish; one JSON-lines request each on the process's original
stdout, answered on its stdin) with the two additions of this station: `fixture_index(target)` (turn the positioner to
`keel_up` or `deck_up`) and the `force` argument of `command` (the grip force per pad, N). A reply that carries an error
raises `CellError` (a call the cell does not offer answers "forbidden" and is counted); a refused turn is an answer,
not an error. `decode_frame(reply)` turns a camera reply into (rgb uint8 H x W x 3, depth float32 H x W in metres with
NaN where there is no reading).

The launcher: `python3 -B -E -s -c <BOOTSTRAP> <lib> assemble` puts <lib> on sys.path and calls `main`, which binds
the wire to the original stdin / stdout (print() goes to stderr), imports assemble.py from the current directory, calls
solve(cell) and finishes the episode whatever solve does. A `run.sh` in the program folder is started as `bash run.sh`
instead.
"""

import os
import sys
import traceback

from ..assembly_cell.client import CellError
from ..assembly_cell.client import connect_stdio as _connect_stdio
from ..assembly_info.client import InfoCell, decode_frame

__all__ = ["StationCell", "CellError", "decode_frame", "connect_stdio", "main", "BOOTSTRAP", "launcher_argv", "INDEX_POSITIONS"]

BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv[1]); from arm.assembly_rov.cell_client_ext import main; "
             "sys.exit(main(sys.argv[2:]))")
INDEX_POSITIONS = ("keel_up", "deck_up")


class StationCell(InfoCell):
    def command(self, q=None, tcp_pos=None, tcp_quat=None, width=None, force=None):
        """q (6 joint targets, joint mode) or tcp_pos (3) + tcp_quat (4, wxyz) (Cartesian mode); width (m) and force (N per
        pad) of the gripper; every value is sent as given (the cell checks it)"""
        params = {}
        if q is not None:
            params["q"] = [float(v) for v in q]
        if tcp_pos is not None:
            params["tcp_pos"] = [float(v) for v in tcp_pos]
        if tcp_quat is not None:
            params["tcp_quat"] = [float(v) for v in tcp_quat]
        if width is not None:
            params["width"] = float(width)
        if force is not None:
            params["force"] = float(force)
        return self._call("command", **params)

    def fixture_index(self, target):
        """start the positioner's turn to `target` ("keel_up" or "deck_up"): {"ok": True} or {"refused": why}, why in
        robot_in_column / moving / no_turns_left; the turn then runs during step()"""
        return self._call("fixture_index", target=str(target))


def connect_stdio():
    """a StationCell on the process's original stdin / stdout; print() goes to stderr from here on"""
    c = _connect_stdio()
    return StationCell(c._send, c._recv)


def launcher_argv(python, lib_dir, program_dir, module="assemble"):
    run_sh = os.path.join(program_dir, "run.sh")
    if os.path.isfile(run_sh) and not os.path.islink(run_sh):
        return ["bash", "run.sh"]
    from ..assembly_cell.runner import PYTHON_FLAGS
    return [python] + PYTHON_FLAGS + ["-c", BOOTSTRAP, lib_dir, module]


def main(argv=None):
    """Launcher: import assemble.py from the current directory and run solve(cell) with a StationCell."""
    argv = list(sys.argv[1:] if argv is None else argv)
    module = argv[0] if argv else "assemble"
    cell = connect_stdio()
    sys.path.insert(0, os.getcwd())
    code = 0
    try:
        mod = __import__(module)
        getattr(mod, "solve")(cell)
    except CellError as e:
        print("[assemble] cell error: %s" % e, file=sys.stderr)
        code = 2
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        code = 1
    finally:
        try:
            cell.finish()
        except Exception:  # noqa: BLE001
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
