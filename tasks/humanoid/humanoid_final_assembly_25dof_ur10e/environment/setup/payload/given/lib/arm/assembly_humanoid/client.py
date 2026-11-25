"""The program's proxy of the assembly cell and its launcher. Each call of the proxy is a request on the process's
original standard output, answered on its standard input; an error raises CellError. The launcher binds the wire
before anything is written, so that neither the program's printing nor native output reaches the wire, imports the
program from the current directory, calls its entry function with the proxy and then ends the episode.
"""
import os
import sys
import traceback

from ..assembly_cell.client import Cell, CellError
from ..assembly_cell.client import connect_stdio as _connect_stdio
from ..assembly_info.client import decode_frame

__all__ = ["HumanoidCell", "CellError", "decode_frame", "connect_stdio", "main", "BOOTSTRAP", "launcher_argv"]

BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv[1]); from arm.assembly_humanoid.client import main; "
             "sys.exit(main(sys.argv[2:]))")
_WIRE = None


class HumanoidCell(Cell):
    def command(self, q=None, tcp_pos=None, tcp_quat=None, width=None, force=None):
        """q (6 joint targets, joint mode) or tcp_pos (3) + tcp_quat (4, wxyz) (Cartesian mode); width (m) and force
        (N per pad) of the gripper; every value is sent as given (the cell checks it)"""
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

    def get_camera(self, name="top", depth=True):
        return self._call("get_camera", name=str(name), depth=bool(depth))


def connect_stdio():
    """the process's one HumanoidCell on its original stdin / stdout (print() and descriptor 1 go to stderr)"""
    global _WIRE
    if _WIRE is None:
        c = _connect_stdio()
        _WIRE = HumanoidCell(c._send, c._recv)
    return _WIRE


def launcher_argv(python, lib_dir, program_dir, module="assemble"):
    run_sh = os.path.join(program_dir, "run.sh")
    if os.path.isfile(run_sh) and not os.path.islink(run_sh):
        return ["bash", "run.sh"]
    from ..assembly_cell.runner import PYTHON_FLAGS
    return [python] + PYTHON_FLAGS + ["-c", BOOTSTRAP, lib_dir, module]


def main(argv=None):
    """Launcher: bind the wire first, import assemble.py from the current directory, run solve(cell), finish."""
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
