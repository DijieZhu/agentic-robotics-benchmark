"""Agent-side proxy of the information-tier cell + its launcher (PUBLIC CODE; the wire of `arm.assembly_cell`).

    def solve(cell):                          # in /home/user/submission/assemble.py
        info = cell.describe()
        f = cell.get_camera("top")            # one frame: PNG rgb + 16-bit PNG sensor depth (mm, 0 = invalid)
        rgb, depth_m = decode_frame(f)        # uint8 H x W x 3, float32 H x W metres (NaN where invalid)
        st = cell.get_state()                 # robot state only (noisy wrench)
        cell.set_control("cartesian", 0.35)
        cell.command(tcp_pos=[...], tcp_quat=[w, x, y, z], width=0.08)
        r = cell.step(20)                     # r["events"]: only "safety:..." and "episode_over"
        cell.finish()
"""

import base64
import io
import os
import sys
import traceback

from ..assembly_cell.client import Cell, CellError, connect_stdio as _connect_stdio

__all__ = ["InfoCell", "CellError", "decode_frame", "connect_stdio", "main", "BOOTSTRAP", "launcher_argv"]

BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv[1]); from arm.assembly_info.client import main; "
             "sys.exit(main(sys.argv[2:]))")


class InfoCell(Cell):
    def get_camera(self, name="overview", depth=True):
        return self._call("get_camera", name=str(name), depth=bool(depth))

    def get_state_privileged(self):
        return self._call("get_state_privileged")

    def describe_privileged(self):
        return self._call("describe_privileged")

    def truth(self):
        return self._call("truth")


def decode_frame(reply):
    """(rgb uint8 H x W x 3, depth float32 H x W in metres with NaN where the sensor reported 0 / None when absent)."""
    import numpy as np
    from PIL import Image
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(reply["rgb_png_b64"]))).convert("RGB"))
    depth = None
    if reply.get("sensor_depth_u16_mm_b64"):
        d = np.asarray(Image.open(io.BytesIO(base64.b64decode(reply["sensor_depth_u16_mm_b64"]))), dtype=np.float32)
        depth = d * float(reply.get("depth_unit_m", 0.001))
        depth[d == float(reply.get("depth_invalid_value", 0))] = np.nan
    return rgb, depth


def connect_stdio():
    c = _connect_stdio()
    cell = InfoCell(c._send, c._recv)
    return cell


def launcher_argv(python, lib_dir, program_dir, module="assemble"):
    run_sh = os.path.join(program_dir, "run.sh")
    if os.path.isfile(run_sh) and not os.path.islink(run_sh):
        return ["bash", "run.sh"]
    from ..assembly_cell.runner import PYTHON_FLAGS
    return [python] + PYTHON_FLAGS + ["-c", BOOTSTRAP, lib_dir, module]


def main(argv=None):
    """Launcher: import assemble.py from the current directory and run solve(cell)."""
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
