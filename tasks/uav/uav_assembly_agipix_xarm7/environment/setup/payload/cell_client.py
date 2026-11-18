"""Workcell connection used by assemble.solve(cell). All motion is remote."""
import base64
import io
import json
import sys


class CellClient:
    def __init__(self, reader=None, writer=None):
        self.reader = reader or sys.stdin
        self.writer = writer or sys.stdout
        self._final_state = None

    def _call(self, op, **kw):
        self.writer.write(json.dumps(dict(op=op, **kw), allow_nan=False) + "\n")
        self.writer.flush()
        line = self.reader.readline()
        if not line:
            raise EOFError("Workcell closed")
        r = json.loads(line)
        if not r.get("ok"):
            raise RuntimeError(r.get("error", "Workcell request rejected"))
        value = r.get("result")
        if isinstance(value, dict) and value.get("done"):
            self._final_state = value
        return value

    def describe(self):
        return self._call("describe")

    def get_state(self):
        return self._call("state")

    def set_control(self, q, gripper_width_m):
        return self._call("control", q=list(q), gripper_width_m=float(gripper_width_m))

    def step(self, ticks=1):
        return self._call("step", ticks=int(ticks))

    def get_camera(self, name="top"):
        import numpy as np
        from PIL import Image
        r = self._call("camera", name=name)
        rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(r.pop("rgb_png")))))
        depth = np.asarray(Image.open(io.BytesIO(base64.b64decode(r.pop("depth_png"))))).astype(np.float32) / 1000
        return dict(r, rgb=rgb, depth_m=depth)

    def finish(self):
        if self._final_state is not None:
            return self._final_state
        return self._call("finish")
