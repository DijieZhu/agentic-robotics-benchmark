"""An in-process cell for development (PUBLIC CODE): the same API as the wire's `Cell` proxy, answered directly by a
practice `ShoeCellServer` on a seed of your choice, plus `truth()` (every pose, the eyelets, the lace passes, the carton
state and the milestone credits) which the graded cell never answers.

    from shoecell.local import LocalCell
    cell = LocalCell(seed=3)                  # builds the cell; ~2-5 s
    info = cell.describe()
    cell.set_control("left", "cartesian", 0.6)
    cell.command("left", tcp_pos=[-0.1, -0.05, 0.25], tcp_quat=[0, 1, 0, 0], width=0.08)
    cell.step(100)
    rgb, depth = cell.frame("overhead")       # decoded numpy arrays (the wire sends PNGs; see client.decode_frame)
    t = cell.truth()                          # development only

Every call still goes through JSON (encode / decode), so what works here works on the wire.
"""

import json
import os

from .client import Cell, CellError, decode_frame

__all__ = ["LocalCell", "given_dir"]


def given_dir():
    """The givens folder (~/given, or ARB_GIVEN_DIR when set)."""
    return os.environ.get("ARB_GIVEN_DIR", os.path.expanduser("~/given"))


class LocalCell(Cell):
    def __init__(self, seed=0, config=None, given=None, log=None, render=True):
        from .server import ShoeCellServer
        self.server = ShoeCellServer(given or given_dir(), config=config, mode="practice", privileged=False,
                                     allow_reset=True, render=render, log=log)
        Cell.__init__(self, self._send, self._recv)
        self._pending = None
        self.server.reset(int(seed))

    # the wire, in process: a request is encoded, dispatched, and the reply decoded like the real thing
    def _send(self, text):
        req = json.loads(text)
        result = self.server.handle(str(req.get("method")), req.get("params") or {})
        if isinstance(result, dict) and "error" in result and len(result) == 1:
            self._pending = json.dumps({"id": req.get("id"), "error": result["error"]})
        else:
            self._pending = json.dumps({"id": req.get("id"), "result": result})

    def _recv(self):
        line, self._pending = self._pending, None
        return line + "\n"

    def frame(self, name="overhead", depth=True):
        """(rgb uint8, depth float32 metres or None) of one camera, decoded."""
        return decode_frame(self.get_camera(name, depth=depth))

    def summary(self):
        """The episode summary the grader computes (work_score, per-pair credits, events); ends the episode."""
        return self.server.finalize()
