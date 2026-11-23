"""In-process practice cell of the information tier (PUBLIC CODE).

    from arm.assembly_info.local import PracticeCell
    cell = PracticeCell(kit_dir, panda_dir, seed=0)          # a practice kit folder (practice/P1 ...), practice seeds 0-19
    cell.describe(); cell.get_camera("top"); cell.step(10); ...
    cell.truth()        # practice only: exact part poses, mate status, every camera's exact CameraSpec, the episode draws
    summary = cell.finalize()

The same server code the grader runs (`cell_info.InfoCellServer`); the practice cell also answers the privileged calls
(`describe_privileged`, `get_state_privileged`), `reset(seed)` and `truth()`, because a practice kit's GT is disclosed.

Budgets (round 3): the practice cell enforces the grader's wall-clock cap and silence rule in-process (the config's
`wall_cap_s` 720 s, `silence_s` 120 s: the time between the end of one call and the start of the next, i.e. your own
computation between requests; rendering inside a call counts towards the wall cap only). The first call after a cap is
exceeded ends the episode (`done_reason` "silence" / "wall_cap", as in the grader's run, where the program is stopped at
that moment) and every later call raises CellError("episode_over: ..."). A cap that is exceeded after the program's last call
is reported by finalize (`cap_exceeded_after_last_call`). The grader's runner also pins the BLAS threads to 2; the
in-process cell cannot (numpy is imported before it): time-critical code should be checked with `practice_grade.py --wire`.
"""

import time

import numpy as np

from ..assembly_cell.client import CellError
from .client import InfoCell as Cell
from .cell_info import InfoCellServer

__all__ = ["PracticeCell", "InfoLocalCell"]


class InfoLocalCell(Cell):
    """An in-process cell on any kit folder (`mode` practice | hidden, `privileged` for the grader's own anchor runs).
    `caps` = {"wall_cap_s", "silence_s"} enforces the grader's time budgets in-process (PracticeCell); None = no caps (the
    grader's anchor runs)."""

    def __init__(self, kit_dir, panda_dir, seed=0, config=None, scope_ids=None, log=None, functional_seed=0, mode="practice",
                 privileged=False, render=True, kit_info=None, caps=None):
        self.server = InfoCellServer(kit_dir, panda_dir, config=config, scope_ids=scope_ids, mode=mode, privileged=privileged,
                                     render=render, log=log, kit_info=kit_info)
        self.functional_seed = functional_seed
        self._log = log or (lambda s: None)
        Cell.__init__(self, None, None)
        self.server.reset(seed)
        self._caps = dict(caps) if caps else None
        self._cap_reason = None
        self._t0 = self._last = time.time()

    def _cap_check(self, now):
        if self._caps is None or self._cap_reason is not None:
            return
        if now - self._t0 > float(self._caps["wall_cap_s"]):
            self._cap_reason = "wall_cap"
        elif now - self._last > float(self._caps["silence_s"]):
            self._cap_reason = "silence"
        if self._cap_reason is not None:
            self.server._end(self._cap_reason)
            self._log("[practice cell] episode over: %s (%.1f s since the last reply, %.1f s since the start; the grader stops the "
                      "program here)" % (self._cap_reason, now - self._last, now - self._t0))

    def _call(self, method, **params):
        now = time.time()
        self.max_silence_s = max(getattr(self, "max_silence_s", 0.0), now - self._last)
        self._cap_check(now)
        if self._cap_reason is not None:
            raise CellError("episode_over: %s (the grader's run stops the program at this point)" % self._cap_reason)
        result = self.server.handle(method, params)
        self._last = time.time()
        if isinstance(result, dict) and "error" in result and len(result) == 1:
            raise CellError(result["error"])
        return result

    def finalize(self, run_functional_check=True):
        late = None
        if self._caps is not None and self._cap_reason is None:
            now = time.time()
            if now - self._last > float(self._caps["silence_s"]) or now - self._t0 > float(self._caps["wall_cap_s"]):
                late = "silence" if now - self._last > float(self._caps["silence_s"]) else "wall_cap"
        rng = np.random.default_rng(self.functional_seed)
        s = self.server.finalize(rng, run_functional_check=run_functional_check)
        if self._caps is not None:
            s["cap_exceeded_after_last_call"] = late
            s["practice_caps"] = dict(self._caps)
        # the longest time the program computed between two requests (the silence rule's quantity) and the wall clock
        s["max_silence_s"] = round(float(getattr(self, "max_silence_s", 0.0)), 2)
        s["episode_wall_s"] = round(float(time.time() - self._t0), 1)
        return s


class PracticeCell(InfoLocalCell):
    def __init__(self, kit_dir, panda_dir, seed=0, **kw):
        if not 0 <= int(seed) <= 19:
            raise ValueError("practice seeds are 0-19")
        kw["mode"] = "practice"
        if "caps" not in kw:
            cfg = kw.get("config") or {}
            from .cell_info import DEFAULT_INFO_CONFIG
            kw["caps"] = {"wall_cap_s": float(cfg.get("wall_cap_s", DEFAULT_INFO_CONFIG["wall_cap_s"])),
                          "silence_s": float(cfg.get("silence_s", DEFAULT_INFO_CONFIG["silence_s"]))}
        InfoLocalCell.__init__(self, kit_dir, panda_dir, seed=seed, **kw)
