"""In-process cell for practice and authoring: the same CellServer behind the same `Cell` API, no wire.

    from arm.assembly_cell.local import LocalCell
    cell = LocalCell(a2_dir, panda_dir, assembly_json, seed=0)   # practice: reset(seed) allowed
    cell.describe(); cell.step(10); cell.finish(); summary = cell.finalize()
"""

import numpy as np

from .client import Cell, CellError
from .server import CellServer

__all__ = ["LocalCell"]


class LocalCell(Cell):
    def __init__(self, a2_dir, panda_dir, assembly_json=None, seed=0, config=None, scope_ids=None, log=None, functional_seed=0):
        self.server = CellServer(a2_dir, panda_dir, assembly_json, config=config, scope_ids=scope_ids, allow_reset=True, log=log)
        self.functional_seed = functional_seed
        Cell.__init__(self, None, None)
        self.server.reset(seed)

    def _call(self, method, **params):
        result = self.server.handle(method, params)
        if isinstance(result, dict) and "error" in result and len(result) == 1:
            raise CellError(result["error"])
        return result

    def finalize(self, run_functional_check=True):
        rng = np.random.default_rng(self.functional_seed)
        return self.server.finalize(rng, run_functional_check=run_functional_check)
