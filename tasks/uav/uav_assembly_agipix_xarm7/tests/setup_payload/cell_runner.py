"""Launch the submitted workcell program; stdout is reserved for the connection. The supplied library (given/lib
beside this file: cell_render, render) is importable after the program's own folder."""
import importlib.util
import os
from pathlib import Path
import sys
from cell_client import CellClient

GIVEN_LIB = Path(__file__).resolve().parent / "given" / "lib"


def main():
    directory = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(directory))
    if GIVEN_LIB.is_dir():
        sys.path.insert(1, str(GIVEN_LIB))
    spec = importlib.util.spec_from_file_location("assemble", directory / "assemble.py")
    module = importlib.util.module_from_spec(spec)
    # The connection keeps the original stdout; file descriptor 1 then points at stderr, so neither ordinary prints
    # nor native libraries' log lines (a renderer's, a simulator's) can corrupt it.
    sys.stdout.flush()
    connection = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    client = CellClient(sys.stdin, connection)
    sys.stdout = sys.stderr
    spec.loader.exec_module(module)
    module.solve(client)
    client.finish()


if __name__ == "__main__":
    main()
