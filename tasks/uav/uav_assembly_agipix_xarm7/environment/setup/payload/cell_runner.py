"""Launch the submitted workcell program; stdout is reserved for the connection."""
import importlib.util
from pathlib import Path
import sys
from cell_client import CellClient


def main():
    directory = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("assemble", directory / "assemble.py")
    module = importlib.util.module_from_spec(spec)
    client = CellClient(sys.stdin, sys.stdout)
    # Ordinary prints from the user's program cannot corrupt the connection.
    sys.stdout = sys.stderr
    spec.loader.exec_module(module)
    module.solve(client)
    client.finish()


if __name__ == "__main__":
    main()
