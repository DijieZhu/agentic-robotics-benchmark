"""The reference submission of the humanoid assembly cell: solve(cell) runs the privileged reference
program of this folder (href/) through the agent's own API over the wire, with the true poses from the privileged
calls of the oracle token. It takes one frame of each camera first."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from href import runner  # noqa: E402


def solve(cell):
    runner.solve(cell)
