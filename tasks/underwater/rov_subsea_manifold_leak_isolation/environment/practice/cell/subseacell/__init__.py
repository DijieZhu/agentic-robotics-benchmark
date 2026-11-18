"""subseacell: the vehicle cell of the subsea intervention task.

A program talks to the vehicle through `subseacell.client` (the `Vehicle` proxy, the launcher and the practice
connections); nothing else of the package is needed on the program's side.  This file imports nothing, so
`import subseacell.client` loads only the client.
"""

CELL_VERSION = "1.0.0"

__all__ = ["CELL_VERSION"]
