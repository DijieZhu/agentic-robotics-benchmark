"""house_client: the program-side library of the house cell (the proxy, frame decoding, the transport).

    from house_client import Cell, CellError, decode_frame, load_frame, given_dir

The graded run hands solve(cell) a connected Cell; load_frame() reads a survey photo as a get_camera reply.
"""

from .client import BOOTSTRAP, PUBLIC_CAMERAS, Cell, CellError, connect_stdio, decode_frame, given_dir, load_frame, main

__all__ = ["Cell", "CellError", "decode_frame", "load_frame", "connect_stdio", "given_dir", "main", "BOOTSTRAP",
           "PUBLIC_CAMERAS"]
