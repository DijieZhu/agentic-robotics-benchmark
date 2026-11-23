"""parcelcell: the program's side of the label-facing station's wire.

The station's cell runs in the grader and in the commissioning service; a program gets this client only:
``parcelcell.client`` (the ``Cell`` proxy that ``solve(cell)`` receives, ``decode_frame``, the launcher and
``connect_commissioning``). See ~/given/cell/cell_api.md.
"""

CELL_VERSION = "parcelcell-1.0"
