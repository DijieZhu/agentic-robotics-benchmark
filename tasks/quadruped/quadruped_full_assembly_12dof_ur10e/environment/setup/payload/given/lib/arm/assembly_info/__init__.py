"""assembly_info: the information-tier assembly stage (the camera-only cell).

    cell_info.py     InfoCellServer: the six-servo cell with the restricted camera-only API (describe / get_state without
                     part poses or mates, events limited to safety, noisy wrench, live RGB-D cameras through the render
                     layer, a frame budget, privileged calls only in the grader's own reference runs)
    cell_render.py   the render twin of the cell (workbench, datum boards, lights) and the per-episode render session
    layout_info.py   per-episode draws (layout, proximal posture, horn side, spare slot, cameras, lights, sensor noise)
"""

CELL_VERSION = "arm-assembly-cell-info/1.0.0"
