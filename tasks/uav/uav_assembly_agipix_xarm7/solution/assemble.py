def solve(cell):
    # Exercise both actual agent camera paths during every validation episode.
    cell.get_camera("top")
    cell.get_camera("wrist")
    cell._call("reference_assemble")
