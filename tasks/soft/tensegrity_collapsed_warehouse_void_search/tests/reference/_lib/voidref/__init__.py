"""voidref: the privileged reference program of tensegrity_collapsed_warehouse_void_search (hidden side only).

    planner     hybrid A* over the site's true occupancy with the robot's own primitive table (its code
                unchanged since the first version)
    controller  the legs pad -> 1 -> 2 -> 3 -> pad through the station's calls (voidcell.client.connect_handler on the
                cell's own handle), replanning from the true pose after every primitive
    monitor     what the episode did to the world (contacts with boxes, strut-axis and tendon penetration) and the
                inline score of the run (a cross-check of tests/voidscore)
    prim_table.json  the primitive table the planner uses (the robot's own flat-floor table, ysim 0.3.1) and the
                envelope numbers of the planning radii

1.1.0: the search points' goal tolerances 0.10 / 0.15 m, the void parts
of the site truth in where() and the rows, the blind leg going in (R5) in the summary, the inline score's constants
from the cell (voidcell/_lib/mission.py) and the site's own weights. The planner and prim_table.json are unchanged.
1.1.1: the same program as 1.1.0; only comments and docstrings differ.
"""

VERSION = "1.1.1"
