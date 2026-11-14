"""skidcell: the pump skid cell (cell internals: the grader and the agent's commissioning service run this package, the agent gets
only client.py; the grader runs the hidden sites, the commissioning service its development sites, and only the grader
holds the hidden reference value and the reference program).

Modules
    client      the agent-side proxy of the cell and the program launcher (JSON-lines wire on stdio)
    layout      the episode's skid drawn from its seed (arrangement, device styles, gauge types, events)
    devices     MjSpec builders of the valves, buttons, lamps, gauges, the rotameter, level gauges, tanks and pumps
    textures    the tag plates, dial faces and scales (Pillow, in memory)
    process     the lumped process model and the local control station's logic
    physics     the scene: the skid, the rail with the Panda, the grip rule, the instruments' visuals
    control     the arm's joint / Cartesian controller (damped least-squares differential IK)
    cameras     RGB-D rendering, intrinsics and extrinsics
    look        the agent cameras' colour (Filament, the dressed water-treatment room) and depth (classic) render twins
    episode     the operating procedure's checks and the episode score
    server      the cell API served over the wire (public calls; privileged calls for the reference program)
"""

CELL_VERSION = "pump-skid-cell/0.6.0"
