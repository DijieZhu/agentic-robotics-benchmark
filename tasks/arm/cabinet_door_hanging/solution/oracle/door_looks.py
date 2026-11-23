"""Cabinet door hanging: the cameras' material library (plain look dictionaries; colours linear).

  base  [r, g, b] linear            col  (map of the render bundle, linear mean the map is tinted to, or None)
  metal, rough                      rgh  (unused, mean roughness, unused): the mean is the surface's roughness
  tile  metres per repeat of a map (part uv are metres); uv "own": the part's uv run 0..1 over the map
A part that carries a task colour keeps it in its family: the new door is the light lacquered one, the old door the
dark one, the doors' knuckles brass, the carcass's steel.
"""


def S(r, g, b):
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]


LOOKS = {
    # ---- the bench (the task's table block: an oak stave worktop on a lacquered cabinet body with a plinth)
    'bench_top': {'rgh': (None, None, 1.0), 'col': ('stave_col', None), 'uv': 'own'},
    'bench_body': {'base': [0.6038, 0.6209, 0.5705], 'rgh': (None, 0.36, 0.6), 'tile': 0.08},
    'bench_plinth': {'base': [0.01, 0.01, 0.0116], 'rgh': (None, 0.5, 0.8), 'tile': 0.06},
    # ---- the wall cabinet, the two doors and their pulls
    'carcass': {'rgh': (None, 0.42, 0.8), 'col': ('stave_col', [0.448, 0.3185, 0.196]), 'tile': 0.9},
    'door_new': {'base': [0.7484, 0.6739, 0.5071], 'rgh': (None, 0.24, 0.4), 'tile': 0.06},
    'door_old': {'rgh': (None, 0.5, 1.0), 'col': ('stave_col', [0.1706, 0.089, 0.047]), 'tile': 0.7},
    'handle': {'base': [0.045, 0.045, 0.05], 'metal': 1.0, 'rgh': (None, 0.38, 0.5), 'tile': 0.1},
    # ---- hinges: turned knuckles, leaves and screws (steel on the carcass, brass on the new door, tarnished on the old one)
    'knuckle_steel': {'base': [0.5, 0.5, 0.52], 'metal': 1.0, 'rgh': (None, 0.28, 0.42), 'tile': 0.04},
    'knuckle_brass': {'base': [0.78, 0.57, 0.24], 'metal': 1.0, 'rgh': (None, 0.24, 0.5), 'tile': 0.05},
    'knuckle_dark': {'base': [0.16, 0.13, 0.1], 'metal': 1.0, 'rgh': (None, 0.48, 0.6), 'tile': 0.05},
    'leaf_steel': {'base': [0.52, 0.52, 0.53], 'metal': 1.0, 'rgh': (None, 0.34, 0.5), 'tile': 0.06},
    'leaf_brass': {'base': [0.78, 0.57, 0.24], 'metal': 1.0, 'rgh': (None, 0.28, 0.5), 'tile': 0.06},
    'leaf_dark': {'base': [0.16, 0.13, 0.1], 'metal': 1.0, 'rgh': (None, 0.5, 0.6), 'tile': 0.05},
    'screw': {'base': [0.7, 0.7, 0.71], 'metal': 1.0, 'rough': 0.28},
    'screw_dark': {'base': [0.14, 0.12, 0.1], 'metal': 1.0, 'rough': 0.5},
    # ---- pins (turned steel; the old door's two are tarnished), the detent groove under a pin's head, stands and racks
    'pin_steel': {'base': [0.5, 0.5, 0.52], 'metal': 1.0, 'rgh': (None, 0.3, 0.5), 'tile': 0.04},
    'pin_old': {'base': [0.2, 0.15, 0.11], 'metal': 1.0, 'rgh': (None, 0.55, 0.7), 'tile': 0.03},
    'groove': {'base': [0.06, 0.06, 0.065], 'metal': 1.0, 'rough': 0.5},
    'stand_steel': {'base': [0.047, 0.0509, 0.0593], 'rgh': (None, 0.45, 1.0), 'tile': 0.03},
    'rack_steel': {'base': [0.0331, 0.0363, 0.0432], 'rgh': (None, 0.45, 1.0), 'tile': 0.03},
    # ---- the vacuum tool: bracket, cup and bellows, push-in fitting, hose and clips
    'cup_alu': {'base': [0.7, 0.71, 0.73], 'metal': 1.0, 'rgh': (None, 0.34, 0.5), 'tile': 0.06},
    'cup_rubber': {'base': [0.02, 0.02, 0.022], 'rgh': (None, 0.66, 0.8), 'tile': 0.02},
    'fitting': {'base': [0.74, 0.66, 0.4], 'metal': 1.0, 'rough': 0.25},
    'hose': {'base': [0.0039, 0.0397, 0.3185], 'rough': 0.38},
    'clip': {'base': [0.03, 0.03, 0.035], 'rough': 0.5},
    # ---- the fitter's things on the bench, the felt pieces, the tool bag, the wall rail
    'level_alu': {'base': [0.6038, 0.0134, 0.01], 'rgh': (None, 0.32, 0.6), 'tile': 0.06},
    'vial': {'base': [0.448, 0.6921, 0.0331], 'rough': 0.08},
    'rubber_black': {'base': [0.02, 0.02, 0.022], 'rgh': (None, 0.66, 0.8), 'tile': 0.02},
    'mallet_head': {'base': [0.7874, 0.7484, 0.6038], 'rgh': (None, 0.55, 0.6), 'tile': 0.03},
    'ash': {'rgh': (None, 0.45, 0.8), 'col': ('stave_col', [0.6038, 0.42, 0.214]), 'tile': 0.5},
    'tool_steel': {'base': [0.45, 0.45, 0.47], 'metal': 1.0, 'rgh': (None, 0.32, 0.5), 'tile': 0.04},
    'tape_yellow': {'base': [0.89, 0.5225, 0.0039], 'rgh': (None, 0.35, 0.6), 'tile': 0.03},
    'pencil': {'base': [0.6921, 0.2633, 0.01], 'rough': 0.4},
    'felt': {'base': [0.0509, 0.089, 0.0637], 'rgh': (None, 0.92, 0.5), 'tile': 0.01},
    'canvas': {'base': [0.1065, 0.089, 0.055], 'rgh': (None, 0.85, 0.6), 'tile': 0.012},
    'strap': {'base': [0.01, 0.01, 0.0116], 'rgh': (None, 0.7, 0.6), 'tile': 0.02},
    'rail_steel': {'base': [0.62, 0.62, 0.63], 'metal': 1.0, 'rgh': (None, 0.42, 0.5), 'tile': 0.06},
    # ---- the Pandas (their own visual meshes; the gripper's tip blocks drawn as soft pads)
    'panda_white': {'base': S(0.93, 0.93, 0.92), 'rough': 0.36},
    'panda_black': {'base': S(0.06, 0.06, 0.065), 'rough': 0.52},
    'panda_grey': {'base': S(0.25, 0.25, 0.26), 'rough': 0.45},
    'panda_blue': {'base': S(0.04, 0.54, 0.78), 'rough': 0.35},
    'panda_led_green': {'base': S(0.10, 0.75, 0.20), 'rough': 0.35},
    'panda_led_red': {'base': S(0.80, 0.08, 0.06), 'rough': 0.35},
    'finger': {'base': [0.70, 0.71, 0.73], 'metal': 1.0, 'rough': 0.36},
    'finger_pad': {'base': [0.022, 0.022, 0.024], 'rough': 0.70},
    'default': {'base': [0.5, 0.5, 0.5], 'rough': 0.5},
}
