"""Harness cell: the cameras' material library (plain look dictionaries; colours linear).

  base  [r, g, b] linear            col  (map of the render bundle, linear mean the map is tinted to, or None)
  metal, rough                      rgh  (unused, mean roughness, unused): the mean is the surface's roughness
  tile  metres per repeat of a map (part uv are metres); uv "own": the part's uv run 0..1 over the map
The cable keeps the task's colour (an orange jacket); a clip's lead-in fins are black nylon (off-white fins on the off-white board gave
a CIELAB distance of 5 to 6 in the overhead camera), its retaining tabs off-white nylon, on steel pins; a post is anodised with a black cap.
"""


def S(r, g, b):
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]


STEEL = [0.56, 0.57, 0.58]

LOOKS = {
    # ---- the bench: a wooden worktop on a steel frame
    'bench_top': {'rgh': (None, 0.42, 1.0), 'col': ('bench_wood_col', None), 'uv': 'own'},
    'bench_steel': {'base': S(0.16, 0.24, 0.36), 'rgh': (None, 0.42, 0.9), 'tile': 0.04},
    'bench_shelf': {'base': S(0.55, 0.56, 0.57), 'metal': 1.0, 'rgh': (None, 0.38, 0.5), 'tile': 0.3},
    'foot': {'base': [0.02, 0.02, 0.022], 'rough': 0.7},
    # ---- the form board: a laminate sheet printed with the harness drawing (the print is the episode's own image)
    'board': {'rgh': (None, 0.38, 1.0), 'uv': 'own'},
    'board_edge': {'base': S(0.62, 0.50, 0.34), 'rough': 0.6},
    'screw': {'base': [0.55, 0.55, 0.56], 'metal': 1.0, 'rough': 0.3},
    # ---- fixtures
    'pin_steel': {'base': STEEL, 'metal': 1.0, 'rgh': (None, 0.24, 0.5), 'tile': 0.02},
    'fin_nylon': {'base': S(0.07, 0.07, 0.08), 'rgh': (None, 0.46, 0.6), 'tile': 0.03},
    'tab_nylon': {'base': S(0.80, 0.79, 0.74), 'rgh': (None, 0.46, 0.6), 'tile': 0.03},
    'post_alu': {'base': S(0.42, 0.36, 0.29), 'metal': 1.0, 'rgh': (None, 0.30, 0.5), 'tile': 0.03},
    'post_cap': {'base': S(0.10, 0.10, 0.11), 'rgh': (None, 0.55, 0.6), 'tile': 0.02},
    # ---- the connector, its bracket and the strain-relief boot; the cable
    'conn_body': {'base': S(0.07, 0.07, 0.08), 'rgh': (None, 0.36, 0.6), 'tile': 0.02},
    'conn_boot': {'base': S(0.05, 0.05, 0.055), 'rgh': (None, 0.62, 0.7), 'tile': 0.01},
    'bracket': {'base': STEEL, 'metal': 1.0, 'rgh': (None, 0.30, 0.5), 'tile': 0.1},
    'cable': {'base': S(0.85, 0.30, 0.10), 'rgh': (None, 0.30, 0.5), 'tile': 0.02},
    'cable_end': {'base': [0.72, 0.45, 0.25], 'metal': 1.0, 'rough': 0.28},
    # ---- the bench's things
    'reel_flange': {'base': S(0.78, 0.66, 0.46), 'rgh': (None, 0.66, 0.8), 'tile': 0.03},
    'reel_cable_blue': {'base': S(0.10, 0.24, 0.62), 'rgh': (None, 0.30, 0.5), 'tile': 0.02},
    'reel_cable_black': {'base': S(0.04, 0.04, 0.045), 'rgh': (None, 0.30, 0.5), 'tile': 0.02},
    'bin_blue': {'base': S(0.08, 0.22, 0.52), 'rgh': (None, 0.36, 0.7), 'tile': 0.04},
    'bin_yellow': {'base': S(0.86, 0.66, 0.08), 'rgh': (None, 0.36, 0.7), 'tile': 0.04},
    'tape': {'base': S(0.05, 0.05, 0.05), 'rgh': (None, 0.32, 0.5), 'tile': 0.02},
    # ---- the robot: the Panda's shells and the cable jaws
    'panda_white': {'base': S(0.93, 0.93, 0.92), 'rough': 0.36},
    'panda_black': {'base': S(0.06, 0.06, 0.065), 'rough': 0.52},
    'panda_grey': {'base': S(0.25, 0.25, 0.26), 'rough': 0.45},
    'panda_blue': {'base': S(0.04, 0.54, 0.78), 'rough': 0.35},
    'panda_led_green': {'base': S(0.10, 0.75, 0.20), 'rough': 0.35},
    'panda_led_red': {'base': S(0.80, 0.08, 0.06), 'rough': 0.35},
    'finger': {'base': [0.70, 0.71, 0.73], 'metal': 1.0, 'rgh': (None, 0.34, 0.5), 'tile': 0.06},
    'finger_pad': {'base': [0.022, 0.022, 0.024], 'rough': 0.70},
    'jaw': {'base': S(0.09, 0.09, 0.10), 'rgh': (None, 0.40, 0.7), 'tile': 0.03},
    'jaw_face': {'base': S(0.03, 0.03, 0.035), 'rough': 0.75},
    'default': {'base': [0.5, 0.5, 0.5], 'rough': 0.5},
}
