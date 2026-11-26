"""Pipe spool push-fit assembly: the cameras' material library (plain look dictionaries; colours linear).

  base  [r, g, b] linear            col  (map of the render bundle, linear mean the map is tinted to, or None)
  metal, rough                      rgh  (unused, mean roughness, unused): the mean is the surface's roughness
  tile  metres per repeat of a map (part uv are metres); uv "own": the part's uv run 0..1 over the map
"""


def S(r, g, b):
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]


STEEL = [0.62, 0.615, 0.60]

LOOKS = {
    # ---- the bench: a worn wooden worktop on a blue steel frame, adapter plates under the robots
    "bench_top": {"rgh": (None, 0.45, 1.0), "col": ("bench_wood_col", None), "uv": "own"},
    "bench_edge": {"base": S(0.66, 0.665, 0.655), "rgh": (None, 0.46, 0.7), "tile": 0.03},
    "bench_steel": {"base": S(0.16, 0.24, 0.36), "rgh": (None, 0.42, 0.9), "tile": 0.04},
    "plate_steel": {"base": [0.36, 0.36, 0.35], "metal": 0.6, "rgh": (None, 0.50, 0.5), "tile": 0.3},
    "foot": {"base": [0.02, 0.02, 0.022], "rgh": (None, 0.7, 0.8), "tile": 0.02},
    "screw": {"base": [0.55, 0.55, 0.56], "metal": 1.0, "rough": 0.3},
    # ---- the parts: 22 mm copper tube, a blue tape band, forged brass push-fit elbows with acetal collars
    "copper": {"base": [0.62, 0.20, 0.09], "metal": 1.0, "rgh": (None, 0.30, 0.6), "tile": 0.02},
    "band_tape": {"base": S(0.12, 0.40, 1.00), "rgh": (None, 0.42, 0.5), "tile": 0.02},
    "fit_brass": {"base": [0.72, 0.50, 0.14], "metal": 0.7, "rgh": (None, 0.44, 0.9), "tile": 0.03},
    "fit_collar": {"base": S(0.16, 0.17, 0.19), "rgh": (None, 0.42, 0.7), "tile": 0.02},
    "fit_bore": {"base": [0.30, 0.21, 0.09], "metal": 1.0, "rough": 0.5},
    "oring": {"base": [0.012, 0.012, 0.013], "rgh": (None, 0.6, 0.8), "tile": 0.01},
    "bin_blue": {"base": S(0.05, 0.16, 0.36), "rgh": (None, 0.38, 0.6), "tile": 0.05},
    # ---- the camera rig
    "post_alu": {"base": [0.70, 0.71, 0.73], "metal": 1.0, "rgh": (None, 0.32, 0.5), "tile": 0.03},
    "boss_black": {"base": S(0.06, 0.06, 0.065), "metal": 1.0, "rgh": (None, 0.42, 0.5), "tile": 0.05},
    "cam_body": {"base": S(0.07, 0.07, 0.075), "metal": 1.0, "rgh": (None, 0.45, 0.5), "tile": 0.04},
    "lens_barrel": {"base": S(0.03, 0.03, 0.032), "metal": 1.0, "rgh": (None, 0.36, 0.5), "tile": 0.01},
    "lens_glass": {"base": [0.01, 0.012, 0.02], "rough": 0.04},
    "stand_black": {"base": S(0.05, 0.05, 0.055), "rgh": (None, 0.42, 0.8), "tile": 0.03},
    "stand_chrome": {"base": [0.72, 0.72, 0.73], "metal": 1.0, "rough": 0.12},
    # ---- the Pandas
    "panda_white": {"base": S(0.93, 0.93, 0.92), "rgh": (None, 0.36, 1.0), "tile": 0.08},
    "panda_black": {"base": S(0.06, 0.06, 0.065), "rgh": (None, 0.52, 1.0), "tile": 0.08},
    "panda_grey": {"base": S(0.25, 0.25, 0.26), "rgh": (None, 0.45, 1.0), "tile": 0.08},
    "panda_blue": {"base": S(0.04, 0.54, 0.78), "rough": 0.35},
    "panda_led_green": {"base": S(0.10, 0.75, 0.20), "rough": 0.35},
    "panda_led_red": {"base": S(0.80, 0.08, 0.06), "rough": 0.35},
    "panda": {"base": [1.0, 1.0, 1.0], "rgh": (None, 0.40, 1.0), "uv": "own"},       # the links: colours from the palette map (spool_scene)
    "finger": {"base": [0.70, 0.71, 0.73], "metal": 1.0, "rgh": (None, 0.36, 0.5), "tile": 0.06},
    "finger_pad": {"base": [0.022, 0.022, 0.024], "rgh": (None, 0.70, 0.8), "tile": 0.02},
    "default": {"base": [0.5, 0.5, 0.5], "rough": 0.5},
}
