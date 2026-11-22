#!/usr/bin/env python3
"""PR #158 OVRTX look: the material library (usdw.py's plain look dictionaries; colours linear, S() converts sRGB).
Parts that carry a task colour keep it: a SKU's body, band and logo are its atlas (tex158.py draws them in the task's
values), the code label its paper and ink, the cap its colour. The bay's own colours (the task: light grey boards,
dark lips, grey cart, blue tote) stay in their families.
"""
import os

def S(r, g, b):
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]

STEEL = [0.62, 0.615, 0.60]

LOOKS = {
    # ---- the gondola: powder-coated steel, a perforated back, a dark ticket rail in place of the lip
    "gondola": {"base": S(0.80, 0.81, 0.80), "rgh": ("pc_rgh", 0.42, 0.8), "nrm": "pc_nrm", "nrm_strength": 0.35, "tile": 0.05},
    "gondola_deck": {"base": S(0.84, 0.85, 0.84), "rgh": ("pc_rgh", 0.38, 0.8), "nrm": "pc_nrm", "nrm_strength": 0.3, "tile": 0.05},
    "gondola_back": {"col": ("perf_col", S(0.78, 0.79, 0.78)), "rgh": ("pc_rgh", 0.45, 0.6), "nrm": "perf_nrm", "nrm_strength": 1.0, "tile": 0.1},
    "gondola_kick": {"base": S(0.22, 0.23, 0.25), "rgh": ("pc_rgh", 0.5, 0.8), "nrm": "pc_nrm", "nrm_strength": 0.3, "tile": 0.05},
    "rail": {"base": S(0.20, 0.22, 0.25), "rgh": ("pc_rgh", 0.40, 0.8), "tile": 0.05},
    "ticket": {"col": ("ticket_col", None), "rough": 0.18, "coat": 0.6, "coat_rough": 0.06, "uv": "own"},
    # ---- products
    "can_metal": {"base": [0.80, 0.80, 0.79], "metal": 1.0, "rgh": ("brushed_rgh", 0.24, 0.4), "tile": 0.05},
    # ---- the cart and the tote
    "cart_deck": {"base": S(0.36, 0.36, 0.40), "rgh": ("pc_rgh", 0.46, 0.8), "nrm": "pc_nrm", "nrm_strength": 0.25, "tile": 0.06},
    "cart_shelf": {"base": S(0.30, 0.30, 0.33), "rgh": ("pc_rgh", 0.5, 0.8), "tile": 0.06},
    "cart_frame": {"base": STEEL, "metal": 1.0, "rgh": ("brushed_rgh", 0.28, 0.5), "nrm": "brushed_nrm", "nrm_strength": 0.4, "tile": 0.1},
    "castor": {"base": [0.03, 0.03, 0.032], "rgh": ("rubber_rgh", 0.62, 0.8), "nrm": "rubber_nrm", "nrm_strength": 0.5, "tile": 0.03},
    "grip": {"base": [0.02, 0.02, 0.022], "rgh": ("rubber_rgh", 0.7, 0.8), "nrm": "rubber_nrm", "nrm_strength": 0.8, "tile": 0.02},
    "tote": {"base": S(0.20, 0.30, 0.55), "rgh": ("pc_rgh", 0.36, 1.0), "nrm": "peel_nrm", "nrm_strength": 0.25, "tile": 0.04,
             "sss": 0.08, "sss_color": S(0.25, 0.36, 0.62), "sss_scale": 0.002},
    # ---- the lift and what rides on it
    "steel_plate": {"base": STEEL, "metal": 1.0, "rgh": ("brushed_rgh", 0.30, 0.5), "nrm": "brushed_nrm", "nrm_strength": 0.6, "tile": 0.3,
                    "aniso": 0.5, "aniso_rot": 0.25},
    "bolt": {"base": [0.55, 0.55, 0.56], "metal": 1.0, "rough": 0.32},
    "lift_alu": {"base": [0.74, 0.75, 0.76], "metal": 1.0, "rgh": ("alu_rough_rgh", 0.34, 0.5), "nrm": "brushed_nrm", "nrm_strength": 0.3, "tile": 0.12},
    "lift_collar": {"base": S(0.10, 0.10, 0.11), "rgh": ("pc_rgh", 0.5, 0.8), "nrm": "pc_nrm", "nrm_strength": 0.3, "tile": 0.05},
    "cam_body": {"base": S(0.08, 0.08, 0.09), "rgh": ("pc_rgh", 0.42, 0.8), "nrm": "pc_nrm", "nrm_strength": 0.3, "tile": 0.03},
    "cam_arm": {"base": S(0.05, 0.05, 0.055), "rgh": ("pc_rgh", 0.38, 0.8), "tile": 0.03},
    "cam_lens": {"base": [0.02, 0.02, 0.022], "metal": 1.0, "rough": 0.3},
    "cam_glass": {"glass": {"color": [0.55, 0.6, 0.75], "ior": 1.5, "rough": 0.0, "thin": True}},
    # ---- the Panda (pr138's)
    "panda_white": {"base": S(0.93, 0.93, 0.92), "rgh": ("shell_rgh", 0.36, 1.0), "nrm": "shell_nrm", "nrm_strength": 0.4, "tile": 0.08,
                    "coat": 0.6, "coat_rough": 0.08},
    "panda_black": {"base": S(0.06, 0.06, 0.065), "rgh": ("shell_rgh", 0.52, 1.0), "nrm": "shell_nrm", "nrm_strength": 0.3, "tile": 0.08},
    "panda_grey": {"base": S(0.25, 0.25, 0.26), "rgh": ("shell_rgh", 0.45, 1.0), "nrm": "shell_nrm", "nrm_strength": 0.3, "tile": 0.08},
    "panda_blue": {"base": S(0.04, 0.54, 0.78), "rough": 0.35},
    "panda_led_green": {"base": S(0.10, 0.75, 0.20), "rough": 0.35},
    "panda_led_red": {"base": S(0.80, 0.08, 0.06), "rough": 0.35},
    "finger": {"base": [0.70, 0.71, 0.73], "metal": 1.0, "rgh": ("alu_rough_rgh", 0.36, 0.5), "nrm": "alu_rough_nrm", "nrm_strength": 0.6, "tile": 0.06},
    "finger_pad": {"base": [0.022, 0.022, 0.024], "rgh": ("rubber_rgh", 0.70, 0.8), "nrm": "rubber_nrm", "nrm_strength": 0.9, "tile": 0.02, "spec": 0.6},
    # ---- the store
    "floor": {"col": ("floor_col", [0.40, 0.385, 0.355]), "rgh": ("floor_rgh", None, 1.0), "nrm": "floor_nrm", "nrm_strength": 0.6, "tile": 2.4,
              "coat": 0.35, "coat_rough": 0.10},
    "ceiling": {"col": ("ceil_col", [0.50, 0.50, 0.49]), "rough": 0.9, "nrm": "ceil_nrm", "nrm_strength": 0.8, "tile": 1.2},
    "wall": {"base": S(0.74, 0.73, 0.70), "rgh": ("pc_rgh", 0.62, 0.6), "nrm": "peel_nrm", "nrm_strength": 0.3, "tile": 0.5},
    "wall_band": {"base": S(0.16, 0.24, 0.32), "rgh": ("pc_rgh", 0.5, 0.6), "tile": 0.5},
    "door_steel": {"base": STEEL, "metal": 1.0, "rgh": ("brushed_rgh", 0.30, 0.6), "nrm": "brushed_nrm", "nrm_strength": 0.5, "tile": 0.4},
    "door_glass": {"glass": {"color": [0.80, 0.86, 0.86], "ior": 1.5, "rough": 0.04, "thin": True}},
    "rubber_black": {"base": [0.025, 0.025, 0.027], "rgh": ("rubber_rgh", 0.66, 0.8), "tile": 0.05},
    "fixture": {"base": S(0.88, 0.88, 0.87), "rgh": ("pc_rgh", 0.45, 0.6), "tile": 0.1},
    "fixture_lit": {"base": [1.0, 1.0, 1.0], "rough": 0.6, "emission": 2600.0, "emission_color": [1.0, 0.97, 0.92]},
    "sign": {"col": ("sign_col", None), "rough": 0.45, "uv": "own"},
    "sign_edge": {"base": S(0.15, 0.23, 0.31), "rough": 0.45},
    "wire": {"base": [0.6, 0.6, 0.6], "metal": 1.0, "rough": 0.3},
    "carton": {"col": ("carton_col", None), "rough": 0.8, "tile": 0.3},
    "default": {"base": [0.5, 0.5, 0.5], "rough": 0.5},
}


def sku_look(tex, name, sku):
    """a SKU's print: its atlas; a carton's satin varnish, a can's or bottle's glossier sleeve"""
    box = sku["kind"] == "box"
    # a plain roughness: a roughness map per SKU cost about 0.5 GB of GPU memory on a GPU shared with other jobs
    return {"col_path": os.path.join(tex, "sku_%s_col.png" % name), "rough": 0.36 if box else 0.28,
            "uv": "own", "coat": 0.25 if box else 0.6, "coat_rough": 0.22 if box else 0.07, "ior": 1.5}


def code_look(tex, code):
    return {"col_path": os.path.join(tex, "code_%02d.png" % code), "rough": 0.62, "uv": "own", "spec": 0.5}
