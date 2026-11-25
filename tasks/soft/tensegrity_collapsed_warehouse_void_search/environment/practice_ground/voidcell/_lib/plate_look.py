"""The agent view's look of the plate backend: the display transform, the robot's materials and the gains
(agent_look.json beside this file, one for every site; steps R5-R8 of plate_filament.py).

    look, look_sha = load_look()                  # agent_look.json and the sha256 of its bytes
    u8 = display(hdr, look)                       # scene-linear HdrColor (the bake's exposure applied) -> sRGB uint8

display() is the ONE display transform of the agent's frame: the bake writes plate_ldr.png = display(plate_lin as
stored in the bundle) with the same agent_look.json, and the backend composes the robot and its shadow in the same
scene-linear space and passes the changed pixels through the same function, so a pixel the robot leaves alone keeps
the plate's byte. The curve is the minimal AgX (the Rec.709 inset matrix, log2 encoding over [-12.47, 4.03] EV, the
6th-order sigmoid fit, the outset matrix, 2.2 decode; look "none"), carried in full here, not imported.
"""
import hashlib
import json
import os

import numpy as np

LOOK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_look.json")
LOOK_FORMAT = "agent-look/1"

_INSET = np.array([[0.842479062253094, 0.0423282422610123, 0.0423756549057051],
                   [0.0784335999999992, 0.878468636469772, 0.0784336],
                   [0.0792237451477643, 0.0791661274605434, 0.879142973793104]])
_OUTSET = np.array([[1.19687900512017, -0.0528968517574562, -0.0529716355144438],
                    [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
                    [-0.0990297440797205, -0.0989611768448433, 1.15107367264116]])
_MIN_EV, _MAX_EV = -12.47393, 4.026069
_SRGB_DECODE = np.where(np.arange(256) / 255.0 <= 0.04045, np.arange(256) / 255.0 / 12.92,
                        ((np.arange(256) / 255.0 + 0.055) / 1.055) ** 2.4).astype(np.float32)

REQUIRED_KEYS = ("format", "display", "robot_layer", "materials", "group_gain", "window", "shadow", "probes")


def load_look(path=None):
    """(look dict, sha256 of the file's bytes). Raises ValueError when a required key is missing."""
    path = path or LOOK_FILE
    with open(path, "rb") as fh:
        raw = fh.read()
    look = json.loads(raw.decode("utf-8"))
    missing = [k for k in REQUIRED_KEYS if k not in look]
    if missing or look.get("format") != LOOK_FORMAT:
        raise ValueError("%s: not an %s file (missing %s)" % (path, LOOK_FORMAT, missing))
    if look["display"].get("curve") != "agx-minimal" or look["display"].get("encode") != "srgb8":
        raise ValueError("%s: display transform %r is not implemented" % (path, look["display"]))
    return look, hashlib.sha256(raw).hexdigest()


def agx(lin):
    """Scene-linear Rec.709 (..., 3) -> display-linear (..., 3): the minimal AgX, look "none"."""
    v = np.maximum(np.asarray(lin, np.float64), 1e-10) @ _INSET
    v = np.clip(np.log2(np.maximum(v, 1e-10)), _MIN_EV, _MAX_EV)
    x = (v - _MIN_EV) / (_MAX_EV - _MIN_EV)
    x2 = x * x
    x4 = x2 * x2
    y = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232
    y = np.clip(y, 0.0, None) @ _OUTSET
    return np.power(np.clip(y, 0.0, None), 2.2)


def srgb_encode(lin):
    """Display-linear -> sRGB uint8 (round half up)."""
    x = np.clip(lin, 0.0, 1.0)
    return (np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055) * 255.0 + 0.5).astype(np.uint8)


def srgb_decode_u8(u8):
    """sRGB uint8 -> linear float32 (lookup table)."""
    return _SRGB_DECODE[np.asarray(u8, np.uint8)]


def display(hdr, look):
    """HdrColor (scene-linear, ... x 3) -> the agent's sRGB uint8: AgX of hdr x the look's fixed gain."""
    return srgb_encode(agx(np.asarray(hdr, np.float64) * float(look["display"]["gain"])))


def box_down(img, ss):
    """Mean over ss x ss blocks (the 4x film -> 1x pixels; the 2 x 2 sub-points -> 1x)."""
    H, W = img.shape[0] // ss, img.shape[1] // ss
    return img[:H * ss, :W * ss].astype(np.float32).reshape(H, ss, W, ss, *img.shape[2:]).mean((1, 3))
