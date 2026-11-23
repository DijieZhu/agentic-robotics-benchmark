"""parcelkit: visual meshes, maps and waybills of the parcel sorting task's parcels.

    from parcelkit import make_parcel
    rec = make_parcel({"kind": "rigid", "size": (0.24, 0.18, 0.12), "label_face": 0}, seed=7,
                      quality=("full", "agent"), out_dir="out/p7")
    rec["waybill"]          # face, centre, direction, paper size, frame: where the simulator puts its label site
    rec["outputs"]["agent"]["files"]   # OBJ, MTL, USDA and the three PNG maps

Modules: ``parcel`` (make_parcel, make_shift_parcels), ``geometry`` (shapes, waybill placement), ``maps`` (atlas and
procedural maps), ``waybill`` (the waybill design and renderer), ``identity`` (the fictional carrier, tracking numbers,
routing codes, address lines), ``code128`` (linear barcode), ``marks`` (pictograms, stamps, stickers), ``export``
(OBJ / USDA / PNG), ``mj`` (MuJoCo MjSpec integration), ``checks`` (mesh versus collision shape).

Dependencies: numpy, scipy, pillow, opencv-python-headless (QR encoder); mujoco only for ``parcelkit.mj``.
"""
from .identity import CARRIER, WaybillIdentity, identities, identity_for_seed
from .parcel import QUALITIES, make_parcel, make_shift_parcels, strip
from .waybill import PAPERS, design as waybill_design

__all__ = ["make_parcel", "make_shift_parcels", "QUALITIES", "strip", "CARRIER", "WaybillIdentity",
           "identities", "identity_for_seed", "PAPERS", "waybill_design"]
__version__ = "0.1.0"
