"""Reader of the maker's published files, the hidden standard of this task.

The standard is the maker's own files, stored byte for byte under ``reference/maker/`` (three large meshes
xz-compressed, pinned on their uncompressed bytes in ``reference/maker/pins.json``). This package reads them as data
and never executes anything from them. It uses only the standard library and numpy and runs on Python 3.11 and later.

Modules:
    pinned       reads a pinned file (xz transparently) and checks its sha256 and byte count
    dae          COLLADA triangles, node matrices, unit, up axis, normals, UVs and materials as written
    xacro_text   the two robot-description files read as XML text with line numbers (nothing evaluated)
    fk           URDF origin and joint conventions, forward kinematics in pure Python floats
    placement    derives the whole content of reference/placement.json from the four pinned texts
    standard     each tag's maker triangles at its assembled place, in millimetres
    seen_codec   the scoring-sample table format ffcad.seen/1

Any disagreement with a pin or with the derived placement raises MakerStdError: a task error, never a score of 0.
"""
from .errors import MakerStdError

VERSION = "ffcad.makerstd/1"

__all__ = ["MakerStdError", "VERSION"]
