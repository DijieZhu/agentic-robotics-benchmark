"""Pinned maker files: read the stored bytes, decompress the xz ones, check the pins.

pins.json (schema ffcad.maker_pins/1) maps each maker file's logical path under the maker root (for example
"meshes/body.dae") to its entry:
    stored_as      the path of the stored file under the maker root ("meshes/body.dae.xz" or the logical path)
    storage        "raw" (byte copy) or "xz" (the byte copy compressed as one .xz stream)
    bytes, sha256  of the maker's file itself, that is of the uncompressed bytes (the pin)
    stored_bytes, stored_sha256   of the stored file (equal to bytes / sha256 for "raw")
    repository, repository_path, role, licence   where the file comes from (informational)

A pins mapping may also be built by a caller (the reference build keeps its own copy of the files): any dict with
the key "files" holding entries of the form above.
"""
import hashlib
import json
import lzma
import os

from .errors import MakerStdError

SCHEMA = "ffcad.maker_pins/1"
PINS_NAME = "pins.json"
MAX_UNCOMPRESSED = 64 * 1024 * 1024


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_pins(maker_root):
    """The pins of a maker root (its pins.json)."""
    path = os.path.join(maker_root, PINS_NAME)
    try:
        with open(path, "rb") as fh:
            pins = json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise MakerStdError("cannot read the pins %s: %s" % (path, exc))
    if pins.get("schema") != SCHEMA or not isinstance(pins.get("files"), dict):
        raise MakerStdError("%s is not a %s file" % (path, SCHEMA))
    return pins


def entry(pins, logical):
    try:
        return pins["files"][logical]
    except (KeyError, TypeError):
        raise MakerStdError("no pin for the maker file %r" % (logical,))


def read(maker_root, logical, pins=None):
    """The maker's bytes of one pinned file (decompressed when stored as xz), after checking every pin."""
    if pins is None:
        pins = load_pins(maker_root)
    e = entry(pins, logical)
    stored = os.path.join(maker_root, e.get("stored_as", logical))
    try:
        with open(stored, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise MakerStdError("cannot read the pinned file %s: %s" % (stored, exc))
    storage = e.get("storage", "raw")
    if "stored_sha256" in e and sha256_bytes(raw) != e["stored_sha256"]:
        raise MakerStdError("pin mismatch: %s stored sha256 %s, pinned %s" % (e.get("stored_as", logical),
                                                                               sha256_bytes(raw), e["stored_sha256"]))
    if storage == "raw":
        data = raw
    elif storage == "xz":
        try:
            dec = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
            data = dec.decompress(raw, max_length=MAX_UNCOMPRESSED + 1)
        except lzma.LZMAError as exc:
            raise MakerStdError("cannot decompress %s: %s" % (stored, exc))
        if not dec.eof or dec.unused_data or len(data) > MAX_UNCOMPRESSED:
            raise MakerStdError("%s is not one complete xz stream within %d bytes" % (stored, MAX_UNCOMPRESSED))
    else:
        raise MakerStdError("unknown storage %r for %s" % (storage, logical))
    if len(data) != int(e["bytes"]):
        raise MakerStdError("pin mismatch: %s has %d bytes, pinned %d" % (logical, len(data), int(e["bytes"])))
    digest = sha256_bytes(data)
    if digest != e["sha256"]:
        raise MakerStdError("pin mismatch: %s sha256 %s, pinned %s" % (logical, digest, e["sha256"]))
    return data


def read_text(maker_root, logical, pins=None):
    """A pinned text file as str (UTF-8; the maker's description, plugin and launch files are ASCII)."""
    data = read(maker_root, logical, pins)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MakerStdError("%s is not UTF-8 text: %s" % (logical, exc))


def check_all(maker_root, pins=None):
    """Read every pinned file once; returns {logical: sha256}. Raises on the first mismatch."""
    if pins is None:
        pins = load_pins(maker_root)
    return {logical: sha256_bytes(read(maker_root, logical, pins)) for logical in sorted(pins["files"])}
