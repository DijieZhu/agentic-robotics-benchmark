"""The access token of the task's solution, standard library only.

    want = read_hash(path)                     # the stage's copy of the token's sha256; a defect is the grader's own
    ok = holds_token(folder, name, want)       # True only for a regular file whose stripped bytes hash to want

The solution's controller folder holds a 256-bit token; the stage holds only its sha256. A folder that holds the
token receives, at set-up, the session's registered reads (grader/fly.py); every other folder is flown exactly as
before. A missing, wrong or malformed token is not an error and changes nothing else: holds_token never raises. The
comparison takes the same time whatever the bytes (hmac.compare_digest on two digests of equal length).
"""
import hashlib
import hmac
import os
import re
import stat as statmod

_HEX64 = re.compile(r"[0-9a-f]{64}")


def read_hash(path):
    """the token's sha256 as 64 lower-case hex characters from the stage's file; ValueError if the file is missing,
    a link, not a regular file or not one such line (the grader's own defect: the verification is void)."""
    st = os.lstat(path)
    if not statmod.S_ISREG(st.st_mode):
        raise ValueError("the token's hash file is not a regular file")
    with open(path, "rb") as fh:
        text = fh.read(4096).decode("ascii", "replace").strip()
    if not _HEX64.fullmatch(text):
        raise ValueError("the token's hash file does not hold one sha256 in hex")
    return text


def holds_token(folder, name, want):
    """True when folder/name is a regular file (by lstat: a link, a folder or a special file is not) whose bytes,
    stripped of surrounding white space, have the sha256 want; False otherwise, and on any error."""
    try:
        if not folder or not isinstance(want, str) or not _HEX64.fullmatch(want):
            return False
        p = os.path.join(folder, name)
        st = os.lstat(p)
        if not statmod.S_ISREG(st.st_mode):
            return False
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            h.update(fh.read().strip())
        return hmac.compare_digest(h.hexdigest().encode("ascii"), want.encode("ascii"))
    except Exception:  # noqa: BLE001 - a token that cannot be read is no token
        return False
