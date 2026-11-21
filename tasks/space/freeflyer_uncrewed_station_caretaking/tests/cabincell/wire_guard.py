"""The grader's guard on the program's request lines (the caretaking cell; shape of the quadruped task's
tests/plantcell/wire_guard.py, which the published arm task's shared kit does not have).

``tests/cellkit/`` is a byte copy of the shared kit, so the guard lives in the cell's package; the
verifier and the cell's wire driver (server.run_wire) use ``GuardedProcess`` in place of ``cellkit.wire.AgentProcess``,
with the shared ``wire.serve_episode`` unchanged:
- a request line the JSON parser cannot read, for any reason, is a protocol end (the program's doing): the shared kit
  catches only ValueError, and a line nested deeper than the parser's limit raises RecursionError out of the wire loop;
- the request's ``method`` reaches the wire loop as a string of at most 64 characters (anything else becomes
  "(not a name)", which the server answers 'forbidden'), and its ``id`` as one the reply can always echo (null,
  true / false, a 64-bit integer, a string of at most 256 characters; anything else is echoed as null).
"""

import queue

from cellkit import wire

METHOD_MAX = 64
ID_STR_MAX = 256


def plain_id(rid):
    """A request id as the reply echoes it: null, true / false, a 64-bit integer or a short string; else null."""
    if rid is None or isinstance(rid, bool):
        return rid
    if isinstance(rid, int) and -(1 << 63) <= rid < (1 << 63):
        return rid
    if isinstance(rid, str) and len(rid) <= ID_STR_MAX:
        return rid
    return None


def plain_request(obj):
    """The request object with a plain ``method`` and ``id`` (in place; returned)."""
    m = obj.get("method")
    obj["method"] = m[:METHOD_MAX] if isinstance(m, str) else "(not a name)"
    obj["id"] = plain_id(obj.get("id"))
    return obj


class GuardedProcess(wire.AgentProcess):
    """``cellkit.wire.AgentProcess`` whose ``next_request`` turns every failure of the JSON parser into a protocol end
    and hands the wire loop plain method names and ids."""

    def next_request(self, timeout):
        try:
            kind, obj = super(GuardedProcess, self).next_request(timeout)
        except queue.Empty:
            raise                                    # silence: the wire loop's own rule
        except Exception:  # noqa: BLE001 - RecursionError (or any other failure) of the parser on the program's line
            return "bad", "(a request line the JSON parser could not read)"
        if kind == "req":
            obj = plain_request(obj)
        return kind, obj
