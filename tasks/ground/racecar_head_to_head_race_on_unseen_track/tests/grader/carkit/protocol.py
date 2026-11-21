"""The wire's vocabulary and its request rules (protocol racecar-wire/1).

Standard library only. The car's server and the protocol stub both answer a request by these rules, so a program
that the stub accepts is accepted by the venue:

    check_envelope(obj)          -> None, or (kind, detail): kind "protocol" (not a request at all) or "bad_params"
    check_params(method, params) -> None, or a detail text (the reply is the error `bad_params`)

Methods outside PUBLIC_METHODS are answered with the error `forbidden` and counted.
"""

import math

PROTOCOL = "racecar-wire/1"
SESSIONS = ("practice", "heat1", "heat2")
PUBLIC_METHODS = ("describe", "open_session", "step", "close_session", "request_pickup")
METHODS = PUBLIC_METHODS
ERROR_CODES = ("forbidden", "bad_params", "no_session", "bad_order", "venue_over")
EVENT_TYPES = ("lap", "stopped", "released", "warning", "phase", "session_end")
SESSION_END_REASONS = ("time", "laps", "warnings", "void")
NO_PARAMS = ("describe", "close_session", "request_pickup")
MAX_LINE_BYTES = 4 * 1024 * 1024
ENVELOPE_KEYS = ("id", "method", "params")


def is_number(v):
    """A finite JSON number (a boolean is not a number)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_envelope(obj):
    """None when `obj` is a well formed request envelope; else ("protocol", detail) when it is not a JSON object with
    a string `method` (the venue ends), or ("bad_params", detail) for a bad `id` or an unknown key."""
    if not isinstance(obj, dict) or not isinstance(obj.get("method"), str):
        return "protocol", "a request is a JSON object with a string `method`"
    if not obj["method"] or len(obj["method"]) > 64:
        return "protocol", "`method` is 1 to 64 characters"
    rid = obj.get("id")
    if not (isinstance(rid, int) and not isinstance(rid, bool) and rid >= 1):
        return "bad_params", "`id` must be an integer counting up from 1"
    extra = sorted(set(obj) - set(ENVELOPE_KEYS))
    if extra:
        return "bad_params", "unknown request key(s): %s" % ", ".join(extra)[:200]
    return None


def check_params(method, params):
    """None when `params` suits `method` (a method of METHODS); else a short detail text for `bad_params`."""
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return "params must be an object"
    if method in NO_PARAMS:
        if params:
            return "%s takes no parameters" % method
        return None
    if method == "open_session":
        if set(params) != {"name"}:
            return "open_session needs exactly the parameter `name`"
        if params["name"] not in SESSIONS:
            return "the session names are practice, heat1, heat2"
        return None
    if method == "step":
        if set(params) != {"steer", "speed"}:
            return "step needs steer and speed, both finite numbers"
        if not (is_number(params["steer"]) and is_number(params["speed"])):
            return "step needs steer and speed, both finite numbers"
        return None
    return "unknown method"


def error_reply(rid, code, detail=None):
    """The error reply object for request id `rid` (None when the request had no usable id)."""
    out = {"id": rid if isinstance(rid, int) and not isinstance(rid, bool) else None, "error": code}
    if detail:
        out["detail"] = str(detail)[:300]
    return out
