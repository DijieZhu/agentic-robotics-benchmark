"""The messages of the head-to-head day against `wire.schema.json` (this folder): the protocol's envelope, methods
and observations as for every venue of the league, with this day's session names, describe table, phase names and
end reasons, and the field of the open heat in the privileged state.

    check_request(obj) -> [error, ...]
    check_reply(method, obj) -> [error, ...]
    check_transcript(lines) -> [error, ...]       # an alternation of request and reply objects, ids counting up
"""

import json
import os

from sketch.examples import check

HERE = os.path.dirname(os.path.abspath(__file__))
WIRE_SCHEMA_PATH = os.path.join(HERE, "wire.schema.json")
_CACHE = {}


def _defs():
    if "s" not in _CACHE:
        with open(WIRE_SCHEMA_PATH, "r", encoding="utf-8") as f:
            _CACHE["s"] = json.load(f)
    return _CACHE["s"]


def _sub(name):
    s = _defs()
    return {"$schema": s["$schema"], "$ref": "#/$defs/" + name, "$defs": s["$defs"]}


def check_request(obj):
    return check.schema_errors(obj, _sub("request"))


def check_reply(method, obj):
    if isinstance(obj, dict) and "error" in obj:
        return check.schema_errors(obj, _sub("error_reply"))
    name = "reply_" + method
    if name not in _defs()["$defs"]:
        return ["method %r is not in the table: the only valid reply is the error `forbidden`" % method]
    return check.schema_errors(obj, _sub(name))


def check_transcript(lines):
    err, pending, last_id = [], None, 0
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError as e:
            err.append("line %d: not JSON (%s)" % (n, e))
            continue
        if pending is None:
            e = check.schema_errors(obj, _sub("envelope"))
            err += ["line %d: %s" % (n, x) for x in e]
            if not e:
                if obj["id"] != last_id + 1:
                    err.append("line %d: request id %r, expected %d" % (n, obj["id"], last_id + 1))
                last_id = obj["id"]
                pending = obj
        else:
            if check_request(pending):
                if not (isinstance(obj, dict) and obj.get("error") == "bad_params"):
                    err.append("line %d: the request before is ill-formed, the reply must be the error `bad_params`" % n)
                err += ["line %d: %s" % (n, x) for x in check.schema_errors(obj, _sub("error_reply"))]
            elif pending["method"] not in check.METHODS:
                if not (isinstance(obj, dict) and obj.get("error") == "forbidden"):
                    err.append("line %d: method %r is not in the table, the reply must be the error `forbidden`" % (n, pending["method"]))
            else:
                err += ["line %d: %s" % (n, x) for x in check_reply(pending["method"], obj)]
            if isinstance(obj, dict) and obj.get("id") != pending["id"]:
                err.append("line %d: reply id %r does not answer request %r" % (n, obj.get("id"), pending["id"]))
            pending = None
    if pending is not None:
        err.append("the last request has no reply")
    return err


__all__ = ["check_request", "check_reply", "check_transcript", "WIRE_SCHEMA_PATH"]
