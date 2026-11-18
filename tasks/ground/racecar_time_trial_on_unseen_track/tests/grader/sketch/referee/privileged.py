"""The cell's privileged calls, server side only: their names and their parameter rule.

The agent-facing client library `carkit` names none of them (its `protocol.py` knows the
five public methods only); the cell answers a privileged call with `forbidden` unless the process unlocked it with the
token whose sha256 the cell holds. The reference program is the only user and sends them with the generic
`car.call(method, **params)`.
"""

from carkit import protocol

PRIVILEGED_METHODS = ("priv_unlock", "priv_world", "priv_state")
METHODS = protocol.PUBLIC_METHODS + PRIVILEGED_METHODS
NO_PARAMS = ("priv_world", "priv_state")
_HEX = frozenset("0123456789abcdef")


def check_params(method, params):
    """None when `params` suits `method` (public or privileged); else a short detail text for `bad_params`."""
    if method not in PRIVILEGED_METHODS:
        return protocol.check_params(method, params)
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return "params must be an object"
    if method in NO_PARAMS:
        return "%s takes no parameters" % method if params else None
    tok = params.get("token")
    if set(params) != {"token"} or not isinstance(tok, str) or len(tok) != 64 or not set(tok) <= _HEX:
        return "priv_unlock needs `token`: 64 lower-case hex digits"
    return None
