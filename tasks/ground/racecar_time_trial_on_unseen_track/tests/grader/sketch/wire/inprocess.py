"""The same messages in-process (the protocol may be carried in-process during development; the content is identical).

    car = connect(venue)                  # a carkit.Car whose requests go through the JSON encoding to venue.handle
    reason = run(venue, solve)            # call solve(car) once; the venue's end reason

Program time is the wall clock between the return of one call and the next call, charged by `venue.account` as on
the process wire; the cell's own work is excluded. Used by tests, ladders and the oracle when a separate process is
not needed; the graded run is `sketch.wire.serve`.
"""

import json
import time

from carkit import CarError, connect_handler, protocol

from .server import encode_reply


class InProcess(object):
    def __init__(self, venue):
        self.venue = venue
        self.t_last = None
        self.requests = 0
        self.program_times = []

    def __call__(self, line):
        now = time.perf_counter()
        v = self.venue
        obj = json.loads(line)
        env_err = protocol.check_envelope(obj)
        if env_err is not None and env_err[0] == "protocol":
            v.end("protocol")
            return b""
        if self.t_last is not None:
            dt = now - self.t_last
            self.program_times.append(dt)
            if v.account(dt) is not None:
                return b""                    # a cap ended the venue: the wire closes
        self.requests += 1
        if env_err is not None:
            v.counts["bad_calls"] += 1
            data = (json.dumps(protocol.error_reply(obj.get("id"), "bad_params", env_err[1])) + "\n").encode("utf-8")
        else:
            data = encode_reply(obj["id"], v.handle(obj["method"], obj.get("params")))
        self.t_last = time.perf_counter()
        return data


def connect(venue, numpy=None):
    return connect_handler(InProcess(venue), numpy=numpy)


def run(venue, solve, numpy=None):
    """Call `solve(car)` in this process; returns the venue's end reason ("eof" when solve returns early)."""
    car = connect(venue, numpy=numpy)
    try:
        solve(car)
    except CarError as e:
        if e.code != "closed":
            raise
    venue.end("eof")
    return venue.end_reason
