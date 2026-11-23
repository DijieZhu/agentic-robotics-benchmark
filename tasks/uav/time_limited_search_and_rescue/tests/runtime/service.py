"""Single-threaded, root-owned aircraft service with a private ledger.

Two modes behind the same Unix socket and the same calls:

* ``commissioning`` (the agent's sandbox): sites drawn at random on request (``new_site``), a total
  budget of actions and of sites, no truth, no score and no debrief: an answer is what the aircraft
  itself returns (two images, the sensors, the step counters, ``done``).
* ``graded`` (the verifier): one site, one flight, no reset. ``--truth`` (set by the grader only
  for the token-gated reference) adds the privileged ``_gt`` block to every answer.

The ledger (``state.json`` in the 0700 directory) is rewritten atomically after every request. It
carries the flight's metrics and the seconds the service spent answering, which the grader subtracts
from wall time to measure the program's own time. It never holds the site's seed.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import socketserver
import sys
import time
import traceback

from rescue import World

REQUESTS = 9000                 # per flight, observations included
COMMISSIONING_ACTIONS = 16000   # over all commissioning sites together
COMMISSIONING_SITES = 40


def commissioning_seed():
    # Bit 63 set: disjoint from the graded sites, which are all below 2**32.
    return secrets.randbits(63) | (1 << 63)


class Runtime:
    def __init__(self, directory, mode, seed=None, truth=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.directory, 0o700)
        self.mode = mode
        self.truth = bool(truth) and mode == "graded"
        self.busy = 0.
        self.sites = 0
        self.actions_spent = 0
        self.world = None
        self.internal_errors = 0
        self.request_started = None
        self._start(seed)

    def _start(self, seed=None):
        if self.world is not None:
            self.actions_spent += self.world.steps
            self.world.close()
            self.world = None
        self.world = World("rescue", commissioning_seed() if seed is None else int(seed), truth=self.truth)
        self.requests = 0
        self.exhausted = False
        self.sites += 1
        self.flush()

    @property
    def actions_left(self):
        return COMMISSIONING_ACTIONS-self.actions_spent-(self.world.steps if self.world else 0)

    def flush(self):
        world = self.world
        state = {"mode": self.mode, "sites": self.sites, "requests": self.requests,
                 "requests_exhausted": self.exhausted, "busy_seconds": round(self.busy, 4),
                 "done": bool(world and (world.done or self.exhausted)),
                 "metrics": world.metrics() if world else {},
                 "internal_errors": self.internal_errors, "request_started": self.request_started}
        temp = self.directory/"state.tmp"
        temp.write_text(json.dumps(state))
        temp.replace(self.directory/"state.json")

    def call(self, req):
        started = time.monotonic()
        # Wall-clock start of the request in flight, for the grader's program-time measure.
        self.request_started = time.time()
        self.flush()
        try:
            if not isinstance(req, dict) or set(req) != {"method", "params"}:
                raise ValueError("a request is {\"method\": ..., \"params\": {...}}")
            method, params = req["method"], req["params"]
            if not isinstance(method, str) or not isinstance(params, dict):
                raise ValueError("a request is {\"method\": ..., \"params\": {...}}")
            if method == "new_site":
                if self.mode != "commissioning":
                    raise ValueError("new_site is a commissioning call; a graded flight is one site")
                if params:
                    raise ValueError("new_site takes no parameters")
                if self.sites >= COMMISSIONING_SITES:
                    raise ValueError(f"all {COMMISSIONING_SITES} commissioning sites are used")
                if self.actions_left <= 0:
                    raise ValueError("the commissioning action budget is spent")
                self._start()
                self.requests = 1
                return self._answer(self.world.observe())
            self.requests += 1
            if self.requests > REQUESTS:
                self.exhausted = True
                raise ValueError(f"this flight's {REQUESTS} requests are spent; it has ended")
            return self._answer(self.dispatch(method, params))
        except (ValueError, TypeError, KeyError, OverflowError):
            raise
        except Exception:
            # A service failure voids verification; it never starts a second flight.
            self.internal_errors += 1
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            raise RuntimeError("internal service error")
        finally:
            self.busy += time.monotonic()-started
            self.request_started = None
            self.flush()

    def _answer(self, observation):
        answer = dict(observation)
        answer["requests_remaining"] = max(0, REQUESTS-self.requests)
        if self.mode == "commissioning":
            answer["commissioning_actions_remaining"] = max(0, self.actions_left)
            answer["commissioning_sites_remaining"] = COMMISSIONING_SITES-self.sites
        return answer

    def dispatch(self, method, params):
        allowed = {"observe": set(), "act": {"action"}}
        if method not in allowed or set(params) != allowed[method]:
            raise ValueError("calls are observe() and act(action)"
                             + (" and new_site()" if self.mode == "commissioning" else ""))
        if method == "act" and self.mode == "commissioning" and self.actions_left <= 0 and not self.world.done:
            raise ValueError("the commissioning action budget is spent")
        return getattr(self.world, method)(**params)


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(3)
        try:
            line = self.rfile.readline(8193)
            if len(line) > 8192:
                raise ValueError("request too large")
            result = self.server.runtime.call(json.loads(line))
        except RecursionError:
            result = {"error": "request too deeply nested"}
        except (ValueError, TypeError, KeyError, OverflowError, RuntimeError) as exc:
            result = {"error": str(exc)}
        except OSError:
            return
        try:
            self.wfile.write((json.dumps(result)+"\n").encode())
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("commissioning", "graded"), required=True)
    parser.add_argument("--directory", default="/var/lib/visual_robotics")
    parser.add_argument("--socket", default="/run/visual_robotics/world.sock")
    parser.add_argument("--seed-file", help="graded: a root-only file holding the site's seed")
    parser.add_argument("--truth", action="store_true", help="graded, token-gated reference only")
    args = parser.parse_args()
    seed = None
    if args.mode == "graded":
        if not args.seed_file:
            parser.error("--seed-file is required in graded mode")
        seed = int(Path(args.seed_file).read_text().strip())
    elif args.truth:
        parser.error("--truth exists only in graded mode")
    runtime = Runtime(args.directory, args.mode, seed, args.truth)
    path = Path(args.socket)
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError("refuse to overwrite a live or pre-existing socket")
    (Path(args.directory)/"pid").write_text(str(os.getpid()))
    with socketserver.UnixStreamServer(str(path), Handler) as server:
        os.chmod(path, 0o666)
        server.runtime = runtime
        (Path(args.directory)/"ready").write_text("1")
        server.serve_forever()


if __name__ == "__main__":
    main()
