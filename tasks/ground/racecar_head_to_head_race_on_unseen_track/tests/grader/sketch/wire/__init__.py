"""sketch.wire: the JSON-lines server in the grader process, the program runner, the in-process
transport and the protocol stub (`stub.py`, standalone: it imports nothing from sketch).

    from sketch.wire import serve, run_venue
    reason = serve(venue, argv, cwd=..., env=...)             # the contract
    result, venue, stats = run_venue(world, program_dir)      # a fresh venue and process, the wire, the result
"""

from .inprocess import InProcess, connect, run
from .runner import child_env, prepare_run_dir, program_argv, run_venue
from .server import ProgramProcess, encode_reply, serve

__all__ = ["serve", "run_venue", "ProgramProcess", "encode_reply", "prepare_run_dir", "child_env", "program_argv",
           "InProcess", "connect", "run"]
