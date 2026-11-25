"""The privileged reference program of the humanoid assembly cell: a framework that drives the cell
through the agent's own API (the client proxy over the wire: describe, get_state, set_control, command, step,
get_camera, finish) and reads the TRUE poses through the privileged calls of the oracle token (describe_privileged,
get_state_privileged). One module = one file (module_<mate>.py); runner.solve(cell) runs them in the reference order."""
VERSION = "humanoid-reference/0.1.0"
