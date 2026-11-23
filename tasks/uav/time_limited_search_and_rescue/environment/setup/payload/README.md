# Aircraft service (commissioning)

In this sandbox the service is a commissioning aircraft on sites drawn at random; none of them is graded.
The graded flights happen later, in a separate sandbox, where your program gets the same calls on other sites.

- `client.py`: `Client().observe()`, `Client().act([right, forward, up, yaw_left, grip])`, `Client().new_site()`.
  The socket path comes from the environment variable `ROBOT_SOCKET` (default `/run/visual_robotics/world.sock`).
- The same three calls are MCP tools: `robot_observe`, `robot_act`, `robot_new_site`.
- Run code with `/opt/venv/bin/python3`.
- An answer holds `images` (two base64 PNGs: down, forward), `sensors`, `frame_id`, `steps_remaining`, `done`,
  `requests_remaining`, and here also `commissioning_actions_remaining` and `commissioning_sites_remaining`.
- There is no state inspection, no score and no report of what a flight achieved: you see what the aircraft sees.
- `new_site` does not exist in a graded flight.
