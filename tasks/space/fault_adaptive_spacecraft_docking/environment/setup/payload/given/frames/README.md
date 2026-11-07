# Sample frames of the docking camera

Forty frames of four example episodes (A to D, none of them graded), as the docking camera delivers them
(`../camera.md`): 640 x 480 lossless 8-bit monochrome PNGs holding exactly the bytes a frame's `data` decodes to,
noise included. Each episode was flown by a controller that docks; per episode there are ten frames:

| kind | how many | what |
|---|---|---|
| `start` | 1 | the first frame (stamp 0), as flown |
| `far` | 3 | the far field (1.6 m or more from the docking target), with the craft turned to look at the target's centre: not as flown, the same positions |
| `mid` | 2 | 0.9 to 1.4 m from the docking target, as flown |
| `after_restart` | 1 | the first frame after the re-key's avionics restart, as flown |
| `final` | 2 | the final approach, 0.4 to 0.9 m from the docking target, as flown |
| `hold` | 1 | during the hold at the port, as flown |

A file is named `<episode>_<stamp>_<kind>.png`. `truth.json` holds, for every frame:

- `stamp`: the step count at capture;
- `craft_x_m`, `craft_y_m`, `craft_yaw_rad`: the craft's position and yaw in the inertial frame at capture (for the
  `far` frames the turned yaw), from which the camera's pose follows (`../camera.md`);
- `target_phase_rad`, `port_radius_m`: the target's phase and the port's radius at capture (`../model.md`);
- `camera_in_port_frame`: the lens centre in the port frame (`../target.md`) and the boresight's yaw in that frame;
- `camera_to_docking_target_m`: the distance from the lens to the docking target's flush LED faces;
- for the frames as flown: `delivered_at_t`, the request that carried the frame, with that request's
  `star_tracker_yaw_at_t` and the craft's true yaw then, `true_yaw_at_t_rad`.
