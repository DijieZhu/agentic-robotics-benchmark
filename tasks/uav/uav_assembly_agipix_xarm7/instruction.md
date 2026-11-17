Build the supplied AGiPix-derived quadrotor mechanical kit with the xArm 7 workcell. The lower chassis is clamped to a raised fixture; the other components start loose on the bench. Write `/home/user/submission/assemble.py` with a synchronous `solve(cell)` function. It must locate, grasp, transport, align and fasten the components through the workcell connection.

You receive:

| Path | Contents |
|---|---|
| `given/kit/` | Fabricated and purchased component meshes, collision pieces, quantities, masses and manufacturing notes |
| `given/xarm7/` | xArm 7 and gripper model, licence and upstream description |
| `given/manual/` | Picture manual and mechanical assembly instructions |
| `given/cell.md` | Cell commissioning, controller conventions, sensing and safety limits |
| `cell_client.py` | Connection interface and image decoding |
| `render/` | Workshop scene assets used by the cameras |

Your program runs separately on three fresh workcells. Parts have new bench locations and yaws; fixture position and camera commissioning also vary. Use the top and wrist RGB-D cameras, robot joint encoders and wrist wrench. No object poses, camera calibration, segmentation labels, coupling events or assembly progress are supplied. There is no trial workcell during programming. You may inspect the supplied CAD and write your own offline geometric and kinematic tools.

Before physical settling, each part is staged in the rest orientation published in `given/kit/kit.json` (`rest_rotation`, or the identity matrix when absent), then rotated by an unknown angle in [-0.2, 0.2] radians about world +z. In matrix notation, `R_world = Rz(theta) @ R_rest`; this is a world-axis rotation. This staging prior helps disambiguate initially symmetric parts after you calibrate the world frame. Parts remain free to settle and move under contact, so estimate and track their actual poses from observations.

The connection provides:

```python
cell.describe()
state = cell.get_state()
cell.set_control(q=[...], gripper_width_m=0.06)  # seven radians; metres
state = cell.step(ticks=100)                   # 10 ms per tick, at most 200
frame = cell.get_camera("top")                # also "wrist"
rgb, depth = frame["rgb"], frame["depth_m"]      # numpy arrays; zero depth invalid
cell.finish()
```

Changing a control target does not advance time. You have 1,800 simulated seconds, 7,200 wall seconds and 900 camera frames per workcell. A program silent for 120 wall seconds ends that workcell. Camera calls and computation use wall time. All command values must be finite and within the supplied robot limits. The controller slews joint targets by at most 0.8 rad/s. The clamp is fixed; you cannot command it.

Fastening is an at-rest mechanical coupling abstraction for small screws and retainers: bring the correct mating features within 1.5 mm, orient them within 3 degrees and hold still for 0.5 seconds. Engagement is recognized only within 3 mm and 8 degrees; it does not pull the part into place. A fastened component remains at the actual pose you achieved. A seated component fastened at 3–8 degrees can be mechanically fixed but receive no credit. Motor and landing-strut mounts have four equivalent orientations, 90 degrees apart, matching their four mounting slots or holes. Rotationally symmetric propeller mounts do not constrain yaw. Propellers attach to an unpowered rotor hinge; CW and CCW propellers are different components. Identical parts within a listed component kind are interchangeable.

The objective is the completed fraction of mechanical connections, averaged across the workcells. Each connection has equal weight. Prerequisites in the manual describe physical assembly dependencies. Missing connections lose their own credit. An unsafe episode receives zero: non-pad robot contact exceeding 30 N, a part moving above 3 m/s, or a part contact above 150 N sustained for 0.2 seconds. Do not push through obstructions. A crash or early return preserves only the physical work already completed; it does not remove that episode from the average.

This task covers mechanical assembly. Wiring, soldering, battery energization, firmware and flight are outside its scope. Tiny fasteners, thin damping pads, captive spacer sets and small mounting brackets (the camera unit's bridge, the guard's sensor-cage cradles) are included in the relevant component, as documented. The supplied geometry defines this adapted unit; use its manual rather than assuming every photo of the original AGiPix has the same avionics placement. Leave ordinary Python source and support files in `submission/` (at most 60 MiB): only that folder is copied out of your session and graded, in a fresh sandbox that holds nothing else of yours. The connection is provided to `solve`; do not instantiate your own `CellClient` or read/write its protocol streams.

Your own `.py` files must import no networking module (`socket`, `ssl`, `urllib`, `http`, `requests`, `asyncio`, `ctypes`, `subprocess`, ...), load no module under a computed name, execute no computed code (`exec` / `eval` / `compile` of a non-literal) and start no other program (`os.system`, `os.exec*`, `os.spawn*`, ...), and your program must cause no network activity while it runs: the grader reads the outbound network counters before and after each workcell and samples the program's sockets. A submission that breaks any of these scores zero.
