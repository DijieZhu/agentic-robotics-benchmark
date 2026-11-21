# spacekit API (v0.4.0)

Pure-Python physics and firmware layer for MuJoCo 3.14 scenes of the cabin free-flyer.
Runtime dependencies: `mujoco==3.14.0`, `numpy>=2`. No scipy, no OpenCV, no rendering in the core
(the optional `spacekit.cameras` uses OpenCV's remap when it is installed and falls back to numpy).
v0.2.0 adds the cabin mechanics (`aero`, `suction`, `gripper`, `grasp_rule`, `fixtures`), the station model
(`station`), the zone validator (`zones`), firmware-level primitives (`primitives`) and the optional camera layer
(`cameras`); v0.3.0 adopts the part-level reference build v1 (reference configuration, finger recipe, the dock v1
berth `dock`, the propulsion power cut, the fan-stop flight rules, the standard stowage bag, fiducial positioning,
the render-side `visual_twin`); v0.4.0 fixes the hover wander and the moves that never stop, adds the firmware
behaviours of freeflyer_uncrewed_station_caretaking as options, speeds up the suction and dock sources and adds
interfaces; see the sections at the end.  Names in these modules are mechanics-level, not an agent-facing API.

Conventions used everywhere:

- SI units (m, s, kg, N, rad). Quaternions are `(w, x, y, z)`, Hamilton product; a pose quaternion rotates
  body vectors into the world frame (as MuJoCo `xquat`).
- World frame = the station module (frame W). Body frame of the robot = the firmware body frame B
  (x forward, y starboard side of propulsion module 0, z down). Nozzles 0-5 belong to propulsion module 0
  (+y side), 6-11 to module 1 (-y side), in the firmware's order.
- A **wrench source** is any object with `accumulate(model, data, W)` that adds a `(model.nbody, 6)` array
  of world-axes `[force, torque]` acting at the body centres of mass. Most sources also have
  `wrench(model, data)` returning a new array, and an `enabled` attribute.
- A **velocity field** is any callable `field(x) -> (N, 3)` with `x` of shape `(N, 3)` (world, m/s).
- Determinism: no wall clock is read; every random draw comes from an explicit `numpy.random.Generator`.
  The only global state ever touched is MuJoCo's `mjcb_passive`, and only inside `stepping.PassiveCallback`.
- Firmware rate: 62.5 Hz (period 0.016 s). Physics step: 2 ms in contact phases (`stepping.DT_CONTACT_MAX`).

Typical use:

```python
import numpy as np, mujoco, spacekit as sk
m = mujoco.MjModel.from_xml_path("scene.xml"); d = mujoco.MjData(m)
rig = sk.robot.FreeFlyerRig(m, d, rng=np.random.default_rng(7),
                            station_frame=sk.frames.RotatingStationFrame(m),
                            flow=field, jet_targets=item_ids, drag_bodies=item_ids)
rig.move_to([1.0, 0.0, 0.0], [1, 0, 0, 0])      # face-forward trapezoid, standard mode
rig.run_until_done(timeout=60.0)
print(rig.firmware.result, rig.battery.soc)
```

---

## spacekit.rotations

| Function | Description |
|---|---|
| `quat_mul(a, b)` | Hamilton product `a * b`. |
| `quat_conj(q)` | Conjugate (inverse of a unit quaternion). |
| `quat_normalize(q, positive_w=False)` | Unit quaternion; optional sign choice `w >= 0`. |
| `quat_to_mat(q)` | 3x3 rotation matrix. |
| `mat_to_quat(R)` | Unit quaternion (`w >= 0`) of a rotation matrix. |
| `quat_from_rotvec(v)` | Quaternion of rotation vector `v` (axis * angle). |
| `rotvec_from_quat(q)` | Rotation vector with angle in [0, pi]. |
| `quat_from_axis_angle(axis, angle)` | Rotation about `axis` by `angle`. |
| `quat_angle(qa, qb)` | Angle [rad] between two orientations. |
| `quat_rotate(q, v)` | Rotate vector(s) `v` (..., 3). |
| `quat_integrate_body(q, omega_body, dt)` | `q * exp(omega_body dt / 2)` (constant body rate). |
| `skew(v)` | Cross-product matrix. |

## spacekit.frames — A. frames and fictitious forces

Constants: `MU_EARTH`, `R_EARTH`, `J2_EARTH`, `DEFAULT_ORBIT_RATE` (1.1314e-3 rad/s, 400 km circular orbit).

| Function / class | Description |
|---|---|
| `mean_motion(a, mu=MU_EARTH)` | Orbit rate of a circular orbit of radius `a`. |
| `radius_for_rate(n, mu=MU_EARTH)` | Radius of a circular orbit with rate `n`. |
| `encke_rel_accel(rho, R, mu=MU_EARTH)` | Exact `g(R + rho) - g(R)` without cancellation (Battin), `rho` (..., 3). |
| `j2_accel(r, ...)` | J2 acceleration (Earth-centred, z = spin axis). |
| `cw_propagate(x0, v0, n, t)` | Clohessy-Wiltshire closed form in LVLH; returns `(pos, vel)`. |
| `CircularChief(orbit_rate=DEFAULT_ORBIT_RATE)` | Analytic circular reference orbit; `.state(t) -> (R, V)`, `.lvlh_basis(t)` (rows = LVLH axes in world), attributes `n`, `a`, `mu`. |
| `body_com_velocity(model, data)` | (nbody, 3) CoM linear velocity, world axes (exact for articulated bodies). |
| `world_inertia(model, data, bodies=None)` | (n, 3, 3) inertia about each CoM in world axes. |
| `free_bodies(model)` | Ids of bodies with mass that move (not welded to world, not mocap). |
| `RotatingStationFrame(model, orbit_rate=..., C_lvlh_from_world=I, origin=0, bodies=None, enabled=True, tidal=True, coriolis=True, centrifugal=True, rotational=True, gravity_gradient=True, linear_tidal=False)` | Wrench source for a world fixed to a station module holding attitude `C_lvlh_from_world` in LVLH (rows = LVLH axes in world axes). Per body: `F = m[tidal - 2 Om x v - Om x (Om x rho)]`, `tau = tau_gg - Om x J Om - Om x J w - w x J Om + J (w x Om)`. `enabled=False` switches everything off; per-term switches for ablation; `linear_tidal` uses `n^2 (2x, -y, -z)` (LVLH). Attributes `n`, `Om` (world axes), `R_world`. |
| `InertialChiefFrame(model, chief=None, bodies=None, gravity_gradient=True, extra_accel=None, enabled=True)` | Wrench source for the chief-centred inertial variant (cross-checks, extra-vehicular work): Encke differential gravity + gravity-gradient torque at `data.time` (stage time inside RK4). `extra_accel(t, rho) -> (N, 3)` adds accelerations. |
| `rotating_to_inertial(p, v, w, orbit_rate=..., C_lvlh_from_world=I)` | Initial-state conversion from the rotating frame to the inertial variant (axes coincide at t = 0). |

## spacekit.stepping — B. stepping

Constants: `DT_CONTACT_MAX = 0.002` s, `DT_FREE_MAX = 0.010` s.

| Function / class | Description |
|---|---|
| `step(model, data, sources=(), nstep=1, W=None)` | `mj_step1` -> sum sources into `xfrc_applied` -> `mj_step2`, `nstep` times. Raises `ValueError` for RK4. `W` optional work buffer. |
| `leapfrog_start(model, data, sources)` | Runs `mj_forward`, then shifts free-joint linear velocities by `-dt/2 * F/m` (half-step start); returns the shifts. |
| `sum_sources(model, data, sources, W=None)` | Zero `W` and accumulate the sources in order. |
| `SlowForceCache(sources, every=10)` | Source wrapper: evaluates the wrapped sources every `every`-th call, replays the cached wrench otherwise; `.invalidate()` forces a refresh. |
| `PassiveCallback(sources, replace=False)` | Context manager: evaluates sources inside `mj_passive` (every RK4 stage). Installs/restores MuJoCo's global `mjcb_passive`; raises `RuntimeError` if a foreign callback is installed. Attribute `calls`. |
| `check_timestep(model, phase="contact")` | `ValueError` if `timestep` exceeds the limit of `'contact'` or `'free'`. |
| `check_finite(data)` | `StateError` on NaN / Inf in `qpos` / `qvel`. |
| `Simulator(model, data, sources=(), slow_sources=(), slow_every=1, phase=None, check_every=1000)` | Deterministic scheduler. `.step(n)`, `.run_until(t)`, `.add_source(src, slow=False)`, `.add_periodic(fn, period, name=None)` (fn(sim) before the physics step whose start time is k * period; period must be an integer multiple of dt), `.steps`. RK4 models: fast sources via `PassiveCallback`, slow ones via `xfrc_applied`. |
| `StateError` | Exception type. |

## spacekit.propulsion — C. fan propulsion and allocation

| Function / class | Description |
|---|---|
| `load_firmware_reference(path=None)` | Deep copy of `data/firmware_reference.json` (propulsion, configured mass properties, controller constants, flight modes `off / standard / agile / hush / fine`, planner and executive constants). |
| `PropulsionParams` | Dataclass of all propulsion constants (`from_dict(block)`, `reference()`, `copy()`, `max_flap_travel`, `width_of(k)`, `max_area()`). Fields: nozzle geometry (`nozzle_position` (12,3), `nozzle_exhaust_dir` (12,3), `nozzle_width` (6,), `discharge_coeff` (2,6), `leak_area` (2,)), flap geometry, servo and fan-motor constants, PID gains, fan speed levels `imp_levels` (3,), ramp, `air_density`, `thrust_error_scale` (1.25), both lookup tables, `null_groups`. |
| `table_lookup(breakpoints, values, x)` | Firmware lookup (clamp outside, binary search + linear interpolation), vectorised. |
| `nozzle_area(theta, width, p)` | Open area of nozzles with flap angle `theta` beyond closed. |
| `flap_angle_for_area(area, width, p)` | Inverse with the firmware's clamping (NaN -> fully open). |
| `plenum_pressure(omega, total_area, p)` | `omega^2 D^2 rho c(A_tot)` [Pa]. |
| `intake_flow(omega, total_area, p)` | Module volume flow `A_tot sqrt(2 dp / rho)` [m^3/s]. |
| `FanModule(p, index, com=0, gyro="physical", thrust_scale=None, aero_load=False, fan_efficiency=0.52)` | One propulsion module, step-for-step port of the firmware plant: `.step(impeller_cmd, servo_cmd(6), omega_body, voltage=None)` at the control period; state/outputs `thrust` (6,), `area`, `nozzle_theta`, `dp`, `impeller_speed`, `motor_torque`, `force`, `torque` (about `com`), `momentum`, `electrical_power`, `impeller_current`, `servo_current`; `.set_com(com)`, `.total_area()`, `.aero_torque(omega)`. `gyro='firmware'` reproduces the firmware plant's `+w x h` sign. |
| `FanPropulsion(p=None, com=0, **module_kw)` | Both modules: `.step(impeller_cmds(2), servo_cmds(12), omega_body, voltage=None)`; properties `thrust` (12,), `force`, `torque`, `momentum`, `impeller_speed` (2,), `electrical_power`, `area` (12,); `.nozzle_forces()` (12,3), `.is_ready(cmds, tol=0.1)`, `.set_com(com)`. |
| `FlightAllocator(p=None, com=0)` | Port of the firmware force allocation: `.allocate(force_body, torque_body, speed_level) -> dict(impeller_cmd(2), servo_pwm(12) float, servo_cmd(12) int, thrust(12), theta(12), area(12), dp(2))`; `.thrusts(F, tau)` (pseudo-inverse + null-group shift), `.plenum_dp(omega, cd, thrusts)`, `.module_command(i, level, thrusts)`, `.update_com(com)`; attributes `B` (6,12), `Binv`. |
| `NNLSAllocator(p=None, com=0, reg=1e-6, margin=0.98)` | Alternative allocator: Lawson-Hanson NNLS + Tikhonov, direction-preserving down-scaling to the module / nozzle capacity; same output dict plus `scale`. |
| `nozzle_matrix(p, com=0)` | (6, 12) thrust-magnitude -> body wrench map. |
| `static_thrust(p, servo_pwm, impeller_speed, thrust_scale=None)` | Steady-state plant thrusts for held commands: `(thrust(12), dp(2), area(12))`. |
| `PropulsionWrench(body_id, propulsion, nozzle_pos_body=None, intake_coeff=0.0, intake_area=None, extra_force_body=0)` | Wrench source applying the plant's nozzle forces at the nozzle points (torque about the body's actual CoM) plus fan reaction and gyroscopic torques; zero-order hold of the latest plant outputs (body loads cached per plant step); optional intake reaction `-k rho Q^2 / A_in` along each fan axis (default k = 0, as the firmware plant) and a constant body-frame force; `.body_wrench(model, data)`. |
| `impeller_power(p, omega, total_area, fan_efficiency=0.52)` | Steady-state electrical power of one fan motor: `dict(power_w, current_a, voltage_v, air_power_w, shaft_power_w, copper_loss_w, flow_m3s, dp_pa)`. |
| `max_along(direction, allocator, speed_level, plant=None, kind="force", rel_tol=0.01, lever=0.1, ...)` | Largest pure force (N) / torque (N m) along a direction that allocator + static plant can hold with closed-loop-like command correction: `(magnitude, delivered wrench, command)`. |
| `wrench_envelope(allocator, speed_level, mass, n_dirs=64, plant=None, rel_tol=0.02, thrust_scale=None)` | Per-signed-axis max force and torque and the pure-force acceleration envelope over directions: `dict(axis_force(6), axis_torque(6), accel_min, accel_max, accel_mean, dirs, accel)`. |

## spacekit.allocation

| Function | Description |
|---|---|
| `thruster_matrix(positions, directions, com=0)` | `B[:, i] = [d_i; (p_i - com) x d_i]`. |
| `nnls(A, b, max_iter=None, tol=None)` | Lawson-Hanson NNLS; returns `(x, residual)`. |
| `allocate(B, wrench, umax=None, reg=1e-4, weights=None)` | NNLS + Tikhonov + uniform saturation scaling: `(u, achieved, scale)`. |
| `pinv_clip(B, wrench, umax=None)` | Naive pseudo-inverse + clip (counter-example). |

## spacekit.cabin — D. cabin air

Constants: `AIR_DENSITY = 1.225`, `DRAG_COEFF` (sphere 0.47, box 1.05, cylinder cross 1.0 / axial 0.9, plate 1.17, capsule cross 0.9).

| Function / class | Description |
|---|---|
| `DragShape(kind, size, cd=None, center=None, rot_coeff=0.0)` | Drag description in the body frame: `'box'` (half extents), `'sphere'` (r), `'cylinder'` / `'capsule'` (r, half length; axis z), `'fixed'` (area). `center` = centre of pressure (body frame, relative to the body origin). `.cd_area(u_body)`. |
| `projected_area(kind, size, u, cd=None)` | Projected area along a unit direction (times Cd if given). |
| `shape_from_body(model, body, cd=None, rot_coeff=0.0)` | Box spanning the body's collision geoms (single sphere -> sphere). |
| `UniformFlow(u)` / `DuctJet(origin, axis, u0, r0, spread, background)` / `ScaledFlow(field, scale)` / `SumFlow(*fields)` | Simple velocity fields (`ScaledFlow.scale` may be changed at run time). |
| `GridFlowField(origin, spacing, values(nx,ny,nz,3), outside="clamp", background=0)` | Regular-grid field with trilinear interpolation; `outside` = `'clamp'`, `'zero'`, `'background'`. |
| `FlowFieldBuilder(lo, hi, h=0.05)` | Divergence-free ventilation field inside a box: `.add_inlet(wall, center, half_size, flow, jet=None)`, `.add_outlet(wall, center, half_size, flow)`, `.add_source(pos, flow)`, `.build(rtol=1e-3, project_jets=True) -> (GridFlowField, report)`. `wall` in `'-x' '+x' '-y' '+y' '-z' '+z'`; `jet=dict(speed=..., spread=0.094)`. Raises `ValueError` if inflow != outflow (rtol). Report keys: `inflow_m3s, outflow_m3s, net_m3s, wall_net_outflow_m3s, max_cell_divergence_m3s, shape, max_speed_ms`; staggered face velocities in `.faces`. |
| `CabinAir(model, bodies, shapes=None, flow=None, rho=1.225)` | Wrench source: quadratic drag relative to the local air velocity at each centre of pressure, torque from its offset, optional rotational drag. |
| `hold_force(u, cd_area=1.05*0.092903, rho=1.225)` | `1/2 rho Cd A u^2`. |
| `RoundJet(origin, direction, momentum_flux, exit_area, spread=0.094, rho=1.225)` | Momentum-conserving Gaussian round jet: `.width(x)`, `.centre_speed(x)`, `.velocity(P)`. |
| `NozzleJets(model, robot_body, nozzle_pos, nozzle_dir, thrust_fn, targets, shapes=None, exit_area_fn=None, n_samples=5, occlusion=True, max_range=3.0, spread=0.094, ambient=None)` | Wrench source: the robot's nozzle exhaust on target bodies (drag of the jet velocity over the target silhouette, increment over the ambient-air drag); one occlusion ray per nozzle-target pair; `last_forces` (n_targets, 3). |
| `duct_operating_point(open_area, p_shutoff, q_free, cd=0.6, rho=1.225)` | `(Q, p_behind)` of a return duct (quadratic fan curve + grille orifice), closed form. |
| `GrilleSuction(model, center, normal, u_axis, half_size, targets, porosity=0.7, p_shutoff=150, q_free=0.12, cd=0.6, samples=16, gap_max=0.01, friction=None, v_slip=0.005, geomgroup=None, exclude_body=0, back=0.005)` | Wrench source: objects within `gap_max` of the grille plane are pulled with `p_behind * w(gap) * dA` at every covered sample (w = `(1 - gap/gap_max)^2`); optional regularised friction `mu N tanh(v/v_slip)`. `.coverage(model, data)`, `.flow_scale()` (duct flow relative to the open grille), `.state` (covered_area, open_area, flow, p_behind), `last_forces`, `.area`. |

## spacekit.control — E. controller, flight modes, planner, executive

Constants: `MODE_IDLE = 0`, `MODE_STOP = 1`, `MODE_NOMINAL = 2`, `MODE_STOPPED = 3`.

| Function / class | Description |
|---|---|
| `FlightMode` | Dataclass: gains (`att_kp`, `att_ki`, `omega_kd`, `pos_kp`, `pos_ki`, `vel_kd`), `speed` (fan level 0-3), tolerances (`tolerance_pos_endpoint`, `tolerance_pos`, `tolerance_vel`, `tolerance_att`, `tolerance_omega`, `tolerance_time`), planner limits (`hard_limit_vel`, `hard_limit_accel`, `hard_limit_omega`, `hard_limit_alpha`), `control_enabled`; `.from_dict(name, d)`. |
| `load_flight_modes(ref=None)` | `{name: FlightMode}` for `off, standard, agile, hush, fine`. |
| `ControlState(pos, quat, vel, omega, confidence=0)` | Estimate given to the controller (world position / velocity, body rate, 0 good / 1 poor / 2 lost). |
| `Setpoint(t, pos, vel, acc, quat, omega, alpha)` | Trajectory sample (angular terms in body axes). |
| `FlightController(ctl=None, mass=None, inertia=None)` | Port of the firmware controller: `.step(time_delta, est, cmd, mode, gains) -> dict(force, torque (body), accel_cmd, alpha_cmd, pos_err, pos_err_int, att_err, att_err_int, att_err_mag, traj_pos, traj_vel, traj_accel, traj_quat, traj_omega, traj_alpha, traj_error_pos, traj_error_vel, traj_error_omega, traj_error_att, ctl_status)`; `.reset()`. |
| `plan_trapezoid(poses, mode, faceforward=True, t0=0.0, times=None, planner=None, desired=None)` | Port of the trapezoidal planner: list of `Setpoint`; limits = mode hard limits (accelerations / 2 if holonomic), optionally lowered by `desired=dict(vel, accel, omega, alpha)` (`ValueError` above the limits). |
| `segment_setpoint(seg, t)` | `(setpoint, index)` active at `t` (latest sample with `t_i <= t`). |
| `Firmware(propulsion=None, allocator="flight", mode="standard", ref=None, overspeed_stop=0.5, overspeed_cutoff=0.75, ready_tol=0.1)` | Executive: `.set_mode(name)`, `.move(poses, faceforward=True, t_now=0, est=None, desired=None)` (plans from the estimate), `.stop()`, `.hold()`, `.idle()`, `.speed_level()`, `.tick(t, est, impeller_speed) -> dict` (controller output + allocation + `state`, `result`, `mode`). `state` in `idle, preparing, moving, stopping, stopped`; `result` in `success, already_there, tolerance_pos, tolerance_att, tolerance_pos_endpoint, cancelled, localization_lost, overspeed_stop, overspeed_cutoff, speed_timeout` or `None`. Attributes `ctl`, `alloc`, `modes`, `mode`, `segment`, `last`. |

## spacekit.sensors — F. sensors (no rendering)

| Function / class | Description |
|---|---|
| `true_state(model, data, body)` | `(pos, quat, vel_world, omega_body)` of a body frame origin (truth, for graders). |
| `Imu(model, site, rng, rate_hz=62.5, frame=None, gyro_noise=2.9e-5, gyro_bias_rw=1e-6, gyro_bias0=1e-4, accel_noise=1e-3, accel_bias_rw=1e-4, accel_bias0=2e-3)` | `.read(model, data) -> (gyro, accel)` noisy (site frame), `.true(model, data)`; with `frame` the gyro includes the station rate and the accelerometer excludes the frame's fictitious / tidal accelerations. |
| `FeaturePointMap(points, cam_axis_body=(1,0,0), fov_deg=120, range_m=(0.3, 6.0), n_good=40)` | Feature richness = mapped points inside the view cone / `n_good`; `.count(pos, R, occlusion=None)`; callable `(pos, R, occlusion=None) -> [0, 1]`. `occlusion=(model, data, exclude_body)` adds ray tests. |
| `GridRichnessMap(origin, spacing, values)` | Richness on a grid (trilinear). |
| `Localizer(rng, richness=None, rate_hz=62.5, sigma_pos=0.0491/sqrt(3), sigma_att=0.0275/sqrt(3), tau=2.0, drift_pos=2*0.2777**2/1800, drift_att=2*0.0696**2/1800, richness_min=0.2, tau_reloc=5.0, vel_sigma=0.003, omega_sigma=2e-4, pos_sigma_lost=0.20, att_sigma_lost=20 deg)` | Per-axis error parameters (defaults: the published map-based RMSE of the error norm 0.0491 m / 0.0275 rad split over three axes; dead-reckoning diffusion that gives the published 0.2777 m / 0.0696 rad as the time-averaged error-norm RMSE of a 600 s activity). `.update(t, model, data, body) -> dict(pos, quat, vel, omega, cov_pos, cov_att, confidence, richness, pos_error, att_error)`; `forced_lost` attribute; `.confidence_from_cov(cov_pos, cov_att, thr_pos, thr_att)` (static, port of the estimator's pose confidence). |
| `DepthCamera(model, name, width, height, fx, fy, cx, cy, kind="camera", stride=1, range_m=(0.1, 4.0), noise=(0.002, 0.002), exclude_body=-1, rng=None, geomgroup=None)` | Ray-cast depth (`mj_multiRay`, one origin): `.read(model, data)` -> z-depth image, NaN invalid; noise `a + b d^2` when `rng` is given. |
| `fovy_from_intrinsics(fy, height)` / `intrinsics_from_fovy(fovy_deg, width, height)` | Pinhole <-> MuJoCo `fovy`. |
| `optical_to_mujoco_camera(R)` / `mujoco_camera_to_optical(R)` | Optical frame (x right, y down, z forward) <-> MuJoCo camera frame (looks along -z, y up). |
| `camera_xml(name, pos, quat_parent_optical, fy, height)` | MJCF `<camera>` element from an optical-frame pose and intrinsics. |
| `project_points(K, R_world_optical, t_world_optical, points)` | `(uv (N,2), depth (N,))`. |
| `distort_fov(xn, w)` / `undistort_fov(xd, w)` / `distort_radtan(xn, k1, k2, p1, p2)` | Lens models on normalised coordinates. |
| `camera_model(role, ref=None)` | Configured camera (`forward_wide`, `aft_wide`, `forward_depth`, `aft_depth`, `forward_imaging`): `dict(K, distortion, distortion_model, R_body_optical, t_body, q_body_optical, placeholder_in_source, width, height)`. |
| `Battery(n_packs=4, capacity_wh=49.0, soc0=1.0, avionics_w=35.2, other_w=6.0, arm_efficiency=0.5, arm_idle_w=1.0, arm_powered=True)` | `.update(dt, impeller_w, arm_mech_w, payload_w) -> W`, `.soc`, `.voltage()`, `.endurance_h(p)`, `.used` (Wh per consumer), `energy_wh`, `last_power_w`. |
| `JointSensors(model, joints, rng=None, resolution=2pi/4096, pos_noise=0, vel_noise=0.002)` | `.read(data) -> (q, qdot)`. |
| `ContactSensor(model, geoms, threshold=0.05)` | `.read(model, data) -> dict(in_contact, normal_force, n_contacts, partners)`. |

## spacekit.render (optional)

| Class | Description |
|---|---|
| `CameraRenderer(model, width=640, height=480, rng=None, noise_sigma=0.0)` | One `mujoco.Renderer`: `.rgb(data, camera)`, `.gray(...)`, `.depth(...)`, `.close()`; context manager. Needs an OpenGL backend (OSMesa on a host without a GPU). |

## spacekit.robot — G. reference-robot adapter

| Name | Description |
|---|---|
| `NAME_MAP` | Names expected in the free-flyer model: `root_body` (`core`), `root_joint` (`freeflyer_root`), `nozzle_sites` (`nozzle_1_1`..`nozzle_2_6`, firmware order), `imu_site`, `arm_joints`, `arm_actuators`, `gripper_actuator`, `gripper_joints`, `gripper_bodies`, `gripper_site`, `dock_site`, `cameras` (role -> camera name). |
| `VALUE_KEYS`, `NOZZLE_TABLE_KEYS` | Candidate keys looked up in the reference robot's values JSON file (flat dotted keys under `values`, nozzle rows under `tables`) per plant quantity. |
| `FreeFlyerSpec` | Dataclass: `propulsion` (plant `PropulsionParams`), `firmware` (reference dict), `mass_kg`, `com_m`, `inertia_kgm2`, `drag_cd`, `drag_area_m2`, `battery_count`, `battery_capacity_wh`, `sources`; `.reference()`, `.from_values(path, base=None)` (override from a values JSON file; missing keys keep the base value). |
| `FreeFlyerAdapter(model, spec=None, name_map=None, prefix="")` | Resolves names: `body`, `joint`, `qadr`, `dadr`, `subtree`, `nozzle_pos` / `nozzle_dir` (12,3, from the sites if present), `arm_joints`, `arm_actuators`, `gripper_actuator`, `gripper_joints`, `gripper_geoms`, `imu_site`, `cameras`; `.mass`, `.mass_properties(data=None) -> (M, com_body, J_body)`, `.set_pose(data, pos, quat, vel, omega_body)`, `.true_state(data)`. |
| `FreeFlyerRig(model, data, spec=None, name_map=None, prefix="", mode="standard", allocator="flight", rng=None, station_frame=None, flow=None, drag=True, drag_bodies=(), drag_shapes=None, jet_targets=(), suction=(), slow_every=10, spin_up=True, thrust_scale=None, gyro="physical", fan_efficiency=0.52, localizer=None, battery=None, extra_sources=(), extra_slow_sources=(), overspeed_stop=0.5, overspeed_cutoff=0.75)` | Wired simulation: plant (`plant`), wrench source (`prop_src`), `firmware`, `localizer` (noise-free truth if `rng` is None), `imu`, `battery`, `air`, `jets`, `suction`, `sim` (Simulator; firmware task at 62.5 Hz). Commands `.set_mode`, `.move_to(pos, quat, faceforward=True, desired=None)`, `.hold`, `.stop`, `.idle`, `.set_arm(targets)`, `.set_gripper(target)`; stepping `.step(n)`, `.run(duration)`, `.run_until_done(timeout)`; `.spin_up()`, `.estimate()`, `.impeller_power_w()`, `.telemetry()`, `log` / `record_every`. |

## Data

`data/firmware_reference.json` (schema `spacekit.firmware_reference/1`): `propulsion`, `mass_properties_configured`,
`control`, `flight_modes`, `planner`, `executive`, `cameras`. Regenerated from the firmware sources.

---

## v0.2 additions

### spacekit.aero — loose-item drag and the nozzle-jet field

| Name | Description |
|---|---|
| `AIR_DENSITY = 1.2`, `ITEM_CD` | Item drag defaults (plate / box 1.2, cube 1.05, cylinder cross 1.0 / axial 0.9, sphere 0.47). |
| `ItemShape(body, kind, size, grid=None, cd=None, offset=0, quat_wxyz=1)` | One shape of an item body: `'box'` (half extents, grid (nx, ny, nz) sample points, per-axis areas), `'cylinder'` / `'capsule'` (r, half length, n points along z), `'sphere'` (r). |
| `ItemAero(model, shapes, flow=None, jets=None, rho=1.2, shelters=None)` | Wrench source: quadratic drag of every sample point relative to `flow(x) + jets(x)` (zero inside `shelters`); per-axis for boxes, cross / axial for cylinders, isotropic for spheres; torques from the point offsets. `last_points`, `last_air`, `last_force`. |
| `Shelters(boxes)` | Closed volumes (oriented boxes: centre, axes as columns, half sizes) the cabin air and the jets do not reach, e.g. stowage pockets; `.mask(points)`. |
| `jet_velocity(x, noz_pos, noz_dir, thrust, v0, rho=1.2)` | Superposed round jets (core 6.2 d0, then u_c = v0 6.2 d0 / s, Gaussian 0.114 s; momentum flux = thrust). |
| `NozzleJetField(body, nozzle_pos, nozzle_dir, thrust_fn, dp_fn, cd_exit=0.9)` | Velocity field of the robot's 12 jets from a live plant (exit velocity `cd_exit sqrt(2 dp / rho)` per module); `.bind(data)`, `.exit_velocity()`, callable on points. |
| `StillAir`, `UniformAir(u)`, `settle_time_closed_form(m, cdA, u)` | Helpers. |

### spacekit.suction — lumped duct-and-fan network

| Name | Description |
|---|---|
| `INLET_BUILDS`, `MAIN_BUILDS` | Inlet builds as (K_face, r_face [Pa per m/s of face velocity]): `grille`, `screen`, `prefilter`, `filter_face` (the four), intermediates `low`, `mid`, and the smoke demo's `prefilter_double`. |
| `FAN_CURVE_Q`, `FAN_CURVE_DP`, `FAN_DESIGN_POINT`, `fan_dp(Q, omega_frac, size_scale)` | Published cabin-fan curve (affinity scaled). |
| `brent(f, a, b)` | Bracketed root finder (numpy-free Brent). |
| `Inlet(name, center, normal, t1, half, kind, flow)` | Return inlet geometry; `.velocity(x)` near-field sink line; `.local(x)`; `.area`. |
| `FanNetwork(inlets, build, Q_design, tau_fan=3, fan_scale='published'|'design_point'|float)` | One fan with parallel inlets: `.solve()` (operating point, cached), `.advance(dt)` (spin-down / up), `.set_fan(on, t)`, `.set_blocked(name, area)`, `.open_fraction(name)`; state `omega`, `Pc`, `Q`, `blocked`, inlet `flow`. |
| `SuctionFaces(model, items, networks, g0=0.0015, g_max=0.012, cos_min=0.5, sub=4)` | Wrench source: suction on the two large faces of item bodies (cells + sub-points, seal decay 1/(1+(g/g0)^2), force at each cell). `.update_networks()` feeds the sealed areas back; `covered`, `item_force`, `.hold_force(bodies)`. |
| `holding_curve(build, inlet_area, n_inlets, Q_design, phis)` | P_c vs the blocked fraction of one inlet. |

### spacekit.gripper — perching arm

| Name | Description |
|---|---|
| `GRIPPER_DEFAULTS`, `ARM_DEFAULTS`, `ARM_NAMES` | Spring torques at full opening (0.1059 / 0.0443 N m), preload span 2 rad, pad friction 1.8 (reference-build fingers; box fingers 1.2), stroke 1 s; servo kp 20 / kv 0.6, 1 N m, 0.12 rad/s; reference-build names. |
| `tilt_to_q(deg)`, `q_to_tilt(q)` | Firmware tilt (0 deployed aft, 180 stowed) <-> joint angle. |
| `rig_spec(spec, names=None, gripper=None, arm=None, prefix='', add_weld=True)` | Edit an MjSpec: remove the coupled position-actuated gripper (actuator + joint equalities), install the springs, pad friction, limit softness, arm servo gains, intra-robot excludes, inactive grasp weld `grasp_weld`. |
| `TendonGripper(model, joints, stroke_s=1, open0=0)` | Tendon = moving closing-side joint limit: `.set(cmd)`, `.step(dt)`, `.moving`, `.opening(data)`, `.tendon_load(model, data)` (motor-current proxy), `.latch(data)` / `.unlatch()`. |
| `ArmServo(model, data, actuators, joints, vel=0.12)` | Rate-limited setpoints: `.set(tilt=, pan=)`, `.step(data, dt)`, `.done(data, tol)`, `.joint_angles(data)`. |
| `set_fingers(model, data, joints, open_)`, `set_arm(model, data, joints, tilt_deg, pan)` | Scene initialisation only (qpos). |

### spacekit.grasp_rule

| Name | Description |
|---|---|
| `GraspRule(model, data, palm_body, finger_geoms, eq_name='grasp_weld', gripper=None, x_lo, x_hi, y_half, z_plane, angle_deg=20, slip_force_N=6.4, slip_torque_Nm=0.5, slip_ticks=3)` | `.check_capture(center, axis)`, `.attach(body)` (current relative pose; finger contacts off; fingers latched), `.release(why)`, `.load()`, `.check_slip()`; `held`, `events`. |

### spacekit.fixtures — MJCF fragments

| Name | Description |
|---|---|
| `RAIL_DEPTH, RAIL_WIDTH, RAIL_STANDOFF`, `CONTACT`, `MATERIALS` | Measured handrail section (35 x 19 mm, 43.9 mm off the wall); contact recipe. |
| `handrail(name, center_on_wall, axis, normal, length)` | Static stadium handrail on legs -> (xml, bar centre). |
| `handle_geoms`, `plate_item`, `soft_item(name, pos, quat, n, size, mass, hinge_k, centre_handle, corner_loop)`, `small_item` | Items with handrail-profile handles (sites `*_grasp*`). |
| `stowage_bag(name, mouth_center, normal, t1, mouth_half, depth, flap_k, flap_ref, protrude, wall_t, split=False, flap_len=None)` | Pocket + inward-only spring flap (or, `split=True`, short flaps from the +b / -b edges: `flap_len` = L or (L+b, L-b), 0 = bare edge; an elastic slit) -> (geoms, flap bodies, info incl. `preload_Nm`, `flap_joints`, `flap_edges`, `swing_radius`, `gap`). An item stays when it is pushed past the swing and is larger than the gap; the gap must pass the arm and open fingers. |
| `hook_loop_board`, `fragment_model(worldbody, contact)`, `frame_axes(normal, t1)` | Board; stand-alone model for `MjSpec.attach`; axes. |

### spacekit.station — ventilation, fans, CO2

| Name | Description |
|---|---|
| `CABIN_BAND`, `INLET_VICINITY_MAX`, `RETURN_GRILLE_CFM`, `SUPPLY_DIFFUSER_CFM`, `INTER_MODULE_CFM`, `HATCH_OPENING` | Published values used. |
| `StationVentilation(features, modules=('l2','h2','l1'), h=0.05, return_cfm, imv_cfm, builds='screen', circulation=None, band_target=0.69, fan_scale='design_point', tau_fan=3, vent_overrides=None)` | Boxes per module and hatch vestibule, potential-flow bases per fan (mass balance checked), two-cell circulation calibrated to the band; callable field `vent(x)`; `boxes`, `links`, `grilles` {id: Inlet}, `cabin_fans` {module: FanNetwork}, `imv_fans` {(hub, lab): SimpleFan}, `circulation`, `report`; `.band_fraction(m)`, `.fan_fractions()`, `.circulation_velocity(m, x)`. |
| `SimpleFan(name, tau_fan, flow)` | First-order fan (inter-module). |
| `CO2Balance(vent, bulk_fraction=0.9, residual_mix=0.02, scrubber=('l1', Q, eta), c0)` | Two-zone CO2 per module: `.add_source(module, kg_s)`, `.step(dt)`, `.module_mean(m)`, `.snapshot()` (mmHg), `removed_kg`. |
| `StationModel(vent, co2=None)` | `.request_fan(module, on, t, which='cabin'|'imv'|'all')` (logged in `requests`), `.advance(dt)`, `.telemetry()`, `.field`. |
| `pp_mmhg(c)`, `load_features(path)` | Helpers. |

### spacekit.zones — keep-in / keep-out validation

| Name | Description |
|---|---|
| `ZONE_RES = 0.08`, `ZONE_INFLATION = 0.30` | Mapper resolution; robot radius 0.28 + collision distance 0.02. |
| `ZoneMap(zones, res, inflation, inflation_z=None)` | Port of the firmware validator's voxel map (keep-in surfaces occupied one voxel outside, keep-in volumes free, keep-out surfaces occupied, dilation): `.check_point(p)` -> 'ok' / 'keep_in' / 'keep_out', `.check_positions(P)`, `.check_segment(seg, 10 Hz)`, `.value(p)`, `.free_fraction()`. |
| `resample_positions(seg, rate_hz)` | Segment positions at a fixed rate. |

### spacekit.primitives — firmware-level primitives

| Name | Description |
|---|---|
| `DOCK_DEFAULTS` | Approach 0.5 m, initial tolerance 1.0 m, cone 30 deg [G], attitude 45 deg [G], capture 4 cm / release 5 cm / docked after 5 s, hold 5 N, magnet falloff g0 4 mm [G], retract 8 s, charge 72 W. |
| `DockMagnets(robot_body, berth_pose, hold_force, g0, capture)` | Wrench source; `.retract(t, s)`, `gap`, `lateral`, `force`. |
| `FirmwareLayer(rig, names, berth_pose, station, suction, zones, grasp_rule, magnets, log_every, recorder)` | Wires arm servo, tendon gripper, grasp rule, magnets, station and zones around a `FreeFlyerRig`; `.run(action, timeout, label)` -> result; `timeline`, `docked`, `perched`, `thrust_saturation_ticks`, `.palm_frame()`, `.gripper_telemetry()`, `T_body_palm_deployed`. |
| `Action` subclasses | `Wait(s)`, `Call(fn)`, `SetMode(mode)`, `Hold(s)`, `FlyTo(pos, quat, mode, faceforward, validate, desired, resume)` (refuses `refused_keep_in` / `refused_keep_out`; resumes after `localization_lost`), `Undock(mode)`, `Dock(berth_speed, overshoot)` (`refused_outside_cone`, `docked`), `ArmTo(tilt_deg, pan_deg)`, `GripperCmd(open_)` (no grasp feedback), `GraspFeature(feature, standoff, approach_speed, depth, close_x, retarget_s, timeout_s, use_rule, rule_body, lead_s, label, close_on_the_move, keep_attitude, intercept, align_depth, align_tol, align_timeout_s, box_z_tol, align_attitude)` (`lead_s="auto"`: the pre-grasp aims at the feature predicted at the end of its planned move; `intercept`: every approach re-target aims at the feature predicted at the end of the planned rest-to-rest move; `align_depth`: hold the bar outside the fingertips until it is within `align_tol` of the finger plane, then go in; `align_attitude`: follow the item's attitude while aligning; `keep_attitude`: go in without re-orienting; `box_z_tol`: capture tolerance along the bar), `Perch(rail)`, `Unperch(rail)`. |
| `pose_for_palm(T_body_palm, palm_pos, palm_R)` | Robot pose that puts the palm at a target. |
| `carried_mass_properties(model, data, robot_body, extra_bodies)` | Mass, CoM, inertia of robot + held item (what the controller does not know). |
| `NozzleTestMode(params, level)`, `nozzle_factory_test(rig, carriage_dofs, fire_s=1.5, gap_s=2, level=3)` | Factory test on an air-bearing carriage: each nozzle full open for 1.5 s, per-nozzle thrust, impulse, carriage delta-v, yaw-rate change, direction check. |

### spacekit.cameras (optional)

| Name | Description |
|---|---|
| `CAMERA_ROLES`, `IMAGE_SIZE` | Role -> camera name (neutral ids) and published image sizes. |
| `set_intrinsics(model, cam_id, K, size)` | MuJoCo intrinsic pinhole (`cam_sensorsize` = size, `cam_intrinsic` = fx, fy, W/2 - cx, H/2 - cy). |
| `ViewportRenderer(model, max_size=(1366, 1024), max_geom=10000)` | One offscreen OpenGL context for every image size up to `max_size` (one renderer per process on OSMesa hosts): `.render(data, camera_id, (W, H), scene_option) -> (H, W, 3) uint8`, `.close()`. |
| `RobotCameras(model, roles, scale, exclude_body, scene_option, depth_stride, depth_groups=(1,2,3), renderer=None)` | `.rgb(data, role)`, `.depth_image(model, data, role)` (ray cast), `.close()`; `cams[role]` (K, size); shares `renderer` (a `ViewportRenderer`) or makes one at its largest colour size. |
| `fov_model_angles(K, size, w)`, `fisheye_maps(...)`, `fisheye_from_pinhole(img, K_wide, K_fish, size, w, maps)` | FOV-model field of view; fisheye image from a wider pinhole render. |

### Changed in v0.2

- `robot.NAME_MAP["cameras"]` uses the reference build's neutral camera ids (`cam_fwd_wide`, `cam_aft_wide`, `cam_fwd_img`, `cam_fwd_depth`, `cam_aft_depth`).

---

## v0.3 additions (reference build v1)

### spacekit.robot (changed)

| Name | Description |
|---|---|
| `FreeFlyerSpec.reference_configuration()` | Firmware reference constants (propulsion, controller, modes, impeller inertia 0.001 kg m^2 = the firmware's own) with the configured mass properties of the reference configuration (platform + arm module stowed, 10.450 kg, CoM, full inertia tensor: `mass.refcfg.*`) and the default battery count (`pwr.battery.default_count`), from `data/reference_configuration.json` (a copy of the reference robot's values). |
| `FreeFlyerSpec.from_values(path, base=None, firmware_config="refcfg")` | Also configures the firmware with the file's `mass.refcfg.*` (`firmware_config="firmware"` keeps the firmware reference's own); battery count reads `pwr.battery.default_count` first. |
| `FreeFlyerRig(..., spec=None \| "reference_configuration" \| "firmware" \| FreeFlyerSpec, cutoff_cuts_power=True)` | Default spec = `reference_configuration()`; `"firmware"` = the platform values of the firmware reference.  The over-speed cutoff now cuts the propulsion power. |
| `FreeFlyerRig.set_propulsion_power(on, modules=(0, 1))`, `power_events` | Power / cut the propulsion channels (fan drive, nozzle servos, module board) of the given modules. |

### spacekit.propulsion (changed)

| Name | Description |
|---|---|
| `FanModule.set_power(on)`, `.powered`, `coast_aero_load=True` | Unpowered channel: the fan drive is open-circuit (no current, no braking), the impeller coasts against its friction and (coast_aero_load) the air it still pumps (dp Q / (eta omega)); the servos stop driving, the flaps hold.  Re-power restarts the speed loop from the coasting speed. |
| `FanPropulsion.set_power(on, modules)`, `.powered` | Both modules. |
| `coast_down(p=None, level=3, servo_pwm=None, mass=10.449993, duration=120, aero_load=True, modules=(0, 1), settle_s=20)` | Thrust after a power cut with the servo commands held: dict(t, thrust, force_body, impeller_speed, impulse_body, dv_body, t_half_thrust, force_at_cut, speed_at_cut, residual_force_end). |

### spacekit.gripper (changed)

| Name | Description |
|---|---|
| `GRIPPER_DEFAULTS` | `pad_friction` 1.2 (silicone pads; v1 cage 6.47-6.60 N vs the published 6.43 N), `hull_friction` 0.4 [G] (printed phalanges, posts, springs), `pad_suffix` `_pad`, `finger_contact` None (= the two fingers collide iff the model has pad geoms). |
| `GRASP_PROFILES`, `grasp_profile(model)` | Grasp / perch depths and capture-box entries per finger recipe: `pads` (v1: bar 82 mm in front of the palm origin, close at 90 / 88 mm) and `generic` (the v0.2 values). |
| `rig_spec(...)` | With pad geoms: pad friction on the pads only, hull friction on the other finger geoms, the left / right finger excludes removed (summary keys `pads`, `hulls`, `finger_contact`). |
| `has_pad_geoms(model)`, `fingers_collide(model, joints)`, `touch_opening(model, data, joints)` | Recipe detection; smallest coupled opening without finger overlap (bisection on `mj_geomDistance`). |
| `set_fingers(model, data, joints, open_, opening=None)` | Closed = the closing limits, or the touch opening when the fingers collide. |

### spacekit.primitives (changed / new)

| Name | Description |
|---|---|
| `DOCK_DEFAULTS` | Approach cone 6 deg about the berth axis and attitude 20 deg at the start of the berthing move (measured), initial tolerance 1.0 m [R]. |
| `Dock(berth_speed=0.02, overshoot=0.0134, mode="standard", go_to_approach=True, spin_down=True, enforce_cone=True, max_reapproach=2, berth_mode="fine", check_timeout_s=30)` | The firmware's sequence: approach pose (nominal), dock-relative localisation (`fw.dock_fiducial`), cone check (outside: back to the approach pose), berthing move in the precision mode to the complete pose, the dock sees the robot (pins), check attached (the move back must FAIL), spin down.  Results `docked` / `attached` / `refused_outside_cone` / `eps_dock_failed` / `attach_failed`. Works with `DockMagnets` (v0.2) or `dock.BerthDock`. |
| `Undock(mode)` | Releases the dock-relative localisation; `info()`: magnet pull at the retraction, largest pull after it and beyond 5 mm, time to 5 mm out. |
| `Perch(..., check="firmware" \| "displacement", close_x=None)` | Depth / capture entry from the gripper profile; attach check as the firmware: the move back to the approach pose must FAIL (`check_result`, cage play `check_displacement_m`). |
| `GraspFeature(..., depth=None, close_x=None)` | None = the gripper profile's values. |
| `FlyToFiducial(fiducial, pos, quat, settle_s=1, keep=False, validate=False, mode, desired, faceforward=False)` | Fly to a pose next to a fiducial plate on the fiducial-relative estimate (about 1 cm instead of the map's 5 cm); plain FlyTo without a fiducial-capable localiser. |
| `FirmwareLayer.grasp_defaults`, `.dock_fiducial` | The gripper profile of the model; the dock target fiducial (set by the scene adapter). |

### spacekit.dock (new)

| Name | Description |
|---|---|
| `DOCK_V1` | Magnets 4 x 12.1 N / (1 + g / 1 mm)^2 at a 0.39 mm working gap, pins 12010 N/m (12.0 N at 1.0 mm, 1.4 mm travel, damping 2 N s/m), retraction 30 mm at 6.2 mm/s re-extending after 8 s, robot cups (mouth r 13.42 mm, 17.4 deg, 22.3 mm deep, 13.2 mm hole), robot dock face at body x -0.1461 m, berth contact poses. |
| `dock_model_path(default=None)`, `attach_dock_v1(spec, dock_pos, dock_quat, xml=None, visual=True)` | Attach the reference build's dock (its robot, keyframes and the robot's mesh assets dropped; moving parts, pins, magnets and the merged screw-head hull without contacts). |
| `rig_dock_spec(spec, robot_root="core", n_funnel=16)` | Open the robot's dock face for the lances: the dock-plate and mid-band proxies replaced by boxes around two lance channels, cup funnels, through-hole rings, face annuli (an adapter edit; summary). |
| `berth_contact_pose(dock_pos, dock_quat, berth=1)` | Robot body pose resting on a berth. |
| `BerthDock(robot_body, berth_pose, params=None)` | Wrench source: magnets (strike-plate overlap) and pins on the berth axis, retraction state; `gap`, `lateral`, `force`, `pins_N`, `carrier`, `eps_contact`, `retract(t, s)`, `net_hold_N()` (the `DockMagnets` interface). |
| `DockRelativeLocalizer(rng, base=None, rmse_m=0.01, rmse_range_m=0.6, floor_m=0.002, att_rmse_rad=0.5 deg, tau=1, half_fov_deg=40, range_m=(0.05, 1.5), hold_s=20, hold_drift=1e-6)` | Map localiser + fiducial-relative mode: `reference(name, points, normal, cam_offset_body, cam_axis_body)`, `release()`, `update(...)` (Localizer signature, `source`); inertial hold after the last fix. |
| `fiducial_visible(cam_pos, cam_axis, points, normal, ...)`, `camera_in_body(model, cam, body)`, `plate_points(model, data, geom, normal_hint)` | Visibility of marker points; a camera in the body frame; marker points on a plate geom. |

### spacekit.station (changed)

| Name | Description |
|---|---|
| `DESIGN_RETURN_INLET_CFM = 40`, `FAN_STOP_RULES` | The task station's design flow per return inlet and the fan-stop flight rules (2 stops per unit per mission, restart after 60 s, 120 s from a restart to the next stop). |
| `StationModel(vent, co2=None, rules=None)` | `request_fan_stop(module, t)` -> 'stopped' or 'refused_*'; `request_fan_restart(module, t)`; automatic restart in `advance`; `fan_stop_state(module)`; `fan_events`; `request_fan` stays the unruled setup call. |

### spacekit.suction, spacekit.fixtures (changed)

| Name | Description |
|---|---|
| `INLET_BUILDS["prefilter_light"]` | K_face 8, r_face 64 Pa per m/s: at 40 cfm per inlet the soft blocker (0.036 m^2 sealed) is held with 1.15 N (window 0.85-1.45 N) [G]. |
| `fixtures.STANDARD_BAG`, `standard_bag(name, mouth_center, normal, t1, params=None)` | The standard stowage bag (the short-flap rule): mouth 0.36 x 0.36 m, 0.24 m deep, two 0.08 m flaps (0.20 m slit), fiducial plate beside the mouth; info has `fiducial`. |
| `fixtures.fiducial_plate(name, center, normal, t1, size)` | Display-only marker plate with a site. |

### spacekit.sensors, spacekit.cameras (changed)

| Name | Description |
|---|---|
| `DepthCamera(..., exclude_body=-1, ..., exclude_bodies=None)` | `exclude_bodies`: further bodies the rays do not see; the rays skip the camera's mount body when it is in the set (else `exclude_body`) and continue past any other excluded geom they meet first (`n_continued` per read). |
| `RobotCameras(..., exclude_body=core)` | The depth rays ignore the core and every body rigidly attached to it (`rigid_bodies`: same weld body), not only the core: on the reference build v1 the forward depth camera sits on the forward sensor module inside its collision proxy and v0.2 read an all-invalid image there.  Moving parts (arm, gripper) stay visible. |

### spacekit.visual_twin (new)

| Name | Description |
|---|---|
| `VisualTwin(model, prefix="", display_rev_s=1.5)` | Display joints (two impellers, 48 nozzle joints) and status lights (full-detail set) of the reference build v1. |
| `VisualTwin.sample(rig, layer=None)` | 17-vector: impeller speeds, the plant's flap angles, executive state code (+8 docked, +16 perched), channel power; reads only. |
| `.advance_phase(phase, sample, dt_video)`, `.apply(model, data, sample, phase, lights=True)`, `.reset_lights(model)` | Slowed impeller spin for video; writes a RENDER copy only, so the physics is bit-identical with and without the twin. |

### Changed in v0.3

- Default model paths point to the reference build v1 (`build_v1/mjcf/freeflyer_v1.xml`); v0 stays usable (its fingers need `gripper=dict(pad_friction=1.8)`).
- The rig's default firmware configuration is the reference configuration (`mass.refcfg.*`).
- `Dock` follows the firmware's sequence (approach pose, dock-relative localisation, precision berthing, failing check move); the v0.2 30 / 45 deg cone check at the start is replaced by the measured 6 / 20 deg cone at the berthing start.
- `Perch` checks attachment as the firmware does (the move back must fail).
- `RobotCameras` depth rays exclude the robot's whole rigid structure (the v1 forward depth camera was blind in v0.2).


## v0.3.1 (reference build v1.2)

From reference build v1.2 on, the four adapter corrections are in the model files, and every model says which ones in a
custom text element `refbuild` = "reference build v1.2; corrections: <tokens>".  The adapters read it (by file content,
never by the path) and do not apply a correction twice; models without the text (v0, v1, v1.1, the placeholder robot)
behave exactly as in v0.3.0.

| Function | Notes |
|---|---|
| `refbuild.build_info(model_or_spec)` | `{'text', 'version', 'corrections'}` (frozenset of `gripper_pads`, `dock_face_cups`, `berth_display_only`, `fwd_depth_window`); texts named `<prefix>refbuild` (attach prefixes) are read too. |
| `refbuild.has_correction(model_or_spec, token)` | True if the model already contains that correction. |
| `gripper.rig_spec(...)` | Unchanged behaviour (the recipe's values are set again, which changes nothing on v1.2 and still carries a caller's friction override); summary key `build_recipe`. |
| `dock.rig_dock_spec(spec, ...)` | On a model that has the cups (the proxy boxes, or `dock_face_cups` in its text) returns `already_in_model=True` and changes nothing. |
| `dock.attach_dock_v1(...)` | Deletes the dock scene's custom texts before attaching (the scene's robot is dropped; its `refbuild` text would collide with the robot's); summary keys `display_parts_in_model`, `refbuild`. |
| `cameras.RobotCameras(..., exclude_body=core)` | With `fwd_depth_window` in the model the depth rays exclude only `exclude_body` (the forward depth camera looks through its window); `rigid_bodies` is still reported, `depth_window_in_model` says which rule is used. |

---

## v0.4 additions

### spacekit.sensors -- localiser error model (defect fix, default changed)

| Name | Description |
|---|---|
| `Localizer(..., error_model="consistent", corr_time=60.0, corr_length=1.0, corr_angle=1.0, vio_tau=10.0)` | `'consistent'` (default): the estimate is a smooth trajectory whose velocity and body rate are its own derivative (+ the independent `vel_sigma` / `omega_sigma` noise). Map mode: per axis the error is the reported std (the v0.3 covariance bookkeeping, stationary `sigma / sqrt(richness)`, so the 4.91 cm / 2.75 deg RMSE is kept) times a unit-variance critically damped second-order Gauss-Markov state in an error clock running at `1/corr_time + |v|/corr_length + |w|/corr_angle` (at least `2/tau_reloc` while re-converging), so the true error follows the reported covariance when the richness changes or after dead reckoning; dead reckoning: the error integrates an OU velocity error of correlation time `vio_tau` whose long-run diffusion is `drift_pos` / `drift_att` (starting from its stationary distribution when dead reckoning begins). The reported covariance and the confidence flag are computed as in v0.3. `corr_*`, `vio_tau` are [G]. `'legacy'`: v0.3 bit for bit (first-order error, correlation time `tau`, velocity / rate errors independent of it -- a hovering robot then random-walks 0.2-0.7 m in 10 min). `.clock_rate(v, w)`; state `e_pos`, `de_pos` (d/dt), `ds_pos` (d/ds), same for the attitude. |

<!-- spacekit v0.4 patch 02 -->
### spacekit.control, spacekit.robot -- stop feedforward (defect fix, default changed)

| Name | Description |
|---|---|
| `Firmware(..., stop_feedforward="zero")`, `FreeFlyerRig(..., stop_feedforward="zero")` | A move stopped early (`stop()` / `hold()` while moving, a tolerance failure, an over-speed stop, a localisation loss) keeps the active setpoint's pose with zero velocity, acceleration, rate and angular acceleration. `'hold'` = v0.3 and the flight software's controller: the stop mode keeps the stale acceleration feedforward; after a stop inside a rotation ramp the robot then spins at alpha / omega_kd (0.0545 rad/s in STANDARD's face-forward turns) and the executive never reports `stopped`. Moves that end their segment are unaffected. |

<!-- spacekit v0.4 patch 03 -->
### spacekit.primitives -- fly-to timeout (defect fix, default on)

| Name | Description |
|---|---|
| `FlyTo(..., timeout_margin_s=30.0)` | Returns `'timeout'` (and tells the executive to stop) when the move has not reported `stopped` 30 s after the end of its latest planned segment (not while the fans change speed); `info()['timed_out']` has the planned end and the executive's state / result. `None` = v0.3 (the move ends only with the caller's timeout). |

<!-- spacekit v0.4 patch 04 -->
### spacekit.zones, spacekit.primitives -- band exit (defect fix, default on in the firmware layer)

| Name | Description |
|---|---|
| `ZoneMap.check_positions(P, band_exit=False)`, `check_segment(seg, rate_hz, band_exit=False)` | `band_exit=True`: a path that starts inside the inflated band (an occupied cell that is free before the dilation) is accepted if its leading samples inside the band never get closer to the occupied cells (`clearance` strictly increasing between samples that move, equal between samples that do not) and every sample from the first free one on is free. The validator's default stays the v0.3 port. |
| `ZoneMap.raw`, `ZoneMap.clearance(P)` | The occupancy before the dilation; normalised clearance in the dilation's metric (>= 1 outside the band). |
| `FirmwareLayer(..., zone_band_exit=True)`, `FirmwareLayer.check_segment(seg)` | The validation `FlyTo` uses: the zone map with the band-exit rule (False = v0.3). |

<!-- spacekit v0.4 patch 05 -->
### spacekit.zones, spacekit.primitives -- arm-deployed inflation (option, default off)

| Name | Description |
|---|---|
| `ZoneMap(..., inflation_arm=None, inflation_arm_z=None)` | A second map dilated by `inflation_arm` (e.g. 0.50 m) for a robot with its arm deployed; the stowed map is unchanged bit for bit. Query methods take `arm=True` (`value`, `check_point`, `check_positions`, `check_segment`, `clearance`). |
| `FirmwareLayer.arm_deployed(tol_deg=5)`, `check_segment(seg, arm_exempt=False)`, `FlyTo(..., arm_exempt=False)` | With a zone map that has `inflation_arm`, `FlyTo` validates on the arm map while the arm's tilt (encoder or command) is more than 5 deg from stowed. The final segments of capture, perch and dock are not validated at all (as before). |

<!-- spacekit v0.4 patch 06 -->
### spacekit.primitives -- fiducial scope (option, default off)

| Name | Description |
|---|---|
| `FirmwareLayer(..., fiducial_scope="sticky")` | `'action'`: the localiser's fiducial reference is released when the action that set it ends (returns or is abandoned on the caller's timeout), even a `FlyToFiducial(keep=True)`; `fiducial_releases` counts them. `'sticky'` = v0.3 (the reference and its 20 s inertial hold carry into the next action). |

<!-- spacekit v0.4 patch 07 -->
### spacekit.primitives -- holding interlock (option, default off)

| Name | Description |
|---|---|
| `FirmwareLayer(..., holding_interlock=False)`, `holding`, `interlock_refusals`, `arm_command(tilt, pan)` | `True`: after a `GraspFeature` (capture) returned and until the tendon is commanded open, every arm command is refused with `'refused_holding'` and the joints keep their goal: `ArmTo` returns it at once, `fw.arm.set(...)` returns it (the layer's arm is wrapped). On the command state, not on a grasp sensor. |
| `GraspFeature(..., engage_interlock=True)` | Whether the capture engages the interlock; the perch behaviour passes False. |

<!-- spacekit v0.4 patch 08 -->
### spacekit.suction -- faster item suction (bit for bit)

`SuctionFaces.accumulate` gives the same wrench, sealed areas and item forces as v0.3 bit for bit (tested against a
copy of the v0.3 method, also with other sources' wrench already in W): the inlet's network and the candidate cells'
gathers are cached, cells that cannot reach the seal box are skipped, the row-wise cross product replaces `np.cross`,
and each body's wrench row is updated by one running sum (`np.add.accumulate`, the same additions in the same order as
`np.add.at`) together with its item force.  About 1.5 x faster per call on the task-like site (673 -> 446 us CPU per
call; the site with suction every physics step 0.93 -> 0.85 CPU s per simulated s).

<!-- spacekit v0.4 patch 09 -->
### spacekit.dock -- dock model early-out (bit for bit)

| Name | Description |
|---|---|
| `BerthDock(..., far_skip=True)`, `n_skipped` | Skips the magnet and pin evaluation when no magnet can be in range or over its plate and the pins cannot be compressed whatever the attitude (each robot-side point moves at most 2 sin(theta/2) abs(q) relative to the body origin); the same (zero) wrench and state as the full evaluation. |

<!-- spacekit v0.4 patch 10 -->
### spacekit.scenes (new) -- static geometry in cell bodies

| Name | Description |
|---|---|
| `cluster_static_geoms(in_xml, out_xml, cell=1.0, prefix="sk_cell")`, `cluster_static_geoms_xml(text, base_dir, cell, prefix)` | MJCF rewrite: every collision geom that is a direct child of `<worldbody>` moves unchanged into a static body (origin, identity, no joint) per cubic cell; visual-only geoms stay; asset paths made absolute; report `moved`, `cells`, `kept_on_world`. Same compiled geoms; a run without robot-structure contacts is bit-identical; with contacts the contact set is the same and its order differs (round-off). Standard library only. |

<!-- spacekit v0.4 patch 11 -->
### spacekit.station -- per-unit flows, inlet list, inhibited units

| Name | Description |
|---|---|
| `StationVentilation(..., return_cfm=float \| {module: cfm}, imv_cfm=float \| {(hub, lab) or 'hub:lab': cfm}, inlets=None)` | Per cabin air unit return flow per inlet and per link inter-module flow (scalars = v0.3 bit for bit); `inlets`: the station scene's inlet table (`wall_point`, `normal`, `quat_wxyz`, `opening_m`, `frame_m`) or vent dicts, replacing every return grille of the catalogue. `return_q_m`, `imv_q_l`. |
| `StationModel(..., inhibited={module: reason})`, `set_inhibited(module, reason)` | Stop requests of an inhibited unit answer `'refused_inhibited'` (checked first); `fan_stop_state` has `inhibited` for such a unit. |

<!-- spacekit v0.4 patch 12 -->
### spacekit.fixtures -- soft item loop position

| Name | Description |
|---|---|
| `soft_item(..., loop="default", loop_yoff=None)`, `SOFT_ITEM_LOOPS` | `'corner'` (v0.3), `'edge_mid'` (the middle of the end strip, the task's loop; the same XML as the station library's insertion), `None`; the default follows `corner_loop` (v0.3 output byte for byte); `loop_yoff` sets the loop's local y. The loop's site stays `{name}_grasp_corner`. |

<!-- spacekit v0.4 patch 13 -->
### spacekit.cameras -- near clip, exposure, lamps

| Name | Description |
|---|---|
| `RobotCameras(..., near_m=0.015, exposure=None, exposure_range=(1/16, 8), exposure_target=0.18)`, `last_exposure`, `set_lamp(name, on)` | Near clipping distance 1.5 cm (model-wide `vis.map.znear`; None keeps the model's); exposure: a fixed gain or `'auto'` (centre-weighted, the gain of a frame from the camera's previous frame), applied by scaling the lights, the headlight and the material emission during the render. |
| `set_near_clip(model, near_m)`, `exposure_gain(img, gain, target, gain_range, sigma)` | Helpers. |
| `add_robot_lamps(spec, body="core", prefix="", lamps=LAMPS, active=False, **LAMP_DEFAULTS)`, `LAMPS`, `LAMP_DEFAULTS` | The robot's two lamps as spot lights (front / aft lamp windows; cutoff 30 deg, cos^4.5 taper, inverse-square fall-off [G]; `intensity` 452 cd stored for photometric renderers). |

<!-- spacekit v0.4 patch 14 -->
### spacekit.sensors -- robot-item contact events

| Name | Description |
|---|---|
| `ContactEvents(model, bodies_a, bodies_b, min_force=0.0)`, `.attach(sim)`, `.update(data)`, `events`, `active`, `touching(b)` | Per physics step: contact events between two body groups (onset, end, steps, peak normal force, impulse, normal speed at the onset); reads only (physics bit-identical). |

<!-- spacekit v0.4 patch 16 -->
### Changed in v0.4

- Defaults that change results (defect fixes): the localiser's `error_model='consistent'`; the firmware's
  `stop_feedforward='zero'`; `FlyTo(timeout_margin_s=30)`; `FirmwareLayer(zone_band_exit=True)`;
  `RobotCameras(near_m=0.015)` (images only).  Each has its v0.3 value as an option.
- Options that default to v0.3: `ZoneMap(inflation_arm)`, `FirmwareLayer(fiducial_scope, holding_interlock)`,
  `RobotCameras(exposure)`, `add_robot_lamps`.
- Bit for bit: the suction and dock speed-ups, the station's scalar flows, the soft item's default XML.

<!-- spacekit v0.4 -->
## v0.4 additions (the localiser models, the suction layering, the band rule)

v0.3.1 (reference build v1.2's `refbuild` text) and the sixteen v0.4 patches are merged; the version is 0.4.0.

| Function | Notes |
|---|---|
| `dock.DockRelativeLocalizer(..., error_model="consistent", corr_time=60, corr_length=1, corr_angle=1, vio_tau=10)` | the fiducial-relative estimate uses the map localiser's consistent model -- per axis error = the reported std at the current range (1 cm at 0.6 m, proportional, floor 2 mm; 0.5 deg) x a unit-variance critically damped second-order Gauss-Markov state in the error clock ds/dt = 1 / corr_time + \|v\| / corr_length + \|w\| / corr_angle [G]; velocity / rate estimates carry its derivative plus the v0.3 independent noise; within `hold_s` of the last fix the position error integrates an Ornstein-Uhlenbeck velocity error (vio_tau, long-run diffusion `hold_drift`), the attitude error is held; back in view the held error is carried over and re-converges with the clock >= 2 / tau for 2 tau.  `error_model="legacy"` = v0.3 bit for bit.  `de` (the error's rate), `last_clock`, `clock_rate(v, w)`. |
| `suction.SuctionFaces(..., occlusion=True, occl_tol=1e-4)` | only the layer that directly faces an inlet or a holding-screen face is sucked: a sub-point whose path to the face along the inlet normal crosses the large face of another item in the seal box gets no suction and adds no sealed area (a pouch on a sheet that seals the face: 0.12 N in v0.3, 0 N now).  Bit-identical to v0.3 when no item lies between another and the face (one item, side-by-side items, a soft item's strips lying on or peeling from the face).  `shadowed` {(inlet, body): shadowed sub-points} after each evaluation.  `occlusion=False` = v0.3 / the v0.4 patches' v0.4.  Cost: about 25 us per call on an inlet with a multi-body item (the necessary conditions from the faces' corners), more only when something is shadowed. |
| `zones.ZoneMap(..., approach_tol=0.01)`, `ZoneMap.check_positions / check_segment(..., band_exit=True \| "exit" \| False)`, `FirmwareLayer(zone_band_exit=True \| "exit" \| False)` | `band_exit=True` (the firmware layer's default) accepts a segment that starts inside the inflated band if it leaves the band by the rule, or if no sample is unknown or inside an occupied cell, it never re-enters the band after leaving it and no in-band sample is closer to the occupied cells than any earlier sample by more than `approach_tol` (along the wall, staying in the band, turning in place: accepted; any approach: refused; the end point may be in the band).  `"exit"` = the departure rule only (the v0.4 patches' draft); `False` = v0.3.  Free starts are validated exactly as before. |

### Changed in v0.4

- Defaults that change results: `DockRelativeLocalizer(error_model="consistent")` (records with a fiducial reference: the dock
  and fly-to-plate records); `SuctionFaces(occlusion=True)` and the merged band rule of `FirmwareLayer(zone_band_exit=True)`
  change results only when their condition occurs (an item between another item and a face; a segment that starts in the
  band and does not get closer).  Each has its earlier value as an option.
