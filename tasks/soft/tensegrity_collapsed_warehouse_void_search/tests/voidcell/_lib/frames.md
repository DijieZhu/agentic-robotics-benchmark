# Frames and numbering of the cell (hidden side; the agent's handbook and data sheets are written from this file)

All lengths in metres unless a field name says `_mm`; angles in radians unless `_deg` / `_dps`.

## Site frame

x east, y north, z up; origin = the south-west floor corner of the site rectangle (4.4 x 4.0 m in the W2 family);
the floor is z = 0. Every position the station gives (pad, search points, camera extrinsic, `pose()` nodes) is in this
frame.

## Camera frame and pixels

OpenCV convention: camera x = image right (+u), y = image down (+v), z = forward along the optical axis.
`T_world_camera` (4 x 4) maps camera-frame points to the site frame. Pixel (u, v) with integer u, v is the centre of
column u, row v. A site point P projects as u = fx X / Z + cx, v = fy Y / Z + cy with (X, Y, Z) = R^T (P - t),
T_world_camera = [R t; 0 1]. Depth = Z (along the optical axis, not the ray length), uint16 counts x 0.00025 m;
0 = no reading. The cell renders with the TRUE extrinsic (site.json camera.true); mission() gives the calibrated
extrinsic (site.json camera.given: the true one with an error of at most 5 mm and 0.2 deg).

## Robot: struts, nodes, end-cap colours

| strut | nodes (even, odd) | end-cap colour |
|---|---|---|
| 0 | 0, 1 | red |
| 1 | 2, 3 | green |
| 2 | 4, 5 | blue |

A node is the centre of an end cap (the 17.5 mm end-cap sphere; 145.4 mm from the strut's middle along its axis).
Both ends of one strut have the same colour. The even node of each strut is at the +z end of the strut mesh, the end
the control board's USB socket points to (measured; ysim parameter `mesh_end_to_node`, ysim/params.py).

Planar pose (the scoring's and the primitive table's): position = planar centroid of the six nodes; heading =
direction of mean(nodes 0, 2, 4) - mean(nodes 1, 3, 5), counter-clockwise from site +x.

Rollable faces (bottom triples a primitive may start from): (0,2,5) (0,3,5) (1,2,4) (1,2,5) (0,3,4) (1,3,4). Each holds
exactly one node of every strut. The two end triangles (0,2,4) and (1,3,5) (also one node per strut) are not
rollable: no primitive starts there.

Lines and sensors (the cell's numbering = the robot's): motor / line i runs from the motor node to the line node of
`CABLES[i]` = (5,3) (3,1) (1,5) (2,0) (0,4) (4,2); short sensor i (telemetry `sensor_length_mm[i]`, i = 0-5) runs
parallel to line i; long sensors 6, 7, 8 join nodes (2,5) (0,3) (1,4).

## IMU frames (one per strut, `imu_accel_g[k]`, `imu_gyro_dps[k]` of strut k)

As built in the vendored simulation base (ysim `geometry.strut_features`, the site `imu<k>` of the model): origin at
the centroid of the strut's control board inside its electronics enclosure;

- **x = along the strut axis, pointing from the odd node to the even node** (2k+1 -> 2k);
- **z = the board's normal** (the board's principal-axis fit; which way the chip faces is not verified);
- y = z x x (right-handed).

So the x reading of the accelerometer alone tells which end of a strut is lower: at rest the accelerometer reads the
specific force (+1 g pointing up), so `imu_accel_g[k][0] > 0` means the even node 2k is the higher end and the odd node
2k+1 the lower one (and the other way round). The y / z axes are an ASSUMPTION of the base (`imu_site`: the chip's
axes on the board are not verified; the board's principal axes are used); only x is tied to the node numbering.

Noise (rated values of this robot, telemetry.py): accelerometer 2 mg rms per axis; gyroscope 0.1 deg/s rms per axis plus
one constant bias per axis and episode, uniform in [-0.5, 0.5] deg/s. Sample rate 100 Hz (one per control period).

## Times

Episode clock: 0 at the start of the episode (after the set-down settled); advances only in `step(dt)`, in control
periods of 10 ms (the physics step is 0.25 ms in ysim 0.3.0 and 0.3.1). Camera exposure k at the first control tick at or after
k / 7 s; the frame's `t` is that tick's time. Telemetry record = the state after a control period, `t` = its end.
