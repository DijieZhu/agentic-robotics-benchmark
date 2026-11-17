# DA-6 desktop arm - mass properties

Document DA6-MAS-001, revision A. Per link: mass, centre of mass (CoM) and inertia tensor about the CoM, all in the **link CAD frame** (kinematics sheet, section 6: parallel to the base frame at the zero pose, origin on the joint axis that carries the link). The link totals include the printed parts, the fasteners and the BUS-SERVO-12V-A servo body mounted in the link (each link except the jaw carries the servo that drives the next joint: base - servo 1, shoulder - servo 2, upper arm - servo 3, forearm - servo 4, wrist - servo 5, gripper - servo 6). The jaw is listed separately and is not included in the gripper.

Inertia tensor convention: I = [[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]] with Ixx = integral of (y^2 + z^2) dm and the products of inertia Ixy = -integral of x y dm (and likewise Ixz, Iyz), about the CoM, axes parallel to the link CAD frame.

| link | mass [g] | CoM x [mm] | CoM y [mm] | CoM z [mm] | Ixx [kg m^2] | Iyy [kg m^2] | Izz [kg m^2] | Ixy [kg m^2] | Ixz [kg m^2] | Iyz [kg m^2] |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 135.449 | 0.023 | -21.952 | 38.413 | 1.2365e-04 | 1.1746e-04 | 1.1425e-04 | 2.5085e-07 | -3.3725e-08 | -2.4155e-06 |
| shoulder | 99.635 | -0.133 | 30.822 | 30.276 | 9.9200e-05 | 9.9486e-05 | 2.0426e-05 | 4.0624e-08 | 4.7372e-07 | -1.2102e-06 |
| upper arm | 115.778 | -18.652 | -14.281 | 93.348 | 1.9141e-04 | 1.9877e-04 | 4.3627e-05 | 9.7511e-08 | -2.4661e-07 | 9.3507e-06 |
| forearm | 110.403 | -18.308 | 0.764 | 106.869 | 2.0745e-04 | 2.2273e-04 | 2.8400e-05 | -2.8022e-08 | 1.6172e-06 | 2.7476e-06 |
| wrist | 75.256 | -28.480 | -0.079 | 46.228 | 1.9325e-05 | 3.3313e-05 | 2.2770e-05 | 2.5776e-08 | 3.2035e-06 | -4.0848e-08 |
| gripper | 84.167 | -0.155 | -1.464 | 24.647 | 3.8485e-05 | 2.2164e-05 | 3.1480e-05 | -1.3430e-07 | -2.1964e-08 | -4.4697e-06 |
| jaw | 8.634 | 19.293 | 1.582 | 28.415 | 3.4296e-06 | 4.4275e-06 | 1.4077e-06 | 4.4712e-09 | 7.7315e-08 | -2.2622e-07 |

| total | mass [g] |
|---|---|
| base to gripper (without the jaw) | 620.689 |
| jaw | 8.634 |
| complete arm (all seven links) | 629.323 |
| moving mass (all links above the base) | 493.874 |

Tools are not included: the dispensing needle with its holder block (20 g, centre of mass on the needle axis 8.0 mm beyond the TCP along its +z; kinematics sheet, section 5.3, and needle drawing) and gripped parts are additional loads on the gripper.

**Tolerance**: each link's mass and each component of its inertia tensor: +-5 % of the tabulated value (printed parts: infill and wall variation). The CoM positions are the CAD values.

