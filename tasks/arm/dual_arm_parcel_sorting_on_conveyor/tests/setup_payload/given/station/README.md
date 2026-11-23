# The station's geometry

`station.xml` is the station's geometry as a MuJoCo model (MJCF): the conveyor's collision boxes exactly as the cell
has them (`describe()["station"]["hardware"]` and `cell_manual.md` section 2), the belt surface as a fixed box, the
floor, the `infeed` camera (`inspect_cam`, at the installation drawing's pose with the data sheet's field of view and
image size; the installed camera differs within the tolerances of `cell_manual.md` section 6) and marker sites for the lines
(`entry_line`: the upstream end of the set-down range, `scan_line`, `belt_end`). `station_with_robot.xml` is the
same with the robot of
`../robot/dual_arm_with_grippers.xml` attached at the origin.

It is geometry only: the friction, contact and drive settings in these files are MuJoCo's defaults, not the cell's;
the belt does not move; there is no parcel and no light. World frame and units as in `cell_manual.md` section 1.
