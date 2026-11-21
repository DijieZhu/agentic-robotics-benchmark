"""The grader of freeflyer_cad_design_from_photos: one extracted FreeCAD document against the maker's published
appearance meshes and robot description (read by ../makerstd), in millimetres.

Modules:
    util        bands, rigid transforms, small helpers, the two error kinds (TaskError, item errors)
    geom        B-rep reading and checking (OCP), tessellation, seeded surface samples, the tangent-disk distance
    reference   the standard: every tag's maker triangles at configuration Z, its dense samples, the graded samples of
                the seen tables, the six standard axis lines
    submission  the extraction (fcstd_extract.py's model.json): members, tags, solids, joints, grounded members
    align       the rigid alignment of the whole unit onto the standard
    score       the combining layer: surface, joint_axes, structure, the floor and the design score
    run         the grade of one extraction and the command line (grade.py)

The rules are stated in run.py's docstring; every number is in grader_config.json.
"""
VERSION = "ffcad.grade/1"
