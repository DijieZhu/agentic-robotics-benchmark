"""twinprobe: the probe library of the flight twin, shared by the grader and the check tool.

The same files run in both, so a twin that passes the public gate here passes it there, and the rays, the hazard
check and the one-flight fit computed here are the ones the score is computed from. The library reads no ground
truth and holds no numbers of its own: every number comes from the task's configuration, a flat mapping
{key: value} of the public entries (for example "hazard.voxel"), handed to each module's from_config / params
function.

    posechain    frames, quaternions, the camera mounting, the two-camera transform, the time convention,
                 cameras.json
    widemodel    the wide camera's one-coefficient wide-angle model and its pixel convention
    raycast      rays against a twin's collision geometry (meshes as their convex hulls, every geom opaque)
    insidesolid  is a point inside the solid of a geom (closed forms; meshes by their hull planes)
    hazard       the single-frame hazard check of straight moves
    oneflight    the one-flight fit of declared camera poses against the on-board estimate
    gate         the validity gate of one flight's scene.xml, its mesh files and cameras.json
"""

LIBRARY_VERSION = "twinprobe/1.0"


def need(cfg, key):
    """The value of a configuration key; a missing key is the task's error, never a default."""
    try:
        return cfg[key]
    except KeyError:
        raise KeyError("the configuration has no entry %r" % key) from None
