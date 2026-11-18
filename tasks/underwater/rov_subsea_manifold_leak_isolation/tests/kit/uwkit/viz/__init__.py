"""uwkit.viz - render-only animation layer for the reference vehicle (visual twin, rotor animation, wake, replay).

    from uwkit import viz
    rec = viz.Recorder(sim)                 # read-only logger; call rec.step() after every sim.step()
    log = rec.finish()
    twin = viz.VisualTwin(vb)                           # render-only copy of the physics spec with animated parts
    viz.render_video(log, twin, "out.mp4", camera=viz.FollowCamera(), mode="limited", wake=True)

Nothing here changes the physics model: the twin is a copy of the MjSpec, the recorder only reads the simulator.
Maintainer data: rotor_split.json (built by build_rotor_split.py from the digest's per-solid CAD cache).
"""
from .rotors import (RotorSet, Rotor, DisplaySpeed, rpm_from_command, rpm_from_thrust, rpm_reference,  # noqa: F401
                     integrate_angles, load_rotor_split)
from .log import Recorder, SimLog, joint_table  # noqa: F401
from .twin import VisualTwin  # noqa: F401
from .wake import Wake, WakeStyle  # noqa: F401,E402
from .render import (TwinRenderer, FollowCamera, FixedCamera, VideoJob, AngleSource, FrameState,  # noqa: F401,E402
                     render_jobs, render_video, render_thruster_tiles)
from .export import static_scene, part_transforms, save_animation  # noqa: F401,E402
