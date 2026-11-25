"""The cell's library (grader side). voidcell/__init__.py puts this folder on sys.path, so the vendored packages
`ysim` and `cellcore` import by their own names; the cell's modules are imported as voidcell._lib.<name>:

    protocol     the calls, their arguments, the reply-key whitelist and the error enumeration (given/station_api.md)
    sitepkg      site packages: a folder or a .tar.xz archive with site.json + site.xml, and their sha256
    camera       the search camera: 7 Hz exposures, renderer-agnostic backends, the depth sensor model
    telemetry    100 Hz telemetry with the rated noise (given/station_api.md); frames.md states the frames
    budgets      frames, reports, the simulated-time limit
    mission      mission(): keys only from the whitelist
    pose_source  the pose switch (grader_config.json `pose_source`): "estimated" (default) or "cell"
"""
