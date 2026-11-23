"""assembly_info: the information-tier assembly stage (lane W1-ASM; task `arm_assemble_six_servo_from_manual_panda`).

PUBLIC (shipped with the practice cell; no instance numbers):
    cell_info.py     InfoCellServer: the six-servo cell with the restricted camera-only API (describe / get_state without
                     part poses or mates, events limited to safety, noisy wrench, live RGB-D cameras through shared/render,
                     a frame budget, privileged calls only with an oracle token or in the practice cell)
    cell_render.py   the render twin of the cell (workbench, datum boards, lights) and the per-episode render session
    layout_info.py   per-episode draws (layout, proximal posture, horn side, spare slot, cameras, lights, sensor noise)
    local.py         PracticeCell (in-process practice cell over a practice kit; the wall-clock cap and the silence rule
                     enforced in-process)
    plan.py          the assembly-plan schema check and the plan grader (arm-assembly-plan/1, symmetries, the horn zero
                     bands relative to the practice kits' no-photograph estimates)
GRADER / AUTHOR ONLY (import arm.gt_arm; never shipped):
    instances.py     the lane's rows in the tier registry (the instance draws, the feasibility predicate)
    variant.py       a GT arm -> cell design, A2 layers, print-frame kit, GT plan, symmetries
    manual.py        the picture manual (25 photos through render.photo_set.build_photo_set)
    rigs.py          the practice cells' camera rigs (per practice seed, away from the hidden rig) and the hidden rig's draw
    handoff.py       the A2-info adapter (hardware-design -> the assembly stage's kit and grader GT) and the e2e mode's
                     GT from the agent's own CAD (README.md, "The hand-off in")
"""

CELL_VERSION = "arm-assembly-cell-info/1.0.0"
