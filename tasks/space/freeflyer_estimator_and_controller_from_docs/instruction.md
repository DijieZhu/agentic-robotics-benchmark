# Flight software for the cabin free-flyer: its estimator and its controller

You write the flight software of the cabin free-flyer that the acceptance flies: one program that is both the state estimator and the controller. The built-in estimator's output is not on the wire: it is the part this deliverable replaces, not a withheld document. Optical-flow and marker measurements are not on the wire either. The acceptance flies the submitted controller on hidden sessions (6 of them) after your working session ends; the score is gates x mean of the six sessions.

## What you are given (in `/home/user/`)

- `payload/`: the maker's documents of the robot and of the cabin, the unit's own files and its earlier records (`payload/unit/records/about_the_records.md` describes them), the campaign's photographs, and the acceptance's rules in `payload/acceptance/`.
- `controller/`: `interface.md` (the wire, the frame and the replies), an example controller and example messages, and the toy self-test `selftest/`.
- Paths on the task's pages (payload/..., controller/...) are relative to the folder that holds payload and controller; in your home that folder is `/home/user/`.

## What you deliver

The folder `/home/user/submission/controller/` holding `controller.py`; its class (interface.class): Controller with setup(session), plan(job, frame), step(frame). It may import these libraries (interface.libraries): python standard library, numpy, scipy; and its own modules (the static import audit of score.gates). At most 200 regular files and 16 megabytes (interface.submission_cap; caps on the submission folder (regular files; symbolic links, special files and __pycache__ are not copied)); an over-cap submission scores 0 in every session. The integrity rule (interface.submission_integrity): the submission folder is final when the working session ends: if anything in it is written, renamed, linked or has its permissions changed after that, if the folder itself is replaced or changed after that, or if it is a symbolic link, is not a folder or leads outside the home folder, the reward is 0 and no session is flown.

## The budgets and the wall caps

- A step every 16.0 ms (interface.step_period_ms); 13.5 ms of processor time per step (interface.step_cpu_ms); at least a share of 0.9 of the steps answered inside that budget over the run (interface.on_time_share_gate).
- Set-up: 60.0 s of processor time and 180.0 s of wall clock per session (interface.setup_cpu_s, interface.setup_wall_s); a plan call: 5.0 s of processor time and 20.0 s of wall clock per job (interface.plan_cpu_s, interface.plan_wall_s).
- The wall rule (interface.block_wall_rule): at every step of a stretch (each block, from its first step to the step at which it ends; the start hold with the check-out; each gap between blocks; the time after the last block), the wall-clock time of the stretch's step replies so far may not exceed the stretch's world time so far plus 5 s; a call that would exceed it is cut there: in a block it is a protocol error and the block ends, outside a block the controller's process is ended and started again (interface.restart) and no job is lost.
- The controller's sandbox (interface.sandbox): 4 processors, 8 GiB of address space per process, 1 thread per numerics library, network: none (no socket of any family).
- The verification's limit (interface.verify_timeout_s): 18000 s; if the verification reaches this limit it is void: a task error of the run, not a score (no reward is written).

## No practice flight, no simulator

Nothing flies your controller on the robot, or on a model of it, before the acceptance: there is no practice flight and no simulator of the robot in your sandbox. The self-test in `controller/selftest/` is a toy rigid body with toy sensors in the same message formats and conventions, with arbitrary noise and no calibrated value of this robot; for checking protocol and signs only (interface.self_test).

## Where the rules are

`payload/acceptance/acceptance_specification.md`: what the acceptance does with the submitted controller, every rule that ends a job, a block, a session or the run, the block budget's formula, the plan rules, every printed range of the sensors and the unit, and the score. `controller/interface.md`: how the submitted controller is run and what passes between it and the acceptance's runner. `payload/acceptance/session_paper_format.md`: the session paper that setup receives, its format and two filled examples.

## Session

Your working session: 4 processors, 16 GB of memory, no graphics processor, no network, 6 hours. Installed in the agent's sandbox: Python 3 with numpy and scipy, the libraries a controller may import (interface.libraries: python standard library, numpy, scipy). When the time limit ends the working session, the submission folder as it stands at that moment is what the acceptance flies.
