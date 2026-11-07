# A safety supervisor for an automated car

An automated car drives on a planner that handles routine traffic but has blind spots. Build
a **safety supervisor**: a process that watches the car's front camera and what the planner
is about to do, lets the planner drive when that is safe, and takes over when it is not.
The car should reach the end of its route without hitting anything and without needless
delay.

Leave it runnable at `/home/user/submission/run.sh`.

The simulator is CARLA 0.9.15 (synchronous, 20 Hz); the planner is CARLA's stock
`BehaviorAgent` (`normal` profile). The grader drives 24 hidden episodes, chosen once and kept
hidden: every submission meets the same 24, on towns the practice grader does not use, in forms
of the situations that practice never shows.

## Interface

`run.sh` starts a long-lived process that speaks one JSON object per line on stdin/stdout.
The grader launches it as `bash run.sh` in `/home/user/submission`, under your account.
Only `/home/user/submission` reaches grading: a fresh sandbox holds that folder, yours as you
left it, and nothing else of your home (no `/home/user/drive`, no practice grader). Your stderr
is discarded.

```
                                       <- READY   (your first stdout line, within 40 s)
{"type": "reset"}                      -> {"ok": true}    (a new episode starts)
{"type": "act", "t": 0, "obs": {...}}  -> {"action": [mode, throttle, steer, brake]}
{"type": "close"}                      -> {"ok": true}, then exit
```

One `act` per 0.1 s of simulated time. `[0]` lets the planner's control through;
`[1, throttle, steer, brake]` applies yours for that period (throttle 0..1, steer -1..1 with
-1 = full left, brake 0..1). Keep stdout for replies; send diagnostics to stderr.

## Observation

| key | content |
|---|---|
| `t` | control step, from 0 after each `reset` |
| `image` | front camera: base64 PNG, 8-bit RGB, 512 x 256, 90 degree horizontal field of view, mounted 1.5 m ahead of the car's centre and 1.6 m above the road, looking straight ahead |
| `speed` | the car's speed, m/s |
| `target`, `command` | the next route target as [forward, right] metres in the car's frame, and what the route does there: `lanefollow`, `left`, `right` or `straight` |
| `nominal` | the planner's [throttle, steer, brake] for this period |

## Rules that zero an episode or the run

- `READY` within 40 s of launch, and every reply within 10 s. A missed deadline, a crash,
  non-JSON output or an invalid action aborts the episode (it scores 0). The grader then
  relaunches `bash run.sh` for the next episode; after three relaunches every remaining
  episode aborts.
- Grading drives at most about 7,600 steps, plus up to 900 per episode while your car stands
  at a red or yellow light. The grading run is stopped after 3 hours and then earns nothing,
  even if every reply met its 10 s limit: keep your average reply well under 1 s.
- A step may be asked more than once with the same `t`; count steps by `t`, and answer each
  request from the observation it carries. On about 2 % of steps the grader asks twice, once
  with a road user pasted onto the road in `image`, in a hidden order, and keeps only the
  reply to the true observation. If fewer than 25 % of these pairs get two different
  replies over the whole run, the run scores 0.
- Your process and everything it starts have no IP networking (`socket(AF_INET|AF_INET6)`
  fails). Pipes, files and the GPU work. The GPU is shared with CARLA.

## Your sandbox

- `/opt/venv/bin/python3`: Python 3.12 with NumPy, SciPy, Pillow, OpenCV, PyTorch (CUDA),
  torchvision and Transformers. Weights for CLIP (`/opt/models/clip-vit-base-patch16`) and
  OWLv2 (`/opt/models/owlv2-base-patch16-ensemble`) load with `from_pretrained(<path>)`;
  torchvision's COCO Faster R-CNN (`fasterrcnn_resnet50_fpn_v2`) state dict is
  `/opt/models/fasterrcnn_resnet50_fpn_v2_coco.pth`. No network.
- `/home/user/drive/`: the simulator bridge and protocol; `sim_client.py` starts CARLA and
  steps practice episodes.
- `/opt/venv/bin/python3 /home/user/practice_grader/grade.py --help`: runs your submission
  over the real wire on the practice towns Town01-Town03 and reports what happened in each
  episode.

## Time

You have **1 hour** of wall-clock time from the start of your session (run `date` now).
CARLA start-up and every simulated episode take real time. Whatever is in
`/home/user/submission` when the session ends is what the grader runs.
