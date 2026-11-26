"""The runner of the task: the wire between the grader and a submitted controller (JSON lines over standard streams),
its processor and wall accounting, its sandbox, the set-up allowance and the restart rule, the plan validity rules,
the event rule, the block clock and budget, the wall rule per stretch, the watchdog's count and the step-time report.
Every number comes from runner_config.json (written from the task's table by key); every rule the runner applies is
listed by its key in PROTOCOL.json."""
