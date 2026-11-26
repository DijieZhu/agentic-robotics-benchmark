"""The flight-twin grader's modules (run by the worker that verify.py starts).

    config   grader_config.json: the public numbers, read without defaults
    givens   the grader's copies of the givens it reads (frame stamps, the on-board estimate, the ray table)
    truth    the truth files of a flight, checked against the configuration at load
    guard    stages under alarms; what a failure means (the flight's zero, or a void verification)
    score    the depth term, the hazard term, the key poses (rule P) and every registered aggregation
    grade    a submission through the gate (rule L's entry list of cameras.json among it), the one-flight fit, the
             frame rules and the two terms
    metrics  the metric whitelist of the verdict
    worker   the worker process: the reference first, then the submission
    twinprobe  the probe library shared with the agent's check tool (a byte copy)
"""
