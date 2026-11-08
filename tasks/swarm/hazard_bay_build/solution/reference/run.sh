#!/bin/bash
# The controller: the grader launches this as `bash run.sh` in the submission folder and speaks JSON lines to it.
exec /opt/venv/bin/python3 -I controller.py
