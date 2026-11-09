#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${ALE_HOME:-/home/user}/submission"
cp submission/* "${ALE_HOME:-/home/user}/submission/"
