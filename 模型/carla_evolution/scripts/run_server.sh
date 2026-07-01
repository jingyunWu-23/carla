#!/usr/bin/env bash
set -euo pipefail

CARLA_ROOT="${CARLA_ROOT:-/path/to/CARLA_0.9.15}"
exec "${CARLA_ROOT}/CarlaUE4.sh" -quality-level=Low

