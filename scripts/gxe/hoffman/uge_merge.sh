#!/bin/bash
set -euo pipefail
umask 077
: "${GXE_FROZEN_PYTHON:?GXE_FROZEN_PYTHON must name the sealed interpreter}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${GXE_FROZEN_PYTHON}" -I "${SCRIPT_DIR}/hoffman_deploy.py" run --task merge "$@"
