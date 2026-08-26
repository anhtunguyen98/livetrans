#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")"
"$SCRIPT_DIR/start-ai-services.sh"
"$SCRIPT_DIR/start-ui.sh"
"$SCRIPT_DIR/status.sh"
