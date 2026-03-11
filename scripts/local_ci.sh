#!/usr/bin/env bash
# SCRIPT: local_ci.sh
# DESCRIPTION: Run local CI sequence: submodules, build, lint, and tests.
# USAGE: ./scripts/local_ci.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "$ROOT_DIR/scripts/update_submodules.sh"
bash "$ROOT_DIR/scripts/build.sh"
bash "$ROOT_DIR/scripts/lint.sh"
bash "$ROOT_DIR/scripts/test.sh"

echo "Local CI complete."
