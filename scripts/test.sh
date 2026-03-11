#!/usr/bin/env bash
# SCRIPT: test.sh
# DESCRIPTION: Run the finetorch Rust test suite.
# USAGE: ./scripts/test.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp}"
export CARGO_INCREMENTAL=0
mkdir -p "$TMPDIR"

cargo test
