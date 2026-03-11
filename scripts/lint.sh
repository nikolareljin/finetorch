#!/usr/bin/env bash
# SCRIPT: lint.sh
# DESCRIPTION: Run formatting and lint checks for finetorch.
# USAGE: ./scripts/lint.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp}"
export CARGO_INCREMENTAL=0
mkdir -p "$TMPDIR"

cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
