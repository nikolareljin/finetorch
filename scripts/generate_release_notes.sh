#!/usr/bin/env bash
# SCRIPT: generate_release_notes.sh
# DESCRIPTION: Extract the matching release section from CHANGELOG.md for a given tag/version.
# USAGE: ./scripts/generate_release_notes.sh <tag-or-version> [--output PATH]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHANGELOG_PATH="$ROOT_DIR/CHANGELOG.md"
OUTPUT_PATH=""

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <tag-or-version> [--output PATH]" >&2
  exit 2
fi

RAW_VERSION="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$CHANGELOG_PATH" ]]; then
  echo "CHANGELOG.md not found." >&2
  exit 1
fi

VERSION="${RAW_VERSION#v}"

release_notes="$(
  python3 - "$CHANGELOG_PATH" "$VERSION" <<'PY'
import sys
from pathlib import Path

changelog_path = Path(sys.argv[1])
version = sys.argv[2]
lines = changelog_path.read_text(encoding="utf-8").splitlines()

target_prefix = f"## [{version}]"
capturing = False
captured = []

for line in lines:
    if line.startswith("## ["):
        if capturing:
            break
        if line.startswith(target_prefix):
            capturing = True
            continue
    if capturing:
        captured.append(line)

body = "\n".join(captured).strip()
if not capturing or not body:
    raise SystemExit(1)

print(body)
PY
)"

if [[ -z "$release_notes" ]]; then
  echo "No release notes found for version $VERSION in CHANGELOG.md." >&2
  exit 1
fi

if [[ -n "$OUTPUT_PATH" ]]; then
  printf '%s\n' "$release_notes" >"$OUTPUT_PATH"
else
  printf '%s\n' "$release_notes"
fi
