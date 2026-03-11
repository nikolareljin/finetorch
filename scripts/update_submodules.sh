#!/usr/bin/env bash
# SCRIPT: update_submodules.sh
# DESCRIPTION: Sync and initialize git submodules recursively.
# USAGE: ./scripts/update_submodules.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: $ROOT_DIR is not a git worktree" >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/.gitmodules" ]]; then
  echo "No .gitmodules found; nothing to update."
  exit 0
fi

configured_paths=()
git_config_output="$(git -C "$ROOT_DIR" config -f .gitmodules --get-regexp '^submodule\..*\.path$' 2>&1)" || {
  git_config_status=$?
  if [[ "$git_config_status" -eq 1 ]]; then
    echo "No configured submodules found in .gitmodules."
    exit 0
  fi
  echo "error: failed to read submodule paths from .gitmodules:" >&2
  echo "$git_config_output" >&2
  exit "$git_config_status"
}
if [[ -z "$git_config_output" ]]; then
  echo "No configured submodules found in .gitmodules."
  exit 0
fi

while IFS= read -r path; do
  [[ -n "$path" ]] || continue
  configured_paths+=("$path")
done < <(printf '%s\n' "$git_config_output" | awk '{print $2}')

if [[ "${#configured_paths[@]}" -eq 0 ]]; then
  echo "No configured submodules found in .gitmodules."
  exit 0
fi

for path in "${configured_paths[@]}"; do
  git -C "$ROOT_DIR" submodule sync --recursive -- "$path"
  git -C "$ROOT_DIR" submodule update --init --recursive -- "$path"
done

echo "Submodules updated."
