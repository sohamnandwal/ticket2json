#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${UV:-$(command -v uv || true)}"
VENV="${TICKET2JSON_VENV:-$HOME/.venvs/ticket2json}"

if [[ -z "$UV" && -x "$HOME/.local/bin/uv" ]]; then
  UV="$HOME/.local/bin/uv"
fi
if [[ -z "$UV" ]]; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/." >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  "$UV" venv "$VENV" --python 3.13
fi
"$UV" pip install \
  --python "$VENV/bin/python" \
  --torch-backend=auto \
  -r "$ROOT/requirements-training.txt"
echo "Training environment ready. Activate with: source \"$VENV/bin/activate\""
