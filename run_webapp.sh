#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Virtual environment not found. Run: uv venv -p python3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt"
  exit 1
fi
exec .venv/bin/python webapp/app.py
