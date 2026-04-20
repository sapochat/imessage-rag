#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .

echo
echo "Setup complete."
echo "Activate the environment with: source .venv/bin/activate"
echo "Then check the app with: imessage-rag doctor"
