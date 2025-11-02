#!/usr/bin/env bash
source "$(dirname "$0")/nemotron-env/bin/activate"
echo "=== 🚀 Running Nemotron Probability Stasis v2 ==="
python3 run_nemotron.py "$@"
