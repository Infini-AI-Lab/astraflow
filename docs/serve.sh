#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PORT=${1:-8000}
ROOT="$SCRIPT_DIR/build/en"

if [ ! -d "$ROOT" ]; then
  echo "Directory not found: $ROOT"
  echo "Run docs build first: bash docs/build.sh"
  exit 1
fi

echo "Serving $ROOT at http://127.0.0.1:$PORT"
cd "$ROOT"
python -m http.server "$PORT"
