#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "$SCRIPT_DIR"
export LC_ALL="${LC_ALL:-C.UTF-8}"
PROJECT_DOC_LANG=en sphinx-build -b html -D language=en --conf-dir ./ "./en" "./build/en"

echo "Built docs to: $SCRIPT_DIR/build/en"
