#!/usr/bin/env bash

# Simple C++ compilation helper for gentrade modules
# Usage: ./op/compile_cpp_module.sh <module_name>
# Example: ./op/compile_cpp_module.sh eval_signals_sharpe

set -euo pipefail

MODULE_NAME=${1:-}

if [ -z "$MODULE_NAME" ]; then
    echo "Usage: $0 <module_name>"
    exit 1
fi

# Resolve paths (script is located in op/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
TARGET_DIR="$PROJECT_ROOT/src/gentrade"

# Use venv python explicitly for deterministic behavior
PY="$PROJECT_ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "Error: Python executable not found at $PY"
    echo "Create or point to a virtualenv at $PROJECT_ROOT/.venv or adjust the script."
    exit 1
fi

# Preflight checks
if ! command -v g++ >/dev/null 2>&1; then
    echo "Error: g++ not found in PATH. Install a C++ compiler."
    exit 1
fi

if ! "$PY" -c "import pybind11" >/dev/null 2>&1; then
    echo "Error: pybind11 is not installed in the virtualenv ($PY)."
    echo "Install it with: $PY -m pip install pybind11"
    exit 1
fi

# Check presence of Python C headers (Python.h)
INCLUDE_DIR=$($PY -c 'import sysconfig, json; print(sysconfig.get_paths()["include"])')
if [ ! -f "$INCLUDE_DIR/Python.h" ]; then
    echo "Error: Python headers not found in $INCLUDE_DIR (Python.h missing)."
    echo "Install the Python development headers for this interpreter."
    exit 1
fi

INCLUDES=$($PY -m pybind11 --includes)
EXT_SUFFIX=$($PY -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

SRC_FILE="$TARGET_DIR/$MODULE_NAME.cpp"
OUT_FILE="$TARGET_DIR/$MODULE_NAME$EXT_SUFFIX"

if [ ! -f "$SRC_FILE" ]; then
    echo "Error: source file not found: $SRC_FILE"
    exit 1
fi

echo "Compiling $SRC_FILE -> $OUT_FILE"

g++ -O3 -Wall -shared -std=c++11 -fPIC \
    $INCLUDES \
    "$SRC_FILE" \
    -o "$OUT_FILE"

echo "Successfully compiled $MODULE_NAME -> $OUT_FILE"

# Final import check (use venv python directly)
echo "Testing Python import of gentrade.$MODULE_NAME using $PY"
if $PY -c "import gentrade.$MODULE_NAME; print('import OK')"; then
    echo "Import test passed: gentrade.$MODULE_NAME"
else
    echo "Error: Python import failed for gentrade.$MODULE_NAME"
    echo "Run the following to see import error details:"
    echo "  $PY -c \"import gentrade.$MODULE_NAME\""
    exit 1
fi
