#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
SCRIPT_NAME="main.py"
PYTHON_GUI_EXE="$VENV_DIR/bin/python3"
PYTHON_CONSOLE_EXE="$VENV_DIR/bin/python"

"$(dirname "$0")/setup_env.sh"

echo
echo "Launching $SCRIPT_NAME..."
if [ -x "$PYTHON_GUI_EXE" ]; then
    "$PYTHON_GUI_EXE" "$SCRIPT_NAME"
else
    "$PYTHON_CONSOLE_EXE" "$SCRIPT_NAME"
fi
