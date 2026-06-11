#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
REQUIREMENTS_INPUT_FILE="requirements.in"
REQUIREMENTS_LOCK_FILE="requirements.lock.txt"
RUNTIME_LOCK_FILE="$VENV_DIR/.requirements.runtime.lock.txt"
DEPS_STAMP="$VENV_DIR/.deps_installed"
TORCH_STAMP="$VENV_DIR/.torch_checked"
LOG_DIR="logs"
REQUIRED_PYTHON_MINOR="3.11"
CREATED_VENV=0
UV_APP_DRY="${UV_APP_DRY:-0}"

mkdir -p "$LOG_DIR"
LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/installer_${LOG_TIMESTAMP}.log"
export CHATTERBOX_INSTALL_LOG="$(pwd)/$LOG_FILE"

log_message() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" | tee -a "$LOG_FILE"
}

pick_python() {
    if command -v python3.11 >/dev/null 2>&1; then
        printf 'python3.11'
        return
    fi

    if command -v python3 >/dev/null 2>&1; then
        local version
        version="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
        if [ "$version" = "$REQUIRED_PYTHON_MINOR" ]; then
            printf 'python3'
            return
        fi
    fi

    return 1
}

is_newer_than() {
    [ ! -e "$2" ] && return 0
    [ "$1" -nt "$2" ]
}

build_runtime_lock() {
    log_message "Building runtime lock without torch, torchaudio, and torchvision pins."
    grep -Ev '^(torch|torchaudio|torchvision)==' "$REQUIREMENTS_LOCK_FILE" > "$RUNTIME_LOCK_FILE"
}

log_message "=== Starting Chatterbox TTS Installer ==="
log_message "Dry Run Mode: $UV_APP_DRY"
echo
echo "=== Starting Chatterbox TTS Installer ==="
echo
echo "Installer log: $LOG_FILE"

PYTHON_BIN="$(pick_python || true)"
if [ -z "${PYTHON_BIN:-}" ]; then
    log_message "ERROR: Python 3.11 not found."
    echo "ERROR: Python 3.11 not found."
    echo "Please install Python 3.11 and re-run this script."
    exit 1
fi
log_message "Using Python interpreter: $PYTHON_BIN"

if ! command -v uv >/dev/null 2>&1; then
    log_message "ERROR: uv is missing."
    echo "ERROR: uv is not installed."
    echo "Install it from: https://github.com/astral-sh/uv#getting-started/installation"
    exit 1
fi
log_message "uv found on PATH."

if [ ! -d "$VENV_DIR" ]; then
    log_message "Creating venv at $VENV_DIR."
    uv venv "$VENV_DIR" --python "$PYTHON_BIN" >> "$LOG_FILE" 2>&1
    CREATED_VENV=1
fi

PYTHON_CONSOLE_EXE="$VENV_DIR/bin/python"
if [ ! -x "$PYTHON_CONSOLE_EXE" ]; then
    log_message "ERROR: venv python executable missing."
    echo "ERROR: venv python executable missing."
    exit 1
fi

VENV_PYTHON_VERSION="$("$PYTHON_CONSOLE_EXE" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
log_message "Detected venv Python version $VENV_PYTHON_VERSION."
if [ "$VENV_PYTHON_VERSION" != "$REQUIRED_PYTHON_MINOR" ]; then
    log_message "Rebuilding venv because Python version does not match requirement."
    rm -rf "$VENV_DIR"
    uv venv "$VENV_DIR" --python "$PYTHON_BIN" >> "$LOG_FILE" 2>&1
    CREATED_VENV=1
    PYTHON_CONSOLE_EXE="$VENV_DIR/bin/python"
fi

if [ "$CREATED_VENV" -eq 1 ]; then
    rm -f "$DEPS_STAMP" "$TORCH_STAMP"
fi

NEED_LOCK_COMPILE=0
[ ! -f "$REQUIREMENTS_LOCK_FILE" ] && NEED_LOCK_COMPILE=1
if [ -f "$REQUIREMENTS_INPUT_FILE" ] && is_newer_than "$REQUIREMENTS_INPUT_FILE" "$REQUIREMENTS_LOCK_FILE"; then
    NEED_LOCK_COMPILE=1
fi

if [ "$NEED_LOCK_COMPILE" -eq 1 ]; then
    log_message "Running uv pip compile because the lock file is missing or stale."
    uv pip compile "$REQUIREMENTS_INPUT_FILE" -o "$REQUIREMENTS_LOCK_FILE" >> "$LOG_FILE" 2>&1
else
    log_message "Lock file already up to date."
fi

build_runtime_lock

NEED_DEPS_SYNC=0
[ ! -f "$DEPS_STAMP" ] && NEED_DEPS_SYNC=1
if is_newer_than "$REQUIREMENTS_LOCK_FILE" "$DEPS_STAMP"; then
    NEED_DEPS_SYNC=1
fi
[ "$CREATED_VENV" -eq 1 ] && NEED_DEPS_SYNC=1

if [ "$NEED_DEPS_SYNC" -eq 1 ]; then
    log_message "Installing runtime dependencies from lock file."
    uv pip install --python "$PYTHON_CONSOLE_EXE" -r "$RUNTIME_LOCK_FILE" --strict >> "$LOG_FILE" 2>&1
    printf 'Dependencies synced from %s\n' "$RUNTIME_LOCK_FILE" > "$DEPS_STAMP"
else
    log_message "Dependencies already installed."
fi

if [ "$UV_APP_DRY" = "0" ]; then
    NEED_TORCH_INSTALL=0
    [ ! -f "$TORCH_STAMP" ] && NEED_TORCH_INSTALL=1
    if ! "$PYTHON_CONSOLE_EXE" -c "import torch, torchaudio, torchvision" >/dev/null 2>&1; then
        log_message "Torch runtime import check failed. Forcing reinstall."
        rm -f "$TORCH_STAMP"
        NEED_TORCH_INSTALL=1
    fi
    if [ -f install_torch.py ] && is_newer_than install_torch.py "$TORCH_STAMP"; then
        NEED_TORCH_INSTALL=1
    fi
    if is_newer_than "$REQUIREMENTS_LOCK_FILE" "$TORCH_STAMP"; then
        NEED_TORCH_INSTALL=1
    fi

    if [ "$NEED_TORCH_INSTALL" -eq 1 ]; then
        log_message "Running install_torch.py."
        "$PYTHON_CONSOLE_EXE" install_torch.py >> "$LOG_FILE" 2>&1
        printf 'PyTorch checked by install_torch.py\n' > "$TORCH_STAMP"
    else
        log_message "PyTorch already checked."
    fi
else
    log_message "[Dry Run] Skipped PyTorch installation."
fi

log_message "Setup complete."
echo "Setup complete."
