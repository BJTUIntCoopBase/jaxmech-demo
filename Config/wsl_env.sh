#!/bin/bash
# Load repository machine config without hardcoded machine details.

if [ -n "$HOME" ] && [ ! -d "$HOME" ]; then
    _user_home="$(getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6)"
    if [ -n "$_user_home" ] && [ -d "$_user_home" ]; then
        export HOME="$_user_home"
    fi
fi

_detect_wsl_python() {
    local candidate
    local configured_python="$1"

    if [ -n "$configured_python" ] && [ -x "$configured_python" ]; then
        echo "$configured_python"
        return
    fi

    if [ -n "$WSL_PYTHON" ] && [ -x "$WSL_PYTHON" ]; then
        echo "$WSL_PYTHON"
        return
    fi

    candidate=$(command -v python3 2>/dev/null)
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        echo "$candidate"
        return
    fi

    candidate=$(command -v python 2>/dev/null)
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        echo "$candidate"
        return
    fi

    echo "python3"
}

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$_script_dir/.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/Config"
ENV_TEMPLATE="$CONFIG_DIR/env.template.cfg"
ENV_CFG="$CONFIG_DIR/env.cfg"
PROJECT_PYTHONPATH="$PROJECT_ROOT/external:$PROJECT_ROOT"

if [ ! -f "$ENV_CFG" ] && [ -f "$ENV_TEMPLATE" ]; then
    mkdir -p "$CONFIG_DIR"
    cp "$ENV_TEMPLATE" "$ENV_CFG"
fi

_get_cfg_value() {
    local key="$1"
    local file="$2"
    if [ ! -f "$file" ]; then
        return
    fi
    awk -F= -v key="$key" '
        {
            line=$0
            sub(/[[:space:]]*#.*/, "", line)
            if (line ~ /^[[:space:]]*$/) next
            split(line, a, "=")
            k=a[1]
            sub(/^[[:space:]]+/, "", k)
            sub(/[[:space:]]+$/, "", k)
            if (k != key) next
            val=substr(line, index(line, "=")+1)
            sub(/^[[:space:]]+/, "", val)
            sub(/[[:space:]]+$/, "", val)
            print val
            exit
        }
    ' "$file"
}

CFG_WSL_PYTHON="$(_get_cfg_value wsl_python "$ENV_CFG")"
WINDOWS_PYTHON_EXE="$(_get_cfg_value windows_python_exe "$ENV_CFG")"
WIN_STORED_ROOT="$(_get_cfg_value win_stored_root "$ENV_CFG")"
WSL_STORED_ROOT="$(_get_cfg_value wsl_stored_root "$ENV_CFG")"

export PROJECT_ROOT
export PROJECT_PYTHONPATH
export JAX_ENABLE_X64=1
export WINDOWS_PYTHON_EXE
export WIN_STORED_ROOT
export WSL_STORED_ROOT

WSL_PYTHON=$(_detect_wsl_python "$CFG_WSL_PYTHON")
WSL_USER=$(whoami)
export WSL_PYTHON
export WSL_USER

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "$WSL_PYTHON"
fi
