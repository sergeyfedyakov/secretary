#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "ОШИБКА: Python не найден: $PYTHON"
    echo
    echo "Сначала соберите окружение:"
    echo "  cd \"$SCRIPT_DIR\""
    echo "  python3 -m venv .venv"
    echo "  .venv/bin/pip install -r requirements.txt"
    echo "  .venv/bin/pip install -e ."
    exit 1
fi

cd "$SCRIPT_DIR"
exec "$PYTHON" -m secretary "$@"
