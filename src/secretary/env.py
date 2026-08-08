"""Загрузка переменных окружения из .env (без внешних зависимостей)."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | os.PathLike = ".env") -> None:
    """Читает строки KEY=VALUE из файла и заполняет os.environ.

    Уже заданные переменные окружения не перезаписываются.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value
