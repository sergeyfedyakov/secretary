"""Загрузка переменных окружения из .env (без внешних зависимостей)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def load_env_file(path: str | os.PathLike | None = None) -> None:
    """Читает строки KEY=VALUE из файла и заполняет os.environ.

    Если path не указан, ищет .env рядом с исполняемым файлом (актуально
    для PyInstaller exe) и в текущем каталоге. Уже заданные переменные
    окружения не перезаписываются.
    """
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = []
        exe_dir = Path(sys.executable).parent
        candidates.append(exe_dir / ".env")
        candidates.append(Path(".env"))
    for env_path in candidates:
        if not env_path.is_file():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value
        return
