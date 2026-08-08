"""Интерфейс диаризации. MVP: только контракт, реализация — следующей итерацией.

Бэкенд по плану (.ai/plan.md): pyannote/speaker-diarization-3.1, склейка
результатов с сегментами STT по пересечению таймкодов.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpeakerTurn:
    """Интервал, в котором активен говорящий."""

    start: float
    end: float
    speaker: str


class Diarizer:
    """Контракт диаризатора: принимает аудио, возвращает интервалы говорящих."""

    def diarize(self, audio_path: str | Path) -> list[SpeakerTurn]:
        raise NotImplementedError


def create_diarizer(backend: str = "pyannote", **kwargs) -> Diarizer:
    raise NotImplementedError(
        "Диаризация ещё не реализована. Запланирован бэкенд "
        "pyannote/speaker-diarization-3.1 (см. .ai/plan.md)."
    )
