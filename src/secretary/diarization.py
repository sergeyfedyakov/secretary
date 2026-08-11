"""Диаризация: разделение говорящих по аудио (pyannote/speaker-diarization-community-1).

Актуальный стек: pyannote.audio 4.x + пайплайн community-1 (CC-BY-4.0, gated —
нужно принять условия на huggingface.co). Пайплайн 3.1 — legacy, совместим только
с pyannote.audio 3.4.x, поэтому не используется.

Аудио декодируется через PyAV в моно 16 кГц и передаётся пайплайну in-memory
(dict {'waveform', 'sample_rate'}) — это не зависит от torchcodec/ffmpeg-shared.
Модели скачиваются при первом вызове и кэшируются в ~/.cache/secretary/pyannote
(env PYANNOTE_CACHE) — сохраняются между сеансами.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PIPELINE_REPO = "pyannote/speaker-diarization-community-1"

# pyannote.audio 4.x импортирует torchcodec и предупреждает, что декодер не готов;
# мы всегда передаём аудио in-memory, поэтому предупреждения неактуальны.
warnings.filterwarnings("ignore", module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", module="pyannote.audio.models.blocks.pooling")


def _pyannote_cache_dir() -> str:
    """Единый кэш моделей pyannote (сохраняется между сеансами)."""
    return os.environ.setdefault(
        "PYANNOTE_CACHE",
        str(Path.home() / ".cache" / "secretary" / "pyannote"),
    )


@dataclass(frozen=True)
class SpeakerTurn:
    """Интервал, в котором активен говорящий."""

    start: float
    end: float
    speaker: str


class Diarizer:
    """Контракт диаризатора: принимает аудио, возвращает интервалы говорящих."""

    def warmup(self) -> None:
        """Предзагрузить модель (скачать при необходимости). По умолчанию — no-op."""

    def diarize(self, audio_path: str | Path) -> list[SpeakerTurn]:
        raise NotImplementedError


class PyannoteDiarizer(Diarizer):
    """Бэкенд pyannote/speaker-diarization-community-1.

    Пайплайн и модель скачиваются один раз (лениво, при первом diarize)
    и переиспользуются на весь батч.
    """

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("HF_TOKEN")
        self._pipeline: Any = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        if not self.token:
            raise RuntimeError(
                "Для диаризации нужен HF_TOKEN (см. .env): модели pyannote gated, "
                "плюс требуется вручную принять условия на huggingface.co."
            )
        cache_dir = _pyannote_cache_dir()
        self._ensure_model(cache_dir)
        from pyannote.audio import Pipeline

        self._pipeline = Pipeline.from_pretrained(
            PIPELINE_REPO, token=self.token, cache_dir=cache_dir
        )

    def _ensure_model(self, cache_dir: str) -> None:
        """Предзагружает пайплайн (все компоненты внутри репозитория) с прогресс-баром."""
        if _repo_cached(cache_dir, PIPELINE_REPO):
            return
        from huggingface_hub import snapshot_download

        from .engine import ModelDownloadBar

        ModelDownloadBar.label = f"Загрузка {PIPELINE_REPO}"
        try:
            snapshot_download(
                repo_id=PIPELINE_REPO,
                token=self.token or None,
                cache_dir=cache_dir,
                tqdm_class=ModelDownloadBar,
            )
        finally:
            ModelDownloadBar.label = ""

    def warmup(self) -> None:
        """Предзагружает модель диаризации (скачивает при необходимости).

        Вызов до начала обработки файлов — чтобы ошибка токена или загрузки
        всплыла сразу, а не после долгой транскрибации первого файла.
        """
        self._load()

    def diarize(self, audio_path: str | Path) -> list[SpeakerTurn]:
        self._load()
        waveform = _load_waveform(audio_path)
        output = self._pipeline(waveform)
        # exclusive_speaker_diarization — без перекрывающихся реплик,
        # проще и точнее склеивается с таймкодами whisper.
        annotation = output.exclusive_speaker_diarization
        return [
            SpeakerTurn(turn.start, turn.end, speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]


def create_diarizer(backend: str = "pyannote", **kwargs) -> Diarizer:
    if backend == "pyannote":
        return PyannoteDiarizer(**kwargs)
    raise ValueError(f"Неизвестный бэкенд диаризации: {backend}")


def _repo_cached(cache_dir: str, repo_id: str) -> bool:
    """True, если в кэше уже есть снапшот репозитория с config.yaml."""
    repo = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not repo.is_dir():
        return False
    return any((snap / "config.yaml").is_file() for snap in repo.iterdir())


def _load_waveform(audio_path: str | Path) -> dict:
    """Декодирует аудио через PyAV в моно 16 кГц float32 и отдаёт in-memory dict."""
    import av
    import numpy as np
    import torch

    with av.open(str(audio_path)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        samples = []
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                samples.append(out.to_ndarray())
    if not samples:
        raise ValueError(f"Не удалось декодировать аудио: {audio_path}")
    audio = np.concatenate(samples, axis=1)[0]
    return {"waveform": torch.from_numpy(audio.copy()).unsqueeze(0), "sample_rate": 16000}
