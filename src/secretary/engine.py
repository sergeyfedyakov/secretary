"""Обёртка над faster-whisper: загрузка модели с прогрессом и транскрибация."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# На Windows без Developer Mode os.symlink кидает WinError 1314. huggingface_hub
# кэширует модели симлинками по умолчанию — выставляем флаг до импорта hub,
# чтобы файлы копировались, а не линковались (иначе падение при загрузке pyannote).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import ctranslate2
from huggingface_hub import snapshot_download
from tqdm import tqdm

from .model_registry import DEFAULT_MODEL, resolve_model_repo


class ModelDownloadBar(tqdm):
    """Прогресс-бар загрузки модели: полоса, скорость (MB/s), ETA."""

    label: str = ""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("unit", "B")
        kwargs.setdefault("unit_scale", True)
        kwargs.setdefault("unit_divisor", 1024)
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("mininterval", 0.2)
        if self.__class__.label:
            desc = kwargs.get("desc") or ""
            kwargs["desc"] = f"{self.__class__.label}: {desc}".strip(": ")
        super().__init__(*args, **kwargs)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    return device


def _resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type == "auto":
        return "int8" if device == "cpu" else "float16"
    return compute_type


def _default_cache_dir() -> Path:
    override = os.environ.get("SECRETARY_MODEL_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "secretary" / "models"


def _model_dir(cache_root: str | None, repo_id: str) -> Path:
    root = Path(cache_root) if cache_root else _default_cache_dir()
    return root / repo_id.replace("/", "--")


def _split_model_ref(model_ref: str) -> tuple[str, str | None]:
    """Разделяет ссылку на модель на (repo_id, subfolder).

    Поддержка ссылок вида 'org/name/subfolder' — для pre-quantized моделей
    (например coriollon/whisper-large-v3-turbo-russian/ct2_int8_float16),
    где model.bin лежит в подпапке репозитория.
    """
    parts = model_ref.split("/")
    if len(parts) == 3:
        return "/".join(parts[:2]), parts[2]
    return model_ref, None


def _ensure_model(model_ref: str, cache_root: str | None) -> str:
    """Возвращает локальный путь к модели, скачивая её при необходимости."""
    repo, subfolder = _split_model_ref(model_ref)
    target = _model_dir(cache_root, repo)

    if subfolder is not None:
        model_path = target / subfolder
        if (model_path / "model.bin").is_file():
            return str(model_path)
        target.mkdir(parents=True, exist_ok=True)
        ModelDownloadBar.label = f"Загрузка модели {model_ref}"
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=str(target),
                allow_patterns=f"{subfolder}/*",
                tqdm_class=ModelDownloadBar,
                token=os.environ.get("HF_TOKEN") or None,
            )
        finally:
            ModelDownloadBar.label = ""
        return str(model_path)

    if (target / "model.bin").is_file():
        return str(target)

    resolved = resolve_model_repo(model_ref) or model_ref
    target.mkdir(parents=True, exist_ok=True)
    ModelDownloadBar.label = f"Загрузка модели {model_ref}"
    try:
        snapshot_download(
            repo_id=resolved,
            local_dir=str(target),
            tqdm_class=ModelDownloadBar,
            token=os.environ.get("HF_TOKEN") or None,
        )
    finally:
        ModelDownloadBar.label = ""
    return str(target)


class TranscriptionEngine:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        compute_type: str = "auto",
        cache_dir: str | None = None,
        language: str | None = None,
        vad_filter: bool = True,
        verbose: bool = False,
    ):
        self.model_ref = model
        self.device = _resolve_device(device)
        self.compute_type = _resolve_compute_type(compute_type, self.device)
        self.cache_dir = cache_dir
        self.language = None if language in (None, "auto", "") else language
        self.vad_filter = vad_filter
        self.verbose = verbose
        self._model: Any = None

    def _load_model(self) -> None:
        from faster_whisper import WhisperModel

        local = Path(self.model_ref)
        if local.is_dir():
            path = str(local)
        else:
            path = _ensure_model(self.model_ref, self.cache_dir)
        if self.verbose:
            print(f"[модель] {self.model_ref} | device={self.device} compute={self.compute_type}\n  {path}")
        self._model = WhisperModel(path, device=self.device, compute_type=self.compute_type)

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
    ) -> dict:
        """Возвращает {language, language_probability, segments, words}.

        segments — [(start, end, text)]; words — [(start, end, word)] из
        словесных таймкодов (только при word_timestamps=True).
        """
        if self._model is None:
            self._load_model()
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=self.language,
            vad_filter=self.vad_filter,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
        )
        segments: list[tuple[float, float, str]] = []
        words: list[tuple[float, float, str]] = []
        for segment in segments_iter:
            segments.append((segment.start, segment.end, segment.text))
            if word_timestamps and segment.words:
                words.extend((word.start, word.end, word.word) for word in segment.words)
        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": segments,
            "words": words,
        }
