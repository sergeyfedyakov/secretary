"""Опциональный бэкенд OpenVINO GenAI (Intel GPU/XPU, NPU, CPU) для транскрибации.

Использует `openvino_genai.WhisperPipeline` вместо CTranslate2. Аудио декодируется
и фильтруется по тишине теми же средствами, что и в faster-whisper
(`faster_whisper.audio` / `faster_whisper.vad`), поэтому поведение VAD и таймкоды
согласованы с базовым движком. Пакет `openvino-genai` ставится отдельно
(requirements-openvino.txt или pip install -e .[openvino]).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Алиасы в стиле faster-whisper -> готовые OV-IR репозитории на HF.
OV_ALIASES: dict[str, str] = {
    "tiny": "OpenVINO/whisper-tiny-int8-ov",
    "base": "OpenVINO/whisper-base-int8-ov",
    "small": "OpenVINO/whisper-small-int8-ov",
    "medium": "OpenVINO/whisper-medium-int8-ov",
    "large-v3": "OpenVINO/whisper-large-v3-int8-ov",
    "large-v3-turbo": "OpenVINO/whisper-large-v3-turbo-int8-ov",
}

_OV_MODEL_MARKER = "openvino_encoder_model.xml"


def default_ov_device() -> str:
    """Устройство OpenVINO по умолчанию (env SECRETARY_OV_DEVICE, иначе GPU)."""
    return os.environ.get("SECRETARY_OV_DEVICE", "GPU")


def resolve_ov_model(model_ref: str) -> str:
    """Алиас/hf-ссылка -> repo_id или путь как есть."""
    if model_ref in OV_ALIASES:
        return OV_ALIASES[model_ref]
    return model_ref


def _ensure_ov_model(model_ref: str, cache_root: str | None) -> str:
    """Возвращает локальный путь к OV-модели, скачивая её при необходимости.

    Маркер OV-IR — `openvino_encoder_model.xml` (вместо `model.bin` у CT2).
    """
    from huggingface_hub import snapshot_download

    from .engine import ModelDownloadBar, _split_model_ref, _model_dir

    repo, subfolder = _split_model_ref(model_ref)
    target = _model_dir(cache_root, repo)

    if subfolder is not None:
        model_path = target / subfolder
        if (model_path / _OV_MODEL_MARKER).is_file():
            return str(model_path)
        target.mkdir(parents=True, exist_ok=True)
        ModelDownloadBar.label = f"Загрузка модели {model_ref}"
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=str(target),
                allow_patterns=f"{subfolder}/*",
                tqdm_class=ModelDownloadBar,
                token=os.environ.get("HF_TOKEN") or False,
            )
        finally:
            ModelDownloadBar.label = ""
        return str(model_path)

    if (target / _OV_MODEL_MARKER).is_file():
        return str(target)

    target.mkdir(parents=True, exist_ok=True)
    ModelDownloadBar.label = f"Загрузка модели {model_ref}"
    try:
        snapshot_download(
            repo_id=repo,
            local_dir=str(target),
            tqdm_class=ModelDownloadBar,
            token=os.environ.get("HF_TOKEN") or None,
        )
    finally:
        ModelDownloadBar.label = ""
    return str(target)


def _decode_audio(audio_path: str | Path):
    """PyAV -> моно 16 кГц float32 (та же функция, что у faster-whisper)."""
    from faster_whisper.audio import decode_audio

    return decode_audio(str(audio_path), sampling_rate=16000)


def _speech_only(audio, vad_options):
    """Оставляет только речь по Silero VAD; возвращает (аудио, карта таймкодов)."""
    import numpy as np
    from faster_whisper.vad import SpeechTimestampsMap, get_speech_timestamps

    speech = get_speech_timestamps(audio, vad_options, sampling_rate=16000)
    if not speech:
        return audio, None
    parts = [audio[s["start"]:s["end"]] for s in speech]
    return np.concatenate(parts), SpeechTimestampsMap(speech, 16000)


class OpenVINOEngine:
    """Движок транскрибации на OpenVINO GenAI (интерфейс как у TranscriptionEngine)."""

    def __init__(
        self,
        model: str,
        device: str | None = None,
        cache_dir: str | None = None,
        language: str | None = None,
        vad_filter: bool = True,
        verbose: bool = False,
        num_beams: int = 1,
    ):
        self.model_ref = resolve_ov_model(model)
        self.device = device or default_ov_device()
        self.cache_dir = cache_dir
        self.language = None if language in (None, "auto", "") else language
        self.vad_filter = vad_filter
        self.verbose = verbose
        self.num_beams = num_beams
        self._pipe: Any = None

    def _load_model(self) -> None:
        try:
            import openvino_genai as ov_genai
        except ImportError:
            raise RuntimeError(
                "OpenVINO-бэкенд не установлен. Установите его отдельно: "
                "pip install -r requirements-openvino.txt "
                "(или pip install -e .[openvino])."
            ) from None

        local = Path(self.model_ref)
        path = str(local) if local.is_dir() else _ensure_ov_model(
            self.model_ref, self.cache_dir
        )
        if self.verbose:
            print(f"[модель] {self.model_ref} | backend=openvino device={self.device}\n  {path}")
        self._pipe = ov_genai.WhisperPipeline(path, self.device)

    def _make_config(self, word_timestamps: bool, initial_prompt: str | None):
        cfg = self._pipe.get_generation_config()
        if self.language:
            cfg.language = self.language
        cfg.task = "transcribe"
        cfg.num_beams = self.num_beams
        cfg.return_timestamps = True
        cfg.word_timestamps = word_timestamps
        if initial_prompt:
            cfg.initial_prompt = initial_prompt
        return cfg

    def _generate(self, audio, word_timestamps: bool, initial_prompt: str | None):
        """generate с откатом на сегментные таймкоды, если модель без alignment heads."""
        cfg = self._make_config(word_timestamps, initial_prompt)
        try:
            return self._pipe.generate(audio, cfg), word_timestamps
        except RuntimeError as exc:
            if not word_timestamps:
                raise
            if self.verbose:
                print(f"  [ov] словарные таймкоды недоступны ({exc}); сегментные.")
            cfg = self._make_config(False, initial_prompt)
            return self._pipe.generate(audio, cfg), False

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
        progress_bar=None,
    ) -> dict:
        if self._pipe is None:
            self._load_model()

        audio = _decode_audio(audio_path)
        mapping = None
        if self.vad_filter:
            from faster_whisper.vad import VadOptions

            audio, mapping = _speech_only(audio, VadOptions())

        result, got_words = self._generate(audio, word_timestamps, initial_prompt)

        def to_orig(start: float, end: float) -> tuple[float, float]:
            if mapping is None:
                return start, end
            return (
                mapping.get_original_time(start, is_end=False),
                mapping.get_original_time(end, is_end=True),
            )

        segments: list[tuple[float, float, str]] = []
        for chunk in result.chunks:
            start, end = to_orig(float(chunk.start_ts), float(chunk.end_ts))
            segments.append((start, end, chunk.text))
            if progress_bar is not None:
                progress_bar.n = min(end, progress_bar.total)
                progress_bar.refresh()

        words: list[tuple[float, float, str]] = []
        if got_words:
            for word in result.words or []:
                start, end = to_orig(float(word.start_ts), float(word.end_ts))
                words.append((start, end, word.word))

        if progress_bar is not None:
            progress_bar.n = progress_bar.total
            progress_bar.refresh()

        language = getattr(result, "language", None) or self.language or "?"
        return {
            "language": language,
            "language_probability": 1.0,
            "segments": segments,
            "words": words,
        }
