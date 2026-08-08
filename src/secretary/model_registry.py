"""Каталог моделей и аудиоформатов."""

from faster_whisper.utils import _MODELS

DEFAULT_MODEL = "large-v3-turbo"

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".m4a", ".mp4", ".ogg", ".opus", ".aac",
    ".wma", ".webm", ".aiff", ".aif", ".m4b", ".wv", ".caf", ".ape",
}


def resolve_model_repo(model_ref: str) -> str | None:
    """Возвращает HF repo_id для известного алиаса, иначе None.

    None означает, что model_ref — это либо путь к локальной папке с моделью,
    либо произвольный HF repo_id (передаём как есть).
    """
    return _MODELS.get(model_ref)
