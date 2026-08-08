"""Склейка сегментов STT с диаризацией и форматирование результата.

Поддерживаются два формата:
- plain: сплошной текст (без диаризации) либо строки [HH:MM:SS] SPEAKER_nn: текст
  (по сегментам whisper);
- srt: «под SRT» — строки [HH:MM:SS] [SPEAKER_nn] текст, длинные реплики
  и монологи нарезаются на участки ~max_duration секунд (по словесным таймкодам).
"""

from __future__ import annotations

from typing import Iterable

from .diarization import SpeakerTurn


def assign_speaker(start: float, end: float, turns: Iterable[SpeakerTurn]) -> str | None:
    """Говорящий с максимальным пересечением интервала; None, если пересечения нет."""
    best: str | None = None
    best_overlap = 0.0
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = turn.speaker
    return best


def _fmt_ts(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_plain(
    segments: list[tuple[float, float, str]],
    turns: Iterable[SpeakerTurn] | None = None,
) -> str:
    """Сплошной текст; с диаризацией — строки [HH:MM:SS] SPEAKER_nn: текст."""
    if not turns:
        return " ".join(text.strip() for _, _, text in segments)
    turns = list(turns)
    lines = []
    for start, end, text in segments:
        text = text.strip()
        if not text:
            continue
        speaker = assign_speaker(start, end, turns)
        prefix = f"[{_fmt_ts(start)}] {speaker}: " if speaker else f"[{_fmt_ts(start)}] "
        lines.append(f"{prefix}{text}")
    return "\n".join(lines)


def format_srt(
    segments: list[tuple[float, float, str]],
    words: list[tuple[float, float, str]],
    turns: Iterable[SpeakerTurn] | None = None,
    max_duration: float = 10.0,
) -> str:
    """«Под SRT»: строки [HH:MM:SS] [SPEAKER_nn] текст, участки ~max_duration.

    При наличии словесных таймкодов (words) нарезка идёт по словам и смена
    говорящего внутри сегмента whisper учитывается точно. Иначе — по сегментам.
    """
    turns = list(turns) if turns else []

    def speaker_for(start: float, end: float) -> str | None:
        return assign_speaker(start, end, turns) if turns else None

    tokens: list[tuple[float, float, str]] = words if words else segments

    lines: list[str] = []
    chunk_start: float | None = None
    chunk_speaker: str | None = None
    chunk_words: list[str] = []

    def flush() -> None:
        nonlocal chunk_start, chunk_speaker, chunk_words
        if chunk_words and chunk_start is not None:
            text = " ".join(w.strip() for w in chunk_words if w.strip())
            if text:
                ts = _fmt_ts(chunk_start)
                prefix = f"[{ts}] {chunk_speaker}: " if chunk_speaker else f"[{ts}] "
                lines.append(f"{prefix}{text}")
        chunk_start = None
        chunk_speaker = None
        chunk_words = []

    for start, end, word in tokens:
        text = word.strip()
        if not text:
            continue
        speaker = speaker_for(start, end)
        if chunk_start is None:
            chunk_start, chunk_speaker, chunk_words = start, speaker, [text]
            continue
        if speaker != chunk_speaker or (end - chunk_start) > max_duration:
            flush()
            chunk_start, chunk_speaker, chunk_words = start, speaker, [text]
            continue
        chunk_words.append(text)
    flush()

    return "\n".join(lines)
