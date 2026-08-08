"""Тесты склейки STT с диаризацией и форматирования."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secretary.diarization import SpeakerTurn
from secretary.merge import assign_speaker, format_plain, format_srt

TURN_A = SpeakerTurn(0.0, 10.0, "SPEAKER_00")
TURN_B = SpeakerTurn(10.0, 20.0, "SPEAKER_01")
TURN_C = SpeakerTurn(30.0, 40.0, "SPEAKER_02")


def test_assign_speaker_basic():
    assert assign_speaker(1.0, 5.0, [TURN_A, TURN_B]) == "SPEAKER_00"
    assert assign_speaker(11.0, 15.0, [TURN_A, TURN_B]) == "SPEAKER_01"


def test_assign_speaker_no_overlap_returns_none():
    assert assign_speaker(21.0, 25.0, [TURN_A, TURN_B]) is None


def test_assign_speaker_prefers_max_overlap():
    turns = [SpeakerTurn(0.0, 20.0, "SPEAKER_00"), SpeakerTurn(10.0, 12.0, "SPEAKER_01")]
    assert assign_speaker(9.0, 13.0, turns) == "SPEAKER_00"


def test_assign_speaker_partial_edge_overlap():
    assert assign_speaker(9.0, 10.5, [TURN_A, TURN_B]) == "SPEAKER_00"
    assert assign_speaker(9.9, 10.1, [TURN_A, TURN_B]) == "SPEAKER_00"


def test_format_plain_without_diarization_joins_text():
    segments = [(0.0, 1.0, "Привет. "), (1.0, 2.0, "Как дела?")]
    assert format_plain(segments) == "Привет. Как дела?"


def test_format_plain_with_diarization_labels_lines():
    segments = [(1.0, 5.0, "Первый говорит "), (11.0, 15.0, "второй отвечает")]
    text = format_plain(segments, [TURN_A, TURN_B])
    lines = text.splitlines()
    assert lines[0].startswith("[00:00:01] SPEAKER_00: ")
    assert lines[1].startswith("[00:00:11] SPEAKER_01: ")


def test_format_plain_ts_rounds_to_seconds():
    segments = [(59.6, 60.4, "тест ")]
    lines = format_plain(segments, [TURN_A]).splitlines()
    assert lines[0].startswith("[00:01:00] ")


def test_format_srt_splits_long_monologue_into_chunks():
    words = [(0.0, 0.5, "слово"), (0.5, 1.0, "два"), (12.0, 12.5, "хвост")]
    text = format_srt(segments=[], words=words, max_duration=10.0)
    lines = text.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("[00:00:00] ")
    assert lines[1].startswith("[00:00:12] ")


def test_format_srt_splits_on_speaker_change():
    words = [(0.0, 1.0, "привет"), (11.0, 12.0, "пока")]
    turns = [SpeakerTurn(0.0, 10.0, "SPEAKER_00"), SpeakerTurn(10.0, 20.0, "SPEAKER_01")]
    lines = format_srt(segments=[], words=words, turns=turns).splitlines()
    assert "SPEAKER_00:" in lines[0]
    assert "SPEAKER_01:" in lines[1]


def test_format_srt_falls_back_to_segments_without_words():
    segments = [(0.0, 15.0, "длинная реплика"), (25.0, 35.0, "ещё реплика")]
    turns = [TURN_A, TURN_C]
    lines = format_srt(segments=segments, words=[], turns=turns, max_duration=10.0).splitlines()
    assert lines[0].startswith("[00:00:00] SPEAKER_00: ")
    assert lines[1].startswith("[00:00:25] SPEAKER_02: ")


def test_format_srt_empty():
    assert format_srt(segments=[], words=[]) == ""
