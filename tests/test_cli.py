"""Тесты сбора файлов (collect_files): файлы, папки, glob-паттерны."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secretary.cli import collect_files, _filter_by_mtime, _parse_time_filter


def test_single_file(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"")
    assert collect_files([str(f)]) == [f]


def test_folder_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    a = tmp_path / "a.mp3"
    b = tmp_path / "sub" / "b.m4a"
    c = tmp_path / "notes.txt"
    a.write_bytes(b"")
    b.write_bytes(b"")
    c.write_bytes(b"")
    assert collect_files([str(tmp_path)]) == sorted([a, b], key=str)


def test_wildcard_mp4(tmp_path):
    a = tmp_path / "one.mp4"
    b = tmp_path / "two.mp4"
    c = tmp_path / "skip.mp3"
    a.write_bytes(b"")
    b.write_bytes(b"")
    c.write_bytes(b"")
    pattern = str(tmp_path / "*.mp4")
    assert collect_files([pattern]) == sorted([a, b], key=str)


def test_wildcard_nested(tmp_path):
    (tmp_path / "sub").mkdir()
    a = tmp_path / "a.mp3"
    b = tmp_path / "sub" / "b.mp3"
    a.write_bytes(b"")
    b.write_bytes(b"")
    pattern = str(tmp_path / "**" / "*.mp3")
    assert collect_files([pattern]) == sorted([a, b], key=str)


def test_folder_with_brackets(tmp_path):
    folder = tmp_path / "LostFilm.TV [MP4]"
    folder.mkdir()
    a = folder / "one.mp4"
    b = folder / "two.mp4"
    a.write_bytes(b"")
    b.write_bytes(b"")
    assert collect_files([str(folder)]) == sorted([a, b], key=str)


def test_glob_in_folder_with_brackets(tmp_path):
    folder = tmp_path / "LostFilm.TV [MP4]"
    folder.mkdir()
    a = folder / "one.mp4"
    b = folder / "two.mp4"
    c = folder / "skip.mp3"
    a.write_bytes(b"")
    b.write_bytes(b"")
    c.write_bytes(b"")
    pattern = str(folder / "*.mp4")
    assert collect_files([pattern]) == sorted([a, b], key=str)


def test_missing_and_empty_glob(tmp_path, capsys):
    missing = tmp_path / "nope.mp3"
    files = collect_files([str(missing), str(tmp_path / "*.zzz")])
    err = capsys.readouterr().err
    assert files == []
    assert "не найдено" in err


def test_filter_by_mtime_passes_recent(tmp_path):
    f = tmp_path / "new.mp3"
    f.write_bytes(b"")
    assert _filter_by_mtime([f], 0) == [f]


def test_filter_by_mtime_drops_old(tmp_path):
    f = tmp_path / "old.mp3"
    f.write_bytes(b"")
    assert _filter_by_mtime([f], time.time() + 3600) == []


def test_parse_time_filter_relative_hours():
    expected = time.time() - 6 * 3600
    result = _parse_time_filter("6h")
    assert abs(result - expected) < 1


def test_parse_time_filter_relative_days():
    expected = time.time() - 2 * 86400
    result = _parse_time_filter("2d")
    assert abs(result - expected) < 1


def test_parse_time_filter_iso_date():
    result = _parse_time_filter("2026-08-10")
    assert result == _parse_time_filter("2026-08-10T00:00")
