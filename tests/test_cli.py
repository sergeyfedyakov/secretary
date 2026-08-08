"""Тесты сбора файлов (collect_files): файлы, папки, glob-паттерны."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secretary.cli import collect_files


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


def test_missing_and_empty_glob(tmp_path, capsys):
    missing = tmp_path / "nope.mp3"
    files = collect_files([str(missing), str(tmp_path / "*.zzz")])
    err = capsys.readouterr().err
    assert files == []
    assert "не найдено" in err
