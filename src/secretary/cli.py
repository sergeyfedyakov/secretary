"""CLI: пакетная транскрибация аудио в текст."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from . import __version__
from .diarization import create_diarizer
from .engine import TranscriptionEngine
from .env import load_env_file
from .model_registry import AUDIO_EXTENSIONS, DEFAULT_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="secretary",
        description="Пакетная транскрибация аудио в текст (локально, faster-whisper).",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="ФАЙЛ_ИЛИ_ПАПКА",
        help="Аудиофайлы или папки (сканируются рекурсивно)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Модель: алиас (tiny/base/small/medium/large-v3/large-v3-turbo), "
             f"HF-репо или локальный путь. По умолчанию: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Код языка (ru, en, ...). По умолчанию — автоопределение.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="auto|int8|float16|int8_float16|float32 ... (auto: int8 на CPU, float16 на CUDA)",
    )
    parser.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Фильтр тишины (по умолчанию включён, отключить: --no-vad)",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Диаризация (разметка говорящих). Пока не реализована.",
    )
    parser.add_argument("--out-dir", default=None, help="Папка для результатов (по умолчанию — рядом с файлом)")
    parser.add_argument("--model-cache", default=None, help="Каталог кэша моделей")
    parser.add_argument(
        "--ffmpeg-path",
        default=None,
        help="Путь к бинарю ffmpeg (резерв; декодирование идёт через PyAV, обычно не нужен)",
    )
    parser.add_argument("--verbose", action="store_true", help="Подробный лог")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def collect_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)
        elif path.is_file():
            files.append(path)
        else:
            print(f"ПРЕДУПРЕЖДЕНИЕ: не найдено: {raw}", file=sys.stderr)
    return sorted(set(files), key=str)


def output_path_for(src: Path, out_dir: str | None) -> Path:
    if out_dir:
        folder = Path(out_dir)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{src.stem}.txt"
    return src.with_suffix(".txt")


def format_text(segments: list[tuple[float, float, str]]) -> str:
    return " ".join(seg[2].strip() for seg in segments)


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass

    load_env_file()
    args = build_parser().parse_args(argv)

    if args.diarize:
        try:
            create_diarizer()
        except NotImplementedError as exc:
            print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 2

    if args.ffmpeg_path:
        os.environ.setdefault("FFMPEG_BINARY", args.ffmpeg_path)

    files = collect_files(args.inputs)
    if not files:
        print("ОШИБКА: не найдено ни одного аудиофайла по заданным путям.", file=sys.stderr)
        return 1

    engine = TranscriptionEngine(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        cache_dir=args.model_cache,
        language=args.language,
        vad_filter=args.vad,
        verbose=args.verbose,
    )

    total = len(files)
    ok = 0
    for index, src in enumerate(files, 1):
        print(f"[{index}/{total}] {src}")
        try:
            result = engine.transcribe(src)
            text = format_text(result["segments"])
            out = output_path_for(src, args.out_dir)
            out.write_text(text, encoding="utf-8")
            if args.verbose:
                print(
                    f"  язык: {result['language']} "
                    f"(p={result['language_probability']:.2f}), "
                    f"сегментов: {len(result['segments'])}"
                )
            print(f"  -> {out}")
            ok += 1
        except Exception as exc:
            print(f"  ОШИБКА: {exc}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()

    print(f"Готово: {ok}/{total} файлов.")
    return 0 if ok == total else 2
