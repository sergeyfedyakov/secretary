"""CLI: пакетная транскрибация аудио в текст."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from glob import glob, has_magic
from pathlib import Path

from tqdm import tqdm

from . import __version__
from .diarization import create_diarizer
from .engine import TranscriptionEngine, get_audio_duration
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
        help="Аудиофайлы, папки (сканируются рекурсивно) или glob-паттерны, "
             "например \"D:\\записи\\*.mp4\"",
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
        help="Диаризация: метки SPEAKER_nn (бэкенд pyannote, нужен HF_TOKEN)",
    )
    parser.add_argument(
        "--format",
        choices=("plain", "srt"),
        default="plain",
        help="Формат вывода: plain — сплошной текст; srt — строки "
             "[HH:MM:SS] [SPEAKER_nn] текст с нарезкой длинных реплик на ~10 с",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Системный промпт для распознавания, например "
             "'транскрипция лекции по программированию'",
    )
    parser.add_argument("--out-dir", default=None, help="Папка для результатов (по умолчанию — рядом с файлом)")
    parser.add_argument("--model-cache", default=None, help="Каталог кэша моделей")
    parser.add_argument(
        "--ffmpeg-path",
        default=None,
        help="Путь к бинарю ffmpeg (резерв; декодирование идёт через PyAV, обычно не нужен)",
    )
    parser.add_argument("--verbose", action="store_true", help="Подробный лог")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Не показывать прогресс-бар транскрибации (полезно в CI/логах)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def collect_files(inputs: list[str]) -> list[Path]:
    """Собирает аудиофайлы из путей/папок/glob-паттернов (рекурсивно)."""
    files: list[Path] = []
    for raw in inputs:
        candidates = [Path(p) for p in glob(raw, recursive=True)] if has_magic(raw) else [Path(raw)]
        if not candidates:
            print(f"ПРЕДУПРЕЖДЕНИЕ: ничего не найдено: {raw}", file=sys.stderr)
            continue
        for path in candidates:
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


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass

    load_env_file()
    args = build_parser().parse_args(argv)

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

    diarizer = None
    if args.diarize:
        try:
            diarizer = create_diarizer()
        except Exception as exc:
            print(f"ОШИБКА инициализации диаризации: {exc}", file=sys.stderr)
            return 2

    from .merge import format_plain, format_srt

    total = len(files)
    ok = 0
    for index, src in enumerate(files, 1):
        print(f"[{index}/{total}] {src}")
        try:
            duration = get_audio_duration(src) if not args.no_progress else None
            desc = f"  {src.stem}" if duration else None
            pbar = (
                tqdm(
                    total=duration,
                    unit="s",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt:.1f}/{total_fmt:.1f}s [{elapsed}<{remaining}]",
                    desc=desc,
                    dynamic_ncols=True,
                    mininterval=0.5,
                )
                if duration
                else None
            )
            result = engine.transcribe(
                src,
                initial_prompt=args.prompt,
                word_timestamps=args.format == "srt",
                progress_bar=pbar,
            )
            if pbar:
                pbar.close()
            turns = diarizer.diarize(src) if diarizer else None
            if args.format == "srt":
                text = format_srt(result["segments"], result["words"], turns)
            else:
                text = format_plain(result["segments"], turns)
            out = output_path_for(src, args.out_dir)
            out.write_text(text, encoding="utf-8")
            if args.verbose:
                print(
                    f"  язык: {result['language']} "
                    f"(p={result['language_probability']:.2f}), "
                    f"сегментов: {len(result['segments'])}"
                    + (f", говорящих: {len({t.speaker for t in turns})}" if turns else "")
                )
            print(f"  -> {out}")
            ok += 1
        except Exception as exc:
            print(f"  ОШИБКА: {exc}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()

    print(f"Готово: {ok}/{total} файлов.")
    return 0 if ok == total else 2
