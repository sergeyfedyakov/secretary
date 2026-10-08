"""Конвертация HF-модели Whisper в OpenVINO IR для бэкенда `--backend ov`.

Build-time утилита: требует отдельного окружения с optimum-intel (тянет
torch/transformers/nncf), поэтому НЕ входит в рантайм secretary.

Пример (создаём одноразовое окружение):
    py -3.12 -m venv .conv-venv
    .conv-venv\\Scripts\\pip install "optimum[openvino]"
    .conv-venv\\Scripts\\python tools\\convert_openvino.py \
        --model coriollon/whisper-large-v3-turbo-russian \
        --output models\\whisper-large-v3-turbo-russian-fp16-ov \
        --weight-format fp16

Готовую папку затем указывают в secretary:
    secretary запись.m4a --backend ov --model <путь-к-папке>

Либо берут уже готовое OV-IR прямо с HF (без конвертации):
    secretary запись.m4a --backend ov --model OpenVINO/whisper-large-v3-turbo-int8-ov
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convert_openvino",
        description="Экспорт Whisper в OpenVINO IR через optimum-cli (optimum-intel).",
    )
    p.add_argument("--model", required=True,
                   help="HF repo_id или локальный путь к transformers-модели "
                        "(нужны полные веса: model.safetensors)")
    p.add_argument("--output", required=True, help="Папка для OV-IR")
    p.add_argument("--weight-format", choices=("fp16", "int8", "int4"), default="fp16",
                   help="Точность весов (fp16 — без калибровки, рекомендуемое)")
    p.add_argument("--task", default="automatic-speech-recognition-with-past")
    p.add_argument("--dataset", default=None,
                   help="Датасет калибровки для int8/int4 (без него — data-free)")
    p.add_argument("--num-samples", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "optimum.commands.optimum_cli", "export", "openvino",
        "-m", args.model, "--task", args.task,
        "--weight-format", args.weight_format,
    ]
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    if args.num_samples is not None:
        cmd += ["--num-samples", str(args.num_samples)]
    cmd.append(str(out))

    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ.copy())
    print("DONE:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
