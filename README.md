# secretary

Пакетная транскрибация аудиофайлов в текст — локально, без облаков.

- Движок: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2).
- Модель по умолчанию: `large-v3-turbo` (хорошее качество, многоязычная, отлично понимает русский).
- Модель скачивается автоматически при первом запуске с прогресс-баром (полоса, скорость, ETA).
- Язык по умолчанию определяется автоматически.
- Диаризация (метки говорящих) запланирована — флаг `--diarize` пока зарезервирован.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .                # консольная команда `secretary`
```

## Использование

```bash
# один файл
secretary запись.mp3

# папка (рекурсивно) — по .txt на каждый аудиофайл
secretary ./recordings

# явный язык + результат в отдельную папку
secretary ./recordings --language ru --out-dir ./transcripts

# модель поменьше для CPU, отключить фильтр тишины
secretary a.mp3 --model small --compute-type int8 --no-vad
```

`python -m secretary ...` работает так же.

## Опции

| Опция | По умолчанию | Описание |
|---|---|---|
| `--model` | `large-v3-turbo` | алиас (`tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`), HF-репо или локальный путь |
| `--language` | авто | код языка (`ru`, `en`, ...) |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` |
| `--compute-type` | `auto` | `auto` (int8 на CPU, float16 на CUDA), `int8`, `float16`, `float32`, ... |
| `--vad` / `--no-vad` | включён | фильтр тишины |
| `--diarize` | выкл | диаризация (пока не реализована, выход 2) |
| `--out-dir` | рядом с файлом | папка результатов |
| `--model-cache` | `~/.cache/secretary/models` | каталог моделей (или env `SECRETARY_MODEL_CACHE`) |
| `--ffmpeg-path` | — | путь к ffmpeg (резерв) |
| `--verbose` | выкл | подробный лог (язык, число сегментов) |

## ffmpeg

Отдельный бинарь не обязателен: faster-whisper декодирует mp3/m4a/mp4 и др. через PyAV.
Если всё же нужен CLI-конверт (экзотические форматы): `winget install ffmpeg` или сборки [gyan.dev](https://www.gyan.dev/ffmpeg/builds/),
путь можно передать в `--ffmpeg-path` или переменную окружения `FFMPEG_BINARY`.

## Токен Hugging Face (`HF_TOKEN`)

Модели Whisper (faster-whisper) публичные — токен для распознавания не нужен.
Токен понадобится для **диаризации** (pyannote, в разработке) и для любых закрытых (gated) моделей.

Где взять:
1. Войти/зарегистрироваться на [huggingface.co](https://huggingface.co).
2. [Settings → Access Tokens](https://huggingface.co/settings/tokens) → **New token**,
   роль **Read** → создать и скопировать (`hf_...`).
3. Одноразово принять условия моделей pyannote — открыть и нажать
   **«Agree and access repository»**:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

Как передать (токен не коммитится — см. `.gitignore`):

```bash
# файл .env в корне проекта (пример — .env.example)
HF_TOKEN=hf_xxxxxxxxxxxxx
```

Или переменная окружения ОС: `$env:HF_TOKEN = "hf_xxx"` (PowerShell), `set HF_TOKEN=hf_xxx` (cmd).
Файл `.env` подхватывается автоматически при запуске.

## Диаризация (дорожная карта)

Бэкенд выбран: `pyannote/speaker-diarization-3.1` (исследование — в `.ai/research.md`, план — в `.ai/plan.md`).
Схема: сегменты STT → диаризация по той же записи → склейка по таймкодам → метки `SPEAKER_nn`.
Потребуется токен Hugging Face (`HF_TOKEN`) и принятие условий моделей pyannote.
