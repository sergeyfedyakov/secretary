# secretary

[English version](README.md)

Пакетная транскрибация аудиофайлов в текст — локально, без облаков.

- Движок: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2).
- Модель по умолчанию: `large-v3-turbo` (хорошее качество, многоязычная, отлично понимает русский).
- Модель скачивается автоматически при первом запуске с прогресс-баром (полоса, скорость, ETA).
- Язык по умолчанию определяется автоматически.
- Диаризация (метки говорящих): флаг `--diarize`, бэкенд `pyannote/speaker-diarization-community-1`.

## Готовые сборки (Windows)

Скачай готовый exe из [GitHub Releases](https://github.com/sergeyfedyakov/secretary/releases) — Python не нужен:

| Файл | Размер | Диаризация |
|---|---|---|
| `secretary-light.exe` | ~100 МБ | нет |
| `secretary.exe` | ~300 МБ | ✓ |

Модели скачиваются при первом запуске автоматически. Для диаризации положи `.env` с токеном рядом с exe (см. шаблон `.env.sample` в релизе) или задай переменную окружения `HF_TOKEN`.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest (для тестов)
pip install -e .                # консольная команда `secretary`
```

После установки можно запускать одной командой через `run.cmd` (Windows) или `run.sh` (Linux/macOS) — без ручной активации venv:

```cmd
:: Windows
run.cmd запись.mp3
run.cmd ./recordings --language ru --out-dir ./transcripts
```

```bash
# Linux / macOS
./run.sh запись.mp3
./run.sh ./recordings --language ru --out-dir ./transcripts
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

# диаризация: метки говорящих [HH:MM:SS] SPEAKER_nn: текст
secretary запись.mp3 --diarize

# формат «под SRT» с нарезкой длинных реплик на ~10-сек участки
secretary лекция.mp3 --diarize --format srt

# системный промпт для распознавания
secretary лекция.mp3 --prompt "Транскрипция лекции по программированию"
```

`python -m secretary ...` работает так же.

## Модели с подпапками (pre-quantized, faster-whisper)

Некоторые HF-репозитории содержат несколько готовых CTranslate2-вариантов в
подпапках (например `coriollon/whisper-large-v3-turbo-russian` — русский
fine-tune с пунктуацией). В корне такого репозитория `model.bin` нет, поэтому
нужно указывать подпапку явно:

```bash
# вариант "int8 веса + fp16 вычисления" (782 МБ, актуальная v2)
secretary запись.mp3 --model coriollon/whisper-large-v3-turbo-russian/ct2_int8_float16

# вариант "int16" (1.6 ГБ, в карточке модели отмечен как устаревшая v1-сборка)
secretary запись.mp3 --model coriollon/whisper-large-v3-turbo-russian/ct2-int16 --compute-type int16
```

Скачивается только выбранная подпапка. `--compute-type` на CPU автоматически
`int8` (проверено: работает с `ct2_int8_float16`); на CUDA для этого варианта
указывайте `--compute-type int8_float16`, как в карточке модели.

## Оффлайн-установка моделей

Если целевая машина без интернета — скачайте модели на машине с доступом
и перенесите папку кэша.

### Способ 1: huggingface-cli (рекомендуется)

```bash
pip install huggingface_hub
# Базовая модель Whisper (3 ГБ)
huggingface-cli download Systran/faster-whisper-large-v3-turbo --local-dir ./models/faster-whisper-large-v3-turbo

# Русский fine-tune с пунктуацией (782 МБ)
huggingface-cli download coriollon/whisper-large-v3-turbo-russian --local-dir ./models/whisper-large-v3-turbo-russian --include "ct2_int8_float16/*"

# Модель диаризации (нужен HF_TOKEN, ~400 МБ)
huggingface-cli download pyannote/speaker-diarization-community-1 --local-dir ./models/speaker-diarization --token hf_xxx
```

Перенесите папку `./models/` на целевую машину и укажите путь:

```cmd
set SECRETARY_MODEL_CACHE=D:\models
secretary запись.mp3 --model D:\models\faster-whisper-large-v3-turbo
```

### Способ 2: через первый запуск на машине с интернетом

Запустите `secretary` один раз на машине с интернетом — модель скачается
в `~/.cache/secretary/models/`. Скопируйте эту папку на целевую машину
в то же расположение, либо укажите `--model-cache` / `SECRETARY_MODEL_CACHE`.

### Способ 3: готовый exe + модели отдельно

Скачайте `secretary.exe` из [GitHub Releases](../../releases). Модели в exe
не вшиты — при первом запуске они скачаются автоматически, либо перенесите
их вручную одним из способов выше. Переменная `SECRETARY_MODEL_CACHE` работает
и с exe-версией.

## Опции

| Опция | По умолчанию | Описание |
|---|---|---|
| `-m`, `--model` | `large-v3-turbo` | алиас (`tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`), HF-репо, `репо/подпапка` или локальный путь. Также через env `SECRETARY_MODEL` |
| `-l`, `--language` | авто | код языка (`ru`, `en`, ...) |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` |
| `--compute-type` | `auto` | `auto` (int8 на CPU, float16 на CUDA), `int8`, `float16`, `int8_float16`, `float32`, ... |
| `--vad` / `--no-vad` | включён | фильтр тишины |
| `--diarize` | выкл | диаризация: метки `SPEAKER_nn` (бэкенд pyannote community-1, нужен `HF_TOKEN`) |
| `-f`, `--format` | `plain` | `plain` — сплошной текст; `srt` — строки `[HH:MM:SS] [SPEAKER_nn] текст` с нарезкой длинных реплик на ~10 с |
| `--prompt` | — | системный промпт для распознавания (например, «Транскрипция лекции по программированию») |
| `-o`, `--out-dir` | рядом с файлом | папка результатов |
| `--model-cache` | `~/.cache/secretary/models` | каталог моделей (или env `SECRETARY_MODEL_CACHE`) |
| `--ffmpeg-path` | — | путь к ffmpeg (резерв) |
| `-v`, `--verbose` | выкл | подробный лог (язык, число сегментов, говорящих) |
| `-q`, `--no-progress` | выкл | не показывать прогресс-бар транскрибации (полезно в CI/логах) |
| `-n`, `--newer-than` | — | только файлы новее даты: `2026-08-10`, `2026-08-10T14:30`, `2d`, `6h`, `30m` |
| `-?`, `--help` | — | показать справку и выйти (также при запуске без аргументов) |

## ffmpeg

Отдельный бинарь не обязателен: faster-whisper декодирует mp3/m4a/mp4 и др. через PyAV.
Если всё же нужен CLI-конверт (экзотические форматы): `winget install ffmpeg` или сборки [gyan.dev](https://www.gyan.dev/ffmpeg/builds/),
путь можно передать в `--ffmpeg-path` или переменную окружения `FFMPEG_BINARY`.

## Токен Hugging Face (`HF_TOKEN`)

Модели Whisper (faster-whisper) публичные — токен для распознавания не нужен.
Токен нужен для **диаризации** (модели pyannote gated).

Где взять:
1. Войти/зарегистрироваться на [huggingface.co](https://huggingface.co).
2. [Settings → Access Tokens](https://huggingface.co/settings/tokens) → **New token**,
   роль **Read** → создать и скопировать (`hf_...`).
3. Одноразово принять условия модели pyannote — открыть и нажать
   **«Agree and access repository»**:
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   (все компоненты пайплайна лежат внутри этого репозитория)

Как передать (токен не коммитится — см. `.gitignore`):

```bash
# файл .env в корне проекта (пример — .env.example)
HF_TOKEN=hf_xxxxxxxxxxxxx
```

Или переменная окружения ОС: `$env:HF_TOKEN = "hf_xxx"` (PowerShell), `set HF_TOKEN=hf_xxx` (cmd).
Файл `.env` подхватывается автоматически при запуске.

## Диаризация

Бэкенд: `pyannote/speaker-diarization-community-1` (pyannote.audio 4.x).
Схема: STT-сегменты → диаризация по той же записи → каждому слову/сегменту
присваивается говорящий по пересечению таймкодов (используется
`exclusive_speaker_diarization` — без перекрывающихся реплик, точнее склеивается
с whisper).

- Модели скачиваются при первом `--diarize` и кэшируются в
  `~/.cache/secretary/pyannote` (env `PYANNOTE_CACHE`) — сохраняются между сеансами.
- Паpаметры числа говорящих (`--min-speakers`/`--max-speakers`) намеренно не вынесены
  в CLI: при пакетной обработке они заранее неизвестны.
