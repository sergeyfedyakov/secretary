@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ОШИБКА: Python не найден: %PYTHON%
    echo.
    echo Сначала соберите окружение:
    echo   cd /d "%SCRIPT_DIR%"
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    echo   .venv\Scripts\pip install -e .
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"
"%PYTHON%" -m secretary %*
exit /b %ERRORLEVEL%
