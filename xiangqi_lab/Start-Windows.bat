@echo off
setlocal
cd /d "%~dp0"
set "CHESS_PYTHON=%~dp0runtime\python\python.exe"
if not exist "%CHESS_PYTHON%" set "CHESS_PYTHON=%~dp0..\.runtime\python\python.exe"
if not exist "%CHESS_PYTHON%" (
    echo Please extract the complete Windows release ZIP first.
    pause
    exit /b 1
)
"%CHESS_PYTHON%" -E -X utf8 "%~dp0app.py"
if errorlevel 1 pause
