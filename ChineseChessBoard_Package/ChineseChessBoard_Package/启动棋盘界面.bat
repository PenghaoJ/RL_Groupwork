@echo off
setlocal
cd /d "%~dp0"
set "CHESS_PYTHON=%~dp0..\..\.runtime\python\python.exe"
if not exist "%CHESS_PYTHON%" (
    echo Python runtime is missing: "%CHESS_PYTHON%"
    pause
    exit /b 1
)
set "PYTHONHOME="
set "PYTHONPATH="
set "TCL_LIBRARY="
set "TK_LIBRARY="
"%CHESS_PYTHON%" -E "%~dp0pikafish_gui.py"
if errorlevel 1 (
    echo Chess board failed to start. See the error above.
    pause
    exit /b 1
)
