"""Keep each Windows entry point self-contained, including when opened inside a ZIP."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ('Start-Xiangqi-Windows.bat', '启动残局实验室.bat', 'xiangqi_lab/Start-Windows.bat')
HEADER = r'''@echo off
setlocal DisableDelayedExpansion
set "XIANGQI_LAUNCHER_PATH=%~f0"
set "XIANGQI_ARCHIVE_PATH=%~1"
powershell.exe -NoLogo -NoProfile -STA -ExecutionPolicy Bypass -Command "$raw=[IO.File]::ReadAllText($env:XIANGQI_LAUNCHER_PATH); $marker='#==XIANGQI_POWERSHELL=='; & ([scriptblock]::Create($raw.Substring($raw.LastIndexOf($marker)+$marker.Length)))"
set "XIANGQI_EXIT=%ERRORLEVEL%"
if "%XIANGQI_EXIT%"=="0" exit /b 0
echo.
echo Launch stopped. Read the message above before closing this window.
pause
exit /b %XIANGQI_EXIT%
#==XIANGQI_POWERSHELL==
'''


def launcher_bytes():
    source = (ROOT / 'tools/windows_launcher.ps1').read_text(encoding='utf-8')
    return (HEADER + source).replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')


def write_launchers():
    data = launcher_bytes()
    for name in TARGETS:
        (ROOT / name).write_bytes(data)


if __name__ == '__main__':
    write_launchers()
