"""Exercise the real BAT, auto extraction, HTTP server and Pikafish on Windows.

Run after build_release.py. Creates isolated fixtures under .setup and never stops
an existing user server. Only the process trees started by this script are stopped.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def request(url, data=None):
    body = None if data is None else json.dumps(data).encode()
    headers = {} if data is None else {'Content-Type': 'application/json'}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=20) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'{url}: {error.read().decode("utf-8", errors="replace")}') from error


def check_bat(bat, folder, label, expect_text, cache):
    env = os.environ.copy()
    env['XIANGQI_LAUNCHER_HOME'] = str(cache)
    env['XIANGQI_NO_BROWSER'] = '1'
    env['PYTHONHOME'] = str(folder / 'invalid-python-home')
    env['PYTHONPATH'] = str(folder / 'invalid-python-path')
    # A machine without Python on PATH must still be able to launch the offline ZIP.
    system = Path(os.environ['SystemRoot']) / 'System32'
    env['PATH'] = str(system) + ';' + str(system / 'WindowsPowerShell/v1.0')
    log = folder / (label + '.log')
    with log.open('wb') as output:
        process = subprocess.Popen(
            f'"{system / "cmd.exe"}" /d /s /c ""{bat}""',
            cwd=folder, env=env, stdout=output, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            deadline = time.monotonic() + 75
            url = None
            while time.monotonic() < deadline:
                text = log.read_text(encoding='utf-8', errors='replace')
                match = re.search(r'http://127\.0\.0\.1:\d+', text)
                if match:
                    url = match.group()
                    break
                if process.poll() is not None:
                    raise RuntimeError(f'{label} stopped ({process.returncode}):\n{text}')
                time.sleep(.2)
            if not url:
                raise RuntimeError(f'{label} did not start:\n{log.read_text(errors="replace")}')
            assert expect_text in text, text
            meta = request(url + '/api/meta')
            assert meta['version'] == '1.2.0' and meta['engine_available'], meta
            with urllib.request.urlopen(url + '/', timeout=10) as response:
                assert b'Choose a board' in response.read()
            session = request(url + '/api/session', {})
            state = session['state']
            moved = request(url + '/api/engine', {
                'session': session['session'], 'revision': state['revision'],
                'model': 'pikafish', 'milliseconds': 100,
            })['state']
            assert moved['revision'] == state['revision'] + 1
            assert moved['fen'] != state['fen']
            model = request(url + '/api/models/check', {'model': 'custom-team-template'})
            assert model.get('passed'), model
            print(f'PASS {label}: website, Pikafish move and team model; {url}', flush=True)
        finally:
            if process.poll() is None:
                subprocess.run([str(system / 'taskkill.exe'), '/PID', str(process.pid), '/T', '/F'],
                               capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            process.wait(timeout=15)


def main():
    if os.name != 'nt':
        raise SystemExit('Run this smoke check on Windows.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'dist/XiangqiLab-Windows-v1.2.0.zip')
    args = parser.parse_args()
    folder = ROOT / '.setup' / ('launcher 中文 & (team) ! ' + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    archive = folder / args.archive.name
    shutil.copy2(args.archive, archive)
    bat = folder / 'Start-Xiangqi-Windows.bat'
    shutil.copy2(ROOT / 'dist/Start-Xiangqi-Windows.bat', bat)
    cache = folder / 'cache'
    check_bat(bat, folder, 'first-auto-extract', 'Unpacking the full Windows package', cache)
    check_bat(bat, folder, 'repeat-use-cache', 'Using the unpacked Windows package', cache)
    # Simulate a ZIP viewer extracting only its internal BAT to a temporary child folder.
    isolated = folder / 'zip-viewer-temp'
    isolated.mkdir()
    with zipfile.ZipFile(archive) as zipped:
        internal = isolated / 'Start-Windows.bat'
        internal.write_bytes(zipped.read('XiangqiLab/Start-Windows.bat'))
    check_bat(internal, folder, 'only-bat-from-zip', 'Using the unpacked Windows package', cache)
    with archive.open('rb') as stream:
        archive_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    app = cache / 'packages' / ('windows-' + archive_hash[:16]) / 'XiangqiLab'
    check_bat(app / 'Start-Windows.bat', folder, 'fully-extracted-package', 'App folder:', cache)
    print(f'Logs and isolated extracted package: {folder}', flush=True)


if __name__ == '__main__':
    main()
