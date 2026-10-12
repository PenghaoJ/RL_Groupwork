"""Regression checks for Windows package recovery, without requiring Pester."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'tools/windows_launcher.ps1'
HAS_SOURCE = SOURCE.is_file()


@unittest.skipUnless(HAS_SOURCE, 'Launcher source checks run from the repository')
class GeneratedLaunchers(unittest.TestCase):
    def test_standalone_launchers_are_current(self):
        sys.path.insert(0, str(ROOT / 'tools'))
        try:
            from build_launchers import TARGETS, launcher_bytes
            for name in TARGETS:
                with self.subTest(name=name):
                    self.assertEqual((ROOT / name).read_bytes(), launcher_bytes())
        finally:
            sys.path.pop(0)


@unittest.skipUnless(os.name == 'nt' and HAS_SOURCE, 'Windows source checkout required')
class WindowsArchiveRecovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='xiangqi launcher ')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    @staticmethod
    def quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    def run_ps(self, body):
        script = self.folder / 'check.ps1'
        script.write_text(
            "$ErrorActionPreference='Stop'\n. " + self.quote(SOURCE) + ' -LibraryOnly\n' + body,
            encoding='utf-8-sig',
        )
        result = subprocess.run(
            ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', str(script)],
            capture_output=True, text=True, errors='replace', timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def make_zip(self, name='XiangqiLab-Windows-v1.2.0.zip', extra=None, omit=None, platform='Windows'):
        files = [
            'app.py', 'rules.py', 'engines.py', 'model_registry.py', 'model_worker.py',
            'policy_process.py', 'static/index.html', 'static/app.js', 'static/style.css',
            'engine/Pikafish-Windows-x86-64-universal.exe', 'engine/pikafish.nnue',
            'runtime/python/python.exe', 'runtime/python/python313.dll',
            'runtime/python/Lib/encodings/__init__.py',
        ]
        path = self.folder / name
        with zipfile.ZipFile(path, 'w') as archive:
            for entry in files:
                if entry != omit:
                    archive.writestr('XiangqiLab/' + entry, 'fixture')
            archive.writestr('XiangqiLab/release.json', json.dumps({'platform': platform, 'version': '1.2.0', 'offline': True}))
            if extra:
                archive.writestr(*extra)
        return path

    def validate(self, archive, expected=None):
        assertion = "if ($meta.version -ne '1.2.0') { throw 'Wrong metadata' }"
        operation = '$meta = Test-ReleaseArchive $zip ' + self.quote(self.folder / 'target') + '\n' + assertion
        if expected:
            operation = (
                "$rejected=$false\ntry { " + operation + " } catch {\n"
                "if ($_.Exception.Message -notlike " + self.quote('*' + expected + '*') + ") { throw }\n"
                "$rejected=$true\n}\nif (-not $rejected) { throw 'Unsafe archive was accepted' }"
            )
        self.run_ps('$zip=[IO.Compression.ZipFile]::OpenRead(' + self.quote(archive) + ')\ntry {\n' + operation + '\n} finally { $zip.Dispose() }')

    def test_accepts_complete_windows_layout(self):
        self.validate(self.make_zip())

    def test_rejects_source_or_missing_runtime(self):
        self.validate(self.make_zip(omit='runtime/python/python.exe'), 'missing runtime/python/python.exe')

    def test_rejects_wrong_platform(self):
        self.validate(self.make_zip(platform='macOS'), 'Windows offline ZIP')

    def test_rejects_traversal_before_writing_files(self):
        archive = self.make_zip(extra=('XiangqiLab/../../escaped.txt', 'never write this'))
        cache = self.folder / 'packages'
        self.run_ps(
            "$rejected=$false\ntry { Expand-Release " + self.quote(archive) + ' ' + self.quote(cache) + " } catch {\n"
            "if ($_.Exception.Message -notlike '*Unsafe ZIP path*') { throw }; $rejected=$true\n}\n"
            "if (-not $rejected) { throw 'Traversal accepted' }"
        )
        self.assertFalse((self.folder / 'escaped.txt').exists())
        self.assertEqual(list(cache.iterdir()), [])

    def test_rejects_case_insensitive_duplicate(self):
        self.validate(self.make_zip(extra=('XiangqiLab/APP.PY', 'duplicate')), 'Duplicate paths')

    def test_rejects_links(self):
        info = zipfile.ZipInfo('XiangqiLab/link')
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        self.validate(self.make_zip(extra=(info, '../../elsewhere')), 'Links are not supported')

    def test_cleans_own_stage_after_python_failure(self):
        archive = self.make_zip()
        cache = self.folder / 'packages'
        cache.mkdir()
        keep = cache / 'my-model.txt'
        keep.write_text('keep this')
        self.run_ps(
            'function Test-Python([string]$Python) { return $false }\n'
            "$rejected=$false\ntry { Expand-Release " + self.quote(archive) + ' ' + self.quote(cache) + " } catch {\n"
            "if ($_.Exception.Message -notlike '*failed the Python check*') { throw }; $rejected=$true\n}\n"
            "if (-not $rejected) { throw 'Broken runtime accepted' }"
        )
        self.assertEqual(list(cache.iterdir()), [keep])
        self.assertEqual(keep.read_text(), 'keep this')

    def test_search_is_bounded_and_prefers_new_version(self):
        newer = self.make_zip(name='XiangqiLab-Windows-v1.10.0.zip')
        older = self.make_zip(name='XiangqiLab-Windows-v1.2.0.zip')
        nested = self.folder / 'unrelated'
        nested.mkdir()
        (nested / 'XiangqiLab-Windows-v99.0.0.zip').write_bytes(b'not searched')
        output = self.run_ps('@(Find-ReleaseArchives @(' + self.quote(self.folder) + ')) | ConvertTo-Json -Compress')
        self.assertEqual(json.loads(output), [str(newer), str(older)])

    def test_adjacent_package_wins_over_other_downloads(self):
        adjacent = self.make_zip()
        downloads = self.folder / 'downloads'
        downloads.mkdir()
        unrelated = downloads / 'XiangqiLab-Windows-v99.0.0.zip'
        unrelated.write_bytes(b'not the intended package')
        output = self.run_ps('@(Find-ReleaseArchives @(' + self.quote(self.folder) + ',' + self.quote(downloads) + ')) | ConvertTo-Json -Compress')
        self.assertEqual(json.loads(output), [str(adjacent), str(unrelated)])


if __name__ == '__main__':
    unittest.main()
