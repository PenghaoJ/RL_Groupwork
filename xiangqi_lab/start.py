"""Prepare a local CPU inference environment, then open the website. Never train."""
import os
from pathlib import Path
import platform
import subprocess
import sys

APP = Path(__file__).resolve().parent
ENV = APP / ".venv-models"
PROBE = "import torch, numpy, gymnasium, sb3_contrib; print('XIANGQI_MODELS_READY')"


def runtime_python(folder):
    return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ready(python):
    if not Path(python).is_file():
        return False
    try:
        result = subprocess.run([str(python), "-I", "-c", PROBE],
                                capture_output=True, text=True, timeout=90)
        return result.returncode == 0 and "XIANGQI_MODELS_READY" in result.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def install_commands(python, intel_mac):
    pip = [str(python), "-I", "-m", "pip", "install", "--only-binary=:all:"]
    if intel_mac:
        # Intel Mac wheels stopped at torch 2.2.2 / Python 3.12.
        return [pip + ["--index-url", "https://pypi.org/simple", "torch==2.2.2",
                       "numpy==1.26.4", "gymnasium==0.29.1",
                       "stable-baselines3==2.3.2", "sb3-contrib==2.3.0"]]
    torch_index = "https://pypi.org/simple" if sys.platform == "darwin" else "https://download.pytorch.org/whl/cpu"
    return [pip + ["--index-url", torch_index, "torch>=2.6,<3"],
            pip + ["--index-url", "https://pypi.org/simple", "numpy==2.2.6",
                   "gymnasium==1.2.1", "stable-baselines3==2.7.0", "sb3-contrib==2.7.0"]]


def prepare(allow_install=True):
    for candidate in (runtime_python(ENV), Path(sys.executable)):
        if ready(candidate):
            return candidate
    if not allow_install:
        raise RuntimeError("Model dependencies are missing. Run the normal Start launcher with internet access once.")
    intel_mac = sys.platform == "darwin" and platform.machine() == "x86_64"
    maximum = 12 if intel_mac else 13
    if not (3, 10) <= sys.version_info[:2] <= (3, maximum):
        raise RuntimeError(f"Use Python 3.10-3.{maximum} for these models on this computer.")
    python = runtime_python(ENV)
    print(f"First-time model setup in {ENV}\nInternet access is required. This may take several minutes.", flush=True)
    print("Only this local environment is changed; no training is started.", flush=True)
    # Re-running venv also repairs an interrupted first setup with missing pip.
    subprocess.run([sys.executable, "-I", "-m", "venv", str(ENV)], check=True)
    for command in install_commands(python, intel_mac):
        subprocess.run(command, check=True)
    if not ready(python):
        raise RuntimeError("Model setup did not pass the import check. Keep this window open and share its error message.")
    return python


def main():
    args = sys.argv[1:]
    check_only = "--check-models" in args
    setup_only = "--setup-models" in args
    if check_only:
        args.remove("--check-models")
    if setup_only:
        args.remove("--setup-models")
    try:
        python = prepare(allow_install=not check_only)
        print(f"Model Python: {python}", flush=True)
        if check_only or setup_only:
            return 0
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        env["PYTHONUTF8"] = "1"
        engine = APP / "engine" / "Pikafish-MacOS-universal"
        if sys.platform == "darwin" and engine.exists():
            engine.chmod(engine.stat().st_mode | 0o111)
        return subprocess.call([str(python), "-E", "-s", "-X", "utf8", str(APP / "app.py"), *args], cwd=APP, env=env)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Could not start: {error}\nFix the error and run Start again. No models were changed.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
