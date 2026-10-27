#!/bin/bash
cd "$(dirname "$0")" || exit 1
arch="$(uname -m)"
case "$arch" in
  arm64) bundled="runtime/macos-arm64/python/bin/python3"; maximum=13 ;;
  x86_64) bundled="runtime/macos-x64/python/bin/python3"; maximum=12 ;;
  *) printf 'Unsupported CPU architecture.\n'; read -r -p 'Press Enter to close.'; exit 1 ;;
esac
python=""
for candidate in ".venv-models/bin/python" "$bundled" "../.venv/bin/python" python3.12 python3.13 python3.11 python3.10 python3; do
  if "$candidate" -I -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,$maximum)" 2>/dev/null; then
    python="$candidate"
    break
  fi
done
if [ -z "$python" ]; then
  printf 'Install Python 3.12 from python.org, then open this launcher again.\n'
  read -r -p 'Press Enter to close.'
  exit 1
fi
unset PYTHONHOME PYTHONPATH
"$python" -E -s -X utf8 start.py "$@"
status=$?
if [ "$status" -ne 0 ]; then
  printf '\nStartup failed. Read the error above.\n'
  read -r -p 'Press Enter to close.'
fi
exit "$status"
