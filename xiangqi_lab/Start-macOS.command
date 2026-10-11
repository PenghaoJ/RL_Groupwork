#!/bin/bash
set -e
cd "$(dirname "$0")"
case "$(uname -m)" in
  arm64) runtime="runtime/macos-arm64/python/bin/python3" ;;
  x86_64) runtime="runtime/macos-x64/python/bin/python3" ;;
  *) printf 'Unsupported CPU architecture.\n'; exit 1 ;;
esac
if [ ! -f "$runtime" ]; then
  printf 'Please extract the complete macOS release ZIP first.\n'
  read -r -p 'Press Enter to close.'
  exit 1
fi
chmod +x "$runtime" engine/Pikafish-MacOS-universal
exec "$runtime" -E -X utf8 app.py
