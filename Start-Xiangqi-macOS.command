#!/bin/bash
cd "$(dirname "$0")" || exit 1
exec /bin/bash xiangqi_lab/Start-macOS.command "$@"
