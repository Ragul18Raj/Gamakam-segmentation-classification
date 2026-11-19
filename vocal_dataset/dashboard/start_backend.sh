#!/bin/sh
# Start the local analysis service used by dashboard/index.html.
cd "$(dirname "$0")/.." || exit 1
exec /usr/bin/python3 dashboard/server.py
