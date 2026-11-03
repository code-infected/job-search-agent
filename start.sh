#!/bin/sh
# Sets up the agent on first use, then starts it and opens it in the browser.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "First run: creating a Python environment. This takes a few minutes."
    python3.14 -m venv .venv || python3 -m venv .venv || {
        echo "Python 3.14 is needed. Install it from https://www.python.org/downloads/ and run this again."
        exit 1
    }
fi
.venv/bin/python -m pip install --quiet --disable-pip-version-check -r requirements.lock
[ -f .env ] || cp .env.example .env
exec .venv/bin/python app.py "$@"
