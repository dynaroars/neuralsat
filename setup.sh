#!/bin/bash
set -e

if ! command -v uv &> /dev/null; then
    echo "[setup.sh] uv not found, installing via the official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv python list
uv python pin 3.12
uv sync
