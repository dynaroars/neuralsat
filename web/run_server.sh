#!/bin/bash
# NeuralSAT Web Server Startup Script
# Usage: ./run_server.sh [port]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PORT="${1:-5000}"

echo "=== NeuralSAT Web Server ==="
echo "Repo:   $REPO_DIR"
echo "Port:   $PORT"

# Activate the NeuralSAT virtual environment
if [ -d "$REPO_DIR/neuralsat_env" ]; then
    echo "Activating neuralsat_env..."
    source "$REPO_DIR/neuralsat_env/bin/activate"
elif [ -d "$HOME/neuralsat_env" ]; then
    echo "Activating ~/neuralsat_env..."
    source "$HOME/neuralsat_env/bin/activate"
else
    echo "[!] No virtual environment found. Using system Python."
fi

# Install web dependencies if needed
pip install -q -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null || true

# Export repo root so server.py can find NeuralSAT
export NEURALSAT_ROOT="$REPO_DIR"

echo "Starting server on port $PORT..."
cd "$SCRIPT_DIR"
exec gunicorn --bind "0.0.0.0:$PORT" --timeout 600 --workers 1 --threads 2 server:app
