#!/bin/bash
# Invoked as a forced SSH command (see authorized_keys on the deploy host) by
# the "Deploy backend to taco" GitHub Actions workflow. Pulls the latest
# develop branch and restarts the neuralsat-backend systemd service, which
# is permitted passwordlessly for the deploying user via a scoped sudoers
# NOPASSWD rule (sudo -l shows exactly which commands are allowed).
set -e
cd "$(dirname "$0")/.."
git pull origin develop
sudo systemctl restart neuralsat-backend
echo "[deploy] neuralsat-backend restarted at $(date)"
