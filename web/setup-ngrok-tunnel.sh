#!/bin/bash
# One-time (idempotent) setup for NeuralSAT's dedicated ngrok tunnel.
#
# Migrates neuralsat-ngrok-tunnel.service off the domain previously shared
# with the unrelated "CS Scheduler" service (oarless-chafflike-chung.ngrok
# -free.dev, fanned out via nginx) onto its own dedicated domain
# (shingle-unhinge-concert.ngrok-free.dev), tunneled directly to gunicorn
# on port 5050 -- matching dynaroars/dig's dig-ngrok-tunnel.service, which
# tunnels straight to its backend with no nginx in the path. See
# ANALYSIS.md Sec 9 for the full history.
#
# The new domain lives under a *different* ngrok account than whatever
# else webapp runs, so (also mirroring dig-ngrok-tunnel.service, which
# uses its own --config=/home/webapp/ngrok-dig.yml for the same reason)
# the tunnel authenticates via its own config file,
# /home/webapp/ngrok-neuralsat.yml, instead of webapp's default ngrok
# config -- so setting this up never touches or overwrites whatever
# account/token webapp's other tunnel(s) rely on.
#
# Run this ON taco, as root, from a checkout of this repo at (or past) the
# commit that introduced it:
#   sudo ./web/setup-ngrok-tunnel.sh
#
# It only installs/restarts NeuralSAT's own systemd unit. It does not
# touch nginx or CS Scheduler -- that service's continued public access
# after losing the shared tunnel is a separate task, not handled here.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Must be run as root (sudo ./web/setup-ngrok-tunnel.sh)." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNIT_NAME="neuralsat-ngrok-tunnel.service"
UNIT_SRC="$SCRIPT_DIR/$UNIT_NAME"
UNIT_DST="/etc/systemd/system/$UNIT_NAME"
DOMAIN="shingle-unhinge-concert.ngrok-free.dev"
NGROK_CONFIG="/home/webapp/ngrok-neuralsat.yml"

if [ ! -f "$UNIT_SRC" ]; then
    echo "Expected $UNIT_SRC -- run this from a checkout of the repo, not a copied script." >&2
    exit 1
fi

echo "== NeuralSAT ngrok tunnel setup =="

echo "-> Checking ngrok config at $NGROK_CONFIG (webapp)..."
if ! sudo -u webapp ngrok config check --config "$NGROK_CONFIG" >/dev/null 2>&1; then
    echo "[!] No valid ngrok config found at $NGROK_CONFIG for 'webapp'." >&2
    echo "    This domain was reserved under a separate ngrok account, so it" >&2
    echo "    needs its own authtoken in its own config file (not webapp's" >&2
    echo "    default one, which other tunnels may still depend on). Get the" >&2
    echo "    authtoken from that account's dashboard.ngrok.com, then run:" >&2
    echo "      sudo -u webapp ngrok config add-authtoken <TOKEN> --config $NGROK_CONFIG" >&2
    echo "    Continuing anyway -- the service restart below will fail" >&2
    echo "    (and keep retrying via Restart=always) if this isn't set up." >&2
fi

echo "-> Installing $UNIT_DST from $UNIT_SRC"
install -m 0644 "$UNIT_SRC" "$UNIT_DST"

echo "-> Reloading systemd and (re)starting $UNIT_NAME"
systemctl daemon-reload
systemctl enable "$UNIT_NAME"
systemctl restart "$UNIT_NAME"

sleep 2
if systemctl is-active --quiet "$UNIT_NAME"; then
    echo "-> $UNIT_NAME is active."
else
    echo "[!] $UNIT_NAME failed to start. Check: journalctl -u $UNIT_NAME -n 50" >&2
    exit 1
fi

cat <<EOF

Done. Verify from outside taco with:
  curl -s https://$DOMAIN/api/health

Reminder: CS Scheduler previously rode on the shared tunnel/nginx setup
that this unit used to run. Nothing here touches nginx or CS Scheduler --
sort out its continued public access separately.
EOF
