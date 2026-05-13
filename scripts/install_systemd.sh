#!/usr/bin/env bash
# Install + start the sglang and rollout systemd services with a single
# command. Idempotent — safe to re-run.
#
# Default behavior:
#   1. Generate unit files with paths from this user's HOME
#   2. Install to /etc/systemd/system/ (sudo)
#   3. Start sglang.service
#   4. Start rollout.service (its ExecStartPre waits for sglang to be ready)
#   5. Print follow-up commands
#
# Usage:
#   bash scripts/install_systemd.sh           # install + start
#   INSTALL_ONLY=1 bash scripts/install_systemd.sh   # install only, no start
#
# Override batch args via env vars:
#   N_TASKS=500 SAMPLES_PER_TASK=4 CONCURRENCY=4 bash scripts/install_systemd.sh

set -euo pipefail

USER_HOME="$HOME"
SC_REPO="${SC_REPO:-$USER_HOME/structured-cot}"
R2E_REPO="${R2E_REPO:-$USER_HOME/R2E-Gym}"
RUN_USER="${RUN_USER:-$(whoami)}"
SYSTEMD_DIR="/etc/systemd/system"
INSTALL_ONLY="${INSTALL_ONLY:-0}"

N_TASKS="${N_TASKS:-200}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-4}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_TURNS="${MAX_TURNS:-30}"
SEED="${SEED:-0}"

echo "[install] config:"
echo "  structured-cot = $SC_REPO"
echo "  R2E-Gym        = $R2E_REPO"
echo "  user           = $RUN_USER"
echo "  batch args     = --n-tasks $N_TASKS --samples-per-task $SAMPLES_PER_TASK \\"
echo "                   --concurrency $CONCURRENCY --max-turns $MAX_TURNS --shuffle --seed $SEED"
echo

for path in "$SC_REPO" "$R2E_REPO"; do
    if [ ! -d "$path" ]; then
        echo "[install] ERROR: $path does not exist"
        exit 1
    fi
done

TMPDIR="$(mktemp -d)"
SGLANG_UNIT="$TMPDIR/sglang.service"
ROLLOUT_UNIT="$TMPDIR/rollout.service"

# bash -lc ensures uv (in ~/.local/bin) is on PATH; systemd's default PATH is
# minimal and would otherwise miss it.

cat > "$SGLANG_UNIT" <<EOF
[Unit]
Description=SGLang inference server (Qwen 3.6 27B, tp=2)
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$SC_REPO
ExecStart=/bin/bash -lc 'cd $SC_REPO && bash run_sglang_server.sh'
Restart=on-failure
RestartSec=15s
TimeoutStartSec=600
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

cat > "$ROLLOUT_UNIT" <<EOF
[Unit]
Description=Grammar-constrained R2E-Gym rollout (overnight SFT generation)
After=sglang.service network-online.target
Requires=sglang.service

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$R2E_REPO
ExecStartPre=/bin/bash -lc 'until curl -s -f http://127.0.0.1:30000/v1/models > /dev/null; do echo "waiting for sglang..."; sleep 5; done'
ExecStart=/bin/bash -lc 'cd $R2E_REPO && uv run python $SC_REPO/scripts/grammar_rollout_batch.py --n-tasks $N_TASKS --samples-per-task $SAMPLES_PER_TASK --concurrency $CONCURRENCY --max-turns $MAX_TURNS --shuffle --seed $SEED'
Restart=no
TimeoutStartSec=14400
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "[install] generated unit files at $TMPDIR/"

if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

$SUDO cp "$SGLANG_UNIT" "$SYSTEMD_DIR/sglang.service"
$SUDO cp "$ROLLOUT_UNIT" "$SYSTEMD_DIR/rollout.service"
$SUDO systemctl daemon-reload

rm -rf "$TMPDIR"

echo "[install] units installed:"
echo "  $SYSTEMD_DIR/sglang.service"
echo "  $SYSTEMD_DIR/rollout.service"
echo

if [ "$INSTALL_ONLY" = "1" ]; then
    echo "[install] INSTALL_ONLY=1 — skipping start. Start later with:"
    echo "  sudo systemctl start sglang.service rollout.service"
    exit 0
fi

# Make sure any panes running these services manually are stopped first.
# It's harmless if they aren't running.
echo "[install] starting services..."
$SUDO systemctl start sglang.service
echo "[install] sglang started (it will take 30-90s to load weights)"

# rollout's ExecStartPre polls /v1/models until sglang is ready, so we can
# start it immediately — systemd holds it in 'activating' until sglang answers
$SUDO systemctl start rollout.service
echo "[install] rollout queued (will activate once sglang answers /v1/models)"
echo

echo "[install] status:"
$SUDO systemctl --no-pager status sglang.service | head -8
echo
$SUDO systemctl --no-pager status rollout.service | head -8
echo

echo "[install] follow logs (Ctrl-C to detach; services keep running):"
echo "  sudo journalctl -u sglang.service -f"
echo "  sudo journalctl -u rollout.service -f"
echo "  watch -n 5 \"cat $SC_REPO/traces/\$(ls -t $SC_REPO/traces/ | head -1)/_status.json | python3 -m json.tool | head -30\""
echo
echo "[install] stop everything:"
echo "  sudo systemctl stop rollout.service sglang.service"
