#!/usr/bin/env bash
set -u

APP_DIR="/home/fontaine-decal/Documents/Final_Pipeline"
LOG_FILE="$APP_DIR/terminal.log"
SERVER_PATTERN="uvicorn app.gui:app"

{
    echo "Force restart requested at $(date)"
    sleep 2

    if command -v pkill >/dev/null 2>&1; then
        pkill -TERM -f "$SERVER_PATTERN" || true
        sleep 1
        pkill -KILL -f "$SERVER_PATTERN" || true
    fi

    cd "$APP_DIR" || exit 1
    exec /bin/bash "$APP_DIR/startup.sh"
} >> "$LOG_FILE" 2>&1
