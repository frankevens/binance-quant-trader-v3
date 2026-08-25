#!/bin/bash
# Binance Quant Trader V2 - Start Script
# ========================================
# Usage:
#   ./start.sh          # Start in foreground
#   ./start.sh --bg     # Start in background
#   ./start.sh --stop   # Stop background process
#   ./start.sh --status # Check if running

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/trader_data/trader.pid"
LOG_FILE="$SCRIPT_DIR/trader_data/trader.log"

mkdir -p "$SCRIPT_DIR/trader_data"

# Load .env if exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

start_foreground() {
    echo "Starting Binance Quant Trader V2 (foreground)..."
    cd "$SCRIPT_DIR"
    python main.py "$@"
}

start_background() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Trader already running (PID: $PID)"
            exit 1
        fi
    fi

    echo "Starting Binance Quant Trader V2 (background)..."
    cd "$SCRIPT_DIR"
    nohup python main.py >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Started with PID: $(cat $PID_FILE)"
    echo "Log: $LOG_FILE"
}

stop_trader() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No PID file found. Trader may not be running."
        exit 0
    fi

    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping trader (PID: $PID)..."
        kill "$PID"
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            echo "Force killing..."
            kill -9 "$PID"
        fi
        rm -f "$PID_FILE"
        echo "Stopped."
    else
        echo "Process $PID not running. Cleaning up PID file."
        rm -f "$PID_FILE"
    fi
}

check_status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Trader is running (PID: $PID)"
            echo "Log: $LOG_FILE"
            echo ""
            echo "=== Last 10 log lines ==="
            tail -n 10 "$LOG_FILE" 2>/dev/null || echo "(no log yet)"
        else
            echo "PID file exists but process $PID is not running."
        fi
    else
        echo "Trader is not running."
    fi
}

case "${1:-}" in
    --bg|background)
        start_background
        ;;
    --stop|stop)
        stop_trader
        ;;
    --status|status)
        check_status
        ;;
    *)
        start_foreground "$@"
        ;;
esac
