#!/bin/bash

# Exhaustive Optimization Launcher
# Usage: bash start_optimization.sh --task <task> --model <model> [options]

set -e

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR/logs"

# Parse arguments to extract task and model for log filename
TASK=""
MODEL=""
FOREGROUND=false

for arg in "$@"; do
    if [ "$prev_arg" = "--task" ]; then
        TASK="$arg"
    elif [ "$prev_arg" = "--model" ]; then
        MODEL="$arg"
    elif [ "$arg" = "--foreground" ]; then
        FOREGROUND=true
    fi
    prev_arg="$arg"
done

# Activate .venv virtual environment if exists and not already activated
if [ -d "$PROJECT_ROOT/.venv" ]; then
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "[INFO] Activating virtual environment: $PROJECT_ROOT/.venv"
        source "$PROJECT_ROOT/.venv/bin/activate"
    else
        echo "[INFO] Virtual environment already activated: $VIRTUAL_ENV"
    fi
else
    echo "[WARNING] No .venv found at $PROJECT_ROOT/.venv, using system Python"
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Generate log filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$TASK" ] && [ -n "$MODEL" ]; then
    LOG_FILE="${LOG_DIR}/opt_${TASK}_${MODEL}_${TIMESTAMP}.log"
else
    LOG_FILE="${LOG_DIR}/opt_${TIMESTAMP}.log"
fi

# Run optimizer
cd "$SCRIPT_DIR"

if [ "$FOREGROUND" = true ]; then
    # Foreground mode
    echo "[INFO] Running in foreground mode"
    python3 -u run_optimization.py "$@"
else
    # Default: background mode with nohup
    echo "============================================"
    echo "🚀 Exhaustive Optimization"
    echo "============================================"
    echo "📋 Task: ${TASK:-auto-detect}"
    echo "🤖 Model: ${MODEL:-auto-detect}"
    echo "📝 Log file: $LOG_FILE"
    echo "🌐 Mode: Background (nohup)"
    echo "============================================"
    echo ""
    
    # Use -u flag for unbuffered output (real-time logging)
    nohup python3 -u run_optimization.py "$@" > "$LOG_FILE" 2>&1 &
    PID=$!
    
    echo "✅ Started successfully (PID: $PID)"
    echo ""
    echo "📌 Useful commands:"
    echo "  View log:    tail -f $LOG_FILE"
    echo "  Check status: ps -p $PID"
    echo "  Stop process: kill $PID"
    echo ""
    echo "💡 To run in foreground, add --foreground flag"
fi
