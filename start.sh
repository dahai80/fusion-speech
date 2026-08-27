#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# fusion monorepo shares one venv at the repo root (per CLAUDE.md).
VENV_DIR="${FUSION_VENV_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)/.venv}"
PID_FILE="$SCRIPT_DIR/.fusion-speech.pid"
LOG_DIR="${FUSION_SPEECH_LOG_DIR:-$HOME/.fusion-speech/logs}"
BOOT_LOG="$LOG_DIR/boot.log"
ENTRY="python3 -m fusion_speech.server"
SOCK_DIR="${FUSION_SOCKET_DIR:-$HOME/.fusion-speech/run}"
SOCK_PATH="$SOCK_DIR/fusion-speech.sock"
DEFAULT_PORT="${FUSION_SPEECH_PORT:-11465}"
export FUSION_SPEECH_PORT="$DEFAULT_PORT"
mkdir -p "$SOCK_DIR" && chmod 700 "$SOCK_DIR" 2>/dev/null || true

mkdir -p "$LOG_DIR"

# process-identity check: live PID is ours only if cmdline references our daemon
# entry. prevents stale-PID reuse -> false "already running".
_pid_is_daemon() {
    local pid="$1"
    local cmd
    cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    [ -n "$cmd" ] || return 1
    case "$cmd" in
        *fusion_speech.server*) return 0 ;;
        *) return 1 ;;
    esac
}

is_running() {
    [ -f "$PID_FILE" ] || return 1
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    _pid_is_daemon "$pid"
}

do_start() {
    if is_running; then
        echo "fusion-speech already running (pid $(cat "$PID_FILE"))"
        return 0
    fi
    rm -f "$PID_FILE"
    if [ -d "$VENV_DIR" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    fi
    # mkdir-based atomic lock guards against two start invocations racing to spawn.
    if ! mkdir "$PID_FILE.lock" 2>/dev/null; then
        echo "fusion-speech start lock held by another process, aborting" >&2
        return 1
    fi
    rm -f "$SOCK_PATH"
    nohup $ENTRY >> "$BOOT_LOG" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    sleep 1
    if is_running; then
        echo "fusion-speech started (pid $pid, port $DEFAULT_PORT, sock $SOCK_PATH)"
    else
        echo "fusion-speech failed to start, see $BOOT_LOG" >&2
        rm -f "$PID_FILE"
        rmdir "$PID_FILE.lock" 2>/dev/null || true
        return 1
    fi
    rmdir "$PID_FILE.lock" 2>/dev/null || true
}

do_stop() {
    if ! is_running; then
        echo "fusion-speech not running"
        rm -f "$PID_FILE"
        return 0
    fi
    local pid
    pid="$(cat "$PID_FILE")"
    kill "$pid" 2>/dev/null || true
    # E8: 15s grace before SIGKILL — accommodates R6 bounded drain (10s) plus
    # fusion-mlx response latency. 30 polls x 0.5s.
    for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE" "$SOCK_PATH"
    rmdir "$PID_FILE.lock" 2>/dev/null || true
    echo "fusion-speech stopped"
}

do_status() {
    if is_running; then
        echo "fusion-speech running (pid $(cat "$PID_FILE"), port $DEFAULT_PORT, sock $SOCK_PATH)"
    else
        echo "fusion-speech stopped"
        return 1
    fi
}

do_doctor() {
    local ok=1
    echo "== fusion-speech doctor =="
    if is_running; then
        echo "[ok] daemon running (pid $(cat "$PID_FILE"))"
    else
        echo "[fail] daemon not running"
        ok=0
    fi
    if [ -d "$VENV_DIR" ]; then
        echo "[ok] venv at $VENV_DIR"
    else
        echo "[fail] venv missing at $VENV_DIR"
        ok=0
    fi
    if source "$VENV_DIR/bin/activate" 2>/dev/null && python3 -c "import fusion_core, fastapi, httpx" 2>/dev/null; then
        echo "[ok] core deps importable (fusion_core/fastapi/httpx)"
    else
        echo "[fail] core deps missing"
        ok=0
    fi
    if python3 -c "import sounddevice" 2>/dev/null; then
        echo "[ok] sounddevice importable"
    else
        echo "[warn] sounddevice missing (capture disabled until M2)"
    fi
    if curl -sf "http://127.0.0.1:${DEFAULT_PORT}/health" >/dev/null 2>&1; then
        echo "[ok] /health reachable on $DEFAULT_PORT"
    else
        echo "[warn] /health not reachable on $DEFAULT_PORT (daemon down or still booting)"
    fi
    if curl -sf "http://127.0.0.1:11434/v1/models" >/dev/null 2>&1; then
        echo "[ok] fusion-mlx reachable on 11434"
    else
        echo "[warn] fusion-mlx not reachable on 11434 (start with ~/claude-home/fusion-mlx/start.sh start)"
    fi
    [ "$ok" -eq 1 ]
}

do_log() {
    local follow="${1:-}"
    if [ "$follow" = "-f" ] || [ "$follow" = "--follow" ]; then
        tail -f "$LOG_DIR"/daemon.log "$BOOT_LOG" 2>/dev/null || tail -f "$BOOT_LOG"
    else
        tail -n 100 "$LOG_DIR"/daemon.log "$BOOT_LOG" 2>/dev/null || tail -n 100 "$BOOT_LOG"
    fi
}

case "${1:-status}" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; do_start ;;
    status)  do_status ;;
    doctor)  do_doctor ;;
    log)     shift; do_log "$@" ;;
    *) echo "Usage: $0 {start|stop|restart|status|doctor|log [-f]}" >&2; exit 1 ;;
esac
