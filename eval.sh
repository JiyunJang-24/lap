#!/usr/bin/env bash
set -e

SESSION_NAME="lap_libero"

# 이미 같은 세션이 있으면 종료
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$SESSION_NAME"
fi

# 현재 작업 디렉토리 저장
WORKDIR="$(pwd)"

# 새 tmux 세션 생성
tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"

# pane 0: policy server 실행
tmux send-keys -t "$SESSION_NAME":0.0 \
"JAX_PLATFORMS=cuda uv run --group cuda --active scripts/serve_policy.py policy:checkpoint --policy.config=lap_libero --policy.dir=checkpoints/lap --policy.type=dual" C-m #dual mode : LAP + LAP_AR로 demo, Action, 실제 Language Output 생성

# 오른쪽 또는 아래로 pane 분할
tmux split-window -h -t "$SESSION_NAME":0 -c "$WORKDIR"

# pane 1: sim 실행
tmux send-keys -t "$SESSION_NAME":0.1 \
"source $WORKDIR/scripts/libero/.venv/bin/activate && export LIBERO_CONFIG_PATH=$WORKDIR/third_party/openpi/third_party/libero && export PYTHONPATH=\$PYTHONPATH:$WORKDIR/third_party/openpi/third_party/libero && python scripts/libero/main_custom.py" C-m

# 레이아웃 정리
tmux select-layout -t "$SESSION_NAME" tiled

# attach
tmux attach-session -t "$SESSION_NAME"