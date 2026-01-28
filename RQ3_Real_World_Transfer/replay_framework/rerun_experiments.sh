#!/bin/bash

# 重新运行已保存的Guided组配置 - 后台运行模式

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${SCRIPT_DIR}"
SCRIPT="${WORKSPACE}/rerun_experiments.py"
LOG_DIR="${WORKSPACE}/logs"
PID_FILE="${LOG_DIR}/rerun_experiment.pid"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/rerun_experiment_${TIMESTAMP}.log"

echo "======================================"
echo "重新运行Guided组实验 - 后台运行模式"
echo "======================================"
echo "主日志文件: ${MAIN_LOG}"
echo "PID文件: ${PID_FILE}"
echo ""

# 检查是否已有实验在运行
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if ps -p "${OLD_PID}" > /dev/null 2>&1; then
        echo -e "\033[0;31m错误: 已有实验正在运行 (PID: ${OLD_PID})\033[0m"
        echo "请先停止正在运行的实验，或等待其完成"
        exit 1
    else
        echo "清理旧的PID文件..."
        rm -f "${PID_FILE}"
    fi
fi

# 默认参数
SCENES=${1:-"--all"}  # 默认运行所有场景
CHECKPOINT_DIR=${2:-"./checkpoints"}

if [[ "$SCENES" != "--all" ]] && [[ "$SCENES" != --* ]]; then
    SCENES="--scenes $SCENES"
fi

echo "实验参数:"
echo "  场景选择:         ${SCENES}"
echo "  模型路径:         ${CHECKPOINT_DIR}"
echo ""

# 切换到工作目录
cd "${WORKSPACE}"

# 激活conda环境（需要用户根据自己的环境修改）
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate env_isaaclab

# 启动后台任务
nohup stdbuf -oL -eL python3 "${SCRIPT}" \
    ${SCENES} \
    --config-dir ./saved_configs \
    --output-dir ./new_results \
    --checkpoint-dir ${CHECKPOINT_DIR} \
    > "${MAIN_LOG}" 2>&1 &

# 记录PID
MAIN_PID=$!
echo ${MAIN_PID} > "${PID_FILE}"

echo "实验已启动 (PID: ${MAIN_PID})"
echo ""
echo "查看实时日志:"
echo "  tail -f ${MAIN_LOG}"
echo ""
echo "停止实验:"
echo "  kill ${MAIN_PID}"
echo "  或运行: pkill -f rerun_experiments.py"
echo ""
echo "查看进程状态:"
echo "  ps -p ${MAIN_PID}"
echo "======================================"

# 等待2秒，检查进程是否正常启动
sleep 2

if ps -p ${MAIN_PID} > /dev/null 2>&1; then
    echo -e "\033[0;32m✓ 实验进程正常运行\033[0m"
    echo ""
    echo "显示最新日志（按Ctrl+C退出查看，不会停止实验）:"
    echo ""
    tail -f "${MAIN_LOG}"
else
    echo -e "\033[0;31m✗ 实验进程启动失败\033[0m"
    echo "请检查日志文件: ${MAIN_LOG}"
    rm -f "${PID_FILE}"
    exit 1
fi
