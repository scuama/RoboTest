#!/bin/bash

# 优化实验启动脚本
# 使用nohup后台运行，记录实时日志

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${SCRIPT_DIR}"
SCRIPT="${WORKSPACE}/run_optimization.py"
LOG_DIR="${WORKSPACE}/logs"
PID_FILE="${LOG_DIR}/optimization_experiment.pid"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/optimization_experiment_${TIMESTAMP}.log"

echo "======================================"
echo "启动优化实验 - 后台运行模式"
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
NUM_SCENES=${1:-100}
NUM_DISTRACTORS=${2:-1}
MAX_GUIDED_ATTEMPTS=${3:-3}
SEED=${4:-42}

echo "实验参数:"
echo "  场景数量:         ${NUM_SCENES}"
echo "  干扰物数量:       ${NUM_DISTRACTORS}"
echo "  Guided最多尝试:   ${MAX_GUIDED_ATTEMPTS}"
echo "  随机种子:         ${SEED}"
echo ""

# 切换到工作目录
cd "${WORKSPACE}"

# 激活conda环境（需要用户根据自己的环境修改）
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate env_isaaclab

# 启动后台任务
nohup stdbuf -oL -eL python3 "${SCRIPT}" \
    --config-dir configs \
    --num-scenes ${NUM_SCENES} \
    --num-distractors ${NUM_DISTRACTORS} \
    --max-guided-attempts ${MAX_GUIDED_ATTEMPTS} \
    --checkpoint-dir ${CHECKPOINT_DIR:-./checkpoints} \
    --output-dir ./optimization_results \
    --seed ${SEED} \
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
echo "  或运行: pkill -f run_optimization.py"
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
