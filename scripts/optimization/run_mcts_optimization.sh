#!/bin/bash

# MCTS with Two-Stage Optimization Runner
# 每次变异后自动运行baseline和guided优化
# 使用nohup后台运行，实时记录日志
# 
# 使用方法：
#   bash experiments/obstruction/run_mcts_with_optimization.sh [NUM_FAILURES]
# 
# 示例：
#   收集80个失败案例（默认）:
#     bash experiments/obstruction/run_mcts_with_optimization.sh
#   
#   收集100个失败案例:
#     bash experiments/obstruction/run_mcts_with_optimization.sh 100
#   
#   测试模式（收集1个失败案例）:
#     bash experiments/obstruction/run_mcts_with_optimization.sh 1

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${SCRIPT_DIR}"
LOG_DIR="${WORKSPACE}/logs"
PID_FILE="${LOG_DIR}/mcts_optimization.pid"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 获取失败案例数（默认80）
NUM_FAILURES=${1:-80}

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="fuzzer_output/mcts_optimized_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/mcts_optimization_${TIMESTAMP}.log"

# 检查是否已有MCTS实验在运行
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if ps -p "${OLD_PID}" > /dev/null 2>&1; then
        echo -e "\033[0;31m错误: 已有MCTS实验正在运行 (PID: ${OLD_PID})\033[0m"
        echo "请先停止正在运行的实验，或等待其完成"
        echo "停止命令: kill ${OLD_PID}"
        exit 1
    else
        echo "清理旧的PID文件..."
        rm -f "${PID_FILE}"
    fi
fi

echo "======================================"
echo "MCTS with Two-Stage Optimization"
echo "======================================"
echo "目标失败案例数: ${NUM_FAILURES}"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志文件: ${LOG_FILE}"
echo "PID文件: ${PID_FILE}"
echo ""
echo "启动后台运行..."
echo "======================================"

# 创建后台运行脚本
RUNNER_SCRIPT="${LOG_DIR}/runner_mcts_${TIMESTAMP}.sh"
cat > "${RUNNER_SCRIPT}" << 'RUNNER_EOF'
#!/bin/bash

# 激活conda环境（需要用户根据自己的环境修改）
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate env_isaaclab

echo "=================================="
echo "MCTS with Two-Stage Optimization"
echo "=================================="
echo "启动时间: $(date)"
echo "Python: $(which python3)"
echo "目标失败案例数: NUM_FAILURES_PLACEHOLDER"
echo "=================================="
echo ""

cd WORKSPACE_PLACEHOLDER

stdbuf -oL -eL python3 ../fuzzing/run_mcts_fuzzer.py \
    --task-suite libero_90 \
    --task-type put-in \
    --checkpoint-dir ${CHECKPOINT_DIR:-./checkpoints} \
    --energy 1 \
    --max-iter 1000 \
    --output OUTPUT_DIR_PLACEHOLDER \
    --seed 42 \
    --max-steps 300 \
    --strategies "expand_random,rephrase" \
    --num-failures NUM_FAILURES_PLACEHOLDER \
    --verbose

echo ""
echo "=================================="
echo "Run completed!"
echo "完成时间: $(date)"
echo "Results saved to: OUTPUT_DIR_PLACEHOLDER"
echo "Check final_summary.json for statistics"
echo "=================================="
RUNNER_EOF

# 替换占位符
sed -i "s|WORKSPACE_PLACEHOLDER|${WORKSPACE}|g" "${RUNNER_SCRIPT}"
sed -i "s|OUTPUT_DIR_PLACEHOLDER|${OUTPUT_DIR}|g" "${RUNNER_SCRIPT}"
sed -i "s|NUM_FAILURES_PLACEHOLDER|${NUM_FAILURES}|g" "${RUNNER_SCRIPT}"

# 添加执行权限
chmod +x "${RUNNER_SCRIPT}"

# 使用nohup后台运行
nohup bash "${RUNNER_SCRIPT}" > "${LOG_FILE}" 2>&1 &
RUNNER_PID=$!

# 保存PID
echo ${RUNNER_PID} > "${PID_FILE}"

echo "✓ 后台任务已启动"
echo "  PID: ${RUNNER_PID}"
echo ""
echo "监控命令:"
echo "  查看日志: tail -f ${LOG_FILE}"
echo "  查看进程: ps -p ${RUNNER_PID}"
echo "  停止实验: kill ${RUNNER_PID}"
echo ""
echo "======================================"
