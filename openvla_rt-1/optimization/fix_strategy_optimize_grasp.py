#!/usr/bin/env python3
"""
自动寻找成功抓取配置的工具
功能：在指定方向上搜索能成功抓取的物体位置配置
"""

import os
import json
import random
import subprocess
import argparse
import numpy as np
from pathlib import Path


# ==================== 默认配置参数 ====================
# 使用相对路径获取项目根目录
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# 虚拟环境 Python（如果存在）
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python3")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = "python3"  # 回退到系统 Python

DEFAULT_BASE_DIR = "newresult/t-grasp_n-100_o-0_s-170912623/openvla-7b_2024/t-grasp_n-100_o-0_s-170912623/openvla-7b_2024"
DEFAULT_TASK_NAME = "google_robot_pick_customizable"
DEFAULT_TASK_TYPE = "grasp"
# 使用同目录下的 replay_vla_actions.py
REPLAY_SCRIPT = str(SCRIPT_DIR / "replay_vla_actions.py")

# 成功判断标准
MIN_CONSECUTIVE_GRASP_STEPS = 5  # 连续抓取的最小步数

# 搜索参数
DEFAULT_TOTAL_ATTEMPTS = 20  # 总搜索次数（粗搜索+精细搜索）
COARSE_FINE_RATIO = 0.5  # 粗搜索占比，剩余为精细搜索

# 方向定义 (机械臂视角)
# 左(left): x负方向, 右(right): x正方向
# 上(up): y负方向, 下(down): y正方向
# 范围是相对原始位置的偏移量
DIRECTION_OFFSETS = {
    "left": {"x": (-0.015, -0.003), "y": (-0.003, 0.003)},    # 左：x负向0.3-1.5cm，y±0.3cm
    "right": {"x": (0.003, 0.015), "y": (-0.003, 0.003)},     # 右：x正向0.3-1.5cm，y±0.3cm
    "up": {"x": (-0.003, 0.003), "y": (-0.015, -0.003)},      # 上：y负向0.3-1.5cm，x±0.3cm
    "down": {"x": (-0.003, 0.003), "y": (0.003, 0.015)},      # 下：y正向0.3-1.5cm，x±0.3cm
    "left-up": {"x": (-0.015, -0.003), "y": (-0.015, -0.003)}, # 左上：x负向0.3-1.5cm，y负向0.3-1.5cm
    "left-down": {"x": (-0.015, -0.003), "y": (0.003, 0.015)}, # 左下：x负向0.3-1.5cm，y正向0.3-1.5cm
    "right-up": {"x": (0.003, 0.015), "y": (-0.015, -0.003)},  # 右上：x正向0.3-1.5cm，y负向0.3-1.5cm
    "right-down": {"x": (0.003, 0.015), "y": (0.003, 0.015)},  # 右下：x正向0.3-1.5cm，y正向0.3-1.5cm
}

# 精细搜索范围（在粗搜索最佳点附近）
FINE_SEARCH_RANGE = {"x": (-0.003, 0.003), "y": (-0.003, 0.003)}


# ==================== 工具函数 ====================

def load_options(episode_dir):
    """加载 options.json"""
    options_path = os.path.join(episode_dir, "options.json")
    with open(options_path, 'r') as f:
        return json.load(f)


def save_options(episode_dir, options):
    """保存 options.json"""
    options_path = os.path.join(episode_dir, "options.json")
    with open(options_path, 'w') as f:
        json.dump(options, f, indent=2)


def backup_original_options(episode_dir):
    """备份原始 options.json"""
    options_path = os.path.join(episode_dir, "options.json")
    backup_path = os.path.join(episode_dir, "origin.json")
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy(options_path, backup_path)
        print(f"✅ 已备份原始配置到: {backup_path}")


def adjust_position(original_xy, x_range, y_range):
    """按指定方向范围调整位置
    
    Args:
        original_xy: 原始位置 [x, y]
        x_range: (min, max) x方向偏移范围（米）
        y_range: (min, max) y方向偏移范围（米）
    
    Returns:
        新位置 [x, y]
    """
    x_offset = random.uniform(x_range[0], x_range[1])
    y_offset = random.uniform(y_range[0], y_range[1])
    
    new_x = original_xy[0] + x_offset
    new_y = original_xy[1] + y_offset
    
    # 确保在合理的桌面范围内
    new_x = np.clip(new_x, -0.5, 0.3)
    new_y = np.clip(new_y, -0.3, 0.5)
    
    return [float(new_x), float(new_y)]


def modify_object_position(options, object_name, x_range, y_range):
    """修改指定物体的位置
    
    Args:
        options: options字典
        object_name: 物体名称
        x_range: X轴偏移范围
        y_range: Y轴偏移范围
    
    Returns:
        (new_xy, original_xy): 新位置和原始位置
    """
    obj_init_options = options.get("obj_init_options", {})
    if object_name in obj_init_options:
        obj_opts = obj_init_options[object_name]
        original_xy = obj_opts["init_xy"].copy()
        new_xy = adjust_position(original_xy, x_range, y_range)
        obj_opts["init_xy"] = new_xy
        return new_xy, original_xy
    # grasp 场景：obj_init_options 直接是 init_xy/orientation
    if "init_xy" in obj_init_options:
        original_xy = obj_init_options["init_xy"].copy()
        new_xy = adjust_position(original_xy, x_range, y_range)
        obj_init_options["init_xy"] = new_xy
        return new_xy, original_xy
    return None, None


def run_replay(episode_dir, task_name):
    """运行重放脚本"""
    cmd = [
        VENV_PYTHON, REPLAY_SCRIPT,
        "--episode_dir", episode_dir,
        "--task", task_name,
        "--render_every", "1"
    ]
    
    try:
        # 设置 PYTHONPATH 确保能找到 simpler_env 模块
        env = os.environ.copy()
        env['PYTHONPATH'] = PROJECT_ROOT
        
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        if result.returncode != 0:
            print(f"\n❌ 重放命令失败 (返回码: {result.returncode})")
            print(f"命令: {' '.join(cmd)}")
            if result.stdout:
                print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏱️  重放超时")
        return False
    except Exception as e:
        print(f"❌ 重放执行出错: {e}")
        return False


def check_grasp_success(episode_dir, task_type, min_steps=MIN_CONSECUTIVE_GRASP_STEPS):
    """检查是否成功抓取
    
    Args:
        episode_dir: episode目录
        task_type: 任务类型 "grasp" 或 "move"
        min_steps: 最小连续抓取步数
    
    Returns:
        (success, consecutive_steps, details): 成功标志、连续步数和详细信息
    """
    replay_log_path = os.path.join(episode_dir, "replay_log.json")
    log_path = os.path.join(episode_dir, "log.json")
    
    log_to_read = replay_log_path if os.path.exists(replay_log_path) else log_path
    
    if not os.path.exists(log_to_read):
        return False, 0, {}
    
    try:
        with open(log_to_read, 'r') as f:
            log_data = json.load(f)
        
        def to_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() == "true"
            return False
        
        if task_type == "grasp":
            # Grasp任务：检查 is_grasped 或 lifted_object
            for step_key, step_info in log_data.items():
                if isinstance(step_info, dict):
                    is_grasped = to_bool(step_info.get("is_grasped", False))
                    lifted = to_bool(step_info.get("lifted_object", False))
                    
                    if is_grasped or lifted:
                        return True, 1, {
                            "step": step_key,
                            "is_grasped": is_grasped,
                            "lifted_object": lifted,
                        }
            return False, 0, {}
        
        elif task_type == "move":
            # Move任务：计算最长连续抓取序列
            grasp_steps = []
            sorted_steps = sorted([(int(k), v) for k, v in log_data.items() 
                                  if isinstance(v, dict)], key=lambda x: x[0])
            
            for step_num, step_info in sorted_steps:
                # 兼容两种字段名：is_src_obj_grasped 或 is_grasped
                is_grasped = step_info.get("is_src_obj_grasped")
                if is_grasped is None:
                    is_grasped = to_bool(step_info.get("is_grasped", False))
                if is_grasped is True:
                    grasp_steps.append(step_num)
            
            if not grasp_steps:
                return False, 0, {}
            
            # 查找连续抓取序列
            consecutive_sequences = []
            current_seq = [grasp_steps[0]]
            
            for i in range(1, len(grasp_steps)):
                if grasp_steps[i] == grasp_steps[i-1] + 1:
                    current_seq.append(grasp_steps[i])
                else:
                    consecutive_sequences.append(current_seq)
                    current_seq = [grasp_steps[i]]
            consecutive_sequences.append(current_seq)
            
            longest_seq = max(consecutive_sequences, key=len)
            consecutive_steps = len(longest_seq)
            
            success = consecutive_steps >= min_steps
            
            details = {
                "consecutive_grasp_steps": consecutive_steps,
                "grasp_step_range": f"{longest_seq[0]}-{longest_seq[-1]}",
                "is_src_obj_grasped": True,
            }
            
            return success, consecutive_steps, details
        
        return False, 0, {}
    except Exception as e:
        print(f"⚠️  读取日志失败: {e}")
        return False, 0, {}


def run_single_attempt(attempt, total_attempts, stage_name, episode_dir, task_name, 
                      task_type, original_options, source_obj, x_range, y_range):
    """执行单次尝试
    
    Returns:
        结果字典或None（如果重放失败）
    """
    print(f"\n🔄 [{stage_name}] 尝试 {attempt}/{total_attempts}")
    
    # 修改物体位置
    options = json.loads(json.dumps(original_options))  # 深拷贝
    new_xy, orig_xy = modify_object_position(options, source_obj, x_range, y_range)
    
    print(f"   📍 新位置: [{new_xy[0]:.4f}, {new_xy[1]:.4f}]")
    print(f"   📏 偏移: Δx={new_xy[0]-orig_xy[0]:+.4f}m, Δy={new_xy[1]-orig_xy[1]:+.4f}m")
    
    # 保存修改后的配置
    save_options(episode_dir, options)
    
    # 运行重放
    print("   ⏳ 执行重放...", end=" ", flush=True)
    replay_success = run_replay(episode_dir, task_name)
    
    if not replay_success:
        print("❌ 重放失败")
        return None
    
    print("✅")
    
    # 检查是否成功抓取
    print("   🔍 检查抓取...", end=" ", flush=True)
    is_success, grasp_steps, details = check_grasp_success(episode_dir, task_type)
    
    if is_success:
        print(f"✅ 成功！连续抓取 {grasp_steps} 步")
    else:
        print(f"❌ 失败（连续抓取 {grasp_steps} 步）")
    
    return {
        'attempt': attempt,
        'stage': stage_name,
        'position': new_xy,
        'original_position': orig_xy,
        'options': options,
        'success': is_success,
        'grasp_steps': grasp_steps,
        'details': details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="自动寻找成功抓取配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
方向说明（机械臂视角）:
  left:       向左（x负方向）
  right:      向右（x正方向）
  up:         向上（y负方向）
  down:       向下（y正方向）
  left-up:    左上
  left-down:  左下
  right-up:   右上
  right-down: 右下

示例用法:
  # 在场景13中向左上方向搜索
  python3 auto_find_successful_grasp.py 13 left-up
  
  # 在场景7中向右方向搜索，总共搜索20次
  python3 auto_find_successful_grasp.py 7 right --attempts 20
  
  # 指定基础目录
  python3 auto_find_successful_grasp.py 13 left-up --base_dir results/t-grasp_n-100/openvla-7b_2024
        """
    )
    
    parser.add_argument(
        'episode_dir',
        type=str,
        help="Episode工作目录（包含options.json）"
    )
    
    parser.add_argument(
        'direction',
        type=str,
        choices=list(DIRECTION_OFFSETS.keys()),
        help="搜索方向"
    )
    
    parser.add_argument(
        '--attempts',
        type=int,
        default=DEFAULT_TOTAL_ATTEMPTS,
        help=f"总搜索次数（默认: {DEFAULT_TOTAL_ATTEMPTS}）"
    )
    
    parser.add_argument(
        '--min_steps',
        type=int,
        default=MIN_CONSECUTIVE_GRASP_STEPS,
        help=f"最小连续抓取步数（默认: {MIN_CONSECUTIVE_GRASP_STEPS}）"
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default=DEFAULT_TASK_TYPE,
        help=f"任务类型 grasp/move（默认: {DEFAULT_TASK_TYPE}）"
    )
    
    parser.add_argument(
        '--task_name',
        type=str,
        default=DEFAULT_TASK_NAME,
        help=f"任务名称（默认: {DEFAULT_TASK_NAME}）"
    )
    
    args = parser.parse_args()
    
    episode_dir = args.episode_dir
    
    if not os.path.exists(episode_dir):
        print(f"❌ Episode目录不存在: {episode_dir}")
        return 1
    
    # 获取方向偏移范围
    direction_offset = DIRECTION_OFFSETS[args.direction]
    
    # 计算粗搜索和精细搜索的次数
    coarse_attempts = int(args.attempts * COARSE_FINE_RATIO)
    fine_attempts = args.attempts - coarse_attempts
    
    print("=" * 70)
    print("🔍 自动寻找成功抓取配置")
    print("=" * 70)
    print(f"📍 目录: {episode_dir}")
    print(f"🔍 方向: {args.direction}")
    print(f"🧭 方向: {args.direction}")
    print(f"   X范围: [{direction_offset['x'][0]:+.3f}, {direction_offset['x'][1]:+.3f}]m")
    print(f"   Y范围: [{direction_offset['y'][0]:+.3f}, {direction_offset['y'][1]:+.3f}]m")
    print(f"🔢 搜索策略:")
    print(f"   粗搜索: {coarse_attempts} 次")
    print(f"   精细搜索: {fine_attempts} 次（在最佳点附近）")
    print(f"✅ 成功标准: 连续抓取 >= {args.min_steps} 步")
    print("=" * 70)
    
    # 备份原始配置
    backup_original_options(episode_dir)
    
    # 加载原始备份配置（用于获取真正的原始位置）
    backup_path = os.path.join(episode_dir, "origin.json")
    if os.path.exists(backup_path):
        with open(backup_path, 'r') as f:
            original_options = json.load(f)
        print(f"✅ 从备份加载原始配置: origin.json")
    else:
        original_options = load_options(episode_dir)
        print(f"⚠️  备份不存在，使用当前配置")
    
    if "model_ids" in original_options and "source_obj_id" in original_options:
        source_obj = original_options["model_ids"][original_options["source_obj_id"]]
        original_xy = original_options['obj_init_options'][source_obj]['init_xy']
    else:
        source_obj = original_options.get("model_id", "unknown")
        original_xy = original_options['obj_init_options']['init_xy']
    
    print(f"\n🎯 源物体: {source_obj}")
    print(f"📍 原始位置: [{original_xy[0]:.4f}, {original_xy[1]:.4f}]")
    
    # 记录所有尝试的结果
    all_attempts = []
    
    # ========== 阶段1: 粗搜索 ==========
    if coarse_attempts > 0:
        print("\n" + "=" * 70)
        print(f"🔍 阶段1: 粗搜索（指定方向）")
        print("=" * 70)
        
        for attempt in range(1, coarse_attempts + 1):
            result = run_single_attempt(
                attempt, coarse_attempts, "粗搜索",
                episode_dir, args.task_name, args.task,
                original_options, source_obj,
                x_range=direction_offset['x'],
                y_range=direction_offset['y']
            )
            if result:
                all_attempts.append(result)
                
                # 如果找到成功的配置，立即进入精细搜索
                if result['success']:
                    print(f"\n🎉 找到成功配置！立即开始精细搜索...")
                    break
    
    # ========== 阶段2: 精细搜索 ==========
    if fine_attempts > 0 and all_attempts:
        # 找到粗搜索的最佳结果
        all_attempts.sort(key=lambda x: x['grasp_steps'], reverse=True)
        best_coarse = all_attempts[0]
        
        print("\n" + "=" * 70)
        print(f"📊 阶段1完成！最佳结果: {best_coarse['grasp_steps']} 步")
        print(f"   位置: [{best_coarse['position'][0]:.4f}, {best_coarse['position'][1]:.4f}]")
        print("=" * 70)
        
        print("\n" + "=" * 70)
        print(f"🔍 阶段2: 精细搜索（在最佳点附近微调）")
        print(f"   基准点: [{best_coarse['position'][0]:.4f}, {best_coarse['position'][1]:.4f}]")
        print("=" * 70)
        
        # 在最佳点附近精细搜索
        best_position = best_coarse['position']
        
        for attempt in range(1, fine_attempts + 1):
            # 修改物体位置（在最佳点附近微调）
            options = json.loads(json.dumps(original_options))
            obj_init_options = options["obj_init_options"]
            if source_obj in obj_init_options:
                obj_opts = obj_init_options[source_obj]
            else:
                obj_opts = obj_init_options
            
            x_offset = random.uniform(FINE_SEARCH_RANGE['x'][0], FINE_SEARCH_RANGE['x'][1])
            y_offset = random.uniform(FINE_SEARCH_RANGE['y'][0], FINE_SEARCH_RANGE['y'][1])
            
            new_xy = [
                float(np.clip(best_position[0] + x_offset, -0.5, 0.3)),
                float(np.clip(best_position[1] + y_offset, -0.3, 0.5))
            ]
            obj_opts['init_xy'] = new_xy
            
            print(f"\n🔄 [精细搜索] 尝试 {attempt}/{fine_attempts}")
            print(f"   📍 新位置: [{new_xy[0]:.4f}, {new_xy[1]:.4f}]")
            print(f"   📏 相对基准点: Δx={new_xy[0]-best_position[0]:+.4f}m, Δy={new_xy[1]-best_position[1]:+.4f}m")
            print(f"   📏 相对原始点: Δx={new_xy[0]-original_xy[0]:+.4f}m, Δy={new_xy[1]-original_xy[1]:+.4f}m")
            
            save_options(episode_dir, options)
            
            print("   ⏳ 执行重放...", end=" ", flush=True)
            replay_success = run_replay(episode_dir, args.task_name)
            
            if not replay_success:
                print("❌ 重放失败")
                continue
            
            print("✅")
            
            print("   🔍 检查抓取...", end=" ", flush=True)
            is_success, grasp_steps, details = check_grasp_success(episode_dir, args.task, args.min_steps)
            
            if is_success:
                print(f"✅ 成功！连续抓取 {grasp_steps} 步")
            else:
                print(f"❌ 失败（连续抓取 {grasp_steps} 步）")
            
            all_attempts.append({
                'attempt': coarse_attempts + attempt,
                'stage': '精细搜索',
                'position': new_xy,
                'original_position': original_xy,
                'options': options,
                'success': is_success,
                'grasp_steps': grasp_steps,
                'details': details,
            })
    
    # ========== 分析结果 ==========
    print("\n" + "=" * 70)
    print("📊 搜索完成，分析结果...")
    print("=" * 70)
    
    if not all_attempts:
        print("😞 所有尝试都失败了（重放错误）")
        return 1
    
    # 按连续抓取步数排序
    all_attempts.sort(key=lambda x: x['grasp_steps'], reverse=True)
    best = all_attempts[0]
    
    print(f"\n🏆 最优结果:")
    print(f"   尝试: #{best['attempt']}/{args.attempts}")
    print(f"   连续抓取: {best['grasp_steps']} 步")
    print(f"   位置: [{best['position'][0]:.4f}, {best['position'][1]:.4f}]")
    print(f"   偏移: Δx={best['position'][0]-original_xy[0]:+.4f}m, Δy={best['position'][1]-original_xy[1]:+.4f}m")
    
    # 显示前5名结果
    print(f"\n📈 Top 5 结果:")
    for i, result in enumerate(all_attempts[:5]):
        status = "✅" if result['success'] else "⚠️"
        print(f"   {i+1}. {status} #{result['attempt']:2d}: {result['grasp_steps']:2d}步 at [{result['position'][0]:+.4f}, {result['position'][1]:+.4f}]")
    
    # 保存最优配置
    save_options(episode_dir, best['options'])
    
    print("\n" + "=" * 70)
    print("🎊 已保存最优配置！")
    print("=" * 70)
    print(f"📍 最优位置: [{best['position'][0]:.4f}, {best['position'][1]:.4f}]")
    print(f"📏 相对偏移: Δx={best['position'][0]-original_xy[0]:+.4f}m, Δy={best['position'][1]-original_xy[1]:+.4f}m")
    print(f"📊 连续抓取: {best['grasp_steps']} 步")
    print(f"\n💾 配置文件: {os.path.join(episode_dir, 'options.json')}")
    print(f"📦 原始备份: {os.path.join(episode_dir, 'origin.json')}")
    print("=" * 70)
    
    if best['success']:
        print(f"\n✅ 成功！已达到目标（>= {args.min_steps} 步）")
    else:
        print(f"\n⚠️  注意: 最优结果({best['grasp_steps']}步)仍未达到目标({args.min_steps}步)")
        print(f"   但已保留最优配置，将继续推理")
    
    # 无论是否达到目标，都返回成功（因为已保存最优配置）
    return 0


if __name__ == "__main__":
    exit(main())
