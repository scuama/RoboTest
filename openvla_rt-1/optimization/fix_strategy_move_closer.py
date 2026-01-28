#!/usr/bin/env python3
"""
简化版：将物品移近机械臂
功能：仅修改配置，将物品向机械臂方向移动指定距离
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path


# ==================== 默认配置参数 ====================
DEFAULT_BASE_DIR = "newresult/t-grasp_n-100_o-0_s-170912623/openvla-7b_2024/t-grasp_n-100_o-0_s-170912623/openvla-7b_2024"

# 机械臂中心位置（google_robot 默认位置范围的中点）
# 来自 base_env.py line 310-311: init_x ∈ [0.30, 0.40], init_y ∈ [0.0, 0.2]
ROBOT_CENTER = [0.35, 0.1]

# 默认移动比例（0-1），0.3表示移动30%的距离
DEFAULT_MOVE_RATIO = 0.3

# 碰撞检测安全距离（物体之间最小间隔，单位：米）
COLLISION_SAFE_DISTANCE = 0.08  # 8cm 安全距离


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
        print(f"✅ 已备份原始配置: {backup_path}")


def calculate_distance(pos1, pos2):
    """计算两点之间的欧氏距离"""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return np.sqrt(dx**2 + dy**2)


def get_all_object_positions(options, exclude_obj=None):
    """获取所有物体的位置
    
    Args:
        options: 配置字典
        exclude_obj: 要排除的物体名称
    
    Returns:
        物体位置列表 [(obj_name, [x, y]), ...]
    """
    positions = []
    obj_init_options = options.get("obj_init_options", {})
    
    # move 场景：多个物体
    if "model_ids" in options:
        model_ids = options["model_ids"]
        if isinstance(model_ids, list):
            for obj_name in model_ids:
                if obj_name != exclude_obj and obj_name in obj_init_options:
                    pos = obj_init_options[obj_name].get("init_xy")
                    if pos:
                        positions.append((obj_name, pos))
    # grasp 场景：单个物体
    elif "init_xy" in obj_init_options:
        obj_name = options.get("model_id", "object")
        if obj_name != exclude_obj:
            positions.append((obj_name, obj_init_options["init_xy"]))
    
    return positions


def check_collision(new_pos, other_positions, safe_distance=COLLISION_SAFE_DISTANCE):
    """检查新位置是否会与其他物体碰撞
    
    Args:
        new_pos: 新位置 [x, y]
        other_positions: 其他物体位置列表 [(name, [x, y]), ...]
        safe_distance: 安全距离（米）
    
    Returns:
        (is_collision, collision_obj_name)
    """
    for obj_name, pos in other_positions:
        dist = calculate_distance(new_pos, pos)
        if dist < safe_distance:
            return True, obj_name
    return False, None


def move_closer_to_robot(original_xy, move_ratio=0.3, other_positions=None):
    """将物体向机械臂方向移动
    
    Args:
        original_xy: 原始位置 [x, y]
        move_ratio: 移动比例 (0-1)，0.3表示移动30%的距离
        other_positions: 其他物体位置列表，用于碰撞检测
    
    Returns:
        新位置 [x, y]
    """
    if other_positions is None:
        other_positions = []
    
    # 计算朝向机械臂的方向向量
    dx = ROBOT_CENTER[0] - original_xy[0]
    dy = ROBOT_CENTER[1] - original_xy[1]
    
    # 尝试不同的移动比例，从期望值逐渐减小，避免碰撞
    for ratio in np.linspace(move_ratio, 0.05, 20):
        new_x = original_xy[0] + dx * ratio
        new_y = original_xy[1] + dy * ratio
        
        # 确保在合理的桌面范围内
        new_x = np.clip(new_x, -0.4, 0.3)
        new_y = np.clip(new_y, -0.2, 0.5)
        
        new_pos = [float(new_x), float(new_y)]
        
        # 检查碰撞
        is_collision, collision_obj = check_collision(new_pos, other_positions)
        if not is_collision:
            if ratio < move_ratio * 0.9:  # 如果调整了移动比例
                print(f"  ⚠️  为避免碰撞，移动比例调整为 {ratio:.2f} (原计划 {move_ratio:.2f})")
            return new_pos
    
    # 如果所有尝试都失败，返回原始位置
    print(f"  ⚠️  无法找到安全位置，保持原位不变")
    return [float(original_xy[0]), float(original_xy[1])]


def main():
    parser = argparse.ArgumentParser(description="将物品移近机械臂（仅修改配置）")
    
    parser.add_argument('episode_dir', type=str, help="Episode工作目录（包含options.json）")
    parser.add_argument('--move_ratio', type=float, default=DEFAULT_MOVE_RATIO, 
                       help=f"移动比例 (0-1)，默认: {DEFAULT_MOVE_RATIO}")
    
    args = parser.parse_args()
    
    episode_dir = args.episode_dir
    
    if not os.path.exists(episode_dir):
        print(f"❌ Episode目录不存在: {episode_dir}")
        return 1
    
    print("=" * 70)
    print("🎯 将物品移近机械臂")
    print("=" * 70)
    print(f"📍 目录: {episode_dir}")
    print(f"📊 移动比例: {args.move_ratio:.1%}")
    print("=" * 70)
    
    # 备份原始配置
    backup_original_options(episode_dir)
    
    # 加载原始备份配置（用于获取真正的原始位置）
    backup_path = os.path.join(episode_dir, "origin.json")
    if os.path.exists(backup_path):
        with open(backup_path, 'r') as f:
            options = json.load(f)
        print(f"✅ 从备份加载原始配置: origin.json")
    else:
        options = load_options(episode_dir)
        print(f"⚠️  备份不存在，使用当前配置")
    
    # 获取源物体和初始位置（兼容 grasp/move 两种配置结构）
    source_obj = None
    obj_init_options = options.get("obj_init_options", {})
    if "model_ids" in options and "source_obj_id" in options:
        source_obj_id = options["source_obj_id"]
        if isinstance(options["model_ids"], list):
            source_obj = options["model_ids"][source_obj_id]
        else:
            source_obj = options["model_ids"][source_obj_id]
        original_xy = obj_init_options[source_obj]["init_xy"]
    else:
        # grasp 场景：obj_init_options 直接包含 init_xy
        source_obj = options.get("model_id", "unknown")
        original_xy = obj_init_options["init_xy"]
    
    original_dist = calculate_distance(original_xy, ROBOT_CENTER)
    
    print(f"\n🎯 源物体: {source_obj}")
    print(f"📍 原始位置: [{original_xy[0]:.4f}, {original_xy[1]:.4f}]")
    print(f"📐 到机械臂距离: {original_dist:.3f}m")
    
    # 获取其他物体的位置（用于碰撞检测）
    other_positions = get_all_object_positions(options, exclude_obj=source_obj)
    if other_positions:
        print(f"\n🔍 检测到 {len(other_positions)} 个其他物体：")
        for obj_name, pos in other_positions:
            dist_to_source = calculate_distance(original_xy, pos)
            print(f"  - {obj_name}: [{pos[0]:.4f}, {pos[1]:.4f}], 距离: {dist_to_source:.3f}m")
    
    # 计算新位置（带碰撞检测）
    new_xy = move_closer_to_robot(original_xy, args.move_ratio, other_positions)
    new_dist = calculate_distance(new_xy, ROBOT_CENTER)
    
    # 再次检查碰撞（用于报告）
    is_collision, collision_obj = check_collision(new_xy, other_positions)
    if is_collision:
        print(f"  ⚠️  警告：新位置可能与 {collision_obj} 太近！")
    
    print(f"\n📍 新位置: [{new_xy[0]:.4f}, {new_xy[1]:.4f}]")
    print(f"📏 偏移: Δx={new_xy[0]-original_xy[0]:+.4f}m, Δy={new_xy[1]-original_xy[1]:+.4f}m")
    print(f"📐 新距离: {new_dist:.3f}m (靠近 {original_dist-new_dist:.3f}m)")
    
    # 更新配置
    if "model_ids" in options and "source_obj_id" in options:
        options["obj_init_options"][source_obj]["init_xy"] = new_xy
    else:
        options["obj_init_options"]["init_xy"] = new_xy
    save_options(episode_dir, options)
    
    print("\n" + "=" * 70)
    print("✅ 配置已更新！")
    print("=" * 70)
    print(f"💾 配置文件: {os.path.join(episode_dir, 'options.json')}")
    print(f"📦 原始备份: {os.path.join(episode_dir, 'origin.json')}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
