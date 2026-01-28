#!/usr/bin/env python3
"""
物体旋转优化策略
功能：调整物体的旋转角度以优化抓取成功率

适用场景：
1. 物体形状不对称（如瓶子、盒子、工具等）
2. 夹爪与物体角度不匹配
3. 需要特定方向才能成功抓取的物体

使用方法：
    python3 fix_strategy_rotate_object.py <episode_dir> --rotation_mode <mode> --angle <degrees>

旋转模式：
    z_axis: 绕Z轴旋转（桌面水平旋转，最常用）
    x_axis: 绕X轴旋转（翻转）
    y_axis: 绕Y轴旋转（侧翻）
    random_z: 随机Z轴旋转
    preset: 使用预设方向（upright/laid_vertically/lr_switch）
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from transforms3d.quaternions import quat2axangle, axangle2quat
from transforms3d.euler import euler2quat, quat2euler


# ==================== 默认配置 ====================

# 预设方向（来自底层实现）
PRESET_ORIENTATIONS = {
    "upright": [0.707, 0.707, 0, 0],           # 直立（X轴90度）
    "laid_vertically": [0.5, 0.5, 0.5, 0.5],  # 侧躺（Y轴90度）
    "lr_switch": [1, 0, 0, 0],                 # 左右翻转（无旋转）
}

# 常用旋转角度（度）
COMMON_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]


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
    print(f"✅ 已保存配置: {options_path}")


def backup_original_options(episode_dir):
    """备份原始配置"""
    options_path = os.path.join(episode_dir, "options.json")
    backup_path = os.path.join(episode_dir, "origin.json")
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy(options_path, backup_path)
        print(f"✅ 已备份原始配置: {backup_path}")


def quaternion_to_euler_degrees(quat):
    """四元数转欧拉角（度）"""
    # quat: [w, x, y, z]
    euler_rad = quat2euler([quat[0], quat[1], quat[2], quat[3]])
    euler_deg = [np.degrees(angle) for angle in euler_rad]
    return euler_deg


def euler_degrees_to_quaternion(euler_deg):
    """欧拉角（度）转四元数"""
    euler_rad = [np.radians(angle) for angle in euler_deg]
    quat = euler2quat(euler_rad[0], euler_rad[1], euler_rad[2])
    return [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


def rotate_quaternion(original_quat, axis, angle_degrees):
    """将四元数绕指定轴旋转
    
    Args:
        original_quat: 原始四元数 [w, x, y, z]
        axis: 旋转轴 'x', 'y', 或 'z'
        angle_degrees: 旋转角度（度）
    
    Returns:
        新的四元数 [w, x, y, z]
    """
    # 定义轴向量
    axis_vectors = {
        'x': [1, 0, 0],
        'y': [0, 1, 0],
        'z': [0, 0, 1]
    }
    
    axis_vec = axis_vectors[axis.lower()]
    angle_rad = np.radians(angle_degrees)
    
    # 创建旋转四元数
    rotation_quat = axangle2quat(axis_vec, angle_rad)
    
    # 组合旋转：new = rotation * original
    # 注意：transforms3d 使用 [w, x, y, z] 格式
    original_wxyz = [original_quat[0], original_quat[1], original_quat[2], original_quat[3]]
    
    # 四元数乘法
    from transforms3d.quaternions import qmult
    new_quat = qmult(rotation_quat, original_wxyz)
    
    return [float(new_quat[0]), float(new_quat[1]), float(new_quat[2]), float(new_quat[3])]


def get_object_rotation_info(obj_init_options, source_obj=None):
    """获取物体当前旋转信息
    
    Returns:
        (current_quat, rotation_source): 当前四元数和来源
    """
    if source_obj and source_obj in obj_init_options:
        obj_opts = obj_init_options[source_obj]
    else:
        obj_opts = obj_init_options
    
    # 检查 init_rot_quat
    if "init_rot_quat" in obj_opts:
        return obj_opts["init_rot_quat"], "init_rot_quat"
    
    # 检查 orientation（字段或预设名称）
    if "orientation" in obj_opts:
        orientation = obj_opts["orientation"]
        if isinstance(orientation, list):
            return orientation, "orientation (list)"
        elif orientation in PRESET_ORIENTATIONS:
            return PRESET_ORIENTATIONS[orientation], f"orientation (preset: {orientation})"
    
    # 默认无旋转
    return [1, 0, 0, 0], "default (no rotation)"


def set_object_rotation(obj_init_options, new_quat, source_obj=None):
    """设置物体旋转
    
    Args:
        obj_init_options: obj_init_options 字典
        new_quat: 新的四元数 [w, x, y, z]
        source_obj: 物体名称（可选）
    """
    if source_obj and source_obj in obj_init_options:
        obj_opts = obj_init_options[source_obj]
    else:
        obj_opts = obj_init_options
    
    # 统一使用 orientation 字段（底层会优先读取此字段）
    obj_opts["orientation"] = new_quat
    
    # 如果存在 init_rot_quat，也更新它（兼容性）
    if "init_rot_quat" in obj_opts:
        obj_opts["init_rot_quat"] = new_quat


# ==================== 主要功能 ====================

def rotate_object_z_axis(options, source_obj, angle_degrees):
    """绕Z轴旋转物体（桌面水平旋转）
    
    Args:
        options: options 字典
        source_obj: 物体名称
        angle_degrees: 旋转角度（度）
    """
    obj_init_options = options.get("obj_init_options", {})
    
    # 获取当前旋转
    current_quat, rotation_source = get_object_rotation_info(obj_init_options, source_obj)
    
    print(f"\n🔄 绕Z轴旋转（桌面水平）")
    print(f"📍 物体: {source_obj}")
    print(f"📊 当前旋转: {current_quat} (来源: {rotation_source})")
    
    # 显示当前欧拉角
    current_euler = quaternion_to_euler_degrees(current_quat)
    print(f"   当前欧拉角: X={current_euler[0]:.1f}°, Y={current_euler[1]:.1f}°, Z={current_euler[2]:.1f}°")
    
    # 应用旋转
    new_quat = rotate_quaternion(current_quat, 'z', angle_degrees)
    new_euler = quaternion_to_euler_degrees(new_quat)
    
    print(f"🎯 旋转角度: {angle_degrees:+.1f}°")
    print(f"📊 新旋转: {[f'{x:.4f}' for x in new_quat]}")
    print(f"   新欧拉角: X={new_euler[0]:.1f}°, Y={new_euler[1]:.1f}°, Z={new_euler[2]:.1f}°")
    
    # 更新配置
    set_object_rotation(obj_init_options, new_quat, source_obj)
    
    return new_quat


def rotate_object_arbitrary_axis(options, source_obj, axis, angle_degrees):
    """绕任意轴旋转物体
    
    Args:
        options: options 字典
        source_obj: 物体名称
        axis: 'x', 'y', 或 'z'
        angle_degrees: 旋转角度（度）
    """
    obj_init_options = options.get("obj_init_options", {})
    
    # 获取当前旋转
    current_quat, rotation_source = get_object_rotation_info(obj_init_options, source_obj)
    
    axis_name = {'x': 'X轴（翻转）', 'y': 'Y轴（侧翻）', 'z': 'Z轴（水平）'}
    
    print(f"\n🔄 绕{axis_name[axis.lower()]}旋转")
    print(f"📍 物体: {source_obj}")
    print(f"📊 当前旋转: {current_quat} (来源: {rotation_source})")
    
    # 应用旋转
    new_quat = rotate_quaternion(current_quat, axis, angle_degrees)
    new_euler = quaternion_to_euler_degrees(new_quat)
    
    print(f"🎯 旋转角度: {angle_degrees:+.1f}°")
    print(f"📊 新旋转: {[f'{x:.4f}' for x in new_quat]}")
    print(f"   新欧拉角: X={new_euler[0]:.1f}°, Y={new_euler[1]:.1f}°, Z={new_euler[2]:.1f}°")
    
    # 更新配置
    set_object_rotation(obj_init_options, new_quat, source_obj)
    
    return new_quat


def set_preset_orientation(options, source_obj, preset_name):
    """设置预设方向
    
    Args:
        options: options 字典
        source_obj: 物体名称
        preset_name: 预设名称（upright/laid_vertically/lr_switch）
    """
    if preset_name not in PRESET_ORIENTATIONS:
        print(f"❌ 未知预设: {preset_name}")
        print(f"   可用预设: {list(PRESET_ORIENTATIONS.keys())}")
        return None
    
    obj_init_options = options.get("obj_init_options", {})
    
    # 获取当前旋转
    current_quat, rotation_source = get_object_rotation_info(obj_init_options, source_obj)
    
    print(f"\n🎯 设置预设方向: {preset_name}")
    print(f"📍 物体: {source_obj}")
    print(f"📊 当前旋转: {current_quat} (来源: {rotation_source})")
    
    new_quat = PRESET_ORIENTATIONS[preset_name]
    new_euler = quaternion_to_euler_degrees(new_quat)
    
    print(f"📊 新旋转: {new_quat}")
    print(f"   新欧拉角: X={new_euler[0]:.1f}°, Y={new_euler[1]:.1f}°, Z={new_euler[2]:.1f}°")
    
    # 更新配置
    set_object_rotation(obj_init_options, new_quat, source_obj)
    
    return new_quat


def random_z_rotation(options, source_obj):
    """随机Z轴旋转（0-360度）
    
    Args:
        options: options 字典
        source_obj: 物体名称
    """
    angle = np.random.uniform(0, 360)
    print(f"🎲 随机Z轴旋转: {angle:.1f}°")
    return rotate_object_z_axis(options, source_obj, angle)


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="物体旋转优化策略",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
旋转模式说明:
  z_axis:    绕Z轴旋转（桌面水平旋转，最常用）
  x_axis:    绕X轴旋转（翻转）
  y_axis:    绕Y轴旋转（侧翻）
  random_z:  随机Z轴旋转（0-360度）
  preset:    使用预设方向（需配合 --preset 参数）

预设方向:
  upright:         直立（X轴90度）
  laid_vertically: 侧躺（Y轴90度）
  lr_switch:       左右翻转（无旋转）

示例用法:
  # 绕Z轴旋转45度
  python3 fix_strategy_rotate_object.py /path/to/episode --rotation_mode z_axis --angle 45
  
  # 绕Z轴旋转90度（最常用，夹爪换个角度）
  python3 fix_strategy_rotate_object.py /path/to/episode --rotation_mode z_axis --angle 90
  
  # 随机Z轴旋转
  python3 fix_strategy_rotate_object.py /path/to/episode --rotation_mode random_z
  
  # 设置为直立姿态
  python3 fix_strategy_rotate_object.py /path/to/episode --rotation_mode preset --preset upright
  
  # 翻转物体（X轴180度）
  python3 fix_strategy_rotate_object.py /path/to/episode --rotation_mode x_axis --angle 180
        """
    )
    
    parser.add_argument(
        'episode_dir',
        type=str,
        help="Episode目录（包含options.json）"
    )
    
    parser.add_argument(
        '--rotation_mode',
        type=str,
        choices=['z_axis', 'x_axis', 'y_axis', 'random_z', 'preset'],
        default='z_axis',
        help="旋转模式（默认: z_axis）"
    )
    
    parser.add_argument(
        '--angle',
        type=float,
        default=90.0,
        help="旋转角度（度，默认: 90）"
    )
    
    parser.add_argument(
        '--preset',
        type=str,
        choices=list(PRESET_ORIENTATIONS.keys()),
        help="预设方向名称（仅当 rotation_mode=preset 时使用）"
    )
    
    args = parser.parse_args()
    
    episode_dir = args.episode_dir
    
    # 检查目录
    if not os.path.exists(episode_dir):
        print(f"❌ Episode目录不存在: {episode_dir}")
        return 1
    
    print("=" * 70)
    print("🔄 物体旋转优化策略")
    print("=" * 70)
    print(f"📍 目录: {episode_dir}")
    print(f"🔧 旋转模式: {args.rotation_mode}")
    if args.rotation_mode == 'preset':
        print(f"🎯 预设方向: {args.preset}")
    elif args.rotation_mode != 'random_z':
        print(f"📐 旋转角度: {args.angle}°")
    print("=" * 70)
    
    # 备份原始配置
    backup_original_options(episode_dir)
    
    # 加载配置（优先从备份加载原始配置）
    backup_path = os.path.join(episode_dir, "origin.json")
    if os.path.exists(backup_path):
        with open(backup_path, 'r') as f:
            options = json.load(f)
        print(f"✅ 从备份加载原始配置: origin.json")
    else:
        options = load_options(episode_dir)
        print(f"⚠️  备份不存在，使用当前配置")
    
    # 确定物体名称
    if "model_ids" in options and "source_obj_id" in options:
        # Move任务
        source_obj = options["model_ids"][options["source_obj_id"]]
    else:
        # Grasp任务
        source_obj = options.get("model_id", "object")
    
    # 执行旋转
    try:
        if args.rotation_mode == 'z_axis':
            rotate_object_z_axis(options, source_obj, args.angle)
        
        elif args.rotation_mode == 'x_axis':
            rotate_object_arbitrary_axis(options, source_obj, 'x', args.angle)
        
        elif args.rotation_mode == 'y_axis':
            rotate_object_arbitrary_axis(options, source_obj, 'y', args.angle)
        
        elif args.rotation_mode == 'random_z':
            random_z_rotation(options, source_obj)
        
        elif args.rotation_mode == 'preset':
            if not args.preset:
                print(f"❌ 使用 preset 模式时必须指定 --preset 参数")
                return 1
            set_preset_orientation(options, source_obj, args.preset)
        
        # 保存配置
        save_options(episode_dir, options)
        
        print("\n" + "=" * 70)
        print("✅ 旋转配置完成！")
        print("=" * 70)
        print(f"💡 提示: 运行推理验证效果")
        print(f"📦 原始配置备份: {os.path.join(episode_dir, 'origin.json')}")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 旋转失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
