#!/usr/bin/env python3
"""
简化版：替换物品策略
功能：仅修改配置，将物品替换成指定物品（确保物品在场景中有定义）
"""

import os
import json
import argparse
from pathlib import Path


# ==================== 默认配置参数 ====================
DEFAULT_BASE_DIR = "newresult/t-grasp_n-100_o-0_s-170912623/openvla-7b_2024/t-grasp_n-100_o-0_s-170912623/openvla-7b_2024"

# 物品类别映射
OBJECT_CATEGORIES = {
    "bottle": ["apple_juice", "orange_juice", "milk_bottle", "water_bottle"],
    "can": ["coke_can", "pepsi_can", "sprite_can", "redbull_can"],
    "fruit": ["apple", "orange", "banana"],
    "container": ["bowl", "cup", "mug"],
}

# 通用物品列表
COMMON_OBJECTS = [
    "apple_juice", "orange_juice", "coke_can", "pepsi_can",
    "apple", "orange", "bowl", "cup"
]


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


def get_available_objects_in_scene(options):
    """获取场景中已定义的所有物品
    
    Returns:
        set: 场景中所有物品名称
    """
    available_objects = set()
    
    # 从 obj_init_options 中获取
    if 'obj_init_options' in options:
        obj_init_options = options['obj_init_options']
        # grasp 场景：obj_init_options 直接是 init_xy/orientation
        if "init_xy" in obj_init_options:
            model_id = options.get("model_id")
            if model_id:
                available_objects.add(model_id)
        else:
            available_objects.update(obj_init_options.keys())
    
    # 从 model_ids 中获取（model_ids 是列表）
    if 'model_ids' in options:
        if isinstance(options['model_ids'], list):
            available_objects.update(options['model_ids'])
        elif isinstance(options['model_ids'], dict):
            available_objects.update(options['model_ids'].values())
    
    return available_objects


def get_similar_objects(object_name, available_objects=None):
    """获取与指定物品类似的物品列表
    
    Args:
        object_name: 原始物品名称
        available_objects: 场景中可用的物品集合（可选）
    
    Returns:
        类似物品列表（仅包含场景中可用的）
    """
    similar = []
    
    # 查找物品所属类别
    for category, objects in OBJECT_CATEGORIES.items():
        if object_name in objects:
            # 返回同类别的其他物品
            similar = [obj for obj in objects if obj != object_name]
            break
    
    # 如果找不到类别，使用通用物品列表
    if not similar:
        similar = [obj for obj in COMMON_OBJECTS if obj != object_name]
    
    # 如果提供了可用物品列表，只返回场景中存在的物品
    if available_objects:
        similar = [obj for obj in similar if obj in available_objects]
    
    return similar


def replace_object(options, old_object_name, new_object_name):
    """替换物品
    
    Args:
        options: options字典
        old_object_name: 原物品名称
        new_object_name: 新物品名称
    
    Returns:
        是否替换成功
    """
    obj_init_options = options.get("obj_init_options", {})
    if "init_xy" in obj_init_options:
        # grasp 场景：仅替换 model_id，保持位置配置不变
        if options.get("model_id") != old_object_name:
            print(f"❌ 原物品 '{old_object_name}' 不在配置中")
            return False
        options["model_id"] = new_object_name
        return True
    if old_object_name not in obj_init_options:
        print(f"❌ 原物品 '{old_object_name}' 不在配置中")
        return False
    
    # 保存原物品的位置配置
    old_config = options["obj_init_options"][old_object_name].copy()
    
    # 删除原物品
    del options["obj_init_options"][old_object_name]
    
    # 添加新物品（使用相同的位置配置）
    options["obj_init_options"][new_object_name] = old_config
    
    # 更新 model_ids（model_ids 是列表）
    source_obj_id = options["source_obj_id"]
    if isinstance(options["model_ids"], list):
        if options["model_ids"][source_obj_id] == old_object_name:
            options["model_ids"][source_obj_id] = new_object_name
    elif isinstance(options["model_ids"], dict):
        if options["model_ids"][source_obj_id] == old_object_name:
            options["model_ids"][source_obj_id] = new_object_name
    
    return True


def main():
    parser = argparse.ArgumentParser(description="替换物品（仅修改配置）")
    
    parser.add_argument('episode_dir', type=str, help="Episode工作目录（包含options.json）")
    parser.add_argument('--new_object', type=str, help="新物品名称（如果不指定，显示可用物品列表）")
    parser.add_argument('--list_available', action='store_true', help="列出场景中所有可用物品")
    
    args = parser.parse_args()
    
    episode_dir = args.episode_dir
    
    if not os.path.exists(episode_dir):
        print(f"❌ Episode目录不存在: {episode_dir}")
        return 1
    
    print("=" * 70)
    print("🔄 替换物品策略")
    print("=" * 70)
    print(f"📍 目录: {episode_dir}")
    print("=" * 70)
    
    # 备份原始配置
    backup_original_options(episode_dir)
    
    # 加载原始备份配置（用于获取真正的原始配置）
    backup_path = os.path.join(episode_dir, "origin.json")
    if os.path.exists(backup_path):
        with open(backup_path, 'r') as f:
            options = json.load(f)
        print(f"✅ 从备份加载原始配置: origin.json")
    else:
        options = load_options(episode_dir)
        print(f"⚠️  备份不存在，使用当前配置")
    
    # 获取源物体（兼容 grasp/move 两种配置结构）
    if "model_ids" in options and "source_obj_id" in options:
        source_obj_id = options["source_obj_id"]
        if isinstance(options["model_ids"], list):
            source_obj = options["model_ids"][source_obj_id]
        else:
            source_obj = options["model_ids"][source_obj_id]
    else:
        source_obj = options.get("model_id", "unknown")
    
    print(f"\n🎯 当前源物体: {source_obj}")
    
    # 获取场景中可用的物品
    available_objects = get_available_objects_in_scene(options)
    print(f"\n📦 场景中已定义的物品 ({len(available_objects)}个):")
    for obj in sorted(available_objects):
        marker = "⭐" if obj == source_obj else "  "
        print(f"   {marker} {obj}")
    
    # 如果只是列出可用物品
    if args.list_available:
        return 0
    
    # 获取类似物品（仅场景中存在的）
    similar_objects = get_similar_objects(source_obj, available_objects)
    
    # 确定新物品
    if args.new_object:
        new_object = args.new_object
    else:
        print(f"\n💡 推荐的类似物品:")
        if similar_objects:
            for i, obj in enumerate(similar_objects, 1):
                print(f"   {i}. {obj}")
        else:
            print("   （未找到场景中可用的类似物品）")
        
        print(f"\n❌ 请使用 --new_object 参数指定要替换的物品")
        print(f"   示例: python3 {os.path.basename(__file__)} <episode_dir> --new_object coke_can")
        return 1
    
    # 验证新物品是否在场景中（仅警告，不阻止）
    if new_object not in available_objects:
        print(f"\n⚠️  警告: 物品 '{new_object}' 不在场景的已定义物品中")
        print(f"   可用的物品: {', '.join(sorted(available_objects))}")
        print(f"   继续执行替换...")
    
    print(f"\n🔄 执行替换: {source_obj} → {new_object}")
    
    # 执行替换
    if not replace_object(options, source_obj, new_object):
        return 1
    
    # 保存配置
    save_options(episode_dir, options)
    
    print("\n" + "=" * 70)
    print("✅ 配置已更新！")
    print("=" * 70)
    print(f"🔄 替换: {source_obj} → {new_object}")
    print(f"💾 配置文件: {os.path.join(episode_dir, 'options.json')}")
    print(f"📦 原始备份: {os.path.join(episode_dir, 'origin.json')}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
