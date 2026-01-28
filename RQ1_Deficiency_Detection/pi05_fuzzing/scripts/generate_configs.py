#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成遮挡实验配置文件
根据物品组合批量生成BDDL和YAML配置
"""

import os
from pathlib import Path
from datetime import datetime

# 物品配置
OBJECTS_CONFIG = {
    # 下方物品（目标物）
    "bottom_objects": {
        "cream_cheese": {
            "bddl_name": "cream_cheese_1",
            "mujoco_pattern": "cream_cheese_1",
            "description": "cream cheese box",
            "init_region": "(0.025 -0.125 0.075 -0.075)"
        },
        "butter": {
            "bddl_name": "butter_1",
            "mujoco_pattern": "butter_1",
            "description": "butter box",
            "init_region": "(0.025 -0.125 0.075 -0.075)"
        },
        "chocolate_pudding": {
            "bddl_name": "chocolate_pudding_1",
            "mujoco_pattern": "chocolate_pudding_1",
            "description": "chocolate pudding",
            "init_region": "(0.025 -0.125 0.075 -0.075)"
        }
    },
    
    # 上方物品（遮挡物）
    "top_objects": {
        "akita_black_bowl": {
            "bddl_name": "akita_black_bowl_1",
            "mujoco_pattern": "akita_black_bowl_1",
            "description": "black bowl",
            "init_region": "(-0.175 0.035 -0.125 0.085)"
        },
        "alphabet_soup": {
            "bddl_name": "alphabet_soup_1",
            "mujoco_pattern": "alphabet_soup_1",
            "description": "alphabet soup can",
            "init_region": "(-0.175 0.035 -0.125 0.085)"
        },
        "tomato_sauce": {
            "bddl_name": "tomato_sauce_1",
            "mujoco_pattern": "tomato_sauce_1",
            "description": "tomato sauce jar",
            "init_region": "(-0.175 0.035 -0.125 0.085)"
        },
        "ketchup": {
            "bddl_name": "ketchup_1",
            "mujoco_pattern": "ketchup_1",
            "description": "ketchup bottle",
            "init_region": "(-0.175 0.035 -0.125 0.085)"
        },
        "milk": {
            "bddl_name": "milk_1",
            "mujoco_pattern": "milk_1",
            "description": "milk carton",
            "init_region": "(-0.175 0.035 -0.125 0.085)"
        }
    }
}

# 篮子配置（固定）
BASKET_CONFIG = {
    "bddl_name": "basket_1",
    "init_region": "(-0.01 0.25 0.01 0.27)"
}

# Plate配置（固定，用于放置遮挡物）
PLATE_CONFIG = {
    "bddl_name": "plate_1",
    "mujoco_pattern": "plate_1",
    "description": "plate",
    "init_region": "(-0.17 -0.15 -0.15 -0.13)"  # 左下方，容易放置遮挡物
}

# 场景类型选择
SCENE_INFO = {
    # 大部分物品在 LIVING_ROOM_SCENE2
    "default": {
        "scene": "LIVING_ROOM_SCENE2",
        "table": "living_room_table",
        "available_objects": ["alphabet_soup", "cream_cheese", "ketchup", "butter", "milk", "basket"]
    },
    # plate 和 bowl 在 LIVING_ROOM_SCENE4
    "with_plate_bowl": {
        "scene": "LIVING_ROOM_SCENE4",
        "table": "living_room_table",
        "available_objects": ["akita_black_bowl", "chocolate_pudding", "wooden_tray"]
    }
}

def select_scene_for_objects(bottom_obj, top_obj):
    """根据物品组合选择合适的场景"""
    # 所有配置都使用 SCENE4（因为需要 bowl 和 plate）
    return "with_plate_bowl"

def generate_bddl(bottom_obj, top_obj, output_path):
    """生成BDDL文件"""
    
    bottom_cfg = OBJECTS_CONFIG["bottom_objects"][bottom_obj]
    top_cfg = OBJECTS_CONFIG["top_objects"][top_obj]
    scene_type = select_scene_for_objects(bottom_obj, top_obj)
    scene_info = SCENE_INFO[scene_type]
    
    # 确定需要的所有物品
    task_objects = [bottom_obj, top_obj]
    
    # 构建BDDL内容
    bddl_content = f"""(define (problem LIBERO_Living_Room_Tabletop_Manipulation)
    (:domain robosuite)
    (:language put the {bottom_cfg['description']} in the basket)
    
    (:regions
      (basket_init_region
          (:target {scene_info['table']})
          (:ranges (
              {BASKET_CONFIG['init_region']}
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (plate_init_region
          (:target {scene_info['table']})
          (:ranges (
              {PLATE_CONFIG['init_region']}
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      ({bottom_cfg['bddl_name']}_init_region
          (:target {scene_info['table']})
          (:ranges (
              {bottom_cfg['init_region']}
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      ({top_cfg['bddl_name']}_init_region
          (:target {scene_info['table']})
          (:ranges (
              {top_cfg['init_region']}
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (contain_region
          (:target {BASKET_CONFIG['bddl_name']})
      )
    )

    (:fixtures
        {scene_info['table']} - {scene_info['table']}
    )

    (:objects
        {BASKET_CONFIG['bddl_name']} - basket
        {PLATE_CONFIG['bddl_name']} - plate
        {bottom_cfg['bddl_name']} - {bottom_obj}
        {top_cfg['bddl_name']} - {top_obj}
    )

    (:obj_of_interest 
        {bottom_cfg['bddl_name']}
        {BASKET_CONFIG['bddl_name']}
        {PLATE_CONFIG['bddl_name']}
        {top_cfg['bddl_name']}
    )

    (:init
        (On {BASKET_CONFIG['bddl_name']} {scene_info['table']}_basket_init_region)
        (On {PLATE_CONFIG['bddl_name']} {scene_info['table']}_plate_init_region)
        (On {bottom_cfg['bddl_name']} {scene_info['table']}_{bottom_cfg['bddl_name']}_init_region)
        (On {top_cfg['bddl_name']} {scene_info['table']}_{top_cfg['bddl_name']}_init_region)
    )

    (:goal
        (And
            (In {bottom_cfg['bddl_name']} {BASKET_CONFIG['bddl_name']}_contain_region)
            (On {top_cfg['bddl_name']} {PLATE_CONFIG['bddl_name']})
        )
    )
)
"""
    
    # 写入文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(bddl_content)
    
    print(f"  ✓ 生成BDDL: {output_path.name}")

def generate_yaml(bottom_obj, top_obj, exp_id, output_path, bddl_rel_path):
    """生成YAML配置文件"""
    
    bottom_cfg = OBJECTS_CONFIG["bottom_objects"][bottom_obj]
    top_cfg = OBJECTS_CONFIG["top_objects"][top_obj]
    
    yaml_content = f"""# ============================================================
# 自动生成的遮挡实验配置
# 实验ID: {exp_id}
# 下方物品: {bottom_obj}
# 上方物品: {top_obj}
# 任务类型: put-in (篮子)
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# ============================================================

experiment:
  name: "exp{exp_id}_{top_obj}_on_{bottom_obj}_basket"
  description: "测试{top_cfg['description']}遮挡{bottom_cfg['description']}放入篮子任务的影响"

# 任务定义
task:
  suite: "custom"
  task_name: "put_the_{bottom_obj}_in_the_basket"
  bddl_file: "{bddl_rel_path}"

# 场景物体配置
scene:
  # 目标物体（被操作物，下方）
  target_object:
    bddl_name: "{bottom_cfg['bddl_name']}"
    mujoco_body_pattern: "{bottom_cfg['mujoco_pattern']}"
    description: "{bottom_cfg['description']}"
    
  # 遮挡物（上方）
  obstruction_object:
    bddl_name: "{top_cfg['bddl_name']}"
    mujoco_body_pattern: "{top_cfg['mujoco_pattern']}"
    description: "{top_cfg['description']}"
    
  # 遮挡配置
  obstruction:
    enabled: true
    type: "stack_on_top"
    offset: [0.0, 0.0, 0.08]  # 只改变z轴，xy自动对齐

# 实验组配置
groups:
  # Group 1: 基线组（无遮挡）
  - name: "baseline"
    description: "基线组：无遮挡场景"
    use_obstruction: false
    instruction: "pick up the {bottom_cfg['description']} and put it in the basket"
    
  # Group 2: 遮挡组 - 原始指令
  - name: "obstructed_original"
    description: "遮挡组：{bottom_cfg['description']}被{top_cfg['description']}遮挡 + 原始指令"
    use_obstruction: true
    instruction: "pick up the {bottom_cfg['description']} and put it in the basket"
    
  # Group 3: 遮挡组 - 引导指令
  - name: "obstructed_guided"
    description: "遮挡组：{bottom_cfg['description']}被{top_cfg['description']}遮挡 + 引导指令"
    use_obstruction: true
    instruction: "put the {top_cfg['description']} on the plate and pick up the {bottom_cfg['description']} and put it in the basket"

# 运行参数
execution:
  episodes_per_group: 3
  max_steps_per_episode: 400
  checkpoint_dir: "./pi05_libero"
  libero_env: "libero"
  seed_start: {42 + exp_id}

# 输出配置
output:
  results_dir: "./experiments/obstruction/results"
  save_images: true
  save_videos: false

# 模型配置
model:
  name: "pi0.5"
  config_path: "./pi05_libero/config.yaml"
  use_cache: true
  cache_length: 10
"""
    
    # 写入文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"  ✓ 生成YAML: {output_path.name}")

def generate_all_configs():
    """生成所有配置文件"""
    
    # 获取脚本目录，向上一级到实验根目录
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    bddl_dir = project_root / "custom_bddl_files"
    yaml_dir = project_root / "configs/task2_put_in_basket"
    
    print("="*60)
    print("🚀 开始生成遮挡实验配置")
    print("="*60)
    print(f"\n配置方案:")
    print(f"  下方物品（3种）: {list(OBJECTS_CONFIG['bottom_objects'].keys())}")
    print(f"  上方物品（5种）: {list(OBJECTS_CONFIG['top_objects'].keys())}")
    print(f"  辅助物品: plate (固定位置，用于放置遮挡物)")
    print(f"  总计: 3 × 5 = 15 组配置\n")
    
    exp_id = 1  # 从1开始
    combinations = []
    
    for bottom_obj in OBJECTS_CONFIG["bottom_objects"].keys():
        for top_obj in OBJECTS_CONFIG["top_objects"].keys():
            print(f"\n[实验 {exp_id}] {top_obj} → {bottom_obj} → basket")
            
            # 文件名
            combo_name = f"{top_obj}_on_{bottom_obj}_basket"
            bddl_filename = f"{combo_name}.bddl"
            yaml_filename = f"exp{exp_id}_{combo_name}.yaml"
            
            # 路径
            bddl_path = bddl_dir / bddl_filename
            yaml_path = yaml_dir / yaml_filename
            bddl_rel_path = f"experiments/obstruction/custom_bddl_files/{bddl_filename}"
            
            # 生成文件
            generate_bddl(bottom_obj, top_obj, bddl_path)
            generate_yaml(bottom_obj, top_obj, exp_id, yaml_path, bddl_rel_path)
            
            combinations.append({
                "exp_id": exp_id,
                "top": top_obj,
                "bottom": bottom_obj,
                "bddl": str(bddl_path),
                "yaml": str(yaml_path)
            })
            
            exp_id += 1
    
    print("\n" + "="*60)
    print("✅ 配置生成完成！")
    print("="*60)
    print(f"\n生成文件:")
    print(f"  BDDL文件: {len(combinations)} 个")
    print(f"  YAML文件: {len(combinations)} 个")
    print(f"\n文件位置:")
    print(f"  BDDL: {bddl_dir}")
    print(f"  YAML: {yaml_dir}")
    
    # 生成实验列表文件
    list_file = project_root / "experiments/obstruction/实验配置清单.txt"
    with open(list_file, 'w') as f:
        f.write("遮挡实验配置清单\n")
        f.write("="*60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总计: {len(combinations)} 组实验\n\n")
        
        for combo in combinations:
            f.write(f"实验 {combo['exp_id']}: {combo['top']} → {combo['bottom']} → basket\n")
            f.write(f"  YAML: configs/task2_put_in_basket/exp{combo['exp_id']}_{combo['top']}_on_{combo['bottom']}_basket.yaml\n")
            f.write(f"  BDDL: custom_bddl_files/{combo['top']}_on_{combo['bottom']}_basket.bddl\n\n")
    
    print(f"\n📋 实验清单已保存: {list_file}")
    
    # 生成批量运行脚本
    run_script = project_root / "run_all_experiments.sh"
    with open(run_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# 批量运行所有{len(combinations)}组遮挡实验\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("set -e\n\n")
        f.write("SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n")
        f.write("cd \"${SCRIPT_DIR}\"\n\n")
        f.write(f"echo \"开始运行{len(combinations)}组遮挡实验...\"\n\n")
        
        for combo in combinations:
            yaml_rel = f"configs/task2_put_in_basket/exp{combo['exp_id']}_{combo['top']}_on_{combo['bottom']}_basket.yaml"
            f.write(f"# 实验 {combo['exp_id']}: {combo['top']} → {combo['bottom']}\n")
            f.write(f"echo \"\\n{'='*60}\"\n")
            f.write(f"echo \"运行实验 {combo['exp_id']}/{len(combinations)}: {combo['top']} → {combo['bottom']}\"\n")
            f.write(f"echo \"{'='*60}\\n\"\n")
            f.write(f"# source ~/anaconda3/etc/profile.d/conda.sh && \\\n")
            f.write(f"# conda activate env_isaaclab && \\\n")
            f.write(f"python scripts/run_experiment.py \\\n")
            f.write(f"    --config {yaml_rel}\n\n")
        
        f.write(f'echo "\\n✅ 所有{len(combinations)}组实验运行完成！"\n')
    
    os.chmod(run_script, 0o755)
    print(f"🚀 批量运行脚本已生成: {run_script}")
    print(f"\n运行命令:")
    print(f"  ./experiments/obstruction/run_all_experiments.sh")

if __name__ == "__main__":
    generate_all_configs()
