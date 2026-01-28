#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成单个遮挡物实验的BDDL文件和配置文件
将不同物品放到黄油上，遮挡物数量为1
"""

import os
from pathlib import Path
from typing import List, Dict

# 物体配置
OBJECTS = {
    'butter': {
        'type': 'butter',
        'name': 'butter',
        'display_name': 'butter box'
    },
    'pudding': {
        'type': 'chocolate_pudding',
        'name': 'chocolate_pudding',
        'display_name': 'chocolate pudding'
    },
    'cheese': {
        'type': 'cream_cheese',
        'name': 'cream_cheese',
        'display_name': 'cream cheese'
    },
    'plate': {
        'type': 'plate',
        'name': 'plate',
        'display_name': 'plate'
    },
    'bowl': {
        'type': 'akita_black_bowl',
        'name': 'akita_black_bowl',
        'display_name': 'black bowl'
    },
    'tomato_sauce': {
        'type': 'tomato_sauce',
        'name': 'tomato_sauce',
        'display_name': 'tomato sauce'
    },
    'ketchup': {
        'type': 'ketchup',
        'name': 'ketchup',
        'display_name': 'ketchup'
    },
    'alphabet_soup': {
        'type': 'alphabet_soup',
        'name': 'alphabet_soup',
        'display_name': 'alphabet soup'
    },
    'orange_juice': {
        'type': 'orange_juice',
        'name': 'orange_juice',
        'display_name': 'orange juice'
    },
    'milk': {
        'type': 'milk',
        'name': 'milk',
        'display_name': 'milk'
    }
}

# 位置配置（参考标准LIBERO和备份文件）
POSITION = "(0.025 -0.125 0.075 -0.075)"
BASKET_POSITION = "(-0.01 0.25 0.01 0.27)"

# BDDL模板 - Baseline版本（1个遮挡物在黄油上）
BDDL_TEMPLATE_BASELINE = """(define (problem LIBERO_Living_Room_Tabletop_Manipulation)
  (:domain robosuite)
  (:language put the {bottom_display} in the basket)
    (:regions
      (basket_init_region
          (:target living_room_table)
          (:ranges (
              {basket_pos}
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      ({bottom_name}_init_region
          (:target living_room_table)
          (:ranges (
              {obj_pos}
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (contain_region
          (:target basket_1)
      )
    )

  (:fixtures
    living_room_table - living_room_table
  )

  (:objects
    {bottom_name}_1 - {bottom_type}
    {top_name}_1 - {top_type}
    basket_1 - basket
  )

  (:obj_of_interest
    {bottom_name}_1
    basket_1
  )

  (:init
    (On {bottom_name}_1 living_room_table_{bottom_name}_init_region)
    (On {top_name}_1 {bottom_name}_1)
    (On basket_1 living_room_table_basket_init_region)
  )

  (:goal
    (And (In {bottom_name}_1 basket_1_contain_region))
  )

)
"""

# BDDL模板 - Guided Stage 1（移除遮挡物）
BDDL_TEMPLATE_GUIDED_STAGE1 = """(define (problem LIBERO_Living_Room_Tabletop_Manipulation)
  (:domain robosuite)
  (:language put the {top_display} in the basket)
    (:regions
      (basket_init_region
          (:target living_room_table)
          (:ranges (
              {basket_pos}
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      ({bottom_name}_init_region
          (:target living_room_table)
          (:ranges (
              {obj_pos}
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (contain_region
          (:target basket_1)
      )
    )

  (:fixtures
    living_room_table - living_room_table
  )

  (:objects
    {bottom_name}_1 - {bottom_type}
    {top_name}_1 - {top_type}
    basket_1 - basket
  )

  (:obj_of_interest
    {top_name}_1
    basket_1
  )

  (:init
    (On {bottom_name}_1 living_room_table_{bottom_name}_init_region)
    (On {top_name}_1 {bottom_name}_1)
    (On basket_1 living_room_table_basket_init_region)
  )

  (:goal
    (And (In {top_name}_1 basket_1_contain_region))
  )

)
"""

# BDDL模板 - Guided Stage 2（只有底部物体）
BDDL_TEMPLATE_STAGE2 = """(define (problem LIBERO_Living_Room_Tabletop_Manipulation)
  (:domain robosuite)
  (:language put the {bottom_display} in the basket)
    (:regions
      (basket_init_region
          (:target living_room_table)
          (:ranges (
              {basket_pos}
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      ({bottom_name}_init_region
          (:target living_room_table)
          (:ranges (
              {obj_pos}
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (contain_region
          (:target basket_1)
      )
    )

  (:fixtures
    living_room_table - living_room_table
  )

  (:objects
    {bottom_name}_1 - {bottom_type}
    basket_1 - basket
  )

  (:obj_of_interest
    {bottom_name}_1
    basket_1
  )

  (:init
    (On {bottom_name}_1 living_room_table_{bottom_name}_init_region)
    (On basket_1 living_room_table_basket_init_region)
  )

  (:goal
    (And (In {bottom_name}_1 basket_1_contain_region))
  )

)
"""

# YAML配置模板
YAML_CONFIG_TEMPLATE = """experiment:
  name: "single_{bottom_key}_{top_key}"
  description: "单遮挡物实验：{bottom_display}(底) + 1个{top_display}(顶)"

task:
  suite: "custom"
  task_name: "single_{bottom_key}_{top_key}"

groups:
  # Guided: 分两步，先移除1个{top_display}，再移{bottom_display}
  - name: "guided"
    description: "分两步：先移除1个{top_display}，再移动{bottom_display}"
    use_obstruction: false
    stages:
      - stage_name: "remove_top_{top_key}"
        bddl_file: "experiments/obstruction/custom_bddl_files/single/{bottom_key}_{top_key}_stacked.bddl"
        instruction: "put the {top_display} in the basket"
        target_object: "{top_name}_1"
      
      - stage_name: "move_bottom_{bottom_key}"
        bddl_file: "experiments/obstruction/custom_bddl_files/single/{bottom_key}_only.bddl"
        instruction: "put the {bottom_display} in the basket"
        target_object: "{bottom_name}_1"
  
  # Baseline: 直接移动底层{bottom_display}（1个{top_display}遮挡）
  - name: "baseline"
    description: "直接移动底层的{bottom_display}到篮子（1个{top_display}在上方遮挡）"
    bddl_file: "experiments/obstruction/custom_bddl_files/single/{bottom_key}_{top_key}_stacked_baseline.bddl"
    instruction: "put the {bottom_display} in the basket"
    use_obstruction: false
    use_bddl_stacking: true

execution:
  episodes_per_group: 3
  max_steps_per_episode: 300
  seed_start: 43
  checkpoint_dir: "./pi05_libero"

output:
  results_dir: "./experiments/obstruction/results/single/{bottom_key}_{top_key}"
  save_images: true
  save_videos: false
"""


def generate_bddl_files(bottom_key: str, top_key: str, output_dir: Path):
    """生成一组实验的BDDL文件"""
    
    bottom_obj = OBJECTS[bottom_key]
    top_obj = OBJECTS[top_key]
    
    # 准备替换参数
    params = {
        'bottom_name': bottom_obj['name'],
        'bottom_type': bottom_obj['type'],
        'bottom_display': bottom_obj['display_name'],
        'top_name': top_obj['name'],
        'top_type': top_obj['type'],
        'top_display': top_obj['display_name'],
        'obj_pos': POSITION,
        'basket_pos': BASKET_POSITION
    }
    
    # 1. Baseline BDDL (堆叠状态，goal是bottom)
    baseline_bddl = BDDL_TEMPLATE_BASELINE.format(**params)
    baseline_file = output_dir / f"{bottom_key}_{top_key}_stacked_baseline.bddl"
    baseline_file.write_text(baseline_bddl)
    print(f"  ✓ 创建: {baseline_file.name}")
    
    # 2. Guided Stage 1 BDDL (堆叠状态，goal是top)
    guided_stage1_bddl = BDDL_TEMPLATE_GUIDED_STAGE1.format(**params)
    guided_stage1_file = output_dir / f"{bottom_key}_{top_key}_stacked.bddl"
    guided_stage1_file.write_text(guided_stage1_bddl)
    print(f"  ✓ 创建: {guided_stage1_file.name}")
    
    # 3. Guided Stage 2 BDDL (只有bottom) - 所有配置共用同一个butter_only.bddl
    if not (output_dir / f"{bottom_key}_only.bddl").exists():
        stage2_bddl = BDDL_TEMPLATE_STAGE2.format(**params)
        stage2_file = output_dir / f"{bottom_key}_only.bddl"
        stage2_file.write_text(stage2_bddl)
        print(f"  ✓ 创建: {stage2_file.name}")


def generate_config_file(bottom_key: str, top_key: str, output_dir: Path):
    """生成配置文件"""
    
    bottom_obj = OBJECTS[bottom_key]
    top_obj = OBJECTS[top_key]
    
    params = {
        'bottom_key': bottom_key,
        'bottom_name': bottom_obj['name'],
        'bottom_display': bottom_obj['display_name'],
        'top_key': top_key,
        'top_name': top_obj['name'],
        'top_display': top_obj['display_name']
    }
    
    config_content = YAML_CONFIG_TEMPLATE.format(**params)
    config_file = output_dir / f"exp_single_{bottom_key}_{top_key}.yaml"
    config_file.write_text(config_content)
    print(f"  ✓ 创建: {config_file.name}")


def main():
    """批量生成所有单遮挡物实验"""
    
    # 设置路径
    base_dir = Path(__file__).parent.parent
    bddl_dir = base_dir / "custom_bddl_files" / "single"
    config_dir = base_dir / "configs"
    
    bddl_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义底部物体（可以有多个底部物体）
    bottom_objects = [
        'butter',          # 黄油
        'pudding',         # 巧克力布丁
        'cheese'           # 奶油芝士
    ]
    
    # 定义所有遮挡物
    top_objects = [
        'plate',           # 盘子
        'bowl',            # 碗
        'tomato_sauce',    # 番茄酱
        'ketchup',         # 番茄酱2
        'alphabet_soup',   # 字母汤罐头
        'orange_juice',    # 橙汁
        'milk'             # 牛奶
    ]
    
    print("\n" + "="*60)
    print("🚀 开始批量生成单遮挡物实验")
    print(f"   底部物体: {len(bottom_objects)}种 - {', '.join(bottom_objects)}")
    print(f"   遮挡物数量: 1个")
    print(f"   遮挡物种类: {len(top_objects)}种")
    print("="*60 + "\n")
    
    total_count = 0
    all_combinations = []
    
    for bottom_key in bottom_objects:
        bottom_display = OBJECTS[bottom_key]['display_name']
        print(f"📦 底部物体: {bottom_key} ({bottom_display})")
        print("-" * 60)
        
        for top_key in top_objects:
            total_count += 1
            all_combinations.append((bottom_key, top_key))
            top_display = OBJECTS[top_key]['display_name']
            print(f"  [{total_count}] 生成: {bottom_key}(底) + {top_key}(顶)")
            
            # 生成BDDL文件
            generate_bddl_files(bottom_key, top_key, bddl_dir)
            
            # 生成配置文件
            generate_config_file(bottom_key, top_key, config_dir)
        
        print()
    
    print("="*60)
    print("✅ 批量生成完成！")
    print("="*60)
    print(f"\nBDDL文件位置: {bddl_dir}")
    print(f"配置文件位置: {config_dir}")
    print(f"\n共生成 {total_count} 组实验:")
    print(f"  底部物体: {len(bottom_objects)}种")
    print(f"  遮挡物: {len(top_objects)}种")
    print(f"  总组合: {len(bottom_objects)} × {len(top_objects)} = {total_count}组")
    
    print("\n详细列表:")
    for bottom, top in all_combinations:
        print(f"  - {bottom}_{top}")
    
    print("\n📋 运行命令示例:")
    print("  # 单个实验")
    print(f"  python3 experiments/obstruction/scripts/run_experiment.py --config experiments/obstruction/configs/exp_single_{all_combinations[0][0]}_{all_combinations[0][1]}.yaml")
    print("\n  # 批量运行")
    print("  bash experiments/obstruction/run_all_pair_experiments.sh")
    print()
    for top in top_objects:
        print(f"  python3 experiments/obstruction/scripts/run_experiment.py --config experiments/obstruction/configs/exp_single_obs_{bottom_key}_{top}.yaml")
    
    print("\n或者创建批量运行脚本 (run_all_single_obs_experiments.sh)")
    print()


if __name__ == "__main__":
    main()
