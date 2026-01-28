#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新运行已保存的Guided组配置
功能：读取 saved_configs/ 中已保存的 config_guided.yaml，重新运行实验
"""

import argparse
import json
import yaml
import sys
import time
from pathlib import Path
from typing import List, Dict

# 添加路径
_ROOT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.validation import run_experiment


def parse_scene_list(scenes_str: str) -> List[int]:
    """解析场景列表字符串
    
    支持格式：
    - "0,4,7,9" -> [0, 4, 7, 9]
    - "0-10" -> [0, 1, 2, ..., 10]
    - "0,5-10,15" -> [0, 5, 6, 7, 8, 9, 10, 15]
    """
    scenes = []
    parts = scenes_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            scenes.extend(range(start, end + 1))
        else:
            scenes.append(int(part))
    
    return sorted(set(scenes))


def get_available_scenes(config_dir: Path) -> List[int]:
    """获取所有可用的场景编号"""
    scenes = []
    for scene_dir in sorted(config_dir.glob("scene_*")):
        if scene_dir.is_dir():
            scene_id = int(scene_dir.name.split('_')[1])
            scenes.append(scene_id)
    return sorted(scenes)


def fix_bddl_paths(config: Dict, scene_id: int, config_dir: Path) -> Dict:
    """修正配置中的BDDL文件路径"""
    scene_dir = config_dir / f"scene_{scene_id:04d}"
    
    for group in config.get('groups', []):
        if 'stages' in group:
            for stage in group['stages']:
                if 'bddl_file' in stage:
                    old_path = stage['bddl_file']
                    # 提取文件名
                    filename = Path(old_path).name
                    # 构建新的绝对路径
                    new_path = scene_dir / filename
                    stage['bddl_file'] = str(new_path.absolute())
    
    return config


def run_single_scene(
    scene_id: int,
    config_dir: Path,
    output_dir: Path,
    model,
    model_config
) -> Dict:
    """运行单个场景的guided组"""
    
    print(f"\n{'='*70}")
    print(f"Scene {scene_id:04d} - Guided组")
    print(f"{'='*70}")
    
    # 读取配置文件
    config_path = config_dir / f"scene_{scene_id:04d}" / "config_guided.yaml"
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return {
            'scene_id': scene_id,
            'success': False,
            'error': 'config_not_found'
        }
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修正BDDL路径
    config = fix_bddl_paths(config, scene_id, config_dir)
    
    # 修改输出路径
    scene_output_dir = output_dir / f"scene_{scene_id:04d}"
    config['output']['results_dir'] = str(scene_output_dir / "guided")
    
    # 提取guided组配置
    guided_group_config = None
    for group in config.get('groups', []):
        if group.get('name') == 'guided':
            guided_group_config = group
            break
    
    if not guided_group_config:
        print(f"❌ 配置中找不到guided组")
        return {
            'scene_id': scene_id,
            'success': False,
            'error': 'guided_group_not_found'
        }
    
    # 运行实验
    try:
        env, summary = run_experiment.run_group_with_cache(
            env=None,
            config=config,
            group_config=guided_group_config,
            group_name="guided",
            output_dir=scene_output_dir,
            model=model,
            model_config=model_config,
            task_suite=None,
            task_id=0
        )
        
        if env is not None:
            try:
                env.close()
            except:
                pass
        
        success = summary.get('success_count', 0) > 0
        
        print(f"\n{'='*70}")
        if success:
            print(f"✅ Scene {scene_id:04d} 成功")
        else:
            print(f"❌ Scene {scene_id:04d} 失败")
        print(f"{'='*70}\n")
        
        return {
            'scene_id': scene_id,
            'success': success,
            'summary': summary
        }
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ Scene {scene_id:04d} 运行异常: {e}")
        print(f"{'='*70}\n")
        
        import traceback
        traceback.print_exc()
        
        return {
            'scene_id': scene_id,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="重新运行已保存的Guided组配置"
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help='指定运行的场景（如 "0,4,7,9" 或 "0-10"）'
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="运行所有场景"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="./saved_configs",
        help="保存的配置目录（默认：./saved_configs）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./new_results",
        help="新结果输出目录（默认：./new_results）"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Pi0.5模型checkpoint目录（默认：./checkpoints）"
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    config_dir = Path(args.config_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not config_dir.exists():
        print(f"❌ 配置目录不存在: {config_dir}")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有可用场景
    available_scenes = get_available_scenes(config_dir)
    print(f"找到 {len(available_scenes)} 个可用场景")
    
    # 确定要运行的场景
    if args.all:
        selected_scenes = available_scenes
    elif args.scenes:
        requested_scenes = parse_scene_list(args.scenes)
        selected_scenes = [s for s in requested_scenes if s in available_scenes]
        if len(selected_scenes) < len(requested_scenes):
            missing = set(requested_scenes) - set(selected_scenes)
            print(f"⚠️  警告：部分场景不存在: {sorted(missing)}")
    else:
        print("❌ 请指定 --scenes 或 --all")
        return
    
    print(f"将运行 {len(selected_scenes)} 个场景: {selected_scenes}\n")
    
    # 加载模型（一次性）
    print("="*70)
    print("加载Pi0.5模型...")
    print("="*70)
    model, model_config = run_experiment.load_pi05_model(args.checkpoint_dir)
    
    if model is None:
        print("❌ 模型加载失败")
        return
    
    print("✅ 模型加载完成\n")
    
    # 运行所有场景（串行）
    results = []
    start_time = time.time()
    
    for idx, scene_id in enumerate(selected_scenes, 1):
        print(f"\n进度: {idx}/{len(selected_scenes)}")
        result = run_single_scene(
            scene_id=scene_id,
            config_dir=config_dir,
            output_dir=output_dir,
            model=model,
            model_config=model_config
        )
        results.append(result)
    
    elapsed_time = time.time() - start_time
    
    # 生成总结报告
    success_count = sum(1 for r in results if r.get('success', False))
    total_count = len(results)
    
    summary_report = {
        'total_scenes': total_count,
        'success_count': success_count,
        'failed_count': total_count - success_count,
        'success_rate': success_count / total_count if total_count > 0 else 0,
        'elapsed_time_seconds': elapsed_time,
        'results': results
    }
    
    # 保存报告
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    # 打印统计
    print(f"\n{'='*70}")
    print(f"运行完成！")
    print(f"{'='*70}")
    print(f"总场景数: {total_count}")
    print(f"成功: {success_count}")
    print(f"失败: {total_count - success_count}")
    print(f"成功率: {success_count/total_count*100:.1f}%")
    print(f"总耗时: {elapsed_time/60:.1f} 分钟")
    print(f"\n结果保存在: {output_dir}")
    print(f"总结报告: {summary_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
