#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的优化实验脚本 - 不使用MCTS
功能：
1. 随机选择配置文件
2. 向三个BDDL文件添加相同的干扰物（位置随机，但都是干扰物不是遮挡物）
3. 运行baseline，如果失败则运行guided（最多3次）
4. 统计优化成功率
"""

import argparse
import json
import random
import re
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

_ROOT_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = _ROOT_DIR
sys.path.insert(0, str(_REPO_ROOT))

from utils.bddl_variation import BDDLDocument, BDDLVariation, _NAME_TO_TYPE_MAP
from scripts.fuzzing import run_experiment as obstruction_run


class OptimizationExperiment:
    def __init__(self, checkpoint_dir: str, output_dir: Path):
        """初始化实验"""
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        print("Loading Pi0.5 model...")
        model, model_config = obstruction_run.load_pi05_model(checkpoint_dir)
        if model is None:
            raise SystemExit("Failed to load Pi0.5 model.")
        self.model = model
        self.model_config = model_config
        
        # 统计信息 - 注意：total_scenes只统计baseline失败的场景
        self.stats = {
            "total_scenes": 0,  # 只统计baseline失败的场景数
            "baseline_success": 0,  # baseline成功的场景数（不计入total_scenes）
            "baseline_failed": 0,  # baseline失败的场景数（等于total_scenes）
            "optimization_attempted": 0,
            "optimization_success": 0,
            "scenes": [],
            # 按策略统计（仅baseline失败的场景）
            "by_strategy": {
                "random_table": {
                    "total": 0, 
                    "baseline_failed": 0, 
                    "optimization_success": 0,
                    "success_scenes": [],  # 成功的场景编号
                    "failed_scenes": []    # 失败的场景编号
                },
                "near_target": {
                    "total": 0, 
                    "baseline_failed": 0, 
                    "optimization_success": 0,
                    "success_scenes": [],
                    "failed_scenes": []
                },
                "stacked": {
                    "total": 0, 
                    "baseline_failed": 0, 
                    "optimization_success": 0,
                    "success_scenes": [],
                    "failed_scenes": []
                }
            }
        }
    
    def parse_bddl_paths(self, config: Dict) -> Dict[str, str]:
        """从配置文件中提取三个BDDL路径"""
        bddl_paths = {}
        
        for group in config.get('groups', []):
            if group.get('name') == 'guided':
                stages = group.get('stages', [])
                if len(stages) >= 2:
                    bddl_paths['stacked'] = stages[0].get('bddl_file', '')
                    bddl_paths['only'] = stages[1].get('bddl_file', '')
            elif group.get('name') == 'baseline':
                bddl_paths['baseline'] = group.get('bddl_file', '')
        
        # 验证是否找到所有路径
        if len(bddl_paths) != 3:
            raise ValueError(f"配置文件缺少必要的BDDL路径: {bddl_paths}")
        
        return bddl_paths
    
    def add_stacked_obstacle_on_top(self, doc: BDDLDocument, variation: BDDLVariation) -> BDDLDocument:
        """在最顶层遮挡物上堆叠一个新遮挡物（不修改goal）
        
        注意：堆叠的物体不需要独立的region，而是直接使用On关系
        """
        import re
        from utils.bddl_variation import _OBSTACLE_POOL
        
        mutated = doc.clone()
        
        # 1. 找到现有的堆叠关系
        stacking_relations = []  # [(object, base), ...]
        init_block = mutated._get_block("init")
        if not init_block:
            print("[策略3] 找不到init section，跳过")
            return mutated
        
        _, _, init_text = init_block
        for line in init_text.splitlines():
            match = re.match(r'\(On\s+(\w+)\s+(\w+)\)', line.strip())
            if match:
                obj = match.group(1)
                base = match.group(2)
                # 只关注物体之间的堆叠（不包括物体与fixture的关系）
                if not base.endswith("_region"):
                    stacking_relations.append((obj, base))
        
        if not stacking_relations:
            print("[策略3] 没有物体间的堆叠关系，跳过")
            return mutated
        
        print(f"[策略3] 找到堆叠关系: {stacking_relations}")
        
        # 2. 找到最顶层的物体（没有其他物体堆叠在它上面）
        all_objects = set(obj for obj, _ in stacking_relations)
        base_objects = set(base for _, base in stacking_relations)
        top_objects = [obj for obj in all_objects if obj not in base_objects]
        
        if not top_objects:
            print("[策略3] 找不到顶层物体，跳过")
            return mutated
        
        top_object = top_objects[0]  # 选择第一个顶层物体
        print(f"[策略3] 在顶层物体 {top_object} 上堆叠新遮挡物")
        
        # 3. 选择一个新的遮挡物类型
        obstacle_type = variation.rng.choice(_OBSTACLE_POOL)
        obstacle_name = f"{obstacle_type}_stacked_1"
        counter = 1
        existing_objects = mutated.get_objects()
        while obstacle_name in existing_objects:
            counter += 1
            obstacle_name = f"{obstacle_type}_stacked_{counter}"
        
        # 4. 添加到objects section
        mutated.add_object(obstacle_name, obstacle_type)
        
        # 5. 添加堆叠关系到init section（直接堆叠在顶层物体上）
        # 不需要创建region，因为堆叠物体的位置由底层物体决定
        mutated.add_init_condition(f"(On {obstacle_name} {top_object})")
        
        print(f"[策略3] ✓ 成功添加堆叠遮挡物: {obstacle_name} 在 {top_object} 上")
        
        return mutated
    
    def mutate_three_bddls(
        self, 
        stacked_path: str, 
        only_path: str, 
        baseline_path: str, 
        num_distractors: int,
        output_dir: Path,
        strategy: str = "random",
        exclude_types: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Path], str]:
        """向三个BDDL文件添加物体，支持三种策略
        
        Args:
            strategy: 变异策略
                - "random_table": 桌面随机干扰物
                - "near_target": 靠近目标物的干扰物
                - "stacked": 堆叠在顶层遮挡物上
                - "random": 随机选择上述三种之一
        
        Returns:
            (mutated_paths, actual_strategy): 变异后的BDDL路径和实际使用的策略
        """
        
        # 1. 读取三个BDDL
        stacked_doc = BDDLDocument(Path(stacked_path).read_text())
        only_doc = BDDLDocument(Path(only_path).read_text())
        baseline_doc = BDDLDocument(Path(baseline_path).read_text())
        
        # 2. 初始化变异器
        base_seed = int(time.time() * 1000) % (2**32)
        
        # 3. 如果是随机策略，随机选择一种（使用两种策略之一）
        if strategy == "random":
            variation = BDDLVariation(seed=base_seed, debug=False)
            strategies = ["random_table", "near_target"]
            actual_strategy = variation.rng.choice(strategies)
        else:
            actual_strategy = strategy
        
        print(f"使用变异策略: {actual_strategy}")
        
        # 4. 对三个BDDL应用相应策略
        mutated_stacked = stacked_doc.clone()
        mutated_only = only_doc.clone()
        mutated_baseline = baseline_doc.clone()
        
        if actual_strategy == "stacked":
            # 策略3: 堆叠在顶层遮挡物上（只对stacked和baseline，only不需要）
            variation = BDDLVariation(seed=base_seed, debug=False)
            mutated_stacked = self.add_stacked_obstacle_on_top(mutated_stacked, variation)
            mutated_baseline = self.add_stacked_obstacle_on_top(mutated_baseline, variation)
            # only不添加，因为它只有目标物
            
        else:
            # 策略1和2: 添加干扰物（使用四方位分散策略）
            # 目标干扰物数量：1到4个（随机）
            variation_init = BDDLVariation(seed=base_seed, debug=False)
            target_count = variation_init.rng.randint(1, 4)
            print(f"目标添加 {target_count} 个干扰物")
            
            # 根据策略决定位置模式
            if actual_strategy == "random_table":
                place_near_target = False
            elif actual_strategy == "near_target":
                place_near_target = True
            else:
                place_near_target = False
            
            added = 0
            attempts = 0
            max_attempts = target_count * 3  # 每个干扰物最多尝试3次
            
            # 如果是near_target策略，使用方位轮换确保分散
            direction_offset = 0
            
            while added < target_count and attempts < max_attempts:
                attempts += 1
                position_seed = base_seed + attempts
                variation = BDDLVariation(seed=position_seed, debug=False)
                
                # 保存变异前的文本，用于检测是否成功添加
                before_stacked = mutated_stacked.text
                before_only = mutated_only.text
                before_baseline = mutated_baseline.text
                
                # 对三个BDDL使用相同的seed尝试添加
                # 使用direction_hint参数指导方位选择
                variation.rng = random.Random(position_seed)
                mutated_stacked = variation.add_obstacle(
                    mutated_stacked,
                    place_near_target=place_near_target,
                    enforce_clear=False,
                    direction_hint=added if place_near_target else None,
                    exclude_types=exclude_types
                )
                
                variation.rng = random.Random(position_seed)
                mutated_only = variation.add_obstacle(
                    mutated_only,
                    place_near_target=place_near_target,
                    enforce_clear=False,
                    direction_hint=added if place_near_target else None,
                    exclude_types=exclude_types
                )
                
                variation.rng = random.Random(position_seed)
                mutated_baseline = variation.add_obstacle(
                    mutated_baseline,
                    place_near_target=place_near_target,
                    enforce_clear=False,
                    direction_hint=added if place_near_target else None,
                    exclude_types=exclude_types
                )
                
                # 检查是否成功添加（至少有一个文档变化了）
                if (mutated_stacked.text != before_stacked or 
                    mutated_only.text != before_only or 
                    mutated_baseline.text != before_baseline):
                    added += 1
                    print(f"  ✓ 成功添加第 {added} 个干扰物（尝试 {attempts} 次）")
                else:
                    print(f"  ✗ 第 {attempts} 次尝试失败，未找到合适位置")
            
            print(f"完成添加：目标 {target_count} 个，实际 {added} 个，共尝试 {attempts} 次")
        
        # 5. 保存三个变异后的BDDL
        output_dir.mkdir(parents=True, exist_ok=True)
        mutated_paths = {
            'stacked': output_dir / "mutated_stacked.bddl",
            'only': output_dir / "mutated_only.bddl",
            'baseline': output_dir / "mutated_baseline.bddl"
        }
        
        mutated_paths['stacked'].write_text(mutated_stacked.text)
        mutated_paths['only'].write_text(mutated_only.text)
        mutated_paths['baseline'].write_text(mutated_baseline.text)
        
        return mutated_paths, actual_strategy
    
    def create_configs(
        self, 
        original_config: Dict, 
        mutated_bddls: Dict[str, Path], 
        output_dir: Path
    ) -> Tuple[Dict, Dict]:
        """创建baseline和guided配置（只替换bddl_file路径）"""
        import copy
        
        # 深拷贝原始配置
        baseline_config = copy.deepcopy(original_config)
        guided_config = copy.deepcopy(original_config)
        
        # 更新baseline配置
        for group in baseline_config.get('groups', []):
            if group.get('name') == 'baseline':
                group['bddl_file'] = str(mutated_bddls['baseline'])
                break
        baseline_config['execution']['episodes_per_group'] = 1
        baseline_config['execution']['seed_start'] = 0
        baseline_config['output']['results_dir'] = str(output_dir / "baseline")
        
        # 更新guided配置
        for group in guided_config.get('groups', []):
            if group.get('name') == 'guided':
                stages = group.get('stages', [])
                if len(stages) >= 2:
                    stages[0]['bddl_file'] = str(mutated_bddls['stacked'])
                    stages[1]['bddl_file'] = str(mutated_bddls['only'])
                break
        guided_config['execution']['episodes_per_group'] = 1
        guided_config['execution']['seed_start'] = 0
        guided_config['output']['results_dir'] = str(output_dir / "guided")
        
        return baseline_config, guided_config
    
    def run_experiment(self, config: Dict, group_name: str, output_dir: Path) -> Dict:
        """运行实验并返回结果"""
        # 保存配置文件
        config_file = output_dir / f"config_{group_name}.yaml"
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)
        
        # 获取group配置
        group_config = None
        for group in config.get('groups', []):
            if group.get('name') == group_name:
                group_config = group
                break
        
        if not group_config:
            raise ValueError(f"找不到group: {group_name}")
        
        # 运行实验
        env = None
        group_output_dir = Path(config['output']['results_dir'])
        group_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            env, summary = obstruction_run.run_group_with_cache(
                env,
                config,
                group_config,
                f"group1_{group_name}",
                group_output_dir,
                self.model,
                self.model_config,
                None,
                0,
            )
        except Exception as e:
            # 捕获RandomizationError等异常
            error_msg = str(e)
            if "Cannot place all objects" in error_msg:
                print(f"⚠️  场景初始化失败：物体放置空间不足")
                return {
                    "success": False,
                    "summary": {"success_count": 0, "error": "placement_failed"},
                    "output_dir": str(group_output_dir),
                    "placement_error": True
                }
            else:
                # 其他异常继续抛出
                raise
        finally:
            if env is not None:
                try:
                    env.close()
                except AttributeError:
                    # 某些环境可能没有正确的close方法
                    pass
        
        success = summary.get("success_count", 0) > 0
        return {
            "success": success,
            "summary": summary,
            "output_dir": str(group_output_dir)
        }
    
    def run_scene(
        self, 
        scene_id: int, 
        config_file: Path, 
        num_distractors: int,
        max_guided_attempts: int,
        strategy: str = "random"
    ) -> Dict:
        """运行单个场景的实验"""
        print(f"\n{'='*70}")
        print(f"Scene {scene_id:04d} - Config: {config_file.name}")
        print(f"{'='*70}")
        
        # 创建场景输出目录
        scene_dir = self.output_dir / f"scene_{scene_id:04d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载原始配置
        with open(config_file) as f:
            original_config = yaml.safe_load(f)
        
        # 解析BDDL路径
        try:
            bddl_paths = self.parse_bddl_paths(original_config)
        except ValueError as e:
            print(f"跳过配置 {config_file.name}: {e}")
            return None
        
        # 从配置文件名提取场景中的物品类型，避免添加重复物品
        # 例如: exp_single_cheese_milk.yaml -> ['cream_cheese', 'milk']
        # 例如: exp_pair_butter_pudding.yaml -> ['butter', 'chocolate_pudding']
        exclude_types = []
        config_name = config_file.stem  # 去掉.yaml后缀
        # 使用正则提取物品名称（跳过exp_single/exp_pair前缀）
        match = re.search(r'exp_(single|pair)_(.+)', config_name)
        if match:
            items_str = match.group(2)
            item_names = items_str.split('_')
            for name in item_names:
                if name in _NAME_TO_TYPE_MAP:
                    exclude_types.append(_NAME_TO_TYPE_MAP[name])
        
        if exclude_types:
            print(f"从配置中检测到场景物品: {exclude_types}，将避免添加这些类型作为障碍物")
        
        # 变异BDDL
        print(f"使用策略: {strategy}")
        mutated_bddls, actual_strategy = self.mutate_three_bddls(
            bddl_paths['stacked'],
            bddl_paths['only'],
            bddl_paths['baseline'],
            num_distractors,
            scene_dir,
            strategy,
            exclude_types=exclude_types
        )
        
        # 创建配置
        baseline_config, guided_config = self.create_configs(
            original_config, mutated_bddls, scene_dir
        )
        
        # 保存原始配置信息
        scene_info = {
            "scene_id": scene_id,
            "config_file": str(config_file),
            "num_distractors": num_distractors,
            "strategy": actual_strategy,  # 记录实际使用的策略
            "bddl_paths": bddl_paths,
            "mutated_bddls": {k: str(v) for k, v in mutated_bddls.items()}
        }
        
        # 运行Baseline
        print("\n[Baseline] 运行中...")
        try:
            baseline_result = self.run_experiment(baseline_config, "baseline", scene_dir)
        except Exception as e:
            print(f"[Baseline] ❌ 运行失败，异常: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # 检查是否是placement错误
        if baseline_result.get("placement_error"):
            print(f"[Baseline] ⚠️  场景跳过：无法放置所有物体（干扰物太多）")
            return None
        
        scene_info["baseline_success"] = baseline_result["success"]
        scene_info["baseline_summary"] = baseline_result["summary"]
        
        if baseline_result["success"]:
            print(f"[Baseline] ✓ 成功")
            scene_info["optimization_needed"] = False
            scene_info["optimization_success"] = None
        else:
            print(f"[Baseline] ✗ 失败")
            scene_info["optimization_needed"] = True
            
            # 运行Guided（最多max_guided_attempts次）
            optimization_success = False
            guided_attempts = []
            
            for attempt in range(max_guided_attempts):
                print(f"\n[Guided] 尝试 {attempt + 1}/{max_guided_attempts}...")
                
                # 更新输出目录（每次尝试使用不同目录）
                guided_config['output']['results_dir'] = str(scene_dir / f"guided_attempt_{attempt}")
                
                try:
                    guided_result = self.run_experiment(guided_config, "guided", scene_dir)
                except Exception as e:
                    print(f"[Guided] ❌ 尝试 {attempt + 1} 运行失败，异常: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # 检查是否是placement错误
                if guided_result.get("placement_error"):
                    print(f"[Guided] ⚠️  尝试 {attempt + 1} 跳过：无法放置所有物体")
                    continue
                
                guided_attempts.append({
                    "attempt": attempt,
                    "success": guided_result["success"],
                    "summary": guided_result["summary"]
                })
                
                if guided_result["success"]:
                    print(f"[Guided] ✓ 成功（第 {attempt + 1} 次尝试）")
                    optimization_success = True
                    break
                else:
                    print(f"[Guided] ✗ 失败（第 {attempt + 1} 次尝试）")
            
            scene_info["optimization_success"] = optimization_success
            scene_info["guided_attempts"] = guided_attempts
            
            if not optimization_success:
                print(f"\n[Guided] 所有 {max_guided_attempts} 次尝试均失败")
        
        # 保存场景结果
        with open(scene_dir / "scene_result.json", 'w') as f:
            json.dump(scene_info, f, indent=2)
        
        return scene_info
    
    def run(
        self, 
        config_dir: Path, 
        num_scenes: int, 
        num_distractors: int,
        max_guided_attempts: int
    ):
        """运行完整实验"""
        # 加载所有配置文件
        config_files = list(config_dir.glob("exp_*.yaml"))
        if not config_files:
            raise ValueError(f"在 {config_dir} 中找不到配置文件（exp_*.yaml）")
        
        print(f"找到 {len(config_files)} 个配置文件")
        print(f"计划运行 {num_scenes} 个场景")
        print(f"每个场景添加 {num_distractors} 个干扰物")
        print(f"Guided最多尝试 {max_guided_attempts} 次\n")
        
        # 运行所有场景
        for scene_id in range(num_scenes):
            try:
                # 随机选择配置
                config_file = random.choice(config_files)
                
                # 运行场景（使用随机策略）
                scene_info = self.run_scene(
                    scene_id, config_file, num_distractors, max_guided_attempts, 
                    strategy="random"
                )
                
                if scene_info is None:
                    print(f"⚠️  Scene {scene_id} 跳过（配置错误或placement失败）")
                    continue
                
                # 更新统计 - 注意：只有baseline失败的才算入total_scenes和策略统计
                self.stats["scenes"].append(scene_info)
                
                # 获取使用的策略
                strategy = scene_info.get("strategy", "unknown")
                
                if scene_info["baseline_success"]:
                    self.stats["baseline_success"] += 1
                    # Baseline成功的不计入total_scenes和策略统计
                else:
                    # 只有Baseline失败时才算入总运行组数
                    self.stats["total_scenes"] += 1
                    self.stats["baseline_failed"] += 1
                    self.stats["optimization_attempted"] += 1
                    
                    # 更新按策略统计
                    if strategy in self.stats["by_strategy"]:
                        self.stats["by_strategy"][strategy]["total"] += 1
                        self.stats["by_strategy"][strategy]["baseline_failed"] += 1
                    
                    if scene_info.get("optimization_success", False):
                        self.stats["optimization_success"] += 1
                        
                        # 更新按策略统计 - 记录成功的场景编号
                        if strategy in self.stats["by_strategy"]:
                            self.stats["by_strategy"][strategy]["optimization_success"] += 1
                            self.stats["by_strategy"][strategy]["success_scenes"].append(scene_id)
                    else:
                        # 记录失败的场景编号
                        if strategy in self.stats["by_strategy"]:
                            self.stats["by_strategy"][strategy]["failed_scenes"].append(scene_id)
                
                # 打印当前统计
                self.print_current_stats()
                
            except Exception as e:
                print(f"\n{'='*70}")
                print(f"❌ Scene {scene_id} 发生异常，跳过此场景")
                print(f"错误信息: {e}")
                print(f"{'='*70}\n")
                import traceback
                traceback.print_exc()
                # 继续下一个场景
                continue
        
        # 保存最终统计
        self.save_final_stats()
    
    def print_current_stats(self):
        """打印当前统计"""
        print(f"\n{'='*70}")
        print(f"当前统计")
        print(f"{'='*70}")
        
        if self.stats['optimization_attempted'] > 0:
            success_rate = (self.stats['optimization_success'] / 
                          self.stats['optimization_attempted'] * 100)
            print(f"已运行: {self.stats['total_scenes']}  |  "
                  f"成功: {self.stats['optimization_success']}  |  "
                  f"成功率: {success_rate:.2f}%")
        else:
            print(f"已运行: {self.stats['total_scenes']}  |  成功: 0  |  成功率: 0.00%")
        
        # 打印各策略统计
        print(f"\n按策略统计:")
        has_data = False
        for strategy, stats in self.stats["by_strategy"].items():
            if stats["total"] > 0:
                has_data = True
                if stats["baseline_failed"] > 0:
                    strategy_rate = (stats["optimization_success"] / 
                                   stats["baseline_failed"] * 100)
                else:
                    strategy_rate = 0.0
                print(f"  {strategy:15s}: 运行={stats['total']:3d}, "
                      f"成功={stats['optimization_success']:3d}, "
                      f"成功率={strategy_rate:5.1f}%")
                
                # 打印成功的场景编号
                if stats["success_scenes"]:
                    scenes_str = ", ".join(str(s) for s in stats["success_scenes"])
                    print(f"    ✓ 成功: Scene {scenes_str}")
                
                # 打印失败的场景编号
                if stats["failed_scenes"]:
                    scenes_str = ", ".join(str(s) for s in stats["failed_scenes"])
                    print(f"    ✗ 失败: Scene {scenes_str}")
                
                print()  # 策略之间空一行
        
        if not has_data:
            print("  (暂无数据)")
        
        print(f"{'='*70}\n")
    
    def save_final_stats(self):
        """保存最终统计"""
        stats_file = self.output_dir / "final_stats.json"
        
        # 计算优化成功率
        if self.stats['optimization_attempted'] > 0:
            optimization_rate = (self.stats['optimization_success'] / 
                               self.stats['optimization_attempted'] * 100)
        else:
            optimization_rate = 0.0
        
        # 计算各策略成功率
        by_strategy_stats = {}
        for strategy, stats in self.stats["by_strategy"].items():
            if stats["baseline_failed"] > 0:
                strategy_rate = (stats["optimization_success"] / 
                               stats["baseline_failed"] * 100)
            else:
                strategy_rate = 0.0
            
            by_strategy_stats[strategy] = {
                "total": stats["total"],
                "baseline_failed": stats["baseline_failed"],
                "optimization_success": stats["optimization_success"],
                "optimization_failed": stats["baseline_failed"] - stats["optimization_success"],
                "optimization_success_rate": strategy_rate,
                "success_scenes": stats["success_scenes"],
                "failed_scenes": stats["failed_scenes"]
            }
        
        final_stats = {
            "total_scenes": self.stats['total_scenes'],
            "baseline_success": self.stats['baseline_success'],
            "baseline_failed": self.stats['baseline_failed'],
            "optimization_attempted": self.stats['optimization_attempted'],
            "optimization_success": self.stats['optimization_success'],
            "optimization_failed": (self.stats['optimization_attempted'] - 
                                  self.stats['optimization_success']),
            "optimization_success_rate": optimization_rate,
            "by_strategy": by_strategy_stats,
            "scenes": self.stats['scenes']
        }
        
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"最终统计结果")
        print(f"{'='*70}")
        print(f"总运行: {final_stats['total_scenes']}  |  "
              f"成功: {final_stats['optimization_success']}  |  "
              f"成功率: {final_stats['optimization_success_rate']:.2f}%")
        
        print(f"\n按策略统计:")
        for strategy, stats in by_strategy_stats.items():
            if stats['total'] > 0:
                print(f"  {strategy:15s}: 运行={stats['total']:3d}, "
                      f"成功={stats['optimization_success']:3d}, "
                      f"成功率={stats['optimization_success_rate']:5.2f}%")
                
                # 打印成功的场景编号
                if stats["success_scenes"]:
                    scenes_str = ", ".join(str(s) for s in stats["success_scenes"])
                    print(f"    ✓ 成功: Scene {scenes_str}")
                
                # 打印失败的场景编号
                if stats["failed_scenes"]:
                    scenes_str = ", ".join(str(s) for s in stats["failed_scenes"])
                    print(f"    ✗ 失败: Scene {scenes_str}")
                
                print()  # 策略之间空一行
        
        print(f"{'='*70}")
        print(f"\n结果已保存到: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="优化实验 - 测试两阶段优化策略的有效性"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="experiments/obstruction/configs",
        help="配置文件目录"
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=100,
        help="运行的场景数量"
    )
    parser.add_argument(
        "--num-distractors",
        type=int,
        default=5,
        help="目标干扰物数量的上限（实际会在2到此值之间随机，容错机制可能导致实际数量更少）"
    )
    parser.add_argument(
        "--max-guided-attempts",
        type=int,
        default=3,
        help="Guided组最多尝试次数"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./pi05_libero",
        help="Pi0.5模型checkpoint目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/obstruction/optimization_results",
        help="输出目录"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        print(f"使用随机种子: {args.seed}")
    
    # 创建输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"optimization_{timestamp}"
    
    # 运行实验
    experiment = OptimizationExperiment(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=output_dir
    )
    
    experiment.run(
        config_dir=Path(args.config_dir),
        num_scenes=args.num_scenes,
        num_distractors=args.num_distractors,
        max_guided_attempts=args.max_guided_attempts
    )


if __name__ == "__main__":
    main()
