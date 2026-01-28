"""
VLA Fuzzing 穷举优化系统
作者: AI Assistant
日期: 2026-01-10

该脚本实现了一个穷举优化系统，功能包括：
1. 自动从 results/ 目录查找失败的 episode
2. 调用外部策略脚本尝试所有可能的优化
3. 管理工作目录和成功目录
4. 跟踪优化历史
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ExhaustiveOptimizer:
    """失败 VLA episode 的穷举优化器"""
    
    def __init__(self, task: str, model: str, base_results_dir: str = None):
        """
        初始化优化器
        
        参数:
            task: 任务名称 (grasp, move, put-on, put-in)
            model: 模型名称 (openvla-7b, rt_1_x, 等)
            base_results_dir: 基础结果目录（默认：从 results/ 自动检测）
        """
        self.task = task
        self.model = model
        # 脚本在 optimization/ 目录，父目录是项目根目录
        self.project_root = Path(__file__).parent.parent.resolve()
        
        # 如果未提供，自动检测基础结果目录
        if base_results_dir is None:
            self.base_results_dir = self._find_base_results_dir()
        else:
            self.base_results_dir = Path(base_results_dir)
        
        # 设置优化目录结构
        self.opt_root = self.project_root / "optimization" / model / task
        self.working_dir = self.opt_root / "working"
        self.success_dir = self.opt_root / "success"
        self.history_file = self.opt_root / "history.json"
        self.logs_dir = self.project_root / "optimization" / "logs"
        self.config_file = self.opt_root / "config.json"
        
        # 加载配置（如果存在）或使用穷举策略
        self.config = self._load_config()
        self.episode_strategies = {}  # episode_id -> strategy mapping
        
        if self.config:
            print(f"[信息] 加载任务配置: {self.config_file}")
            self._parse_config()
            print(f"  配置的 episode 数: {len(self.episode_strategies)}")
        else:
            print(f"[信息] 未找到配置文件，使用穷举策略")
            # 定义所有优化策略
            self.strategies = self._define_strategies()
            print(f"  策略总数: {len(self.strategies)}")
        
        print(f"[信息] 优化器已初始化")
        print(f"  任务: {task}")
        print(f"  模型: {model}")
        print(f"  基础结果目录: {self.base_results_dir}")
        print(f"  优化根目录: {self.opt_root}")
    
    def _find_base_results_dir(self) -> Path:
        """
        从 results/ 自动检测基础结果目录
        
        返回:
            基础结果目录的路径
        """
        results_root = self.project_root / "results"
        
        # 搜索匹配的目录: t-{task}_n-*_o-*_s-*
        pattern = f"t-{self.task}_n-*"
        matching_dirs = list(results_root.glob(pattern))
        
        if not matching_dirs:
            raise FileNotFoundError(
                f"未找到任务 '{self.task}' 的结果目录: {results_root}"
            )
        
        # 使用最新的一个（按修改时间）
        latest_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)
        
        # 查找模型子目录
        model_dirs = [d for d in latest_dir.iterdir() if d.is_dir() and self.model in d.name]
        
        if not model_dirs:
            raise FileNotFoundError(
                f"未找到模型 '{self.model}' 的目录: {latest_dir}"
            )
        
        base_dir = model_dirs[0]
        print(f"[信息] 自动检测到结果目录: {base_dir}")
        return base_dir
    
    def _load_config(self) -> Optional[Dict]:
        """
        加载任务配置文件（如果存在）
        
        返回:
            配置字典，如果不存在返回 None
        """
        if not self.config_file.exists():
            return None
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"[警告] 配置文件加载失败: {e}")
            return None
    
    def _parse_config(self):
        """
        解析配置文件，构建 episode -> strategy 映射
        
        配置格式:
        {
            "task": "move",
            "min_grasp_steps": 5,
            "cases": [
                {
                    "episode_id": "1",
                    "strategy": "optimize_grasp",
                    "direction": "right",
                    "max_episode_steps": 80
                },
                ...
            ]
        }
        """
        if not self.config:
            return
        
        cases = self.config.get("cases", [])
        
        for case in cases:
            episode_id = str(case.get("episode_id"))
            strategy_type = case.get("strategy")
            
            if not episode_id or not strategy_type:
                print(f"[警告] 配置项缺少 episode_id 或 strategy: {case}")
                continue
            
            # 构建策略对象
            strategy = self._build_strategy_from_config(case)
            if strategy:
                self.episode_strategies[episode_id] = strategy
    
    def _build_strategy_from_config(self, case: Dict) -> Optional[Dict]:
        """
        从配置项构建策略字典
        
        参数:
            case: 配置项
            
        返回:
            策略字典
        """
        strategy_type = case.get("strategy")
        
        # 根据策略类型构建策略
        if strategy_type == "move_closer":
            move_ratio = case.get("move_ratio", 0.3)
            return {
                "name": f"move_closer_{move_ratio}",
                "type": "move_closer",
                "script": str(self.project_root / "optimization" / "fix_strategy_move_closer.py"),
                "params": {"move_ratio": move_ratio}
            }
        
        elif strategy_type == "rotate_object":
            rotation_mode = case.get("rotation_mode", "yaw")
            angle = case.get("angle", 45)
            return {
                "name": f"rotate_{rotation_mode}_{angle}",
                "type": "rotate_object",
                "script": str(self.project_root / "optimization" / "fix_strategy_rotate_object.py"),
                "params": {
                    "rotation_mode": rotation_mode,
                    "angle": angle
                }
            }
        
        elif strategy_type == "replace_object":
            new_object = case.get("new_object")
            if not new_object:
                print(f"[警告] replace_object 策略缺少 new_object: {case}")
                return None
            return {
                "name": f"replace_to_{new_object}",
                "type": "replace_object",
                "script": str(self.project_root / "optimization" / "fix_strategy_replace_object.py"),
                "params": {"new_object": new_object}
            }
        
        elif strategy_type == "optimize_grasp":
            direction = case.get("direction", "right")
            attempts = case.get("attempts", 20)
            max_steps = case.get("max_episode_steps", 80)
            return {
                "name": f"optimize_grasp_{direction}",
                "type": "optimize_grasp",
                "script": str(self.project_root / "optimization" / "fix_strategy_optimize_grasp.py"),
                "params": {
                    "direction": direction,
                    "attempts": attempts
                },
                "max_episode_steps": max_steps
            }
        
        else:
            print(f"[警告] 未知的策略类型: {strategy_type}")
            return None
    
    def _define_strategies(self) -> List[Dict]:
        """
        定义所有优化策略（调用外部脚本）
        
        返回:
            策略字典列表
        """
        strategies = []
        opt_dir = self.project_root / "optimization"
        
        # 1. 移近机械臂（3种）
        for ratio in [0.2, 0.3, 0.4]:
            strategies.append({
                "name": f"移近_{ratio:.1f}",
                "type": "move_closer",
                "script": str(opt_dir / "fix_strategy_move_closer.py"),
                "params": {"move_ratio": ratio}
            })
        
        # 2. 旋转物体（8种，仅Z轴）
        for angle in [15, 30, 45, 90, -15, -30, -45, -90]:
            strategies.append({
                "name": f"旋转_z轴_{angle:+d}度",
                "type": "rotate_object",
                "script": str(opt_dir / "fix_strategy_rotate_object.py"),
                "params": {"rotation_mode": "z_axis", "angle": angle}
            })
        
        # 3. 替换物体（1种，根据物品类型动态决定是否应用）
        strategies.append({
            "name": "替换物体",
            "type": "replace_object",
            "script": str(opt_dir / "fix_strategy_replace_object.py"),
            "params": {}
        })
        
        # 4. 优化抓取（8个方向）
        direction_names = {
            "left": "左", "right": "右", "up": "上", "down": "下",
            "left-up": "左上", "left-down": "左下", 
            "right-up": "右上", "right-down": "右下"
        }
        for direction, name in direction_names.items():
            strategies.append({
                "name": f"搜索最佳位置_{name}",
                "type": "optimize_grasp",
                "script": str(opt_dir / "fix_strategy_optimize_grasp.py"),
                "params": {"direction": direction, "attempts": 20}
            })
        
        return strategies
    
    def initialize(self):
        """
        初始化优化：从 results/ 复制失败的 episode 到 working/
        只在 working/ 不存在时运行一次
        """
        if self.working_dir.exists():
            print(f"[信息] 工作目录已存在，跳过初始化")
            return
        
        print(f"[信息] 正在初始化优化目录...")
        
        # 创建目录结构
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.success_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 扫描失败的 episode
        failed_episodes = self._find_failed_episodes()
        
        if not failed_episodes:
            print(f"[警告] 在 {self.base_results_dir} 中未找到失败的 episode")
            return
        
        print(f"[信息] 找到 {len(failed_episodes)} 个失败的 episode")
        
        # 复制失败的 episode 到 working/
        for episode_id in failed_episodes:
            src_dir = self.base_results_dir / str(episode_id)
            dst_dir = self.working_dir / str(episode_id)
            
            # 创建目标目录
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            # 从基础结果目录路径提取配置
            options = self._extract_options_from_path(src_dir)
            
            # 保存为 options.json 和 origin.json
            options_file = dst_dir / "options.json"
            origin_file = dst_dir / "origin.json"
            
            with open(options_file, 'w') as f:
                json.dump(options, f, indent=2)
            
            with open(origin_file, 'w') as f:
                json.dump(options, f, indent=2)
            
            print(f"  已复制 episode {episode_id}")
        
        # 初始化 history.json
        history = {
            "task": self.task,
            "model": self.model,
            "base_results_dir": str(self.base_results_dir),
            "total_episodes": len(failed_episodes),
            "episodes": {str(eid): {"status": "pending", "attempts": []} for eid in failed_episodes}
        }
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"[信息] 初始化完成")
    
    def _find_failed_episodes(self) -> List[int]:
        """
        在 base_results_dir 中查找所有失败的 episode
        
        返回:
            失败的 episode ID 列表
        """
        failed = []
        
        for episode_dir in sorted(self.base_results_dir.iterdir()):
            if not episode_dir.is_dir():
                continue
            
            # 从目录名提取 episode ID
            try:
                episode_id = int(episode_dir.name)
            except ValueError:
                continue
            
            # 检查 log.json
            log_file = episode_dir / "log.json"
            if not log_file.exists():
                continue
            
            # 检查是否有任何步骤成功
            if not self._check_success(log_file):
                failed.append(episode_id)
        
        return failed
    
    def _check_success(self, log_file: Path) -> bool:
        """
        检查 log.json 中是否有任何步骤 success=True
        
        参数:
            log_file: log.json 的路径
            
        返回:
            如果有任何步骤成功则返回 True
        """
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            for step_key in log_data.keys():
                success_val = log_data[step_key].get("success", False)
                # 处理布尔值和字符串 "true"/"false"
                if success_val is True or success_val == "true" or success_val == True:
                    return True
            
            return False
        except Exception as e:
            print(f"[警告] 读取 {log_file} 出错: {e}")
            return False
    
    def _extract_options_from_path(self, episode_dir: Path) -> Dict:
        """
        从原始结果目录提取配置（model_id, obj_init_options）
        需要读取原始数据集文件并匹配 episode
        
        参数:
            episode_dir: results/ 中 episode 目录的路径
            
        返回:
            包含 model_id 和 obj_init_options 的字典
        """
        # 从 base_results_dir 路径提取数据集名称
        base_name = self.base_results_dir.parent.name
        dataset_file = self.project_root / "data" / f"{base_name}.json"
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"数据集文件未找到: {dataset_file}")
        
        # 加载数据集
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        # 提取 episode ID
        episode_id = int(episode_dir.name)
        
        # 获取此 episode 的配置
        if str(episode_id) not in dataset:
            raise KeyError(f"Episode {episode_id} 在数据集 {dataset_file} 中未找到")
        
        return dataset[str(episode_id)]
    
    def _extract_seed_from_path(self, base_dir: Path) -> int:
        """
        从结果目录路径提取 seed
        模式: t-{task}_n-{num}_o-{obj}_s-{seed}
        
        参数:
            base_dir: 结果目录的路径
            
        返回:
            Seed 值
        """
        dir_name = base_dir.parent.name
        match = re.search(r's-(\d+)', dir_name)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"无法从目录名提取 seed: {dir_name}")
    
    def run_optimization(self, max_episodes: Optional[int] = None):
        """
        对所有工作中的 episode 运行穷举优化
        
        参数:
            max_episodes: 最多处理的 episode 数量（None = 全部）
        """
        # 加载历史
        if not self.history_file.exists():
            print(f"[错误] 历史文件未找到。请先运行 initialize()")
            return
        
        with open(self.history_file, 'r') as f:
            history = json.load(f)
        
        # 获取工作中的 episode 列表（不在 success/ 中的）
        working_episodes = [
            int(d.name) for d in self.working_dir.iterdir()
            if d.is_dir() and not (self.success_dir / d.name).exists()
        ]
        
        if max_episodes:
            working_episodes = working_episodes[:max_episodes]
        
        print(f"[信息] 开始优化 {len(working_episodes)} 个 episode")
        
        # 处理每个 episode
        for episode_id in working_episodes:
            print(f"\n{'='*60}")
            print(f"处理 episode {episode_id}")
            print(f"{'='*60}")
            
            episode_success = False
            episode_id_str = str(episode_id)
            
            # 检查是否有配置的策略
            if self.config and episode_id_str in self.episode_strategies:
                # 使用配置指定的策略
                strategy = self.episode_strategies[episode_id_str]
                print(f"[配置模式] 使用指定策略: {strategy['name']}")
                
                # 重置为原始配置
                self._reset_options(episode_id)
                
                # 应用策略
                strategy_applied = self._apply_strategy(episode_id, strategy)
                
                if strategy_applied:
                    # 运行推理
                    success = self._run_inference(episode_id, strategy)
                    
                    # 记录尝试历史
                    self._record_attempt(history, episode_id, strategy, success)
                    
                    if success:
                        print(f"✅ [成功] 策略 '{strategy['name']}' 成功！")
                        # 移动到成功目录
                        self._move_to_success(episode_id, strategy)
                        episode_success = True
                    else:
                        print(f"❌ [失败] 策略 '{strategy['name']}' 推理失败（任务未成功完成）")
                else:
                    print(f"  ⏭️  [跳过] 策略不适用于当前episode")
            
            elif self.config and episode_id_str not in self.episode_strategies:
                # 配置模式，但该 episode 未在配置中
                print(f"[配置模式] Episode {episode_id} 未在配置中，跳过")
                continue
            
            else:
                # 穷举模式：尝试所有策略
                print(f"[穷举模式] 尝试所有策略")
                for strategy_idx, strategy in enumerate(self.strategies):
                    print(f"\n[{strategy_idx+1}/{len(self.strategies)}] 尝试策略: {strategy['name']}")
                    
                    # 重置为原始配置
                    self._reset_options(episode_id)
                    
                    # 应用策略
                    strategy_applied = self._apply_strategy(episode_id, strategy)
                    
                    # 如果策略不适用则跳过
                    if not strategy_applied:
                        print(f"  ⏭️  [跳过] 策略不适用于当前episode")
                        continue
                    
                    # 运行推理
                    success = self._run_inference(episode_id, strategy)
                    
                    # 记录尝试历史
                    self._record_attempt(history, episode_id, strategy, success)
                    
                    if success:
                        print(f"✅ [成功] 策略 '{strategy['name']}' 成功！")
                        
                        # 移动到成功目录
                        self._move_to_success(episode_id, strategy)
                        episode_success = True
                        break
                    else:
                        print(f"❌ [失败] 策略 '{strategy['name']}' 推理失败（任务未成功完成）")
            
            if not episode_success:
                print(f"[失败] Episode {episode_id} 的策略失败了")
            
            # 每个 episode 后保存历史
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"优化完成！")
        self._print_summary(history)
    
    def _reset_options(self, episode_id: int):
        """
        将 options.json 重置为 origin.json
        
        参数:
            episode_id: Episode ID
        """
        episode_dir = self.working_dir / str(episode_id)
        origin_file = episode_dir / "origin.json"
        options_file = episode_dir / "options.json"
        
        shutil.copy(origin_file, options_file)
    
    def _apply_strategy(self, episode_id: int, strategy: Dict) -> bool:
        """
        通过调用外部脚本应用优化策略
        
        参数:
            episode_id: Episode ID
            strategy: 包含 'type', 'script', 'params' 的策略字典
            
        返回:
            如果策略已应用返回 True，如果跳过返回 False
        """
        episode_dir = self.working_dir / str(episode_id)
        strategy_type = strategy["type"]
        script_path = strategy["script"]
        params = strategy["params"]
        
        # 检查是否应跳过某些物品的 replace_object
        if strategy_type == "replace_object":
            if not self._should_apply_replace_strategy(episode_dir):
                print(f"  [跳过] 物品不适合替换")
                return False
        
        # 根据策略类型构建命令
        cmd = [sys.executable, script_path, str(episode_dir)]
        
        if strategy_type == "move_closer":
            cmd.extend(["--move_ratio", str(params["move_ratio"])])
        
        elif strategy_type == "rotate_object":
            cmd.extend([
                "--rotation_mode", params["rotation_mode"],
                "--angle", str(params["angle"])
            ])
        
        elif strategy_type == "replace_object":
            # 根据替换规则获取新物品名称
            new_object = self._get_replacement_object(episode_dir)
            if new_object is None:
                print(f"  [跳过] 没有定义替换规则")
                return False
            cmd.extend(["--new_object", new_object])
        
        elif strategy_type == "optimize_grasp":
            # direction 是位置参数，不是选项参数
            cmd.append(params["direction"])
            cmd.extend(["--attempts", str(params["attempts"])])
        
        # 执行策略脚本
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
                cwd=str(self.project_root)
            )
            
            if result.returncode != 0:
                print(f"  [错误] 策略脚本失败:")
                print(f"  [STDOUT]\n{result.stdout}")
                print(f"  [STDERR]\n{result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            print(f"  [错误] 策略脚本超时 (>5分钟)")
            return False
        except Exception as e:
            print(f"  [错误] 执行策略失败: {e}")
            return False
    
    def _should_apply_replace_strategy(self, episode_dir: Path) -> bool:
        """
        检查是否应该应用 replace_object 策略
        
        参数:
            episode_dir: Episode 目录
            
        返回:
            如果适用替换则返回 True
        """
        options_file = episode_dir / "options.json"
        with open(options_file, 'r') as f:
            options = json.load(f)
        
        # 获取源物体名称
        if "model_id" in options:
            source_obj = options["model_id"]
        elif "model_ids" in options and "source_obj_id" in options:
            source_id = options["source_obj_id"]
            source_obj = options["model_ids"][source_id]
        else:
            return False
        
        # 跳过不应替换的物品
        skip_objects = {
            "bridge_carrot_generated_modified",
            "eggplant",
            "green_cube_3cm",
            "yellow_cube_3cm",
            "bridge_spoon_generated_modified",
            "sponge",
            "coke_can",  # 已经是目标
            "opened_coke_can",  # 已经是目标
            "apple",  # 已经是目标
        }
        
        return source_obj not in skip_objects
    
    def _get_replacement_object(self, episode_dir: Path) -> Optional[str]:
        """
        根据规则获取替换物品名称
        
        参数:
            episode_dir: Episode 目录
            
        返回:
            新物品名称，如果没有规则则返回 None
        """
        options_file = episode_dir / "options.json"
        with open(options_file, 'r') as f:
            options = json.load(f)
        
        # 获取源物体名称
        if "model_id" in options:
            source_obj = options["model_id"]
        elif "model_ids" in options and "source_obj_id" in options:
            source_id = options["source_obj_id"]
            source_obj = options["model_ids"][source_id]
        else:
            return None
        
        # 替换规则
        replacement_rules = {
            # 球状 → 苹果
            "orange": "apple",
            
            # 瓶子 → 可乐罐
            "blue_plastic_bottle": "coke_can",
            
            # 所有罐头（未开封） → 可乐罐
            "pepsi_can": "coke_can",
            "sprite_can": "coke_can",
            "fanta_can": "coke_can",
            "7up_can": "coke_can",
            "redbull_can": "coke_can",
            
            # 所有罐头（开封） → 开封可乐罐
            "opened_pepsi_can": "opened_coke_can",
            "opened_sprite_can": "opened_coke_can",
            "opened_fanta_can": "opened_coke_can",
            "opened_7up_can": "opened_coke_can",
            "opened_redbull_can": "opened_coke_can",
        }
        
        return replacement_rules.get(source_obj)
    
    def _run_inference(self, episode_id: int, strategy: Dict) -> bool:
        """
        使用当前配置为单个 episode 运行推理
        
        参数:
            episode_id: Episode ID
            strategy: 正在测试的策略
            
        返回:
            如果推理成功返回 True
        """
        episode_dir = self.working_dir / str(episode_id)
        options_file = episode_dir / "options.json"
        
        # 创建临时数据集文件（单个 episode）
        with open(options_file, 'r') as f:
            options = json.load(f)
        
        # 如果策略包含 max_episode_steps，应用它
        if "max_episode_steps" in strategy:
            options["max_episode_steps"] = strategy["max_episode_steps"]
            print(f"  设置 max_episode_steps = {strategy['max_episode_steps']}")
        
        # 从基础目录提取 seed
        seed = self._extract_seed_from_path(self.base_results_dir)
        
        temp_dataset = {
            "0": options,
            "seed": seed,
            "num": 1
        }
        
        # 在文件名中使用任务名，这样 run_fuzzer.py 可以检测任务类型
        temp_data_file = episode_dir / f"temp_{self.task}_data.json"
        with open(temp_data_file, 'w') as f:
            json.dump(temp_dataset, f, indent=2)
        
        # 运行 fuzzer
        cmd = [
            sys.executable,
            str(self.project_root / "experiments" / "run_fuzzer.py"),
            "--data", str(temp_data_file),
            "--model", self.model,
            "--output", str(episode_dir) + "/",
            "--seed", str(seed),  # 传递固定的 seed
            "--resume", "False"
        ]
        
        print(f"  运行: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True
                # 无超时 - 让推理运行所需的时间
            )
            
            # run_fuzzer.py 创建: {output}/{dataset_name}/{model}_{seed}/0/log.json
            dataset_name = f"temp_{self.task}_data"
            log_file = episode_dir / dataset_name / f"{self.model}_{seed}" / "0" / "log.json"
            
            if log_file.exists():
                success = self._check_success(log_file)
                
                # 如果成功，复制 log.json 到 episode 根目录以便记录
                if success:
                    shutil.copy(str(log_file), str(episode_dir / "log.json"))
                
                return success
            else:
                print(f"  [错误] log.json 未找到: {log_file}")
                # 打印完整的 stderr 和 stdout 用于调试
                if result.stdout:
                    print(f"  [STDOUT]\n{result.stdout}")
                if result.stderr:
                    print(f"  [STDERR]\n{result.stderr}")
                return False
        
        except Exception as e:
            print(f"  [错误] 推理失败: {e}")
            return False
    
    def _record_attempt(self, history: Dict, episode_id: int, strategy: Dict, success: bool):
        """
        在历史中记录优化尝试
        
        参数:
            history: 历史字典
            episode_id: Episode ID
            strategy: 策略字典
            success: 尝试是否成功
        """
        episode_key = str(episode_id)
        
        if episode_key not in history["episodes"]:
            history["episodes"][episode_key] = {
                "status": "pending",
                "attempts": []
            }
        
        attempt = {
            "strategy": strategy["name"],
            "success": success
        }
        
        history["episodes"][episode_key]["attempts"].append(attempt)
        
        if success:
            history["episodes"][episode_key]["status"] = "success"
    
    def _move_to_success(self, episode_id: int, strategy: Dict):
        """
        将 episode 从 working/ 移动到 success/
        
        参数:
            episode_id: Episode ID
            strategy: 成功的策略
        """
        src_dir = self.working_dir / str(episode_id)
        dst_dir = self.success_dir / str(episode_id)
        
        # 移动目录
        shutil.move(str(src_dir), str(dst_dir))
        
        # 保存策略信息
        strategy_file = dst_dir / "strategy.json"
        with open(strategy_file, 'w') as f:
            json.dump(strategy, f, indent=2)
        
        print(f"  已移动到 success/")
    
    def _print_summary(self, history: Dict):
        """
        打印优化摘要
        
        参数:
            history: 历史字典
        """
        total = len(history["episodes"])
        success = sum(1 for ep in history["episodes"].values() if ep["status"] == "success")
        failed = total - success
        
        print(f"总 episode 数: {total}")
        print(f"成功: {success}")
        print(f"失败: {failed}")
        print(f"成功率: {success/total*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLA fuzzing 失败案例的穷举优化"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["grasp", "move", "put-on", "put-in"],
        help="任务名称"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["rt_1_x", "rt_1_400k", "rt_1_58k", "rt_1_1k", "octo-base", "octo-small", "openvla-7b"],
        help="模型名称"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="基础结果目录（未指定则自动检测）"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="最多处理的 episode 数量"
    )
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="仅初始化目录结构，不运行优化"
    )
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = ExhaustiveOptimizer(
        task=args.task,
        model=args.model,
        base_results_dir=args.base_dir
    )
    
    # 初始化
    optimizer.initialize()
    
    # 运行优化
    if not args.initialize_only:
        optimizer.run_optimization(max_episodes=args.max_episodes)
