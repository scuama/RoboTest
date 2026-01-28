import argparse
import json
import random
import time
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import re
import tempfile
import yaml
import subprocess

import sys


_ROOT_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _ROOT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
from experiments.obstruction.bddl_variation import BDDLDocument, BDDLVariation
from experiments.obstruction.scripts import run_experiment as obstruction_run


class StableJSONizer(json.JSONEncoder):
    def default(self, obj):
        return super().encode(bool(obj)) if isinstance(obj, np.bool_) else super().default(obj)


def _load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def _extract_language(text: str) -> str:
    match = re.search(r"\(:language\s+([^\n\)]+)\)", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip().lower()


def _has_goal_predicate(text: str, predicate: str) -> bool:
    return re.search(rf"\({predicate}\s", text, flags=re.IGNORECASE) is not None


def _classify_task_type(text: str, filename: str) -> str:
    language = _extract_language(text)
    name = filename.lower()

    if _has_goal_predicate(text, "in") or " into " in language or " inside " in language:
        return "put-in"
    if _has_goal_predicate(text, "on") and (
        " on " in language
        or " place " in language
        or " put " in language
        or " stack " in language
    ):
        return "put-on"
    if any(word in language for word in [" move ", " near ", " closer ", " next to ", " bring ", " shift "]):
        return "move"
    if any(word in language for word in [" pick ", " pick up ", " grasp ", " grab ", " lift "]):
        return "grasp"

    if "move" in name or "near" in name:
        return "move"
    if "put" in name and "in" in name:
        return "put-in"
    if "put" in name and "on" in name:
        return "put-on"
    if "pick" in name or "grasp" in name:
        return "grasp"

    if _has_goal_predicate(text, "on"):
        return "put-on"
    return "grasp"


def _load_libero_task_map(task_suite: str) -> Tuple[Dict[str, int], Dict[int, Path]]:
    try:
        from libero.libero import benchmark
    except Exception as exc:
        raise RuntimeError("LIBERO not available; cannot map BDDL files to task ids.") from exc

    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite not in benchmark_dict:
        raise ValueError(f"Unknown task_suite: {task_suite}")
    suite = benchmark_dict[task_suite]()
    file_to_id: Dict[str, int] = {}
    id_to_path: Dict[int, Path] = {}
    for task_id in range(suite.n_tasks):
        bddl_file = suite.get_task_bddl_files()[task_id]
        file_to_id[bddl_file] = task_id
        id_to_path[task_id] = Path(suite.get_task_bddl_file_path(task_id))
    return file_to_id, id_to_path


class PromptNode:
    def __init__(
        self,
        fuzzer: "OptionsFuzzer",
        prompt: Dict[str, Any],
        response: Optional[Dict[str, Any]] = None,
        results: Optional[List[int]] = None,
        parent: Optional["PromptNode"] = None,
        mutator: Optional[str] = None,
    ):
        self.fuzzer: "OptionsFuzzer" = fuzzer
        self.prompt: Dict[str, Any] = prompt
        self.response: Optional[Dict[str, Any]] = response
        self.results: Optional[List[int]] = results
        self.visited_num = 0
        self.parent: Optional["PromptNode"] = parent
        self.mutator: Optional[str] = mutator
        self.child: List["PromptNode"] = []
        self.level: int = 0 if parent is None else parent.level + 1
        self._index: Optional[int] = None

    @property
    def index(self) -> Optional[int]:
        return self._index

    @index.setter
    def index(self, index: int) -> None:
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

    @property
    def num_jailbreak(self) -> int:
        return sum(self.results or [])

    @property
    def num_reject(self) -> int:
        return len(self.results or []) - sum(self.results or [])

    @property
    def num_query(self) -> int:
        return len(self.results or [])


class SelectPolicy:
    def __init__(self, fuzzer: Optional["OptionsFuzzer"] = None):
        self.fuzzer = fuzzer

    def select(self) -> PromptNode:
        raise NotImplementedError("SelectPolicy must implement select method.")

    def update(self, prompt_nodes: List[PromptNode]) -> None:
        pass


class MCTSExploreSelectPolicy(SelectPolicy):
    def __init__(
        self,
        fuzzer: Optional["OptionsFuzzer"] = None,
        ratio: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.2,
    ):
        super().__init__(fuzzer)
        self.step = 0
        self.mctc_select_path: List[PromptNode] = []
        self.last_choice_index: Optional[int] = None
        self.rewards: List[float] = []
        self.ratio = ratio
        self.alpha = alpha
        self.beta = beta

    def select(self) -> PromptNode:
        self.step += 1
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend(
                [0.0 for _ in range(len(self.fuzzer.prompt_nodes) - len(self.rewards))]
            )

        self.mctc_select_path.clear()
        cur = max(
            self.fuzzer.initial_prompts_nodes,
            key=lambda pn: self.rewards[pn.index] / (pn.visited_num + 1)
            + self.ratio * np.sqrt(2 * np.log(self.step) / (pn.visited_num + 0.01)),
        )
        self.mctc_select_path.append(cur)

        while len(cur.child) > 0:
            if np.random.rand() < self.alpha:
                break
            cur = max(
                cur.child,
                key=lambda pn: self.rewards[pn.index] / (pn.visited_num + 1)
                + self.ratio * np.sqrt(2 * np.log(self.step) / (pn.visited_num + 0.01)),
            )
            self.mctc_select_path.append(cur)

        for pn in self.mctc_select_path:
            pn.visited_num += 1

        self.last_choice_index = cur.index
        return cur

    def update(self, prompt_nodes: List[PromptNode]) -> None:
        succ_num = sum(prompt_node.num_jailbreak for prompt_node in prompt_nodes)
        last_choice_node = self.fuzzer.prompt_nodes[self.last_choice_index]
        for prompt_node in reversed(self.mctc_select_path):
            reward = succ_num / (len(self.fuzzer.questions) * len(prompt_nodes))
            self.rewards[prompt_node.index] += reward * max(
                self.beta, (1 - 0.1 * last_choice_node.level)
            )


class OptionsMutatePolicy:
    def __init__(self, variation: BDDLVariation, strategies: List[str]):
        self.variation = variation
        self.strategies = strategies
        self._fuzzer = None

    @property
    def fuzzer(self):
        return self._fuzzer

    @fuzzer.setter
    def fuzzer(self, fuzzer):
        self._fuzzer = fuzzer

    def mutate_single(self, prompt_node: PromptNode) -> List[PromptNode]:
        results: List[PromptNode] = []
        for _ in range(self._fuzzer.energy):
            strategy = random.choice(self.strategies)
            if getattr(self._fuzzer, "verbose", False):
                print(f"[mutate] strategy={strategy} parent_index={prompt_node.index}")

            variation_seed = time.time_ns() % (2**32)
            variation = BDDLVariation(seed=variation_seed)
            mutated = dict(prompt_node.prompt)
            seed_doc = BDDLDocument(mutated["bddl_text"])
            if strategy == "expand_random":
                mutated_doc = variation.expand_random(seed_doc)
            elif strategy == "rephrase":
                mutated_doc = variation.rephrase(seed_doc)
            else:
                continue

            mutated["bddl_text"] = mutated_doc.text
            language = mutated_doc.get_language()
            if language:
                mutated["task_instruction"] = language
            mutated["mutator"] = strategy
            results.append(PromptNode(self._fuzzer, mutated, parent=prompt_node, mutator=strategy))
        return results


class OptionsFuzzer:
    def __init__(
        self,
        initial_seeds: List[Dict[str, Any]],
        mutate_policy: OptionsMutatePolicy,
        select_policy: MCTSExploreSelectPolicy,
        runner: "MCTSRunner",
        task_suite: str,
        energy: int = 1,
        max_iteration: int = 100,
        output_dir: Optional[Path] = None,
        run_seed: Optional[int] = None,
        num_failures: int = 80,  # 新增：目标失败案例数
    ):
        self.questions = [None]
        self.prompt_nodes = [PromptNode(self, seed) for seed in initial_seeds]
        self.initial_prompts_nodes = self.prompt_nodes.copy()
        for i, prompt_node in enumerate(self.prompt_nodes):
            prompt_node.index = i

        self.mutate_policy = mutate_policy
        self.select_policy = select_policy
        self.mutate_policy.fuzzer = self
        self.select_policy.fuzzer = self

        self.energy = energy
        self.max_iteration = max_iteration
        self.current_iteration = 0
        self.run_seed = run_seed
        self.task_suite = task_suite
        self.runner = runner
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_counter = 0
        self.verbose = False
        self.rng = random.Random(run_seed)
        self.num_failures = num_failures  # 保存目标失败案例数

    def is_stop(self) -> bool:
        return self.max_iteration != -1 and self.current_iteration >= self.max_iteration

    def evaluate(self, prompt_nodes: List[PromptNode]) -> None:
        """
        两阶段评估：
        1. 运行baseline（1次）
        2. 如果失败，运行guided（最多5次，成功1次即停）
        """
        for idx, prompt_node in enumerate(prompt_nodes):
            options = prompt_node.prompt
            
            # 准备输出目录
            scene_idx = self.runner.scene_counter
            self.runner.scene_counter += 1
            
            run_dir = None
            if self.output_dir is not None:
                run_dir = self.output_dir / f"scene_{scene_idx:04d}"
                run_dir.mkdir(parents=True, exist_ok=True)
            
            if run_dir is None:
                # 如果没有输出目录，跳过
                prompt_node.response = {"success": False}
                prompt_node.results = [1]
                continue
            
            # 保存变异后的BDDL
            bddl_file = run_dir / "mutated.bddl"
            bddl_file.write_text(options["bddl_text"])
            
            # 保存options
            with open(run_dir / "options.json", 'w') as f:
                json.dump({k: v for k, v in options.items() if k != "bddl_text"}, f, indent=2, cls=StableJSONizer)
            
            print(f"\n{'='*70}")
            print(f"[{idx+1}/{len(prompt_nodes)}] 评估场景 {scene_idx}")
            print(f"{'='*70}")
            
            # === Stage 1: Baseline ===
            print(f"\n[Stage 1] 运行Baseline配置...")
            baseline_config, guided_config = self.runner.create_dual_configs(options, bddl_file, run_dir)
            
            baseline_config_file = run_dir / "config_baseline.yaml"
            with open(baseline_config_file, 'w') as f:
                yaml.safe_dump(baseline_config, f, sort_keys=False)
            
            baseline_output_dir = run_dir / "baseline"
            baseline_output_dir.mkdir(parents=True, exist_ok=True)
            baseline_success = self.runner._run_single_config(baseline_config, baseline_output_dir)
            
            result = {
                "baseline_success": baseline_success,
                "guided_success": False,
                "guided_attempts": 0,
                "status": "unknown"
            }
            
            if baseline_success:
                print(f"✅ Baseline成功，丢弃此样例")
                result["status"] = "baseline_success"
                prompt_node.response = {"success": True, "result": result}
                prompt_node.results = [0]  # 成功 = 0 reward
                
                # 删除整个目录（丢弃成功样例）
                shutil.rmtree(run_dir)
                continue
            
            # Baseline失败，计数+1，保存样例
            print(f"❌ Baseline失败，保存baseline结果")
            self.runner.fail_count += 1
            print(f"\n*** 累计Baseline失败案例: {self.runner.fail_count} / {self.num_failures} ***\n")
            
            # === Stage 2: Guided ===
            print(f"\n[Stage 2] 运行Guided配置（最多5次，成功1次即停）...")
            guided_config_file = run_dir / "config_guided.yaml"
            
            guided_success = False
            for attempt in range(1, 6):
                print(f"  尝试 {attempt}/5...")
                
                # 更新输出目录（每次尝试独立）
                guided_output_dir = run_dir / "guided" / f"attempt_{attempt}"
                guided_output_dir.mkdir(parents=True, exist_ok=True)
                guided_config["output"]["results_dir"] = str(guided_output_dir)
                
                with open(guided_config_file, 'w') as f:
                    yaml.safe_dump(guided_config, f, sort_keys=False)
                
                episode_success = self.runner._run_single_config(guided_config, guided_output_dir)
                result["guided_attempts"] = attempt
                
                if episode_success:
                    print(f"✅ Guided成功（第{attempt}次尝试）")
                    guided_success = True
                    result["guided_success"] = True
                    result["status"] = "guided_success"
                    
                    # 只保留成功的那次尝试
                    final_guided_dir = run_dir / "guided"
                    # 删除其他尝试
                    for a in range(1, attempt):
                        other_dir = run_dir / "guided" / f"attempt_{a}"
                        if other_dir.exists():
                            shutil.rmtree(other_dir)
                    # 重命名成功的尝试
                    shutil.move(str(guided_output_dir), str(final_guided_dir.parent / "final"))
                    if (final_guided_dir.parent / "final").exists():
                        if final_guided_dir.exists():
                            shutil.rmtree(final_guided_dir)
                        shutil.move(str(final_guided_dir.parent / "final"), str(final_guided_dir))
                    break
            
            if not guided_success:
                print(f"❌ Guided失败（5次均失败）")
                result["status"] = "failed"
                
                # 只保留最后一次尝试
                final_guided_dir = run_dir / "guided"
                last_attempt_dir = run_dir / "guided" / f"attempt_{result['guided_attempts']}"
                # 删除其他尝试
                for a in range(1, result['guided_attempts']):
                    other_dir = run_dir / "guided" / f"attempt_{a}"
                    if other_dir.exists():
                        shutil.rmtree(other_dir)
                # 重命名最后一次
                if last_attempt_dir.exists():
                    shutil.move(str(last_attempt_dir), str(final_guided_dir.parent / "final"))
                    if (final_guided_dir.parent / "final").exists():
                        if final_guided_dir.exists():
                            shutil.rmtree(final_guided_dir)
                        shutil.move(str(final_guided_dir.parent / "final"), str(final_guided_dir))
            else:
                # Guided成功，保留此样例
                print(f"✅ Guided成功，保留此样例")
            
            # 保存结果
            if run_dir.exists():
                with open(run_dir / "result.json", 'w') as f:
                    json.dump(result, f, indent=2)
            
            # 更新节点统计
            prompt_node.response = {"success": not (result["status"] == "failed"), "result": result}
            prompt_node.results = [1 if result["status"] == "failed" else 0]
            
            # 达到目标失败案例数，停止
            if self.runner.fail_count >= self.num_failures:
                print(f"\n{'='*70}")
                print(f"已收集{self.num_failures}个失败案例（baseline失败的），停止MCTS搜索")
                print(f"{'='*70}")
                break

    def update(self, prompt_nodes: List[PromptNode]) -> None:
        for prompt_node in prompt_nodes:
            if prompt_node.num_jailbreak > 0:
                prompt_node.index = len(self.prompt_nodes)
                self.prompt_nodes.append(prompt_node)
                if self.verbose:
                    print(f"[update] keep node_index={prompt_node.index} reward={prompt_node.num_jailbreak}")
        self.select_policy.update(prompt_nodes)

    def run(self) -> None:
        while not self.is_stop():
            # 检查是否达到目标失败案例数
            if self.runner.fail_count >= self.num_failures:
                print(f"\n已达到{self.num_failures}个失败案例，停止MCTS")
                break
                
            if self.verbose:
                print(
                    f"[run] iter={self.current_iteration} "
                    f"total_nodes={len(self.prompt_nodes)} "
                    f"fail_count={self.runner.fail_count}/{self.num_failures}"
                )
            seed = self.select_policy.select()
            if self.verbose:
                print(
                    f"[select] node_index={seed.index} "
                    f"visited={seed.visited_num} level={seed.level}"
                )
            mutated_nodes = self.mutate_policy.mutate_single(seed)
            self.evaluate(mutated_nodes)
            self.update(mutated_nodes)
            self.current_iteration += 1


_OBSTRUCTION_CONFIG_NAMES = [
    "exp_pair_butter_pudding",
    "exp_pair_butter_cheese",
    "exp_pair_pudding_butter",
    "exp_pair_pudding_cheese",
    "exp_pair_cheese_butter",
    "exp_pair_cheese_pudding",
    "exp_single_butter_plate",
    "exp_single_butter_bowl",
    "exp_single_butter_tomato_sauce",
    "exp_single_butter_ketchup",
    "exp_single_butter_alphabet_soup",
    "exp_single_butter_orange_juice",
    "exp_single_butter_milk",
    "exp_single_pudding_plate",
    "exp_single_pudding_bowl",
    "exp_single_pudding_tomato_sauce",
    "exp_single_pudding_ketchup",
    "exp_single_pudding_alphabet_soup",
    "exp_single_pudding_orange_juice",
    "exp_single_pudding_milk",
    "exp_single_cheese_plate",
    "exp_single_cheese_bowl",
    "exp_single_cheese_tomato_sauce",
    "exp_single_cheese_ketchup",
    "exp_single_cheese_alphabet_soup",
    "exp_single_cheese_orange_juice",
    "exp_single_cheese_milk",
]


def _select_config_group(config: Dict[str, Any]) -> Dict[str, Any]:
    for group in config.get("groups", []):
        if group.get("name") == "baseline" and group.get("bddl_file"):
            return group
    for group in config.get("groups", []):
        if group.get("bddl_file"):
            return group
    for group in config.get("groups", []):
        if group.get("stages"):
            return group
    return {}


def _extract_bddl_and_instruction(group: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    if group.get("bddl_file"):
        return group["bddl_file"], group.get("instruction")
    stages = group.get("stages") or []
    if stages:
        stage = stages[0]
        return stage["bddl_file"], stage.get("instruction")
    raise KeyError("No bddl_file or stages found in group.")


def _load_seeds_from_configs(
    config_root: Path,
    config_names: Iterable[str],
    task_suite: str,
    file_to_id: Dict[str, int],
    task_type: str,
) -> List[Dict[str, Any]]:
    seeds: List[Dict[str, Any]] = []
    missing_configs: List[str] = []
    missing_bddl: List[str] = []
    skipped: List[str] = []

    for name in config_names:
        config_path = config_root / f"{name}.yaml"
        if not config_path.exists():
            missing_configs.append(str(config_path))
            continue
        config = yaml.safe_load(_load_text(config_path)) or {}
        group = _select_config_group(config)
        if not group:
            raise SystemExit(f"Config has no usable group: {config_path}")
        bddl_file, instruction = _extract_bddl_and_instruction(group)
        bddl_path = Path(bddl_file)
        if not bddl_path.is_absolute():
            bddl_path = _REPO_ROOT / bddl_path
        if not bddl_path.exists():
            missing_bddl.append(str(bddl_path))
            continue
        task_id = file_to_id.get(bddl_path.name)
        bddl_text = _load_text(bddl_path)
        seeds.append(
            {
                "task_suite": task_suite,
                "task_id": task_id,
                "bddl_name": bddl_path.name,
                "bddl_path": str(bddl_path),
                "bddl_text": bddl_text,
                "task_type": task_type,
                "task_instruction": instruction,
                "task_instruction_original": instruction,
                "config_name": name,
            }
        )

    if missing_configs:
        raise SystemExit("Missing config files:\n" + "\n".join(missing_configs))
    if missing_bddl:
        raise SystemExit("Missing BDDL files referenced by configs:\n" + "\n".join(missing_bddl))
    if not seeds:
        detail = ""
        if skipped:
            detail = "\nNot found in task suite: " + ", ".join(sorted(set(skipped)))
        raise SystemExit("No seeds found from obstruction configs." + detail)
    return seeds


class MCTSRunner:
    def __init__(self, checkpoint_dir: str):
        model, model_config = obstruction_run.load_pi05_model(checkpoint_dir)
        if model is None:
            raise SystemExit("Failed to load Pi0.5 model.")
        self.model = model
        self.model_config = model_config
        self.fail_count = 0  # 失败案例计数
        self.scene_counter = 0  # 场景计数
        
    def extract_target_object(self, options: Dict) -> str:
        """从options中提取目标物体"""
        instruction = options.get("task_instruction_original", options.get("task_instruction", ""))
        
        # 解析 "put the <target> in the basket"
        match = re.search(r'put the (.+?) in the basket', instruction)
        if match:
            target_desc = match.group(1)
            mapping = {
                'butter': 'butter',
                'butter box': 'butter',
                'cream cheese': 'cream_cheese',
                'chocolate pudding': 'chocolate_pudding',
            }
            return mapping.get(target_desc, target_desc.replace(' ', '_'))
        return "unknown"

    def extract_obstacles_from_bddl(self, bddl_text: str) -> List[tuple]:
        """从BDDL中提取障碍物（返回(实例名, 类型名)）- 已废弃，请使用extract_stacking_obstacles"""
        obstacles = []
        pattern = r'(\w+_obstacle_\d+)\s+-\s+(\w+)'
        for match in re.finditer(pattern, bddl_text):
            instance_name = match.group(1)  # 如 "wooden_tray_obstacle_1"
            type_name = match.group(2)       # 如 "wooden_tray"
            obstacles.append((instance_name, type_name))
        return obstacles
    
    def extract_stacking_obstacles(self, bddl_text: str, target_name: str) -> List[tuple]:
        """从BDDL的(:init)部分提取真正堆叠在目标物上的遮挡物
        
        Args:
            bddl_text: BDDL文件内容
            target_name: 目标物名称（如 "butter"）
            
        Returns:
            List[tuple]: 堆叠的遮挡物列表 [(实例名, 类型名), ...]
        """
        # 解析所有对象的类型映射（处理多个实例在一行的情况）
        objects_map = {}  # instance_name -> type_name
        lines = bddl_text.split('\n')
        in_objects = False
        
        for line in lines:
            stripped = line.strip()
            if '(:objects' in stripped:
                in_objects = True
                continue
            elif in_objects and stripped == ')':
                break
            elif in_objects and stripped and not stripped.startswith('('):
                # 匹配形如 "obj1 obj2 obj3 - type" 或 "obj1 - type"
                match = re.match(r'(.+)\s+-\s+(\w+)', stripped)
                if match:
                    instances_str = match.group(1)
                    type_name = match.group(2)
                    # 分割所有实例名
                    instances = instances_str.split()
                    for instance in instances:
                        objects_map[instance] = type_name
        
        # 解析init section中的堆叠关系
        stacking_relations = []  # [(object, base), ...]
        in_init = False
        
        for line in lines:
            stripped = line.strip()
            if '(:init' in stripped:
                in_init = True
                continue
            elif in_init and stripped.startswith(')'):
                break
            elif in_init:
                # 匹配 (On object base) 的形式
                match = re.match(r'\(On\s+(\w+)\s+(\w+)\)', stripped)
                if match:
                    obj = match.group(1)
                    base = match.group(2)
                    stacking_relations.append((obj, base))
        
        # 从目标物开始，递归找出所有堆叠在上面的物体
        target_instance = f"{target_name}_1"
        stacked_objects = []
        
        def find_stacked_on(base_obj: str):
            """递归查找堆叠在base_obj上的所有物体"""
            for obj, base in stacking_relations:
                if base == base_obj:
                    # 找到一个堆叠的物体
                    if obj in objects_map:
                        obj_type = objects_map[obj]
                        stacked_objects.append((obj, obj_type))
                        # 继续递归查找堆叠在这个物体上的
                        find_stacked_on(obj)
        
        find_stacked_on(target_instance)
        
        return stacked_objects

    def generate_obstacle_clear_instruction(self, obstacles: List[tuple]) -> str:
        """生成清除遮挡物的指令（处理真正堆叠在目标物上的对象）"""
        if not obstacles:
            return "no obstacles to clear"
        
        # 使用类型名映射到显示名称（扩展以支持非obstacle对象）
        display_names = {
            # 原有的obstacle对象
            'wooden_tray': 'wooden tray',
            'porcelain_mug': 'porcelain mug',
            'red_coffee_mug': 'red coffee mug',
            'white_yellow_mug': 'white and yellow mug',
            'chefmate_8_frypan': 'frying pan',
            'moka_pot': 'moka pot',
            'wine_bottle': 'wine bottle',
            'black_book': 'black book',
            # 真正的目标物（可能堆叠在底层目标物上）
            'chocolate_pudding': 'chocolate pudding',
            'butter': 'butter box',
            'cream_cheese': 'cream cheese',
        }
        
        # 获取所有遮挡物的显示名称（去重）
        obstacle_display_names = list(set([
            display_names.get(type_name, type_name.replace('_', ' '))
            for _, type_name in obstacles
        ]))
        
        # 生成指令
        if len(obstacle_display_names) == 1:
            # 单个遮挡物
            obj_name = obstacle_display_names[0]
            # 检查是否需要复数（如chocolate_pudding有多个实例）
            count = sum(1 for _, t in obstacles if display_names.get(t, t.replace('_', ' ')) == obj_name)
            if count > 1:
                return f"put the {obj_name} in the basket"
            else:
                return f"put the {obj_name} in the basket"
        else:
            # 多个不同类型的遮挡物
            return "put the obstacles in the basket"
    
    def modify_bddl_goal_for_obstacles(self, original_bddl_text: str, obstacles: List[tuple]) -> str:
        """修改BDDL的goal和obj_of_interest部分，将目标改为清除障碍物"""
        lines = original_bddl_text.split('\n')
        new_lines = []
        
        in_goal_section = False
        in_obj_interest_section = False
        goal_indent = ''
        obj_interest_indent = ''
        
        for line in lines:
            stripped = line.strip()
            
            # 检测obj_of_interest section开始
            if '(:obj_of_interest' in stripped:
                in_obj_interest_section = True
                obj_interest_indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(line)
                
                # 生成新的obj_of_interest：所有障碍物 + basket
                if obstacles:
                    for instance_name, _ in obstacles:
                        new_lines.append(f"{obj_interest_indent}    {instance_name}")
                    new_lines.append(f"{obj_interest_indent}    basket_1")
                continue
            
            # 跳过原有的obj_of_interest内容
            if in_obj_interest_section:
                if stripped == ')':
                    new_lines.append(line)
                    in_obj_interest_section = False
                continue
            
            # 检测goal section开始
            if '(:goal' in stripped:
                in_goal_section = True
                # 提取缩进
                goal_indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(line)
                
                # 生成新的goal：所有障碍物都要放入basket
                if obstacles:
                    # 如果只有一个障碍物
                    if len(obstacles) == 1:
                        instance_name = obstacles[0][0]
                        new_lines.append(f"{goal_indent}    (In {instance_name} basket_1_contain_region)")
                    else:
                        # 多个障碍物，使用And
                        goal_parts = []
                        for instance_name, _ in obstacles:
                            goal_parts.append(f"(In {instance_name} basket_1_contain_region)")
                        
                        # 如果只有2个障碍物
                        if len(obstacles) == 2:
                            new_lines.append(f"{goal_indent}    (And {goal_parts[0]}")
                            new_lines.append(f"{goal_indent}         {goal_parts[1]})")
                        else:
                            # 3个或更多障碍物
                            new_lines.append(f"{goal_indent}    (And {goal_parts[0]}")
                            for i in range(1, len(goal_parts) - 1):
                                new_lines.append(f"{goal_indent}         {goal_parts[i]}")
                            new_lines.append(f"{goal_indent}         {goal_parts[-1]})")
                continue
            
            # 跳过原有的goal内容
            if in_goal_section:
                if stripped == ')':
                    new_lines.append(line)
                    in_goal_section = False
                continue
            
            new_lines.append(line)
        
        return '\n'.join(new_lines)
    
    def create_cleared_bddl(self, original_bddl_text: str, target_name: str, stacked_obstacles: List[tuple]) -> str:
        """创建清除遮挡物后的BDDL（只移除堆叠的遮挡物，保留干扰物）
        
        Args:
            original_bddl_text: 原始BDDL内容
            target_name: 目标物名称
            stacked_obstacles: 堆叠的遮挡物列表 [(instance_name, type_name), ...]
        """
        lines = original_bddl_text.split('\n')
        new_lines = []
        
        # 获取要移除的实例名集合
        obstacles_to_remove = set(instance_name for instance_name, _ in stacked_obstacles)
        
        in_objects_section = False
        in_init_section = False
        in_regions_section = False
        
        for line in lines:
            stripped = line.strip()
            
            # 检测section开始
            if '(:regions' in stripped:
                in_regions_section = True
                new_lines.append(line)
                continue
            elif '(:objects' in stripped:
                in_regions_section = False
                in_objects_section = True
                new_lines.append(line)
                continue
            elif '(:init' in stripped:
                in_objects_section = False
                in_init_section = True
                new_lines.append(line)
                continue
            elif stripped.startswith('(:') or stripped.startswith('(define'):
                in_regions_section = False
                in_objects_section = False
                in_init_section = False
            
            # 处理regions section：移除遮挡物的region定义
            if in_regions_section:
                # 检查是否是遮挡物的region
                is_obstacle_region = False
                for obs_name in obstacles_to_remove:
                    if f"{obs_name}_init_region" in stripped:
                        is_obstacle_region = True
                        break
                
                if is_obstacle_region:
                    # 跳过整个region块
                    depth = 1 if '(' in stripped else 0
                    while depth > 0:
                        continue
                else:
                    new_lines.append(line)
            
            # 处理objects section：移除堆叠的遮挡物
            elif in_objects_section:
                if stripped == ')':
                    new_lines.append(line)
                    in_objects_section = False
                else:
                    # 检查这行是否包含要移除的对象
                    should_remove = False
                    for obs_name in obstacles_to_remove:
                        if obs_name in stripped:
                            should_remove = True
                            break
                    
                    if not should_remove:
                        new_lines.append(line)
            
            # 处理init section：移除堆叠遮挡物的初始化
            elif in_init_section:
                # 检查是否包含要移除的对象
                should_remove = False
                for obs_name in obstacles_to_remove:
                    if obs_name in stripped:
                        should_remove = True
                        break
                
                if not should_remove:
                    new_lines.append(line)
            
            else:
                new_lines.append(line)
        
        return '\n'.join(new_lines)

    def create_dual_configs(self, options: Dict, bddl_file: Path, output_base: Path) -> Tuple[Dict, Dict]:
        """创建baseline和guided两组配置（Guided使用多阶段结构）"""
        bddl_text = bddl_file.read_text()
        baseline_instruction = options.get("task_instruction_original", options.get("task_instruction", ""))
        
        # 提取目标物信息
        target = self.extract_target_object(options)
        
        # 提取真正堆叠在目标物上的遮挡物（不是所有obstacle对象）
        stacked_obstacles = self.extract_stacking_obstacles(bddl_text, target)
        
        # 生成目标物显示名称
        display_names = {
            'butter': 'butter box',
            'cream_cheese': 'cream cheese',
            'chocolate_pudding': 'chocolate pudding',
        }
        target_display = display_names.get(target, target.replace('_', ' '))
        
        # Baseline配置（单阶段）
        baseline_config = {
            "experiment": {
                "name": "baseline",
                "description": "baseline test",
            },
            "task": {
                "suite": "custom",
                "task_name": options.get("config_name", "baseline_task"),
            },
            "groups": [{
                "name": "baseline",
                "description": "直接移动目标物（有遮挡物堆叠）",
                "bddl_file": str(bddl_file),
                "instruction": baseline_instruction,
                "use_obstruction": False,
                "use_bddl_stacking": True,  # 使用BDDL中定义的堆叠
            }],
            "execution": {
                "episodes_per_group": 1,
                "max_steps_per_episode": 300,
                "seed_start": 0,
                "checkpoint_dir": "",
            },
            "output": {
                "results_dir": str(output_base / "baseline"),
                "save_images": True,
                "save_videos": False,
            },
        }
        
        # 生成清除遮挡物后的BDDL文件
        cleared_bddl_file = output_base / "mutated_cleared.bddl"
        stage1_bddl_file = output_base / "mutated_stage1.bddl"  # 第一阶段专用BDDL
        
        if stacked_obstacles:
            # 第一阶段BDDL：修改goal和obj_of_interest为堆叠的遮挡物
            stage1_bddl_text = self.modify_bddl_goal_for_obstacles(bddl_text, stacked_obstacles)
            stage1_bddl_file.write_text(stage1_bddl_text)
            
            # 第二阶段BDDL：清除堆叠的遮挡物（保留干扰物）
            cleared_bddl_text = self.create_cleared_bddl(bddl_text, target, stacked_obstacles)
            cleared_bddl_file.write_text(cleared_bddl_text)
            
            # 生成遮挡物清除指令
            obstacle_clear_instruction = self.generate_obstacle_clear_instruction(stacked_obstacles)
            
            # Guided配置（两阶段：先清除堆叠遮挡物，再移动目标）
            guided_config = {
                "experiment": {
                    "name": "guided",
                    "description": "guided two-stage optimization",
                },
                "task": {
                    "suite": "custom",
                    "task_name": options.get("config_name", "guided_task"),
                },
                "groups": [{
                    "name": "guided",
                    "description": "两阶段：先清除堆叠遮挡物，再移动目标物",
                    "use_obstruction": False,
                    "stages": [
                        {
                            "stage_name": "remove_stacked_obstacles",
                            "bddl_file": str(stage1_bddl_file),  # 使用修改过goal的BDDL
                            "instruction": obstacle_clear_instruction,
                            "target_object": f"{stacked_obstacles[0][0]}" if stacked_obstacles else None,
                        },
                        {
                            "stage_name": "move_target",
                            "bddl_file": str(cleared_bddl_file),  # 清除遮挡物后的BDDL
                            "instruction": f"put the {target_display} in the basket",
                            "target_object": f"{target}_1",
                        }
                    ]
                }],
                "execution": {
                    "episodes_per_group": 1,
                    "max_steps_per_episode": 300,
                    "seed_start": 0,
                    "checkpoint_dir": "",
                },
                "output": {
                    "results_dir": str(output_base / "guided"),
                    "save_images": True,
                    "save_videos": False,
                },
            }
        else:
            # 没有障碍物时，guided和baseline相同
            guided_config = baseline_config.copy()
            guided_config["experiment"]["name"] = "guided"
            guided_config["output"]["results_dir"] = str(output_base / "guided")
        
        return baseline_config, guided_config

    def _run_single_config(self, config: Dict, output_dir: Path) -> bool:
        """运行单个配置，返回是否成功"""
        try:
            group_config = config["groups"][0]
            env = None
            env, summary = obstruction_run.run_group_with_cache(
                env,
                config,
                group_config,
                "group1_" + group_config["name"],
                output_dir,
                self.model,
                self.model_config,
                None,
                0,
            )
            if env is not None:
                env.close()
            
            success = summary.get("success_count", 0) > 0
            return success
        except Exception as e:
            print(f"运行失败: {e}")
            return False

    def run_episode(
        self,
        options: Dict[str, Any],
        max_steps: int,
        seed: Optional[int],
        output_dir: Optional[Path],
        run_index: int,
    ) -> Dict[str, Any]:
        instruction = options.get("task_instruction") or options.get("task_instruction_original") or ""
        bddl_text = options["bddl_text"]

        run_output_dir: Optional[Path]
        temp_output_dir: Optional[tempfile.TemporaryDirectory] = None
        if output_dir is None:
            temp_output_dir = tempfile.TemporaryDirectory()
            run_output_dir = Path(temp_output_dir.name)
        else:
            run_output_dir = output_dir

        seed_start = seed if seed is not None else 0

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_bddl_path = Path(temp_dir) / "mutated.bddl"
            temp_bddl_path.write_text(bddl_text, encoding="utf-8")

            config = {
                "experiment": {
                    "name": options.get("config_name", options.get("bddl_name", "mcts_seed")),
                    "description": "mcts_mutation",
                },
                "task": {
                    "suite": "custom",
                    "task_name": options.get("config_name", options.get("bddl_name", "mcts_task")),
                },
                "groups": [
                    {
                        "name": "baseline",
                        "description": "mcts_eval",
                        "bddl_file": str(temp_bddl_path),
                        "instruction": instruction,
                        "use_obstruction": False,
                    }
                ],
                "execution": {
                    "episodes_per_group": 1,
                    "max_steps_per_episode": max_steps,
                    "seed_start": seed_start,
                    "checkpoint_dir": "",
                },
                "output": {
                    "results_dir": str(run_output_dir),
                    "save_images": True,
                    "save_videos": False,
                },
            }

            if run_output_dir is not None:
                with (run_output_dir / "config.yaml").open("w") as handle:
                    yaml.safe_dump(config, handle, sort_keys=False)

            group_config = config["groups"][0]
            env = None
            env, summary = obstruction_run.run_group_with_cache(
                env,
                config,
                group_config,
                "group1_baseline",
                run_output_dir,
                self.model,
                self.model_config,
                None,
                0,
            )
            if env is not None:
                env.close()

        if temp_output_dir is not None:
            temp_output_dir.cleanup()

        success = summary.get("success_count", 0) > 0
        return {
            "success": success,
            "summary": summary,
                "output_dir": str(run_output_dir) if run_output_dir else None,
        }


def print_final_summary(output_dir: Path):
    """打印最终统计"""
    if not output_dir or not output_dir.exists():
        print("No output directory to summarize.")
        return
    
    # 统计所有场景
    all_scenes = list(output_dir.glob("scene_*"))
    baseline_fail_count = 0
    guided_success_count = 0
    
    for scene_dir in all_scenes:
        result_file = scene_dir / "result.json"
        if result_file.exists():
            result = json.load(open(result_file))
            if not result.get("baseline_success", True):
                baseline_fail_count += 1
                if result.get("guided_success", False):
                    guided_success_count += 1
    
    print(f"\n{'='*70}")
    print(f"最终统计结果")
    print(f"{'='*70}")
    print(f"Baseline失败案例数: {baseline_fail_count}")
    print(f"Guided成功救回:     {guided_success_count}")
    print(f"最终失败数:         {baseline_fail_count - guided_success_count}")
    if baseline_fail_count > 0:
        print(f"Guided成功率:       {guided_success_count / baseline_fail_count * 100:.2f}%")
    print(f"{'='*70}")
    
    # 保存到文件
    summary = {
        "baseline_failed": baseline_fail_count,
        "guided_success": guided_success_count,
        "final_failed": baseline_fail_count - guided_success_count,
        "guided_success_rate": guided_success_count / baseline_fail_count * 100 if baseline_fail_count > 0 else 0,
    }
    
    with open(output_dir / "final_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果已保存到: {output_dir / 'final_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MCTS fuzzing with Pi0.5 using BDDL seeds.")
    parser.add_argument("--task-suite", type=str, default="libero_90")
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["put-in", "put-on", "grasp", "move"],
        default="put-in",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="/mnt/disk1/decom/VLATest/IsaacLab/pi05_libero")
    parser.add_argument("--energy", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--output", type=str, default="./fuzzer_output")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--strategies",
        type=str,
        default="expand_random,rephrase",
    )
    parser.add_argument("--num-failures", type=int, default=80,
                       help="目标失败案例数量（收集多少个baseline失败的案例后停止）")
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    file_to_id, _ = _load_libero_task_map(args.task_suite)
    config_root = _ROOT_DIR / "obstruction" / "configs"
    seeds = _load_seeds_from_configs(
        config_root=config_root,
        config_names=_OBSTRUCTION_CONFIG_NAMES,
        task_suite=args.task_suite,
        file_to_id=file_to_id,
        task_type=args.task_type,
    )
    for seed in seeds:
        seed["max_steps"] = args.max_steps

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    invalid = [s for s in strategies if s not in {"expand_random", "rephrase"}]
    if invalid:
        raise SystemExit(f"Unsupported strategies: {', '.join(invalid)}")
    variation = BDDLVariation(seed=args.seed)
    mutate_policy = OptionsMutatePolicy(variation, strategies=strategies)
    select_policy = MCTSExploreSelectPolicy()

    output_dir = None
    if args.output:
        run_tag = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        output_dir = Path(args.output) / f"mcts_{run_tag}"

    try:
        runner = MCTSRunner(checkpoint_dir=args.checkpoint_dir)
        fuzzer = OptionsFuzzer(
            initial_seeds=seeds,
            mutate_policy=mutate_policy,
            select_policy=select_policy,
            runner=runner,
            task_suite=args.task_suite,
            energy=args.energy,
            max_iteration=args.max_iter,
            output_dir=output_dir,
            run_seed=args.seed,
            num_failures=args.num_failures,  # 传递目标失败案例数
        )
        fuzzer.verbose = args.verbose
        fuzzer.run()
        
        # 打印最终统计
        if output_dir:
            print_final_summary(output_dir)
    finally:
        pass


if __name__ == "__main__":
    main()
