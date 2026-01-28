import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import sys

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))

from utils.interface import VLAInterface, VLAInterfaceLM
from utils.variation import Variation


class StableJSONizer(json.JSONEncoder):
    def default(self, obj):
        return super().encode(bool(obj)) if isinstance(obj, np.bool_) else super().default(obj)


def _iter_seed_options(root: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(root.glob("**/options.json")):
        try:
            with path.open("r") as handle:
                yield json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue


def _compute_success(episode_stats: Any) -> bool:
    if not isinstance(episode_stats, dict) or not episode_stats:
        return False
    try:
        keys = sorted(episode_stats.keys(), key=lambda k: int(k))
    except Exception:
        keys = list(episode_stats.keys())
    for key in reversed(keys):
        info = episode_stats.get(key, {})
        if isinstance(info, dict) and "success" in info:
            return bool(info.get("success"))
    for info in episode_stats.values():
        if isinstance(info, dict) and info.get("success"):
            return True
    return False


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
    def __init__(self, variation: Variation, strategies: List[str]):
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
            # Sample one of the Variation strategies to mutate the seed options.
            strategy = random.choice(self.strategies)
            if getattr(self._fuzzer, "verbose", False):
                print(f"[mutate] strategy={strategy} parent_index={prompt_node.index}")
            seed_options = prompt_node.prompt
            if strategy == "generate":
                mutated = self.variation.generate(seed_options)
            elif strategy == "crossover":
                other = random.choice(self._fuzzer.prompt_nodes).prompt
                mutated = self.variation.crossover(seed_options, other)
            elif strategy == "expand":
                mutated = self.variation.expand(seed_options, None)
            elif strategy == "rephrase":
                task_type = seed_options.get("task_type", "grasp")
                mutated = self.variation.rephrase(task_type, seed_options)
            elif strategy == "reset":
                mutated = self.variation.reset(seed_options)
            else:
                mutated = self.variation.generate(seed_options)
            # Ensure model_id is present and consistent with the task database.
            self.variation._ensure_model_id(mutated)
            results.append(PromptNode(self._fuzzer, mutated, parent=prompt_node, mutator=strategy))
        return results


class OptionsFuzzer:
    def __init__(
        self,
        initial_seeds: List[Dict[str, Any]],
        mutate_policy: OptionsMutatePolicy,
        select_policy: MCTSExploreSelectPolicy,
        model_name: str,
        lora_path: Optional[str],
        energy: int = 1,
        max_iteration: int = 100,
        output_dir: Optional[Path] = None,
        run_seed: Optional[int] = None,
        save_images: bool = False,
    ):
        self.questions = [None]
        # prompt_nodes: 所有种子及其后续变异节点的集合
        self.prompt_nodes = [PromptNode(self, seed) for seed in initial_seeds]
        # initial_prompts_nodes: 初始种子节点（MCTS 只从这些根节点开始选择）
        self.initial_prompts_nodes = self.prompt_nodes.copy()
        for i, prompt_node in enumerate(self.prompt_nodes):
            # index: 节点在全局池中的编号，MCTS 依赖该索引维护奖励表
            prompt_node.index = i

        # mutate_policy: 变异策略（调用 variation.py 的五种方法之一）
        self.mutate_policy = mutate_policy
        # select_policy: 选择策略（MCTSExploreSelectPolicy）
        self.select_policy = select_policy
        self.mutate_policy.fuzzer = self
        self.select_policy.fuzzer = self

        # energy: 每轮从同一个 seed 生成多少个变异样本
        self.energy = energy
        # max_iteration: 最大迭代轮数
        self.max_iteration = max_iteration
        # current_iteration: 当前迭代计数
        self.current_iteration = 0
        # run_seed: 环境 reset 时使用的随机种子
        self.run_seed = run_seed

        # model_name: OpenVLA/RT-1/Octo 模型名称
        self.model_name = model_name
        # lora_path: OpenVLA 的 LoRA 权重路径（可选）
        self.lora_path = lora_path
        # save_images: 是否保存 rollout 图像
        self.save_images = save_images
        # output_dir: 保存 log/options/actions 的输出目录（可选）
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # run_counter: 递增的场景编号，用于保存结果到 scene_x 目录
        self.run_counter = 0

        # _vla_cache: 按 task 缓存 VLAInterface，避免重复初始化
        self._vla_cache: Dict[str, VLAInterface] = {}
        self.verbose = False

    def _get_vla(self, task: str) -> VLAInterface:
        if task not in self._vla_cache:
            if self.verbose:
                print(f"[vla] init task={task} model={self.model_name} lora={self.lora_path}")
            self._vla_cache[task] = VLAInterface(task=task, model_name=self.model_name, lora_path=self.lora_path)
        return self._vla_cache[task]

    def prewarm(self, tasks: Iterable[str]) -> None:
        for task in tasks:
            self._get_vla(task)

    def is_stop(self) -> bool:
        return self.max_iteration != -1 and self.current_iteration >= self.max_iteration

    def evaluate(self, prompt_nodes: List[PromptNode]) -> None:
        for idx, prompt_node in enumerate(prompt_nodes):
            options = prompt_node.prompt
            task = options.get("task")
            vla = self._get_vla(task)
            instruction = options.get("task_instruction")
            if self.verbose:
                task_type = options.get("task_type", "unknown")
                print(
                    f"[eval] idx={idx} node_index={prompt_node.index} "
                    f"task_type={task_type} task={task} instruction={instruction} "
                    f"options:{options}"
                )
            options.pop("robot_init_options", None)
            images, episode_stats, actions = vla.run_interfaceWithPromot(
                seed=self.run_seed, options=options,promot=None
            )
            success = _compute_success(episode_stats)
            prompt_node.response = {"success": success}
            # Treat success as reward (jailbreak=1) to keep successful cases.
            prompt_node.results = [1 if success else 0]

            if self.output_dir is None:
                continue
            task_type = options.get("task_type", "unknown")
            outcome = "success" if success else "failure"
            run_dir = (
                self.output_dir
                / task_type
                / outcome
                / f"scene_{self.run_counter:04d}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "log.json").open("w") as handle:
                json.dump(episode_stats, handle, cls=StableJSONizer)
            with (run_dir / "options.json").open("w") as handle:
                json.dump(options, handle, cls=StableJSONizer)
            actions_list = [a.tolist() if hasattr(a, "tolist") else a for a in actions]
            with (run_dir / "actions.json").open("w") as handle:
                json.dump(actions_list, handle, cls=StableJSONizer)
            if self.save_images and images:
                image_dir = run_dir / "image"
                image_dir.mkdir(parents=True, exist_ok=True)
                for img_idx, img in enumerate(images):
                    from PIL import Image
                    Image.fromarray(img).save(image_dir / f"{img_idx}.jpg")
            self.run_counter += 1

    def update(self, prompt_nodes: List[PromptNode]) -> None:
        for prompt_node in prompt_nodes:
            # Only keep nodes that achieved "reward" according to the scoring rule.
            if prompt_node.num_jailbreak > 0:
                prompt_node.index = len(self.prompt_nodes)
                self.prompt_nodes.append(prompt_node)
                if self.verbose:
                    print(f"[update] keep node_index={prompt_node.index} reward={prompt_node.num_jailbreak}")
        self.select_policy.update(prompt_nodes)

    def run(self) -> None:
        while not self.is_stop():
            if self.verbose:
                print(
                    f"[run] iter={self.current_iteration} "
                    f"total_nodes={len(self.prompt_nodes)}"
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


def _load_seeds(
    root: Optional[str],
    task_type: str,
    env_task: str,
) -> List[Dict[str, Any]]:
    if not root:
        return []
    seeds: List[Dict[str, Any]] = []
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = _ROOT_DIR / root_path
    for options in _iter_seed_options(root_path):
        options = dict(options)
        options["task"] = env_task
        options["task_type"] = task_type
        seeds.append(options)
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="MCTS fuzzing with VLA options.")
    parser.add_argument("--grasp-seed-root", type=str, default="result/openvla-7b_2024/grasp/failure")
    parser.add_argument("--move-seed-root", type=str, default="result/openvla-7b_2024/move/failure")
    parser.add_argument("--grasp-task", type=str, default="google_robot_pick_customizable")
    parser.add_argument("--move-task", type=str, default="google_robot_move_near_customizable")
    parser.add_argument("--model", type=str, default="openvla-7b")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--energy", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save-images", action="store_true", default=True)
    parser.add_argument("--no-save-images", dest="save_images", action="store_false")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--strategies", type=str, default="generate,crossover,expand,rephrase,reset")
    parser.add_argument("--task-type", type=str, default="grasp", choices=["grasp", "move", "both"])
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    seeds = []
    if args.task_type in ("grasp", "both"):
        seeds.extend(_load_seeds(args.grasp_seed_root, "grasp", args.grasp_task))
    if args.task_type in ("move", "both"):
        seeds.extend(_load_seeds(args.move_seed_root, "move", args.move_task))
    if not seeds:
        raise SystemExit("No seeds found. Check your seed roots.")

    variation = Variation(seed=args.seed)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    mutate_policy = OptionsMutatePolicy(variation, strategies=strategies)
    select_policy = MCTSExploreSelectPolicy()

    output_dir = None
    if args.output:
        run_tag = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        output_dir = Path(args.output) / f"mcts_{run_tag}"

    fuzzer = OptionsFuzzer(
        initial_seeds=seeds,
        mutate_policy=mutate_policy,
        select_policy=select_policy,
        model_name=args.model,
        lora_path=args.lora_path,
        energy=args.energy,
        max_iteration=args.max_iter,
        output_dir=output_dir,
        run_seed=args.seed,
        save_images=args.save_images,
    )
    fuzzer.verbose = args.verbose
    fuzzer.prewarm(sorted({seed.get("task") for seed in seeds if seed.get("task")}))
    fuzzer.run()


if __name__ == "__main__":
    main()
