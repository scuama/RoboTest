import copy
import json
import os
import re
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


import numpy as np
import simpler_env

from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


import cv2
import numpy as np

try:
    from utils.camera import RandomCamera
    from utils.lighting import RandomLighting
except ModuleNotFoundError:
    # Allow running this file directly from the utils directory.
    from camera import RandomCamera
    from lighting import RandomLighting

_COLOR_BACKGROUNDS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "gray": (128, 128, 128),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "navy": (0, 0, 128),
    "teal": (0, 128, 128),
    "olive": (128, 128, 0),
    "maroon": (128, 0, 0),
    "gold": (255, 215, 0),
    "lime": (191, 255, 0),
    "sky": (135, 206, 235),
}

_COLOR_BG_DIR = Path(__file__).parent / "assets" / "color_backgrounds"
_COLOR_OCCLUSION_DIR = Path(__file__).parent / "assets" / "color_occlusions"


_TEMPLATES = {
    "grasp": [
        "pick [OBJECT]",
        "grab [OBJECT]",
        "can you pick up [OBJECT]",
        "fetch [OBJECT]",
        "get [OBJECT]",
        "lift [OBJECT]",
        "take [OBJECT]",
        "retrieve [OBJECT]",
        "let's pick up [OBJECT]",
        "would you grab [OBJECT]",
    ],
    "move": [
        "Take [OBJECT A] to [OBJECT B]",
        "Bring [OBJECT A] close to [OBJECT B]",
        "Position [OBJECT A] near [OBJECT B]",
        "Move [OBJECT A] closer to [OBJECT B]",
        "Put [OBJECT A] by [OBJECT B]",
        "Place [OBJECT A] near [OBJECT B]",
        "Set [OBJECT A] next to [OBJECT B]",
        "Can you move [OBJECT A] near [OBJECT B]",
        "Shift [OBJECT A] near [OBJECT B]",
        "Let's move [OBJECT A] near [OBJECT B]",
    ],
    "put-on": [
        "place [OBJECT A] on [OBJECT B]",
        "set [OBJECT A] on [OBJECT B]",
        "move [OBJECT A] onto [OBJECT B]",
        "position [OBJECT A] on [OBJECT B]",
        "put [OBJECT A] onto [OBJECT B]",
        "could you put [OBJECT A] on [OBJECT B]",
        "let's put [OBJECT A] on [OBJECT B]",
        "please place [OBJECT A] on [OBJECT B]",
        "can you place [OBJECT A] on [OBJECT B]",
        "would you move [OBJECT A] onto [OBJECT B]",
    ],
    "put-in": [
        "take [OBJECT] into the yellow basket",
        "bring [OBJECT] into the yellow basket",
        "place [OBJECT] in the yellow basket",
        "move [OBJECT] inside the yellow basket",
        "put [OBJECT] inside the yellow basket",
        "drop [OBJECT] into the yellow basket",
        "insert [OBJECT] into the yellow basket",
        "can you put [OBJECT] into the yellow basket",
        "please put [OBJECT] into the yellow basket",
        "let's put [OBJECT] into the yellow basket",
    ],
}


def _get_instruction_obj_name(name: str) -> str:
    parts = name.split("_")
    rm_list = {
        "opened",
        "light",
        "generated",
        "modified",
        "objaverse",
        "bridge",
        "baked",
        "v2",
    }
    cleaned = []
    for word in parts:
        if word.endswith("cm"):
            continue
        if word not in rm_list:
            cleaned.append(word)
    return " ".join(cleaned)


def _gpt_rephrase_instruction(instruction: str) -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    model = os.environ.get("OPENAI_REPHRASE_MODEL", "gpt-3.5-turbo")
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite robot task instructions while preserving meaning. "
                        "Keep it concise, imperative, and do not add new objects or constraints."
                        "When translating, you need to expand the instructional information. For example, when the original command is 'pick xxx', you can elaborate it into something like 'grasp xxx and lift it up', rather than just translating it word-for-word by simply adding a single equivalent term"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Instruction: {instruction}\nRewritten instruction:",
                },
            ],
            temperature=0.2,
            max_tokens=64,
        )
    except Exception:
        return None

    if not response.choices:
        return None

    rewritten = response.choices[0].message.content.strip()
    return rewritten or None


def _normalize_instruction(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _random_xy(rng: np.random.RandomState, xy_range: Tuple[float, float]) -> List[float]:
    return [float(rng.uniform(*xy_range)), float(rng.uniform(*xy_range))]


def _random_z_quat(rng: np.random.RandomState) -> List[float]:
    yaw = float(rng.uniform(-np.pi, np.pi))
    return [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]

def _ensure_color_patch(path: Path, rgb: Tuple[int, int, int], size: int = 128) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = np.zeros((size, size, 3), dtype=np.uint8)
    bgr[:, :, 0] = rgb[2]
    bgr[:, :, 1] = rgb[1]
    bgr[:, :, 2] = rgb[0]
    cv2.imwrite(str(path), bgr)


class Variation:
    def __init__(
        self,
        seed: Optional[int] = None,
        camera_base: Optional[str] = None,
        lighting_direction: Optional[str] = None,
        task_name: Optional[str] = None,
        model_pool: Optional[Iterable[str]] = None,
        rephrase_fn: Optional[Any] = None,
        distractor_count_range: Tuple[int, int] = (1, 3),
        robot_init_xy_range: Tuple[float, float] = (0.30, 0.45),
        robot_init_yaw_range: Tuple[float, float] = (-np.pi, np.pi),
        distractor_xy_offset_range: Tuple[float, float] = (-0.3, 0.3),
        target_xy_offset_range: Tuple[float, float] = (-0.15, 0.15),
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)
        self.camera_fuzzer = RandomCamera(camera_base) if camera_base else None
        self.lighting_fuzzer = RandomLighting(lighting_direction)
        self.task_name = task_name
        self.model_pool = list(model_pool) if model_pool else None
        self.rephrase_fn = rephrase_fn
        self.distractor_count_range = distractor_count_range
        self.robot_init_xy_range = robot_init_xy_range
        self.robot_init_yaw_range = robot_init_yaw_range
        self.distractor_xy_offset_range = distractor_xy_offset_range
        self.target_xy_offset_range = target_xy_offset_range

    def generate(
        self,
        options: Dict[str, Any],
        background_choices: Optional[Iterable[str]] = None,
        occlusion_choices: Optional[Iterable[Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a new test combo by updating lighting and robot placement."""
        mutated = copy.deepcopy(options)

        self._ensure_model_id(mutated)

        lighting_options = self.lighting_fuzzer.generate_options()
        mutated.update(lighting_options)

        self._randomize_robot(mutated)

        return mutated

    def crossover(self, parent_a: Dict[str, Any], parent_b: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse two test combos into a new one."""
        merged = self._crossover_dict(parent_a, parent_b)

        target_source = self.py_rng.choice([parent_a, parent_b])
        for key in ("model_id", "obj_init_options", "model_ids", "source_obj_id", "target_obj_id"):
            if key in target_source:
                merged[key] = copy.deepcopy(target_source[key])

        distractors: List[str] = []
        for parent in (parent_a, parent_b):
            for obj in parent.get("distractor_model_ids", []):
                if obj not in distractors:
                    distractors.append(obj)

        if distractors:
            merged["distractor_model_ids"] = distractors

        merged_distractors: Dict[str, Any] = {}
        for obj in distractors:
            a_cfg = parent_a.get("distractor_obj_init_options", {}).get(obj)
            b_cfg = parent_b.get("distractor_obj_init_options", {}).get(obj)
            if a_cfg is not None and b_cfg is not None:
                merged_distractors[obj] = copy.deepcopy(self.py_rng.choice([a_cfg, b_cfg]))
            elif a_cfg is not None:
                merged_distractors[obj] = copy.deepcopy(a_cfg)
            elif b_cfg is not None:
                merged_distractors[obj] = copy.deepcopy(b_cfg)

        if merged_distractors:
            merged["distractor_obj_init_options"] = merged_distractors

        return merged

    def expand(
        self,
        options: Dict[str, Any],
        distractor_pool: Optional[Iterable[str]] = None,
        num: int = 1,
        xy_range: Tuple[float, float] = (-0.5, 0.5),
    ) -> Dict[str, Any]:
        """Add distractor objects to expand a test combo."""
        mutated = copy.deepcopy(options)
        if distractor_pool is None:
            distractor_pool = self._get_model_pool(mutated)
        distractor_pool = list(distractor_pool)
        num = self.py_rng.randint(1, 5)
        existing = set(mutated.get("distractor_model_ids", []))
        choices = [obj for obj in distractor_pool if obj not in existing]
        if not choices or num <= 0:
            return mutated

        selected = self.py_rng.sample(choices, min(num, len(choices)))
        mutated.setdefault("distractor_model_ids", [])
        mutated.setdefault("distractor_obj_init_options", {})
        for obj in selected:
            mutated["distractor_model_ids"].append(obj)
            mutated["distractor_obj_init_options"][obj] = {
                "init_xy": _random_xy(self.rng, xy_range),
                "init_rot_quat": _random_z_quat(self.rng),
            }
        return mutated

    def rephrase(self, task: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Rewrite task instruction while preserving the semantics."""
        mutated = copy.deepcopy(options)
        if self.rephrase_fn:
            try:
                instruction = self.rephrase_fn(task, mutated)
            except Exception:
                return mutated
            if instruction:
                mutated["task_instruction"] = instruction
            return mutated

        if task not in _TEMPLATES:
            return mutated

        try:
            if task == "grasp":
                obj = _get_instruction_obj_name(options["model_id"])
                instruction = self.py_rng.choice(_TEMPLATES[task]).replace("[OBJECT]", obj)
            elif task == "move":
                ids = options["model_ids"]
                obj_a = _get_instruction_obj_name(ids[options["source_obj_id"]])
                obj_b = _get_instruction_obj_name(ids[options["target_obj_id"]])
                instruction = (
                    self.py_rng.choice(_TEMPLATES[task])
                    .replace("[OBJECT A]", obj_a)
                    .replace("[OBJECT B]", obj_b)
                )
            elif task == "put-on":
                obj_a = _get_instruction_obj_name(options["model_ids"][0])
                obj_b = _get_instruction_obj_name(options["model_ids"][1])
                instruction = (
                    self.py_rng.choice(_TEMPLATES[task])
                    .replace("[OBJECT A]", obj_a)
                    .replace("[OBJECT B]", obj_b)
                )
            elif task == "put-in":
                obj = _get_instruction_obj_name(options["model_ids"][0])
                instruction = self.py_rng.choice(_TEMPLATES[task]).replace("[OBJECT]", obj)
            else:
                return mutated
        except (KeyError, IndexError, TypeError):
            return mutated
  
        gpt_instruction = _gpt_rephrase_instruction(instruction)
        if gpt_instruction and _normalize_instruction(gpt_instruction) != _normalize_instruction(instruction):
            mutated["task_instruction"] = gpt_instruction
        else:
            mutated["task_instruction"] = self._expand_instruction(instruction)
        return mutated

    def _expand_instruction(self, instruction: str) -> str:
        prefixes = [
            "Please",
            "Carefully",
            "Gently",
            "Smoothly",
            "With a steady grip,",
        ]
        suffixes = [
            "in a smooth motion",
            "with a steady grip",
            "gently and steadily",
            "without hesitation",
        ]
        variants: List[str] = []
        for prefix in prefixes:
            variants.append(f"{prefix} {instruction}")
        for suffix in suffixes:
            variants.append(f"{instruction} {suffix}")
        for prefix in prefixes:
            for suffix in suffixes:
                variants.append(f"{prefix} {instruction} {suffix}")

        base_norm = _normalize_instruction(instruction)
        candidates = [
            v for v in variants if _normalize_instruction(v) != base_norm
        ]
        if not candidates:
            return instruction
        return self.py_rng.choice(candidates)

    def reset(
        self,
        options: Dict[str, Any],
        xy_range: Tuple[float, float] = (-0.5, 0.5),
    ) -> Dict[str, Any]:
        """Rearrange target/distractor positions in a test combo."""
        mutated = copy.deepcopy(options)

        obj_opts = mutated.get("obj_init_options")
        if isinstance(obj_opts, dict):
            if self._is_xy(obj_opts.get("init_xy")):
                obj_opts["init_xy"] = _random_xy(self.rng, xy_range)
            else:
                model_id = mutated.get("model_id")
                if model_id and isinstance(obj_opts.get(model_id), dict):
                    obj_opts[model_id]["init_xy"] = _random_xy(self.rng, xy_range)
                else:
                    for cfg in obj_opts.values():
                        if isinstance(cfg, dict) and self._is_xy(cfg.get("init_xy")):
                            cfg["init_xy"] = _random_xy(self.rng, xy_range)
                            break
            mutated["obj_init_options"] = obj_opts

        if "distractor_obj_init_options" in mutated:
            for name, cfg in mutated["distractor_obj_init_options"].items():
                if isinstance(cfg, dict):
                    cfg["init_xy"] = _random_xy(self.rng, xy_range)
                    mutated["distractor_obj_init_options"][name] = cfg

        return mutated

    def _crossover_dict(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        keys = set(a.keys()) | set(b.keys())
        for key in keys:
            if key in a and key in b:
                va, vb = a[key], b[key]
                if isinstance(va, dict) and isinstance(vb, dict):
                    merged[key] = self._crossover_dict(va, vb)
                else:
                    merged[key] = copy.deepcopy(self.py_rng.choice([va, vb]))
            else:
                merged[key] = copy.deepcopy(a.get(key, b.get(key)))
        return merged

    @staticmethod
    def _choose_existing_key(options: Dict[str, Any], candidates: List[str]) -> str:
        for key in candidates:
            if key in options:
                return key
        return candidates[0]

    @staticmethod
    def _get_color_background_path(color_name: str, rgb: Tuple[int, int, int], options: Dict[str, Any]) -> Path:
        _COLOR_BG_DIR.mkdir(parents=True, exist_ok=True)
        camera_cfgs = options.get("camera_cfgs", {})
        width = int(camera_cfgs.get("width", 256)) if isinstance(camera_cfgs, dict) else 256
        height = int(camera_cfgs.get("height", 256)) if isinstance(camera_cfgs, dict) else 256
        path = _COLOR_BG_DIR / f"{color_name}_{width}x{height}.png"
        if not path.exists():
            bgr = np.zeros((height, width, 3), dtype=np.uint8)
            bgr[:, :, 0] = rgb[2]
            bgr[:, :, 1] = rgb[1]
            bgr[:, :, 2] = rgb[0]
            cv2.imwrite(str(path), bgr)
        return path

    def _randomize_objects(self, options: Dict[str, Any]) -> None:
        if not any(k in options for k in ("model_id", "distractor_model_ids", "distractor_obj_init_options")):
            return

        pool = self._get_model_pool(options)
        if not pool:
            return

        model_id = self.py_rng.choice(pool)
        options["model_id"] = model_id

        robot_xy = None
        robot_opts = options.get("robot_init_options")
        if isinstance(robot_opts, dict):
            robot_xy = robot_opts.get("init_xy")

        if robot_xy is not None and len(robot_xy) == 2:
            dx = float(self.rng.uniform(*self.target_xy_offset_range))
            dy = float(self.rng.uniform(*self.target_xy_offset_range))
            init_xy = [float(robot_xy[0]) + dx, float(robot_xy[1]) + dy]
        else:
            init_xy = _random_xy(self.rng, (-0.2, 0.2))
        options["obj_init_options"] = {
            "init_xy": init_xy,
            "orientation": _random_z_quat(self.rng),
        }

        pool_no_target = [m for m in pool if m != model_id]
        if not pool_no_target:
            pool_no_target = pool

        min_k, max_k = self.distractor_count_range
        if max_k < min_k:
            min_k, max_k = max_k, min_k
        max_k = min(max_k, len(pool_no_target))
        if max_k < min_k:
            min_k = max_k
        num = self.py_rng.randint(min_k, max_k + 1) if max_k > 0 else 0

        if num > 0:
            if len(pool_no_target) >= num:
                distractors = self.py_rng.sample(pool_no_target, num)
            else:
                distractors = [self.py_rng.choice(pool_no_target) for _ in range(num)]
        else:
            distractors = []

        options["distractor_model_ids"] = distractors
        options["distractor_obj_init_options"] = {}
        for obj in distractors:
            if robot_xy is not None and len(robot_xy) == 2:
                dx = float(self.rng.uniform(*self.distractor_xy_offset_range))
                dy = float(self.rng.uniform(*self.distractor_xy_offset_range))
                init_xy = [float(robot_xy[0]) + dx, float(robot_xy[1]) + dy]
            else:
                init_xy = _random_xy(self.rng, (-0.5, 0.5))
            options["distractor_obj_init_options"][obj] = {
                "init_xy": init_xy,
                "init_rot_quat": _random_z_quat(self.rng),
            }

    def _get_model_pool(self, options: Dict[str, Any]) -> List[str]:
        if self.model_pool:
            return self.model_pool
        pool = options.get("model_pool")
        if isinstance(pool, (list, tuple)) and pool:
            return list(pool)

        model_json = options.get("model_json") or options.get("model_json_path")
        if isinstance(model_json, str):
            model_path = Path(model_json)
            if model_path.exists():
                return self._load_model_json(model_path)

        task_name = self._get_task_name(options)
        task_path = self._get_model_json_from_task(task_name) if task_name else None
        if task_path is not None and task_path.exists():
            return self._load_model_json(task_path)

        default_path = Path(__file__).parent.parent / "ManiSkill2_real2sim" / "data" / "ycb-dataset" / "info_ycb.json"
        if default_path.exists():
            return self._load_model_json(default_path)

        return []

    @staticmethod
    def _load_model_json(path: Path) -> List[str]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return list(data.keys())
        except (OSError, json.JSONDecodeError):
            return []

    def _ensure_model_id(self, options: Dict[str, Any]) -> None:
        if options.get("model_id") is not None:
            return
        pool = self._get_model_pool(options)
        if pool:
            options["model_id"] = self.py_rng.choice(pool)

    def _get_task_name(self, options: Dict[str, Any]) -> Optional[str]:
        for key in ("task", "task_name", "env_name"):
            value = options.get(key)
            if isinstance(value, str) and value:
                return value
        return self.task_name

    def _get_model_json_from_task(self, task_name: str) -> Optional[Path]:
        name = task_name.lower()
        is_ycb = "ycb" in name
        is_put = "put-on" in name or "put_on" in name or "put-in" in name or "put_in" in name
        is_move = "move" in name
        is_pick = "pick" in name or "grasp" in name

        base = Path(__file__).parent.parent / "ManiSkill2_real2sim" / "data"
        if is_ycb and is_put:
            return base / "ycb-dataset" / "info_ycb_put.json"
        if is_ycb and (is_pick or is_move):
            return base / "ycb-dataset" / "info_ycb.json"
        if is_put:
            return base / "custom" / "info_bridge_custom_v0.json"
        if is_pick or is_move:
            return base / "custom" / "info_pick_custom_v0.json"
        return None

    def _randomize_robot(self, options: Dict[str, Any]) -> None:
        robot_opts = options.get("robot_init_options")
        if not isinstance(robot_opts, dict):
            robot_opts = {}

        target_xy = self._get_target_xy(options)
        if target_xy is not None:
            dx = float(self.rng.uniform(*self.target_xy_offset_range))
            dy = float(self.rng.uniform(*self.target_xy_offset_range))
            init_xy = [float(target_xy[0]) + dx, float(target_xy[1]) + dy]
        else:
            init_xy = _random_xy(self.rng, self.robot_init_xy_range)

        if "google_robot" in (self.task_name or ""):
            init_xy = [
                float(np.clip(init_xy[0], 0.30, 0.40)),
                float(np.clip(init_xy[1], 0.0, 0.2)),
            ]

        yaw = float(self.rng.uniform(*self.robot_init_yaw_range))
        init_rot_quat = [
            float(np.cos(yaw / 2.0)),
            0.0,
            0.0,
            float(np.sin(yaw / 2.0)),
        ]

        robot_opts["init_xy"] = init_xy
        robot_opts["init_rot_quat"] = init_rot_quat
        options["robot_init_options"] = robot_opts

    @staticmethod
    def _is_xy(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) == 2

    def _get_target_xy(self, options: Dict[str, Any]) -> Optional[List[float]]:
        obj_opts = options.get("obj_init_options")
        if isinstance(obj_opts, dict):
            if self._is_xy(obj_opts.get("init_xy")):
                return [float(obj_opts["init_xy"][0]), float(obj_opts["init_xy"][1])]
            model_id = options.get("model_id")
            if model_id and isinstance(obj_opts.get(model_id), dict):
                xy = obj_opts[model_id].get("init_xy")
                if self._is_xy(xy):
                    return [float(xy[0]), float(xy[1])]
            for cfg in obj_opts.values():
                if isinstance(cfg, dict):
                    xy = cfg.get("init_xy")
                    if self._is_xy(xy):
                        return [float(xy[0]), float(xy[1])]
        return None

if __name__ == "__main__":
    env = simpler_env.make("google_robot_pick_customizable")
    variation = Variation(seed=123, camera_base=None,task_name="google_robot_pick_customizable")
    base_options = {
                    "model_id": "7up_can",
                    "obj_init_options": {
                        "init_xy": [
                        -0.2874544205649647,
                        0.24473519451947678
                        ],
                        "orientation": [
                        0.7071067811865476,
                        0.7071067811865475,
                        0.0,
                        0.0
                        ]
                    },
                    "max_episode_steps": 80,
                    "robot_init_options": {
                        "init_xy": [
                        -0.0416844647096154,
                        0.03392580027476842
                        ],
                        "init_rot_quat": [
                        1.0,
                        0.0,
                        0.0,
                        0.0
                        ]
                    }
                }
    def _allclose(a, b, tol=1e-6):
        if a is None or b is None:
            return a == b
        return np.allclose(np.array(a), np.array(b), atol=tol, rtol=0)

    def _check(label, expected, actual):
        ok = _allclose(expected, actual) if isinstance(expected, (list, tuple, np.ndarray)) else expected == actual
        status = "OK" if ok else "MISMATCH"
        print(f"[{status}] {label}: expected={expected} actual={actual}")
        return ok

    def validate_base_options(env, base_options, reset_info):
        env_u = env.unwrapped
        print("=== validate base_options ===")
        if "model_id" in base_options:
            _check("model_id", base_options["model_id"], getattr(env_u, "model_id", None))
        if "model_scale" in base_options:
            _check("model_scale", base_options["model_scale"], getattr(env_u, "model_scale", None))
        if "obj_init_options" in base_options:
            obj_init = getattr(env_u, "obj_init_options", {})
            exp_init_xy = base_options["obj_init_options"].get("init_xy")
            if exp_init_xy is not None:
                _check("obj_init_options.init_xy", exp_init_xy, obj_init.get("init_xy"))
            exp_orient = base_options["obj_init_options"].get("orientation")
            if exp_orient is not None:
                _check("obj_init_options.orientation->init_rot_quat", exp_orient, obj_init.get("init_rot_quat"))
        if "robot_init_options" in base_options:
            robot_init = getattr(env_u, "robot_init_options", {})
            exp_robot_xy = base_options["robot_init_options"].get("init_xy")
            if exp_robot_xy is not None:
                _check("robot_init_options.init_xy", exp_robot_xy, robot_init.get("init_xy"))
            exp_robot_rot = base_options["robot_init_options"].get("init_rot_quat")
            if exp_robot_rot is not None:
                _check("robot_init_options.init_rot_quat", exp_robot_rot, robot_init.get("init_rot_quat"))
        if "max_episode_steps" in base_options:
            max_steps = getattr(env, "_max_episode_steps", None)
            spec_steps = getattr(getattr(env, "spec", None), "max_episode_steps", None)
            expected = base_options["max_episode_steps"]
            actual = max_steps if max_steps is not None else spec_steps
            if actual is None:
                print("[IGNORED] max_episode_steps not exposed on env; option may be ignored.")
            else:
                _check("max_episode_steps", expected, actual)

    base_options2 = {"model_id": "sprite_can", "obj_init_options": {"init_xy": [-0.476108366133701, 0.3097114542828694], "orientation": [0.7829300091446386, 0.0, 0.0, -0.6221097980105892]}, "max_episode_steps": 80}
    result = variation.rephrase(task="grasp",options=base_options)
    obs, reset_info = env.reset(seed=123, options=result)
    validate_base_options(env, result, reset_info)
    print(json.dumps(result, indent=2, sort_keys=True))
