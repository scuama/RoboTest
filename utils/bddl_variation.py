import copy
import os
import random
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


_OBSTACLE_POOL = [
    # 容器类
    "porcelain_mug",      # 白色瓷杯
    "red_coffee_mug",     # 红色咖啡杯
    "white_yellow_mug",   # 黄白色杯子
    "plate",              # 盘子
    "akita_black_bowl",   # 黑碗
    # 厨具类
    "moka_pot",           # 摩卡壶
    "wine_bottle",        # 葡萄酒瓶
    # 食品类
    "butter",             # 黄油
    "chocolate_pudding",  # 巧克力布丁
    "cream_cheese",       # 奶油芝士
    "tomato_sauce",       # 番茄酱
    "ketchup",            # 番茄酱
    "alphabet_soup",      # 字母汤
    "orange_juice",       # 橙汁
    "milk",               # 牛奶
    # 其他类
    "black_book",         # 黑色书籍
]

# 配置文件名到物品类型的映射
_NAME_TO_TYPE_MAP = {
    'butter': 'butter',
    'pudding': 'chocolate_pudding',
    'cheese': 'cream_cheese',
    'plate': 'plate',
    'bowl': 'akita_black_bowl',
    'tomato_sauce': 'tomato_sauce',
    'ketchup': 'ketchup',
    'alphabet_soup': 'alphabet_soup',
    'orange_juice': 'orange_juice',
    'milk': 'milk',
}

_DEFAULT_TABLE_RANGE = (-0.5, -0.5, 0.5, 0.5)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _expand_language(text: str, rng: random.Random) -> str:
    prefixes = [
        "Please",
        "Carefully",
        "Gently",
        "Slowly",
        "With a steady grip,",
    ]
    suffixes = [
        "in a smooth motion",
        "with a steady grip",
        "gently and steadily",
        "without hesitation",
    ]
    variants = []
    for prefix in prefixes:
        variants.append(f"{prefix} {text}")
    for suffix in suffixes:
        variants.append(f"{text} {suffix}")
    for prefix in prefixes:
        for suffix in suffixes:
            variants.append(f"{prefix} {text} {suffix}")
    base = _normalize_text(text)
    candidates = [v for v in variants if _normalize_text(v) != base]
    if not candidates:
        return text
    return rng.choice(candidates)


def _gpt_rephrase_instruction(instruction: str) -> Optional[str]:
    # Get API key from environment variable for security
    api_key = os.environ.get("OPENAI_API_KEY", None)
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    model =  "gpt-4.0"
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite robot task instructions while preserving meaning.Expand it.Don't just add adjectives. "
                        "Keep it concise, imperative, and do not add new objects or constraints."
                        "Don't just add adjectives. You should rephrase the entire sentence while preserving the original meaning."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Don't just add adjectives. You should rephrase the entire sentence while preserving the original meaning.Instruction: {instruction}\nRewritten instruction:",
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


def _format_range(values: Tuple[float, float, float, float]) -> str:
    return f"({values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f})"


def _find_block(text: str, token: str) -> Optional[Tuple[int, int, str]]:
    start = text.find(f"(:{token}")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "(":
            depth += 1
        elif text[idx] == ")":
            depth -= 1
            if depth == 0:
                end = idx + 1
                return start, end, text[start:end]
    return None


def _indent_for_block(block_text: str) -> str:
    for line in block_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("(") and not stripped.startswith("(:"):
            return line[: len(line) - len(stripped)]
    return "  "


@dataclass
class Region:
    name: str
    target: Optional[str]
    ranges: Optional[Tuple[float, float, float, float]]
    block_text: str


class BDDLDocument:
    def __init__(self, text: str):
        self.text = text

    def _get_block(self, token: str) -> Optional[Tuple[int, int, str]]:
        return _find_block(self.text, token)

    def _replace_block(self, start: int, end: int, new_block: str) -> None:
        self.text = self.text[:start] + new_block + self.text[end:]

    def get_language(self) -> Optional[str]:
        match = re.search(r"\(:language\s+([^\n\)]+)\)", self.text)
        if not match:
            return None
        return match.group(1).strip()

    def set_language(self, new_text: str) -> None:
        def repl(match):
            return f"(:language {new_text})"

        self.text = re.sub(r"\(:language\s+[^\n\)]+\)", repl, self.text, count=1)

    def get_objects(self) -> List[str]:
        block = self._get_block("objects")
        if not block:
            return []
        _, _, block_text = block
        names: List[str] = []
        for line in block_text.splitlines():
            if "-" not in line:
                continue
            left = line.split("-")[0].strip()
            if not left:
                continue
            names.extend(left.split())
        return names

    def get_fixtures(self) -> List[str]:
        block = self._get_block("fixtures")
        if not block:
            return []
        _, _, block_text = block
        names: List[str] = []
        for line in block_text.splitlines():
            if "-" not in line:
                continue
            left = line.split("-")[0].strip()
            if not left:
                continue
            names.extend(left.split())
        return names

    def get_object_types(self) -> List[str]:
        block = self._get_block("objects")
        if not block:
            return []
        _, _, block_text = block
        types: List[str] = []
        for line in block_text.splitlines():
            if "-" not in line:
                continue
            right = line.split("-")[-1].strip()
            if right:
                types.append(right)
        return types

    def add_object(self, name: str, obj_type: str) -> None:
        block = self._get_block("objects")
        if not block:
            return
        start, end, block_text = block
        lines = block_text.splitlines()
        indent = "  "
        for line in lines:
            if line.strip().startswith("(:objects"):
                indent = line[: len(line) - len(line.lstrip())] + "  "
                break

        updated = False
        new_lines: List[str] = []
        type_pattern = re.compile(rf"^(?P<indent>\s*)(?P<names>.+?)\s*-\s*{re.escape(obj_type)}\s*$")
        for line in lines:
            match = type_pattern.match(line)
            if match and not updated:
                names = match.group("names").split()
                if name not in names:
                    names.append(name)
                new_line = f"{match.group('indent')}{' '.join(names)} - {obj_type}"
                new_lines.append(new_line)
                updated = True
            else:
                new_lines.append(line)

        if not updated:
            insertion = f"{indent}{name} - {obj_type}"
            new_lines = new_lines[:-1] + [insertion] + [new_lines[-1]]
        new_block = "\n".join(new_lines)
        self._replace_block(start, end, new_block)

    def get_regions(self) -> List[Region]:
        block = self._get_block("regions")
        if not block:
            return []
        _, _, block_text = block
        regions: List[Region] = []
        lines = block_text.splitlines()
        in_region = False
        region_lines: List[str] = []
        depth = 0
        for line in lines[1:-1]:
            if not in_region:
                stripped = line.lstrip()
                if stripped.startswith("(") and not stripped.startswith("(:"):
                    in_region = True
                    region_lines = [line]
                    depth = stripped.count("(") - stripped.count(")")
                    if depth == 0:
                        in_region = False
                        region_text = "\n".join(region_lines)
                        regions.append(self._parse_region(region_text))
            else:
                region_lines.append(line)
                depth += line.count("(") - line.count(")")
                if depth == 0:
                    in_region = False
                    region_text = "\n".join(region_lines)
                    regions.append(self._parse_region(region_text))
        return regions

    def get_region_ref_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for region in self.get_regions():
            if region.target:
                mapping[f"{region.target}_{region.name}"] = region.name
        return mapping

    def get_fixture_init_regions(self) -> List[str]:
        fixtures = set(self.get_fixtures())
        region_ref_map = self.get_region_ref_map()
        fixed_regions: List[str] = []
        for match in re.finditer(r"\(On\s+([^\s\)]+)\s+([^\s\)]+)\)", self.text):
            obj_name = match.group(1)
            region_ref = match.group(2)
            if obj_name in fixtures:
                region_name = region_ref_map.get(region_ref)
                if region_name:
                    fixed_regions.append(region_name)
        return fixed_regions

    def get_init_region_refs(self) -> List[str]:
        region_refs: List[str] = []
        for match in re.finditer(r"\(On\s+([^\s\)]+)\s+([^\s\)]+)\)", self.text):
            region_refs.append(match.group(2))
        return region_refs

    def _parse_region(self, region_text: str) -> Region:
        first_line = region_text.strip().splitlines()[0]
        name = first_line.strip().lstrip("(").split()[0]
        target_match = re.search(r"\(:target\s+([^\s\)]+)\)", region_text)
        target = target_match.group(1) if target_match else None
        range_match = re.search(
            r"\(:ranges\s+\(\s*\(\s*(.*?)\s*\)\s*\)\s*\)",
            region_text,
            flags=re.DOTALL,
        )
        ranges = None
        if range_match:
            parts = range_match.group(1).split()
            if len(parts) >= 4:
                try:
                    ranges = tuple(float(p) for p in parts[:4])
                except ValueError:
                    ranges = None
        return Region(name=name, target=target, ranges=ranges, block_text=region_text)

    def update_region_ranges(self, region_name: str, new_ranges: Tuple[float, float, float, float]) -> None:
        block = self._get_block("regions")
        if not block:
            return
        start, end, block_text = block
        region_blocks = self.get_regions()
        updated_blocks = []
        for region in region_blocks:
            region_text = region.block_text
            if region.name == region_name and region.ranges is not None:
                formatted = _format_range(new_ranges)
                region_text = re.sub(
                    r"\(:ranges\s+\(\s*\(\s*(.*?)\s*\)\s*\)\s*\)",
                    f"(:ranges (\n              {formatted}\n            )\n          )",
                    region_text,
                    count=1,
                    flags=re.DOTALL,
                )
            updated_blocks.append(region_text)
        indent = _indent_for_block(block_text)
        new_lines = [block_text.splitlines()[0]]
        for region_text in updated_blocks:
            for line in region_text.splitlines():
                new_lines.append(line)
        new_lines.append(block_text.splitlines()[-1])
        new_block = "\n".join(new_lines)
        self._replace_block(start, end, new_block)

    def add_region_block(self, region_text: str) -> None:
        block = self._get_block("regions")
        if not block:
            return
        start, end, block_text = block
        lines = block_text.splitlines()
        new_lines = lines[:-1] + [region_text] + [lines[-1]]
        new_block = "\n".join(new_lines)
        self._replace_block(start, end, new_block)

    def add_init_condition(self, condition: str) -> None:
        block = self._get_block("init")
        if not block:
            return
        start, end, block_text = block
        lines = block_text.splitlines()
        indent = "  "
        for line in lines:
            if line.strip().startswith("(:init"):
                indent = line[: len(line) - len(line.lstrip())] + "  "
                break
        insertion = f"{indent}{condition}"
        new_lines = lines[:-1] + [insertion] + [lines[-1]]
        new_block = "\n".join(new_lines)
        self._replace_block(start, end, new_block)

    def add_goal_condition(self, condition: str) -> None:
        block = self._get_block("goal")
        if not block:
            return
        start, end, block_text = block
        if "(And" in block_text:
            new_block = re.sub(
                r"\(And\s*(.*)\)",
                lambda m: f"(And {m.group(1).strip()} {condition})",
                block_text,
                count=1,
                flags=re.DOTALL,
            )
        else:
            inner = block_text.strip()[len("(:goal"):].strip()
            if inner.endswith(")"):
                inner = inner[:-1].strip()
            new_block = block_text
            new_block = re.sub(
                r"\(:goal\s*\(([^)]+)\)\s*\)",
                f"(:goal (And (\\1) {condition}))",
                block_text,
                count=1,
                flags=re.DOTALL,
            )
        self._replace_block(start, end, new_block)

    def clone(self) -> "BDDLDocument":
        return BDDLDocument(copy.deepcopy(self.text))


class BDDLVariation:
    def __init__(
        self,
        seed: Optional[int] = None,
        skip_regions: Optional[List[str]] = None,
        debug: bool = False,
    ):
        self.rng = random.Random(seed)
        self.skip_regions = set(skip_regions or ["flat_stove_init_region"])
        self.debug = debug

    def generate(self, doc: BDDLDocument) -> BDDLDocument:
        return self.jitter_regions(doc)

    def expand(self, doc: BDDLDocument) -> BDDLDocument:
        mutated = doc.clone()
        count = self.rng.randint(1, 3)
        for _ in range(count):
            mutated = self.add_obstacle(mutated, place_near_target=True, enforce_clear=True)
        return mutated

    def expand_random(self, doc: BDDLDocument) -> BDDLDocument:
        mutated = doc.clone()
        count = self.rng.randint(3, 6)
        if self.debug:
            print(f"[expand_random] adding {count} obstacles")
        added = 0
        attempts = 0
        max_attempts = 10
        while added < count and attempts < max_attempts:
            attempts += 1
            before = mutated.text
            mutated = self.add_obstacle(
                mutated,
                place_near_target=False,
                enforce_clear=False,
            )
            if mutated.text != before:
                added += 1
        if self.debug:
            print(f"[expand_random] added={added} attempts={attempts}")
        return mutated

    def rephrase(self, doc: BDDLDocument) -> BDDLDocument:
        return self.rephrase_language(doc)

    def jitter_regions(self, doc: BDDLDocument, max_shift: float = 0.05) -> BDDLDocument:
        mutated = doc.clone()
        fixed_regions = set(mutated.get_fixture_init_regions())
        regions = mutated.get_regions()
        for region in regions:
            if region.name in self.skip_regions or region.name in fixed_regions:
                continue
            if region.ranges is None:
                continue
            x1, y1, x2, y2 = region.ranges
            dx = self.rng.uniform(-max_shift, max_shift)
            dy = self.rng.uniform(-max_shift, max_shift)
            new_ranges = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            mutated.update_region_ranges(region.name, new_ranges)
        return mutated

    def rephrase_language(self, doc: BDDLDocument) -> BDDLDocument:
        mutated = doc.clone()
        language = mutated.get_language()
        if not language:
            return mutated
        gpt_text = _gpt_rephrase_instruction(language)
        if gpt_text and _normalize_text(gpt_text) != _normalize_text(language):
            mutated.set_language(gpt_text)
        else:
            mutated.set_language(_expand_language(language, self.rng))
        return mutated

    def add_obstacle(
        self,
        doc: BDDLDocument,
        place_near_target: bool = True,
        enforce_clear: bool = True,
        direction_hint: Optional[int] = None,
        exclude_types: Optional[List[str]] = None,
    ) -> BDDLDocument:
        mutated = doc.clone()
        regions = mutated.get_regions()
        if self.debug:
            print(f"[add_obstacle] regions={len(regions)} place_near_target={place_near_target} enforce_clear={enforce_clear}")
        region_map = {}
        for region in regions:
            if region.target:
                region_map[f"{region.target}_{region.name}"] = region

        objects = mutated.get_objects()
        
        # 过滤掉场景中已有的物品类型
        available_pool = _OBSTACLE_POOL
        if exclude_types:
            available_pool = [obj for obj in _OBSTACLE_POOL if obj not in exclude_types]
            if self.debug:
                print(f"[add_obstacle] exclude_types={exclude_types}, available_pool size={len(available_pool)}")
        
        if not available_pool:
            if self.debug:
                print("[add_obstacle] no available obstacle types after filtering")
            return mutated
        
        # 从可用池中选择障碍物类型
        obstacle_type = self.rng.choice(available_pool)
        obstacle_name = f"{obstacle_type}_obstacle_1"
        counter = 1
        while obstacle_name in objects:
            counter += 1
            obstacle_name = f"{obstacle_type}_obstacle_{counter}"

        init_match = re.search(r"\(On\s+([^\s\)]+)\s+([^\s\)]+)\)", doc.text)
        target_fixture = None
        target_region = None
        if init_match and place_near_target:
            target_region = init_match.group(2)
            region = region_map.get(target_region)
            if region and region.target:
                target_fixture = region.target

        if not place_near_target:
            base_range = _DEFAULT_TABLE_RANGE
            target_fixture = target_fixture or "living_room_table"
            if self.debug:
                print(f"[add_obstacle] base_range={base_range} target_fixture={target_fixture}")
        else:
            base_range = None

        if place_near_target and target_region and target_region in region_map and region_map[target_region].ranges:
            base_range = region_map[target_region].ranges
        if base_range is None:
            base_range = (-0.1, -0.1, 0.1, 0.1)

        def _boxes_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
            return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

        occupied: List[Tuple[float, float, float, float]] = []
        region_ref_map = mutated.get_region_ref_map()
        for ref in mutated.get_init_region_refs():
            region_name = region_ref_map.get(ref)
            if not region_name:
                continue
            for region in regions:
                if region.name == region_name and region.ranges is not None:
                    occupied.append(region.ranges)
                    break
        if self.debug:
            print(f"[add_obstacle] occupied_regions={len(occupied)}")

        if place_near_target:
            x1, y1, x2, y2 = base_range
            width = x2 - x1
            height = y2 - y1
            pad = 0.1  # 安全距离设置为10cm
            size = max(0.015, min(0.025, min(width, height) * 0.3))  # 增加干扰物尺寸

            if width <= 0 or height <= 0:
                return mutated

            cx_target = (x1 + x2) / 2.0
            cy_target = (y1 + y2) / 2.0
            offset_x = width / 2.0 + size + pad
            offset_y = height / 2.0 + size + pad

            existing = re.findall(r"_obstacle_\\d+_init_region", mutated.text)
            idx = len(existing)
            
            # 如果提供了direction_hint，使用它来指导方向选择
            # 这样可以确保多个干扰物分散在四个方位
            if direction_hint is not None:
                start_direction = direction_hint % 4
            else:
                start_direction = idx % 4
            
            ring = idx // 4
            extra = ring * (size * 2 + pad)

            init_range = None
            clear_range = None
            
            # 尝试8个方向（优先使用direction_hint指定的方向）
            for attempt in range(8):
                direction = (start_direction + attempt) % 4
                if direction == 0:  # 右侧
                    cx = cx_target + offset_x + extra
                    cy = cy_target
                elif direction == 1:  # 左侧
                    cx = cx_target - offset_x - extra
                    cy = cy_target
                elif direction == 2:  # 上方
                    cx = cx_target
                    cy = cy_target + offset_y + extra
                else:  # 下方
                    cx = cx_target
                    cy = cy_target - offset_y - extra
                candidate = (cx - size, cy - size, cx + size, cy + size)
                if any(_boxes_overlap(candidate, box) for box in occupied):
                    continue
                init_range = candidate
                clear_range = (cx + 0.2, cy + 0.2, cx + 0.23, cy + 0.23)
                break

            if init_range is None:
                if self.debug:
                    print("[add_obstacle] no free near-target region found")
                return mutated
        else:
            if target_fixture is None:
                target_fixture = region.target if regions else "main_table"
            x1, y1, x2, y2 = base_range
            dx = self.rng.uniform(0.015, 0.03)
            dy = self.rng.uniform(0.015, 0.03)
            init_range = None
            clear_range = None
            # 增加重试次数到20次，以提高成功率
            for _ in range(20):
                cx = self.rng.uniform(x1, x2)
                cy = self.rng.uniform(y1, y2)
                candidate = (cx - dx, cy - dy, cx + dx, cy + dy)
                if any(_boxes_overlap(candidate, box) for box in occupied):
                    continue
                init_range = candidate
                clear_range = (cx + 0.1, cy + 0.1, cx + 0.13, cy + 0.13)
                break

            if init_range is None:
                if self.debug:
                    print("[add_obstacle] no free random region found")
                return mutated

        if not target_fixture:
            target_fixture = "main_table"

        init_region_name = f"{obstacle_name}_init_region"
        clear_region_name = f"{obstacle_name}_clear_region"
        suffix = 1
        if enforce_clear:
            while init_region_name in mutated.text or clear_region_name in mutated.text:
                suffix += 1
                init_region_name = f"{obstacle_name}_init_region_{suffix}"
                clear_region_name = f"{obstacle_name}_clear_region_{suffix}"
        else:
            while init_region_name in mutated.text:
                suffix += 1
                init_region_name = f"{obstacle_name}_init_region_{suffix}"

        region_indent = "      "
        init_region_block = (
            f"{region_indent}({init_region_name}\n"
            f"{region_indent}    (:target {target_fixture})\n"
            f"{region_indent}    (:ranges (\n"
            f"{region_indent}        {_format_range(init_range)}\n"
            f"{region_indent}      )\n"
            f"{region_indent}    )\n"
            f"{region_indent})"
        )
        mutated.add_region_block(init_region_block)
        if enforce_clear:
            clear_region_block = (
                f"{region_indent}({clear_region_name}\n"
                f"{region_indent}    (:target {target_fixture})\n"
                f"{region_indent}    (:ranges (\n"
                f"{region_indent}        {_format_range(clear_range)}\n"
                f"{region_indent}      )\n"
                f"{region_indent}    )\n"
                f"{region_indent})"
            )
            mutated.add_region_block(clear_region_block)

        mutated.add_init_condition(
            f"(On {obstacle_name} {target_fixture}_{init_region_name})"
        )

        mutated.add_object(obstacle_name, obstacle_type)
        if enforce_clear:
            mutated.add_goal_condition(
                f"(On {obstacle_name} {target_fixture}_{clear_region_name})"
            )

            language = mutated.get_language()
            if language:
                prefix = "Move the"
                if not place_near_target:
                    prefix = "Clear the"
                new_lang = f"{prefix} {obstacle_type.replace('_', ' ')} away first, then {language}"
                mutated.set_language(new_lang)

        return mutated


def _diff_text(a: str, b: str) -> str:
    import difflib
    return "\n".join(
        difflib.unified_diff(
            a.splitlines(),
            b.splitlines(),
            fromfile="original",
            tofile="mutated",
            lineterm="",
        )
    )


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="BDDL variation debug runner.")
    parser.add_argument("--bddl", type=str, required=False, help="Path to a BDDL file.",default="./custom_bddl_files/pair/butter_pudding_stacked_baseline.bddl")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["expand_random", "expand", "rephrase"],
        default="expand_random",
    )
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--show-diff", action="store_true", default=True)
    args = parser.parse_args()

    doc = BDDLDocument(Path(args.bddl).read_text())
    seed = args.seed
    if seed is None:
        seed = time.time_ns() % (2**32)
        print(f"[seed] auto_seed={seed}")
    variation = BDDLVariation(seed=seed, debug=args.debug)

    if args.mode == "expand_random":
        mutated = variation.expand_random(doc)
    elif args.mode == "expand":
        mutated = variation.expand(doc)
    else:
        mutated = variation.rephrase(doc)

    if args.show_diff:
        print(_diff_text(doc.text, mutated.text))
    else:
        print(mutated.text)


if __name__ == "__main__":
    main()

    def mutate_goal_region(self, doc: BDDLDocument) -> BDDLDocument:
        mutated = doc.clone()
        regions = mutated.get_regions()
        region_map = {}
        for region in regions:
            if region.target:
                region_map[f"{region.target}_{region.name}"] = region

        goal_match = re.search(r"\(On\s+([^\s\)]+)\s+([^\s\)]+)\)", doc.text)
        if not goal_match:
            return mutated
        goal_region = goal_match.group(2)
        region = region_map.get(goal_region)
        if not region or region.ranges is None or not region.target:
            return mutated

        x1, y1, x2, y2 = region.ranges
        dx = self.rng.uniform(0.05, 0.1)
        dy = self.rng.uniform(0.05, 0.1)
        new_ranges = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        new_region_name = f"{region.name}_alt"
        region_indent = "      "
        new_region_block = (
            f"{region_indent}({new_region_name}\n"
            f"{region_indent}    (:target {region.target})\n"
            f"{region_indent}    (:ranges (\n"
            f"{region_indent}        {_format_range(new_ranges)}\n"
            f"{region_indent}      )\n"
            f"{region_indent}    )\n"
            f"{region_indent})"
        )
        mutated.add_region_block(new_region_block)
        new_goal_region = f"{region.target}_{new_region_name}"
        mutated.text = re.sub(
            re.escape(goal_region),
            new_goal_region,
            mutated.text,
            count=1,
        )
        return mutated

    def mutate(
        self,
        doc: BDDLDocument,
        strategy: str,
        other_doc: Optional[BDDLDocument] = None,
    ) -> BDDLDocument:
        if strategy == "generate":
            return self.generate(doc)
        if strategy == "expand":
            return self.expand(doc)
        if strategy == "expand_random":
            return self.expand_random(doc)
        if strategy == "rephrase":
            return self.rephrase(doc)
        return self.generate(doc)


    
