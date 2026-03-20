"""
CHORD/GRPO 用的 reward 插件：根据模型生成的 JSON 与数据集中的 solution 比较，
对 shape / orientation / completion / stage 四字段全对给 1.0，否则 0.0。

用法：
  --external_plugins /path/to/grpo_reward_plugin.py --reward_funcs cloth_json_match
数据集需包含 solution 列（与 make_cloth_dataset.py 输出的格式一致）。
"""

import json
import os
import sys
from typing import List

# 解析逻辑与 prompt-parse 一致
_dir = os.path.dirname(os.path.abspath(__file__))
_spec = __import__("importlib.util").spec_from_file_location(
    "prompt_parse", os.path.join(_dir, "prompt-parse.py")
)
_prompt_parse = __import__("importlib.util").module_from_spec(_spec)
_spec.loader.exec_module(_prompt_parse)
_parser = _prompt_parse.PromptParserClothFolding()


def _parse_completion(text: str) -> dict:
    return _parser._parse_llm_stage_response(text)


def _normalize_stage(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 1


class ClothJsonMatchORM:
    """Reward: 1.0 if parsed JSON matches solution (shape, orientation, completion, stage)."""

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        for completion, sol_str in zip(completions, solution):
            try:
                pred = _parse_completion(completion)
                gt = json.loads(sol_str)
            except Exception:
                rewards.append(0.0)
                continue
            if pred.get("shape") != gt.get("shape"):
                rewards.append(0.0)
                continue
            if pred.get("orientation") != gt.get("orientation"):
                rewards.append(0.0)
                continue
            if pred.get("completion") != gt.get("completion"):
                rewards.append(0.0)
                continue
            if _normalize_stage(pred.get("stage")) != _normalize_stage(gt.get("stage")):
                rewards.append(0.0)
                continue
            rewards.append(1.0)
        return rewards


try:
    from swift.rewards import orms
    orms["cloth_json_match"] = ClothJsonMatchORM
except ImportError:
    pass
