# GRPO 外部奖励插件：将模型输出解析为 JSON 后与数据集中的 solution 四字段比对。
# 用法：--external_plugins bash/grpo_cloth_reward_plugin.py --reward_funcs cloth_json_match
# 数据需含 solution 列（与 datasets/processed/cloth_debug/*.jsonl 一致）。
# 依赖仓库内 datasets/prompt-parse.py 的解析逻辑。

import importlib.util
import json
import os
from typing import List

from swift.rewards import ORM, orms

_bash_dir = os.path.dirname(os.path.abspath(__file__))
_prompt_parse_path = os.path.normpath(os.path.join(_bash_dir, '..', 'datasets', 'prompt-parse.py'))
_spec = importlib.util.spec_from_file_location('prompt_parse_cloth', _prompt_parse_path)
_prompt_parse = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_prompt_parse)
_parser = _prompt_parse.PromptParserClothFolding()


def _normalize_stage(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


class ClothJsonMatchORM(ORM):
    """Reward: 1.0 if parsed JSON matches solution (shape, orientation, completion, stage)."""

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        for completion, sol_str in zip(completions, solution):
            try:
                pred = _parser._parse_llm_stage_response(completion)
                gt = json.loads(sol_str)
            except Exception:
                rewards.append(0.0)
                continue
            if pred.get('shape') != gt.get('shape'):
                rewards.append(0.0)
                continue
            if pred.get('orientation') != gt.get('orientation'):
                rewards.append(0.0)
                continue
            if pred.get('completion') != gt.get('completion'):
                rewards.append(0.0)
                continue
            if _normalize_stage(pred.get('stage')) != _normalize_stage(gt.get('stage')):
                rewards.append(0.0)
                continue
            rewards.append(1.0)
        return rewards


orms['cloth_json_match'] = ClothJsonMatchORM
