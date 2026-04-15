"""Custom Dataset for TTT-Discover algorithm problems.

Wraps problems loaded from ``datasets/final_test/`` into a
``torch.utils.data.Dataset`` compatible with verl's data pipeline.

Each item returns the same dict structure as ``RLHFDataset.__getitem__``::

    {
        "input_ids":      (max_prompt_length,) long tensor,
        "attention_mask":  (max_prompt_length,) long tensor,
        "position_ids":   (max_prompt_length,) long tensor,
        "raw_prompt_ids": list[int],
        "extra_info":     {"problem_id": str},
    }

The prompt is built from the problem text and (optionally) the best code
found so far, then tokenized.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Default prompt template — instructs the model to produce a ``solve`` function.
_DEFAULT_PROMPT_TEMPLATE = """\
You are given an algorithm problem. Write a Python function `solve` that solves it.

## Problem

{problem_text}

## Start Code (do NOT modify, your solve function will be called by this code)

```python
{start_code}
```

{best_code_section}

## Your Task

Write the `solve` function in Python. Only output the function definition inside a ```python code block.
"""

_BEST_CODE_SECTION = """\
## Previous Best Solution (reward={best_reward:.2f})

```python
{best_code}
```

Improve upon this solution if possible.
"""


class TTTDiscoverDataset(Dataset):
    """Dataset that wraps algorithm problems for TTT-Discover training.

    Parameters
    ----------
    data_files : str or list[str]
        Unused — kept for API compatibility with ``create_rl_dataset``.
    tokenizer : PreTrainedTokenizer
        Tokenizer for encoding prompts.
    processor : optional
        Unused (no multimodal data).
    config : DictConfig
        Must contain ``ttt_project_root`` and ``ttt_problem_dir``.
        Optional: ``ttt_levels`` (comma-separated), ``max_prompt_length``.
    """

    def __init__(self, data_files, tokenizer, processor=None, config=None):
        self.tokenizer = tokenizer
        self.config = config

        self.max_prompt_length = int(config.get("max_prompt_length", 1024)) if config else 1024
        self.truncation = config.get("truncation", "error") if config else "error"

        # --- Read TTT-specific config ---
        ttt_project_root = config.get("ttt_project_root", "") if config else ""
        ttt_problem_dir = config.get("ttt_problem_dir", "") if config else ""
        ttt_levels = config.get("ttt_levels", None) if config else None

        if not ttt_project_root or not ttt_problem_dir:
            raise ValueError(
                "TTTDiscoverDataset requires data.ttt_project_root and "
                "data.ttt_problem_dir in the config."
            )

        # Make ttt_discover importable
        algo_test_root = os.path.dirname(ttt_project_root) if not os.path.isdir(
            os.path.join(ttt_project_root, "ttt_discover")
        ) else ttt_project_root
        if algo_test_root not in sys.path:
            sys.path.insert(0, algo_test_root)

        from ttt_discover.data.problem_loader import load_problems

        level_list: Optional[list[str]] = None
        if ttt_levels:
            level_list = [l.strip() for l in str(ttt_levels).split(",") if l.strip()]

        self.problems = load_problems(
            project_root=ttt_project_root,
            problem_dir=ttt_problem_dir,
            levels=level_list,
        )

        if not self.problems:
            raise RuntimeError(
                f"No problems loaded from {ttt_project_root}/{ttt_problem_dir} "
                f"(levels={level_list})"
            )

        logger.info("TTTDiscoverDataset: loaded %d problems", len(self.problems))

        # Build index mappings
        self._problem_ids: list[str] = [p.problem_id for p in self.problems]
        self.problem_id_to_idx: dict[str, int] = {
            pid: i for i, pid in enumerate(self._problem_ids)
        }

        # Per-problem best code cache (updated by PUCTSampler)
        self._best_codes: dict[str, str] = {}
        self._best_rewards: dict[str, float] = {}

        # Pre-tokenize all prompts
        self._cached_items: list[Optional[dict]] = [None] * len(self.problems)
        for idx in range(len(self.problems)):
            self._cached_items[idx] = self._build_item(idx)

    def _build_prompt(self, idx: int) -> str:
        """Build the text prompt for the problem at *idx*."""
        problem = self.problems[idx]
        pid = problem.problem_id

        best_code = self._best_codes.get(pid, "")
        best_reward = self._best_rewards.get(pid, 0.0)

        if best_code:
            best_section = _BEST_CODE_SECTION.format(
                best_code=best_code,
                best_reward=best_reward,
            )
        else:
            best_section = ""

        return _DEFAULT_PROMPT_TEMPLATE.format(
            problem_text=problem.problem_text,
            start_code=problem.start_code,
            best_code_section=best_section,
        )

    def _build_item(self, idx: int) -> dict:
        """Tokenize the prompt for problem *idx* and return a sample dict."""
        from verl.utils.model import compute_position_id_with_mask
        import verl.utils.torch_functional as verl_F

        prompt_text = self._build_prompt(idx)

        model_inputs = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False,
        )
        input_ids = model_inputs["input_ids"]            # (1, seq_len)
        attention_mask = model_inputs["attention_mask"]  # (1, seq_len)

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "error":
                # Silently truncate right to avoid crashing
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "extra_info": {"problem_id": self._problem_ids[idx]},
        }

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> dict:
        item = self._cached_items[idx]
        if item is None:
            item = self._build_item(idx)
            self._cached_items[idx] = item
        return item

    # ------------------------------------------------------------------
    # Called by PUCTSampler after each training step
    # ------------------------------------------------------------------

    def update_prompt(self, problem_id: str, best_code: str, best_reward: float) -> None:
        """Update the prompt for *problem_id* to include the best code so far.

        This invalidates the cached tokenized item so that the next
        ``__getitem__`` call rebuilds it with the new prompt.
        """
        self._best_codes[problem_id] = best_code
        self._best_rewards[problem_id] = best_reward

        idx = self.problem_id_to_idx.get(problem_id)
        if idx is not None:
            self._cached_items[idx] = self._build_item(idx)
            logger.debug(
                "Updated prompt for problem %s (reward=%.4f)",
                problem_id,
                best_reward,
            )
