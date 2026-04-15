"""PUCT-based curriculum sampler for TTT-Discover.

Extends verl's ``AbstractCurriculumSampler`` so that after every training
step the trainer automatically calls ``sampler.update(batch=batch)``
(see ``ray_trainer.py`` L1466-1467).

The sampler:
- Uses ``PUCTSelector`` to choose which problem to train on next.
- Yields the chosen problem index ``batch_size`` times (all samples in a
  batch target the same problem — this is central to TTT-Discover).
- On ``update()``, extracts rewards from the batch, updates the PUCT
  Archive, and refreshes the dataset prompt with the new best code.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Sized
from typing import Iterator, Optional

from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler

logger = logging.getLogger(__name__)


class PUCTSampler(AbstractCurriculumSampler):
    """Curriculum sampler driven by PUCT tree search.

    Parameters
    ----------
    data_source : TTTDiscoverDataset
        The dataset (must expose ``.problem_id_to_idx``, ``.problems``,
        ``.update_prompt()``).
    data_config : DictConfig
        Hydra config; reads ``puct_c`` (default 1.4) and
        ``ttt_project_root`` from here.
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        # Make ttt_discover importable
        ttt_project_root = data_config.get("ttt_project_root", "")
        algo_test_root = os.path.dirname(ttt_project_root) if ttt_project_root and not os.path.isdir(
            os.path.join(ttt_project_root, "ttt_discover")
        ) else ttt_project_root
        if algo_test_root and algo_test_root not in sys.path:
            sys.path.insert(0, algo_test_root)

        from ttt_discover.puct.archive import Archive
        from ttt_discover.puct.puct_selector import PUCTSelector

        self._dataset = data_source
        self._data_config = data_config

        # Read batch size from config: train_batch_size * rollout.n gives the
        # total number of samples per step, but the DataLoader only yields
        # train_batch_size items (rollout.n is handled by vLLM's n parameter).
        self._batch_size = int(data_config.get("train_batch_size", 1))

        # Initialise PUCT components
        puct_c = float(data_config.get("puct_c", 1.4))
        self.selector = PUCTSelector(c=puct_c)
        self.archive = Archive(max_size=len(data_source))

        # Register every problem in the archive
        for problem in data_source.problems:
            self.archive.add_state(problem.problem_id)

        logger.info(
            "PUCTSampler: %d problems, batch_size=%d, c=%.2f",
            len(data_source),
            self._batch_size,
            puct_c,
        )

        # Keep track of the tokenizer for decoding responses in update()
        self._tokenizer = getattr(data_source, "tokenizer", None)

    # ------------------------------------------------------------------
    # Sampler interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        # Return a large number so the DataLoader doesn't think we're done.
        # In practice the trainer controls the epoch length.
        return len(self._dataset) * 1000

    def __iter__(self) -> Iterator[int]:
        """Yield dataset indices for the DataLoader.

        Each "batch" consists of ``_batch_size`` copies of the same index
        (all samples target the same problem).  The DataLoader calls this
        lazily so we can select a fresh problem for every batch.
        """
        while True:
            selected_pid = self.selector.select_state(self.archive)
            idx = self._dataset.problem_id_to_idx[selected_pid]
            for _ in range(self._batch_size):
                yield idx

    # ------------------------------------------------------------------
    # Curriculum update — called by ray_trainer after each training step
    # ------------------------------------------------------------------

    def update(self, batch: DataProto) -> None:
        """Process the completed batch and update the PUCT Archive.

        Parameters
        ----------
        batch : DataProto
            The full batch that just finished training.  Contains:
            - ``batch["token_level_rewards"]`` — (bs, resp_len) rewards
            - ``batch["responses"]`` — (bs, resp_len) token IDs
            - ``batch["response_mask"]`` — (bs, resp_len)
            - ``non_tensor_batch["extra_info"]`` — list of dicts with problem_id
        """
        try:
            self._update_impl(batch)
        except Exception as exc:
            logger.error("PUCTSampler.update failed: %s", exc, exc_info=True)

    def _update_impl(self, batch: DataProto) -> None:
        import torch

        # Make ttt_discover importable
        from ttt_discover.utils.code_extraction import extract_solve_function

        # Extract per-sample scalar rewards
        response_mask = batch.batch.get("response_mask")
        token_rewards = batch.batch.get("token_level_rewards")

        if token_rewards is None:
            # Fallback: use "acc" field if present (set by BatchRewardManager)
            if "acc" in batch.batch:
                rewards = batch.batch["acc"]  # (bs,)
            else:
                logger.warning("No rewards found in batch — skipping PUCT update")
                return
        else:
            rewards = (token_rewards * response_mask).sum(dim=-1)  # (bs,)

        # Get extra_info for problem_id
        extra_infos = batch.non_tensor_batch.get("extra_info", [None] * len(rewards))

        # Group by problem_id and find best sample per problem
        problem_results: dict[str, dict] = {}  # pid -> {"best_reward", "best_idx"}

        for i in range(len(rewards)):
            extra = extra_infos[i] if extra_infos[i] is not None else {}
            if isinstance(extra, dict):
                pid = extra.get("problem_id", "")
            else:
                pid = ""

            if not pid:
                continue

            r = float(rewards[i].item()) if isinstance(rewards[i], torch.Tensor) else float(rewards[i])

            if pid not in problem_results or r > problem_results[pid]["best_reward"]:
                problem_results[pid] = {"best_reward": r, "best_idx": i}

        if not problem_results:
            logger.warning("No valid problem IDs in batch — skipping PUCT update")
            return

        # Decode best response and extract solve code for each problem
        for pid, info in problem_results.items():
            best_idx = info["best_idx"]
            best_reward = info["best_reward"]

            # Decode the response to extract code
            best_code = ""
            if self._tokenizer is not None:
                try:
                    response_ids = batch.batch["responses"][best_idx]
                    if response_mask is not None:
                        valid_len = int(response_mask[best_idx].sum().item())
                        response_ids = response_ids[:valid_len]
                    response_str = self._tokenizer.decode(
                        response_ids, skip_special_tokens=True
                    )
                    best_code = extract_solve_function(response_str)
                except Exception as exc:
                    logger.debug(
                        "Failed to decode/extract code for problem %s: %s",
                        pid, exc,
                    )

            # Update PUCT Archive
            if best_code or best_reward > 0:
                try:
                    q_mode = str(self._data_config.get("puct_q_mode", "max"))
                    self.archive.update_state(
                        problem_id=pid,
                        reward=best_reward,
                        code=best_code if best_code else "",
                        q_mode=q_mode,
                    )
                except KeyError:
                    logger.warning("Problem %s not in archive — skipping", pid)
                    continue

                # Update dataset prompt with new best code
                state = self.archive.get_state(pid)
                if state is not None and state.best_code:
                    self._dataset.update_prompt(
                        problem_id=pid,
                        best_code=state.best_code,
                        best_reward=state.best_reward,
                    )

        logger.info(
            "PUCT update: %d problems, total_visits=%d, "
            "best_global=%.4f",
            len(problem_results),
            self.archive.total_visits,
            max(s.best_reward for s in self.archive.get_all_states()),
        )
