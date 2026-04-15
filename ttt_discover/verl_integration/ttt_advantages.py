"""Entropic advantage estimator registered into verl's core_algos registry.

Usage:
    Simply ``import ttt_discover.verl_integration.ttt_advantages`` early in the
    process (e.g. in ``main_ttt.py``) so that the ``@register_adv_est("entropic")``
    decorator fires before ``RayPPOTrainer`` resolves the estimator name.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est

# Re-use the existing TTT-Discover implementations
from ttt_discover.entropic_rl.advantages import (
    compute_adaptive_beta,
    compute_loo_entropic_advantages,
)

logger = logging.getLogger(__name__)


@register_adv_est("entropic")
def compute_entropic_outcome_advantage(
    token_level_rewards: torch.Tensor,  # (bs, response_length)
    response_mask: torch.Tensor,        # (bs, response_length)
    index: Optional[np.ndarray] = None, # (bs,) — uid grouping
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token advantages using the entropic utility objective.

    This mirrors GRPO's interface but replaces the mean/std normalisation with:
      1. Adaptive beta selection via binary search on KL(q_beta || uniform).
      2. Leave-one-out (LOO) entropic advantages.

    Groups are identified by *index* (uid); samples within a group share the
    same prompt/problem and their rewards are compared against each other.

    Parameters
    ----------
    token_level_rewards : torch.Tensor
        Shape ``(bs, response_length)`` with the scalar reward placed at
        the last valid token position (verl convention).
    response_mask : torch.Tensor
        Shape ``(bs, response_length)``, 1 for valid response tokens.
    index : np.ndarray, optional
        Shape ``(bs,)`` grouping key (uid).  If absent, all samples are
        treated as one group.
    config : AlgoConfig, optional
        Algorithm-level config.  Reads ``entropic_gamma`` (default ln2)
        from here when present.

    Returns
    -------
    advantages : torch.Tensor
        Shape ``(bs, response_length)``.
    returns : torch.Tensor
        Same as ``token_level_rewards`` (no value baseline).
    """
    # --- extract per-sample scalar scores (same as GRPO) ---
    scores = (token_level_rewards * response_mask).sum(dim=-1)  # (bs,)

    # --- read gamma from config if available ---
    gamma = math.log(2)
    if config is not None:
        gamma = float(getattr(config, "entropic_gamma", gamma))

    # --- group by uid ---
    if index is None:
        index = np.zeros(scores.shape[0], dtype=np.int64)

    unique_ids = np.unique(index)
    advantages_flat = torch.zeros_like(scores)  # (bs,)

    for uid in unique_ids:
        mask = index == uid
        idx = np.where(mask)[0]
        group_scores = scores[idx]  # (group_size,)

        if group_scores.numel() <= 1:
            # Single sample — advantage is zero
            advantages_flat[idx] = 0.0
            continue

        # 1) Adaptive beta via binary search
        beta = compute_adaptive_beta(group_scores, gamma=gamma)

        # 2) LOO entropic advantages
        adv = compute_loo_entropic_advantages(group_scores, beta)

        advantages_flat[idx] = adv

    # --- expand to token dimension ---
    advantages = advantages_flat.unsqueeze(-1) * response_mask  # (bs, response_length)
    returns = token_level_rewards.clone()

    logger.debug(
        "entropic advantages: mean=%.4f std=%.4f, %d groups",
        advantages_flat.mean().item(),
        advantages_flat.std().item() if advantages_flat.numel() > 1 else 0.0,
        len(unique_ids),
    )

    return advantages, returns
