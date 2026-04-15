"""Entropic policy gradient loss.

Implements the loss from "Learning to Discover at Test Time":

    loss = -E[ A_β(τ) · log π_θ(τ|s) ]

where A_β already includes the KL penalty term (folded into the advantage
by the caller in trainer.py):

    A(a;s) = w_β(a) - 1 - λ · log(π_θ(a|s) / π_init(a|s))

No PPO clipping or importance sampling ratio is used — this is on-policy.
"""

import torch
import torch.nn.functional as F


def compute_policy_loss(
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute entropic policy gradient loss.

    The KL penalty against π_init is already folded into the per-sample
    advantages by the caller (see trainer.py), matching the paper's
    formulation where A(a;s) = w_β(a) - 1 - λ·KL_per_sample.

    Args:
        new_log_probs: Tensor of shape (batch, seq_len) with log probs from
            the current policy π_θ (with gradients).
        advantages: Tensor of shape (batch,) with per-sample advantages
            that already include the KL penalty term.
        response_mask: Tensor of shape (batch, seq_len) with 1.0 for
            response tokens and 0.0 for padding / prompt tokens. If None,
            all tokens are treated as response tokens.

    Returns:
        A tuple of:
          - loss: Scalar tensor (mean masked loss), suitable for .backward().
          - metrics: Dict with diagnostic scalars.
    """
    # Expand per-sample advantages to per-token: (batch,) -> (batch, seq_len)
    seq_len = new_log_probs.shape[1]
    advantages_expanded = advantages.unsqueeze(1).expand(-1, seq_len)

    # On-policy policy gradient: -A · log π_θ
    per_token_loss = -advantages_expanded * new_log_probs

    # Apply response mask
    if response_mask is None:
        response_mask = torch.ones_like(per_token_loss)

    masked_loss = per_token_loss * response_mask

    # Average over valid (masked) tokens
    num_valid = response_mask.sum().clamp(min=1.0)
    loss = masked_loss.sum() / num_valid

    metrics = {
        "clip_fraction": 0.0,
        "approx_kl": 0.0,
        "mean_ratio": 1.0,
        "kl_ref": 0.0,
    }

    return loss, metrics


def compute_log_probs_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_start_indices: torch.Tensor,
) -> torch.Tensor:
    """Extract per-token log probabilities for the response portion.

    For each position t, the log-prob is log P(input_ids[t] | context up to t),
    taken from logits[t-1] (the autoregressive shift). Positions before the
    response start index are zeroed out.

    Args:
        logits: Tensor of shape (batch, seq_len, vocab_size) — raw model
            output logits.
        input_ids: Tensor of shape (batch, seq_len) — token ids that were
            fed to the model (prompt + response + padding).
        response_start_indices: Tensor of shape (batch,) — the index in
            each sequence where the response begins. Tokens before this
            position will have log_prob = 0.

    Returns:
        Tensor of shape (batch, seq_len) with per-token log probabilities.
        Prompt positions (before response_start_indices) are zeroed out.
    """
    batch_size, seq_len, vocab_size = logits.shape

    log_probs_all = F.log_softmax(logits, dim=-1)

    shifted_ids = input_ids[:, 1:]
    shifted_log_probs = log_probs_all[:, :-1, :]

    gathered = shifted_log_probs.gather(
        dim=-1, index=shifted_ids.unsqueeze(-1)
    ).squeeze(-1)

    per_token_log_probs = torch.zeros(
        batch_size, seq_len, device=logits.device, dtype=logits.dtype
    )
    per_token_log_probs[:, 1:] = gathered

    positions = torch.arange(seq_len, device=logits.device).unsqueeze(0)
    response_mask = positions >= response_start_indices.unsqueeze(1)

    per_token_log_probs = per_token_log_probs * response_mask.to(per_token_log_probs.dtype)

    return per_token_log_probs
