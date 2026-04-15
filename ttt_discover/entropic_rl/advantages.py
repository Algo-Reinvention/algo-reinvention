"""Entropic Utility Objective (Appendix A.1).

Implements adaptive temperature selection via KL-constrained binary search
and leave-one-out entropic advantage computation.
"""

import math
import torch


def compute_kl_for_beta(rewards: torch.Tensor, beta: float) -> float:
    """Compute KL(q_beta || uniform) for a given beta.

    q_beta(n) = exp(beta * r_n) / sum_m exp(beta * r_m)

    KL(q_beta || u) = sum_n q_beta(n) * log(N * q_beta(n))

    Uses log-sum-exp trick for numerical stability.

    Args:
        rewards: Tensor of shape (N,) containing reward values.
        beta: Inverse temperature parameter.

    Returns:
        KL divergence as a float.
    """
    N = rewards.shape[0]
    if N <= 1:
        return 0.0

    # Log-sum-exp trick: log_probs = beta * r_n - log(sum_m exp(beta * r_m))
    scaled = beta * rewards
    log_Z = torch.logsumexp(scaled, dim=0)  # log(sum_m exp(beta * r_m))
    log_q = scaled - log_Z                  # log q_beta(n)

    # KL(q_beta || uniform) = sum_n q_beta(n) * [log(q_beta(n)) + log(N)]
    #                       = sum_n q_beta(n) * log(N * q_beta(n))
    q = torch.exp(log_q)
    kl = torch.sum(q * (log_q + math.log(N))).item()

    return kl


def compute_adaptive_beta(
    rewards: torch.Tensor,
    gamma: float = math.log(2),
    beta_min: float = 0.1,
    beta_max: float = 100.0,
    tol: float = 0.01,
) -> float:
    """Find beta such that KL(q_beta || uniform) = gamma via binary search.

    The entropic utility objective selects an inverse temperature beta that
    concentrates probability mass on high-reward samples while maintaining
    a bounded KL divergence from the uniform distribution. The target
    gamma = ln(2) means roughly half the probability mass is effectively used.

    KL(q_beta || uniform) is monotonically increasing in beta:
      - beta -> 0: q_beta -> uniform, KL -> 0
      - beta -> inf: q_beta -> argmax, KL -> log(N)

    Special case: if all rewards are equal, any beta yields KL = 0, so we
    return beta_min (no useful signal to differentiate samples).

    Args:
        rewards: Tensor of shape (N,) containing reward values.
        gamma: Target KL divergence. Default is ln(2) ≈ 0.693.
        beta_min: Lower bound for binary search. Default 0.1.
        beta_max: Upper bound for binary search. Default 100.0.
        tol: Convergence tolerance. Default 0.01.

    Returns:
        The adaptive beta value as a float.
    """
    N = rewards.shape[0]
    if N <= 1:
        return beta_min

    # Special case: all rewards identical — no signal to exploit
    if torch.all(rewards == rewards[0]).item():
        return beta_min

    # Check boundary conditions
    kl_low = compute_kl_for_beta(rewards, beta_min)
    if kl_low > gamma:
        return beta_min

    kl_high = compute_kl_for_beta(rewards, beta_max)
    if kl_high < gamma:
        return beta_max

    # Binary search
    lo, hi = beta_min, beta_max
    while (hi - lo) > 1e-12:  # guard against infinite loop
        mid = (lo + hi) / 2.0
        kl_mid = compute_kl_for_beta(rewards, mid)

        if abs(kl_mid - gamma) < tol:
            return mid

        if kl_mid < gamma:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def compute_loo_entropic_advantages(
    rewards: torch.Tensor,
    beta: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute Leave-One-Out entropic advantages (paper formulas 6-7).

    For each sample n, the advantage is computed relative to the average
    exponentiated reward of all *other* samples:

        A_n = exp(beta * (r_n - r_max)) / (Z_{-n} + epsilon) - 1

    where Z_{-n} = (1 / (N-1)) * sum_{m != n} exp(beta * (r_m - r_max))

    The subtraction of r_max inside the exp is for numerical stability and
    does not change the relative ordering (it cancels in the ratio).

    Args:
        rewards: Tensor of shape (N,) containing reward values.
        beta: Inverse temperature parameter (typically from compute_adaptive_beta).
        epsilon: Small constant for numerical stability. Default 1e-8.

    Returns:
        Tensor of shape (N,) containing per-sample advantages.
    """
    N = rewards.shape[0]

    if N <= 1:
        return torch.zeros_like(rewards)

    # Numerical stability: subtract r_max before exponentiating
    r_max = rewards.max()
    exp_scaled = torch.exp(beta * (rewards - r_max))  # shape (N,)

    # Total sum of exponentiated rewards
    total_sum = exp_scaled.sum()  # scalar

    # Z_{-n} = (1 / (N-1)) * (total_sum - exp_scaled[n])  for each n
    z_loo = (total_sum - exp_scaled) / (N - 1)  # shape (N,)

    # A_n = exp(beta * (r_n - r_max)) / (Z_{-n} + epsilon) - 1
    advantages = exp_scaled / (z_loo + epsilon) - 1.0

    return advantages
