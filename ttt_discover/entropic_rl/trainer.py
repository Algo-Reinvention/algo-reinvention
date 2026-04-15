"""Single-step Entropic RL trainer.

Orchestrates the full pipeline for one problem:
  1. Build prompt (with optional previous best code)
  2. Sample N responses from the current policy via vLLM
  3. Extract solve functions from each response
  4. Execute code against test cases and compute verifier rewards
  5. Compute adaptive inverse-temperature beta
  6. Compute leave-one-out entropic advantages
  7. Fold KL(π_θ || π_init) penalty into advantages
  8. Run num_inner_epochs of on-policy entropic PG updates
  9. Return the best code, reward, and training metrics
"""

import logging
import torch
from dataclasses import dataclass
from typing import Optional

from ..utils.code_extraction import extract_solve_function
from ..reward.verifier import compute_batch_rewards
from .advantages import compute_adaptive_beta, compute_loo_entropic_advantages
from .policy_loss import compute_policy_loss

logger = logging.getLogger(__name__)


@dataclass
class RLStepResult:
    """Result from a single Entropic RL step."""
    best_reward: float
    best_code: str
    mean_reward: float
    beta: float
    mean_advantage: float
    policy_loss: float
    clip_fraction: float
    approx_kl: float
    num_samples: int
    execution_success_rate: float
    rewards: list  # all N rewards
    num_truncated: int = 0  # how many responses were truncated by max_tokens
    kl_ref: float = 0.0     # mean KL divergence against reference policy
    responses: list = None   # all N raw responses (full model output)
    solve_codes: list = None # all N extracted solve functions


def _empty_result() -> RLStepResult:
    """Return a zeroed-out result used when the step cannot produce anything useful."""
    return RLStepResult(
        best_reward=0.0,
        best_code="",
        mean_reward=0.0,
        beta=0.0,
        mean_advantage=0.0,
        policy_loss=0.0,
        clip_fraction=0.0,
        approx_kl=0.0,
        num_samples=0,
        execution_success_rate=0.0,
        rewards=[],
    )


class EntropicRLStep:
    """Performs a single Entropic RL step for one problem.

    Pipeline:
    1. Build prompt (including previous best code if available)
    2. Sample N responses from model
    3. Extract solve functions from responses
    4. Execute code and compute verifier rewards
    5. Compute adaptive beta
    6. Compute LOO entropic advantages
    7. Fold KL(π_θ || π_init) into per-sample advantages
    8. Run num_inner_epochs of on-policy entropic PG updates
    9. Return best result
    """

    def __init__(self, config):
        """
        Args:
            config: A namespace / dataclass / dict-like object exposing at least:
                - num_samples_per_step: int (N)
                - kl_budget: float           (gamma, default ln(2))
                - beta_min: float
                - beta_max: float
                - beta_search_tol: float
                - loo_epsilon: float
                - clip_ratio: float
                - num_inner_epochs: int
                - mini_batch_size: int
                - temperature: float
                - max_new_tokens: int
                - execution_timeout: float
                - execution_memory_limit: int
                - max_execution_workers: int
        """
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        problem,
        model,
        ref_model,
        tokenizer,
        inference_client,
        optimizer,
        gpu_offload_fn=None,
    ) -> RLStepResult:
        """Execute one Entropic RL step.

        Args:
            problem:          Problem instance (from data.problem_loader).
            model:            Trainable policy model.
            ref_model:        Frozen reference model for KL regularization.
                              When not None and ``config.kl_coeff > 0``, a
                              KL(π_θ || π_ref) penalty is added to the loss.
            tokenizer:        Tokenizer associated with *model*.
            inference_client: InferenceClient for sampling & log-prob queries.
            optimizer:        ``torch.optim.Optimizer`` bound to *model*.
            gpu_offload_fn:   Optional tuple ``(to_gpu, to_cpu)`` of callables.
                              Called to move models to GPU before log-prob /
                              training computation, and back to CPU afterwards
                              (verl-style offloading for GPU sharing with vLLM).

        Returns:
            RLStepResult containing metrics and the best code found.
        """
        config = self.config

        # ── 1. Build prompt ───────────────────────────────────────────
        # The caller (ttt_discover.py) attaches _current_best_code /
        # _current_best_reward to the problem before calling us.
        prompt = inference_client.build_prompt(
            problem.problem_text,
            best_code=getattr(problem, "_current_best_code", ""),
            best_reward=getattr(problem, "_current_best_reward", 0.0),
        )

        # ── 2. Sample N responses ────────────────────────────────────
        problem_id = getattr(problem, "problem_id", "<unknown>")
        logger.info(
            "Sampling %d responses for %s ...",
            config.num_samples_per_step,
            problem_id,
        )
        try:
            responses_list = inference_client.sample(
                [prompt],
                n=config.num_samples_per_step,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
            )
            responses = responses_list[0] if responses_list else []
        except Exception:
            logger.exception("Sampling failed for %s", problem_id)
            return _empty_result()

        if not responses:
            logger.warning("No responses generated for %s", problem_id)
            return _empty_result()

        # Check truncation stats from inference client
        num_truncated = getattr(inference_client, "last_num_truncated", 0)
        num_total = getattr(inference_client, "last_num_total", len(responses))
        if num_truncated > 0:
            logger.warning(
                "Problem %s: %d / %d responses were TRUNCATED (hit max_tokens). "
                "These likely don't contain a solve function.",
                problem_id,
                num_truncated,
                num_total,
            )

        # Log first response for debugging
        logger.debug(
            "Sample response #0 (first 500 chars):\n%s",
            responses[0][:500] if responses[0] else "<empty>",
        )

        # ── 3. Extract solve functions ────────────────────────────────
        solve_codes = []
        for i, r in enumerate(responses):
            try:
                code = extract_solve_function(r)
                solve_codes.append(code)
            except Exception:
                logger.debug("Code extraction failed for response #%d; treating as empty.", i)
                solve_codes.append("")

        # If every extraction failed, log the first response for debugging.
        if all(code == "" for code in solve_codes):
            logger.warning(
                "All code extractions failed for %s. First response (500 chars):\n%s",
                problem_id,
                responses[0][:500] if responses else "<no responses>",
            )
            return RLStepResult(
                best_reward=0.0,
                best_code="",
                mean_reward=0.0,
                beta=0.0,
                mean_advantage=0.0,
                policy_loss=0.0,
                clip_fraction=0.0,
                approx_kl=0.0,
                num_samples=len(responses),
                execution_success_rate=0.0,
                rewards=[0.0] * len(responses),
                num_truncated=num_truncated,
                responses=responses,
                solve_codes=solve_codes,
            )

        # ── 4. Compute verifier rewards ──────────────────────────────
        logger.info("Computing rewards for %d samples ...", len(solve_codes))
        try:
            rewards = compute_batch_rewards(
                solve_codes,
                problem,
                timeout=config.execution_timeout,
                memory_limit_mb=config.execution_memory_limit,
                max_workers=config.max_execution_workers,
                reward_mode=getattr(config, "reward_mode", "binary"),
                time_limit=getattr(config, "reward_time_limit", 0.9),
                time_ceiling=getattr(config, "reward_time_ceiling", 1.5),
            )
        except Exception:
            logger.exception("Reward computation failed; returning zero rewards.")
            rewards = [0.0] * len(solve_codes)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        execution_success_rate = (
            sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0
        )

        # If all rewards are identical, inject a phantom reward=1.0 sample
        # so that the real samples all get negative advantage ("move away
        # from these outputs").  The phantom sample is never used for
        # gradient computation — it only influences the advantage calculation.
        phantom_injected = False
        if rewards_tensor.std() < 1e-8:
            phantom_reward = 1.0
            logger.info(
                "All rewards identical (%.3f). Injecting phantom reward=%.1f "
                "to create gradient signal (all real samples get negative advantage).",
                rewards_tensor[0].item(),
                phantom_reward,
            )
            rewards_tensor = torch.cat([
                rewards_tensor,
                torch.tensor([phantom_reward], dtype=torch.float32),
            ])
            phantom_injected = True

        # ── 5. Compute adaptive beta ─────────────────────────────────
        beta = compute_adaptive_beta(
            rewards_tensor,
            gamma=config.kl_budget,
            beta_min=config.beta_min,
            beta_max=config.beta_max,
            tol=config.beta_search_tol,
        )

        # ── 6. Compute LOO entropic advantages ───────────────────────
        advantages = compute_loo_entropic_advantages(
            rewards_tensor, beta, epsilon=config.loo_epsilon
        )

        logger.info(
            "beta=%.3f, mean_adv=%.3f, max_adv=%.3f, min_adv=%.3f",
            beta,
            advantages.mean().item(),
            advantages.max().item(),
            advantages.min().item(),
        )

        # Remove the phantom sample's advantage — it has no corresponding
        # response/log-probs and must not participate in the gradient update.
        # But first, re-scale the real advantages so that the phantom doesn't
        # cause an extreme imbalance (phantom adv ~28 vs real ~-0.3 → the
        # gradient is dominated by the phantom, leading to model collapse).
        #
        # Strategy: the phantom only exists to make LOO produce non-zero
        # advantages for the real samples.  We keep the *sign and relative
        # ordering* of the real advantages, but normalise them so that the
        # magnitude is reasonable: mean |A_real| ≈ 1.
        if phantom_injected:
            real_adv = advantages[:-1]  # all real samples
            adv_abs_mean = real_adv.abs().mean().clamp(min=1e-8)
            real_adv = real_adv / adv_abs_mean  # normalise to ~unit scale
            advantages = real_adv
            rewards_tensor = rewards_tensor[:-1]
            logger.info(
                "Phantom advantage normalisation: abs_mean_before=%.4f, "
                "mean_adv_after=%.4f, max=%.4f, min=%.4f",
                adv_abs_mean.item(),
                advantages.mean().item(),
                advantages.max().item(),
                advantages.min().item(),
            )

        # ── 6b. Move models to GPU (verl-style offloading) ─────────────
        # Sampling, code extraction, reward computation are done — none of
        # them needed the local model on GPU.  Now move models to GPU for
        # log-prob computation and entropic PG training.
        _to_gpu, _to_cpu = gpu_offload_fn if gpu_offload_fn else (None, None)
        if _to_gpu is not None:
            _to_gpu()

        device = next(model.parameters()).device
        prompt_texts = [prompt] * len(responses)

        # ── 7. Compute reference log probs (for KL penalty) ──────────
        # In entropic PG (paper), the only "old" policy we need is π_init
        # (the initial/reference policy) for the KL(π_θ || π_init) penalty.
        # There is no PPO-style importance sampling ratio.
        response_masks = None
        ref_log_probs = None
        if ref_model is not None and config.kl_coeff > 0:
            try:
                original_model = inference_client.model
                inference_client.model = ref_model
                ref_log_probs, response_masks = inference_client.compute_log_probs(
                    prompt_texts, responses
                )
                inference_client.model = original_model
                ref_log_probs = ref_log_probs.detach().to(device)
                response_masks = response_masks.to(device)
            except Exception:
                logger.exception(
                    "Failed to compute ref log-probs; disabling KL penalty this step."
                )
                inference_client.model = model
                ref_log_probs = None

        # If we didn't get response_masks from ref computation, get them now
        if response_masks is None:
            try:
                _, response_masks = inference_client.compute_log_probs(
                    prompt_texts[:1], responses[:1]
                )
                # We need masks for all samples; compute properly
                _, response_masks = inference_client.compute_log_probs(
                    prompt_texts, responses
                )
                response_masks = response_masks.to(device)
            except Exception:
                logger.exception("Failed to compute response masks; aborting RL update.")
                if _to_cpu is not None:
                    _to_cpu()
                best_idx = int(rewards_tensor.argmax().item())
                return RLStepResult(
                    best_reward=rewards[best_idx],
                    best_code=solve_codes[best_idx],
                    mean_reward=rewards_tensor.mean().item(),
                    beta=beta,
                    mean_advantage=advantages.mean().item(),
                    policy_loss=0.0,
                    clip_fraction=0.0,
                    approx_kl=0.0,
                    num_samples=len(rewards),
                    execution_success_rate=execution_success_rate,
                    rewards=rewards,
                    responses=responses,
                    solve_codes=solve_codes,
                )

        # ── 7b. Fold KL(π_θ || π_init) into per-sample advantages ───────
        # Paper: A(a;s) = w_β(a) - 1 - λ · log(π_θ(a|s) / π_init(a|s))
        # The LOO entropic part (w_β - 1) is already in `advantages`.
        # Now subtract λ · KL_n for each sample, using frozen current log-probs.
        if ref_log_probs is not None and config.kl_coeff > 0:
            try:
                # Compute frozen (detached) log-probs from the current policy
                cur_log_probs_list = []
                for idx in range(len(responses)):
                    cur_lp, _ = inference_client.compute_log_probs(
                        [prompt_texts[idx]], [responses[idx]],
                        no_grad=True,
                    )
                    cur_log_probs_list.append(cur_lp.detach().to(device))

                # Per-sample KL: sum of (log π_θ - log π_init) over response tokens
                for idx in range(len(responses)):
                    cur_lp = cur_log_probs_list[idx]  # (1, seq_len)
                    ref_lp = ref_log_probs[idx:idx+1]  # (1, seq_len)
                    mask = response_masks[idx:idx+1]    # (1, seq_len)

                    min_len = min(cur_lp.shape[1], ref_lp.shape[1])
                    kl_per_token = (cur_lp[:, :min_len] - ref_lp[:, :min_len]) * mask[:, :min_len]
                    kl_n = kl_per_token.sum() / mask[:, :min_len].sum().clamp(min=1.0)

                    # A(a;s) = (w_β - 1) - λ · KL_n
                    advantages[idx] = advantages[idx] - config.kl_coeff * kl_n.item()

                logger.info(
                    "Folded KL into advantages: mean_adv_after=%.4f",
                    advantages.mean().item(),
                )
            except Exception:
                logger.exception(
                    "Failed to fold KL into advantages; proceeding without KL penalty."
                )

        advantages_expanded = advantages.to(device)

        # ── 8. Entropic PG training with gradient accumulation ─────────
        total_loss = 0.0
        total_clip_fraction = 0.0
        total_approx_kl = 0.0
        total_kl_ref = 0.0
        num_updates = 0

        N = len(responses)
        mbs = min(config.mini_batch_size, N)
        early_stop = False

        model.train()
        # Enable gradient checkpointing to reduce GPU memory for long sequences.
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        for inner_epoch in range(config.num_inner_epochs):
            if early_stop:
                break

            # Shuffle sample indices for this epoch
            indices = torch.randperm(N)
            num_mini_batches = (N + mbs - 1) // mbs  # ceiling division

            for mb_idx in range(num_mini_batches):
                if early_stop:
                    break

                mb_start = mb_idx * mbs
                mb_end = min(mb_start + mbs, N)
                mb_indices = indices[mb_start:mb_end].tolist()
                mb_size = len(mb_indices)

                # Slice frozen tensors for this mini-batch
                mb_adv = advantages_expanded[mb_indices]
                mb_mask = response_masks[mb_indices]

                # --- Gradient accumulation: process one sample at a time ---
                optimizer.zero_grad()
                accum_loss = 0.0
                accum_metrics = {"clip_fraction": 0.0, "approx_kl": 0.0, "kl_ref": 0.0, "mean_ratio": 0.0}
                accum_ok = 0

                for si, sample_idx in enumerate(mb_indices):
                    try:
                        s_lp, s_mask = inference_client.compute_log_probs(
                            [prompt_texts[sample_idx]], [responses[sample_idx]],
                            no_grad=False,
                        )
                    except Exception:
                        logger.debug("log-prob failed for sample %d; skipping.", sample_idx)
                        continue

                    s_lp = s_lp.to(device)

                    # Align seq_len
                    s_adv = mb_adv[si: si + 1]
                    s_msk = mb_mask[si: si + 1]

                    min_len = min(s_lp.shape[1], s_msk.shape[1])
                    s_lp = s_lp[:, :min_len]
                    s_msk = s_msk[:, :min_len]

                    s_loss, s_metrics = compute_policy_loss(
                        new_log_probs=s_lp,
                        advantages=s_adv,
                        response_mask=s_msk,
                    )

                    # Scale loss by 1/mb_size for proper averaging
                    (s_loss / mb_size).backward()
                    accum_loss += s_loss.item()
                    for k in accum_metrics:
                        accum_metrics[k] += s_metrics.get(k, 0.0)
                    accum_ok += 1

                if accum_ok == 0:
                    logger.warning("All samples failed in mb %d; skipping.", mb_idx)
                    optimizer.zero_grad()
                    continue

                # Optimizer step with accumulated gradients
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=config.max_grad_norm
                    )
                    optimizer.step()
                except Exception:
                    logger.exception(
                        "Optimizer step failed at epoch %d, mb %d; skipping.",
                        inner_epoch, mb_idx,
                    )
                    optimizer.zero_grad()
                    continue

                avg_mb_loss = accum_loss / accum_ok
                metrics = {k: v / accum_ok for k, v in accum_metrics.items()}

                total_loss += avg_mb_loss
                total_clip_fraction += metrics.get("clip_fraction", 0.0)
                total_approx_kl += metrics.get("approx_kl", 0.0)
                total_kl_ref += metrics.get("kl_ref", 0.0)
                num_updates += 1

                logger.debug(
                    "  Epoch %d, MB %d/%d: loss=%.4f, kl=%.4f",
                    inner_epoch, mb_idx + 1, num_mini_batches,
                    avg_mb_loss, metrics.get("approx_kl", 0.0),
                )

                # Early-stop if KL divergence exceeds safety threshold.
                if metrics.get("approx_kl", 0.0) > 0.1:
                    logger.info(
                        "Early stopping inner loop: KL=%.4f > 0.1 "
                        "(epoch %d, mb %d)",
                        metrics["approx_kl"], inner_epoch, mb_idx,
                    )
                    early_stop = True

        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

        # Sync the inference client so subsequent sampling uses the updated
        # weights.
        try:
            inference_client.update_model(model)
        except Exception:
            logger.exception("Failed to push updated model to inference client.")

        # ── 8b. Move models back to CPU (verl-style offloading) ───────
        if _to_cpu is not None:
            _to_cpu()

        # ── 9. Assemble & return result ───────────────────────────────
        best_idx = int(rewards_tensor.argmax().item())

        avg_loss = total_loss / num_updates if num_updates > 0 else 0.0
        avg_clip = total_clip_fraction / num_updates if num_updates > 0 else 0.0
        avg_kl = total_approx_kl / num_updates if num_updates > 0 else 0.0
        avg_kl_ref = total_kl_ref / num_updates if num_updates > 0 else 0.0

        return RLStepResult(
            best_reward=rewards[best_idx],
            best_code=solve_codes[best_idx],
            mean_reward=rewards_tensor.mean().item(),
            beta=beta,
            mean_advantage=advantages.mean().item(),
            policy_loss=avg_loss,
            clip_fraction=avg_clip,
            approx_kl=avg_kl,
            num_samples=len(rewards),
            execution_success_rate=execution_success_rate,
            rewards=rewards,
            kl_ref=avg_kl_ref,
            num_truncated=num_truncated,
            responses=responses,
            solve_codes=solve_codes,
        )
