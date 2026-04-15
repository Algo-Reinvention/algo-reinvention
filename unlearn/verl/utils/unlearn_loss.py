import torch
import torch.nn.functional as F


def ga_loss_per_token_from_ce(ce_loss: torch.Tensor) -> torch.Tensor:
    """Gradient-ascent loss on forget tokens is the sign-flipped CE loss."""
    return -ce_loss


def npo_loss_per_token_from_logits(
    student_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute token-level NPO loss from student/reference logits."""
    if beta <= 0:
        raise ValueError(f"beta must be positive, but got {beta}")

    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    reference_log_probs = F.log_softmax(reference_logits.float(), dim=-1)

    gather_index = labels.unsqueeze(-1)
    student_selected_log_probs = student_log_probs.gather(dim=-1, index=gather_index).squeeze(-1)
    reference_selected_log_probs = reference_log_probs.gather(dim=-1, index=gather_index).squeeze(-1)
    log_ratio = student_selected_log_probs - reference_selected_log_probs
    return -F.logsigmoid(-beta * log_ratio)


def masked_token_mean(token_loss: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    token_mask = token_mask.to(device=token_loss.device, dtype=token_loss.dtype)
    denom = token_mask.sum()
    if float(denom.detach().item()) <= 0:
        return token_loss.new_zeros(())
    return (token_loss * token_mask).sum() / (denom + 1e-8)


def combine_forget_retain_losses(
    forget_token_loss: torch.Tensor,
    retain_token_loss: torch.Tensor,
    forget_token_mask: torch.Tensor,
    retain_token_mask: torch.Tensor,
    forget_weight: float = 1.0,
    retain_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forget_loss = masked_token_mean(forget_token_loss, forget_token_mask)
    retain_loss = masked_token_mean(retain_token_loss, retain_token_mask)
    total_loss = forget_weight * forget_loss + retain_weight * retain_loss
    return total_loss, forget_loss, retain_loss
