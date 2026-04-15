# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from torch import nn


# @codex
def forward_kl_per_token(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, but got {temperature}")

    teacher_logits = teacher_logits.float() / temperature
    student_logits = student_logits.float() / temperature

    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    token_kl = token_kl * (temperature**2)
    return token_kl


# @codex
def flatten_masked_token_loss(token_loss: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    flat_token_loss = token_loss.reshape(-1)
    flat_loss_mask = loss_mask.reshape(-1).to(flat_token_loss.device)
    return flat_token_loss * flat_loss_mask


# @jzhao
def masked_forward_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    token_kl = forward_kl_per_token(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        temperature=temperature,
    )
    return flatten_masked_token_loss(token_loss=token_kl, loss_mask=loss_mask)


# @jzhao
def apply_noise_to_model(model: nn.Module, alpha: float, beta: float):
    if alpha < 0:
        raise ValueError(f"noise_alpha must be non-negative, but got {alpha}")
    if beta < 0:
        raise ValueError(f"noise_beta must be non-negative, but got {beta}")
    if alpha == 0:
        return

    with torch.no_grad():
        for param in model.parameters():
            if not param.is_floating_point():
                continue

            noise = torch.empty_like(param)
            if param.ndim >= 2:
                nn.init.xavier_uniform_(noise)
            else:
                noise.zero_()
            # student = (1 - alpha) * student + alpha * beta * noise
            param.data = (1 - alpha) * param.data + alpha * beta * noise
