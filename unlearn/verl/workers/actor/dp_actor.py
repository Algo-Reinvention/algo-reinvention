"""NOTE @jzhao:
Add the logic of retain_loss calculation
Add the entropy_mask/prefix_mask
"""

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

# ========== @jzhao ==========
import time
import os
import warnings
import json
import torch.distributed as dist
from transformers import AutoTokenizer
from typing import Any
# ========== /@jzhao ==========


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ========== @jzhao ==========
_tokenizer_cache = None

def get_rank_and_world_size():
    """Helper to get rank and world size, especially for Ray/DDP context."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    # Fallback for non-distributed environment (e.g., local Ray actor)
    raise ValueError("non-distributed ERROR")
    return 0, 1

def get_cached_tokenizer(path):
    global _tokenizer_cache
    if _tokenizer_cache is None:
        try:
            warnings.warn(f"Loading tokenizer from: {path}", UserWarning)
            _tokenizer_cache = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except Exception as e:
            warnings.warn(f"Failed to load tokenizer from {path}. Error: {e}", UserWarning)
            _tokenizer_cache = None
    return _tokenizer_cache

def log_debug_retain_info(config, data, log_dir, epoch_info='N/A'):
    """Logs raw retain_inputs Tensors for the current micro-batch to a JSON file."""
    rank, _ = get_rank_and_world_size()

    # 1. Create the filename and path.
    # Add the _retain marker so it is distinguishable from policy debug logs.
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    random_suffix = torch.randint(0, 10000, (1,)).item()
    filename = f"rank{rank}_epoch{epoch_info}_retain_ts{timestamp}_rand{random_suffix}.json"
    log_path = os.path.join(log_dir, filename)

    # 2. Prepare data by iterating over the batch dimension and converting tensors to lists.
    log_list = []

    # Assume data is a dictionary containing keys such as 'input_ids',
    # 'responses', 'attention_mask', and 'position_ids', where dimension 0 is batch_size.
    first_key = next(iter(data))
    batch_size = data[first_key].size(0)

    for i in range(batch_size):
        sample_data = {
            'sample_idx': i,
            'epoch_info': epoch_info,
        }

        # Handle every key-value pair in data dynamically.
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                # Move tensors to CPU and convert them to lists.
                sample_data[key] = value[i].cpu().tolist()
            else:
                # Record non-tensor content directly.
                sample_data[key] = value

        log_list.append(sample_data)

    # 3. Ensure the directory exists and save the JSON file.
    os.makedirs(log_dir, exist_ok=True)
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_list, f, ensure_ascii=False, indent=4)
    except Exception as e:
        warnings.warn(f"Warning: Failed to save retain debug log to {log_path}. Error: {e}", UserWarning)

def log_debug_info(config, data, combined_mask, entropy, log_dir, epoch_info='N/A'):
    """Logs detailed debug information for the current micro-batch to a JSON file."""
    rank, world_size = get_rank_and_world_size()
    tokenizer = get_cached_tokenizer(config.get('model_path', False))

    # 1. Create the filename and path.
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    random_suffix = torch.randint(0, 10000, (1,)).item()
    filename = f"rank{rank}_epoch{epoch_info}_ts{timestamp}_rand{random_suffix}.json" # Use the .json suffix.
    log_path = os.path.join(log_dir, filename)

    # 2. Prepare sample data.
    responses_ids = data['responses'].cpu()
    response_masks = data['response_mask'].cpu()
    token_entropy = entropy.cpu()
    grad_compute_mask = combined_mask.cpu()

    # 3. Process samples one by one and build a JSON list.
    log_list = []

    # Convert all tensors to Python lists (list of lists).
    responses_ids_list = responses_ids.tolist()
    response_masks_list = response_masks.tolist()
    token_entropy_list = token_entropy.tolist()
    grad_compute_mask_list = grad_compute_mask.tolist()

    for i in range(responses_ids.size(0)):
        # Extract response IDs excluding padding and prompt tokens.
        sample_ids = responses_ids_list[i]
        sample_mask = response_masks_list[i]

        # Find the end of the actual response (last mask=1 index + 1).
        actual_length = sum(sample_mask)
        actual_ids = sample_ids[:actual_length]

        # Extract the matching entropy and mask slices.
        sample_entropy = token_entropy_list[i][:actual_length]
        sample_grad_mask = grad_compute_mask_list[i][:actual_length]

        # Tokenize/Decode
        response_text = "N/A (No Tokenizer)"
        response_tokens = ["N/A"]

        if tokenizer:
            # Convert to token strings for easier inspection.
            try:
                # Use convert_ids_to_tokens to get token strings.
                response_tokens = tokenizer.convert_ids_to_tokens(actual_ids)
                # Use decode to obtain the full text.
                response_text = tokenizer.decode(actual_ids, skip_special_tokens=True)
            except Exception as e:
                warnings.warn(f"Tokenizer operation failed for sample {i}. Error: {e}", UserWarning)


        sample_data = {
            'sample_idx': i,
            'epoch_info': epoch_info,
            'response_text': response_text,      # Str
            # 'original_response_ids': actual_ids, # List of Ints
            # 'response_tokens': response_tokens,  # List of Strs
            # 'token_entropy_list': sample_entropy, # List of Floats
            # 'response_mask_list': sample_mask,
            # 'grad_compute_mask_list': sample_grad_mask, # List of Bools/Ints
            # # Record other key debugging information.
            # 'old_log_probs_slice': data['old_log_probs'][i,:actual_length].cpu().tolist(),
            # 'advantages_slice': data['advantages'][i,:actual_length].cpu().tolist(),
        }
        log_list.append(sample_data)

    # 4. Ensure the directory exists and save the JSON file.
    os.makedirs(log_dir, exist_ok=True)
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            # Use indent so the JSON remains readable.
            json.dump(log_list, f, ensure_ascii=False, indent=4)
    except Exception as e:
        warnings.warn(f"Warning: Failed to save debug log to {log_path}. Error: {e}", UserWarning)

# ========== /@jzhao ==========


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    """NOTE @jzhao:
    - self.config is config.actor_rollout_ref.actor
    - batch: {
        bacth: {
            "retain_input_ids",
            "retain_attention_mask",
            "retain_position_ids",
            "prompts": prompt_ids,
            "responses": response_ids,
            "input_ids": sequence_ids,  # here input_ids become the whole sentences
            "attention_mask": attention_mask,
            "response_mask"
            "position_ids": position_ids,
            (maybe) "rollout_log_probs"
            (maybe) "entropys"
            "old_log_probs"
            (maybe) "ref_log_prob"
            (maybe) "values"
            (raw_reward) "token_level_scores"
            "token_level_rewards"
            "advantages"
            "returns"
        }
        non_tensor_batch: {
            "uid"
            "raw_prompt_ids"
            <reward_extra_info>
        }
        meta_info: {
            "epoch"
            "global_steps"
            "eos_token_id"
            "pad_token_id"
            "micro_batch_size"
            "temperature"
            ...
        }
    }
    """
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        # ========== @jzhao ============
        debug_log_dir = getattr(self.config, 'policy_debug_log_dir', None)
        epoch_info = data.meta_info.get('epoch', '0')

        pos_ratio = getattr(self.config, 'policy_pos_ratio', 1.0)
        entropy_ratio = getattr(self.config, 'policy_entropy_ratio', 1.0)
        # ========== /@jzhao ===========

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "retain_input_ids",
            "retain_response_ids",
            "retain_attention_mask",
            "retain_position_ids",
            "retain_loss_mask"
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    warnings.warn(f"Grad_accumulation: {self.gradient_accumulation}")

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    # @jzhao
                    retain_micro_batch = micro_batch.pop(
                        batch_keys=["retain_input_ids", "retain_response_ids", "retain_attention_mask", "retain_position_ids", "retain_loss_mask"],
                        non_tensor_batch_keys=[]
                    )
                    retain_inputs = {
                        "input_ids": retain_micro_batch.batch["retain_input_ids"],
                        "responses": retain_micro_batch.batch["retain_response_ids"],
                        "attention_mask": retain_micro_batch.batch["retain_attention_mask"],
                        "position_ids": retain_micro_batch.batch["retain_position_ids"]
                    }
                    _, retain_log_probs = self._forward_micro_batch(
                        retain_inputs, temperature=1.0, calculate_entropy=False
                    )
                    retain_inputs["loss_mask"] = retain_micro_batch.batch["retain_loss_mask"]
                    retain_inputs["log_probs"] = retain_log_probs
                    loss_mask = retain_micro_batch.batch["retain_loss_mask"]
                    # retain_loss = -(retain_log_probs * loss_mask).sum() / loss_mask.sum()
                    retain_loss_sum = loss_mask.sum()
                    if retain_loss_sum > 0:
                        retain_loss = -(retain_log_probs * loss_mask).sum() / retain_loss_sum
                    else:
                        retain_loss = torch.tensor(0.0, device=loss_mask.device)
                    # /@jzhao

                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)

                    # ========== @jzhao ==========
                    # calculate_entropy = False
                    # if entropy_coeff != 0:
                    calculate_entropy = True
                    # ========== /@jzhao ==========

                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # ========== @jzhao ============
                    responses = model_inputs['responses']
                    response_length = responses.size(1)

                    # --- 1. Compute the positional mask. ---
                    non_padding_response_mask = response_mask.to(torch.int)
                    actual_response_lengths = non_padding_response_mask.sum(dim=1) # (B,)

                    positional_mask = torch.zeros_like(response_mask, dtype=torch.bool, device=response_mask.device)

                    for i in range(responses.size(0)):
                        length = actual_response_lengths[i].item()
                        if length == 0:
                            continue

                        keep_count = int(torch.ceil(torch.tensor(length * pos_ratio)).item())
                        start_index = length - keep_count

                        start_index = max(0, start_index)

                        positional_mask[i, start_index:length] = True

                    positional_mask = positional_mask & response_mask


                    # --- 2. Compute the entropy mask. ---

                    entropy_mask = torch.zeros_like(response_mask, dtype=torch.bool, device=response_mask.device)

                    rank, _ = get_rank_and_world_size()

                    # Iterate over each sample in the batch.
                    for i in range(responses.size(0)):

                        # 1. Extract valid entropy values for the current sample.
                        sample_entropy = entropy[i]
                        num_masked_tokens = sample_entropy.numel() # Number of valid tokens in the current sample.

                        if num_masked_tokens == 0:
                            continue

                        num_tokens_to_keep = int(torch.ceil(torch.tensor(num_masked_tokens * entropy_ratio)).item())

                        if debug_log_dir:
                            # Debug logging is now emitted per sample.
                            warnings.warn(f"Rank {rank} | MicroBatch {batch_idx} Sample {i}: "
                                        f"Valid Tokens: {num_masked_tokens}, "
                                        f"Ratio: {entropy_ratio}, "
                                        f"Tokens to Keep (ceil): {num_tokens_to_keep}")

                        if num_tokens_to_keep > 0 and num_tokens_to_keep < num_masked_tokens:

                            # Keep the num_tokens_to_keep tokens with the smallest entropy.
                            threshold_index = max(0, num_tokens_to_keep - 1)

                            sorted_entropy, _ = torch.sort(sample_entropy) # Sort only the valid entropy values for this sample.

                            if threshold_index >= sorted_entropy.size(0):
                                # This should not happen, but keep the guard for safety.
                                warnings.warn(f"Rank {rank} | CRITICAL ERROR STATE (Sample {i}): Index {threshold_index} >= Size {sorted_entropy.size(0)}. Skipping sample.")
                                continue

                            # The threshold is the num_tokens_to_keep-th smallest entropy.
                            entropy_threshold = sorted_entropy[threshold_index]

                            # 2. Apply the threshold to the original entropy row and intersect it with response_mask.
                            sample_mask = (entropy[i] <= entropy_threshold) & response_mask[i]

                            if debug_log_dir:
                                # Debug logging is now emitted per sample.
                                warnings.warn(f"Entropy Threshold: {entropy_threshold}")
                                warnings.warn(f"Finally retrained: {sum(sample_mask)}")

                            # 3. Store the result.
                            entropy_mask[i] = sample_mask

                        elif num_tokens_to_keep >= num_masked_tokens and num_masked_tokens > 0:
                            # If all valid tokens should be kept (ratio >= 1.0),
                            # use the full response_mask row.
                            entropy_mask[i] = response_mask[i]

                        else:
                            # num_tokens_to_keep == 0
                            pass # Keep this row entirely False.

                    # --- 3. Combine masks and apply them. ---

                    combined_mask = positional_mask & entropy_mask

                    # # --- Prevent <im_end> from being unlearned. ---
                    # for i in range(combined_mask.size(0)):
                    #     length = actual_response_lengths[i].item()
                    #     if length > 0:
                    #         # The last valid token index is length - 1. Force it to False.
                    #         combined_mask[i, length - 1] = False

                    if not combined_mask.any():
                        warnings.warn(f"Rank {rank} | Warning: Combined mask is empty for batch {batch_idx}.")

                    # --- 4. Debug logging. ---
                    if debug_log_dir and get_rank_and_world_size()[0] == 0:
                        log_debug_info(self.config, model_inputs, combined_mask, entropy, debug_log_dir, epoch_info)
                        log_debug_retain_info(self.config, retain_inputs, debug_log_dir, epoch_info)

                    # ========== /@jzhao ===========

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    # NOTE @jzhao: pipeline / compute loss
                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=combined_mask, # <--- @jzhao
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        
                        # [New] Extract ref_log_prob, or use None if it is missing.
                        ref_log_prob = model_inputs.get("ref_log_prob", None)

                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=combined_mask, # <--- @jzhao
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            ref_log_prob=ref_log_prob  # [New] Pass ref_log_prob through.
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=combined_mask, loss_agg_mode=loss_agg_mode) # <--- @jzhao

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=combined_mask, loss_agg_mode=loss_agg_mode) # <--- @jzhao

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (combined_mask.shape[0] / self.config.ppo_mini_batch_size) # <--- @jzhao
                        retain_loss = retain_loss * (combined_mask.shape[0] / self.config.ppo_mini_batch_size)  #@jzhao
                    else:
                        loss = policy_loss / self.gradient_accumulation
                        retain_loss = retain_loss / self.gradient_accumulation  #@jzhao

                    loss = self.config.unlearn_loss_coef * loss + self.config.retain_loss_coef * retain_loss  #@jzhao

                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                            'actor/masked_token_ratio': combined_mask.sum().item() / max(1, response_mask.sum().item()),  # <--- @jzhao
                            'actor/retain_loss': retain_loss.detach().item()  # <--- @jzhao
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
