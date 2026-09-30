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

import itertools
import logging
import os
from typing import Optional, Tuple
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import verl.utils.torch_functional as verl_F
from vista.ema import update_ema_teacher
from vista.objectives import full_vocab_distillation_loss
from vista.schedule import linear_warmup_value
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    chunked_vocab_logsumexp,
    compute_opd_jsd_loss,
    compute_policy_loss,
    kl_penalty,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input
__all__ = ["DataParallelPPOActor"]
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.teacher_module = None
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        module=None,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        full_vocab: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_log_probs: # (bs, response_len, top_k), or None
            topk_indices: # (bs, response_len, top_k), only for student selection
        """
        use_topk = distill_topk is not None or topk_indices is not None
        return_topk_indices = use_topk and topk_indices is None
        if full_vocab and (self.use_fused_kernels or self.use_ulysses_sp):
            raise ValueError("Full-vocabulary distillation requires unfused logits and sequence parallel size 1")
        if use_topk and self.use_fused_kernels:
            raise ValueError("CVE-Distill Top-K JSD requires use_fused_kernels=False")
        if distill_topk is not None and int(distill_topk) <= 0:
            raise ValueError(f"distill_topk must be positive, got {distill_topk}")
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )
        forward_module = module if module is not None else self.actor_module
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            multi_modal_inputs = {
                key: value.to(input_ids.device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in multi_modal_inputs.items()
            }
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)
            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
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
                        input_ids_rmpad_rolled, position_ids_rmpad=None, sp_size=self.ulysses_sequence_parallel_size
                    )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                output = forward_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )
                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)
                    entropy_rmpad = output.entropy.squeeze(0)
                else:
                    logits_rmpad = output.logits.squeeze(0)
                    logits_rmpad.div_(temperature)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(int(distill_topk), logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size, seqlen, topk, device=topk_indices.device, dtype=topk_indices.dtype
                            )
                            full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        topk_log_probs_rmpad = topk_logits_rmpad - chunked_vocab_logsumexp(logits_rmpad)
                    inplace_backward = not (calculate_entropy or use_topk or full_vocab)
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward
                    )
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                if self.use_ulysses_sp:
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    if use_topk:
                        topk_log_probs_rmpad = gather_outpus_and_unpad(
                            topk_log_probs_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outpus_and_unpad(
                                topk_indices_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                            )
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                if use_topk:
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                        )
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
                if full_vocab:
                    # Gather only response positions; never pad the full [B, S, V] tensor.
                    inverse = torch.zeros(batch_size * seqlen, dtype=torch.long, device=indices.device)
                    inverse[indices] = torch.arange(len(indices), device=indices.device)
                    response_indices = inverse.reshape(batch_size, seqlen)[:, -response_length - 1 : -1]
                    response_logits = logits_rmpad[response_indices]
                if use_topk:
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]
                    if return_topk_indices:
                        topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]
            else:
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = forward_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )
                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]
                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]
                    log_probs = logprobs_from_logits(
                        logits,
                        micro_batch["responses"],
                        inplace_backward=not (use_topk or full_vocab or calculate_entropy),
                    )
                    if full_vocab:
                        response_logits = logits
                    if use_topk:
                        if topk_indices is None:
                            topk = min(int(distill_topk), logits.shape[-1])
                            topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
                        else:
                            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                        topk_log_probs = topk_logits - chunked_vocab_logsumexp(logits)
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)
            return (
                entropy,
                log_probs,
                response_logits if full_vocab else (topk_log_probs if use_topk else None),
                topk_indices if return_topk_indices else None,
            )

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        if not torch.isfinite(grad_norm):
            logger.warning("Skipping optimizer step: non-finite grad_norm=%s", grad_norm)
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def _update_collective_experience_teacher(self) -> float:
        experience_config = self.config.get("collective_experience", {})
        if not experience_config.get("enabled", False):
            return 0.0
        if experience_config.get("teacher_update", "ema") == "frozen":
            return 0.0
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("collective-experience EMA requires a distinct teacher module")
        return update_ema_teacher(
            teacher=self.teacher_module,
            student=self.actor_module,
            decay=float(experience_config.get("ema_decay", 0.99)),
        )

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
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        selected_data = data.select(select_keys, non_tensor_select_keys)
        batch = selected_data.batch
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
            if has_multi_modal_inputs:
                micro_batches = [selected_data.select_idxs(index) for index in indices]
        elif has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            micro_batches = selected_data.chunk(num_micro_batches)
        else:
            micro_batches = batch.split(micro_batch_size)
        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, _, _ = self._forward_micro_batch(
                    micro_batch, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]
        return (log_probs, entropys)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        multi_turn = "loss_mask" in data.batch
        group_normalization = bool(self.config.get("group_normalization", False))
        dp_scale = int(data.meta_info.get("vista_dp_world_size", 1))
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if group_normalization:
            select_keys.extend(["vista_pg_weight", "vista_opd_weight"])
        experience_config = self.config.get("collective_experience", {})
        experience_enabled = bool(experience_config.get("enabled", False))
        experience_keys = {
            "teacher_input_ids",
            "teacher_attention_mask",
            "teacher_position_ids",
            "collective_experience_teacher_mask",
        }
        if experience_enabled:
            missing_experience_keys = experience_keys.difference(data.batch.keys())
            if missing_experience_keys:
                raise ValueError(f"collective-experience actor batch is missing {sorted(missing_experience_keys)}")
            select_keys.extend(sorted(experience_keys))
        opd_loss_target_coef = float(self.config.get("opd_loss_coef", 0.0) or 0.0)
        opd_loss_warmup_steps = int(self.config.get("opd_loss_warmup_steps", 0) or 0)
        global_step = int(data.meta_info.get("global_steps", 0) or 0)
        opd_loss_coef = linear_warmup_value(
            target=opd_loss_target_coef, step=global_step, warmup_steps=opd_loss_warmup_steps
        )
        full_vocab = self.config.get("opd_distribution", "full") == "full"
        opd_topk = int(self.config.get("opd_topk", 128) or 128)
        opd_jsd_alpha = float(self.config.get("opd_jsd_alpha", 0.5))
        opd_jsd_add_tail = bool(self.config.get("opd_jsd_add_tail", True))
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        non_tensor_select_keys = [
            key for key in ("multi_modal_inputs", "teacher_multi_modal_inputs") if key in data.non_tensor_batch
        ]
        has_multi_modal_inputs = bool(non_tensor_select_keys)
        if group_normalization:
            dataloader = [data.select(select_keys, non_tensor_select_keys) if has_multi_modal_inputs else batch]
        elif has_multi_modal_inputs:
            num_mini_batches = max(1, len(data) // self.config.ppo_mini_batch_size)
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    tensors = mini_batch.batch if isinstance(mini_batch, DataProto) else mini_batch
                    lengths = tensors.select("attention_mask").clone(recurse=False)
                    if experience_enabled and opd_loss_coef > 0:
                        # Teacher prompts contain extra visual evidence and are longer.
                        lengths["attention_mask"] = tensors["teacher_attention_mask"]
                    _, indices = rearrange_micro_batches(batch=lengths, max_token_len=max_token_len)
                    micro_batches = [
                        mini_batch.select_idxs(index) if isinstance(mini_batch, DataProto) else mini_batch[index]
                        for index in indices
                    ]
                elif has_multi_modal_inputs:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = [
                        mini_batch.select_idxs(
                            list(range(i, min(i + self.config.ppo_micro_batch_size_per_gpu, len(mini_batch))))
                        )
                        for i in range(0, len(mini_batch), self.config.ppo_micro_batch_size_per_gpu)
                    ]
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                self.actor_optimizer.zero_grad()
                for micro_batch_idx, data in enumerate(micro_batches):
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]
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
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    distill_topk = opd_topk if experience_enabled and opd_loss_coef > 0 and not full_vocab else None
                    entropy, log_prob, student_topk_log_prob, student_topk_indices = self._forward_micro_batch(
                        micro_batch=data,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        distill_topk=distill_topk,
                        full_vocab=full_vocab and experience_enabled and opd_loss_coef > 0,
                    )
                    policy_loss_fn = compute_policy_loss
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                        row_weights=data["vista_pg_weight"] if group_normalization else None,
                    )
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss
                    opd_loss = log_prob.new_tensor(0.0)
                    opd_active_token_ratio = log_prob.new_tensor(0.0)
                    opd_weight_abs_mean = log_prob.new_tensor(0.0)
                    opd_raw_jsd_token_mean = log_prob.new_tensor(0.0)
                    teacher_log_prob = None
                    teacher_topk_log_prob = None
                    active_opd_mask = None
                    if experience_enabled and opd_loss_coef > 0:
                        if self.teacher_module is None:
                            raise ValueError("collective-experience teacher module was not initialized")
                        teacher_batch = {
                            "responses": responses,
                            "input_ids": data["teacher_input_ids"],
                            "attention_mask": data["teacher_attention_mask"],
                            "position_ids": data["teacher_position_ids"],
                        }
                        if "teacher_multi_modal_inputs" in data:
                            teacher_batch["multi_modal_inputs"] = data["teacher_multi_modal_inputs"]
                        self.teacher_module.eval()
                        with torch.no_grad():
                            _, teacher_log_prob, teacher_topk_log_prob, _ = self._forward_micro_batch(
                                micro_batch=teacher_batch,
                                temperature=temperature,
                                calculate_entropy=False,
                                module=self.teacher_module,
                                distill_topk=distill_topk,
                                topk_indices=student_topk_indices,
                                full_vocab=full_vocab,
                            )
                        active_opd_mask = data["collective_experience_teacher_mask"]
                    if opd_loss_coef > 0 and teacher_log_prob is not None and (active_opd_mask is not None):
                        if full_vocab:
                            opd_loss, opd_jsd_metrics = full_vocab_distillation_loss(
                                student_topk_log_prob,
                                teacher_topk_log_prob,
                                response_mask,
                                active_opd_mask,
                                row_weights=data["vista_opd_weight"] if group_normalization else None,
                                alpha=opd_jsd_alpha,
                                token_chunk_size=int(self.config.get("opd_token_chunk_size", 16)),
                            )
                        else:
                            opd_loss, opd_jsd_metrics = compute_opd_jsd_loss(
                                student_topk_log_prob=student_topk_log_prob,
                                teacher_topk_log_prob=teacher_topk_log_prob,
                                response_mask=response_mask,
                                opd_step_mask=active_opd_mask,
                                alpha=opd_jsd_alpha,
                                add_tail=opd_jsd_add_tail,
                                loss_agg_mode=loss_agg_mode,
                                row_weights=data["vista_opd_weight"] if group_normalization else None,
                            )
                        opd_active_token_ratio = opd_jsd_metrics["opd_active_token_ratio"]
                        opd_raw_jsd_token_mean = opd_jsd_metrics["opd_raw_jsd_token_mean"]
                        opd_weight_abs_mean = opd_raw_jsd_token_mean.abs()
                        policy_loss = policy_loss + opd_loss_coef * opd_loss
                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef
                    if group_normalization:
                        loss = policy_loss * dp_scale
                    elif self.config.use_dynamic_bsz:
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()
                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/opd_loss": opd_loss.detach().item(),
                        "actor/opd_loss_coef": opd_loss_coef,
                        "actor/opd_loss_target_coef": opd_loss_target_coef,
                        "actor/opd_loss_warmup_progress": 1.0
                        if opd_loss_warmup_steps == 0
                        else min(max(global_step, 0) / opd_loss_warmup_steps, 1.0),
                        "actor/opd_active_token_ratio": opd_active_token_ratio.detach().item(),
                        "actor/opd_weight_abs_mean": opd_weight_abs_mean.detach().item(),
                        "actor/opd_raw_jsd_token_mean": opd_raw_jsd_token_mean.detach().item(),
                        "actor/opd_topk": 0 if full_vocab else opd_topk,
                        "actor/opd_jsd_alpha": opd_jsd_alpha,
                    }
                    append_to_dict(metrics, data)
                grad_norm = self._optimizer_step()
                if experience_enabled and torch.isfinite(grad_norm):
                    teacher_delta = self._update_collective_experience_teacher()
                    append_to_dict(metrics, {"cve/ema_preupdate_delta": teacher_delta})
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        if group_normalization:
            for key in ("actor/pg_loss", "actor/opd_loss"):
                metrics[key] = [sum(metrics[key]) * dp_scale]
        return metrics
