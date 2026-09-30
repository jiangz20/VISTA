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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import os
import math
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Dict, Optional, Type
import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_subtask_success_rate_mean,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch
from vista.experience import (
    CollectiveExperienceCompiler,
    DecisionAligner,
    ExperienceSerializer,
    ExperienceViewProjector,
)
from vista.hindsight import derive_trajectory_outcome
from vista.supervision import training_route
from vista.objectives import prepare_group_batch
from vista.schedule import linear_decay_value
from vista.teacher import append_identical_responses

WorkerType = Type[Worker]
module_logger = logging.getLogger(__name__)


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    GRPO = "grpo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = (process_on_nodes[0], len(process_on_nodes))
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)
    kld = kld * response_mask
    beta = kl_ctrl.value
    token_level_rewards = token_level_scores - beta * kld
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards
    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}
    return (data, metrics)


def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch["token_level_scores"]
    if "step_rewards" in data.batch.keys():
        step_rewards = data.batch["step_rewards"]
    for i in range(len(data)):
        data_item = data[i]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        action_valids = data_item.non_tensor_batch["is_action_valid"].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids
        if "step_rewards" in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    valid_action_ratio = np.mean(data.non_tensor_batch["is_action_valid"].astype(np.float32)).item()
    metrics = {"episode/valid_action_ratio": valid_action_ratio}
    return (data, metrics)


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data, adv_estimator, norm_adv_by_std_in_grpo=True, **kwargs):
    if adv_estimator != "grpo":
        raise ValueError("VISTA uses GRPO with HAPI routing")
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    mask = data.batch["response_mask"]
    if "loss_mask" in data.batch:
        mask = data.batch["loss_mask"][:, -mask.size(1) :]
    advantages, returns = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=mask,
        index=data.non_tensor_batch["uid"],
        traj_index=data.non_tensor_batch["traj_uid"],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        compute_mean_std_cross_steps=False,
        sample_mask=data.batch.get("collective_experience_grpo_mask"),
    )
    data.batch["advantages"], data.batch["returns"] = (advantages, returns)
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"role_worker_mapping.keys()={role_worker_mapping.keys()!r}"
            )
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)
        self.use_critic = False
        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _is_collective_experience_enabled(self) -> bool:
        return bool(OmegaConf.select(self.config, "algorithm.collective_experience.enabled", default=False))

    @staticmethod
    def _collective_repair_instruction(status: str, failure_type: Any | None) -> str:
        """Return a short, failure-specific instruction for the visual teacher."""
        normalized = str(failure_type or status).strip().lower()
        instructions = {
            "wrong_after_crop": "The complete failed response is supervised. Focus the correction on its final answer decision: re-check successful-peer crop evidence and change the answer if that evidence supports a different choice; keep already valid crops as context.",
            "wrong_direct_answer": "The complete failed response is supervised. Focus the correction on its wrong final-answer decision using successful-peer visual evidence, keep the surrounding history fixed, and emit the smallest valid corrected answer.",
            "invalid_action": "The complete invalid response is supervised. Focus the correction on the malformed or protocol-invalid action, using successful-peer visual evidence to choose a legal crop or final answer action; earlier valid turns remain useful context.",
            "no_valid_final_answer": "The complete incomplete response is supervised. Focus the correction on the unresolved final decision, using successful-peer visual evidence to emit either one legal crop that resolves the uncertainty or the smallest valid boxed answer.",
        }
        return instructions.get(
            normalized,
            "Repair only the current non-success local decision using successful-peer visual evidence and preserve the surrounding trajectory history.",
        )

    def _prepare_collective_experience_teacher_inputs(
        self, batch: DataProto, metrics: Dict[str, float], timing_raw: Optional[Dict[str, float]] = None
    ) -> DataProto:
        """Attach routed peer-evidence supervision to the rollout batch.

        Successful trajectories retain the GRPO objective.  Every response row
        belonging to a non-success trajectory is eligible for collective-
        experience OPD, so the complete failed sample is supervised.  The
        failure-specific instruction and peer visual evidence tell the teacher
        which local decision should be corrected; the response tokens remain
        unchanged while the teacher scores them.
        """
        if not self._is_collective_experience_enabled():
            return batch
        experience_cfg = OmegaConf.select(self.config, "algorithm.collective_experience")
        required = {
            "uid",
            "traj_uid",
            "experience_event",
            "experience_task_id",
            "obs_images",
            "active_visual_messages_json",
        }
        missing = required.difference(batch.non_tensor_batch)
        if missing:
            raise ValueError(f"Collective-experience rollout batch is missing fields: {sorted(missing)}")
        compiler = CollectiveExperienceCompiler(
            near_duplicate_iou=float(experience_cfg.get("near_duplicate_iou", 0.9)),
            phash_distance=int(experience_cfg.get("phash_distance", 4)),
            semantic_deduper=None,
            max_intent_chars=int(experience_cfg.get("max_intent_chars", 240)),
        )
        projector = ExperienceViewProjector(
            artifact_budget=int(experience_cfg.get("teacher_artifact_budget", 4)),
            min_visual_novelty=float(experience_cfg.get("min_visual_novelty", 0.0)),
        )
        serializer = ExperienceSerializer(
            include_outcomes=True,
            compact_ids=bool(experience_cfg.get("compact_ids", True)),
            compact_graph=bool(experience_cfg.get("compact_graph_text", True)),
        )
        aligner = DecisionAligner()
        uids = np.asarray(batch.non_tensor_batch["uid"], dtype=object)
        trajectory_ids = np.asarray(batch.non_tensor_batch["traj_uid"], dtype=object)
        events = list(batch.non_tensor_batch["experience_event"])
        task_ids = np.asarray(batch.non_tensor_batch["experience_task_id"], dtype=object)
        episode_trajectory_valid = np.asarray(
            batch.non_tensor_batch.get("episode_trajectory_valid", np.ones(len(batch), dtype=bool)), dtype=bool
        )
        episode_returns = np.asarray(
            batch.non_tensor_batch.get("episode_rewards", np.zeros(len(batch), dtype=np.float32)), dtype=np.float32
        )
        policy_version = int(batch.meta_info.get("global_steps", self.global_steps))
        active_visual_cfg = OmegaConf.select(self.config, "env.active_visual", default={})
        tool_reward_coef = linear_decay_value(
            float(active_visual_cfg.get("tool_reward_start", 0.0)),
            float(active_visual_cfg.get("tool_reward_end", 0.0)),
            step=max(policy_version - 1, 0),
            decay_steps=int(active_visual_cfg.get("tool_reward_decay_steps", 0)),
        )
        group_events: dict[str, list[dict]] = defaultdict(list)
        group_task_ids: dict[str, set[str]] = defaultdict(set)
        routing_events: dict[int, dict[str, Any]] = {}
        for index, (uid, task_id, event) in enumerate(zip(uids, task_ids, events)):
            group_task_ids[str(uid)].add(str(task_id))
            if event is not None:
                event_copy = dict(event)
                group_events[str(uid)].append(event_copy)
            if isinstance(event, Mapping):
                routing_events[index] = dict(event)
        outcome_by_uid: dict[str, dict[str, Any]] = {}
        last_row_by_uid: dict[str, int] = {}
        for index, uid_raw in enumerate(uids):
            last_row_by_uid[str(uid_raw)] = index
        for uid in sorted({str(value) for value in uids}):
            raw_events = group_events.get(uid, [])
            row_indices = np.flatnonzero(uids == uid)
            rows_by_traj: dict[str, list[dict[str, Any]]] = defaultdict(list)
            valid_by_traj: dict[str, bool] = {}
            representative_by_traj: dict[str, int] = {}
            for row_index in row_indices:
                trajectory_id = str(trajectory_ids[int(row_index)])
                representative_by_traj.setdefault(trajectory_id, int(row_index))
                event = events[int(row_index)]
                if isinstance(event, Mapping):
                    rows_by_traj[trajectory_id].append({"experience_event": event})
                valid_by_traj[trajectory_id] = bool(episode_trajectory_valid[int(row_index)])
                rows_by_traj.setdefault(trajectory_id, [])
            for trajectory_id, trajectory_rows in rows_by_traj.items():
                row_index = representative_by_traj[trajectory_id]
                outcome = derive_trajectory_outcome(
                    trajectory_rows,
                    episode_return=float(episode_returns[row_index]),
                    trajectory_valid=valid_by_traj[trajectory_id],
                )
                outcome = dict(outcome)
                outcome_by_uid[trajectory_id] = outcome
                for current_index in row_indices:
                    if str(trajectory_ids[int(current_index)]) != trajectory_id:
                        continue
                    event = events[int(current_index)]
                    synthetic_kind = None
                    if outcome["status"] == "invalid" and (not isinstance(event, Mapping)):
                        synthetic_kind = "invalid_action"
                    elif outcome["status"] == "incomplete" and int(current_index) == max(
                        (i for i in row_indices if str(trajectory_ids[int(i)]) == trajectory_id)
                    ):
                        synthetic_kind = "incomplete"
                    if synthetic_kind is None:
                        continue
                    previous_step = int(event.get("step", -1)) if isinstance(event, Mapping) else -1
                    synthetic_event = {
                        "kind": synthetic_kind,
                        "task_id": str(task_ids[int(current_index)]),
                        "trajectory_id": trajectory_id,
                        "step": previous_step + 1,
                        "policy_version": policy_version,
                        "visible_artifact_ids": tuple(event.get("visible_artifact_ids", ("A0",)))
                        if isinstance(event, Mapping)
                        else ("A0",),
                        "failure_type": outcome.get("failure_type"),
                        "trajectory_outcome": dict(outcome),
                        "action": {"rationale": ""},
                    }
                    group_events[uid].append(synthetic_event)
                    routing_events[int(current_index)] = synthetic_event
        inconsistent_groups = {uid: sorted(values) for uid, values in group_task_ids.items() if len(values) != 1}
        if inconsistent_groups:
            raise ValueError(f"one rollout group must correspond to exactly one task: {inconsistent_groups}")
        teacher_message_payloads: list[str] = []
        teacher_target_images: list[list] = []
        teacher_experience_texts: list[str] = []
        teacher_experience_images: list[list] = []
        teacher_alignment_texts: list[str] = []
        repair_instructions: list[str] = []
        teacher_sources: list[object] = []
        teacher_mask = np.zeros(len(batch), dtype=bool)
        grpo_mask = np.zeros(len(batch), dtype=bool)
        repair_failure_types: dict[str, int] = defaultdict(int)
        experience_cache: dict[str, dict[str, Any]] = {}
        experience_row_indices: dict[str, int] = {}
        full_artifact_counts, selected_artifact_counts = ([], [])
        occurrence_counts, relation_counts = ([], [])
        for index, (uid, task_id) in enumerate(zip(uids, task_ids)):
            uid = str(uid)
            experience_row_indices.setdefault(uid, index)
            if uid not in experience_cache:
                assert compiler is not None and projector is not None
                if timing_raw is None:
                    experience = compiler.compile(
                        task_id=str(task_id), policy_version=policy_version, events=group_events[uid]
                    )
                else:
                    with _timer("cve_compile", timing_raw):
                        experience = compiler.compile(
                            task_id=str(task_id), policy_version=policy_version, events=group_events[uid]
                        )
                outcome_statuses = [
                    trajectory.outcome.status
                    for trajectory in experience.trajectories
                    if trajectory.outcome is not None
                ]
                association_types = [association.association_type for association in experience.associations]
                experience_cache[uid] = {
                    "experience": experience,
                    "renders": {},
                    "text": "",
                    "images": [],
                    "view": None,
                    "full_artifact_count": len(experience.artifacts),
                    "selected_artifact_count": 0,
                    "occurrence_count": len(experience.occurrences),
                    "relation_count": len(experience.relations),
                    "outcome_statuses": outcome_statuses,
                    "association_types": association_types,
                    "diagnostic_count": len(experience.diagnostics),
                }
            event = routing_events.get(index, events[index])
            outcome = outcome_by_uid.get(str(trajectory_ids[index]))
            if outcome is None:
                event_rows = [{"experience_event": event} for event in [event] if isinstance(event, Mapping)]
                outcome = derive_trajectory_outcome(
                    event_rows,
                    episode_return=float(episode_returns[index]),
                    trajectory_valid=bool(episode_trajectory_valid[index]),
                )
                outcome_by_uid[str(trajectory_ids[index])] = outcome
            status = str(outcome.get("status", "unknown"))
            failure_type = outcome.get("failure_type")
            route = training_route(outcome, self.config.algorithm.get("learning_rule", "hapi"))
            repair_row = route["repair"]
            grpo_mask[index] = route["grpo"]
            teacher_mask[index] = repair_row
            if repair_row:
                repair_failure_types[str(failure_type or status)] += 1
            repair_instructions.append(self._collective_repair_instruction(status, failure_type) if repair_row else "")
            cached = experience_cache[uid]
            if isinstance(event, Mapping):
                try:
                    event_key = (
                        str(event.get("trajectory_id", "")),
                        int(event.get("step", -1)),
                        str(event.get("kind", "")),
                    )
                except (TypeError, ValueError):
                    event_key = (str(index),)
            else:
                event_key = ("shared",)
            rendered = cached["renders"].get(event_key)
            if rendered is None:
                if timing_raw is None:
                    view = projector.project(cached["experience"], event)
                    experience_text, experience_images = serializer.render_view(cached["experience"], view)
                else:
                    with _timer("cve_project", timing_raw):
                        view = projector.project(cached["experience"], event)
                    with _timer("cve_serialize", timing_raw):
                        experience_text, experience_images = serializer.render_view(cached["experience"], view)
                rendered = (experience_text, experience_images, view)
                cached["renders"][event_key] = rendered
            experience_text, experience_images, view = rendered
            if not cached["text"]:
                cached["text"] = experience_text
                cached["images"] = experience_images
                cached["view"] = view
            alignment = aligner.align(cached["experience"], view, event)
            alignment_text = serializer.render_alignment(alignment, serializer.trajectory_aliases(cached["experience"]))
            full_artifact_count = cached["full_artifact_count"]
            selected_artifact_count = len(view.artifact_ids)
            occurrence_count = cached["occurrence_count"]
            relation_count = cached["relation_count"]
            target_images_raw = batch.non_tensor_batch["obs_images"][index]
            target_images = list(target_images_raw) if target_images_raw is not None else []
            teacher_message_payloads.append(str(batch.non_tensor_batch["active_visual_messages_json"][index]))
            teacher_target_images.append(target_images)
            teacher_experience_texts.append(experience_text)
            teacher_experience_images.append(list(experience_images))
            teacher_alignment_texts.append(alignment_text)
            teacher_sources.append(
                batch.non_tensor_batch["data_source"][index] if "data_source" in batch.non_tensor_batch else None
            )
            full_artifact_counts.append(full_artifact_count)
            selected_artifact_counts.append(selected_artifact_count)
            occurrence_counts.append(occurrence_count)
            relation_counts.append(relation_count)
        prompt_kwargs = dict(
            message_payloads=teacher_message_payloads,
            target_images=teacher_target_images,
            experience_texts=teacher_experience_texts,
            experience_images=teacher_experience_images,
            alignment_texts=teacher_alignment_texts,
            repair_instructions=repair_instructions,
            data_sources=teacher_sources,
            meta_info=deepcopy(batch.meta_info),
            max_length=int(experience_cfg.get("teacher_max_prompt_length", self.config.data.max_prompt_length)),
            reuse_image_preprocessing=bool(experience_cfg.get("reuse_image_preprocessing", False)),
        )
        if timing_raw is None:
            teacher_prompt = self.traj_collector.build_collective_experience_teacher_prompt_batch(**prompt_kwargs)
        else:
            with _timer("cve_prompt_preprocess", timing_raw):
                teacher_prompt = self.traj_collector.build_collective_experience_teacher_prompt_batch(**prompt_kwargs)
        responses = batch.batch["responses"]
        response_length = responses.size(1)
        response_mask = batch.batch["attention_mask"][:, -response_length:]
        teacher_input_ids, teacher_attention_mask, teacher_position_ids = append_identical_responses(
            prompt_input_ids=teacher_prompt.batch["input_ids"],
            prompt_attention_mask=teacher_prompt.batch["attention_mask"],
            prompt_position_ids=teacher_prompt.batch["position_ids"],
            responses=responses,
            response_mask=response_mask,
        )
        batch.batch["teacher_input_ids"] = teacher_input_ids
        batch.batch["teacher_attention_mask"] = teacher_attention_mask
        batch.batch["teacher_position_ids"] = teacher_position_ids
        batch.batch["collective_experience_teacher_mask"] = torch.as_tensor(
            teacher_mask, dtype=torch.bool, device=responses.device
        )
        batch.batch["collective_experience_grpo_mask"] = torch.as_tensor(
            grpo_mask, dtype=torch.bool, device=responses.device
        )
        batch.batch["collective_experience_repair_mask"] = torch.as_tensor(
            teacher_mask, dtype=torch.bool, device=responses.device
        )
        batch.non_tensor_batch["teacher_multi_modal_inputs"] = teacher_prompt.non_tensor_batch["multi_modal_inputs"]
        batch.non_tensor_batch.pop("experience_event", None)
        batch.non_tensor_batch.pop("experience_task_id", None)
        batch.non_tensor_batch.pop("obs_images", None)
        batch.non_tensor_batch.pop("active_visual_messages_json", None)
        batch.non_tensor_batch.pop("experience_ground_truth", None)
        metrics.update(
            {
                "cve/teacher_active_ratio": float(teacher_mask.mean()),
                "cve/grpo_success_only_ratio": float(grpo_mask.mean()),
                "cve/repair_active_ratio": float(teacher_mask.mean()),
                "cve/repair_failure_ratio": float(
                    repair_failure_types.get("wrong_after_crop", 0) + repair_failure_types.get("wrong_direct_answer", 0)
                )
                / max(1, len(batch)),
                "cve/repair_invalid_ratio": float(repair_failure_types.get("invalid_action", 0)) / max(1, len(batch)),
                "cve/repair_incomplete_ratio": float(repair_failure_types.get("no_valid_final_answer", 0))
                / max(1, len(batch)),
                "cve/full_artifacts_mean": float(np.mean(full_artifact_counts)),
                "cve/selected_artifacts_mean": float(np.mean(selected_artifact_counts)),
                "cve/occurrences_mean": float(np.mean(occurrence_counts)),
                "cve/relations_mean": float(np.mean(relation_counts)),
                "cve/shared_experiences": float(len(experience_cache)),
                "cve/decision_conditioned_projector": float(True),
                "cve/outcome_privileged_ratio": 1.0,
                "cve/tool_reward_coef": tool_reward_coef,
            }
        )
        if experience_cache:
            cached_experiences = list(experience_cache.values())
            total_outcomes = sum((len(cached["outcome_statuses"]) for cached in cached_experiences))
            total_artifacts = sum((len(cached["association_types"]) for cached in cached_experiences))
            status_count = lambda status: sum(
                (cached["outcome_statuses"].count(status) for cached in cached_experiences)
            )
            association_count = lambda status: sum(
                (cached["association_types"].count(status) for cached in cached_experiences)
            )
            metrics.update(
                {
                    "cve/success_trajectory_ratio": status_count("success") / total_outcomes if total_outcomes else 0.0,
                    "cve/failure_trajectory_ratio": status_count("failure") / total_outcomes if total_outcomes else 0.0,
                    "cve/invalid_trajectory_ratio": status_count("invalid") / total_outcomes if total_outcomes else 0.0,
                    "cve/incomplete_trajectory_ratio": status_count("incomplete") / total_outcomes
                    if total_outcomes
                    else 0.0,
                    "cve/success_observation_ratio": association_count("observed_in_success") / total_artifacts
                    if total_artifacts
                    else 0.0,
                    "cve/mixed_outcome_observation_ratio": association_count("mixed_outcome_observation")
                    / total_artifacts
                    if total_artifacts
                    else 0.0,
                    "cve/diagnostics_mean": float(
                        np.mean([cached["diagnostic_count"] for cached in cached_experiences])
                    ),
                }
            )
        if "tool_callings" in batch.non_tensor_batch:
            _, unique_trajectory_indices = np.unique(trajectory_ids, return_index=True)
            first_crop_ratio = float(
                (
                    np.asarray(batch.non_tensor_batch["tool_callings"], dtype=np.float32)[unique_trajectory_indices] > 0
                ).mean()
            )
            metrics["cve/tool_reward_granted_ratio"] = first_crop_ratio
            metrics["cve/tool_reward_mean"] = tool_reward_coef * first_crop_ratio
        return batch

    def _validate_config(self):
        config = self.config
        if config.algorithm.adv_estimator != "grpo":
            raise ValueError("VISTA requires GRPO with HAPI routing")
        if config.actor_rollout_ref.actor.opd_loss_type != "jsd":
            raise ValueError("VISTA requires JSD distillation")
        if not 0 <= config.actor_rollout_ref.actor.opd_jsd_alpha <= 1:
            raise ValueError("Distillation mixture weight must be in [0, 1]")
        if config.actor_rollout_ref.actor.get("opd_distribution", "full") == "topk" and not config.actor_rollout_ref.actor.opd_jsd_add_tail:
            raise ValueError("The distributed JSD approximation requires its tail bucket")
        if config.actor_rollout_ref.actor.policy_loss.loss_mode != "vanilla":
            raise ValueError("VISTA uses the clipped GRPO policy objective")
        if not config.algorithm.collective_experience.enabled:
            raise ValueError("VISTA requires collective experience")
        actor = config.actor_rollout_ref.actor
        if actor.get("opd_distribution", "full") not in {"full", "topk"}:
            raise ValueError("opd_distribution must be full or topk")
        if actor.get("group_normalization", False):
            if actor.ulysses_sequence_parallel_size != 1 or actor.ppo_epochs != 1:
                raise ValueError("Paper group normalization requires sequence parallel size 1 and one PPO epoch")
            if actor.entropy_coeff or actor.use_kl_loss:
                raise ValueError("Paper objective uses neither entropy bonus nor reference KL loss")
        if actor.collective_experience.get("teacher_update", "ema") not in {"ema", "frozen"}:
            raise ValueError("teacher_update must be ema or frozen")
        training_route({"status": "success"}, config.algorithm.get("learning_rule", "hapi"))
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        experience_algorithm_enabled = bool(
            OmegaConf.select(config, "algorithm.collective_experience.enabled", default=False)
        )
        experience_actor_enabled = bool(
            OmegaConf.select(config, "actor_rollout_ref.actor.collective_experience.enabled", default=False)
        )
        if experience_algorithm_enabled != experience_actor_enabled:
            raise ValueError(
                "algorithm.collective_experience.enabled and actor_rollout_ref.actor.collective_experience.enabled must match"
            )
        if experience_algorithm_enabled:
            if int(OmegaConf.select(config, "algorithm.collective_experience.max_intent_chars", default=240)) < 0:
                raise ValueError("max_intent_chars must be non-negative")
            min_visual_novelty = float(
                OmegaConf.select(config, "algorithm.collective_experience.min_visual_novelty", default=0.0) or 0.0
            )
            if not 0.0 <= min_visual_novelty <= 1.0:
                raise ValueError("min_visual_novelty must be in [0, 1]")
            if str(config.env.env_name).lower() != "active_visual/crop":
                raise ValueError("collective-experience distillation requires env.env_name=active_visual/crop")
            if config.actor_rollout_ref.actor.strategy != "fsdp":
                raise ValueError("EMA checkpointing currently supports actor.strategy=fsdp")
            if int(config.env.rollout.n) < 1:
                raise ValueError("collective experience requires env.rollout.n >= 1")
            if float(config.actor_rollout_ref.actor.opd_loss_coef or 0.0) < 0:
                raise ValueError("actor.opd_loss_coef must be non-negative")
            if int(config.actor_rollout_ref.actor.opd_loss_warmup_steps or 0) < 0:
                raise ValueError("actor.opd_loss_warmup_steps must be non-negative")
            if int(config.algorithm.collective_experience.teacher_artifact_budget) < 1:
                raise ValueError("teacher_artifact_budget must be at least 1")
            if int(config.algorithm.collective_experience.teacher_max_prompt_length) < 1:
                raise ValueError("algorithm.collective_experience.teacher_max_prompt_length must be at least 1")
        env_rollout_n = int(OmegaConf.select(config, "env.rollout.n", default=1) or 1)
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n * env_rollout_n
        assert real_train_batch_size % n_gpus == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        )

        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }
            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"
                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )
                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'"
                        + "is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )
            if self.use_reference_policy:
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )
        if self.use_critic and (not config.critic.use_dynamic_bsz):
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )
        if config.reward_model.enable and (not config.reward_model.use_dynamic_bsz):
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus
        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"
        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")
        if self.use_critic and (not config.critic.use_dynamic_bsz):
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )
        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )
        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, (
                "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            )
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], (
                "only GRPO is tested for multi-turn with tool"
            )
        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = (train_dataset, val_dataset)
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}"
        )
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(
        self,
        inputs,
        outputs,
        scores,
        reward_extra_infos_dict,
        dump_path,
        rollout_extra_infos_dict=None,
        max_entries=None,
    ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        def _json_safe(value):
            if isinstance(value, np.generic):
                return value.item()
            if torch.is_tensor(value):
                return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
            return value

        def _sort_key(entry):
            try:
                return (int(entry.get("sample_id", 0)), int(entry.get("rollout_id", 0)), int(entry.get("step_num", 0)))
            except (TypeError, ValueError):
                step_id = str(entry.get("step_id", "0_0_0")).split("_")
                padded = (step_id + ["0", "0", "0"])[:3]
                try:
                    return tuple((int(part) for part in padded))
                except ValueError:
                    return (0, 0, 0)

        n = len(inputs)
        base_data = {"input": inputs, "output": outputs, "score": scores, "step": [self.global_steps] * n}
        rollout_extra_infos_dict = rollout_extra_infos_dict or {}
        for k, v in rollout_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v
        if all((key in base_data for key in ("sample_id", "rollout_id", "step_num"))) and "step_id" not in base_data:
            base_data["step_id"] = [
                f"{int(base_data['sample_id'][i])}_{int(base_data['rollout_id'][i])}_{int(base_data['step_num'][i])}"
                for i in range(n)
            ]
        entries = []
        for i in range(n):
            entries.append({k: _json_safe(v[i]) for k, v in base_data.items()})
        entries.sort(key=_sort_key)
        if max_entries is not None:
            max_entries = int(max_entries)
            if max_entries < 1:
                raise ValueError("rollout_data_max_samples_per_step must be positive or null")
            entries = entries[:max_entries]
        with open(filename, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Dumped generations to {filename}")

    @staticmethod
    def _metric_value_to_json(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.numel() == 1:
                return tensor.item()
            return tensor.tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): RayPPOTrainer._metric_value_to_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [RayPPOTrainer._metric_value_to_json(item) for item in value]
        return str(value)

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return
        import numpy as np

        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        samples = samples[:generations_to_log]
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop
            )
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                gen_batch=test_gen_batch, actor_rollout_wg=self.actor_rollout_wg, envs=self.val_envs, is_train=False
            )
            print("validation generation end")
            del test_batch
            test_batch = test_output_gen_batch
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch["tool_callings"])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch["traj_uid"])
            for k in test_batch.non_tensor_batch.keys():
                if "success_rate" in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], (
                            f"not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}"
                        )
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]
        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())
        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/{data_source}/test_score"] = np.mean(rewards)
        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f"val/{data_source}/tool_call_count/mean"] = np.mean(tool_calls)
        for k, v in success_rate.items():
            metric_dict[f"val/{k}"] = v
        subtask_success_rate_mean = compute_subtask_success_rate_mean(success_rate)
        if subtask_success_rate_mean is not None:
            metric_dict["val/subtask_success_rate_mean"] = subtask_success_rate_mean
        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()
        if self.use_reference_policy and (not self.ref_in_actor):
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref, worker_group=self.actor_rollout_wg
            )

    def _save_checkpoint(self):
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")
        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        self._load_checkpoint()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"val_metrics={val_metrics!r}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                input_group_count = len(batch)
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)
                is_last_step = self.global_steps >= self.total_training_steps
                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch.meta_info["global_steps"] = int(self.global_steps)
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch, actor_rollout_wg=self.actor_rollout_wg, envs=self.envs, is_train=True
                        )
                    del batch
                    batch = gen_batch_output
                    if self._is_collective_experience_enabled():
                        with _timer("collective_experience_teacher_context", timing_raw):
                            batch.meta_info["global_steps"] = int(self.global_steps)
                            batch = self._prepare_collective_experience_teacher_inputs(batch, metrics, timing_raw)
                    if self.config.actor_rollout_ref.actor.get("group_normalization", False):
                        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                        divisor = world_size
                        for worker_config, key in (
                            (self.config.actor_rollout_ref.rollout, "log_prob_micro_batch_size_per_gpu"),
                            (self.config.actor_rollout_ref.ref, "log_prob_micro_batch_size_per_gpu"),
                            (self.config.actor_rollout_ref.actor, "ppo_micro_batch_size_per_gpu"),
                        ):
                            divisor = math.lcm(divisor, world_size * int(worker_config.get(key) or 1))
                        batch = prepare_group_batch(batch, world_size, divisor, input_group_count)
                    else:
                        batch = adjust_batch(self.config, batch, track_source_indices=False)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob_non_tensor_keys = (
                            ["multi_modal_inputs"] if "multi_modal_inputs" in batch.non_tensor_batch else []
                        )
                        old_log_prob_batch = batch.select(
                            batch_keys=["responses", "input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=old_log_prob_non_tensor_keys,
                        )
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(old_log_prob_batch)
                        del old_log_prob_batch
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                        )
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        if "rollout_log_probs" in batch.batch.keys():
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]
                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )
                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor
                        print(f"list(reward_extra_infos_dict.keys())={list(reward_extra_infos_dict.keys())!r}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        if self.config.actor_rollout_ref.actor.get("use_invalid_action_penalty", True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        episode_skill_teacher_advantage_w = 0.0
                        step_skill_teacher_advantage_w = 0.0
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    rollout_data_freq = int(self.config.trainer.get("rollout_data_freq", 1))
                    if rollout_data_freq < 1:
                        raise ValueError("trainer.rollout_data_freq must be positive")
                    if rollout_data_dir and (is_last_step or self.global_steps % rollout_data_freq == 0):
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            rollout_extra_infos_dict = {
                                key: batch.non_tensor_batch[key]
                                for key in ("sample_id", "rollout_id", "step_num", "step_id", "uid", "traj_uid")
                                if key in batch.non_tensor_batch
                            }
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                                rollout_extra_infos_dict=rollout_extra_infos_dict,
                                max_entries=self.config.trainer.get("rollout_data_max_samples_per_step", None),
                            )
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
