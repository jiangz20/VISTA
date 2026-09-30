"""OPID-compatible environment manager for crop-based VQA trajectories."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .crop import CropRuntime, parse_action, parse_final_answer
from .prompts import (
    assistant_tool_message,
    build_initial_messages,
    tool_error_message,
    tool_image_message,
)
from .schedule import linear_decay_value


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def _normalize_answer(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"\b([A-Za-z])\b", text)
    return match.group(1).upper() if match else text.casefold()


@dataclass
class EpisodeState:
    task_id: str
    question: str
    ground_truth: str
    runtime: CropRuntime
    max_crop_calls: int
    min_crop_calls: int
    crop_calls: int = 0
    trajectory_valid: bool = True
    done: bool = False
    won: float = 0.0
    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages = build_initial_messages(
                self.question,
                self.runtime.artifacts["A0"].image,
            )

    def observation(self) -> tuple[list[dict[str, Any]], str, list[Any]]:
        images = [self.runtime.artifacts[artifact_id].image for artifact_id in self.runtime.artifact_order]
        audit_text = (
            f"Question:\n{self.question.replace('<image>', '').strip()}\n\n"
            f"Images currently in the native Qwen conversation: {len(images)}"
        )
        return self.messages, audit_text, images


class ActiveVisualEnvironmentManager:
    """A vectorized in-process environment matching OPID's manager protocol."""

    def __init__(self, config, is_train: bool = True):
        self.config = config
        self.is_train = is_train
        active_visual_config = config.env.get("active_visual", {})
        self.max_crop_calls = int(active_visual_config.get("max_crop_calls", 4))
        self.min_crop_calls = int(active_visual_config.get("min_crop_calls", 0))
        self.min_dimension = int(active_visual_config.get("min_crop_dimension", 32))
        self.invalid_action_reward = float(active_visual_config.get("invalid_action_reward", 0.0))
        self.cropped_wrong_reward = float(active_visual_config.get("cropped_wrong_reward", 0.0))
        self.tool_reward_start = float(active_visual_config.get("tool_reward_start", 0.0))
        self.tool_reward_end = float(active_visual_config.get("tool_reward_end", 0.0))
        self.tool_reward_decay_steps = int(active_visual_config.get("tool_reward_decay_steps", 0))
        # Validate the schedule eagerly so a malformed formal run fails before
        # producing any rollout or checkpoint.
        linear_decay_value(
            self.tool_reward_start,
            self.tool_reward_end,
            step=0,
            decay_steps=self.tool_reward_decay_steps,
        )
        self.global_step = 0
        self.states: list[EpisodeState] = []

    def set_global_step(self, global_step: int) -> None:
        """Set the trainer update used by the tool-reward schedule."""
        self.global_step = max(0, int(global_step))

    def current_tool_reward(self) -> float:
        if not self.is_train:
            return 0.0
        return linear_decay_value(
            self.tool_reward_start,
            self.tool_reward_end,
            # Trainer updates are one-indexed: update 1 should receive the
            # configured starting value, update decay_steps + 1 the end value.
            step=max(self.global_step - 1, 0),
            decay_steps=self.tool_reward_decay_steps,
        )

    def reset(self, kwargs):
        if kwargs is None:
            raise ValueError("active visual exploration requires per-sample env_kwargs")
        rows = list(kwargs)
        self.states = []
        root_runtimes: dict[str, CropRuntime] = {}
        for index, raw_row in enumerate(rows):
            row = _plain(raw_row)
            if isinstance(row, list) and len(row) == 1 and isinstance(row[0], dict):
                row = row[0]
            if not isinstance(row, dict):
                raise TypeError(f"env_kwargs[{index}] must be a mapping, got {type(row)!r}")
            image_path = row.get("image_path")
            if not image_path:
                raise ValueError(f"env_kwargs[{index}] is missing image_path")
            image_path = str(image_path)
            root_runtime = root_runtimes.get(image_path)
            if root_runtime is None:
                root_runtime = CropRuntime(image_path, min_dimension=self.min_dimension)
                root_runtimes[image_path] = root_runtime
            state = EpisodeState(
                task_id=str(row.get("task_id", uuid.uuid4().hex)),
                question=str(row.get("question", "")),
                ground_truth=str(row.get("ground_truth", "")),
                runtime=root_runtime.fork_root(),
                max_crop_calls=self.max_crop_calls,
                min_crop_calls=self.min_crop_calls,
            )
            self.states.append(state)
        observations = self._observations()
        infos = [
            {
                "task_id": state.task_id,
                "ground_truth": state.ground_truth,
                "won": 0.0,
            }
            for state in self.states
        ]
        return observations, infos

    def _observations(self):
        messages, texts, images = [], [], []
        for state in self.states:
            state_messages, text, state_images = state.observation()
            messages.append(state_messages)
            texts.append(text)
            images.append(state_images)
        return {
            "messages": messages,
            "text": texts,
            "text_base": [state.question for state in self.states],
            "image": images,
            "anchor": [None] * len(texts),
        }

    def step(self, text_actions: list[str]):
        rewards = np.zeros(len(self.states), dtype=np.float32)
        dones = np.zeros(len(self.states), dtype=bool)
        infos = []
        for index, (state, text) in enumerate(zip(self.states, text_actions)):
            visible_artifact_ids = tuple(state.runtime.artifact_order)
            info = {
                "task_id": state.task_id,
                "ground_truth": state.ground_truth,
                "won": state.won,
                "is_action_valid": True,
                "tool_calling": 0.0,
                "tool_reward": 0.0,
                "tool_reward_coef": self.current_tool_reward(),
                "experience_event": None,
            }
            if state.done:
                dones[index] = True
                infos.append(info)
                continue
            tool_call_intent = bool(re.search(r"<\s*/?\s*tool_call\b", str(text), re.IGNORECASE))
            parse_error = None
            try:
                action = parse_action(text, state.runtime.artifact_order)
            except (ValueError, TypeError, KeyError) as exc:
                action = None
                parse_error = exc
                info["is_action_valid"] = False
                state.messages.extend(
                    [
                        {"role": "assistant", "content": str(text)},
                        {
                            "role": "user",
                            "content": (
                                f"Invalid tool call: {exc}. Use image_zoom_in_tool with "
                                "bbox_2d, label, and img_idx, or provide a boxed answer."
                            ),
                        },
                    ]
                )
            if action is not None:
                source_index = state.runtime.artifact_order.index(action.source_artifact_id)
                assistant_message = assistant_tool_message(text, action, source_index)
                if state.crop_calls >= state.max_crop_calls:
                    info["is_action_valid"] = False
                    state.messages.extend(
                        [
                            assistant_message,
                            tool_error_message("Crop budget exhausted; provide a boxed final answer."),
                        ]
                    )
                else:
                    try:
                        first_successful_crop = state.crop_calls == 0
                        artifact = state.runtime.crop(action)
                        state.crop_calls += 1
                        info["tool_calling"] = 1.0
                        if first_successful_crop:
                            tool_reward = self.current_tool_reward()
                            rewards[index] += tool_reward
                            info["tool_reward"] = tool_reward
                        state.messages.extend([assistant_message, tool_image_message(artifact.image)])
                        info["experience_event"] = {
                            "kind": "crop",
                            "task_id": state.task_id,
                            "step": None,
                            "trajectory_id": None,
                            "policy_version": None,
                            "visible_artifact_ids": visible_artifact_ids,
                            "action": {
                                "source_artifact_id": action.source_artifact_id,
                                "bbox_2d": tuple(action.bbox_2d),
                                "label": action.label,
                                # Preserve the student's stated inspection intent.
                                # The experience compiler sanitizes and length-limits
                                # this untrusted text before teacher serialization.
                                "rationale": assistant_message.get("content", ""),
                            },
                            "artifact": artifact.metadata(),
                            "artifact_image": artifact.image,
                        }
                    except ValueError as exc:
                        info["is_action_valid"] = False
                        state.messages.extend([assistant_message, tool_error_message(str(exc))])
            else:
                # Never let a malformed tool call fall through to a boxed answer in
                # the same response. That previously allowed an unclosed tool block
                # containing a boxed answer to receive the outcome reward.
                if tool_call_intent:
                    if parse_error is None:
                        info["is_action_valid"] = False
                        state.messages.extend(
                            [
                                {"role": "assistant", "content": str(text)},
                                {
                                    "role": "user",
                                    "content": (
                                        "Invalid tool call: expected exactly one complete "
                                        "<tool_call>...</tool_call> block containing one "
                                        "valid image_zoom_in_tool call."
                                    ),
                                },
                            ]
                        )
                    answer = None
                else:
                    answer = parse_final_answer(text)
                if answer is not None:
                    answer_correct = float(_normalize_answer(answer) == _normalize_answer(state.ground_truth))
                    min_crop_satisfied = state.crop_calls >= state.min_crop_calls
                    successful_crop = state.crop_calls > 0
                    if not min_crop_satisfied:
                        # A boxed answer is syntactically valid but violates the
                        # configured action protocol. Keep the episode alive so the
                        # policy can recover with a crop on its next turn; mark the
                        # trajectory invalid so the premature answer cannot later
                        # receive outcome reward.
                        info["is_action_valid"] = False
                        info["answer_correct"] = answer_correct
                        info["min_crop_satisfied"] = False
                        info["successful_crop"] = successful_crop
                        info["answer_reward_scale"] = 0.0
                        info["answer_reward"] = 0.0
                        info["experience_event"] = {
                            "kind": "premature_answer",
                            "task_id": state.task_id,
                            "answer_correct": bool(answer_correct),
                            "successful_crop": successful_crop,
                            "min_crop_satisfied": False,
                            "visible_artifact_ids": visible_artifact_ids,
                        }
                        remaining = state.min_crop_calls - state.crop_calls
                        state.messages.extend(
                            [
                                {"role": "assistant", "content": str(text)},
                                {
                                    "role": "user",
                                    "content": (
                                        "The final answer was rejected because at least "
                                        f"{remaining} more successful image_zoom_in_tool "
                                        "call is required. Do not answer again yet; call "
                                        "image_zoom_in_tool now and end after the tool call."
                                    ),
                                },
                            ]
                        )
                        min_crop_satisfied = False
                    else:
                        state.done = True
                    # Correctness reward is gated by at least the requested
                    # number of successful crops. A valid trajectory that
                    # cropped successfully but answered incorrectly receives a
                    # small shaping reward to keep tool exploration alive.
                    answer_reward_scale = 1.0 if min_crop_satisfied else 0.0
                    state.won = answer_correct * answer_reward_scale * float(state.trajectory_valid)
                    if state.trajectory_valid:
                        if state.won:
                            rewards[index] = state.won
                        elif successful_crop and not answer_correct:
                            rewards[index] = self.cropped_wrong_reward
                    if min_crop_satisfied:
                        info["won"] = state.won
                        info["answer_correct"] = answer_correct
                        info["min_crop_satisfied"] = True
                        info["successful_crop"] = successful_crop
                        info["answer_reward_scale"] = answer_reward_scale
                        info["answer_reward"] = float(rewards[index])
                        info["experience_event"] = {
                            "kind": "final",
                            "task_id": state.task_id,
                            # Store only the parsed answer payload, not the full
                            # reasoning response.  The collective-experience
                            # serializer may expose this to peer trajectories.
                            "final_answer": str(answer),
                            "answer_correct": bool(answer_correct),
                            "successful_crop": successful_crop,
                            "min_crop_satisfied": True,
                            "visible_artifact_ids": visible_artifact_ids,
                        }
                elif info["is_action_valid"]:
                    info["is_action_valid"] = False
                    correction = "No valid image_zoom_in_tool call or boxed final answer was found."
                    state.messages.extend(
                        [
                            {"role": "assistant", "content": str(text)},
                            {"role": "user", "content": correction},
                        ]
                    )
            # Invalid actions receive the configured reward (zero by default). A valid correct answer
            # receives 1.0 only after the requested crop budget; a valid
            # crop-then-wrong trajectory receives the configured shaping
            # reward, and answering before the required crop is rejected.
            # Apply the invalid score only on the first invalid action so
            # repeated errors do not stack.
            if not info["is_action_valid"] and state.trajectory_valid:
                state.trajectory_valid = False
                rewards[index] = self.invalid_action_reward
            info["trajectory_valid"] = state.trajectory_valid
            dones[index] = state.done
            infos.append(info)
        return self._observations(), rewards, dones, infos

    def success_evaluator(self, **kwargs):
        values = np.asarray([state.won for state in self.states], dtype=np.float32)
        return {"success_rate": values}

    def close(self):
        self.states.clear()
