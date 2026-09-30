"""Exercise the actual verl driver and teacher-input path on CPU."""

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from vista.prompts import build_initial_messages, serialize_messages
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage
from test_teacher_prompt import make_collector, fake_rope_index


def test_final_profile_and_trainer_validation():
    cfg = OmegaConf.load(Path(__file__).parents[1] / "verl/trainer/config/ppo_trainer.yaml")
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = cfg
    trainer.global_steps = 1
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer._validate_config()
    assert cfg.env.rollout.n == 8
    assert cfg.algorithm.collective_experience.teacher_artifact_budget == 4
    assert cfg.actor_rollout_ref.actor.opd_loss_coef == 0.1
    assert cfg.actor_rollout_ref.actor.opd_topk == 64


def test_driver_routes_complete_responses_without_answer_injection(monkeypatch):
    monkeypatch.setattr("verl.models.transformers.qwen2_vl.get_rope_index", fake_rope_index)
    cfg = OmegaConf.load(Path(__file__).parents[1] / "verl/trainer/config/ppo_trainer.yaml")
    cfg.algorithm.collective_experience.teacher_max_prompt_length = 4096
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = cfg
    trainer.global_steps = 1
    trainer.traj_collector = make_collector()
    trainer.tokenizer = trainer.traj_collector.tokenizer
    image = Image.new("RGB", (32, 32))
    messages = build_initial_messages("Choose one option", image)
    events = [
        {
            "kind": "final",
            "step": 0,
            "trajectory_id": "good",
            "answer_correct": True,
            "final_answer": "REFERENCE_SECRET",
            "visible_artifact_ids": ("A0",),
        },
        {
            "kind": "final",
            "step": 0,
            "trajectory_id": "bad",
            "answer_correct": False,
            "final_answer": "PEER_SECRET",
            "visible_artifact_ids": ("A0",),
        },
    ]

    def objects(rows):
        result = np.empty(len(rows), dtype=object)
        result[:] = rows
        return result

    responses = torch.tensor([[7, 8, 0], [9, 10, 11]])
    batch = DataProto.from_dict(
        tensors={"responses": responses.clone(), "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])},
        non_tensors={
            "uid": objects(["group", "group"]),
            "traj_uid": objects(["good", "bad"]),
            "experience_event": objects(events),
            "experience_task_id": objects(["task", "task"]),
            "obs_images": objects([[image], [image]]),
            "active_visual_messages_json": objects([serialize_messages(messages)] * 2),
            "episode_trajectory_valid": np.array([True, True]),
            "episode_rewards": np.array([1.0, 0.0]),
            "data_source": objects(["vista_vqa"] * 2),
        },
        meta_info={"global_steps": 1},
    )
    captured = {}
    original = trainer.traj_collector.build_collective_experience_teacher_prompt_batch

    def capture(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    trainer.traj_collector.build_collective_experience_teacher_prompt_batch = capture
    result = trainer._prepare_collective_experience_teacher_inputs(batch, {})
    torch.testing.assert_close(result.batch["responses"], responses)
    torch.testing.assert_close(result.batch["teacher_input_ids"][:, -3:], responses)
    assert result.batch["collective_experience_grpo_mask"].tolist() == [True, False]
    assert result.batch["collective_experience_teacher_mask"].tolist() == [False, True]
    assert "ground_truths" not in captured
    assert "REFERENCE_SECRET" not in str(captured)
    assert "PEER_SECRET" not in str(captured)
    assert captured["repair_instructions"][1]


def test_driver_advantages_use_one_outcome_per_trajectory():
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.ones(3, 2, dtype=torch.long),
            "response_mask": torch.ones(3, 2),
            "token_level_rewards": torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]]),
            "collective_experience_grpo_mask": torch.tensor([True, True, False]),
        },
        non_tensors={"uid": np.array(["x"] * 3, dtype=object), "traj_uid": np.array(["a", "a", "b"], dtype=object)},
    )
    out = compute_advantage(batch, "grpo", norm_adv_by_std_in_grpo=False)
    torch.testing.assert_close(out.batch["advantages"], torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]]))


def test_pipeline_has_no_removed_algorithm_references():
    path = Path(__file__).parents[1] / "verl/trainer/ppo/ray_trainer.py"
    source = path.read_text()
    assert "AdvantageEstimator.OPID" not in source
    assert "from opid" not in source and "from gigpo" not in source
