from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vista.environment import ActiveVisualEnvironmentManager
from vista.prompts import THINK_WITH_IMAGES_SYSTEM_PROMPT


class EnvConfig(dict):
    __getattr__ = dict.__getitem__


def test_environment_crop_then_answer(tmp_path):
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(image_path)
    config = SimpleNamespace(
        env=EnvConfig(active_visual={"max_crop_calls": 2, "min_crop_calls": 1, "min_crop_dimension": 8})
    )
    env = ActiveVisualEnvironmentManager(config)
    obs, reset_infos = env.reset(
        [{"task_id": "sample", "image_path": str(image_path), "question": "Choose A", "ground_truth": "A"}]
    )
    assert reset_infos[0]["ground_truth"] == "A"
    assert len(obs["image"][0]) == 1
    assert [message["role"] for message in obs["messages"][0]] == ["system", "user"]
    assert obs["messages"][0][0]["content"] == THINK_WITH_IMAGES_SYSTEM_PROMPT
    opening_user_content = str(obs["messages"][0][1]["content"])
    assert "[MANDATORY FIRST TURN]" not in opening_user_content
    crop = (
        'I need a closer look. <tool_call>{"name":"image_zoom_in_tool",'
        '"arguments":{"bbox_2d":[0,0,500,500],"label":"upper left",'
        '"img_idx":0}}</tool_call>'
    )
    obs, rewards, dones, infos = env.step([crop])
    assert infos[0]["ground_truth"] == "A"
    assert not dones[0] and rewards[0] == 0
    assert len(obs["image"][0]) == 2
    assert [message["role"] for message in obs["messages"][0]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    tool_call = obs["messages"][0][2]["tool_calls"][0]["function"]
    assert tool_call["name"] == "image_zoom_in_tool"
    assert tool_call["arguments"]["img_idx"] == 0
    assert obs["messages"][0][3]["content"][0]["type"] == "image"
    assert infos[0]["experience_event"]["artifact"]["parent_artifact_id"] == "A0"
    assert infos[0]["experience_event"]["visible_artifact_ids"] == ("A0",)
    _, rewards, dones, infos = env.step([r"\boxed{A}"])
    assert dones[0] and rewards[0] == 1 and infos[0]["won"] == 1
    assert infos[0]["experience_event"]["successful_crop"]
    assert infos[0]["experience_event"]["min_crop_satisfied"]
    assert infos[0]["experience_event"]["answer_correct"]
    assert infos[0]["experience_event"]["final_answer"] == "A"
    assert infos[0]["experience_event"]["visible_artifact_ids"] == ("A0", "A1")


def test_environment_accepts_qwen35_xml_crop_then_answer(tmp_path):
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(image_path)
    config = SimpleNamespace(
        env=EnvConfig(active_visual={"max_crop_calls": 2, "min_crop_calls": 1, "min_crop_dimension": 8})
    )
    env = ActiveVisualEnvironmentManager(config)
    env.reset(
        [
            {
                "task_id": "sample",
                "image_path": str(image_path),
                "question": "Choose A",
                "ground_truth": "A",
            }
        ]
    )
    crop = """
    I need a closer look at the upper-left detail.
    <tool_call>
    <function=image_zoom_in_tool>
    <parameter=label>
    upper left
    </parameter>
    <parameter=bbox_2d>
    [0, 0, 500, 500]
    </parameter>
    <parameter=img_idx>
    0
    </parameter>
    </function>
    </tool_call>
    """

    obs, crop_rewards, crop_dones, crop_infos = env.step([crop])

    assert not crop_dones[0] and crop_rewards[0] == 0
    assert crop_infos[0]["is_action_valid"]
    assert crop_infos[0]["tool_calling"] == 1
    assistant_message = obs["messages"][0][2]
    assert assistant_message["content"] == "I need a closer look at the upper-left detail."
    assert "<tool_call>" not in assistant_message["content"]
    tool_call = assistant_message["tool_calls"][0]["function"]
    assert tool_call["name"] == "image_zoom_in_tool"
    assert tool_call["arguments"] == {
        "bbox_2d": [0.0, 0.0, 500.0, 500.0],
        "label": "upper left",
        "img_idx": 0,
    }

    _, rewards, dones, infos = env.step([r"\boxed{A}"])

    assert dones[0]
    assert rewards[0] == 1
    assert infos[0]["won"] == 1


def test_environment_upscales_small_crop_without_invalidating_trajectory(tmp_path):
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(image_path)
    config = SimpleNamespace(
        env=EnvConfig(
            active_visual={
                "max_crop_calls": 2,
                "min_crop_calls": 1,
                "min_crop_dimension": 32,
            }
        )
    )
    env = ActiveVisualEnvironmentManager(config)
    env.reset(
        [
            {
                "task_id": "sample",
                "image_path": str(image_path),
                "question": "Choose A",
                "ground_truth": "A",
            }
        ]
    )
    small_crop = (
        '<tool_call>{"name":"image_zoom_in_tool",'
        '"arguments":{"bbox_2d":[0,0,10,10],"label":"tiny detail",'
        '"img_idx":0}}</tool_call>'
    )

    obs, crop_rewards, crop_dones, crop_infos = env.step([small_crop])

    assert not crop_dones[0]
    assert crop_rewards[0] == 0
    assert crop_infos[0]["is_action_valid"]
    assert crop_infos[0]["trajectory_valid"]
    assert crop_infos[0]["tool_calling"] == 1
    assert obs["image"][0][-1].size == (32, 32)

    _, rewards, dones, infos = env.step([r"\boxed{A}"])

    assert dones[0]
    assert rewards[0] == 1
    assert infos[0]["won"] == 1
    assert infos[0]["trajectory_valid"]


def test_environment_pads_extreme_strip_crop_for_qwen_processor(tmp_path):
    image_path = tmp_path / "tall.png"
    Image.fromarray(np.zeros((1237, 100, 3), dtype=np.uint8)).save(image_path)
    config = SimpleNamespace(
        env=EnvConfig(
            active_visual={
                "max_crop_calls": 2,
                "min_crop_calls": 1,
                "min_crop_dimension": 32,
            }
        )
    )
    env = ActiveVisualEnvironmentManager(config)
    env.reset(
        [
            {
                "task_id": "sample",
                "image_path": str(image_path),
                "question": "Choose A",
                "ground_truth": "A",
            }
        ]
    )
    strip_crop = (
        '<tool_call>{"name":"image_zoom_in_tool",'
        '"arguments":{"bbox_2d":[0,0,10,1000],"label":"thin strip",'
        '"img_idx":0}}</tool_call>'
    )

    obs, crop_rewards, crop_dones, crop_infos = env.step([strip_crop])

    crop_image = obs["image"][0][-1]
    width, height = crop_image.size
    assert not crop_dones[0]
    assert crop_rewards[0] == 0
    assert crop_infos[0]["is_action_valid"]
    assert crop_infos[0]["trajectory_valid"]
    assert min(width, height) >= 32
    assert max(width / height, height / width) < 200


def test_environment_crop_then_wrong_answer_gets_shaping_reward(tmp_path):
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(image_path)
    config = SimpleNamespace(
        env=EnvConfig(
            active_visual={
                "max_crop_calls": 2,
                "min_crop_calls": 1,
                "min_crop_dimension": 8,
                "cropped_wrong_reward": 0.2,
            }
        )
    )
    env = ActiveVisualEnvironmentManager(config)
    env.reset(
        [
            {
                "task_id": "sample",
                "image_path": str(image_path),
                "question": "Choose A",
                "ground_truth": "A",
            }
        ]
    )
    crop = (
        '<tool_call>{"name":"image_zoom_in_tool",'
        '"arguments":{"bbox_2d":[0,0,500,500],"label":"detail",'
        '"img_idx":0}}</tool_call>'
    )

    _, crop_rewards, crop_dones, _ = env.step([crop])
    _, rewards, dones, infos = env.step([r"\boxed{B}"])

    assert not crop_dones[0] and crop_rewards[0] == 0
    assert dones[0]
    assert rewards[0] == np.float32(0.2)
    assert infos[0]["won"] == 0
    assert infos[0]["answer_correct"] == 0
    assert infos[0]["successful_crop"]
    assert infos[0]["trajectory_valid"]
    assert infos[0]["experience_event"]["successful_crop"]
    assert infos[0]["experience_event"]["min_crop_satisfied"]
    assert not infos[0]["experience_event"]["answer_correct"]


def test_direct_answer_before_min_crop_is_rejected_and_can_recover(tmp_path):
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(image_path)
    config = SimpleNamespace(
        env=EnvConfig(active_visual={"max_crop_calls": 2, "min_crop_calls": 1, "min_crop_dimension": 8})
    )
    env = ActiveVisualEnvironmentManager(config)
    row = {
        "task_id": "sample",
        "image_path": str(image_path),
        "question": "Choose A",
        "ground_truth": "A",
    }

    env.reset([row])
    obs, rewards, dones, infos = env.step([r"\boxed{A}"])

    assert not dones[0]
    assert rewards[0] == 0.0
    assert infos[0]["won"] == 0
    assert infos[0]["answer_correct"] == 1
    assert not infos[0]["min_crop_satisfied"]
    assert not infos[0]["successful_crop"]
    assert infos[0]["answer_reward_scale"] == 0
    assert infos[0]["answer_reward"] == 0
    assert not infos[0]["trajectory_valid"]
    assert infos[0]["experience_event"]["kind"] == "premature_answer"
    assert not infos[0]["experience_event"]["successful_crop"]
    assert not infos[0]["experience_event"]["min_crop_satisfied"]
    assert infos[0]["experience_event"]["answer_correct"]
    assert "at least 1 more successful image_zoom_in_tool" in str(obs["messages"][0])

    crop = (
        '<tool_call>{"name":"image_zoom_in_tool",'
        '"arguments":{"bbox_2d":[0,0,500,500],"label":"upper left",'
        '"img_idx":0}}</tool_call>'
    )
    _, rewards, dones, infos = env.step([crop])
    assert not dones[0] and rewards[0] == 0
    assert infos[0]["tool_calling"] == 1
    _, rewards, dones, infos = env.step([r"\boxed{A}"])
    assert dones[0] and rewards[0] == 0
    assert infos[0]["answer_correct"] == 1
    assert not infos[0]["trajectory_valid"]

    env.reset([row])
    _, rewards, dones, infos = env.step([r"\boxed{B}"])

    assert not dones[0]
    assert rewards[0] == 0.0
    assert infos[0]["won"] == 0
    assert infos[0]["answer_correct"] == 0
    assert not infos[0]["min_crop_satisfied"]
    assert not infos[0]["successful_crop"]
    assert infos[0]["answer_reward"] == 0
    assert not infos[0]["trajectory_valid"]


def test_repeated_rollouts_share_only_the_immutable_root(tmp_path):
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(image_path)
    config = SimpleNamespace(
        env=EnvConfig(active_visual={"max_crop_calls": 2, "min_crop_calls": 0, "min_crop_dimension": 8})
    )
    env = ActiveVisualEnvironmentManager(config)
    row = {
        "task_id": "sample",
        "image_path": str(image_path),
        "question": "Choose A",
        "ground_truth": "A",
    }

    env.reset([row, row])

    first, second = env.states
    assert first.runtime is not second.runtime
    assert first.runtime.artifacts is not second.runtime.artifacts
    assert first.runtime.artifacts["A0"] is second.runtime.artifacts["A0"]

    crop = (
        '<tool_call>{"name":"image_zoom_in_tool",'
        '"arguments":{"bbox_2d":[0,0,500,500],"label":"detail",'
        '"img_idx":0}}</tool_call>'
    )
    env.step([crop, r"\boxed{A}"])

    assert "A1" in first.runtime.artifacts
    assert "A1" not in second.runtime.artifacts


def test_malformed_tool_call_gates_the_complete_trajectory_reward(tmp_path):
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(image_path)
    config = SimpleNamespace(
        env=EnvConfig(active_visual={"max_crop_calls": 2, "min_crop_calls": 0, "min_crop_dimension": 8})
    )
    env = ActiveVisualEnvironmentManager(config)
    env.reset([{"task_id": "sample", "image_path": str(image_path), "question": "Choose A", "ground_truth": "A"}])

    malformed = (
        '<tool_call>{"name":"image_zoom_in_tool","arguments":'
        '{"bbox_2d":[0,0,500,500],"img_idx":0} '
        r"\boxed{A}"
    )
    _, rewards, dones, infos = env.step([malformed])

    assert not dones[0]
    assert rewards[0] == 0.0
    assert not infos[0]["is_action_valid"]
    assert not infos[0]["trajectory_valid"]

    _, rewards, dones, infos = env.step([r"\boxed{A}"])

    assert dones[0]
    assert infos[0]["answer_correct"] == 1
    assert infos[0]["won"] == 0
    assert rewards[0] == 0


def test_valid_boxed_answer_keeps_the_outcome_reward(tmp_path):
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(image_path)
    config = SimpleNamespace(
        env=EnvConfig(active_visual={"max_crop_calls": 2, "min_crop_calls": 0, "min_crop_dimension": 8})
    )
    env = ActiveVisualEnvironmentManager(config)
    env.reset([{"task_id": "sample", "image_path": str(image_path), "question": "Choose A", "ground_truth": "A"}])

    _, rewards, dones, infos = env.step([r"\boxed{A}"])

    assert dones[0]
    assert infos[0]["trajectory_valid"]
    assert rewards[0] == 1

    env.reset([{"task_id": "sample", "image_path": str(image_path), "question": "Choose A", "ground_truth": "A"}])
    _, rewards, dones, infos = env.step([r"\boxed{B}"])

    assert dones[0]
    assert infos[0]["trajectory_valid"]
    assert rewards[0] == 0


def test_repeated_invalid_actions_receive_one_trajectory_penalty(tmp_path):
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(image_path)
    config = SimpleNamespace(
        env=EnvConfig(active_visual={"max_crop_calls": 2, "min_crop_calls": 0, "min_crop_dimension": 8, "invalid_action_reward": -0.5})
    )
    env = ActiveVisualEnvironmentManager(config)
    env.reset([{"task_id": "sample", "image_path": str(image_path), "question": "Choose A", "ground_truth": "A"}])

    _, first_rewards, _, first_infos = env.step(["not an action"])
    _, second_rewards, _, second_infos = env.step(["still not an action"])

    assert first_rewards[0] == -0.5
    assert second_rewards[0] == 0
    assert not first_infos[0]["trajectory_valid"]
    assert not second_infos[0]["trajectory_valid"]


def test_qwen_boxed_prompt_allows_direct_answer_without_mandatory_crop(tmp_path):
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(image_path)
    config = SimpleNamespace(
        env=EnvConfig(
            active_visual={
                "max_crop_calls": 2,
                "min_crop_calls": 0,
                "min_crop_dimension": 8,
            }
        )
    )
    env = ActiveVisualEnvironmentManager(config)
    obs, _ = env.reset(
        [
            {
                "task_id": "sample",
                "image_path": str(image_path),
                "question": "Choose A",
                "ground_truth": "A",
            }
        ]
    )

    system_prompt = obs["messages"][0][0]["content"]
    opening_user_content = str(obs["messages"][0][1]["content"])
    assert r"\boxed{A}" in system_prompt
    assert "VISTA" not in system_prompt
    assert "[OPTIONAL VISUAL TOOL]" not in opening_user_content
    assert "[MANDATORY FIRST TURN]" not in opening_user_content

    _, rewards, dones, infos = env.step([r"Evidence is sufficient. \boxed{A}"])
    assert dones[0]
    assert rewards[0] == 1.0
    assert infos[0]["tool_calling"] == 0.0


def test_tool_reward_decays_and_only_rewards_first_successful_crop(tmp_path):
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(image_path)
    config = SimpleNamespace(
        env=EnvConfig(
            active_visual={
                "max_crop_calls": 3,
                "min_crop_calls": 0,
                "min_crop_dimension": 8,
                "tool_reward_start": 0.2,
                "tool_reward_end": 0.0,
                "tool_reward_decay_steps": 10,
            }
        )
    )
    row = {
        "task_id": "sample",
        "image_path": str(image_path),
        "question": "Choose A",
        "ground_truth": "A",
    }
    first_crop = (
        '<tool_call>{"name":"image_zoom_in_tool",'
        '"arguments":{"bbox_2d":[0,0,500,500],"label":"first",'
        '"img_idx":0}}</tool_call>'
    )
    second_crop = (
        '<tool_call>{"name":"image_zoom_in_tool",'
        '"arguments":{"bbox_2d":[500,500,1000,1000],"label":"second",'
        '"img_idx":0}}</tool_call>'
    )

    env = ActiveVisualEnvironmentManager(config)
    env.set_global_step(1)
    env.reset([row])
    _, first_rewards, _, first_infos = env.step([first_crop])
    _, second_rewards, _, second_infos = env.step([second_crop])

    assert first_rewards[0] == pytest.approx(0.2)
    assert first_infos[0]["tool_reward"] == pytest.approx(0.2)
    assert second_rewards[0] == 0.0
    assert second_infos[0]["tool_reward"] == 0.0

    env.set_global_step(11)
    env.reset([row])
    _, decayed_rewards, _, decayed_infos = env.step([first_crop])
    assert decayed_rewards[0] == 0.0
    assert decayed_infos[0]["tool_reward_coef"] == 0.0


def test_tool_reward_is_disabled_for_evaluation_environment(tmp_path):
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(image_path)
    config = SimpleNamespace(
        env=EnvConfig(
            active_visual={
                "max_crop_calls": 2,
                "min_crop_calls": 0,
                "min_crop_dimension": 8,
                "tool_reward_start": 0.2,
                "tool_reward_end": 0.0,
                "tool_reward_decay_steps": 10,
            }
        )
    )
    env = ActiveVisualEnvironmentManager(config, is_train=False)
    env.set_global_step(1)
    env.reset(
        [
            {
                "task_id": "sample",
                "image_path": str(image_path),
                "question": "Choose A",
                "ground_truth": "A",
            }
        ]
    )
    crop = (
        '<tool_call>{"name":"image_zoom_in_tool",'
        '"arguments":{"bbox_2d":[0,0,500,500],"label":"detail",'
        '"img_idx":0}}</tool_call>'
    )
    _, rewards, _, infos = env.step([crop])
    assert rewards[0] == 0.0
    assert infos[0]["tool_reward"] == 0.0
