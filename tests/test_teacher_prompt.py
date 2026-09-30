import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from agent_system.multi_turn_rollout.rollout_loop import (
    TrajectoryCollector,
    sanitize_generated_vision_tokens,
)
from agent_system.multi_turn_rollout.utils import to_object_array
from vista.prompts import (
    COLLECTIVE_EXPERIENCE_TEACHER_SYSTEM_INSTRUCTION,
    IMAGE_ZOOM_IN_TOOLS,
    QWEN_THINK_WITH_IMAGES_SYSTEM_PROMPT,
    THINK_WITH_IMAGES_SYSTEM_PROMPT,
    append_teacher_context,
    build_initial_messages,
    sanitize_message_text_image_markers,
    strip_textual_image_markers,
    tool_image_message,
)
from vista.teacher import append_identical_responses
from vista.supervision import event_is_teacher_supervisable
from verl.utils.dataset.rl_dataset import collate_fn


class DataConfig(dict):
    __getattr__ = dict.__getitem__


class FakeTokenizer:
    pad_token_id = 0
    unk_token_id = 99
    eos_token_id = 2

    def __init__(self):
        self.tools_seen = None

    @staticmethod
    def _tokens(text):
        return re.findall(r"<[^>]+>|.", text, flags=re.DOTALL)

    def apply_chat_template(self, chat, tools=None, **_kwargs):
        self.tools_seen = tools
        rendered = "T" if tools else ""
        for message in chat:
            content = message.get("content", "")
            if isinstance(content, str):
                if message.get("role") != "system":
                    rendered += content
                continue
            for item in content:
                if item.get("type") == "image":
                    rendered += "<image>"
                elif item.get("type") == "text":
                    rendered += item.get("text", "")
        return rendered

    def __call__(self, text, return_tensors, add_special_tokens):
        del return_tensors, add_special_tokens
        length = len(self._tokens(text))
        return {
            "input_ids": torch.arange(1, length + 1).view(1, -1),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(1, len(self._tokens(text)) + 1))

    @staticmethod
    def convert_tokens_to_ids(token):
        return {
            "<|vision_start|>": 91,
            "<|vision_end|>": 92,
            "<|image_pad|>": 93,
            "<|video_pad|>": 94,
        }.get(token, 99)


class FakeImageProcessor:
    merge_size = 1

    def __init__(self):
        self.calls = 0

    def __call__(self, images, return_tensors):
        del return_tensors
        self.calls += 1
        return {
            "pixel_values": torch.zeros((len(images), 3, 2, 2)),
            "image_grid_thw": torch.tensor([[1, 1, 4]] * len(images)),
        }


class FakeProcessor:
    image_token = "<|image_pad|>"

    def __init__(self):
        self.image_processor = FakeImageProcessor()


def make_collector():
    config = SimpleNamespace(
        data=DataConfig(
            max_prompt_length=32,
            truncation="error",
            apply_chat_template_kwargs={},
        )
    )
    return TrajectoryCollector(config, FakeTokenizer(), FakeProcessor())


def fake_rope_index(_processor, input_ids, **_kwargs):
    return torch.zeros((3, len(input_ids)), dtype=torch.long)


def test_current_student_system_prompt_is_only_the_requested_qwen_boxed_prompt():
    expected = r"""Your role is that of a research assistant specializing in visual information.
Answer questions about images by looking at them closely and then using research tools. Please follow this structured thinking process and show your work.

Start an iterative loop for each question:

- **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question.
List what you can tell just by looking, and what you'll need to look up.
- **Next, find information:** Use a tool to research the things you need to find out.
- **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

Continue this loop until your research is complete.

To finish, put your final answer within \boxed{}, and make sure it contains only the answer itself without extra words or symbols.

For multiple-choice questions, answer with the option's letter from the given choices directly, e.g. \boxed{A}, \boxed{B}, \boxed{C}, \boxed{D} etc. Note that YOU MUST choose One Answer from the Options."""

    assert QWEN_THINK_WITH_IMAGES_SYSTEM_PROMPT == expected
    assert THINK_WITH_IMAGES_SYSTEM_PROMPT == expected
    assert "normalized coordinates in [0, 1000]" not in THINK_WITH_IMAGES_SYSTEM_PROMPT
    assert "VISTA" not in THINK_WITH_IMAGES_SYSTEM_PROMPT


def test_zoom_tool_schema_matches_official_qwen_agent_contract():
    function = IMAGE_ZOOM_IN_TOOLS[0]["function"]

    assert function["description"] == (
        "Zoom in on a specific region of an image by cropping it based on a "
        "bounding box (bbox) and an optional object label"
    )
    assert function["parameters"]["required"] == ["bbox_2d", "label", "img_idx"]


def test_different_sized_images_collate_as_objects():
    first = np.zeros((1603, 1500, 3), dtype=np.uint8)
    second = np.zeros((700, 800, 3), dtype=np.uint8)

    batch = collate_fn(
        [
            {"obs_images": to_object_array([first])},
            {"obs_images": to_object_array([second])},
        ]
    )

    assert batch["obs_images"].shape == (2, 1)
    assert batch["obs_images"][0, 0] is first
    assert batch["obs_images"][1, 0] is second


def test_identical_native_histories_reuse_prompt_processing(monkeypatch):
    monkeypatch.setattr("verl.models.transformers.qwen2_vl.get_rope_index", fake_rope_index)
    collector = make_collector()
    collector.config.data.max_prompt_length = 512
    first_image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    second_image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    obs = {
        "messages": [
            build_initial_messages("Choose A", first_image),
            build_initial_messages("Choose A", second_image),
        ],
        "text": ["audit", "audit"],
        "text_base": ["Choose A", "Choose A"],
        "image": [[first_image], [second_image]],
        "anchor": [None, None],
    }
    gen_batch = SimpleNamespace(
        non_tensor_batch={
            "raw_prompt": [None, None],
            "data_source": ["vista", "vista"],
        }
    )
    cache = {}

    first = collector.preprocess_single_sample(0, gen_batch, obs, cache, 0)
    second = collector.preprocess_single_sample(1, gen_batch, obs, cache, 0)

    assert collector.processor.image_processor.calls == 1
    assert first["index"] == 0
    assert second["index"] == 1
    assert first["active_visual_messages_json"] == second["active_visual_messages_json"]


def test_student_and_teacher_use_independent_prompt_widths(monkeypatch):
    monkeypatch.setattr("verl.models.transformers.qwen2_vl.get_rope_index", fake_rope_index)
    collector = make_collector()
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))

    student = collector.build_multimodal_prompt_sample("question <image>", [image])
    teacher = collector.build_multimodal_prompt_sample("question <image>", [image], max_length=48)

    assert student["input_ids"].shape == (32,)
    assert teacher["input_ids"].shape == (48,)
    assert student["position_ids"].shape == (4, 32)
    assert teacher["position_ids"].shape == (4, 48)
    assert len(teacher["multi_modal_inputs"]["image_grid_thw"]) == 1

    responses = torch.tensor([[71, 72, 0, 0]])
    response_mask = torch.tensor([[1, 1, 0, 0]])
    teacher_ids, teacher_mask, teacher_positions = append_identical_responses(
        teacher["input_ids"].unsqueeze(0),
        teacher["attention_mask"].unsqueeze(0),
        teacher["position_ids"].unsqueeze(0),
        responses,
        response_mask,
    )
    assert teacher_ids.shape[-1] == 48 + 4
    assert teacher_positions.shape == (1, 4, 48 + 4)
    assert torch.equal(teacher_ids[:, -4:], responses)
    assert torch.equal(teacher_mask[:, -4:], response_mask)


def test_teacher_overflow_fails_before_image_token_truncation(monkeypatch):
    monkeypatch.setattr("verl.models.transformers.qwen2_vl.get_rope_index", fake_rope_index)
    collector = make_collector()
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="Image tokens are never truncated"):
        collector.build_multimodal_prompt_sample("question <image>", [image], max_length=8)


def test_teacher_rejects_image_grid_slot_mismatch(monkeypatch):
    monkeypatch.setattr("verl.models.transformers.qwen2_vl.get_rope_index", fake_rope_index)
    collector = make_collector()

    class MissingGridImageProcessor(FakeImageProcessor):
        def __call__(self, images, return_tensors):
            result = super().__call__(images, return_tensors)
            result["image_grid_thw"] = torch.empty((0, 3), dtype=torch.long)
            return result

    collector.processor.image_processor = MissingGridImageProcessor()
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="image grid count"):
        collector.build_multimodal_prompt_sample("question <image>", [image])


def test_native_qwen_history_passes_tools_and_preserves_tool_images(monkeypatch):
    monkeypatch.setattr("verl.models.transformers.qwen2_vl.get_rope_index", fake_rope_index)
    collector = make_collector()
    original = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    crop = Image.fromarray(np.ones((16, 16, 3), dtype=np.uint8))
    messages = build_initial_messages("Choose A", original)
    messages.extend(
        [
            {
                "role": "assistant",
                "content": "I should inspect it.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "image_zoom_in_tool",
                            "arguments": {
                                "bbox_2d": [0, 0, 500, 500],
                                "label": "detail",
                                "img_idx": 0,
                            },
                        },
                    }
                ],
            },
            tool_image_message(crop),
        ]
    )

    sample = collector.build_multimodal_chat_prompt_sample(messages, max_length=512)

    assert collector.tokenizer.tools_seen == IMAGE_ZOOM_IN_TOOLS
    assert len(sample["multi_modal_data"]["image"]) == 2
    assert len(sample["multi_modal_inputs"]["image_grid_thw"]) == 2
    assert sample["multi_modal_inputs"]["pixel_values"].dtype == torch.bfloat16
    assert sample["multi_modal_inputs"]["image_grid_thw"].dtype == torch.long


def test_multimodal_prompt_adaptively_downscales_images_to_fixed_budget(monkeypatch):
    monkeypatch.setattr("verl.models.transformers.qwen2_vl.get_rope_index", fake_rope_index)
    collector = make_collector()

    class SizeAwareImageProcessor(FakeImageProcessor):
        def __init__(self):
            super().__init__()
            self.seen_sizes = []

        def __call__(self, images, return_tensors):
            del return_tensors
            self.calls += 1
            self.seen_sizes.append([(image.width, image.height) for image in images])
            grid_tokens = [max(1, image.width * image.height // 1024) for image in images]
            return {
                "pixel_values": torch.zeros((len(images), 3, 2, 2)),
                "image_grid_thw": torch.tensor([[1, 1, token_count] for token_count in grid_tokens]),
            }

    size_aware = SizeAwareImageProcessor()
    collector.processor.image_processor = size_aware
    image = Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8))

    sample = collector.build_multimodal_prompt_sample(
        "Choose A <image>",
        [image],
        max_length=256,
    )

    assert size_aware.calls > 1
    assert size_aware.seen_sizes[-1][0][0] < size_aware.seen_sizes[0][0][0]
    assert int(sample["attention_mask"].sum()) <= 256
    assert len(sample["multi_modal_data"]["image"]) == 1


def test_teacher_context_adds_collective_experience_without_mutating_student():
    original = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    group_crop = Image.fromarray(np.ones((16, 16, 3), dtype=np.uint8))
    student_messages = build_initial_messages("Choose A", original)

    teacher_messages = append_teacher_context(
        student_messages,
        (
            "[COLLECTIVE_VISUAL_EXPERIENCE]\n"
            "trajectory=T0 outcome=success\nartifact=V1 <image>\n"
            "[/COLLECTIVE_VISUAL_EXPERIENCE]"
        ),
        [group_crop],
        alignment_text=(
            '[DECISION_ALIGNMENT]\nstudent_visible=[]\npeer_only=["V1"]\nself_future=[]\n[/DECISION_ALIGNMENT]'
        ),
    )

    assert COLLECTIVE_EXPERIENCE_TEACHER_SYSTEM_INSTRUCTION not in student_messages[0]["content"]
    assert COLLECTIVE_EXPERIENCE_TEACHER_SYSTEM_INSTRUCTION in teacher_messages[0]["content"]
    normalized_system = " ".join(teacher_messages[0]["content"].lower().split())
    assert "all completed on-policy trajectories" in normalized_system
    assert "same input" in normalized_system
    assert "outcome association is not causal evidence" in normalized_system
    assert "student's exact current response" in normalized_system
    assert "requires at least one successful crop" not in normalized_system
    teacher_content = teacher_messages[-1]["content"]
    assert any(item.get("image") is group_crop for item in teacher_content)
    assert any("COLLECTIVE_EXPERIENCE_TEACHER_CONTEXT" in item.get("text", "") for item in teacher_content)
    rendered_context = " ".join(item.get("text", "").lower() for item in teacher_content if isinstance(item, dict))
    assert "decision_alignment" in rendered_context
    assert "peer_only" in rendered_context
    assert "answer_privilege" not in rendered_context


def test_teacher_instruction_is_inserted_as_system_when_missing():
    messages = [{"role": "user", "content": "Question"}]

    teacher_messages = append_teacher_context(messages, "experience", [])

    assert teacher_messages[0] == {
        "role": "system",
        "content": COLLECTIVE_EXPERIENCE_TEACHER_SYSTEM_INSTRUCTION,
    }
    assert teacher_messages[1]["role"] == "user"


def test_teacher_supervision_covers_valid_crop_and_final_decisions():
    assert event_is_teacher_supervisable({"kind": "crop"})
    assert event_is_teacher_supervisable({"kind": "final", "successful_crop": True})
    assert event_is_teacher_supervisable({"kind": "final", "successful_crop": False})
    assert event_is_teacher_supervisable({"kind": "final"})
    assert not event_is_teacher_supervisable({"kind": "invalid"})
    assert not event_is_teacher_supervisable(None)


def test_generated_visual_tokens_are_sanitized_before_logprob_recompute():
    tokenizer = FakeTokenizer()
    responses = torch.tensor(
        [
            [10, 93, 11, 0],
            [12, 13, 14, 0],
            [91, 15, 92, 94],
        ],
        dtype=torch.long,
    )
    input_ids = torch.tensor(
        [
            [93, 1, 2, 10, 93, 11, 0],
            [93, 1, 2, 12, 13, 14, 0],
            [93, 1, 2, 91, 15, 92, 94],
        ],
        dtype=torch.long,
    )

    invalid = sanitize_generated_vision_tokens(responses, input_ids, tokenizer)

    assert invalid.tolist() == [True, False, True]
    assert responses.tolist() == [
        [10, 99, 11, 0],
        [12, 13, 14, 0],
        [99, 15, 99, 99],
    ]
    assert input_ids[:, :3].tolist() == [[93, 1, 2]] * 3
    assert torch.equal(input_ids[:, -4:], responses)


def test_textual_image_markers_are_removed_without_mutating_structured_images():
    image = object()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": ("fake <|vision_start|><|image_pad|><|vision_end|> and <image>"),
                },
            ],
        },
        {
            "role": "assistant",
            "content": "another literal <|image_pad|>",
        },
    ]

    sanitized = sanitize_message_text_image_markers(messages)

    assert sanitized[0]["content"][0]["image"] is image
    assert sanitized[0]["content"][1]["text"] == "fake  and "
    assert sanitized[1]["content"] == "another literal "
    assert "<|image_pad|>" in messages[1]["content"]
    assert strip_textual_image_markers("<image>A<|vision_start|><|image_pad|><|vision_end|>") == "A"


def test_plaintext_generated_image_marker_does_not_create_a_fake_prompt_slot(
    monkeypatch,
):
    monkeypatch.setattr("verl.models.transformers.qwen2_vl.get_rope_index", fake_rope_index)
    collector = make_collector()
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    messages = build_initial_messages("Choose A", image)
    messages.append(
        {
            "role": "assistant",
            "content": ("I emitted text, not an image: <|vision_start|><|image_pad|><|vision_end|>"),
        }
    )

    sample = collector.build_multimodal_chat_prompt_sample(messages, max_length=512)

    assert len(sample["multi_modal_data"]["image"]) == 1
    assert len(sample["multi_modal_inputs"]["image_grid_thw"]) == 1
