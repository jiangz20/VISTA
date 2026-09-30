"""Active-visual-agent prompts and collective-experience teacher helpers."""

from __future__ import annotations

import copy
import json
import re
from typing import Any


QWEN_THINK_WITH_IMAGES_SYSTEM_PROMPT = r"""
Your role is that of a research assistant specializing in visual information.
Answer questions about images by looking at them closely and then using research tools. Please follow this structured thinking process and show your work.

Start an iterative loop for each question:

- **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question.
List what you can tell just by looking, and what you'll need to look up.
- **Next, find information:** Use a tool to research the things you need to find out.
- **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

Continue this loop until your research is complete.

To finish, put your final answer within \boxed{}, and make sure it contains only the answer itself without extra words or symbols.

For multiple-choice questions, answer with the option's letter from the given choices directly, e.g. \boxed{A}, \boxed{B}, \boxed{C}, \boxed{D} etc. Note that YOU MUST choose One Answer from the Options.
""".strip()


# The student receives only the active visual policy prompt above. Tool syntax
# is supplied through the native function schema.
THINK_WITH_IMAGES_SYSTEM_PROMPT = QWEN_THINK_WITH_IMAGES_SYSTEM_PROMPT


COLLECTIVE_EXPERIENCE_TEACHER_SYSTEM_INSTRUCTION = r"""
[COLLECTIVE_EXPERIENCE_TEACHER_INSTRUCTION]
You are the privileged visual teacher for the student's current decision. The
COLLECTIVE_VISUAL_EXPERIENCE context is a shared, training-only view compiled from
all completed on-policy trajectories for the same input. It contains visual
artifacts, action/provenance relations, and optionally categorical trajectory
outcomes. Generated final answers are excluded from this shared context.
The DECISION_ALIGNMENT block
explicitly distinguishes artifacts already visible to the student from peer-only
and future-self hindsight.

Use collective experience to evaluate the student's exact current response. Do
not generate or imitate a replacement peer trajectory. Outcome association is not
causal evidence: a crop observed in a successful trajectory may be incidental,
and a crop observed in a failed trajectory may still be useful. Inspect the images
and reason about the original question yourself.

Never quote or reveal teacher-only trajectories, outcomes, artifact identifiers,
images, or optional answer privilege in the response. Use outcome labels as
feedback about trajectories, not as substitutes for visual evidence. If
student-visible evidence does not resolve the question, and collective experience
identifies a promising unseen region, prefer a targeted image_zoom_in_tool call.

Choose the next action according to the student's current uncertainty. If visible
evidence is sufficient, prefer a concise boxed
answer. Otherwise prefer one non-redundant crop that is likely to resolve the
remaining uncertainty.
Follow the native tool schema and make at most one tool call in this assistant
turn. Do not favor unexplained bare tool calls or bare boxed answers.
[/COLLECTIVE_EXPERIENCE_TEACHER_INSTRUCTION]
""".strip()


IMAGE_ZOOM_IN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "image_zoom_in_tool",
            "description": (
                "Zoom in on a specific region of an image by cropping it based on a "
                "bounding box (bbox) and an optional object label"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": (
                            "The bounding box of the region to zoom in, as "
                            "[x1, y1, x2, y2], where (x1, y1) is the top-left "
                            "corner and (x2, y2) is the bottom-right corner"
                        ),
                    },
                    "label": {
                        "type": "string",
                        "description": ("The name or label of the object in the specified bounding box"),
                    },
                    "img_idx": {
                        "type": "number",
                        "description": "The index of the zoomed-in image (starting from 0)",
                    },
                },
                "required": ["bbox_2d", "label", "img_idx"],
            },
        },
    }
]

TEXTUAL_IMAGE_MARKERS = (
    "<|vision_start|><|image_pad|><|vision_end|>",
    "<|image_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<image>",
)


_TOOL_CALL_RE = re.compile(
    r"<tool_call\b[^>]*>.*?</tool_call\s*>",
    re.DOTALL | re.IGNORECASE,
)


def strip_textual_image_markers(text: str) -> str:
    """Remove image placeholders that occur in untrusted plain text."""
    clean = str(text)
    for marker in TEXTUAL_IMAGE_MARKERS:
        clean = clean.replace(marker, "")
    return clean


def sanitize_message_text_image_markers(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strip fake image slots from text while preserving structured images.

    A model can spell a Qwen vision marker using ordinary text tokens. On the
    next turn the chat template would otherwise interpret that text as another
    image slot, although no corresponding image feature exists.
    """

    def sanitize(value: Any) -> Any:
        if isinstance(value, str):
            return strip_textual_image_markers(value)
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, tuple):
            return tuple(sanitize(item) for item in value)
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        return value

    return [sanitize(message) for message in messages]


def build_initial_messages(question: str, image: Any) -> list[dict[str, Any]]:
    """Build the current Qwen-boxed student opening turn."""
    clean_question = strip_textual_image_markers(question).strip()
    return [
        {"role": "system", "content": THINK_WITH_IMAGES_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": f"Question:\n{clean_question}",
                },
            ],
        },
    ]


def assistant_tool_message(raw_text: str, action: Any, img_idx: int) -> dict[str, Any]:
    """Convert generated tool XML into the structured form expected by Qwen's template."""
    reasoning = _TOOL_CALL_RE.sub("", str(raw_text), count=1).strip()
    return {
        "role": "assistant",
        "content": reasoning,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "image_zoom_in_tool",
                    "arguments": {
                        "bbox_2d": list(action.bbox_2d),
                        "label": action.label,
                        "img_idx": int(img_idx),
                    },
                },
            }
        ],
    }


def tool_image_message(image: Any) -> dict[str, Any]:
    """Return a zoom result as a native tool message containing the crop image."""
    return {"role": "tool", "content": [{"type": "image", "image": image}]}


def tool_error_message(error: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": [{"type": "text", "text": f"Tool error: {error}"}],
    }


def message_images(messages: list[dict[str, Any]]) -> list[Any]:
    """Extract images in the same order in which the chat template sees them."""
    images: list[Any] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                images.append(item.get("image"))
    return images


def serialize_messages(messages: list[dict[str, Any]]) -> str:
    """Serialize a message history while replacing in-memory images with slot indices."""
    serializable: list[dict[str, Any]] = []
    image_index = 0
    for message in messages:
        serialized_message = {key: copy.deepcopy(value) for key, value in message.items() if key != "content"}
        content = message.get("content")
        if isinstance(content, list):
            serialized_content: list[Any] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    serialized_item = {key: copy.deepcopy(value) for key, value in item.items() if key != "image"}
                    serialized_item["image_index"] = image_index
                    image_index += 1
                    serialized_content.append(serialized_item)
                else:
                    serialized_content.append(copy.deepcopy(item))
            serialized_message["content"] = serialized_content
        else:
            serialized_message["content"] = copy.deepcopy(content)
        serializable.append(serialized_message)
    return json.dumps(serializable, ensure_ascii=False, separators=(",", ":"))


def deserialize_messages(payload: str, images: list[Any]) -> list[dict[str, Any]]:
    """Restore image objects into a history created by :func:`serialize_messages`."""
    messages = json.loads(str(payload))
    used_indices: list[int] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                index = int(item.pop("image_index"))
                if index < 0 or index >= len(images):
                    raise ValueError(f"message image index {index} is out of range")
                item["image"] = images[index]
                used_indices.append(index)
    if used_indices != list(range(len(images))):
        raise ValueError(f"message history references image slots {used_indices}, expected {list(range(len(images)))}")
    return messages


def interleave_text_images(text: str, images: list[Any]) -> list[dict[str, Any]]:
    """Convert a text containing ``<image>`` markers into native content items."""
    parts = str(text).split("<image>")
    if len(parts) - 1 != len(images):
        raise ValueError(f"text has {len(parts) - 1} image markers for {len(images)} images")
    content: list[dict[str, Any]] = []
    for index, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if index < len(images):
            content.append({"type": "image", "image": images[index]})
    return content


def append_teacher_context(
    messages: list[dict[str, Any]],
    experience_text: str,
    experience_images: list[Any],
    *,
    alignment_text: str = "",
) -> list[dict[str, Any]]:
    """Add collective experience without flattening the student history."""
    result = copy.deepcopy(messages)
    if not result:
        raise ValueError("cannot augment an empty message history")

    teacher_instruction = COLLECTIVE_EXPERIENCE_TEACHER_SYSTEM_INSTRUCTION

    system_message = next((message for message in result if message.get("role") == "system"), None)
    if system_message is None:
        result.insert(
            0,
            {"role": "system", "content": teacher_instruction},
        )
    else:
        system_content = system_message.get("content")
        if isinstance(system_content, str):
            if teacher_instruction not in system_content:
                system_message["content"] = f"{system_content.rstrip()}\n\n{teacher_instruction}"
        elif isinstance(system_content, list):
            has_instruction = any(
                isinstance(item, dict)
                and item.get("type") == "text"
                and teacher_instruction in str(item.get("text", ""))
                for item in system_content
            )
            if not has_instruction:
                system_content.append({"type": "text", "text": teacher_instruction})
        else:
            raise TypeError("Collective-experience teacher system content must be text or native content items")

    target = result[-1]
    content = target.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif not isinstance(content, list):
        content = []
    content.append(
        {
            "type": "text",
            "text": (
                "\n[COLLECTIVE_EXPERIENCE_TEACHER_CONTEXT]\n"
                "The following shared experience view is teacher-only privilege:\n"
            ),
        }
    )
    if experience_text:
        content.extend(interleave_text_images(experience_text, experience_images))
    elif experience_images:
        raise ValueError("experience images require a serialized experience view")
    if alignment_text:
        content.append({"type": "text", "text": "\n" + alignment_text})
    content.append(
        {
            "type": "text",
            "text": "\n[/COLLECTIVE_EXPERIENCE_TEACHER_CONTEXT]",
        }
    )
    target["content"] = content
    return result
