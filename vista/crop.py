"""Crop-tool runtime with stable artifact identifiers and provenance tracking."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TOOL_CALL_RE = re.compile(
    r"<tool_call\b[^>]*>\s*(.*?)\s*</tool_call\s*>",
    re.DOTALL | re.IGNORECASE,
)
QWEN35_FUNCTION_RE = re.compile(
    r"<function\s*=\s*(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)\s*>"
    r"(?P<parameters>.*?)"
    r"</function\s*>",
    re.DOTALL | re.IGNORECASE,
)
QWEN35_PARAMETER_RE = re.compile(
    r"<parameter\s*=\s*(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)\s*>"
    r"(?P<value>.*?)"
    r"</parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)
MAX_PROCESSOR_ASPECT_RATIO = 199.0


@dataclass(frozen=True)
class CropAction:
    source_artifact_id: str
    bbox_2d: tuple[float, float, float, float]
    label: str = "region"


@dataclass
class Artifact:
    artifact_id: str
    parent_artifact_id: str | None
    local_bbox: tuple[float, float, float, float]
    root_bbox: tuple[float, float, float, float]
    image: Image.Image
    sha256: str
    phash: int

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("image")
        return data


def _to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1:
            array = array * 255
        array = array.astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _hash_image(image: Image.Image) -> str:
    payload = image.tobytes() + f"{image.mode}:{image.size}".encode()
    return hashlib.sha256(payload).hexdigest()


def _phash(image: Image.Image) -> int:
    # A small dependency-free perceptual hash. It is deterministic and sufficient
    # for detecting near-identical crops in a per-instance graph.
    gray = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.float32)
    mean = float(values.mean())
    bits = values >= mean
    result = 0
    for bit in bits.flat:
        result = (result << 1) | int(bit)
    return result


def _pad_to_safe_aspect_ratio(
    image: Image.Image,
    max_aspect_ratio: float = MAX_PROCESSOR_ASPECT_RATIO,
) -> Image.Image:
    """Pad an extreme strip crop so Qwen's image processor can accept it.

    Padding, unlike anisotropic resizing, preserves every requested source
    pixel and its geometry.  The Qwen image processor requires the absolute
    aspect ratio to be strictly below 200, so use a conservative limit of 199.
    """
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("crop image has zero area")
    if max(width / height, height / width) <= max_aspect_ratio:
        return image

    if width < height:
        padded_width = max(width, int(math.ceil(height / max_aspect_ratio)))
        canvas = Image.new("RGB", (padded_width, height))
        canvas.paste(image, ((padded_width - width) // 2, 0))
    else:
        padded_height = max(height, int(math.ceil(width / max_aspect_ratio)))
        canvas = Image.new("RGB", (width, padded_height))
        canvas.paste(image, (0, (padded_height - height) // 2))
    return canvas


def _parse_qwen35_tool_call(body: str) -> dict[str, Any]:
    """Normalize Qwen3.5's native function/parameter XML to the JSON tool schema."""
    function_match = QWEN35_FUNCTION_RE.fullmatch(body.strip())
    if function_match is None:
        raise ValueError("tool call must contain valid JSON or Qwen3.5 function/parameter XML")

    arguments: dict[str, Any] = {}
    parameters = function_match.group("parameters")
    previous_end = 0
    for parameter_match in QWEN35_PARAMETER_RE.finditer(parameters):
        if parameters[previous_end : parameter_match.start()].strip():
            raise ValueError("unexpected content between Qwen3.5 tool parameters")
        previous_end = parameter_match.end()
        name = parameter_match.group("name")
        if name in arguments:
            raise ValueError(f"duplicate tool parameter {name!r}")
        raw_value = parameter_match.group("value").strip()
        try:
            arguments[name] = json.loads(raw_value)
        except json.JSONDecodeError:
            # Qwen3.5 emits free-form labels and artifact ids without JSON
            # quoting, while arrays and numbers use valid JSON syntax.
            if name in {"label", "source_artifact_id"}:
                arguments[name] = raw_value
            else:
                raise ValueError(f"tool parameter {name!r} must contain a valid JSON value") from None
    if parameters[previous_end:].strip():
        raise ValueError("unexpected content after Qwen3.5 tool parameters")

    return {
        "name": function_match.group("name"),
        "arguments": arguments,
    }


def _parse_tool_payload(body: str) -> dict[str, Any]:
    stripped = body.strip()
    if stripped.startswith("{"):
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError("tool call JSON must be an object")
        return payload
    return _parse_qwen35_tool_call(stripped)


def parse_action(text: str, artifact_order: list[str] | None = None) -> CropAction | None:
    """Parse supported active-visual crop calls from an assistant response."""
    match = TOOL_CALL_RE.search(text)
    if match is None:
        return None
    payload = _parse_tool_payload(match.group(1))
    name = payload.get("name")
    if name not in {"crop", "image_zoom_in_tool"}:
        raise ValueError(f"unsupported tool {name!r}")
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    bbox = arguments.get("bbox_2d")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox_2d must contain four numbers")
    try:
        bbox_tuple = tuple(float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox_2d must contain four numbers") from exc
    source_id = arguments.get("source_artifact_id")
    if source_id is None and "img_idx" in arguments:
        if artifact_order is None:
            raise ValueError("img_idx requires artifact_order")
        index = int(arguments["img_idx"])
        if index < 0 or index >= len(artifact_order):
            raise ValueError(f"img_idx {index} is out of range")
        source_id = artifact_order[index]
    if artifact_order is not None and str(source_id or "A0") not in artifact_order:
        raise ValueError(f"unknown source artifact {source_id!r}")
    return CropAction(
        source_artifact_id=str(source_id or "A0"),
        bbox_2d=bbox_tuple,
        label=str(arguments.get("label", "region")),
    )


def parse_final_answer(text: str) -> str | None:
    marker = re.search(r"\\boxed\s*\{", text)
    if marker is None:
        return None
    start = marker.end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                answer = text[start:index].strip()
                latex_wrapper = re.fullmatch(r"\\(?:text|mathrm|mathbf)\s*\{(.*)\}", answer, re.DOTALL)
                return (latex_wrapper.group(1) if latex_wrapper else answer).strip()
    return None


class CropRuntime:
    """Per-trajectory in-memory artifact store."""

    def __init__(self, image: Any, min_dimension: int = 32):
        root = _to_pil(image)
        self.min_dimension = int(min_dimension)
        self.artifacts: dict[str, Artifact] = {
            "A0": Artifact(
                artifact_id="A0",
                parent_artifact_id=None,
                local_bbox=(0.0, 0.0, 1000.0, 1000.0),
                root_bbox=(0.0, 0.0, 1000.0, 1000.0),
                image=root,
                sha256=_hash_image(root),
                phash=_phash(root),
            )
        }

    def fork_root(self) -> "CropRuntime":
        """Create an isolated trajectory store that shares the immutable root artifact."""
        runtime = object.__new__(type(self))
        runtime.min_dimension = self.min_dimension
        runtime.artifacts = {"A0": self.artifacts["A0"]}
        return runtime

    @property
    def artifact_order(self) -> list[str]:
        return list(self.artifacts)

    @staticmethod
    def _validated_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        if not all(np.isfinite([x1, y1, x2, y2])):
            raise ValueError("bbox_2d contains a non-finite coordinate")
        x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
        y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
        if x1 == x2 or y1 == y2:
            raise ValueError("bbox_2d has zero area")
        return x1, y1, x2, y2

    @staticmethod
    def _compose_root_bbox(parent: Artifact, bbox: tuple[float, float, float, float]):
        px1, py1, px2, py2 = parent.root_bbox
        x1, y1, x2, y2 = bbox
        width, height = px2 - px1, py2 - py1
        return (
            px1 + width * x1 / 1000.0,
            py1 + height * y1 / 1000.0,
            px1 + width * x2 / 1000.0,
            py1 + height * y2 / 1000.0,
        )

    def crop(self, action: CropAction) -> Artifact:
        if action.source_artifact_id not in self.artifacts:
            raise ValueError(f"unknown source artifact {action.source_artifact_id!r}")
        parent = self.artifacts[action.source_artifact_id]
        bbox = self._validated_bbox(action.bbox_2d)
        width, height = parent.image.size
        x1, y1, x2, y2 = bbox
        # Use floor/ceil so every non-empty normalized box covers at least one
        # source pixel. Rounding both edges independently can collapse a very
        # small but valid box to zero pixels.
        pixel_bbox = (
            int(math.floor(x1 * width / 1000.0)),
            int(math.floor(y1 * height / 1000.0)),
            int(math.ceil(x2 * width / 1000.0)),
            int(math.ceil(y2 * height / 1000.0)),
        )
        pixel_width = pixel_bbox[2] - pixel_bbox[0]
        pixel_height = pixel_bbox[3] - pixel_bbox[1]
        image = parent.image.crop(pixel_bbox).convert("RGB")

        # Qwen rejects images whose absolute aspect ratio is >= 200.  A valid
        # one-pixel-wide strip can otherwise hit that guard before the model
        # sees it.  Pad only such extreme crops before applying the existing
        # minimum-dimension upscaling rule.
        image = _pad_to_safe_aspect_ratio(image)
        pixel_width, pixel_height = image.size

        # A small requested region is still a valid observation. Upscale it
        # uniformly instead of rejecting the action, preserving the aspect
        # ratio while ensuring both output dimensions meet the processor's
        # minimum useful size.
        if pixel_width < self.min_dimension or pixel_height < self.min_dimension:
            scale = max(
                self.min_dimension / pixel_width,
                self.min_dimension / pixel_height,
            )
            resized_width = max(self.min_dimension, int(round(pixel_width * scale)))
            resized_height = max(self.min_dimension, int(round(pixel_height * scale)))
            image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        artifact_id = f"A{len(self.artifacts)}"
        artifact = Artifact(
            artifact_id=artifact_id,
            parent_artifact_id=parent.artifact_id,
            local_bbox=bbox,
            root_bbox=self._compose_root_bbox(parent, bbox),
            image=image,
            sha256=_hash_image(image),
            phash=_phash(image),
        )
        self.artifacts[artifact_id] = artifact
        return artifact
