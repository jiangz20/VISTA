"""Categorical hindsight for collective visual experience.

The training reward may use the reference answer, but the collective experience
excludes generated final answers when compiling teacher experience. Raw events
retain answers for verification; ground truth and reward magnitudes never enter
the compiled trajectory outcome.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any


_TOOL_BLOCK_RE = re.compile(r"<tool_call\b[^>]*>.*?</tool_call\s*>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\s*\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)
_VISION_MARKER_RE = re.compile(
    r"<\|(?:vision_start|vision_end|image_pad|video_pad)\|>|<image>",
    re.IGNORECASE,
)
_PRIVILEGE_MARKER_RE = re.compile(
    r"\[/?(?:COLLECTIVE_VISUAL_EXPERIENCE|DECISION_ALIGNMENT|"
    r"VISTA(?:_[A-Z_]+)?)[^\]]*\]",
    re.IGNORECASE,
)


def sanitize_agent_claim(value: Any, max_chars: int = 240) -> str:
    """Sanitize an untrusted student intent before teacher serialization."""

    limit = int(max_chars)
    if limit < 0:
        raise ValueError("max_chars must be non-negative")
    text = _TOOL_BLOCK_RE.sub(" ", str(value or ""))
    text = _BOXED_RE.sub(" ", text)
    text = _VISION_MARKER_RE.sub(" ", text)
    text = _PRIVILEGE_MARKER_RE.sub(" ", text)
    text = " ".join(text.split())
    if limit == 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def derive_trajectory_outcome(
    step_rows: Sequence[Mapping[str, Any]],
    *,
    episode_return: float,
    trajectory_valid: bool,
) -> dict[str, Any]:
    """Derive one categorical hindsight record from a completed rollout."""

    events = [
        row.get("experience_event")
        for row in step_rows
        if isinstance(row, Mapping) and isinstance(row.get("experience_event"), Mapping)
    ]
    crop_count = sum(event.get("kind") == "crop" for event in events)
    final_event = next(
        (event for event in reversed(events) if event.get("kind") == "final"),
        None,
    )
    answer_correct = (
        bool(final_event.get("answer_correct"))
        if isinstance(final_event, Mapping) and final_event.get("answer_correct") is not None
        else None
    )
    valid = bool(trajectory_valid)
    if not valid:
        status = "invalid"
        failure_type = "invalid_action"
    elif final_event is None:
        status = "incomplete"
        failure_type = "no_valid_final_answer"
    elif answer_correct:
        status = "success"
        failure_type = None
    else:
        status = "failure"
        failure_type = "wrong_after_crop" if crop_count else "wrong_direct_answer"

    # episode_return is audit-only. CollectiveExperienceCompiler deliberately
    # drops it so shaping rewards cannot be mistaken for evidence quality.
    return {
        "status": status,
        "answer_correct": answer_correct,
        "trajectory_valid": valid,
        "crop_count": int(crop_count),
        "failure_type": failure_type,
        "episode_return": float(episode_return),
    }


def annotate_trajectory_events(
    step_rows: Sequence[MutableMapping[str, Any]],
    *,
    episode_return: float,
    trajectory_valid: bool,
) -> dict[str, Any]:
    """Backfill completed-trajectory hindsight onto every decision event."""

    outcome = derive_trajectory_outcome(
        step_rows,
        episode_return=episode_return,
        trajectory_valid=trajectory_valid,
    )
    for row in step_rows:
        event = row.get("experience_event") if isinstance(row, MutableMapping) else None
        if isinstance(event, MutableMapping):
            event["trajectory_outcome"] = copy.deepcopy(outcome)
    return outcome


def outcome_status(outcome: Mapping[str, Any] | None) -> str:
    """Return a conservative normalized categorical outcome."""

    if not isinstance(outcome, Mapping):
        return "unknown"
    status = str(outcome.get("status", "unknown")).strip().lower()
    if status not in {"success", "failure", "invalid", "incomplete"}:
        return "unknown"
    return status
