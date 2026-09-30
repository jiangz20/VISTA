"""Small, testable guards for collective-experience supervision."""

from __future__ import annotations

from typing import Any, Mapping


def event_is_teacher_supervisable(
    event: Mapping[str, Any] | None,
    *,
    require_valid_outcome: bool = False,
) -> bool:
    """Return whether an event represents a valid student decision.

    Direct answers without a crop remain supervised deliberately. When
    ``require_valid_outcome`` is set, this helper filters decisions whose
    hindsight outcome is protocol-invalid. The trainer may still apply OPD to
    the complete response separately; this guard is only for event-level use.
    """

    if not isinstance(event, Mapping):
        return False
    if event.get("kind") not in {"crop", "final"}:
        return False
    if require_valid_outcome:
        outcome = event.get("trajectory_outcome")
        if not isinstance(outcome, Mapping) or not bool(outcome.get("trajectory_valid", False)):
            return False
    return True


def training_route(outcome: Mapping[str, Any] | None, learning_rule: str = "hapi") -> dict[str, bool]:
    """Map a categorical rollout outcome to the VISTA training objectives.

    The route is sample/trajectory-level: every response row of a successful
    trajectory participates in GRPO, while every row of a non-success
    trajectory participates in peer-evidence OPD.  Failure-specific local
    repair semantics live in the teacher prompt rather than in a token-level
    truncation of the failed response.
    """

    status = str(outcome.get("status", "unknown")).strip().lower() if isinstance(outcome, Mapping) else "unknown"
    success = status == "success"
    failed = status in {"failure", "invalid", "incomplete"}
    routes = {
        "hapi": (success, failed),
        "grpo_only": (True, False),
        "cved_only": (False, True),
        "uniform_hybrid": (True, True),
        "rl_mask_only": (success, True),
        "distill_mask_only": (True, failed),
    }
    if learning_rule not in routes:
        raise ValueError(f"Unknown learning rule: {learning_rule}")
    grpo, repair = routes[learning_rule]
    return {"grpo": grpo, "repair": repair}
