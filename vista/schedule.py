"""Small, dependency-free schedules used by active-agent training."""

from __future__ import annotations


def linear_warmup_value(target: float, step: int, warmup_steps: int) -> float:
    """Linearly ramp from zero to ``target`` over ``warmup_steps`` updates."""
    target = float(target)
    step = int(step)
    warmup_steps = int(warmup_steps)
    if target < 0:
        raise ValueError("target must be non-negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if warmup_steps == 0:
        return target
    progress = min(max(step, 0) / warmup_steps, 1.0)
    return target * progress


def linear_decay_value(start: float, end: float, step: int, decay_steps: int) -> float:
    """Linearly decay from ``start`` to ``end`` over ``decay_steps`` updates.

    ``decay_steps=0`` means the schedule is already at ``end``.  Values are
    clamped before step zero and after the final decay step, which makes resume
    behavior depend only on the restored global step.
    """
    start = float(start)
    end = float(end)
    step = int(step)
    decay_steps = int(decay_steps)
    if start < 0 or end < 0:
        raise ValueError("decay endpoints must be non-negative")
    if decay_steps < 0:
        raise ValueError("decay_steps must be non-negative")
    if decay_steps == 0:
        return end
    progress = min(max(step, 0) / decay_steps, 1.0)
    return start + (end - start) * progress
