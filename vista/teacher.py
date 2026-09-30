"""Tensor helpers for graph-privileged EMA-teacher scoring."""

from __future__ import annotations

import torch


def append_identical_responses(
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    prompt_position_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append the student's exact response IDs and mask to a teacher prompt."""
    response_length = responses.size(1)
    teacher_input_ids = torch.cat([prompt_input_ids, responses], dim=-1)
    expected_response_mask = response_mask.to(prompt_attention_mask.dtype)
    teacher_attention_mask = torch.cat([prompt_attention_mask, expected_response_mask], dim=-1)

    next_position = prompt_position_ids.max(dim=-1, keepdim=True).values + 1
    response_offsets = torch.arange(
        response_length,
        dtype=prompt_position_ids.dtype,
        device=prompt_position_ids.device,
    )
    response_offsets = (
        response_offsets.view(1, 1, -1) if prompt_position_ids.dim() == 3 else response_offsets.view(1, -1)
    )
    teacher_position_ids = torch.cat([prompt_position_ids, next_position + response_offsets], dim=-1)

    if not torch.equal(teacher_input_ids[:, -response_length:], responses):
        raise AssertionError("teacher and student response IDs must be identical")
    if not torch.equal(teacher_attention_mask[:, -response_length:], expected_response_mask):
        raise AssertionError("teacher and student response masks must be identical")
    return teacher_input_ids, teacher_attention_mask, teacher_position_ids
