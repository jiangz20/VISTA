"""Small, testable EMA primitive used by the collective-experience teacher."""

from __future__ import annotations

import torch
from torch import nn


def update_ema_teacher(teacher: nn.Module, student: nn.Module, decay: float) -> float:
    """Update a distinct no-grad teacher and return its mean pre-update delta."""
    if teacher is student:
        raise ValueError("EMA teacher and student must be distinct modules")
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
    teacher_parameters = list(teacher.parameters())
    student_parameters = list(student.parameters())
    if len(teacher_parameters) != len(student_parameters):
        raise ValueError("EMA teacher and student parameter structures differ")
    teacher_buffers = list(teacher.buffers())
    student_buffers = list(student.buffers())
    if len(teacher_buffers) != len(student_buffers):
        raise ValueError("EMA teacher and student buffer structures differ")

    delta_sum = 0.0
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_parameters, student_parameters, strict=True):
            student_data = student_param.detach().to(teacher_param.device)
            delta_sum += float((teacher_param - student_data).abs().float().mean().item())
            teacher_param.mul_(decay).add_(student_data, alpha=1.0 - decay)
            teacher_param.requires_grad_(False)
        for teacher_buffer, student_buffer in zip(teacher_buffers, student_buffers, strict=True):
            student_data = student_buffer.detach().to(teacher_buffer.device)
            if teacher_buffer.is_floating_point():
                teacher_buffer.mul_(decay).add_(student_data, alpha=1.0 - decay)
            else:
                teacher_buffer.copy_(student_data)
    return delta_sum / max(len(teacher_parameters), 1)
