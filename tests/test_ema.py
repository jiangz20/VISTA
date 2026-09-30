import pytest
import torch
from torch import nn

from vista.ema import update_ema_teacher
from vista.schedule import linear_decay_value, linear_warmup_value


def test_ema_teacher_updates_without_gradients():
    student = nn.Linear(2, 1, bias=False)
    teacher = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        student.weight.fill_(2.0)
        teacher.weight.fill_(0.0)
    teacher.requires_grad_(False)

    delta = update_ema_teacher(teacher, student, decay=0.75)

    assert delta == pytest.approx(2.0)
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 0.5))
    assert teacher.weight.requires_grad is False
    assert teacher.weight.grad is None


def test_ema_rejects_alias_and_bad_decay():
    model = nn.Linear(1, 1)
    with pytest.raises(ValueError):
        update_ema_teacher(model, model, decay=0.95)
    with pytest.raises(ValueError):
        update_ema_teacher(nn.Linear(1, 1), model, decay=1.0)


def test_linear_opd_warmup_reaches_target_without_overshoot():
    assert linear_warmup_value(0.5, step=0, warmup_steps=5) == 0.0
    assert linear_warmup_value(0.5, step=1, warmup_steps=5) == pytest.approx(0.1)
    assert linear_warmup_value(0.5, step=2, warmup_steps=5) == pytest.approx(0.2)
    assert linear_warmup_value(0.5, step=5, warmup_steps=5) == pytest.approx(0.5)
    assert linear_warmup_value(0.5, step=100, warmup_steps=5) == pytest.approx(0.5)
    assert linear_warmup_value(0.5, step=0, warmup_steps=0) == pytest.approx(0.5)


def test_linear_opd_warmup_rejects_invalid_values():
    with pytest.raises(ValueError):
        linear_warmup_value(-0.1, step=1, warmup_steps=10)
    with pytest.raises(ValueError):
        linear_warmup_value(0.5, step=1, warmup_steps=-1)


def test_linear_tool_reward_decay_reaches_end_without_overshoot():
    assert linear_decay_value(0.2, 0.0, step=-1, decay_steps=20) == pytest.approx(0.2)
    assert linear_decay_value(0.2, 0.0, step=0, decay_steps=20) == pytest.approx(0.2)
    assert linear_decay_value(0.2, 0.0, step=10, decay_steps=20) == pytest.approx(0.1)
    assert linear_decay_value(0.2, 0.0, step=20, decay_steps=20) == pytest.approx(0.0)
    assert linear_decay_value(0.2, 0.0, step=100, decay_steps=20) == pytest.approx(0.0)
    assert linear_decay_value(0.2, 0.0, step=0, decay_steps=0) == pytest.approx(0.0)


def test_linear_tool_reward_decay_rejects_invalid_values():
    with pytest.raises(ValueError):
        linear_decay_value(-0.1, 0.0, step=1, decay_steps=10)
    with pytest.raises(ValueError):
        linear_decay_value(0.1, -0.1, step=1, decay_steps=10)
    with pytest.raises(ValueError):
        linear_decay_value(0.1, 0.0, step=1, decay_steps=-1)
