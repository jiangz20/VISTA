import math

import pytest
import torch

from verl.trainer.ppo.core_algos import (
    chunked_vocab_logsumexp,
    compute_grpo_outcome_advantage,
    compute_opd_jsd_loss,
)


def _manual_jsd(student_prob: torch.Tensor, teacher_prob: torch.Tensor) -> torch.Tensor:
    mixture = 0.5 * (student_prob + teacher_prob)
    return 0.5 * (
        torch.sum(student_prob * (student_prob.log() - mixture.log()))
        + torch.sum(teacher_prob * (teacher_prob.log() - mixture.log()))
    )


def test_chunked_vocab_logsumexp_matches_full_reduction_and_gradient():
    torch.manual_seed(7)
    logits = torch.randn(3, 5, 37, dtype=torch.float64, requires_grad=True)
    chunked = chunked_vocab_logsumexp(logits, chunk_size=8)
    full = torch.logsumexp(logits, dim=-1, keepdim=True)
    torch.testing.assert_close(chunked, full)

    chunked.sum().backward(retain_graph=True)
    chunked_grad = logits.grad.detach().clone()
    logits.grad.zero_()
    full.sum().backward()
    torch.testing.assert_close(logits.grad, chunked_grad)


def test_chunked_vocab_logsumexp_validates_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunked_vocab_logsumexp(torch.zeros(1, 4), chunk_size=0)


def test_topk_tail_jsd_matches_manual_categorical_jsd():
    # The last entry in each manual distribution is the non-Top-K tail.
    student_prob = torch.tensor([0.6, 0.2, 0.2], dtype=torch.float64)
    teacher_prob = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    student_topk_log_prob = student_prob[:2].log().view(1, 1, 2)
    student_topk_log_prob.requires_grad_()
    teacher_topk_log_prob = teacher_prob[:2].log().view(1, 1, 2)

    loss, metrics = compute_opd_jsd_loss(
        student_topk_log_prob=student_topk_log_prob,
        teacher_topk_log_prob=teacher_topk_log_prob,
        response_mask=torch.ones(1, 1, dtype=torch.float64),
        opd_step_mask=torch.ones(1, dtype=torch.float64),
        alpha=0.5,
        add_tail=True,
    )

    expected = _manual_jsd(student_prob, teacher_prob)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(metrics["opd_raw_jsd_token_mean"], expected)
    loss.backward()
    assert student_topk_log_prob.grad is not None
    assert torch.isfinite(student_topk_log_prob.grad).all()


def test_jsd_is_zero_for_identical_distributions_and_respects_mask():
    log_prob = torch.tensor(
        [
            [[math.log(0.5), math.log(0.25)]],
            [[math.log(0.4), math.log(0.1)]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss, metrics = compute_opd_jsd_loss(
        student_topk_log_prob=log_prob,
        teacher_topk_log_prob=log_prob.detach().clone(),
        response_mask=torch.ones(2, 1, dtype=torch.float64),
        opd_step_mask=torch.tensor([1.0, 0.0], dtype=torch.float64),
        alpha=0.5,
        add_tail=True,
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-12, rtol=0)
    torch.testing.assert_close(
        metrics["opd_active_token_ratio"],
        torch.tensor(0.5, dtype=torch.float64),
    )


def test_grpo_sample_gate_masks_non_success_trajectories_after_group_baseline():
    token_level_rewards = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    response_mask = torch.ones_like(token_level_rewards)

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=["group", "group"],
        traj_index=["success", "failure"],
        norm_adv_by_std_in_grpo=False,
        sample_mask=torch.tensor([True, False]),
    )

    # The failed trajectory remains in the group baseline but receives no
    # policy-gradient signal; the successful trajectory keeps its centered
    # positive advantage on both response tokens.
    assert torch.allclose(advantages[1], torch.zeros_like(advantages[1]))
    assert torch.all(advantages[0] > 0)
    torch.testing.assert_close(returns, advantages)
