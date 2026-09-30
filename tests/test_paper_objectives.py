import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss, compute_grpo_outcome_advantage
from vista.objectives import full_vocab_distillation_loss, group_token_weights, prepare_group_batch, weighted_token_sum
from vista.supervision import training_route


def test_equal_input_weight_and_full_group_policy_denominator():
    mask = torch.tensor([[1.0, 0, 0], [1, 1, 1], [1, 0, 0]])
    pg, opd = group_token_weights(["a", "a", "b"], mask, torch.tensor([False, True, True]))
    torch.testing.assert_close(pg, torch.tensor([1 / 8, 1 / 8, 1 / 2]))
    torch.testing.assert_close(opd, torch.tensor([0.0, 1 / 6, 1 / 2]))
    success = torch.tensor([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert weighted_token_sum(success, mask, pg).item() == 1 / 8
    assert weighted_token_sum(torch.ones_like(mask), mask, opd).item() == 1
    pg_missing, _ = group_token_weights(["a", "a", "b"], mask, torch.zeros(3), num_groups=4)
    torch.testing.assert_close(pg_missing, pg / 2)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_full_distribution_loss_and_gradient_match_independent_reference(alpha):
    torch.manual_seed(42)
    s = torch.randn(3, 5, 17, dtype=torch.float64, requires_grad=True)
    t = torch.randn_like(s, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    gate = torch.tensor([True, False, True])
    _, weights = group_token_weights(["a", "a", "b"], mask, gate)
    loss, _ = full_vocab_distillation_loss(s, t, mask, gate, weights, alpha, token_chunk_size=2)
    p, q = s.softmax(-1), t.detach().softmax(-1)
    if alpha == 0:
        divergence = (q * (q.log() - p.log())).sum(-1)
    elif alpha == 1:
        divergence = (p * (p.log() - q.log())).sum(-1)
    else:
        m = (p + q) / 2
        divergence = (p * (p.log() - m.log()) + q * (q.log() - m.log())).sum(-1) / 2
    reference = weighted_token_sum(divergence, mask * gate[:, None], weights)
    torch.testing.assert_close(loss, reference)
    grad, teacher_grad = torch.autograd.grad(loss, (s, t), allow_unused=True)
    torch.testing.assert_close(grad, torch.autograd.grad(reference, s)[0])
    assert teacher_grad is None
    assert not grad[1].any()


def test_empty_subset_and_extreme_logits_are_finite():
    s = torch.tensor([[[1000.0, -1000.0]]], requires_grad=True)
    t = -s.detach()
    for gate in [torch.tensor([False]), torch.tensor([True])]:
        loss, _ = full_vocab_distillation_loss(s, t, torch.ones(1, 1), gate)
        assert torch.isfinite(loss)
        grad = torch.autograd.grad(loss, s)[0]
        assert torch.isfinite(grad).all()
        if not gate.any():
            assert loss == 0 and not grad.any()


def make_batch():
    ids = torch.tensor([[1, 2, 3, 4, 1], [2, 3, 2, 1, 0], [3, 1, 4, 2, 1]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    return DataProto.from_dict(
        tensors={
            "input_ids": ids,
            "attention_mask": mask,
            "position_ids": torch.arange(5).expand(3, -1),
            "responses": ids[:, -3:],
            "old_log_probs": torch.zeros(3, 3),
            "advantages": torch.tensor([[1.0, 1, 1], [0, 0, 0], [0.5, 0.5, 0.5]]),
            "teacher_input_ids": (ids + 1) % 5,
            "teacher_attention_mask": mask.clone(),
            "teacher_position_ids": torch.arange(5).expand(3, -1),
            "collective_experience_teacher_mask": torch.tensor([False, True, False]),
        },
        non_tensors={"uid": np.array(["a", "a", "b"], dtype=object)},
        meta_info={"temperature": 1.0},
    )


def test_padding_and_distributed_gradient_average_preserve_global_objective():
    original = prepare_group_batch(make_batch(), 1)
    padded = prepare_group_batch(make_batch(), 2)
    assert len(padded) == 4
    assert padded.batch["vista_padding"].tolist() == [False, False, False, True]
    assert padded.batch["vista_pg_weight"][-1] == 0
    theta = torch.tensor(0.1, requires_grad=True)

    def objective(batch):
        logprob = theta.expand_as(batch.batch["old_log_probs"])
        return compute_policy_loss(
            batch.batch["old_log_probs"],
            logprob,
            batch.batch["advantages"],
            batch.batch["attention_mask"][:, -3:],
            cliprange=0.2,
            clip_ratio_c=None,
            row_weights=batch.batch["vista_pg_weight"],
        )[0]

    expected = torch.autograd.grad(objective(original), theta)[0]
    ranks = padded.chunk(2)
    gradient = sum(torch.autograd.grad(objective(rank) * 2, theta)[0] for rank in ranks) / 2
    torch.testing.assert_close(expected, gradient)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(torch.randn(5, 5) * 0.1)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.table[input_ids] + 0)


@pytest.mark.parametrize("teacher_update", ["ema", "frozen"])
@pytest.mark.parametrize("multimodal", [False, True])
def test_actual_actor_one_update_is_invariant_to_microbatching_and_padding(monkeypatch, teacher_update, multimodal):
    import verl.workers.actor.dp_actor as actor_module
    import verl.utils.debug.performance as performance

    monkeypatch.setattr(performance, "_get_current_mem_info", lambda: ("0", "0", "0", "0"))
    monkeypatch.setattr(actor_module, "get_torch_device", lambda: SimpleNamespace(current_device=lambda: "cpu"))
    torch.manual_seed(13)
    initial = TinyPolicy()
    teacher_initial = TinyPolicy()

    def run(micro, padded=False, dynamic=False):
        config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml").actor_rollout_ref.actor
        config.use_dynamic_bsz = dynamic
        config.ppo_max_token_len_per_gpu = 6
        config.ppo_micro_batch_size_per_gpu = micro
        config.ppo_mini_batch_size = 1  # Must not create extra optimizer steps.
        config.use_remove_padding = False
        config.grad_clip = 100.0
        config.collective_experience.teacher_update = teacher_update
        student, teacher = copy.deepcopy(initial), copy.deepcopy(teacher_initial)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.03)
        counter = []
        optimizer.register_step_post_hook(lambda *args: counter.append(1))
        actor = actor_module.DataParallelPPOActor(config, student, optimizer)
        actor.teacher_module = teacher
        batch = prepare_group_batch(make_batch(), 1, size_divisor=4 if padded else 1)
        if multimodal:
            for key in ("multi_modal_inputs", "teacher_multi_modal_inputs"):
                batch.non_tensor_batch[key] = np.array(
                    [{"pixel_values": torch.ones(1, 2)} for _ in range(len(batch))], dtype=object
                )
        if padded:
            batch = batch.select_idxs([3, 2, 0, 1])
        actor.update_policy(batch)
        assert len(counter) == 1
        assert teacher.table.grad is None
        expected_teacher = (
            teacher_initial.table if teacher_update == "frozen" else 0.99 * teacher_initial.table + 0.01 * student.table
        )
        torch.testing.assert_close(teacher.table, expected_teacher)
        return student.table.detach()

    torch.testing.assert_close(run(1), run(3), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(run(1), run(2, padded=True), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(run(1), run(1, dynamic=True), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize(
    "rule,success,failure",
    [
        ("hapi", (True, False), (False, True)),
        ("grpo_only", (True, False), (True, False)),
        ("cved_only", (False, True), (False, True)),
        ("uniform_hybrid", (True, True), (True, True)),
        ("rl_mask_only", (True, True), (False, True)),
        ("distill_mask_only", (True, False), (True, True)),
    ],
)
def test_paper_learning_rule_controls(rule, success, failure):
    assert tuple(training_route({"status": "success"}, rule).values()) == success
    for status in ["failure", "invalid", "incomplete"]:
        assert tuple(training_route({"status": status}, rule).values()) == failure


def test_single_trajectory_has_zero_relative_advantage():
    advantage, _ = compute_grpo_outcome_advantage(
        torch.ones(1, 1), torch.ones(1, 1), np.array(["a"]), np.array(["t"]), compute_mean_std_cross_steps=False
    )
    assert not advantage.any()


def test_unpadded_full_vocabulary_response_alignment(monkeypatch):
    import verl.workers.actor.dp_actor as actor_module

    def rearrange(value, pattern):
        assert pattern == "b s ... -> (b s) ..."
        return value.reshape(-1, *value.shape[2:])

    def unpad(values, mask):
        indices = mask.reshape(-1).nonzero().flatten()
        return values.reshape(-1, values.shape[-1])[indices], indices, None, None

    def pad(hidden_states, indices, batch, seqlen):
        result = hidden_states.new_zeros(batch * seqlen, hidden_states.shape[-1])
        result[indices] = hidden_states
        return result.reshape(batch, seqlen, -1)

    monkeypatch.setattr(actor_module, "unpad_input", unpad, raising=False)
    monkeypatch.setattr(actor_module, "pad_input", pad, raising=False)
    monkeypatch.setattr(actor_module, "rearrange", rearrange, raising=False)
    monkeypatch.setattr(actor_module, "index_first_axis", lambda x, i: x[i], raising=False)
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml").actor_rollout_ref.actor
    config.use_remove_padding = False
    actor = actor_module.DataParallelPPOActor(config, TinyPolicy())
    data = make_batch().batch
    data["attention_mask"][0, 0] = 0  # Left prompt padding as well as right response padding.
    _, dense_logp, dense_logits, _ = actor._forward_micro_batch(data, 1.0, full_vocab=True)
    actor.use_remove_padding = True
    _, packed_logp, packed_logits, _ = actor._forward_micro_batch(data, 1.0, full_vocab=True)
    mask = data["attention_mask"][:, -3:].bool()
    torch.testing.assert_close(dense_logp[mask], packed_logp[mask])
    torch.testing.assert_close(dense_logits[mask], packed_logits[mask])
    gd = torch.autograd.grad(dense_logits[mask].square().sum(), actor.actor_module.table)[0]
    gp = torch.autograd.grad(packed_logits[mask].square().sum(), actor.actor_module.table)[0]
    torch.testing.assert_close(gd, gp)


def test_zero_lambda_skips_teacher_and_pure_clipping_retains_negative_advantage(monkeypatch):
    import verl.workers.actor.dp_actor as actor_module
    import verl.utils.debug.performance as performance

    monkeypatch.setattr(performance, "_get_current_mem_info", lambda: ("0",) * 4)
    monkeypatch.setattr(actor_module, "get_torch_device", lambda: SimpleNamespace(current_device=lambda: "cpu"))
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml").actor_rollout_ref.actor
    config.opd_loss_coef = 0.0
    config.use_dynamic_bsz = False
    config.use_remove_padding = False
    config.ppo_micro_batch_size_per_gpu = 1
    config.collective_experience.teacher_update = "frozen"
    model = TinyPolicy()
    actor = actor_module.DataParallelPPOActor(config, model, torch.optim.SGD(model.parameters(), lr=0.01))
    metrics = actor.update_policy(prepare_group_batch(make_batch(), 1))
    assert metrics["actor/opd_loss"] == [0.0]
    assert actor.teacher_module is None
    loss, *_ = compute_policy_loss(
        torch.zeros(1, 1),
        torch.full((1, 1), np.log(5.0)),
        -torch.ones(1, 1),
        torch.ones(1, 1),
        cliprange=0.2,
        clip_ratio_c=None,
    )
    torch.testing.assert_close(loss, torch.tensor(5.0))


def test_paper_defaults_and_reproducibility_record(tmp_path):
    import json
    from vista.reproducibility import save_run_record, seed_everything

    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    config.seed = 19
    config.trainer.default_local_dir = str(tmp_path)
    assert (
        config.data.seed
        == config.env.seed
        == config.actor_rollout_ref.seed
        == config.actor_rollout_ref.rollout.seed
        == 19
    )
    assert (config.data.train_batch_size, config.env.rollout.n, config.trainer.total_training_steps) == (96, 8, 65)
    assert (config.data.max_prompt_length, config.algorithm.collective_experience.teacher_max_prompt_length) == (
        16384,
        24576,
    )
    assert config.data.max_pixels == 12845056
    assert config.actor_rollout_ref.actor.opd_distribution == "full"
    assert config.actor_rollout_ref.actor.group_normalization
    assert config.actor_rollout_ref.actor.clip_ratio_c is None
    seed_everything(19)
    a = torch.rand(2)
    seed_everything(19)
    torch.testing.assert_close(a, torch.rand(2))
    record = save_run_record(config)
    assert OmegaConf.load(record / "config.yaml").data.seed == 19
    assert len(json.loads((record / "environment.json").read_text())["source_sha256"]) == 64
    assert save_run_record(config) != record
