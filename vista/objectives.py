"""Paper objectives, with denominators fixed before distributed batching."""

import math

import torch
from torch.utils.checkpoint import checkpoint


def group_token_weights(group_ids, response_mask, distillation_mask, num_groups=None):
    if len(group_ids) != len(response_mask) or distillation_mask.shape != (len(group_ids),):
        raise ValueError("Expected one group ID and distillation gate per response row")
    groups = {}
    for i, uid in enumerate(group_ids):
        groups.setdefault(str(uid), []).append(i)
    count = len(groups) if num_groups is None else int(num_groups)
    if count < len(groups) or count < 1:
        raise ValueError("num_groups must include every observed input group")
    pg = torch.zeros(len(group_ids), device=response_mask.device, dtype=torch.float32)
    opd = torch.zeros_like(pg)
    tokens = response_mask.float().sum(-1)
    for rows in groups.values():
        total = tokens[rows].sum()
        eligible = (tokens[rows] * distillation_mask[rows]).sum()
        if total > 0:
            pg[rows] = 1 / (count * total)
        if eligible > 0:
            opd[rows] = distillation_mask[rows].float() / (count * eligible)
    return pg, opd


def prepare_group_batch(data, dp_world_size, size_divisor=None, num_groups=None):
    from verl import DataProto

    if not len(data):
        raise ValueError("Cannot update from an empty rollout batch")
    response_len = data.batch["responses"].shape[-1]
    mask = data.batch.get("loss_mask", data.batch["attention_mask"])[:, -response_len:]
    pg, opd = group_token_weights(
        data.non_tensor_batch["uid"],
        mask,
        data.batch["collective_experience_teacher_mask"],
        num_groups,
    )
    data.batch["vista_pg_weight"] = pg
    data.batch["vista_opd_weight"] = opd
    data.batch["vista_padding"] = torch.zeros(len(data), dtype=torch.bool, device=pg.device)
    data.meta_info["vista_dp_world_size"] = int(dp_world_size)
    divisor = math.lcm(int(dp_world_size), int(size_divisor or 1))
    extra = (-len(data)) % divisor
    if extra:
        padding = data.select_idxs([i % len(data) for i in range(extra)])
        for key in ("vista_pg_weight", "vista_opd_weight"):
            padding.batch[key] = torch.zeros_like(padding.batch[key])
        padding.batch["vista_padding"] = torch.ones(extra, dtype=torch.bool, device=pg.device)
        data = DataProto.concat([data, padding])
    return data


def weighted_token_sum(values, response_mask, row_weights):
    return (values * response_mask * row_weights.to(values.device).unsqueeze(-1)).sum()


def _divergence(student, teacher, alpha):
    dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    p = student.to(dtype).log_softmax(-1)
    q = teacher.to(dtype).log_softmax(-1)
    if alpha == 0:
        return (q.exp() * (q - p)).sum(-1)
    if alpha == 1:
        return (p.exp() * (p - q)).sum(-1)
    m = torch.logaddexp(p + math.log1p(-alpha), q + math.log(alpha))
    return ((1 - alpha) * p.exp() * (p - m) + alpha * q.exp() * (q - m)).sum(-1)


def full_vocab_distillation_loss(
    student_logits, teacher_logits, response_mask, distillation_mask, row_weights=None, alpha=0.5, token_chunk_size=16
):
    if student_logits.shape != teacher_logits.shape or student_logits.shape[:2] != response_mask.shape:
        raise ValueError("Student/teacher logits and response mask must align")
    if not 0 <= alpha <= 1 or token_chunk_size < 1:
        raise ValueError("Invalid mixture weight or token chunk size")
    active = response_mask * distillation_mask.unsqueeze(-1)
    weights = active.float()
    if row_weights is None:
        weights = weights / weights.sum().clamp_min(1)
    else:
        weights = weights * row_weights.unsqueeze(-1)
    student = student_logits.reshape(-1, student_logits.shape[-1])
    teacher = teacher_logits.detach().reshape_as(student)
    indices = (weights.reshape(-1) != 0).nonzero().flatten()
    loss = student.reshape(-1)[:1].sum() * 0
    raw = loss.detach()
    for ids in indices.split(token_chunk_size):
        if not len(ids):
            continue

        def chunk_loss(s, t):
            return _divergence(s, t, alpha)

        divergence = checkpoint(chunk_loss, student[ids], teacher[ids], use_reentrant=False)
        loss = loss + (divergence * weights.reshape(-1)[ids]).sum()
        raw = raw + divergence.detach().sum()
    return loss, {
        "opd_active_token_ratio": active.sum() / response_mask.sum().clamp_min(1),
        "opd_raw_jsd_token_mean": raw / max(len(indices), 1),
    }
