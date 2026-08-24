from __future__ import annotations

import math
from typing import Sequence


def _softmax(logits: Sequence[float]) -> list[float]:
    peak = max(logits)
    values = [math.exp(value - peak) for value in logits]
    total = sum(values)
    return [value / total for value in values]


def topk_union_kl_lists(teacher_logits: Sequence[float], student_logits: Sequence[float], top_k: int = 64) -> float:
    """Reference implementation of top-k-union KL with one tail bucket."""
    if len(teacher_logits) != len(student_logits) or not teacher_logits:
        raise ValueError("teacher and student logits must have the same non-zero length")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    k = min(top_k, len(teacher_logits))
    teacher_probs, student_probs = _softmax(teacher_logits), _softmax(student_logits)
    teacher_top = sorted(range(len(teacher_logits)), key=lambda index: teacher_logits[index], reverse=True)[:k]
    student_top = sorted(range(len(student_logits)), key=lambda index: student_logits[index], reverse=True)[:k]
    union = set(teacher_top) | set(student_top)
    teacher_tail = max(0.0, 1.0 - sum(teacher_probs[index] for index in union))
    student_tail = max(0.0, 1.0 - sum(student_probs[index] for index in union))
    epsilon = 1e-12
    categories = [(teacher_probs[index], student_probs[index]) for index in union] + [(teacher_tail, student_tail)]
    loss = sum(p * (math.log(max(p, epsilon)) - math.log(max(q, epsilon))) for p, q in categories if p > 0)
    return max(0.0, loss)


def topk_union_kl(teacher_logits, student_logits, top_k: int = 64, token_weights=None):
    """Differentiable PyTorch loss for ``[..., vocab]`` logits.

    Only probabilities in the teacher/student top-k union plus a single tail
    bucket enter KL. Teacher probabilities are detached by design.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only in GPU environments.
        raise RuntimeError("PyTorch is required for the differentiable OPD loss") from exc
    if teacher_logits.shape != student_logits.shape or teacher_logits.ndim < 2:
        raise ValueError("teacher and student logits must have the same [..., vocab] shape")
    vocab = teacher_logits.shape[-1]
    k = min(int(top_k), vocab)
    if k < 1:
        raise ValueError("top_k must be positive")
    teacher_flat = teacher_logits.detach().reshape(-1, vocab).float()
    student_flat = student_logits.reshape(-1, vocab).float()
    losses = []
    for teacher_row, student_row in zip(teacher_flat, student_flat):
        indices = torch.unique(torch.cat((torch.topk(teacher_row, k).indices, torch.topk(student_row, k).indices)))
        teacher_log_z = torch.logsumexp(teacher_row, dim=-1)
        student_log_z = torch.logsumexp(student_row, dim=-1)
        teacher_p = torch.exp(teacher_row[indices] - teacher_log_z)
        student_p = torch.exp(student_row[indices] - student_log_z)
        teacher_other = torch.clamp(1.0 - teacher_p.sum(), min=0.0)
        student_other = torch.clamp(1.0 - student_p.sum(), min=0.0)
        p = torch.cat((teacher_p, teacher_other.unsqueeze(0)))
        q = torch.cat((student_p, student_other.unsqueeze(0)))
        row_loss = torch.sum(torch.where(p > 0, p * (torch.log(p.clamp_min(1e-12)) - torch.log(q.clamp_min(1e-12))), 0.0))
        losses.append(row_loss.clamp_min(0.0))
    loss_tensor = torch.stack(losses).reshape(teacher_logits.shape[:-1])
    if token_weights is None:
        return loss_tensor.mean()
    weights = token_weights.to(device=loss_tensor.device, dtype=loss_tensor.dtype)
    if weights.shape != loss_tensor.shape:
        raise ValueError("token_weights shape must match logits without the vocabulary dimension")
    denominator = weights.sum()
    if denominator.item() <= 0:
        raise ValueError("token_weights must contain a positive weight")
    return (loss_tensor * weights).sum() / denominator

