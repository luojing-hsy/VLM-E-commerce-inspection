import math

import pytest

from src.training.opd_loss import topk_union_kl_lists


def test_identical_distributions_have_zero_loss() -> None:
    logits = [1000.0, 0.0, -1000.0, 3.0]
    assert topk_union_kl_lists(logits, logits, top_k=2) == pytest.approx(0.0, abs=1e-10)


def test_topk_union_loss_is_finite_for_extreme_logits() -> None:
    loss = topk_union_kl_lists([1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 0.0], top_k=1)
    assert math.isfinite(loss)
    assert loss > 0


def test_invalid_top_k_is_rejected() -> None:
    with pytest.raises(ValueError):
        topk_union_kl_lists([1.0], [1.0], top_k=0)

