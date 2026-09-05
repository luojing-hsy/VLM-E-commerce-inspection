# Baseline and SFT evaluation

All evaluations use the 200-sample test split unless noted otherwise. Metrics are exported from the
repository's evaluation scripts; model checkpoints, images, datasets, and runtime caches are not
included in this repository.

## Summary

| Run | Samples | Decision accuracy | Violation macro-F1 | Exact protocol accuracy | Evidence mean | Parse rate | Business risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B baseline | 200 | 0.450 | 0.407 | 0.395 | 0.360 | 0.450 | 0.794 |
| Qwen3.5-9B baseline | 200 | 0.670 | 0.596 | 0.625 | 0.640 | 0.825 | 0.428 |
| Current direct baseline | 200 | 0.640 | 0.499 | - | 0.480 | 1.000 | 0.203 |
| SFT 4B (evaluation) | 200 | 0.850 | 0.736 | 0.740 | 0.740 | 1.000 | 0.127 |
| SFT 4B (direct three-image test) | 200 | 0.800 | 0.748 | 0.730 | 0.720 | 1.000 | 0.154 |

## SFT validation

The SFT validation split contains 100 samples:

- Decision accuracy: 0.910
- Violation macro-F1: 0.837
- Exact protocol accuracy: 0.850
- Evidence mean/recall@0.5: 0.900 / 0.900
- Parse rate: 1.000
- Severe-violation miss rate: 0.059
- False-reject rate: 0.267

The validation result is reported separately from the 200-sample test results and should not be
treated as an unbiased test estimate.

## Training curve

The SFT run recorded 100 training steps (plus two validation records). The full per-step JSONL is
included in sft_training_metrics.jsonl; the final recorded training loss was 0.0334 and the final
semantic token loss was 0.1396. The final validation loss was 0.0251 and validation semantic token
loss was 0.1062. These are training/validation metrics, not an additional held-out test estimate.

## Reproducibility

The raw metric files in this directory preserve the complete per-class breakdown, confusion matrix,
timing, and memory fields emitted by the evaluator. Paths in the original evaluator output refer to
the training server; no checkpoint files are tracked here.

## Provenance and comparability (2026-09-05)

The current Qwen3.5 SFT test is sft_direct_three_image_test.json (80% decision accuracy).
sft_evaluation.json and baseline_current_direct.json are historical reports whose model/run
identity is not sufficiently established for a current-model comparison. The table above
lists archived measurements, not a controlled re-evaluation. Business-risk settings and
protocol/prompt versions may differ. Resumed inference timing may cover only pending samples.
grpo_training_metrics.jsonl records initialization validation and 80 completed steps; its final
validation reward is 0.84960656, not test decision accuracy.
See ../REPRODUCIBILITY.md for configurations and verification boundaries.
