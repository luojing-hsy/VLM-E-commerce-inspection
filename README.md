# Qwen3.5 商品多图一致性质检（当前版本）

完整操作与实验记录见 [复现指南](docs/REPRODUCIBILITY.md)，本次验证见 [检查记录](docs/verification_20260905.md)。

## 当前口径

项目以服务器目录 `/root/autodl-tmp/vlm-qwen3vl` 为准。原始清单为 `data/all_product.jsonl`，共 3,986 个商品；清洗后每个商品保留 1 张 main 和 2 张 detail，共 11,958 张图片。

当前可直接核对的数据集：

| 数据集 | 文件 | 样本 | 商品 | 易/中/难 |
|---|---|---:|---:|---:|
| SFT train | `data/sft/train.jsonl` | 1,000 | 973 | 600/300/100 |
| SFT valid | `data/sft/valid.jsonl` | 100 | 88 | 60/30/10 |
| GRPO train | `data/GRPO/train.jsonl` | 1,000 | 1,000 | 300/400/300 |
| GRPO valid | `data/GRPO/valid.jsonl` | 100 | 100 | 30/40/30 |
| Test | `data/test/test.jsonl` | 200 | 200 | 100/60/40 |

SFT 的类别配额为：训练集易/中类各 150、难类各 50；验证集易/中类各 15、难类各 5。SFT 每个商品保留 0–3 条，当前训练集和验证集最多均为 3 条。GRPO 已从 `data/GRPO_synthesis` 重新抽样，训练集按易类 4×75、中类 2×200、难类 2×150 配额，验证集按易类 8/8/7/7、中类 2×20、难类 2×15 配额；每个来源商品只保留 1 条。

SFT 的逐项审计见 [`docs/sft_dataset_analysis.md`](docs/sft_dataset_analysis.md)。本次快照已覆盖三种错图位置和质量子类型/位置组合，详见报告中的 2026-09-05 更新。

## 标签与难度

| 难度 | 标签 |
|---|---|
| 易 | `pass`、`color_mismatch`、`category_mismatch`、`material_mismatch` |
| 中 | `title_mismatch`、`wrong_image` |
| 难 | `duplicate_detail_image`、`image_quality` |

`image_quality` 的子类型为 `blur`、`occlusion`、`low_resolution`。

## 图片路径

每个当前数据集都带有自己的 `images/` 目录。JSONL 中的 `images.main.image_id` 和 `images.detail[*].image_id` 指向对应数据集目录内的实际文件；SFT 路径统一为 `data/sft/images/...`，GRPO 路径统一为 `data/GRPO/images/...`。GRPO 重构会重新生成 `IMAGE_QUALITY` 退化图，不复用旧的子类型—位置绑定。

## 目录说明

- 当前数据：`data/sft`、`data/GRPO`、`data/test`。
- 源数据与审计：`data/all_product.jsonl`、`data/raw_clean`、`data/manifests`。
- `data/GRPO_synthesis` 是 GRPO 重构的候选池；`data/sft_synthesis`、`data/prepared`、`data/highres_split` 是其他中间产物，不作为最终规范数据集。
- SFT/GRPO 入口、评估配置和基线脚本已切换到当前数据目录；启动前会在 outputs 下生成 veRL 运行时输入。
- 训练和测评直接读取三张原图，不生成或加载合成商品页面。

## Qwen3.5 runtime

The active training path targets Qwen3.5-4B. Model loading uses Transformers AutoConfig/AutoProcessor/AutoModelForImageTextToText and validates the model type before SFT, validation, test, or GRPO.

The verified Linux stack is Python 3.12, PyTorch 2.11.0 (CUDA 12.9), Transformers 5.16.1, veRL 0.8.0, and vLLM 0.20.2. Qwen3.5 Gated DeltaNet requires flash-linear-attention 0.4.2 and causal-conv1d 1.7.0; setup installs these after PyTorch with --no-build-isolation.

```bash
bash scripts/setup.sh
.venv/bin/python scripts/verify_training_stack.py
```

Run the pipeline from the repository root:

- `bash scripts/sft.sh --print-command` validates the SFT inputs and prints the veRL command.
- `bash scripts/sft.sh` trains SFT, normalizes the newest Hugging Face checkpoint, publishes `outputs/sft_qwen35_4b/latest`, and runs validation once.
- `bash scripts/basemodel.sh` runs the fixed test split with the Qwen3.5 base model; use `BASEMODEL_MODEL=outputs/sft_qwen35_4b/latest/huggingface BASEMODEL_PREDICTIONS=outputs/test/sft_predictions.jsonl BASEMODEL_FORCE=1 bash scripts/basemodel.sh` to test the normalized SFT checkpoint directly.
- `bash scripts/grpo.sh --dry-run` checks the GRPO command; `bash scripts/grpo.sh` launches it.

GRPO exports verify every referenced image with Pillow before veRL starts. A corrupt or truncated image now reports its sample ID and path at export time instead of failing inside a Ray worker.

If the fast kernels are unavailable, Qwen3.5 training and inference fail with an actionable setup message. Use `require_gated_deltanet_kernels: false` only for CPU diagnostics; it is not a supported training configuration.
