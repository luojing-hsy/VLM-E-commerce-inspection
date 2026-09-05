# 当前快照更新（2026-09-05）

直接检查 data/sft/train.jsonl 与 valid.jsonl：1000/100 行，source_product_id 为 973/88 个，
两者交集为空。训练来源出现 1/2/3 次的数量为 954/11/8，验证为 81/2/5。
wrong_image 训练 main/detail:1/detail:2 各 50 条，验证各 5 条。
image_quality 训练覆盖全部 9 个子类型/位置组合，每个 5 或 6 条。
以下 2026-08-29 内容是历史审计，不代表当前文件；当前哈希与类别计数见
[数据清单](../materials/dataset_inventory.json)，使用说明见 [复现指南](REPRODUCIBILITY.md)。

---

# SFT 数据集分析（服务器当前快照）

统计对象是服务器 `/root/autodl-tmp/vlm-qwen3vl` 中的 `data/sft/train.jsonl` 和
`data/sft/valid.jsonl`。统计日期为 2026-08-29；文件内标签使用小写，下面沿用文件中的实际值。
这份报告只描述当前文件的事实，不把未修复的问题写成已达到的均衡目标。

## 1. 规模、来源与图片完整性

| 文件 | 样本数 | 唯一 `source_product_id` | 每个来源的行数 | 图片引用 | 缺失路径 |
|---|---:|---:|---|---:|---:|
| `data/sft/train.jsonl` | 1,000 | 992 | 988 个来源各 1 行；4 个来源各 3 行 | 每行 3 张 | 0 |
| `data/sft/valid.jsonl` | 100 | 100 | 每个来源 1 行 | 每行 3 张 | 0 |

训练和验证的 `source_product_id` 交集为空。训练集虽然有 1,000 行，但并不是 1,000 个独立来源商品；其中 4 个来源各出现 3 行，应在解释样本权重时保留这一事实。

## 2. 标签与难度分布

当前难度映射为：易=`pass`、`color_mismatch`、`category_mismatch`、`material_mismatch`；中=`title_mismatch`、`wrong_image`；难=`duplicate_detail_image`、`image_quality`。

| 难度 | 标签 | Train | Validation |
|---|---|---:|---:|
| 易 | `pass` | 150 | 15 |
| 易 | `color_mismatch` | 150 | 15 |
| 易 | `category_mismatch` | 150 | 15 |
| 易 | `material_mismatch` | 150 | 15 |
| **易合计** |  | **600** | **60** |
| 中 | `title_mismatch` | 150 | 15 |
| 中 | `wrong_image` | 150 | 15 |
| **中合计** |  | **300** | **30** |
| 难 | `duplicate_detail_image` | 50 | 5 |
| 难 | `image_quality` | 50 | 5 |
| **难合计** |  | **100** | **10** |
| **总计** |  | **1,000** | **100** |

因此，当前 SFT 文件的难度比例是 60% / 30% / 10%；各难度内部的标签配额分别相等，但不同难度的单类配额不同（易/中/难分别为每类 150/150/50）。

## 3. `image_quality` 子类型与图片位置

### Train

| 子类型 | `main` | `detail:1` | `detail:2` | 合计 |
|---|---:|---:|---:|---:|
| `blur` | 17 | 0 | 0 | 17 |
| `occlusion` | 0 | 0 | 17 | 17 |
| `low_resolution` | 0 | 16 | 0 | 16 |
| **合计** | **17** | **16** | **17** | **50** |

### Validation

| 子类型 | `main` | `detail:1` | `detail:2` | 合计 |
|---|---:|---:|---:|---:|
| `blur` | 2 | 0 | 0 | 2 |
| `occlusion` | 0 | 0 | 2 | 2 |
| `low_resolution` | 0 | 1 | 0 | 1 |
| **合计** | **2** | **1** | **2** | **5** |

当前文件中子类型与位置是完全绑定的：`blur` 只出现在 `main`，`low_resolution` 只出现在 `detail:1`，`occlusion` 只出现在 `detail:2`。这是可被模型利用的 shortcut，不应解释为“子类型和位置已独立随机化”；正式训练前应重新生成或重排这部分样本。

## 4. `wrong_image` 图片位置

| 文件 | `main` | `detail:1` | `detail:2` | 合计 |
|---|---:|---:|---:|---:|
| Train | 0 (0.0%) | 28 (18.7%) | 122 (81.3%) | 150 |
| Validation | 0 (0.0%) | 8 (53.3%) | 7 (46.7%) | 15 |

训练集没有主图错图样本，且明显偏向 `detail:2`；验证集虽然两个 detail 位置接近，但仍没有 `main`。这不满足主图、detail:1、detail:2 均覆盖的要求，不能把位置识别准确率当作独立的错图识别能力。

## 5. 字段缺失与变换字段

| 文件 | `color` 为 null/空 | `material` 为 null/空 | `changed_field` 为空 |
|---|---:|---:|---:|
| Train | 510 | 744 | 550 |
| Validation | 49 | 74 | 55 |

空的 `color` 或 `material` 是来源缺失，不是违规标签。当前 SFT 文件没有缺失图片路径；每行均包含一张 `main` 和两张 `detail` 引用。

## 6. 结论与使用边界

1. 当前 SFT 的总体难度比例可复现，但 `image_quality` 存在子类型—位置绑定，`wrong_image` 缺少 `main` 错图并偏向 `detail:2`。
2. 本报告保留这些问题作为数据审计结果，不将它们包装为已经修复的分布。
3. 本轮重构动作针对 `data/GRPO`：从 `data/GRPO_synthesis` 重新按难度内标签配额抽样，并重新生成 `IMAGE_QUALITY` 组合；不会把 SFT 中的 shortcut 复制到 GRPO。
