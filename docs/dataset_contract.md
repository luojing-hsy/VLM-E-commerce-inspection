# 数据集阶段契约与 ID 追踪

9,000 张 ABO 原始图片先按来源关系图的连通分量分配到 `sft`、`grpo`、`opd`、`test` 四个互斥数据池，再渲染页面。任何原始 `image_id`、ABO `item_id`、`source_component_id` 及 pHash 近重复关系都不得跨数据池。

Base 不是训练阶段，只在唯一的固定 test 上运行推理。主训练路线中，SFT 只读取 `sft` 池；Stage 2 把 `grpo` 与 `opd` 池合并为一个联合 manifest，但每行仍保留原始 `dataset_stage` 和独立损失掩码。Base、SFT 和 SFT+Joint 最终都使用相同的 `samples_test.jsonl` 与 `counterfactuals_test.jsonl`，不生成阶段专属 test。

## 1. ID 层级与平铺目录

```text
ABO item_id / image_id / source_component_id
    -> dataset_stage
        -> sample_id
            -> derived_image_id
```

- 原始商品图统一平铺在 `data/raw/abo/images/`，文件名为 `<safe_image_id>.<ext>`；不会为单张图片创建目录。
- 渲染页平铺在 `data/generated/<dataset_stage>/<split>/pages/<sample_id>.png`。
- crop 平铺在 `data/generated/<dataset_stage>/<split>/crops/<sample_id>__crop_<n>.png`。
- `source_images.jsonl` 保存 ABO 原始 `image_id`、本地文件名、所属阶段、split、S3 object key、SHA-256、pHash、宽高和下载字节数。
- 反事实使用独立 `sample_id`，通过 `counterfactual_of` 和 `pair_id` 指回原样本，并继承完全相同的来源 ID 与 `dataset_stage`。

所有训练导出行必须携带 `dataset_stage` 和以下 lineage：

```json
{
  "lineage": {
    "dataset_stage": "sft",
    "source_product_ids": ["..."],
    "source_image_ids": ["..."],
    "derived_image_id": "page:...",
    "parent_sample_id": null
  }
}
```

## 2. 阶段用途

| 数据池 | 原始导出 | 主训练入口中的用途 |
|---|---|---|
| SFT | `sft_{train,validation}.jsonl` | Stage 1 结构化监督训练 |
| GRPO | `grpo_{train,validation}.jsonl` | Stage 2 规则 reward 子池 |
| OPD | `opd_{train,validation}.jsonl` | Stage 2 privileged-crop 蒸馏子池 |
| Test | `samples_test.jsonl`、`counterfactuals_test.jsonl` | 三个模型阶段共用的最终评测 |

`samples_train.jsonl` 与 `samples_validation.jsonl` 只是生成审计总账，包含三个训练池的页面；它们不是训练入口。

Stage 2 的主入口读取 `joint_train.jsonl` 与 `joint_validation.jsonl`。联合行满足：

- GRPO 行：`dataset_stage=grpo`、`training_stage=joint`、`opd_enabled=false`；
- OPD 行：`dataset_stage=opd`、`training_stage=joint`、`opd_enabled=true`；
- `prompt` 始终只有一张完整页；
- 只有 OPD 行包含 `teacher_prompt`，顺序固定为完整页后跟证据 crop；
- 两类行都保留规则 ground truth 与完整 lineage；
- train 和 validation 各自必须同时包含两个子池。

## 3. 强制检查

1. 每个 `source_image_id` 只能属于一个 `dataset_stage`；
2. SFT、GRPO、OPD、test 的来源图片 ID、商品 ID、来源组件两两无交集；
3. train/validation 的 `sample_id` 和 `derived_image_id` 不重复；
4. test 只能出现在 test 数据池，且只有一份固定 test 清单；
5. 所有图片路径存在、可解码，文件名中的 ID 与清单一致；
6. 每个 crop 的父页面 ID 和原图 ID 可追溯；
7. 训练入口遇到错误阶段或 test 行立即终止；
8. 联合导出不得把 crop 放入 student prompt，且 RL-only 行的 OPD token 权重必须全为 0；
9. Base 只做推理；三个模型阶段最终评测使用同一 test 清单哈希。
