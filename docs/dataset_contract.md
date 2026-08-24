# 数据集阶段契约与 ID 追踪

本项目只维护一套由同一 ABO 候选池和同一生成协议产生的
train/validation/test。Base、SFT、GRPO、OPD 四个模型阶段使用同一份固定 test
进行最终配对评测；同一 source component、`item_id` 或 `image_id` 不得跨 split。

## 1. ID 层级与目录

```text
ABO item_id / image_id / source_component_id
    -> sample_id
        -> derived_image_id
```

- 原始商品图保存为 `data/raw/abo/images/by_id/<image_id>/image.jpg`；清单同时保存
  原始 `image_id`、S3 object key、SHA-256、pHash、宽高和下载字节数。
- 页面保存为 `data/generated/<split>/pages/<sample_id>/page.png`，其
  `derived_image_id` 为 `page:<sample_id>`。
- crop 保存为 `data/generated/<split>/crops/<sample_id>/crop_<n>.png`，其
  `derived_image_id` 为 `crop:<sample_id>:<n>`，并记录父页面 ID。
- 反事实页面使用独立 `sample_id`，记录 `counterfactual_of` 和 `pair_id`，但继承原样本
  的 source component、商品和原图 ID。

所有 SFT、OPD 和 GRPO 行都必须携带：

```json
{
  "lineage": {
    "source_product_ids": ["..."],
    "source_image_ids": ["..."],
    "derived_image_id": "page:...",
    "parent_sample_id": null
  }
}
```

## 2. 各阶段清单

| 阶段 | 训练输入 | 验证输入 | 固定测试输入 | 禁止行为 |
|---|---|---|---|---|
| 原始池 | `products_train.jsonl` | `products_validation.jsonl` | `products_test.jsonl` | 先下载图片再随机划分 |
| 页面/标签 | `samples_train.jsonl` | `samples_validation.jsonl` | `samples_test.jsonl` | 生成混合 `samples.jsonl` 供下游直接读取 |
| 反事实 | `counterfactuals_train.jsonl` | `counterfactuals_validation.jsonl` | `counterfactuals_test.jsonl` | 原样本与反事实跨 split |
| SFT | `sft_train.jsonl` | `sft_validation.jsonl` | `sft_test.jsonl` 只评测 | 训练读取 `sft_test.jsonl` |
| GRPO | `grpo_train.jsonl` | `grpo_validation.jsonl` | `grpo_test.jsonl` 只评测 | 普通 GRPO 混入反事实或 test |
| OPD | `opd_train.jsonl` | `opd_validation.jsonl` | `opd_test.jsonl` 只算 gap | teacher filter 后合并 split |
| 最终评测 | 不适用 | 冻结阈值前使用 validation | 四阶段共用 `samples_test` 与 `counterfactuals_test` | 为不同模型阶段重新抽 test |

`sft_test`、`grpo_test` 和 `opd_test` 是使用相同 test 样本构造的阶段视图，不是三套不同
测试集。最终业务、证据和反事实指标以 `samples_test` 与 `counterfactuals_test` 为唯一
真值源。

## 3. 强制检查

生成结束和训练启动时必须同时检查：

1. 文件名声明的 split 与每一行的 split 相同；
2. train/validation/test 的 `sample_id`、`derived_image_id` 不重复；
3. 三个 split 的 source component、`source_product_ids`、`source_image_ids` 两两无交集；
4. 所有图片路径存在、可解码，目录 ID 与清单 ID 一致；
5. 每个 crop 的父页面 ID 和原图 ID 可追溯；
6. SFT/OPD/GRPO 入口只接受配置规定的 train 和 validation 文件，路径或行内出现 test
   立即终止；
7. 四个阶段最终评测使用相同 `sample_id` 集合和同一份冻结 test 哈希。

