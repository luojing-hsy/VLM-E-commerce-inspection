# 当前数据集契约

## 规范目录

- `data/sft/train.jsonl`：1,000 条，992 个商品。
- `data/sft/valid.jsonl`：100 条，100 个商品。
- `data/joint/train.jsonl`：1,000 条，1,000 个商品。
- `data/joint/valid.jsonl`：100 条，100 个商品。
- `data/test/test.jsonl`：200 条，200 个商品。

每条记录对应一个商品的三张图片：`main`、`detail:1`、`detail:2`。图片实际文件放在同一数据集下的 `images/`。

## 记录字段

当前记录保留：

`product_id`、`source_product_id`、`dataset`、`sample_index`、`difficulty`、`violation_type`、`title`、`category`、`color`、`material`、`images`、`changed_field`、`title_audit`。

图片结构为：

```json
{
  "images": {
    "main": {"image_id": "data/sft/images/product_0003_main.jpg"},
    "detail": [
      {"image_id": "data/sft/images/product_0003_detail_1.jpg"},
      {"image_id": "data/sft/images/product_0003_detail_2.jpg"}
    ]
  }
}
```

`color`、`material` 可以为空；空值不是错误。当前 JSONL 是训练样本记录，不是模型最终输出 JSON。

## 标签和难度

- 易：`pass`、`color_mismatch`、`category_mismatch`、`material_mismatch`
- 中：`title_mismatch`、`wrong_image`
- 难：`duplicate_detail_image`、`image_quality`
- 质量子类型：`blur`、`occlusion`、`low_resolution`

SFT 每个商品允许 0–3 条；当前训练集最多 3 条，验证集最多 1 条。
