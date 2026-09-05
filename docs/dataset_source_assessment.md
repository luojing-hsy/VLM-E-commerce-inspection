# 数据源与当前数据集说明

## 源数据

- `data/all_product.jsonl`：3,986 条商品记录。
- `data/raw_clean`：3,986 个商品、11,958 张图片。
- 每个商品保留 1 张 main 和 2 张 detail；图片不足的记录不进入 clean 数据。
- 原始商品类别约 381 类，字段缺失按源数据保留，不补写颜色或材质。

## 当前数据集

最终使用的数据目录为：

`data/sft`、`data/GRPO`、`data/test`。

三套数据均将 JSONL 实际引用的图片放在对应的 `images/` 目录。当前重构只复用已有源数据和已有图片文件，不新增图片内容。

## 标签

当前统一使用 8 个小写标签：

`pass`、`duplicate_detail_image`、`image_quality`、`wrong_image`、`category_mismatch`、`color_mismatch`、`material_mismatch`、`title_mismatch`。

难度为：

- 易：pass、color、category、material
- 中：title、wrong image
- 难：duplicate image、image quality

## 留存目录

旧的历史生成目录和旧版 data/manifests/samples_*.jsonl 是历史生成、中间处理或旧协议留存，不作为当前规范数据集。训练配置已指向当前数据目录，运行时转换文件写入 outputs/。

## 使用边界

ABO 审计记录显示当前保守许可策略为 CC-BY-NC-4.0，官方来源许可状态为 `official_source_conflict`。数据只用于本项目约定的研究和内部验证，不将 catalog 字段与图片关联扩大解释为人工视觉真值。
