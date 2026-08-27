# Qwen3-VL 商品多图一致性质检（当前版本）

## 当前口径

项目以服务器目录 `/root/autodl-tmp/vlm-qwen3vl` 为准。原始清单为 `data/all_product.jsonl`，共 3,986 个商品；清洗后每个商品保留 1 张 main 和 2 张 detail，共 11,958 张图片。

当前可直接核对的数据集：

| 数据集 | 文件 | 样本 | 商品 | 易/中/难 |
|---|---|---:|---:|---:|
| SFT train | `data/sft/train.jsonl` | 1,000 | 992 | 600/300/100 |
| SFT valid | `data/sft/valid.jsonl` | 100 | 100 | 60/30/10 |
| Joint train | `data/joint/train.jsonl` | 1,000 | 1,000 | 300/400/300 |
| Joint valid | `data/joint/valid.jsonl` | 100 | 100 | 30/40/30 |
| Test | `data/test/test.jsonl` | 200 | 200 | 100/60/40 |

SFT 的类别配额为：训练集易/中类各 150、难类各 50；验证集易/中类各 15、难类各 5。SFT 每个商品保留 0–3 条，当前训练集最多 3 条、验证集最多 1 条。Joint 和 Test 按可用性抽样，不强制八类等量。

## 标签与难度

| 难度 | 标签 |
|---|---|
| 易 | `pass`、`color_mismatch`、`category_mismatch`、`material_mismatch` |
| 中 | `title_mismatch`、`wrong_image` |
| 难 | `duplicate_detail_image`、`image_quality` |

`image_quality` 的子类型为 `blur`、`occlusion`、`low_resolution`。

## 图片路径

每个当前数据集都带有自己的 `images/` 目录。JSONL 中的 `images.main.image_id` 和 `images.detail[*].image_id` 指向对应数据集目录内的实际文件；SFT 路径统一为 `data/sft/images/...`。本轮不重新生成图片。

## 目录说明

- 当前数据：`data/sft`、`data/joint`、`data/test`。
- 源数据与审计：`data/all_product.jsonl`、`data/raw_clean`、`data/manifests`。
- `data/*_synthesis`、`data/prepared`、`data/highres_split`：历史生成或中间产物，不作为当前规范数据集。
- SFT/Joint 入口、评估配置和基线脚本已切换到当前数据目录；启动前会在 outputs 下生成 veRL 运行时输入。
