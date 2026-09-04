# 商品多图一致性质检项目（当前版本）

## 目标

使用 Qwen3.5 对商品标题、属性和三张商品图片进行一致性判断。当前数据、路径和统计以服务器实际文件为准。

## 当前数据

原始数据为 3,986 个商品，每个商品 1 张 main、2 张 detail。当前规范数据集如下：

| 数据集 | 样本 | 商品 | 易/中/难 |
|---|---:|---:|---:|
| SFT train | 1,000 | 992 | 600/300/100 |
| SFT valid | 100 | 100 | 60/30/10 |
| GRPO train | 1,000 | 1,000 | 300/400/300 |
| GRPO valid | 100 | 100 | 30/40/30 |
| Test | 200 | 200 | 100/60/40 |

规范目录为 `data/sft`、`data/GRPO`、`data/test`，每个目录的 JSONL 与 `images/` 配套保存。

## 标签

- 易：`pass`、`color_mismatch`、`category_mismatch`、`material_mismatch`
- 中：`title_mismatch`、`wrong_image`
- 难：`duplicate_detail_image`、`image_quality`

`image_quality` 只使用 `blur`、`occlusion`、`low_resolution` 三种子类型。

## 数据规则

- 一条记录只保留一个目标标签。
- `color` 或 `material` 缺失时，不因缺失生成错误。
- 每条记录包含 main、detail:1、detail:2 三张图片。
- 训练和测评直接使用三张原图，不生成合成商品页面。
- 图片路径指向当前数据集自己的 `images/` 目录。
- SFT 每个商品最多 3 条；当前 SFT 不重新制造图片，只抽取并复制已有数据集内容。

## 当前边界

旧的历史生成目录和旧版 data/manifests/samples_*.jsonl 不属于当前规范数据集。训练入口和配置已切换到当前数据目录，模型训练结果尚未产生。项目不使用 LLM-as-a-Judge、闭源模型 API 或奖励模型。
