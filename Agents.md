# 项目工作约定（当前版本）

- 以服务器项目根目录 `/root/autodl-tmp/vlm-qwen3vl` 为准，不以本地副本推断数据状态。
- 当前规范数据集只有 `data/sft`、`data/GRPO`、`data/test`。
- SFT 为 train 1,000 条、valid 100 条，难度比例 6:3:1；每个商品最多保留 3 条。
- 难度映射固定为：易 `pass/color_mismatch/category_mismatch/material_mismatch`；中 `title_mismatch/wrong_image`；难 `duplicate_detail_image/image_quality`。
- `image_quality` 子类型固定为 `blur`、`occlusion`、`low_resolution`。
- JSONL 的图片必须指向对应数据集的 `images/` 目录；不要把旧路径当作当前数据路径。
- `data/*_synthesis`、`data/prepared`、`data/highres_split` 和旧 manifest 仅作历史或审计留存。
- 未明确要求时不要修改训练代码、模型配置或图片内容；当前文档只描述服务器已经存在的版本。
