# 基于 Qwen3-VL 的低成本电商商品页质检后训练项目

## 1. 项目结论

本项目实现一个面向电商商品详情页的视觉质检模型。模型读取商品图片、标题、规格表和包装标签，判断商品是否应当通过、转人工审核或拒绝，并给出可验证的冲突字段与视觉证据。

主训练路线固定为：

```text
程序化构建数据
    → Qwen3-VL BF16 LoRA SFT
    → Cost-sensitive GRPO
    → Vision-OPD-inspired regional-to-global self-distillation
    → 固定测试集评测
```

项目遵守以下边界：

- 不新增人工标注；
- 不使用 LLM-as-a-Judge；
- 不使用闭源模型 API；
- 不训练奖励模型；
- 不比较其他 VLM；
- 只使用一个 Qwen3-VL 规格；
- 不把 JSON、回答长度或模型自报置信度作为业务奖励；
- 不把 PPT、网站或演示系统列为必做交付物。

默认硬件假设为单张 24 GB GPU，主模型选用 `Qwen/Qwen3-VL-4B-Instruct`。如果实际显存不足，只调整 batch、图像分辨率和梯度累积，不更换模型做横向比较。

## 2. 成功标准

项目完成必须同时满足：

1. 固定随机种子可以重新生成同一批数据和标签；
2. 数据集包含本文确定的全部违规类别；
3. 任一来源商品、同款商品族或近重复图片不会跨 train/validation/test；
4. 所有违规标签、修改字段、证据区域及证据来源均由程序自动生成并通过校验；
5. SFT 后结构化结果可解析率达到 98% 以上；
6. GRPO 后目标类别先验下的业务风险低于 SFT；
7. 最终 OPD 后完整页主指标不下降，regional-to-global gap 相比 SFT+GRPO 达到预先冻结的最小改善，且业务风险相对 GRPO 满足预先冻结的 non-inferiority margin；
8. 最终严重违规漏放率、证据定位和反事实一致性达到预先冻结的最小改善；
9. 完整训练与评测不调用任何 Judge 模型。

第 6～8 项的主指标、最小有意义提升、OPD 视觉与业务风险 non-inferiority margin、每类最小样本数和目标类别先验，在完成原始模型 baseline 后、开始正式训练前写入 `configs/eval.yaml` 并冻结。最终结果同时报告点估计与 paired bootstrap 95% CI；test 只在配置冻结后运行一次，不用于调参。

## 3. 业务输入与输出

### 3.1 输入

每条样本由一张程序渲染的商品详情页组成，页面至少包含：

- 商品主图；
- 商品标题；
- 品牌与类别；
- 规格属性表；
- 包装或商品标签区域；
- 最多三张商品详情图。

页面中的文字和布局均属于图像内容，模型不能直接读取对应的结构化元数据。

### 3.2 输出协议

模型输出以下字段：

```json
{
  "schema_version": "1.0",
  "decision": "reject",
  "violation_type": "ATTRIBUTE_CONFLICT",
  "field": "model",
  "listed_value": "Model Y",
  "observed_value": "Model X",
  "evidence": [
    {
      "role": "listed_value",
      "image_ref": "page",
      "region_type": "bbox",
      "bbox_norm": [505, 288, 688, 346]
    },
    {
      "role": "observed_value",
      "image_ref": "page",
      "region_type": "bbox",
      "bbox_norm": [412, 533, 583, 602]
    }
  ]
}
```

JSON 只是工程接口，不进入 reward。格式通过 SFT、Pydantic 校验和推理阶段 constrained decoding 处理。

协议约束如下：

- `bbox_norm` 为完整原始页面上的 `xyxy` 相对坐标，整数范围 `[0, 1000]`；左上边界包含、右下边界不包含；
- resize、letterbox、crop 和视觉增强必须保存正反变换；计算 IoU 前统一逆变换到完整原始页面像素；
- `evidence` 始终为列表。`bbox` 证据填写单个 `image_ref` 与 `bbox_norm`，图片索引填写单个 `image_ref`，重复图使用 `region_type: "image_pair"` 与 `image_refs: ["gallery:1", "gallery:2"]`，缺失字段使用 `region_type: "missing_field"`；
- `PASS` 的 `violation_type`、`field`、两个 value 均为 `null`，`evidence` 为 `[]`；其他类别不适用的字段也使用 `null`，不得用空字符串代替；
- parser 接受字段顺序和空白差异，但输出对象必须通过同一版本 schema；不受支持的 `schema_version` 直接记为不可解析。

不训练自由文本长理由。最终需要自然语言解释时，根据结构化字段使用固定模板生成：

```text
检测到型号属性冲突：页面标注为 Model Y，标签证据显示为 Model X。
```

## 4. 固定违规类别 V1

V1 使用单标签设计：一个违规样本只注入一种违规。`PASS` 是正常类别，不属于违规。

| 类别 | 定义 | 自动构造方式 | 决策 | 自动证据 |
|---|---|---|---|---|
| `PASS` | 页面信息完全一致 | 使用原始元数据渲染 | `pass` | 无 |
| `PRODUCT_MISMATCH` | 标题/属性与商品图片不是同一商品 | 标题和规格来自商品 A，图片与标签来自商品 B | `reject` | 标题区与商品图区 |
| `ATTRIBUTE_CONFLICT` | 两处结构化属性值冲突 | 修改颜色、尺寸、材质或型号等实际存在字段中的一个 | `reject` | 被修改字段与渲染标签区域 |
| `TEXT_LABEL_CONFLICT` | 页面宣传文字与渲染标签冲突 | 修改标题或宣传区中的可验证值，保留渲染标签真值 | `reject` | 宣传文字区与标签文字区 |
| `MISSING_REQUIRED_FIELD` | 缺少类别要求的必要字段 | 删除该类别的一项必填属性 | `review` | 缺失字段名及属性表区域 |
| `IMAGE_QUALITY` | 关键商品图严重模糊、遮挡或低清 | 对指定图片执行参数化退化 | `review` | 受损图片索引和区域 |
| `IRRELEVANT_IMAGE` | 图集中存在无关商品图 | 插入同一 split 内其他商品的图片 | `review` | 异常图片索引 |
| `DUPLICATE_IMAGE` | 图集中存在重复或近重复图片 | 复制已有图片或施加不改变语义的轻微变换 | `review` | 图片对 |

类别优先级由注入器显式保证：图片来源被替换时只标为 `PRODUCT_MISMATCH`；结构化属性表与渲染标签冲突时标为 `ATTRIBUTE_CONFLICT`；结构化属性保持正确、仅标题或宣传区与渲染标签冲突时标为 `TEXT_LABEL_CONFLICT`。无关图、重复图和质量退化分别使用独立变换。任何 eligibility 检查发现第二种违规时丢弃样本，而不是事后选择标签。

暂不加入以下类别：

- 法律意义上的虚假宣传；
- 商标侵权和假货风险；
- 敏感内容；
- 主观审美或平台风格规范。

这些类别无法在没有人工审核、平台规则库或可靠外部验证器的条件下获得可信标签。

## 5. 数据集构建

### 5.1 数据来源

使用具有公开研究许可的商品图片与元数据作为原始商品池，例如 Amazon Berkeley Objects（ABO）。ABO 暂按 [AWS Registry](https://registry.opendata.aws/amazon-berkeley-objects/) 标注的 CC BY-NC 4.0 管理，仅用于非商业研究并保留完整署名；下载时保存归档内 LICENSE、来源 URL、访问日期和文件哈希。仓库默认只保存下载脚本、数据清单和生成标注，不重新分发受限制的原图。若实际下载归档的许可证与来源页不一致，在取得数据维护方书面澄清前不得放宽使用范围。

ABO 元数据只能证明 catalog 字段及图片的来源关系，不能证明天然包装图上出现了容量、SPF 等文字。V1 因此只使用实际存在且通过 eligibility 检查的 catalog 字段，并由页面渲染器生成可读标签区域；项目结论限定为“合成商品文档的一致性、OCR 与 grounding”，不声称已经验证天然包装文字理解。不得用缺失字段或根据标题猜测出的值生成真值。

目标规模：

```text
原始商品：3,000～5,000 个
最终样本：8,000～12,000 条
train / validation / test：80% / 10% / 10%
```

该比例是数据生成预算，不是固定统计结论。baseline 后应按严重漏放率等主指标所需置信区间宽度反推各类 test 最小样本数；若 10% 不足，优先增加测试商品而不是复用 validation。

### 5.2 原始商品格式

```json
{
  "product_id": "product_00123",
  "images": ["raw/images/product_00123_0.jpg"],
  "title": "Brand A Blue Cotton Shirt Model X",
  "category": "apparel",
  "brand": "Brand A",
  "attributes": {
    "color": "blue",
    "material": "cotton",
    "model": "Model X",
    "dimensions": null
  },
  "source": "abo",
  "license": "CC-BY-NC-4.0",
  "source_record_id": "abo_listing_id",
  "source_archive_sha256": "..."
}
```

每个样本还必须在 manifest 中记录证据来源：

```json
{
  "evidence_source": "rendered_text",
  "source_field": "color",
  "source_confidence": "programmatic_exact"
}
```

`catalog_image` 只表示图片归属关系，`gallery_relation` 只表示图集重复/无关关系；二者都不能被解释为天然图片中的 OCR 真值。

### 5.3 页面渲染

使用 Pillow 渲染，不依赖浏览器。每个模板通过 `ImageDraw.textbbox` 记录所有区域坐标：

```text
title_bbox
product_image_bbox
attribute_table_bbox
attribute_value_bboxes
label_bbox
gallery_image_bboxes
```

至少实现四种模板，变化以下因素：

- 标题和图片相对位置；
- 属性表方向；
- 字体、字号和颜色；
- 页面尺寸和背景；
- 标签区域大小；
- 商品图数量。

测试集保留训练中未使用的模板、字体组合与渲染参数组合，检查模型是否只学习固定布局。JPEG/WebP 压缩、截图缩放、轻透视和色偏等增强必须同步更新 bbox 及坐标变换记录；无法精确逆变换的增强不得用于 evidence 样本。

### 5.4 数据划分顺序

必须先建立来源关系图并按连通分量划分，再渲染与增强：

```text
raw products
    → listing_id ↔ image_id ↔ source/donor product_id 建图
    → 合并 canonical product family 与图片 pHash 近重复边
    → split connected components
    → render clean pages
    → inject violations
    → create crops and counterfactuals
```

`PRODUCT_MISMATCH` 的供体 B 只能从 A 所在 split 选择，并把 A、B 全部写入 `source_product_ids`。禁止先增强再随机划分，否则同一商品、供体图片、同款变体或近重复图片会泄漏到测试集。

供体难度分为：easy（跨类目）、medium（同类目但品牌或型号不同）和 hard（同类目、外观相近、仅局部属性不同）。validation/test 以 medium+hard 为主，easy 只用于训练早期。注入前必须验证供体只产生目标违规，不引入第二种冲突。

### 5.5 违规注入记录

每次变换必须同时产生机器可读的审计记录：

```json
{
  "sample_id": "sample_000001",
  "source_product_ids": ["product_00123"],
  "split": "train",
  "template_id": "template_02",
  "seed": 42,
  "violation_type": "ATTRIBUTE_CONFLICT",
  "decision": "reject",
  "transform": "replace_attribute",
  "changed_fields": {
    "model": {
      "original": "Model X",
      "modified": "Model Y"
    }
  },
  "evidence": [
    {
      "role": "listed_value",
      "value": "Model Y",
      "image_ref": "page",
      "region_type": "bbox",
      "bbox_norm": [505, 288, 688, 346],
      "evidence_source": "rendered_text",
      "source_field": "model"
    },
    {
      "role": "observed_value",
      "value": "Model X",
      "image_ref": "page",
      "region_type": "bbox",
      "bbox_norm": [412, 533, 583, 602],
      "evidence_source": "rendered_text",
      "source_field": "model"
    }
  ]
}
```

### 5.6 反事实样本

每个一致性违规样本生成一个最小修改的反事实版本：

```text
原样本：页面 Model Y，标签 Model X → ATTRIBUTE_CONFLICT
反事实：页面 Model X，标签 Model X → PASS
```

反事实对必须保持模板、商品图片、字体和其他属性不变，只修改决定标签的证据。V1 将反事实对用于 SFT 数据增强和固定评测，不接入 GRPO 训练 reward；veRL 的普通 grouped rollout 不保证原样本与反事实样本及其 completion group 成对同批，后续只有在实现并验证 paired sampler/rollout 后才可启用 paired GRPO。

适用类别：

- `PRODUCT_MISMATCH`；
- `ATTRIBUTE_CONFLICT`；
- `TEXT_LABEL_CONFLICT`。

### 5.7 证据 crop

对具有局部视觉证据的样本，从记录的 bbox 生成 1.5～2.0 倍上下文裁剪：

```text
full_image：完整商品页
crop_image：包含 evidence bbox 的高清区域
```

crop 只能由渲染器坐标生成，不允许人工选择。

### 5.8 数据校验

生成后必须运行以下检查：

- 图片路径存在且可解码；
- bbox 位于图片边界内；
- 被修改值与原始值不同；
- `PASS` 样本没有变换记录；
- 每条违规样本只包含一种违规；
- 反事实样本的目标字段已恢复一致；
- train/validation/test 的全部 `source_product_ids` 交集为空；
- canonical product family 不跨 split，图片 pHash 不存在跨 split 近重复；
- 所有类别数量达到配置下限；
- crop 覆盖对应 evidence bbox；
- 坐标归一化、resize/letterbox 逆变换及 crop→page 映射可往返；
- 每个 evidence 的来源类型、来源字段与类别 eligibility 一致；
- 固定 seed 的数据哈希一致。

允许对少量生成页面做不参与评分的人工工程浏览，用于发现字体溢出、遮挡和渲染异常；人工浏览结果不得修改单条训练/评测标签。若完全不进行人工浏览，报告必须注明标签正确性仅相对于程序化生成规范成立。

## 6. 目标仓库结构

```text
vlm-qwen3vl/
├── configs/
│   ├── data.yaml
│   ├── sft.yaml
│   ├── opd.yaml
│   ├── grpo.yaml
│   └── eval.yaml
├── src/
│   ├── data/
│   │   ├── prepare_products.py
│   │   ├── render_page.py
│   │   ├── inject_violation.py
│   │   ├── build_counterfactual.py
│   │   ├── build_crops.py
│   │   ├── export_sft.py
│   │   ├── export_opd.py
│   │   ├── export_grpo.py
│   │   └── validate_dataset.py
│   ├── training/
│   │   ├── train_sft.py
│   │   ├── train_opd.py
│   │   ├── train_grpo.py
│   │   └── opd_loss.py
│   ├── rewards/
│   │   ├── parser.py
│   │   ├── action_cost.py
│   │   ├── type_reward.py
│   │   ├── evidence_reward.py
│   │   ├── value_reward.py
│   │   └── composite.py
│   └── evaluation/
│       ├── evaluate.py
│       ├── metrics.py
│       ├── counterfactual.py
│       └── slices.py
├── tests/
│   ├── test_violation_generation.py
│   ├── test_dataset_split.py
│   ├── test_value_normalization.py
│   ├── test_rewards.py
│   └── test_opd_loss.py
├── data/
│   ├── raw/
│   ├── generated/
│   └── manifests/
├── outputs/
│   ├── baseline/
│   ├── sft/
│   ├── opd/
│   ├── grpo/
│   └── evaluation/
├── requirements.in
├── requirements.lock
└── VLM_POST_TRAINING_PROJECT.md
```

## 7. 环境与依赖

建议依赖：

```text
Python 3.11
PyTorch
Transformers
PEFT
Accelerate
veRL
Ray
vLLM
Pillow
Pydantic
NumPy
SciPy
RapidFuzz
PyYAML
pytest
```

Qwen3-VL 版本必须与 Transformers、veRL 和 rollout 后端版本匹配。首次搭建时以 Qwen 官方微调仓库与 veRL 的已验证 commit 为基线，锁定 PyTorch、Transformers、PEFT、Accelerate、veRL、Ray、vLLM 以及实际启用的 attention 实现；直接依赖写入 `requirements.in`，完整解析结果与仓库 commit 写入 `requirements.lock`。SFT、GRPO actor 与最终 OPD 均以 BF16 加载，GRPO 使用 FSDP2 + LoRA 和 vLLM rollout，不启用 4-bit/8-bit 量化，也不依赖 bitsandbytes。不得在规格阶段臆造版本号，锁文件必须由实际通过导入和最小前向 smoke test 的 Linux CUDA 环境生成。

启动训练前必须保存模型、processor、chat template、prompt、decoding 配置、依赖 lock hash 和数据 manifest hash。数据 collator 必须断言图像 token、assistant label mask 和 completion 未被静默截断。

计划中的统一命令：

```powershell
python -m pytest -q

python -m src.data.prepare_products --config configs/data.yaml
python -m src.data.render_page --config configs/data.yaml
python -m src.data.validate_dataset --config configs/data.yaml

accelerate launch -m src.training.train_sft --config configs/sft.yaml
python -m src.training.train_grpo --config configs/grpo.yaml
accelerate launch -m src.training.train_opd --config configs/opd.yaml

python -m src.evaluation.evaluate --config configs/eval.yaml
```

这些命令是目标接口；实现对应脚本后，不再增加第二套 Notebook 流程。

## 8. Stage 1：Baseline 与 SFT

### 8.1 Baseline

在任何训练前，用相同 prompt、图片分辨率和解码参数运行原始 Qwen3-VL，保存：

```text
outputs/baseline/predictions.jsonl
outputs/baseline/metrics.json
```

Baseline 是验证微调效果所必需的阶段检查，不属于额外模型比较。

### 8.2 SFT 数据

SFT 目标由数据生成器直接构造，不生成自由文本 CoT：

```json
{
  "image": "data/generated/train/sample_000001.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n检查商品页是否存在违规，并输出规定字段。"
    },
    {
      "from": "gpt",
      "value": "{\"schema_version\":\"1.0\",\"decision\":\"reject\",\"violation_type\":\"ATTRIBUTE_CONFLICT\",\"field\":\"model\",\"listed_value\":\"Model Y\",\"observed_value\":\"Model X\",\"evidence\":[{\"role\":\"listed_value\",\"image_ref\":\"page\",\"region_type\":\"bbox\",\"bbox_norm\":[505,288,688,346]},{\"role\":\"observed_value\",\"image_ref\":\"page\",\"region_type\":\"bbox\",\"bbox_norm\":[412,533,583,602]}]}"
    }
  ]
}
```

### 8.3 推荐配置

```yaml
model_name_or_path: Qwen/Qwen3-VL-4B-Instruct
precision: bf16
quantization: none
gradient_checkpointing: true
freeze_vision_encoder: true
train_mm_projector: false
lora_target_scope: llm_attention
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 1.0e-5
num_train_epochs: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
max_sequence_length: 2048
max_output_length: 256
min_pixels: 784
max_pixels: 50176
```

`lora_target_modules` 必须在启动时与 `named_modules()` 实际匹配，并断言没有命中视觉 encoder 或 multimodal merger/projector。`min_pixels/max_pixels` 是首版预算配置，不得让处理后的最小证据文字或图像 token 被截断；collator 的相关断言失败时直接终止训练。

如果证据定位在 SFT 后完全没有改善，才进行一次受控实验：保持 Vision Encoder 冻结，只解冻多模态投影层，并使用低于 LoRA 的学习率。该操作不是默认步骤。

### 8.4 SFT 验证

- 训练 loss 正常下降且无 NaN；
- validation 结构化结果可解析率大于 98%；
- Macro-F1 和字段抽取准确率高于原始模型；
- 通用 `PASS` 样本没有明显全部判成违规的类别坍塌。

## 9. Stage 3：Vision-OPD-inspired regional-to-global self-distillation

### 9.1 目的

SFT 解决任务定义和输出协议，GRPO 优化成本敏感业务策略，最终 OPD 再解决 RL 后完整页面中局部小证据容易被忽略或退化的问题。

本项目采用受 Vision-OPD 启发的 regional-to-global 自蒸馏。由于 teacher 输入使用完整页加 crop、teacher 冻结且 student 不接收 ROI 提示，这不是原始 Vision-OPD 的等价复现，实验结论只归因于本项目的具体实现：

```text
Student：完整商品页
Teacher：完整商品页 + 高清 evidence crop
Rollout：由 Student 当前策略生成
监督：Teacher 在 Student 前缀上的 token 分布
```

Teacher 不是 Judge。它不输出“好/坏”评分，只提供下一 token 概率分布。

crop-local 坐标与完整页坐标不在同一坐标系。V1 仅蒸馏 `decision`、`violation_type`、`field`、`listed_value` 和 `observed_value` 等语义 token，不蒸馏 bbox token。只有 teacher evidence 经 crop→page 映射并通过完整页坐标验证后，后续版本才可单独启用 bbox 蒸馏。

### 9.2 Teacher 与 Student

```text
共享底座：Qwen3-VL-4B
Teacher adapter：冻结的 GRPO LoRA
Student adapter：从同一 GRPO LoRA 初始化，可训练
```

Teacher 前向使用 `torch.no_grad()`。单个 base 权重通过 PEFT 加载两个 adapter，避免保存两个完整模型。

### 9.3 OPD 数据过滤

只使用：

- `PRODUCT_MISMATCH`；
- `ATTRIBUTE_CONFLICT`；
- `TEXT_LABEL_CONFLICT`；
- 证据区域面积占完整页面 30% 以下的样本。

Teacher 先在 crop 条件下生成一次结果。只有通过现有规则验证器的样本才进入 OPD：

```text
teacher decision 正确
teacher violation_type 正确
teacher observed_value 正确
teacher evidence 角色完整
teacher evidence 经 crop→page 映射后位于目标区域
```

该过滤不使用 LLM Judge。

### 9.4 OPD 损失

Student 生成：

\[
y \sim \pi_S(y\mid x_{full}, q)
\]

在每个学生前缀上计算：

\[
p_T^t=\pi_T(\cdot\mid x_{full},x_{crop},q,y_{<t})
\]

\[
p_S^t=\pi_S(\cdot\mid x_{full},q,y_{<t})
\]

训练目标：

\[
L_{OPD}=\frac{1}{\sum_t w_t}\sum_t w_t
KL(p_T^t\parallel p_S^t)
\]

为了降低显存，只对 teacher/student top-k logits 的并集计算 KL，并增加一个 `other` 概率桶。首版预算配置固定为 `top_k=64`；提高 K 或比较 JSD 只作为后续消融，不是执行主线的阻断项。

top-k 实现必须验证 teacher/student 概率各自和为 1、`other` 尾部质量非负、相同分布时 loss 近似 0，并在极端 logits 下无 NaN/Inf。

### 9.5 Token 权重

| Token 内容 | 权重 |
|---|---:|
| `observed_value` | 2.0 |
| `evidence bbox` | 0.0 |
| `violation_type` | 1.5 |
| `listed_value` | 1.5 |
| 其他语义 token | 1.0 |
| 固定标点、括号、JSON key | 0.0 |

这不是格式奖励。目的只是避免大量固定接口 token 稀释视觉蒸馏信号。

### 9.6 OPD 配置

```yaml
precision: bf16
quantization: none
student_checkpoint: outputs/grpo/best
teacher_checkpoint: outputs/grpo/best
teacher_adapter_frozen: true
student_adapter_trainable: true
num_rollouts_per_prompt: 2
temperature: 0.7
top_p: 0.95
max_completion_length: 256
top_k_logits: 64
learning_rate: 2.0e-6
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

### 9.7 OPD 验证

在同一批细粒度样本上分别输入完整页和 crop：

\[
Gap=M_{crop}-M_{full}
\]

其中 `M` 是字段值准确率和违规类型准确率的平均。同时报告 `M_full`、`M_crop`、`Gap` 的 paired 95% CI 和相对恢复率：

\[
Recovery=\frac{M_{full,post}-M_{full,pre}}{M_{crop,pre}-M_{full,pre}}
\]

OPD 成功的主要标准是完整页指标满足预先冻结的 non-inferiority margin，`Gap` 达到预先冻结的最小改善，且目标先验下的业务风险相对 GRPO 满足预先冻结的 non-inferiority margin；不能用 crop 指标变差造成的 gap 缩小宣称成功。

如果 crop teacher 本身错误率高，停止 OPD，先修复数据、SFT 或 GRPO；不得让错误 teacher 继续蒸馏。

## 10. Stage 2：Cost-sensitive GRPO

### 10.1 Reward 原则

Reward 只衡量：

- 业务动作是否符合成本；
- 违规类型是否正确；
- 证据是否指向真实区域；
- 视觉值是否读取正确。

明确不包含：

- JSON 格式分；
- 回答长度分；
- 自报 confidence；
- 自由文本风格分；
- LLM Judge 分数。

### 10.2 动态组合

并非所有 reward 都适用于所有类别：

\[
R=\frac{\sum_k m_k w_k r_k}{\sum_k m_k w_k}
\]

其中 `m_k∈{0,1}` 表示该项是否适用。

默认权重：

| Reward | 权重 |
|---|---:|
| 业务动作 `r_action` | 0.40 |
| 违规类型 `r_type` | 0.15 |
| 证据 `r_evidence` | 0.25 |
| 属性值 `r_value` | 0.20 |

### 10.3 业务动作 Reward

先定义非负业务成本矩阵 `C[y,a]`：

| 真实\预测 | `pass` | `review` | `reject` |
|---|---:|---:|---:|
| `pass` | 0.00 | 0.60 | 0.90 |
| `review` | 0.90 | 0.00 | 0.65 |
| `reject` | 1.00 | 0.40 | 0.00 |

训练时将成本仿射映射为有界 reward：

\[
r_{action}=1-2C[y,a]
\]

理由：

- 将应拒绝商品直接放行是最高风险；
- 将应拒绝商品转人工仍能拦截风险，因此成本低于直接放行；
- 将正常商品转人工只增加审核成本，因此轻度扣分；
- 将正常商品拒绝会造成商家损失，因此重度扣分。

成本值是项目假设，必须集中保存在配置文件中。评测中的业务风险定义为：

\[
Risk=\sum_y P_{target}(y)\sum_a P(a\mid y)C[y,a]
\]

`P_target(y)` 必须在 validation 调参前写入 `configs/eval.yaml`；如果业务先验未知，则报告多组预先声明的先验敏感性分析。类别均衡集上的平均 reward 只用于诊断，不能称为业务风险。

### 10.4 违规类型 Reward

```text
类型完全正确：1.0
同一大类但子类错误：0.25
完全错误：0.0
```

大类：

```text
consistency：PRODUCT_MISMATCH / ATTRIBUTE_CONFLICT / TEXT_LABEL_CONFLICT
completeness：MISSING_REQUIRED_FIELD
quality：IMAGE_QUALITY / IRRELEVANT_IMAGE / DUPLICATE_IMAGE
normal：PASS
```

业务动作决定是否放行，违规类型决定后续路由和修复方式，因此二者分别计算。

### 10.5 证据 Reward

| 类别 | 计算方式 |
|---|---|
| `PRODUCT_MISMATCH` | 标题区和商品图区的 region-set 匹配 |
| `ATTRIBUTE_CONFLICT` | 预测框与值区域 bbox 的 IoU |
| `TEXT_LABEL_CONFLICT` | evidence 角色与 bbox IoU |
| `MISSING_REQUIRED_FIELD` | 缺失字段名精确匹配 |
| `IMAGE_QUALITY` | 受损图片索引精确匹配 |
| `IRRELEVANT_IMAGE` | 异常图片索引匹配 |
| `DUPLICATE_IMAGE` | 图片对匹配 |
| `PASS` | 不计算 |

单框使用连续 IoU。多框使用 Hungarian matching 后计算平均 IoU。连续值比 `IoU>0.5` 的二值分数提供更平滑的学习信号。

### 10.6 属性值 Reward

只对存在可见字段值的样本启用。比较前进行标准化：

```text
0.25 m == 25 cm
MODEL-X == Model X
navy blue == 深蓝
```

评分：

```text
listed_value 和 observed_value 都正确：1.0
只正确一个：0.5
只判断“存在冲突”但未读出值：0.3
值错误或不可验证：0.0
```

这项奖励用于区分“真正读出图片文字”和“碰巧猜对违规类别”。

### 10.7 反事实固定评测

反事实不参与 V1 GRPO reward，只在固定评测中成对计算：

```text
两条决策和类别都正确：1.0
只正确一条：0.2
两条输出相同但 GT 不同：0.0
两条都错误：0.0
```

如果模型忽略视觉证据，原样本和反事实样本通常会输出相同结果，因此该指标可以检测 image-agnostic shortcut。后续若实现 paired GRPO，必须由自定义 sampler/rollout 保证原样本、反事实样本及各自 `G` 个 completion 同批且有稳定配对 ID，并另行增加配对完整性测试。

### 10.8 证据门控

对于要求可见证据的 `review/reject` 样本：

```text
if r_evidence < 0.3:
    total_reward = min(total_reward, 0.45)
```

这不是额外幻觉惩罚。它表示一个没有可靠证据的高风险结论不能得到高业务收益。

### 10.9 无效输出处理

Reward parser 应宽容处理空格、字段顺序和代码块，不奖励特定序列化形式。

如果无法获得 `decision`：

```text
r_action = -1
其他适用项 = 0
```

这是因为系统无法执行该决策，而不是因为 JSON 外观不合格。

### 10.10 Reward 伪代码

```python
def compute_reward(sample, prediction):
    parsed = tolerant_parse(prediction)

    action = action_reward_from_cost(sample.decision, parsed.decision)
    type_score = violation_type_score(
        sample.violation_type,
        parsed.violation_type,
    )

    components = {
        "action": (0.40, action, True),
        "type": (0.15, type_score, True),
        "evidence": (
            0.25,
            evidence_score(sample, parsed),
            sample.has_evidence_target,
        ),
        "value": (
            0.20,
            value_score(sample, parsed),
            sample.has_value_target,
        ),
    }

    reward = masked_weighted_mean(components)

    if sample.requires_visual_evidence:
        if components["evidence"][1] < 0.3:
            reward = min(reward, 0.45)

    return reward
```

### 10.11 Reward 样例

GT：`reject / ATTRIBUTE_CONFLICT / Model Y / Model X`。

正确预测，bbox IoU 为 0.8：

```text
0.40×1.0 + 0.15×1.0 + 0.25×0.8 + 0.20×1.0
= 0.95
```

决策和类别正确，但观察值错误、bbox IoU 为 0.08：

```text
原始加权分按适用项与 mask 计算
证据门控后为 0.45
```

GT 为 `reject`，模型输出 `pass`：

```text
r_action=-1，其余为0
总 reward=-0.40
```

GT 为 `pass`，模型输出 `review`：

```text
r_action=-0.2
这是额外人工成本，因此轻度惩罚
```

### 10.12 GRPO 配置

```yaml
framework: verl
entrypoint: verl.trainer.main_ppo
dataset: data/manifests/grpo_train.jsonl
validation_dataset: data/manifests/grpo_validation.jsonl
model_name_or_path: Qwen/Qwen3-VL-4B-Instruct
lora_adapter_path: outputs/sft/best
precision: bf16
quantization: none
learning_rate: 2.0e-6
rollout_n: 4
train_batch_size: 1
ppo_mini_batch_size: 4
ppo_epochs: 1
max_prompt_length: 1792
max_response_length: 256
temperature: 0.8
top_p: 0.95
actor_strategy: fsdp2
rollout_backend: vllm
rollout_tensor_model_parallel_size: 1
rollout_gpu_memory_utilization: 0.35
actor_param_offload: true
actor_optimizer_offload: true
ref_param_offload: true
max_grad_norm: 1.0
beta: 0.02
lora_r: 16
lora_alpha: 32
n_gpus_per_node: 1
nnodes: 1
```

启动时必须断言 `rollout_n > 1`、`ppo_mini_batch_size` 可整除 `train_batch_size × rollout_n`，并记录 veRL 实际解析出的 prompt batch、rollout group 和 actor mini-batch。修改 `rollout_n` 时必须同步检查 prompt batch、mini-batch 与显存预算，不能依赖默认值推断。

### 10.13 GRPO 验证

- 每个 reward 分项独立记录；
- 检查 group 内 reward 是否有方差；
- 检查高 reward 样本是否真的有正确证据；
- 检查是否全部预测 `review` 来规避风险；
- 检查是否只学习标题、不读取图片；
- 记录训练后 regional-to-global gap，作为最终 OPD 的 pre 指标。

如果 GRPO 后 crop teacher 的正确率不足以通过 OPD 过滤，先修复 GRPO 的数据、reward 或训练稳定性，不得用低质量 teacher 启动最终 OPD。

## 11. 评测

只比较同一个 Qwen3-VL 的四个阶段：

```text
Base
SFT
SFT + GRPO
SFT + GRPO + OPD
```

### 11.1 业务指标

- 三分类 Accuracy；
- 违规类型 Macro-F1；
- 严重违规漏放率：GT=`reject`、Pred=`pass`；
- 正常商品误拒率：GT=`pass`、Pred=`reject`；
- 不必要人工审核率：GT=`pass`、Pred=`review`；
- 目标类别先验下的业务风险；
- 类别均衡诊断集上的平均 action reward（与业务风险分开报告）。

### 11.2 感知与证据指标

- `observed_value` normalized exact match；
- `listed_value` normalized exact match；
- bbox mean IoU；
- Evidence Recall@IoU 0.5；
- 异常图片索引准确率；
- 缺失字段准确率。

### 11.3 视觉依赖指标

- Counterfactual Pair Accuracy；
- Counterfactual Flip Consistency；
- Full-image Accuracy；
- Crop Accuracy；
- Regional-to-global Gap、paired 95% CI 与相对恢复率；
- title-only、image-masked 和 label-masked 干预测试结果。

### 11.4 工程指标

- 结构化结果可解析率；
- 单样本推理延迟；
- 峰值显存；
- 每个阶段训练时长。

结构化可解析率仅用于工程验收，不作为 GRPO reward。

### 11.5 测试切片

从同一自动生成测试商品池导出：

- 类别均衡测试集，用于诊断各类能力；
- 保留模板测试集，用于检查布局泛化；
- 小证据区域测试集；
- 低质量图片测试集；
- 反事实成对测试集。

不额外引入其他 VLM 或人工测试集。

## 12. 无 LLM-as-a-Judge 说明

全流程监督来源如下：

| 阶段 | 监督来源 | Judge |
|---|---|---:|
| 数据生成 | 原始商品元数据与变换记录 | 否 |
| SFT | 程序生成的结构化目标 | 否 |
| GRPO | 成本矩阵、字段比较、IoU、图片索引、反事实对 | 否 |
| OPD | full-page + crop conditioned teacher token 分布 | 否 |
| OPD 数据过滤 | 同一规则验证器 | 否 |
| 评测 | 固定测试集的程序化标签 | 否 |

Teacher 与 Judge 的区别：

```text
Teacher：提供下一 token 分布，用于 KL 蒸馏。
Judge：阅读整段回答并主观评价质量。
```

本项目只使用前者。

## 13. 实施顺序与阶段验收

### Milestone 0：接口与环境闭环

实现：

- 依赖与 Qwen 微调仓库 commit 锁定；
- schema/parser、坐标变换、split、cost/reward 和 top-k loss 的单元测试；
- Qwen3-VL processor、collator、PEFT target modules、veRL/FSDP2、Ray 与 vLLM 配置导入/最小前向 smoke test；
- 配置、依赖和数据 manifest 哈希记录。

验证：所有不依赖正式训练的测试通过，图像 token、label mask、completion、rollout group 与 actor mini-batch 的启动断言生效。Milestone 0 不包含显卡容量或吞吐验收。

### Milestone 1：数据生成

实现：

- 原始商品标准化；
- 页面渲染；
- 八类标签生成；
- evidence bbox；
- counterfactual；
- crop；
- 数据校验。

验证：

```powershell
python -m pytest tests/test_violation_generation.py tests/test_dataset_split.py -q
python -m src.data.validate_dataset --config configs/data.yaml
```

只有数据校验全部通过后才能开始训练。

### Milestone 2：SFT

实现 BF16 LoRA SFT，保存 baseline、训练日志和 validation 预测。

验证：结构化结果可解析率和 Macro-F1 高于 Base。

### Milestone 3：GRPO

先为每个 reward 写单元测试，再通过 veRL `compute_score` 自定义奖励接口接入 GRPO，并从 SFT checkpoint 初始化 actor。

验证：目标先验下的业务风险低于 SFT，不存在全预测 review 的策略坍塌，并记录 regional-to-global gap 作为最终 OPD 的 pre 指标。

### Milestone 4：OPD

从同一 GRPO checkpoint 初始化冻结 teacher 与可训练 student，实现双 adapter、student rollout、teacher top-k KL 和 token mask。

验证：完整页指标满足预先冻结的 non-inferiority margin，gap 达到预先冻结的最小改善，且目标先验下的业务风险相对 GRPO 满足预先冻结的 non-inferiority margin。

### Milestone 5：最终评测

固定模型、prompt、分辨率和 decoding 参数，生成一张四阶段结果表以及错误案例 JSONL。无需制作 PPT。

## 14. 必需单元测试

至少覆盖：

1. 每种违规注入都能生成正确标签；
2. 一个样本不会同时注入两种违规；
3. 反事实修复后标签变为正确目标；
4. 全部来源商品 ID、同款 family 与图片 pHash 近重复不跨数据 split；
5. `0.25m` 与 `25cm` 得到相同 value reward；
6. bbox 完全重合时 IoU 为 1；
7. bbox 不相交时 IoU 为 0；
8. `reject→pass` 比 `reject→review` 奖励低；
9. 无证据高风险结论触发 0.45 上限；
10. 不适用的 reward 被 mask，而不是记为 0；
11. OPD teacher 无梯度；
12. 固定 token 不参与 OPD KL；
13. top-k KL 的概率和为 1、尾部质量非负、相同分布时接近 0，极端 logits 下无 NaN；
14. 无效输出不会因格式获得额外分数；
15. 原样本和反事实输出相同时反事实评测得分为 0；
16. `bbox_norm` 与完整页像素坐标往返一致；
17. resize/letterbox 与 crop→page 的逆变换正确；
18. `ppo_mini_batch_size` 可整除 `train_batch_size × rollout_n`，且 `rollout_n > 1`；
19. collator 在图像 token、assistant label 或 completion 被截断时失败。

## 15. 主要风险与处理

### 模板捷径

风险：模型记住固定布局而不理解内容。

处理：多模板、测试模板留出、反事实对、随机字体和位置。

### 合成到真实域差距

风险：自动渲染页面不等于真实电商页面。

处理：项目结论限定为“验证后训练方法在可控电商质检任务上的有效性”，不声称直接达到生产可用；使用多模板和视觉退化减少单一合成风格偏差。

### Reward hacking

风险：模型全部输出 review，或者只猜违规类别。

处理：成本矩阵惩罚不必要 review；证据门控；反事实 reward；逐项记录 reward。

### 错误 Teacher

风险：crop teacher 错误会把错误知识蒸馏给 student。

处理：使用程序化 GT 验证 teacher，未通过的样本不进入 OPD。

### 显存不足

处理顺序：

1. 降低 `rollout_n`，但保持大于 1；
2. 降低 `max_pixels`；
3. 降低 top-k logits；
4. 增加梯度累积；
5. 缩短 completion；
6. 最后才考虑调整模型规模。

## 16. 最终交付物

```text
1. 可复现的数据生成与校验脚本
2. 数据集 manifest 和类别统计
3. Qwen3-VL SFT LoRA
4. Evidence-aware OPD 实现和 LoRA
5. Cost-sensitive GRPO reward 与训练脚本
6. 单元测试
7. 四阶段评测结果
8. README 形式的运行命令与实验结论
```

不包含人工标注集、奖励模型、LLM Judge、外部模型对比和 PPT。

=== 范围限制（这些限制了您提出的方案，而非您寻找的目标）===

请在此处报告任何实际存在的问题——包括看似罕见的情况，如果

本项目确实会产生此类问题。然后，请将修复工作控制在本项目范围内：

1. 这不是一篇安全论文。我们欢迎验证，但并不提倡过度防御。

除非本项目另有说明，否则请假定合作方在其自己的机器上进行测试；如果存在真正的对手，他们会明确指出，并且该情况的范围优先。

2. 除非哈希值替换了开销更大的操作，并且其结果会改变后续操作，否则请勿添加哈希值、校验和或指纹。

3) 禁止使用防御性框架：禁止使用功能开关、迁移框架、兼容层或针对此处未出现的情况的包装器。

4. 无需纠结于特殊情况：特殊编码、符号链接竞争、RTL 文本和毫秒级竞争均超出本项目范围，除非该情况可通过本项目支持的用法实现——其文档化的输入、已发布的接口以及真实数据。可实现即可；无需复现。原则上可构建是不够的。

5) 需要判断时，请做出判断。不要用评分表、检查清单或对已确定事项的重复验证循环来代替判断。

6. 以上内容均不凌驾于用户、本项目自身约定或更高优先级规则所要求的安全、迁移、验证或审查之上。这些是用户要求的；它们才是工作内容，而非范围蔓延。已见过的形状，用于校准。示例，而非清单——真正的发现

即使与清单相似，也不能忽略以下情况：

H 对两个电子表格的每一行进行哈希处理，以回答“比较单元格”的问题

H 编写从未被读取的校验和文件

E 强化一个没有用户且未部署的应用程序的帐户

R 在功能尚未编写的情况下，整夜审核自己的补丁

R 一个对所有内容都给出不及格评价的审阅者

O 其理由是前一个守则，而非需求

还有两个看起来像上述示例但实际上并非如此的情况。请报告这些情况：

✓ 允许您跳过重复读取已存在的大文件的摘要

✓ 此项目自身文档示例生成的罕见输入

在运行任何检查之前，请回答：此检查会检测到什么具体故障？

如果发生故障，我会采取什么不同的措施？如果没有答案，则无需运行此检查。

如有任何正确之处，请明确说明。切勿捏造发现。
