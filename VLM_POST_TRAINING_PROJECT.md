# 基于 Qwen3-VL 的商品多图一致性质检后训练项目

## 1. 项目结论

本项目使用 3,986 条商品记录构建商品多图质检数据，并后训练单一规格的 `Qwen/Qwen3-VL-4B-Instruct`。每条原始记录包含标题、品类、可空的颜色和材质，以及 1 张主图和 2 张细节图。

V2 固定为 8 个互斥标签：

1. `PASS`
2. `DUPLICATE_DETAIL_IMAGE`：两张细节图重复或近重复
3. `IMAGE_QUALITY`：主图或细节图被注入模糊、遮挡或低分辨率
4. `WRONG_IMAGE`：主图或一张细节图来自其他商品
5. `CATEGORY_MISMATCH`：页面品类与商品图片不一致
6. `COLOR_MISMATCH`：页面颜色与商品图片不一致
7. `MATERIAL_MISMATCH`：页面材质与商品图片不一致
8. `TITLE_MISMATCH`：标题描述的商品与图片不是同一商品

主训练路线为：

```text
原始数据审计与来源图划分
    → 程序化生成 V2 数据
    → Base 固定验证
    → Stage 1：Qwen3-VL BF16 LoRA SFT
    → Stage 2：Cost-sensitive GRPO + regional-to-global OPD 联合训练
    → 冻结测试集评测
```

本项目继续遵守以下边界：

- 不新增逐条人工标注；
- 不使用 LLM-as-a-Judge；
- 不调用闭源模型 API；
- 不训练奖励模型；
- 不比较其他 VLM；
- 不根据标题或图片补写原始 `null` 颜色、材质；
- JSON 格式、回答长度和模型自报置信度不进入业务 reward；
- Stage 1 默认使用单张 24 GB GPU；当前 veRL 联合实现把 actor/rollout 与冻结 teacher 放在独立资源池，Stage 2 默认需要 2×24 GB。若只有单卡，必须另行实现并验证分时或共置调度，不能把现有双资源池配置描述成可直接单卡运行。

## 2. 已确认的数据事实

原始清单为 `data/all_product.jsonl`。当前审计结果如下：

| 项目 | 数量 |
|---|---:|
| 商品记录 | 3,986 |
| 唯一 `product_id` | 3,986 |
| 商品类别 | 381 |
| 字段映射错误 | 0 |
| 缺失图片 | 0 |
| 每条图片数 | 1 张 main + 2 张 detail |
| 非空颜色 | 1,604 |
| 空颜色 | 2,382 |
| 非空材质 | 847 |
| 空材质 | 3,139 |
| 颜色、材质同时非空 | 702 |
| 仅颜色非空 | 902 |
| 仅材质非空 | 145 |
| 颜色、材质同时为空 | 2,237 |
| 唯一标题 | 3,796 |
| 仅有 1 条商品的品类 | 83 |
| 少于 5 条商品的品类 | 167 |

因此：

- `null` 表示“来源未提供”，不是错误标签；
- `COLOR_MISMATCH` 最多只能从 1,604 条非空颜色记录中筛选；
- `MATERIAL_MISMATCH` 最多只能从 847 条非空材质记录中筛选；
- 颜色和材质样本不能要求覆盖全部商品，也不能要求八类原始商品数完全相等；
- 381 个品类中存在大量稀有类，不能假设每个商品都能找到同品类供体；
- “图片存在”不等于“图片天然合格”。PASS 候选仍需检查分辨率、解码、三图近重复及来源关系。

## 3. 原始数据契约

输入记录保持当前格式，不猜测缺失字段：

```json
{
  "product_id": "product_0001",
  "title": "Classic Accessories Fairway Deluxe 3-Sided 2-Person Golf Cart Enclosure, Tan",
  "category": "sporting_goods",
  "color": "Tan",
  "material": "Fabric",
  "images": {
    "main": {
      "image_id": "data/raw_clean_highres/product_0001_main.jpg"
    },
    "detail": [
      {
        "image_id": "data/raw_clean_highres/product_0001_detail_1.jpg"
      },
      {
        "image_id": "data/raw_clean_highres/product_0001_detail_2.jpg"
      }
    ]
  }
}
```

当前 `image_id` 实际保存的是相对文件路径。导入时保留原字段，同时生成明确的内部图片对象：

```json
{
  "role": "detail:1",
  "path": "data/raw_clean_highres/product_0001_detail_1.jpg",
  "sha256": "...",
  "width": 2000,
  "height": 2000,
  "phash64": "..."
}
```

不得把文件路径当作稳定的来源图片 ID。若有 ABO 原始 image ID，应另存为 `source_image_id`。

## 4. PASS 基线与 eligibility

所有违规样本都从通过基础检查的 clean base 派生。原始记录只有同时满足以下条件，才可生成 `PASS` 或作为其他类别的宿主：

- 三张图片均存在、可解码，且哈希与尺寸已记录；
- 主图和两张细节图属于同一来源商品记录；
- 三张图片之间不存在达到阈值的精确重复或 pHash 近重复；
- 图片有效分辨率达到冻结阈值；
- 标题和品类非空；
- 没有已知的字段映射错误；
- 页面渲染后文字不溢出、不被遮挡；
- 生成器没有注入任何变换。

基础检查不能自动证明天然图片内容一定正确。项目结论只相对于“来源清单中的 catalog-image 关联”成立，不声称完成了人工验证的真实商品事实核验。

颜色和材质 eligibility 另加以下条件：

### 4.1 颜色

只有在 `color != null` 且标准化后不是空值、占位值或不可判定组合时，才可生成 `COLOR_MISMATCH`。

颜色标准化只用于匹配与采样，例如：

```text
grey == gray
navy blue == navy
black/white == black + white
```

多色商品不应被强制压成单色。若无法构造与原值明确不同且视觉上可区分的错误颜色，则丢弃该候选，不猜测。

### 4.2 材质

只有在 `material != null` 且能映射到冻结的材质族时，才可生成 `MATERIAL_MISMATCH`。建议先使用视觉上相对可区分的材质族，例如 metal、wood、leather、plastic、glass、fabric/textile、ceramic/stoneware；具体映射写入配置并经数据审计后冻结。

以下值默认不进入材质违规池：

- 空值或 unknown；
- 只描述表面处理、颜色或风格的值；
- 无法拆解的多材质长字符串；
- 原值与错误值属于同一材质族；
- 仅凭当前图片无法形成可验证对比的值。

材质真值来自 catalog 字段与图片归属关系，不是人工视觉标注。因此报告中必须写“catalog-conditioned material consistency”，不能扩大为真实世界材质识别准确率。

## 5. V2 八类定义

V2 仍采用单标签设计。一个样本只允许一个目标违规；发现第二种违规时丢弃，不做事后优先级裁决。

| 标签 | 程序化构造 | 默认动作 | 证据 |
|---|---|---|---|
| `PASS` | clean base 原样渲染 | `pass` | 空列表 |
| `DUPLICATE_DETAIL_IMAGE` | 将 `detail:2` 替换为 `detail:1`，或施加轻微无语义变换 | `review` | 两张细节图的 `image_pair` |
| `IMAGE_QUALITY` | 对一个 main/detail 注入模糊、遮挡或低分辨率 | `review` | 受损图片索引和 `issue_subtype` |
| `WRONG_IMAGE` | 只替换一个 main/detail，其他两图保持原商品 | main: `reject`；detail: `review` | 错误图片索引与至少一张参考图 |
| `CATEGORY_MISMATCH` | 修改页面 category，图片和其他字段不变 | `reject` | category 文本区与商品图区 |
| `COLOR_MISMATCH` | 对 eligible 商品修改 color，并同步处理标题中的同一颜色词 | `review` | color 文本区与商品图区 |
| `MATERIAL_MISMATCH` | 对 eligible 商品把 material 改为不同材质族，并同步处理标题中的同一材质词 | `review` | material 文本区与商品图区 |
| `TITLE_MISMATCH` | 使用满足隔离条件的错误标题，category/color/material 保持不产生第二冲突 | `reject` | title 区与商品图区 |

默认动作是项目假设，最终成本需在 baseline 后写入 `configs/eval.yaml` 冻结。尤其是主图错误与细节图错误严重度不同，因此 `WRONG_IMAGE` 可由 `target_image_ref` 决定动作，而不是强迫同一类型只有一个动作。

### 5.1 图片质量子类型

`IMAGE_QUALITY` 只有一个主标签，但包含三个辅助子类型：

- `blur`：记录 Gaussian blur 半径或等价参数；
- `occlusion`：记录遮挡形状、面积比例与位置；
- `low_resolution`：先缩小到冻结尺寸，再放大到页面显示尺寸，记录原始和退化尺寸。

子类型必须在 train/validation/test 中分别达到最低数量。增强参数范围在正式生成前冻结，避免测试集使用训练时未记录的任意退化。

### 5.2 供体选择

供体必须与宿主处于同一 split，但不属于同一来源连通分量。

- `WRONG_IMAGE`：优先使用相同 category、相同可用 color/material 的不同商品，减少附带属性冲突；
- `TITLE_MISMATCH`：优先使用相同 category，且非空 color/material 与宿主一致的供体标题；
- `CATEGORY_MISMATCH`：只修改 category 文本，不替换图片；
- `COLOR_MISMATCH`、`MATERIAL_MISMATCH`：不使用缺失值作为原值或错误值。

381 个品类中有 83 个单例品类。找不到合格同品类供体时，该商品对 `WRONG_IMAGE` 或 `TITLE_MISMATCH` 记为 ineligible；不得退化为会引入第二标签的跨类供体。若后续建立冻结的 coarse-category taxonomy，可在同一 coarse category 内选择供体，但必须作为新的、可审计的配置。

### 5.3 互斥规则

- 只复制或近复制细节图时，标签固定为 `DUPLICATE_DETAIL_IMAGE`；
- 只施加退化时，标签固定为 `IMAGE_QUALITY`；
- 替换一个图片时，只允许标为 `WRONG_IMAGE`，且供体字段必须通过冲突筛查；
- 只改 category 时标为 `CATEGORY_MISMATCH`；
- 只改 color 及标题中的同一颜色词时标为 `COLOR_MISMATCH`；
- 只改 material 及标题中的同一材质词时标为 `MATERIAL_MISMATCH`；
- 标题整体被替换且其他字段不产生独立冲突时标为 `TITLE_MISMATCH`；
- 任意 eligibility 检查检测到第二种违规，直接丢弃样本。

## 6. 数据划分

必须先按来源关系划分商品，再生成派生样本：

```text
all_product.jsonl
    → product ↔ source image ↔ canonical family 建图
    → 加入相同来源记录、可靠商品族、图片 SHA-256/pHash 近重复边
    → 按连通分量分配 SFT / Joint(GRPO+OPD) / Test
    → 在各数据池内建立供体池和字段 eligibility 池
    → 注入 V2 单违规
    → 生成反事实与 evidence crop
```

3,986 条商品当前固定目标为：

| 数据池 | 数量 | 用途 |
|---|---:|---|
| SFT train | 1,400 | Stage 1 结构化监督 |
| SFT validation | 120 | SFT 选择与协议检查 |
| Joint train | 2,000 | Stage 2 联合 GRPO+OPD |
| Joint validation | 180 | 联合阶段调参与 teacher gate |
| Test | 286 | 配置冻结后的唯一最终评测 |
| 合计 | 3,986 | — |

划分单位是来源连通分量，不是单条 JSON。任一 source product、source image、canonical family、供体关系或近重复图不得跨 SFT、Joint 和 Test。若连通分量大小使目标数无法精确满足，应优先保证无泄漏，并记录实际偏差。

Joint 数据池内部再导出两个互斥子池：

- `dataset_stage=grpo, opd_enabled=false`：只计算规则任务 reward，所有蒸馏 token 权重为 0；
- `dataset_stage=opd, opd_enabled=true`：同一学生同时计算规则任务损失和 OPD KL；教师比学生多看 renderer-derived evidence crop。

两个子池通过同一个 `joint_train.jsonl`、同一个 `train_joint.py` 和同一次 actor 更新训练，不形成“先 GRPO checkpoint、再 OPD checkpoint”的串行路线。

分配器应在不拆散连通分量的前提下，把以下 eligibility 作为分层目标：

- 八类候选数；
- 颜色非空、材质非空及两者同时非空的商品数；
- 381 个品类的覆盖；
- 主图与细节图错误的覆盖；
- 三种图片质量子类型的覆盖。

如果新的颜色/材质过滤使 Joint validation 或 Test 的少数类不足，应使用 V2 strata 重新运行整个连通分量分配；不得把 SFT 来源复制到 Joint，也不得复制同一商品来虚增独立样本。

## 7. 类别配额

旧的“每类固定 600”不再适用。新配额由 eligibility 审计决定：

1. 先完成 clean base、颜色标准化、材质族映射和供体可行性统计；
2. 再以 source component 为单位完成上述 SFT/Joint/Test 固定划分；
3. SFT train 与 Joint train 分别使用 class-aware sampler 缓解不平衡；
4. validation/test 使用唯一 source component 计数，不通过重复模板凑样本量；
5. baseline 后依据严重漏放率和各类准确率所需置信区间，冻结每类最低测试量。

在正式 power analysis 前，可使用“SFT validation、Joint validation 和 Test 每个违规类至少 50 个唯一 source component”作为预检目标，不作为论文最终统计结论。若 `MATERIAL_MISMATCH` 经严格过滤后达不到最低量，应缩小该类结论或补充合规数据，不能从 `null` 猜材质，也不能复制同一商品来虚增独立样本。
## 8. 页面渲染与输出协议

使用 Pillow 把 title、category、color、material、main 和两个 detail 渲染为完整商品页。颜色或材质为 `null` 时显示 `N/A` 或不显示该行，但必须由模板配置固定；缺失本身不构成 V2 违规。

渲染器记录：

```text
title_bbox
category_value_bbox
color_value_bbox
material_value_bbox
main_image_bbox
detail_1_bbox
detail_2_bbox
```

V2 更改标签枚举和辅助字段，因此协议升级为 `schema_version: "2.0"`：

```json
{
  "schema_version": "2.0",
  "decision": "review",
  "violation_type": "IMAGE_QUALITY",
  "issue_subtype": "blur",
  "field": null,
  "listed_value": null,
  "observed_value": null,
  "evidence": [
    {
      "role": "damaged_image",
      "region_type": "image_ref",
      "image_ref": "detail:1"
    }
  ]
}
```

约束如下：

- `PASS` 的 `violation_type` 可规范化为 `PASS`，其余字段为 `null`，`evidence=[]`；
- `DUPLICATE_DETAIL_IMAGE` 使用 `region_type: "image_pair"`；
- `IMAGE_QUALITY` 和 `WRONG_IMAGE` 至少包含一个 `image_ref`；
- category/color/material/title 文本证据使用完整页 `bbox_norm`；
- `bbox_norm` 为完整原始页面上的 xyxy 相对坐标，整数范围 `[0,1000]`；
- parser 接受字段顺序、空白和代码块差异，但不接受不支持的 schema 版本；
- 并非每类都需要 `listed_value` 和 `observed_value`，reward 与指标必须用 mask 动态启用。

## 9. 审计记录、反事实与 crop

每次注入都写入完整审计记录：

```json
{
  "sample_id": "train_000001",
  "source_product_ids": ["product_0001", "product_1024"],
  "source_image_ids": ["..."],
  "split": "train",
  "seed": 42,
  "violation_type": "WRONG_IMAGE",
  "target_image_ref": "detail:2",
  "transform": "replace_single_image",
  "transform_params": {
    "donor_product_id": "product_1024"
  },
  "changed_fields": {},
  "eligibility_checks": {
    "same_category": true,
    "color_conflict": false,
    "material_conflict": false,
    "same_source_component": false
  },
  "evidence": []
}
```

每个违规样本生成一个只撤销目标变换的反事实 PASS：

- 重复图恢复原 detail；
- 质量退化恢复原图；
- 错图恢复原 main/detail；
- category/color/material/title 恢复原字段。

反事实对保持模板、未修改图片、字体、布局和其他字段不变。V2 只把反事实用于 SFT 数据增强和固定评测；未实现 paired sampler 前，不接入 GRPO reward。

对局部证据生成 1.5～2.0 倍上下文 crop。重复图和错图可以保存多 crop 或图片对，但 crop 只由渲染坐标生成，不允许人工选框。

## 10. 数据校验

正式训练前必须全部通过：

- JSONL 可解析，`product_id` 唯一；
- 三张图片路径存在且可解码；
- 图片路径、SHA-256、尺寸和 pHash 一致；
- train/validation/test 的 source component 交集为空；
- SHA-256 相同或 pHash 达阈值的图片不跨 split；
- 供体与宿主不属于同一 source component；
- 每条违规样本只有一个目标变换；
- `PASS` 没有变换和证据；
- 颜色为 `null` 的记录不能生成 `COLOR_MISMATCH`；
- 材质为 `null` 的记录不能生成 `MATERIAL_MISMATCH`；
- 错误颜色与原颜色标准化后不同；
- 错误材质与原材质属于不同冻结材质族；
- TITLE/WRONG_IMAGE 的供体没有引入 category/color/material 第二冲突；
- duplicate 的两图达到冻结重复阈值；
- blur、occlusion、low_resolution 的参数位于冻结范围；
- 反事实准确恢复为 PASS；
- evidence 引用存在，bbox 在边界内，crop 覆盖目标区域；
- resize、letterbox、crop→page 坐标可往返；
- 固定 seed 可生成相同 manifest hash；
- 每类与每个质量子类型达到配置下限。

允许对少量页面做不参与标签修改的工程浏览，用来发现溢出、遮挡和退化参数过强/过弱。若完全不做人工浏览，报告必须注明标签正确性只相对于程序化规范成立。

## 11. 训练流程

### 11.1 Base 与 SFT

训练前使用相同 prompt、图像分辨率和解码配置运行原始 Qwen3-VL，保存 baseline。

SFT 只训练 schema 2.0 的短结构化答案，不训练自由文本 CoT。默认：

```yaml
model_name_or_path: Qwen/Qwen3-VL-4B-Instruct
precision: bf16
quantization: none
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
max_output_length: 256
```

启动时必须验证 LoRA 只命中语言模型 attention；collator 必须断言图像 token、assistant mask 和 completion 未被静默截断。SFT validation 可解析率目标为至少 98%。

### 11.2 Cost-sensitive GRPO + regional-to-global OPD 联合训练

Stage 2 从同一个 SFT checkpoint 初始化学生与教师：

```text
Student：SFT checkpoint + 可训练 LoRA；输入完整商品页
Teacher：同一 SFT checkpoint，完全冻结；输入完整页 + evidence crop
Rollout：Student 当前策略生成
Task signal：程序化 cost-sensitive reward
Distillation：Teacher 在 Student 同一响应前缀上的 token 分布
```

教师不是 Judge：它不决定奖励，也不替代程序标签。冻结 SFT teacher 的预测只用于确定性 OPD eligibility gate；decision、类型、适用值、证据引用和 crop→page 映射全部通过规则校验的样本，才能设置 `opd_enabled=true`。

联合目标为：

```text
L_joint = L_GRPO + lambda * L_OPD
L_OPD = sum_t(w_t * KL(p_T^t || p_S^t)) / sum_t(w_t)
```

首版 `lambda=0.25`、`top_k=64`，teacher/student top-k 并集增加 `other` 概率桶。GRPO-only 行的全部 `w_t=0`；OPD-enabled 行在同一次 actor 更新中同时贡献任务策略损失与蒸馏损失。没有有效 OPD token 时返回可微零蒸馏损失，不能阻断该 batch 的 GRPO 更新。

规则 reward 只使用程序标签：

```text
action reward
type reward
evidence reward
subtype/value reward（仅适用类别）
```

建议初始权重：

| Reward | 权重 |
|---|---:|
| action | 0.40 |
| type | 0.20 |
| evidence | 0.25 |
| subtype/value | 0.15 |

采用 masked weighted mean。各类证据评分：

| 类别 | 证据评分 |
|---|---|
| PASS | 不计算 |
| DUPLICATE_DETAIL_IMAGE | 图片对无序精确匹配 |
| IMAGE_QUALITY | 图片索引 + subtype |
| WRONG_IMAGE | 错误图片索引，必要时加参考图集合 |
| CATEGORY_MISMATCH | category bbox + 规范化类别值 |
| COLOR_MISMATCH | color bbox + 规范化颜色值 |
| MATERIAL_MISMATCH | material bbox + 材质族值 |
| TITLE_MISMATCH | title bbox 与错误标题判定 |

要求视觉证据的非 PASS 样本若 `r_evidence < 0.3`，总 reward 上限为 0.45。无法解析 decision 时 action=-1，其余适用项为 0。不奖励 JSON 外观、长度或置信度。

V2 OPD 子池优先考虑局部证据面积小于完整页 30% 的 `WRONG_IMAGE`、`COLOR_MISMATCH`、`MATERIAL_MISMATCH`、`CATEGORY_MISMATCH` 和 `TITLE_MISMATCH`。若某类冻结教师正确率不足，则该类保留在 GRPO-only 子池，不启用蒸馏。

只蒸馏 decision、violation_type、issue_subtype、field 和适用的 value token，不蒸馏 bbox token。Student 始终只看完整页，crop 不得进入学生 prompt。

## 12. 评测

只比较同一模型的三个阶段：

```text
Base
SFT
SFT + Joint(GRPO + OPD)
```

主要指标：

- 八类 Accuracy 与 Macro-F1；
- PASS 误报率；
- 各违规类漏检率；
- main 错图与 detail 错图分层准确率；
- blur/occlusion/low_resolution 分层准确率；
- 严重违规漏放率；
- 不必要 review/reject 率；
- 目标先验下的业务风险；
- 图片索引准确率和重复图片对准确率；
- category/color/material 规范化值准确率；
- title/category/color/material 证据 bbox IoU；
- Counterfactual Pair Accuracy 与 Flip Consistency；
- full-image、crop 指标和 regional-to-global gap；
- source category、稀有类别、字段缺失模式切片；
- 可解析率、推理延迟、峰值显存与训练时长。

所有主指标、目标类别先验、最小有意义提升、non-inferiority margin 和每类最小测试量在 baseline 后写入 `configs/eval.yaml` 并冻结。最终报告点估计与 paired bootstrap 95% CI；test 在配置冻结后只运行一次。

字段缺失切片至少包括：

- color、material 都非空；
- 仅 color 非空；
- 仅 material 非空；
- 两者都为空。

该切片用于确认模型没有把 `null` 或 `N/A` 当成违规捷径。

## 13. 当前仓库审阅结果与改造顺序

当前仓库的两阶段联合训练骨架正确，但尚未实现本 V2 标签契约，不能直接开始正式训练。已确认的差异：

1. `configs/data.yaml` 仍配置旧八类和每类 600；
2. `src/data/render_page.py` 仍使用 `PRODUCT_MISMATCH`、`ATTRIBUTE_CONFLICT` 等旧标签，并依赖 `attributes/audit_field/brand`；
3. `src/models/schema.py`、reward、counterfactual、OPD filter 和测试仍绑定 schema 1.0 与旧枚举；
4. `train_joint.py`、`configs/joint.yaml` 和 `scripts/joint.sh` 是唯一 Stage 2 主入口；旧的独立 GRPO/OPD 训练入口已删除；
5. 现有 1,400/120/2,000/180/286 固定划分需加入 V2 color/material eligibility strata 后重新验证少数类容量；
6. 现有高分辨率数据仍需按 V2 PASS 规则复查原生低分辨率与三图近重复，不能仅依据“文件存在”判定 clean。

建议按以下顺序实施，每步都先补测试：

1. 新增 V2 importer 和 eligibility audit；验证 3,986 条统计、字段交集、图片哈希和候选容量；
2. 重写 source-component splitter；验证 train/validation/test 无商品族和近重复泄漏；
3. 将 schema 升级到 2.0，并实现八类互斥注入器；验证每类最小反例；
4. 重写 evidence、counterfactual、crop 与 dataset validator；
5. 更新 SFT exporter、Joint exporter、reward 和 evaluation；
6. 更新 `configs/data.yaml`、`configs/joint.yaml`、README 与全部旧枚举测试；
7. CPU 全量生成与校验通过后，再进入 Linux CUDA smoke test；
8. baseline 后冻结 `configs/eval.yaml`，再进行正式 SFT 与联合训练。

## 14. 目标仓库接口

```powershell
python -m pytest -q

python -m src.data.audit_v2 --config configs/data.yaml
python -m src.data.split_v2 --config configs/data.yaml
python -m src.data.generate_v2 --config configs/data.yaml
python -m src.data.validate_dataset --config configs/data.yaml

python -m src.training.train_sft --config configs/sft.yaml
python -m src.training.train_joint --config configs/joint.yaml

python -m src.evaluation.evaluate --config configs/eval.yaml
```

这些是 V2 目标接口。对应实现和测试完成前，文档不得把命令写成已验证结果。

## 15. 不宣称事项

- 当前 Windows 会话未完成 Qwen3-VL-4B 的正式 GPU 训练；
- 当前 V2 是经数据审阅后的实现规格，不是实验结果；
- catalog 的 color/material 与图片归属关系不等同于人工确认的自然图像属性真值；
- 合成冲突评测商品文档一致性、视觉依赖与 grounding，不等同于法律虚假宣传、假货鉴定或平台规则审核；
- ABO 原图和派生图片受来源许可约束，不进入 GitHub；
- 全流程不使用 Judge 模型。
