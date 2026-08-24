# 真实商品数据源适配性与抽样方案评估

评估日期：2026-08-24

## 结论

**ABO 适合本项目，但只适合作为“真实商品图片与目录元数据池”，不能把它描述成天然包装文字真值数据集。** 在当前可核验的候选中，没有另一个数据集同时在可下载原图、商品—图片关系、结构化字段和许可可审计性上整体优于 ABO。因此，不建议用 Amazon Reviews 2023、Amazon-M2、MAVE 或 Shopping Queries Dataset 整体替换 ABO。

更合理的路线是：

1. 以 ABO 的真实目录图和真实 listing 元数据作为主池；
2. 仍由本项目渲染可精确定位的规格表和标签区；
3. 先构建 listing—image—spin—3D 关系图，再合并同款候选边与图片近重复边，按连通分量划分；
4. 将任务结论严格限定为“**真实商品视觉背景上的合成商品文档一致性、OCR 与证据定位**”；
5. Amazon Reviews 2023 只可作为单独审计、单独披露许可风险的可选补充，不应悄悄并入 ABO 主池。

ABO 的许可存在官方来源冲突：官方 S3 页面及其当前提供的 LICENSE 文件写的是 CC BY 4.0，而 AWS Open Data Registry 和 ABO 论文写的是 CC BY-NC 4.0。项目应在取得维护方书面澄清前按更严格的 **CC BY-NC 4.0** 管理，不用于商业用途，并保存实际下载归档内的 LICENSE、来源 URL、访问日期和归档哈希。这是风险控制建议，不是法律意见。

## 1. ABO 是否满足项目要求

### 1.1 许可与使用边界

官方来源目前互相矛盾：

- [ABO 官方 S3 页面](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)将数据标为 CC BY 4.0，并链接到 [S3 中的 CC BY 4.0 LICENSE 文件](https://amazon-berkeley-objects.s3.amazonaws.com/LICENSE-CC-BY-4.0.txt)。
- [AWS Open Data Registry 的 ABO 条目](https://registry.opendata.aws/amazon-berkeley-objects/)标为 CC BY-NC 4.0。
- [ABO 官方论文](https://assets.amazon.science/9d/ca/350b3ef94691a1aedabb8bfb538d/abo-dataset-and-benchmarks-for-real-world-3d-object-understanding.pdf)也写明按 CC BY-NC 4.0 发布。

因此，不能在项目文档中无条件断言 ABO 已明确允许商业使用。可执行的保守规则是：

- `license: CC-BY-NC-4.0-conservative`；
- `license_status: official_source_conflict`；
- 下载时保存归档内 LICENSE 的原始文件名和 SHA-256；
- 保留 Amazon.com 及数据集构建者署名，记录页面渲染、裁剪和退化等修改；
- 若未来需要商业使用或开放分发派生图片/模型，先向 `amazon-berkeley-objects@amazon.com` 请求书面澄清；
- 仓库默认只保留下载脚本、许可审计记录、清单和程序生成标注，不重新分发原图。

### 1.2 listing、图片、多视图与 3D 关系

[ABO 官方页面和公开 schema](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)给出 147,702 个商品 listing、398,212 张唯一高清目录图片；其中 8,222 个 listing 有 24 或 72 帧的 spin 序列，另有约 7,953 个商品对应 3D 模型。listing 记录直接包含：

- `item_id`；
- `main_image_id`；
- `other_image_id[]`；
- `spin_id`；
- `3dmodel_id`；
- 多语言 `item_name`、`brand`、`color`、`material`、`model_name`、`model_number`、`item_dimensions`、`item_weight`、`product_type`、类目 `node` 等字段。

这些关系足以建立真实的 listing—image 来源图，也足以形成主图和最多三张详情图。spin 图适合做保留视角切片，但因为只有约 5.6% 的 listing 带 spin，不能要求所有样本都有 360° 多视图，否则会严重改变商品分布。

公开 listing schema 中没有 `parent_asin`、parent/child variant 或 canonical-family 字段。因此：

- `item_id` 只能作为 listing 身份，不能自动视为款式族身份；
- 同一 `image_id` 被多个 listing 引用时必须连边；
- 相同 `spin_id`、`3dmodel_id` 必须连边；
- 还需通过图片 pHash 和高置信元数据规则构建 `family_candidate` 边；
- “canonical family”是项目派生的保守防泄漏分组，不应冒充 ABO 官方提供的商品族标签。

#### 1.2.1 2026-08-24 metadata 实测审计

本次从 ABO 官方 S3 下载了 `abo-listings.tar`（仅 listing metadata，未下载商品图）并逐条扫描 147,702 条 JSONL 记录。归档大小为 87,480,320 字节，SHA-256 为 `b7f7ceacb328fa5ab6e143b88e1f948443a877cfc95b67ff09c8ebabd50644e3`；归档内 `LICENSE-CC-BY-4.0.txt` 的 SHA-256 为 `419896aea50c15d6e40c5b4baf4bd346f78223b9a354c7357ad00262afdc08ec`。这进一步确认当前归档内是 CC BY 4.0 文本，但不消除 AWS Registry 与论文所写 CC BY-NC 4.0 的冲突。

基础完整性统计：

- 147,127 条有主图引用，134,650 条至少有两张 `other_image_id`；
- 26,424 条有非空 `en_US` 标题；
- 24,033 条同时有 `en_US` 标题、主图、产品类型和至少一个目标属性；其中 21,243 条还有至少两张详情图；
- 进一步要求目标 locale 的字段值唯一、标题长度可渲染后，仍有 21,201 条“主图 + 两张详情图”候选，覆盖 386 个 `product_type`；
- 上述严格候选中，可用颜色字段 11,249 条、材质 6,720 条、`model_number` 14,276 条、尺寸 16,466 条、重量 15,160 条；
- 若每个 `product_type` 最多保留 100 条，图像下载与 pHash 前仍有 8,407 条候选，足以支撑后续严格过滤后抽取 3,000～5,000 个商品。

普通随机抽样不可取：`CELLULAR_PHONE_CASE` 在全量中有 64,853 条，约占 43.9%。本次实测尚未下载原图，因此 21,201 只是 metadata eligibility 上限；原图可解码性、最小分辨率、文件哈希、pHash 近重复和连通分量大小仍须在正式抽样前复核。

### 1.3 天然包装文字与可验证真值

ABO 提供的是目录属性和图片归属关系。其公开 schema 没有天然图片文字转录、OCR 框或“某属性值确实出现在包装上”的标注。目录图中即使肉眼可见文字，也不能据此自动把 catalog 字段当作天然 OCR 真值。

因此，本项目仍应：

- 用 ABO 元数据渲染标签区和规格表；
- 把这些区域标记为 `evidence_source: rendered_text`；
- 仅把商品图归属标记为 `catalog_image`；
- 不声称验证了天然包装文字理解；
- 若未来增加天然包装 OCR 子集，应另行引入可追溯文字标注或人工验证，而不能从标题猜值。

### 1.4 八类任务的可构造性

| 类别 | ABO 上是否可构造 | 约束与真实含义 |
|---|---|---|
| `PASS` | 是 | 只使用实际存在字段，渲染后的标题、规格表和标签区完全一致。 |
| `PRODUCT_MISMATCH` | 是，且 ABO 很适合 | A 提供标题/规格，B 提供真实目录图；证据是渲染标题区和真实商品图区。不是天然包装 OCR 真值。 |
| `ATTRIBUTE_CONFLICT` | 是 | 只修改实际存在且可规范化的字段；真值来自渲染规格值与渲染标签值。 |
| `TEXT_LABEL_CONFLICT` | 是 | 只改标题或宣传区的可验证值，保留渲染标签真值；不要把目录图中文字当作未标注真值。 |
| `MISSING_REQUIRED_FIELD` | 有条件 | ABO 不提供平台“必填规则”；项目必须为选定类目预先定义、版本化并测试必填字段表。 |
| `IMAGE_QUALITY` | 是 | 对已记录的主图或详情图做参数化模糊、遮挡或降采样。 |
| `IRRELEVANT_IMAGE` | 是 | 插入同一 split 中、且不与目标商品处于同一关系图分量的供体图。 |
| `DUPLICATE_IMAGE` | 是 | 复制现有图或施加不改变语义的轻变换，并记录图片对。 |

结论是：ABO 能覆盖 V1 全部类别，但其中三个一致性类别的精确文字证据仍然来自项目渲染器；它提升的是商品图真实性和商品间差异，而不是天然包装文字监督。

## 2. 候选数据集比较

| 数据源 | 图片与字段 | 关系与许可 | 对本项目的判断 |
|---|---|---|---|
| **ABO** | 打包提供高清目录图、spin、3D；有标题、品牌、颜色、材质、型号、尺寸等 | 有明确 listing→图片 ID；无官方 parent family；许可官方来源冲突，按 BY-NC 保守处理 | **最适合作主池**。图片、元数据和归档可复现性最佳，但必须保留渲染标签。 |
| **Amazon Reviews 2023** | [官方数据卡](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)列出标题、features、description、details、商品图片 URL、review 图片及 `parent_asin` | `parent_asin` 能较好表示颜色/款式/尺寸变体族；但维护者明确表示[无权为该数据集指定许可证](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1)，商品图主要是远程 URL，存在授权、链接失效和内容漂移风险 | family 信息优于 ABO，但许可和图片版本化明显更弱。**不宜整体替换 ABO**；只有在单独完成法律与下载审计后，才可作为独立补充。 |
| **Amazon-M2** | [Amazon Science 官方页面](https://www.amazon.science/publications/amazon-m2-a-multilingual-multi-locale-shopping-session-dataset-for-recommendation-and-text-generation)说明它包含六个 locale 的购物会话和商品文本属性；[作者仓库](https://github.com/HaitaoMao/Amazon-M2-Session-Recommendation)用于复现实验 | 发布结构面向会话推荐和标题生成，没有商品图片；仓库未提供清晰的数据 LICENSE | 可参考多语言标题和字段分布，不能承担视觉主数据源。 |
| **MAVE** | [官方仓库](https://github.com/google-research-datasets/MAVE)提供约 300 万个属性值标注、来源段落和字符级 span，覆盖约 220 万商品；没有图片 | [MAVE LICENSE](https://raw.githubusercontent.com/google-research-datasets/MAVE/main/LICENSE)为 CC BY-NC 4.0；仓库只发布标签和重建代码，完整 profile 还依赖另行获得 Amazon Review Data 2018 | 适合辅助设计属性 eligibility、值归一化和类目字段表；**不能替代图像主池**。 |
| **Shopping Queries Dataset / ESCI** | [Amazon Science 官方仓库](https://github.com/amazon-science/esci-data)只有查询、ESCI 相关性标签，以及商品标题、描述、卖点、品牌、颜色和 locale；没有图片字段 | 仓库为 Apache-2.0；ESCI 标签表达查询—商品的 Exact/Substitute/Complement/Irrelevant 关系 | 许可清晰但任务与模态不匹配。可参考困难负例，不能直接生成视觉证据。 |
| **EMMa** | [官方页面](https://emma.stanford.edu/downloads.html)提供约 280 万商品的 listing 文本、图片哈希、材质、质量、价格、类目及约 185 GB 图片包 | 内容非常相关，但官方下载页没有给出清晰的数据许可证或商品族字段 | 在许可证得到明确确认前，不比 ABO 更稳妥；当前不建议纳入主线。 |

### 为什么不直接换成 Amazon Reviews 2023

它的 `parent_asin` 的确解决了 ABO 最弱的 family 关系问题，也有更丰富的 `details` 字典。但这不足以抵消两个主风险：维护者明确不能授予数据许可证，以及图片是外部 URL 而不是与元数据一起固定版本的官方图片归档。对一个要求固定 seed、固定哈希、可重复生成和许可审计的项目，后者是实质性缺陷。

因此，“ABO + 自建保守 family 图”比“Amazon Reviews 2023 + 不稳定图片 URL”更符合当前成功标准。

## 3. 推荐抽样与防泄漏方案

### 3.1 下载与许可审计

先只下载 ABO listing metadata、image metadata 和 256 px 小图做筛选；确定样本后再按 image ID 获取对应原图。每个源归档记录：

```yaml
source: abo
accessed_at: 2026-08-24
source_url: https://amazon-berkeley-objects.s3.amazonaws.com/index.html
archive_name: abo-listings.tar
archive_sha256: "..."
license_file_name: "..."
license_sha256: "..."
license_status: official_source_conflict
effective_policy: CC-BY-NC-4.0-conservative
```

### 3.2 eligibility 过滤

商品进入候选池前应同时满足：

- `item_id`、主图引用和图片文件存在且可解码；
- 原图达到冻结的最小分辨率；
- 至少一个目标 locale 的非空标题；
- 至少一个可做精确一致性构造的实际字段，例如 `color`、`material`、`model_number`、尺寸或重量；
- 字段值经过单位和文本规范化后仍不为空、不含多值歧义；
- 做 gallery 类任务的商品至少有足够的 `other_image_id`；
- 用作困难供体的商品不能与受体共享任何来源图、spin、3D 或 family 分量；
- 任何变换若触发第二个违规类别，则丢弃样本。

不要把 title 中推断出的颜色、容量、SPF 等补成真值。`MISSING_REQUIRED_FIELD` 的必填表必须由项目配置明确给出，而不是用 ABO 缺失率反推平台规则。

### 3.3 来源图与 family 图

建议建立以下节点和边：

```text
listing:item_id
  ├─ main/other ─ image:image_id
  ├─ spin ─────── spin:spin_id
  └─ 3d ───────── model:3dmodel_id

image:image_id ─ pHash-near-duplicate ─ image:image_id
listing:item_id ─ high-confidence-family-candidate ─ listing:item_id
```

`high-confidence-family-candidate` 只使用保守规则，例如：同细类目、规范化品牌相同且非空、规范化型号相同，或多个高置信字段共同一致。不要仅凭标题相似就合并。pHash 阈值应在小规模工程审计后冻结，并记录算法版本；不要事后按 test 表现调阈值。

先求全部边的连通分量，再按分量划分 train/validation/test。划分后重新扫描跨 split 的共享 ID、相同文件哈希和 pHash 近重复，任何命中都应合并分量后重划，而不是删除 test 中不利样本。

### 3.4 分层抽样

对 3,000～5,000 个原始商品，建议按以下顺序抽样：

1. 以连通分量为抽样单位，而不是以 listing 为单位；
2. 先按 `product_type` 或叶级 `node` 分层，并对超大类目设置上限，避免家具等大类支配数据；
3. 在每个类目内平衡可用证据字段：颜色、材质、型号、尺寸/重量等；
4. 平衡单图、三张以上目录图和 spin 商品，但 spin 只做专项切片，不强制占多数；
5. 先分配 split，再在 split 内生成 PASS、违规、反事实、crop 和供体关系；
6. 测试集保留模板、字体和渲染参数组合，同时保持来源分量隔离。

当前 8 类基本均衡的生成预算可以继续使用，但原始商品池不应为追求类别均衡而重复跨 split。类别平衡由同一 split 内的多种程序变换实现。

### 3.5 donor 难度与严格检查

供体只能从受体所在 split 的不同来源分量中选择：

- easy：不同 `product_type` 或明显不同叶级类目；
- medium：同类目，但品牌或型号不同；
- hard：同细类目、视觉嵌入相近，但品牌/型号或关键结构字段可验证地不同。

hard donor 必须额外排除同 family、共享图、pHash 近重复和只有包装/背景差异的候选。难度标签应由这些可复现规则生成，不由 LLM 判断。每条样本保存 donor 候选集版本、过滤原因、相似度和最终选择 seed。

对于 `PRODUCT_MISMATCH`，图片来源替换本身就是唯一目标违规，其他自然图文字不生成附加标签。对于属性和文本冲突，所有被比较的值必须来自渲染区域，且只修改一个字段。这样才能把“无二次违规”变成程序可验证条件。

## 4. 最终建议

项目应从当前 `synthetic_demo` 升级为一种明确的混合来源模式，例如：

```text
source_mode: abo_rendered_audit
visual_source: abo_catalog_image
text_truth_source: abo_catalog_metadata
ocr_evidence_source: rendered_text
family_source: derived_conservative_graph
license_policy: CC-BY-NC-4.0-conservative
```

这比继续完全合成三视图更能检验真实商品视觉差异，也比整体换成 Amazon Reviews 2023 更可审计、更可复现。它仍不是“真实线上商品页质检”数据集：页面布局、标签文字、冲突和证据框由程序生成，真实的是商品图片和 catalog 元数据。最终论文、README 和评测结论都应保持这一边界。
