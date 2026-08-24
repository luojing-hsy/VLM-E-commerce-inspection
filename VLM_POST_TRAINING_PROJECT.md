# Qwen3-VL 商品页视觉质检两阶段训练设计

## 1. 决策

项目主线重构为：

```text
程序化数据构建
  → Stage 1：BF16 LoRA SFT
  → Stage 2：Cost-sensitive GRPO + regional-to-global OPD 联合训练
  → 固定测试集评测
```

框架采用 `verl==0.8.0`。SFT 使用 `verl.trainer.sft_trainer`；联合阶段使用 `verl.trainer.main_ppo`。这不是“先独立 OPD、再独立 RL”，而是在同一次 Stage 2 actor 更新中组合任务策略损失与蒸馏损失。旧 `grpo.yaml`、`opd.yaml` 和对应入口仅保留为组件调试/消融接口，不再是主训练路线。

成功标准是：SFT 建立稳定的结构化任务协议；联合阶段在不损害完整页业务指标的前提下，同时降低成本敏感风险与 regional-to-global gap。所有最终阈值、类别先验、最小改善和 non-inferiority margin 必须在 baseline 后写入 `configs/eval.yaml` 并冻结。

## 2. 数据集沿用与重组

### 2.1 阶段隔离

沿用既有数据集的来源图划分：原始 listing、image、canonical family、供体关系与 pHash 近重复边先组成连通分量，再把整个分量分配到：

| 数据阶段 | 默认比例 | 用途 |
|---|---:|---|
| `sft` | 50% | 结构化协议、八类任务与反事实监督 |
| `grpo` | 25% | Stage 2 规则 reward rollout |
| `opd` | 15% | Stage 2 区域增强教师蒸馏 |
| `test` | 10% | 冻结后的最终评测 |

每个训练阶段内部再切分 train/validation。任一 `source_product_id`、`source_image_id`、canonical family、供体或近重复图不得跨阶段。测试商品不参与 SFT、reward 调参或 teacher 过滤。

### 2.2 违规与证据

V1 保持单标签：`PASS`、`PRODUCT_MISMATCH`、`ATTRIBUTE_CONFLICT`、`TEXT_LABEL_CONFLICT`、`MISSING_REQUIRED_FIELD`、`IMAGE_QUALITY`、`IRRELEVANT_IMAGE`、`DUPLICATE_IMAGE`。

页面、变换标签、证据 bbox、异常图片索引、重复图片对、缺失字段、反事实和 crop 均由程序生成。`bbox_norm` 是完整原始页面上的 `[0,1000]` xyxy 坐标。无法精确逆映射的增强不得用于 evidence 样本。

### 2.3 Stage 2 联合 manifest

`src.data.export_joint` 合并两个互斥子池：

- `dataset_stage=grpo, opd_enabled=false`：学生看完整页；冻结教师虽由 veRL 服务存在，但该样本的所有蒸馏 token 权重为 0。
- `dataset_stage=opd, opd_enabled=true`：学生只看完整页；教师看完整页与一张或多张 renderer-derived crop。

每行必须携带 `sample_id`、`split`、`training_stage=joint`、完整 lineage、规则 ground truth 和布尔 `opd_enabled`。校验器要求 train 与 validation 都包含两个子池，并检查学生图、教师图和来源隔离。

OPD 候选初始为 `pending_model_inference`，`export_joint` 只接纳 `approved`。`src.data.approve_opd` 读取冻结 SFT 教师预测，以协议、decision、类型、归一化 observed value 和 evidence IoU 做确定性 gate；任何缺失或额外 sample ID 都在写 manifest 前终止。

## 3. Stage 1：veRL 多模态 SFT

输入 JSONL 由 `src.data.export_verl_sft` 转为 Parquet：

```json
{
  "messages": [
    {"role": "user", "content": "<image>\n检查商品页是否存在违规，并输出规定字段。"},
    {"role": "assistant", "content": "{...schema 1.0 target...}"}
  ],
  "images": ["data/generated/.../page.png"]
}
```

veRL `MultiTurnSFTDataset` 对每个 turn 单独构造 loss mask，用户与图像 prompt 不计 assistant loss；`truncation=error`，避免图像 token 或结构化 completion 被静默截断。

默认配置：Qwen3-VL-4B-Instruct、BF16、无量化、vision encoder 冻结、LLM attention 的 `q/k/v/o_proj` LoRA，`r=16`、`alpha=32`、学习率 `1e-5`、2 epoch。启动时仍应在真实模型 `named_modules()` 上确认目标没有命中 visual、merger 或 projector。

veRL 在最后一步保存 `global_step_N/huggingface`；启动器将最后一步记录为 `outputs/sft/latest`。该完整 HF 检查点是 Stage 2 学生与教师共同的初始权重。

SFT 通过条件：validation 可解析率至少 98%，Macro-F1 与字段值准确率高于 Base，且 PASS 不发生类别坍塌。

## 4. Stage 2：GRPO 与 OPD 联合训练

### 4.1 模型与视图

```text
Student：SFT checkpoint + 可训练 LoRA；输入完整商品页
Teacher：同一 SFT checkpoint，完全冻结；输入完整页 + evidence crop
Rollout：Student 当前策略生成
Reward：程序规则与成本矩阵
Distillation：Teacher 在 Student 同一响应前缀上的 token 分布
```

教师不是 Judge：它不选择样本好坏、不输出奖励，也不替代程序标签。OPD 候选仍需用既有确定性规则验证字段、证据角色与 crop→page 映射；最终 teacher correctness gate 需在真实冻结教师推理后完成。

### 4.2 联合损失

veRL 0.8.0 原生支持在 `use_task_rewards=true` 时组合 policy loss 与 distillation loss。本项目使用：

\[
L_{joint}=L_{GRPO}+\lambda L_{OPD},\quad \lambda=0.25
\]

\[
L_{OPD}=\frac{1}{\sum_t w_t}\sum_t w_t KL(p_T^t\parallel p_S^t)
\]

`loss_mode=forward_kl_topk`，`top_k=64`，不使用 distillation policy-gradient。teacher/student top-k 并集以 `other` 桶保留尾部概率质量。

token 权重为：

| 内容 | 权重 |
|---|---:|
| `observed_value` 的 JSON value | 2.0 |
| `violation_type`、`listed_value` 的 JSON value | 1.5 |
| `decision`、`field` 的 JSON value | 1.0 |
| JSON key、固定标点、evidence/bbox、其他内容 | 0.0 |
| 任意 GRPO-only 样本 | 0.0 |

权重由学生实际生成 token 的逐前缀解码映射，不依赖重新 tokenize 文本。没有有效 OPD token 的 batch 返回可微的零蒸馏损失，RL loss 保持正常。

### 4.3 Reward

规则 reward 保持原设计：

\[
R=\frac{\sum_k m_k w_k r_k}{\sum_k m_k w_k}
\]

默认 `action/type/evidence/value = 0.40/0.15/0.25/0.20`。动作项由非负成本矩阵仿射映射；证据使用连续 IoU、集合匹配、图片索引或字段精确匹配；字段值先归一化。需要可见证据但 `r_evidence<0.3` 时，总 reward 上限为 0.45。

不奖励 JSON 外观、长度、自报置信度或自由文本；不调用奖励模型和 LLM Judge。无法解析 `decision` 时动作项为 -1，其余适用项为 0。

### 4.4 veRL 最小补丁

上游 veRL 的 OPD teacher 默认复用 student prompt，不能直接表达 privileged crop。本项目的 `patches/verl-0.8.0-joint-opd.patch` 只改四处：

1. single-turn agent loop 为 `opd_enabled` 样本单独处理 `teacher_prompt`；
2. teacher 请求使用教师多模态输入与学生 response；
3. 拼 batch 前丢弃不同长度的教师 prompt，只把教师 response top-k 对齐到学生 response；
4. distillation loss 使用 `distillation_weights` 做样本与语义 token mask。

安装器锁定 veRL 版本、patch SHA-256、四个原文件和四个补丁后文件 SHA-256，并保留可恢复备份。未知源码会直接失败，避免补丁错位。

## 5. 资源与运行

Stage 1 默认单卡。Stage 2 的 veRL teacher server 与 actor/rollout 使用独立 Ray 资源池，默认配置是 1 张 actor GPU 加 1 张 teacher GPU，即至少 2 张 CUDA GPU。目标为 2×24 GB；显存不足时先降低页面分辨率、rollout batch、vLLM utilization 或增加 offload，不通过量化或替换模型改变主实验。

```bash
bash scripts/setup.sh
bash scripts/sft.sh
bash scripts/joint.sh --dry-run
bash scripts/joint.sh
```

CPU/Windows 只做数据、命令和单元测试：

```powershell
python -m src.training.train_sft --config configs/sft.yaml --prepare-only
python -m src.training.train_joint --config configs/joint.yaml --prepare-only
python -m pytest -q
```

每次训练保存配置 hash、数据 manifest hash、依赖可用性和时间戳。正式 GPU 环境还必须归档模型/processor/chat template、完整 uv lock、CUDA/驱动与峰值显存。

## 6. 评测与验收

只比较 `Base`、`SFT`、`SFT+Joint(RL+OPD)`。主要指标包括：三分类、违规类型 Macro-F1、严重漏放率、正常误拒率、不必要人工审核率、目标先验业务风险、字段值 exact match、bbox IoU、证据 Recall、反事实 pair/flip consistency、完整页/crop 指标与 gap。

联合阶段成功必须同时满足：

1. 完整页主指标不低于冻结的 non-inferiority margin；
2. 业务风险相对 SFT 降低，且 paired bootstrap 95% CI 按预注册方式报告；
3. gap 缩小来自完整页改善，不能来自 crop 指标退化；
4. 严重违规漏放、证据定位与反事实一致性达到冻结的最小改善；
5. 没有全部预测 review 的策略坍塌，group 内 reward 有有效方差；
6. 全流程没有 Judge 调用。

若 crop 条件教师正确率不足，停止联合训练并修复数据或 SFT；不得让错误教师继续蒸馏。

## 7. Git 与数据发布边界

GitHub 只同步源码、配置、测试、文档和 veRL 补丁。`.gitignore` 排除 `/data/`、`/outputs/`、模型权重、Parquet、W&B 与本地环境；提交前还需用 `git ls-files data outputs` 做二次检查。

ABO 原图、渲染页、crop、manifest、训练检查点与预测结果均不上传。ABO 暂按来源页所示 CC BY-NC 4.0 用于非商业研究，并保存 LICENSE、来源 URL、访问日期和归档哈希；若归档许可证与来源页不一致，不得放宽使用范围。

## 8. 未宣称事项

- 当前 Windows 会话未完成 4B GPU SFT 或联合训练，配置不是结果。
- 合成页验证商品文档一致性、OCR 与 grounding，不等价于真实商品包装理解。
- 不比较其他 VLM，不训练奖励模型，不引入闭源 API，不把演示网站或 PPT 作为必做交付物。
