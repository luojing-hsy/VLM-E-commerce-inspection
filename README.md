# Qwen3-VL 商品页视觉质检：SFT → 联合 RL + OPD

本仓库实现一个两阶段、无 LLM-as-a-Judge 的电商商品页视觉质检后训练项目：

```text
Stage 1：Qwen3-VL BF16 LoRA SFT
    ↓ 同一个 SFT 检查点
Stage 2：veRL Cost-sensitive GRPO + regional-to-global OPD 联合更新
    ↓
固定测试集评测
```

可以采用 veRL。仓库固定 `verl==0.8.0`：Stage 1 使用官方多模态 `sft_trainer`，Stage 2 使用 `verl.trainer.main_ppo`、FSDP2、vLLM rollout、规则 reward 和 veRL 原生 `policy_loss + coef × distillation_loss`。由于上游 veRL 默认让教师和学生共享同一 prompt，本项目提供一个经过版本与 SHA-256 校验的最小补丁，使冻结教师可看“完整页 + 高清证据 crop”，学生始终只看完整页。

## 当前实现

- 基于之前的数据集契约，先按来源连通分量划分 `sft / grpo / opd / test`，再渲染页面；同商品族、供体和近重复图片不跨阶段。
- 保留 `PASS` 与七类单标签违规、程序化证据、反事实和 renderer-derived crop。
- `sft_train.jsonl` 转为 veRL `MultiTurnSFTDataset` 所需 Parquet，assistant 以外 token 不进入 SFT loss。
- `joint_train.jsonl` 合并两个互斥子池：GRPO 子池只计算规则 RL；OPD 子池同时计算规则 RL 与蒸馏 KL。
- OPD 学生 prompt 只有完整页；教师 prompt 为完整页加 crop。教师输出不充当评分器。
- OPD 只蒸馏业务语义值：`observed_value=2.0`、`violation_type/listed_value=1.5`、`decision/field=1.0`；JSON key、标点、证据 bbox 与 RL-only 样本权重为 0。
- 数据、模型、检查点和运行输出均被 Git 忽略；仓库只提交代码、配置、文档和小型测试。

本次重构完成的是代码与 CPU 回归验证；4B GPU 训练及最终指标尚未在当前 Windows 环境实际运行，不能把配置值写成实验结论。

## 数据流

```text
ABO / 合规原始商品
    → listing-image-family-pHash 来源图
    → sft / grpo / opd / test 阶段隔离
    → 页面渲染、单违规注入、反事实、crop
    ├─ sft_{train,validation}.jsonl
    ├─ grpo_{train,validation}.jsonl ─┐
    ├─ opd_{train,validation}.jsonl  ─┴─ joint_{train,validation}.jsonl
    └─ samples_test.jsonl（冻结后只评测一次）
```

Stage 2 的两个子池不能互换：GRPO 池用于动作与证据策略学习；OPD 池必须有局部证据 crop，并通过确定性 eligibility。`export_joint.py` 不会把 crop 暴露给学生。

## CPU 数据与测试

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.data.prepare_products --config configs/data.yaml
.\.venv\Scripts\python.exe -m src.data.render_page --config configs/data.yaml
.\.venv\Scripts\python.exe -m src.data.validate_dataset --config configs/data.yaml
```

`render_page` 会同步生成 SFT、GRPO、OPD 和 joint manifest。已有页面只需重新导出联合数据时可运行：

```powershell
.\.venv\Scripts\python.exe -m src.data.export_joint --config configs/data.yaml
```

初次生成的 OPD 候选状态是 `pending_model_inference`，不会进入 joint OPD 子池。用冻结的 SFT checkpoint 对“完整页 + crop”推理后，准备如下 JSONL：

```json
{"sample_id":"...","prediction":{"schema_version":"1.0","decision":"reject","violation_type":"ATTRIBUTE_CONFLICT","field":"model","listed_value":"Model Y","observed_value":"Model X","evidence":[...]}}
```

再运行确定性 gate；脚本校验 decision、类型、归一化 observed value 与证据 IoU，并重新导出 joint manifest：

```powershell
.\.venv\Scripts\python.exe -m src.data.approve_opd --config configs/data.yaml --predictions outputs/sft/opd_teacher_predictions.jsonl
```

训练前预检不会下载模型或启动 GPU：

```powershell
.\.venv\Scripts\python.exe -m src.training.train_sft --config configs/sft.yaml --prepare-only
.\.venv\Scripts\python.exe -m src.training.train_joint --config configs/joint.yaml --prepare-only
.\.venv\Scripts\python.exe -m src.training.train_joint --config configs/joint.yaml --print-command
```

## Linux CUDA 训练

训练依赖只支持项目锁定的 Linux CUDA 环境：

```bash
bash scripts/setup.sh
bash scripts/sft.sh
bash scripts/joint.sh --dry-run
bash scripts/joint.sh
```

Stage 1 结束后，启动器把最后一个含 `huggingface/` 的检查点记录为 `outputs/sft/latest`。Stage 2 的学生与冻结教师均从 `outputs/sft/latest/huggingface` 初始化，学生再训练新的 LoRA。

资源边界必须明确：当前 veRL 原生蒸馏把 actor/rollout 与 teacher 放在独立资源池，默认配置是 actor 1 GPU + teacher 1 GPU，因此完整联合阶段至少需要 2 张 CUDA GPU；目标预算为 2×24 GB，并通过 offload、分辨率和 batch 控制显存。单张 24 GB GPU 可以跑 Stage 1，但不能在不改变算法/资源调度的前提下运行本仓库的精确联合 Stage 2。

`scripts/setup.sh` 会依次校验并应用：

- `patches/verl-0.8.0-jsonl-image-path.patch`：读取 JSONL 图片路径；
- `patches/verl-0.8.0-joint-opd.patch`：特权教师 prompt、教师/学生响应对齐、按样本与语义 token 的 OPD mask。

安装器拒绝未知 veRL 版本或未知目标文件，不会静默修改不匹配的环境。

## 联合目标

对 GRPO 子池，`w_t=0`，只优化成本敏感策略目标。对 OPD 子池：

\[
L = L_{GRPO} + \lambda\frac{\sum_t w_t\,KL(p_T^t\parallel p_S^t)}{\sum_t w_t},
\qquad \lambda=0.25
\]

教师在学生生成的同一前缀上提供 top-k 分布，`top_k=64`。教师 prompt 比学生多 crop，但教师响应 prompt 部分会在拼 batch 前移除，所以只有同一 response token 位置参与 KL。规则 reward 仍只包含业务动作、违规类型、证据和可见字段值，不包含 JSON 格式分、长度分、置信度或 Judge 分。

## 评测与范围

只比较同一模型的 `Base / SFT / SFT+Joint(RL+OPD)`，报告三分类、违规类型 Macro-F1、严重漏放率、业务风险、字段值、证据 IoU、反事实一致性和 regional-to-global gap，并使用 paired bootstrap 95% CI。测试配置冻结后，test 只运行一次。

合成页结论只适用于商品文档一致性、OCR 与 grounding，不代表天然包装文字、法律虚假宣传或真实平台规则效果。ABO 原图与派生数据受来源许可约束，不进入 GitHub。

详细设计见 [VLM_POST_TRAINING_PROJECT.md](VLM_POST_TRAINING_PROJECT.md)，数据字段与隔离规则见 [docs/dataset_contract.md](docs/dataset_contract.md)。veRL 依据为官方 [OPD 文档](https://verl.readthedocs.io/en/latest/algo/opd.html)、[Qwen3-VL SFT 示例](https://github.com/verl-project/verl/blob/v0.8.0/examples/sft/vlm/run_qwen3_vl_2b_fsdp.sh) 和 [LoRA 文档](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html)。
