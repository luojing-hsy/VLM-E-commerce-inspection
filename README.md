# Qwen3-VL 商品页视觉质检后训练项目

这是一个面向个人学习和简历展示的可复现实验项目：用程序生成商品详情页与可验证违规，再为 Qwen3-VL 准备 Hugging Face SFT、基于 veRL 的 cost-sensitive GRPO 和最终 regional-to-global OPD 数据接口。项目不使用人工标注、闭源 API、奖励模型或 LLM-as-a-Judge。

当前版本刻意把重心放在一台普通电脑也能验证的部分。CPU 侧的数据生成、八类违规、证据框、反事实、crop、结构化解析、reward 和评测已实现；4B 模型训练入口负责检查数据/配置并记录运行元数据，不会默认下载模型或启动昂贵训练。

## 已实现

- 默认固定 seed 生成 3,000 个带三视图的合成商品、约 6,000 个主页面和约 2,250 个反事实页面，避免分发第三方图片；
- 四种 Pillow 页面布局，以及 `PASS` 和七类单标签违规；
- 完整页 `[0,1000]` 证据坐标、图片索引、重复图对、缺失字段证据；
- 一致性违规的最小反事实页面与 renderer-derived 高清 crop；
- Qwen conversation SFT、veRL 原生多模态 GRPO，以及最终 OPD JSONL 导出；
- Pydantic 1.0 协议、宽容 JSON parser、成本矩阵 reward、连续 IoU 和证据门控；
- top-k union + `other` 概率桶的 OPD KL（纯 Python 参考版和 PyTorch 可微版）；
- 数据泄漏、图像解码、bbox、crop、类别下限、反事实恢复等校验；
- 无 Judge 的分类、业务风险、值读取、证据和反事实评测。

## 完整 CPU 数据生成

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"

.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m src.data.prepare_products --config configs/data.yaml
.\.venv\Scripts\python -m src.data.render_page --config configs/data.yaml
.\.venv\Scripts\python -m src.data.validate_dataset --config configs/data.yaml
.\.venv\Scripts\python -m src.evaluation.evaluate --config configs/eval.yaml --oracle-smoke
```

`validate_dataset` 会同时生成 `data/manifests/agents_compliance.json`。如需独立复查 `Agents.md` 适用项：

```powershell
.\.venv\Scripts\python -m src.data.audit_agents_compliance --config configs/data.yaml
```

完整重现性复验分两次记录：第一次生成后加 `--record`，再次完整生成后不加该参数比较。它会检查商品清单、SFT/OPD/GRPO 导出和全部生成图片字节：

```powershell
.\.venv\Scripts\python -m src.data.check_reproducibility --config configs/data.yaml --record
# 再次运行 prepare_products 与 render_page
.\.venv\Scripts\python -m src.data.check_reproducibility --config configs/data.yaml
```

默认完整生成会写入上万个图片文件，耗时取决于 CPU 和磁盘。若只想快速检查代码，可临时把 `configs/data.yaml` 的 `num_products` 改为 48、`samples_per_product` 改为 1，并把各类下限改为 1。

`--oracle-smoke` 只验证指标链路，不能当作模型结果。真实评测文件为 JSONL，每行包含 `sample_id` 和 `prediction`（字符串或对象），然后运行：

```powershell
.\.venv\Scripts\python -m src.evaluation.evaluate --config configs/eval.yaml --predictions outputs/baseline/predictions.jsonl
```

训练前的可复现性检查：

```powershell
.\.venv\Scripts\python -m src.training.train_sft --config configs/sft.yaml --prepare-only
.\.venv\Scripts\python -m src.training.train_grpo --config configs/grpo.yaml --prepare-only
.\.venv\Scripts\python -m src.training.train_opd --config configs/opd.yaml --prepare-only
```

GRPO 的 `--prepare-only` 只检查 veRL 数据、分组 batch、LoRA 和自定义 reward 契约，不导入或安装 veRL。Linux CUDA 训练环境使用 Python 3.12，可通过固定入口安装并检查：

```bash
bash scripts/setup.sh
bash scripts/basemodel.sh
bash scripts/sft.sh --prepare-only
bash scripts/opd.sh --prepare-only
bash scripts/grpo.sh --dry-run
```

`setup.sh` 从 `scripts/training-requirements.txt` 安装固定的 SFT、veRL 0.8.0、vLLM 0.25.1 与 Ray 2.56.1 组合。安装后会对 veRL 版本、补丁文件、原目标文件和补丁后目标文件执行 SHA-256 校验，再应用 `patches/verl-0.8.0-jsonl-image-path.patch`，使 veRL 读取本项目 JSONL 中的图片路径字符串。

`basemodel.sh` 评测已生成的 `outputs/baseline/predictions.jsonl`；它不会用 oracle 标签代替模型推理。当前 `train_sft.py` 和 `train_opd.py` 仍是预检入口，因此对应 `.sh` 应先使用 `--prepare-only`。查看正式 GRPO 将执行的 Hydra 命令：

```powershell
.\.venv\Scripts\python -m src.training.train_grpo --config configs/grpo.yaml --print-command
```

正式 GRPO 入口是 `python -m verl.trainer.main_ppo`，训练使用 FSDP2、vLLM rollout 和 LoRA。`src/rewards/verl_reward.py` 将现有成本矩阵、类型、证据及字段值奖励适配到 veRL 的 `compute_score` API，并返回各分项供训练日志记录。当前 Windows/CPU 环境不会安装或运行该 Linux GPU 栈。

## 项目结构

```text
configs/                 数据、SFT、GRPO、OPD、评测配置
src/data/                商品生成、渲染、反事实、crop、导出和校验
src/models/schema.py     结构化输出协议
src/rewards/             无 Judge 的可组合规则 reward
src/training/            OPD loss 与训练前运行契约
src/evaluation/          业务、感知和反事实指标
tests/                   核心不变量的单元测试
```

生成图片按数据集划分组织，`pages` 是完整商品页，`crops` 是由证据框自动裁出的 OPD 局部图：

```text
data/generated/
├── train/{pages,crops}
├── validation/{pages,crops}
└── test/{pages,crops}
```

文件名中的 `_cf` 表示 counterfactual（反事实）样本。它保持商品、模板和其他内容不变，只恢复导致违规的字段，并将目标标签改为 `PASS`。

## 简历表述参考

> 构建基于 Qwen3-VL 的电商商品页视觉质检后训练实验管线；使用 Pillow 程序化生成 8 类可审计样本及证据坐标，通过 source-aware split 和反事实对降低数据泄漏与视觉捷径；实现 top-k regional-to-global 自蒸馏 KL 与成本敏感组合奖励，并以 IoU、字段归一化和业务成本完成无 LLM Judge 评测。

只有实际完成 GPU 实验后，才建议在简历中补充具体 F1、风险下降或显存数据。

## 范围与后续工作

- 合成图验证的是商品文档一致性、OCR 与 grounding，不代表天然包装文字或真实平台规则效果。
- 当前未包含 ABO 下载/许可审计，也未自动运行完整 Qwen3-VL/veRL 训练循环。
- `opd.jsonl` 是 full-page + crop 候选集；其中 `teacher_filter_status` 保持 `pending_model_inference`。只有冻结的 GRPO teacher 推理并通过规则校验后，才能视为最终 OPD 训练集。
- SFT、GRPO 和最终 OPD Student 均使用 BF16 底座上的标准 LoRA；GRPO 从 `outputs/sft/best` 初始化，OPD teacher/student 都从 `outputs/grpo/best` 初始化。训练配置显式设置 `quantization: none`，不依赖 QLoRA 或 bitsandbytes。
- `scripts/training-requirements.txt` 固定 GPU 直接依赖；完成真实 Linux CUDA 前向 smoke test 后，仍应归档 uv 的完整传递依赖解析结果和 CUDA/驱动信息。在 baseline 后冻结 `configs/eval.yaml` 中的阈值，再运行一次 test。
- 根目录原有的 `index.html`、`styles.css`、`script.js` 和 `assets/` 未被本项目修改。
