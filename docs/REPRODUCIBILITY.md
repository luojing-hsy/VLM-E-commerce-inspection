# 一键运行与实验复现指南

核对日期：2026-09-05。配置、原始指标与输入 SHA256 一起归档；详见 materials/dataset_inventory.json。

## 1. 四个入口

| 用途 | 一键执行 | 无训练检查 | 默认配置 |
|---|---|---|---|
| 安装环境及补丁 | bash scripts/setup.sh | bash scripts/setup.sh --check | scripts/training-*.txt |
| Baseline | bash scripts/basemodel.sh | bash scripts/basemodel.sh --dry-run | configs/baseline.yaml |
| SFT | bash scripts/sft.sh | bash scripts/sft.sh --print-command | configs/sft.yaml |
| GRPO | bash scripts/grpo.sh | bash scripts/grpo.sh --dry-run | configs/grpo.yaml |

四个脚本通过 shell 语法检查。安装检查验证已安装包版本与 16 个 veRL 文件哈希；
补丁已在官方干净 wheel 上重放并通过首次应用、重复应用、未知改动拒绝和 Python 语法检查。
SFT/GRPO 完成真实数据准备、命令生成及 Hydra 参数解析。

本次服务器 nvidia-smi 显示 No devices were found。因此没有重新执行 GPU 推理或训练，
也没有重装整套二进制依赖。已有 SFT 100 步、GRPO 80 步及模型导出是历史运行证据，
不能把空跑验证描述为本次 GPU 端到端验证。详见 verification_20260905.md。

## 2. 安装与机器要求

需要 Linux、git、可用 Python 引导环境。实际运行需要 NVIDIA GPU/驱动；
fast kernels 安装可能需要编译器和 CUDA 工具链。setup 用 uv 创建 Python 3.12 的 .venv，
安装固定依赖和 editable 项目，再应用完整补丁。训练进程存在时拒绝修改环境。

```bash
git clone --branch refactor/qwen3.5 git@github.com:luojing-hsy/VLM-E-commerce-inspection.git
cd VLM-E-commerce-inspection
bash scripts/setup.sh
bash scripts/setup.sh --check
nvidia-smi
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

| 依赖 | 固定版本 |
|---|---|
| Python | 3.12 |
| torch / torchvision / torchaudio | 2.11.0 / 0.26.0 / 2.11.0 |
| CUDA runtime | 12.9 |
| transformers / peft | 5.16.1 / 0.20.0 |
| veRL / vLLM | 0.8.0 / 0.20.2 |
| ray / datasets / accelerate | 2.56.1 / 5.0.1 / 1.14.0 |
| flash-linear-attention / causal-conv1d | 0.4.2 / 1.7.0 |
| numpy | 2.2.6，使用项目 uv override |

完整直接依赖见 scripts/training-requirements.txt、training-kernel-requirements.txt。
没有完整传递依赖锁文件，不保证未来解析的全部传递依赖逐字一致。

### 完整 veRL 补丁

新的安装器是 scripts/apply_verl_server_patch.py，正文/哈希在
patches/verl-0.8.0-server.patch 和 verl-0.8.0-server.json。
覆盖图片路径、MM projector、SFT 指标、嵌套张量、GRPO 零方差过滤、最终 checkpoint、
日志、rollout 缓存和 padding。保留服务器现有的受条件控制的历史 teacher/distillation 代码，
本文仅支持纯 SFT/GRPO 配置。

旧分项安装器记录的哈希不再匹配完整环境，保留作历史参考，不要叠加执行。
新安装器只接受官方原始文件或已验证的目标文件，未知修改会停止；
首次替换前保留 .before_server_bundle 备份。

## 3. 模型、图片和路径

SFT model_name_or_path 默认 /root/autodl-tmp/models/Qwen3.5-4B；
换机器请改 configs/sft.yaml 为实际模型目录或 Qwen/Qwen3.5-4B。
baseline 可用 BASEMODEL_MODEL 覆盖。GRPO 需要已经导出的
outputs/sft_qwen35_4b/latest/huggingface。

Git 中包含实际数据 JSONL、运行时 JSONL、配置和指标，不含图片及模型权重。
需要把服务器 data/sft/images、data/GRPO/images、data/test/images 复制到新机器同名目录。
仅克隆仓库不能开始多模态训练。
测试集保留原始绝对路径；迁移时将其中 /root/autodl-tmp/vlm-qwen3vl/ 前缀替换为新仓库路径，
并记录新 SHA256。本次为保留可审计字节，不改写原数据。

```bash
.venv/bin/python scripts/check_pipeline_inputs.py --stage all
```

检查每行三张图片、实际可读取性、样本/类别数量和哈希。当前 6960 个唯一图片路径通过检查；
重复详情图样本允许两处引用同一张图。

## 4. 数据规模和实际输入

| 文件 | 行数 | 唯一 source_product_id |
|---|---:|---:|
| data/sft/train.jsonl | 1000 | 973 |
| data/sft/valid.jsonl | 100 | 88 |
| data/GRPO/train.jsonl | 1000 | 1000 |
| data/GRPO/valid.jsonl | 100 | 100 |
| data/test/test.jsonl | 200 | 200 |

五份来源商品集合两两不重叠；该检查不等于全体图片近重复检测。
SFT 训练来源出现 1/2/3 次的数量分别为 954/11/8，验证为 81/2/5。

| 类别 | SFT train/valid | GRPO train/valid | Test |
|---|---:|---:|---:|
| pass | 150/15 | 75/8 | 25 |
| color_mismatch | 150/15 | 75/8 | 25 |
| category_mismatch | 150/15 | 75/7 | 25 |
| material_mismatch | 150/15 | 75/7 | 25 |
| title_mismatch | 150/15 | 200/20 | 30 |
| wrong_image | 150/15 | 200/20 | 30 |
| duplicate_detail_image | 50/5 | 150/15 | 20 |
| image_quality | 50/5 | 150/15 | 20 |

wrong_image 的 main/detail:1/detail:2 训练各 50、验证各 5。
image_quality 训练覆盖全部 9 种子类型/位置组合，每个 5 或 6 条。
旧报告“质量子类型绑定位置、缺少主图错图”已不适用于当前快照。

materials/runtime 保留四份实际运行输入（SFT/GRPO train/validation），其 SHA256
全部与 materials/runs 中历史 run_manifest 匹配。SFT 输入含 conversations，
GRPO 含 prompt、reward_model.ground_truth、extra_info。Parquet 由入口生成。
正式源数据和 runtime 数据用途不同，不应把 runtime JSONL 当作 source_dataset。

## 5. Baseline 测评

```bash
bash scripts/basemodel.sh --dry-run
bash scripts/basemodel.sh
BASEMODEL_MODEL=/path/to/Qwen3.5-9B EVAL_CONFIG=configs/my_9b_eval.yaml bash scripts/basemodel.sh
```

默认本地相邻 models/Qwen3.5-4B 存在时使用它，否则使用 Hub ID。
9B 请复制 baseline.yaml 并改 manifest/predictions/output 到独立目录；
仅换模型名称不会自动换报告目录。
默认结果在 outputs/baseline/qwen35_4b，与 outputs/test 的 SFT 结果分开。
BASEMODEL_FORCE 默认 1，重新生成；设 0 才复用已存在预测。
复用仅检查样本 ID，必须自行确保模型、prompt 和解码配置未变。
BASEMODEL_PREDICTIONS 只改变预测文件；报告仍由配置 output 决定。
BASEMODEL_LOG_FILE 设置日志，BASEMODEL_NO_LOG=1 关闭 tee。

默认测试 200 条，每条三图，min_pixels=784、max_pixels=65536、
max_new_tokens=256、enable_thinking=false，预测使用 do_sample=False。
模型输出经过项目 parser 后统计决策、类别和证据指标。

## 6. SFT

```bash
bash scripts/sft.sh --print-command
bash scripts/sft.sh
SFT_CONFIG=configs/sft_qwen35_4b.yaml bash scripts/sft.sh
bash scripts/sft.sh --resume-from outputs/sft_qwen35_4b/global_step_50
```

| 参数 | 当前值 |
|---|---|
| 模型 / 精度 | Qwen3.5-4B / BF16 |
| GPU / epoch / train / validation | 1 / 1 / 1000 / 100 |
| global_train_batch_size / micro batch | 10 / 1 |
| 学习率 / 最大序列长度 | 1e-5 / 2048 |
| LoRA rank / alpha / targets | 16 / 32 / q,k,v,o projection |
| vision encoder / MM projector | 冻结 / 训练 |
| 保存、验证间隔 | 每 50 步 |
| 默认 resume_mode | disable |

名义训练步数 1000/10=100，实际日志也记录 100 步。
gradient_accumulation_steps=16 没有直接传给当前 veRL 命令，
不要把它再乘到全局 batch 上；有效 batch 以 global_train_batch_size 和动态微批处理为准。

入口准备 JSONL/Parquet，训练后规范化 HF 导出，发布 latest，再做一次 100 条验证。
加载规范化后的 latest/huggingface，避免把原始 PEFT 全状态当作普通合并模型。
训练指标在 outputs/sft_qwen35_4b/training_metrics.jsonl，
验证报告在 validation_metrics.json。最新 checkpoint 策略会清理旧权重，长期保存需提前归档。

直接三图 SFT 测试可用：
```bash
.venv/bin/python -m src.evaluation.evaluate_direct \
  --config configs/eval.yaml --dataset data/test/test.jsonl \
  --model outputs/sft_qwen35_4b/latest/huggingface \
  --predictions outputs/test/sft_new_predictions.jsonl \
  --metrics outputs/test/sft_new_metrics.json --batch-size 8
```

## 7. GRPO

```bash
bash scripts/grpo.sh --dry-run
bash scripts/grpo.sh
bash scripts/grpo.sh --restart
bash scripts/grpo.sh --resume-from outputs/grpo/global_step_80
```

| 参数 | 当前值 |
|---|---|
| 初始模型 | SFT latest/huggingface |
| GPU / rollout TP | 2 / 1 |
| train_batch_size / rollout_n | 4 / 4 |
| ppo_mini_batch_size / ppo_epochs | 4 / 1 |
| 学习率 / KL beta | 2e-6 / 0.02 |
| temperature / top_p | 0.8 / 0.95 |
| prompt / response 上限 | 1792 / 256 |
| 每实例 max_num_seqs | 16 |
| max_model_len / max_num_batched_tokens | 4096 / 4096 |
| rollout 显存比例 | 0.50 |
| actor / optimizer / ref offload | 全部开启 |
| 动态 batch / 每 GPU token 预算 | 开启 / 2048 |
| 图片长度过滤进程 | 8 |
| 保存、验证间隔 / resume | 20 / auto |

当前 YAML 与最近成功运行的 rollout_n 都是 4。讨论过 n=8 不代表已用于本轮。
每个候选批生成 4 个 prompt x 4 个回答；max_num_seqs 是每个推理实例的调度上限。

启用组内 reward 零方差过滤：std>1e-8 才保留；不够 4 组就继续消耗后续源数据批，
最多 8 次，达到上限按当前代码回退。数据加载器有限，不循环补足整个 epoch。
所以 1000/4=250 是未过滤的名义批数；最近实际完成 80 个 step。
训练调用耗时 8407.364 秒，约 2 小时 20 分钟。末次验证 mean reward=0.84960656，
不能将该数值当作决策准确率。

reward 使用适用分量的加权平均：action/type/evidence/subtype=0.30/0.35/0.20/0.15。
action=1-2*cost，误拒 cost=0.75，漏检 cost=1；证据/子类型只对适用标签参与。
证据不足时正 reward 乘以 0.70。完整定义见 src/rewards/composite.py 和 configs/grpo.yaml。

raw checkpoint: outputs/grpo/global_step_80，latest 指向它；
HF 导出: outputs/grpo/hf_exports/global_step_80，hf_latest 指向它。
恢复训练用 raw checkpoint，不用只含推理权重的 HF 导出。
当前保存策略先删除旧 raw checkpoint，再写新 checkpoint；新保存失败时可能无法回退。
文件日志追加写入，重新启动实验建议使用独立 log_file，避免多个运行混杂。

## 8. 历史结果与解释边界

| 记录 | 样本 | 决策准确率 | macro-F1 | 严格协议准确率 | parse rate |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B baseline | 200 | 45.0% | 0.4067 | 39.5% | 45.0% |
| Qwen3.5-9B baseline | 200 | 67.0% | 0.5961 | 62.5% | 82.5% |
| Qwen3.5-4B SFT，直接三图 test | 200 | 80.0% | 0.7478 | 73.0% | 100% |
| Qwen3.5-4B SFT，validation | 100 | 91.0% | 0.8371 | 85.0% | 100% |

原始指标在 docs/evaluation_metrics。
baseline_current_direct.json、sft_evaluation.json 也保留，但模型/运行身份不足以证明与当前
Qwen3.5 实验同口径，不用于计算本轮提升。历史 baseline 低 parse_rate 包含协议失败影响。
相同文件路径不能证明历史数据字节始终一致，旧 business_risk 成本配置也可能不同；
此表是归档记录，不是本次统一控制变量重测。

SFT 曲线含 100 条 train、2 条 val，末步 train/loss=0.03335389，
train/semantic_token_loss=0.13961682，val/loss=0.02511748，
val/semantic_token_loss=0.10616760。semantic loss 为额外观测，不是另一个已训练的新目标。
GRPO 曲线含初始化验证与 80 步（共 81 条），当前没有确认的 GRPO test200 报告。

直接测评续跑时，inference_seconds/seconds_per_sample 只记录本次待预测部分，
num_evaluated 可以覆盖全部 200 条。因此不能直接用总样本数计算该报告的整轮推理耗时。

## 9. 归档清单和再验证

data 下只纳入五份明确选择的 JSONL；materials/runtime 保存四份 veRL 输入，
materials/runs 保存历史配置与输入哈希；dataset_inventory.json 提供核验。
docs/evaluation_metrics 保存聚合指标和训练曲线；patches 保存环境差异。
模型、图片、Parquet、缓存及密钥不属于本次 Git 材料。

```bash
bash scripts/setup.sh --check
.venv/bin/python scripts/check_pipeline_inputs.py --stage all
bash scripts/basemodel.sh --dry-run
bash scripts/sft.sh --print-command
bash scripts/grpo.sh --dry-run
.venv/bin/python -m pytest tests -q
```

换机器先还原图片与模型，再检查环境和输入，具备 GPU 后再执行实际推理/训练。
