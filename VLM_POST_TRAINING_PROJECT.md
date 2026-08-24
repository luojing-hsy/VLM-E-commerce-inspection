# 实验设计与实现状态

## 目标

用单一 `Qwen/Qwen3-VL-4B-Instruct` 规格验证三阶段后训练路线；SFT 使用 Hugging Face Transformers/PEFT/Accelerate，GRPO 使用 veRL 的 FSDP2 actor、vLLM rollout 与规则奖励接口，最终再用 OPD 做 regional-to-global 策略整形：

```text
程序化页面与标签 → BF16 LoRA SFT → cost-sensitive GRPO → regional-to-global OPD
```

业务输出为 `pass / review / reject`、违规类型、冲突字段和值及可验证证据。JSON 是接口协议，不参与 reward。

## V1 类别

| 类别 | 动作 | 自动变换 | 证据 |
|---|---|---|---|
| PASS | pass | 无 | 无 |
| PRODUCT_MISMATCH | reject | 替换主图和标签商品 | 标题区、商品图区 |
| ATTRIBUTE_CONFLICT | reject | 修改规格表型号 | 规格值、标签值 bbox |
| TEXT_LABEL_CONFLICT | reject | 修改标题型号 | 标题区、标签值 bbox |
| MISSING_REQUIRED_FIELD | review | 删除必填材质 | 缺失字段名 |
| IMAGE_QUALITY | review | 图集高斯模糊 | 图片索引 |
| IRRELEVANT_IMAGE | review | 插入 split 内供体图片 | 图片索引 |
| DUPLICATE_IMAGE | review | 复制图集图片 | 图片对 |

## 已实现边界

- 数据、schema/parser、reward、OPD loss、评测和训练前运行契约已有自动化测试。
- 合成数据只支持个人项目演示，不宣称覆盖真实电商平台规则。
- OPD 导出记录的 teacher filter 状态为 `pending_model_inference`；只有冻结的 GRPO teacher 预测通过规则验证后才能参与训练。
- GRPO 从 `outputs/sft/best` 初始化；最终 OPD 的冻结 teacher 和可训练 student 都从 `outputs/grpo/best` 初始化。
- 三个训练入口的 `--prepare-only` 会校验数据和配置并保存哈希；GRPO 已生成 veRL 原生多模态 JSONL、Hydra 启动参数和 `compute_score` 适配，但当前未安装 veRL 或运行 GPU trainer，因此不能把模型效果写成已完成实验。

## 正式 GPU 实验前检查点

1. 在 CUDA 环境完成 processor、image token、assistant mask、LoRA module 命中和最小前向测试。
2. 生成包含 torch/transformers/peft/accelerate、veRL/vLLM/Ray 与 Qwen commit 的 Linux GPU lock；底座使用 BF16，不启用 4-bit/8-bit 量化。
3. 先运行 Base baseline，再冻结 `configs/eval.yaml` 的最小提升、OPD 视觉与业务风险 non-inferiority margin 和目标先验。
4. validation 用于调参；test 只在冻结配置后运行一次。
5. 分阶段保存 predictions、metrics、数据 hash、配置 hash、显存和耗时。
