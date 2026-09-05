# 2026-09-05 服务器检查记录

## 已完成

| 检查 | 结果 |
|---|---|
| setup/basemodel/sft/grpo 四个 shell 语法 | 通过 |
| setup.sh --check：直接依赖版本、CUDA runtime 构建和 fast kernels 导入 | 通过 |
| 完整 veRL 补丁 16 个目标文件 SHA256 | 通过 |
| 官方原始 wheel 重放补丁，重复应用，未知修改拒绝，Python 语法 | 通过 |
| baseline --dry-run，测试集三图路径检查 | 200 行，580 个唯一路径，通过 |
| baseline 复用历史预测的实际 launcher 评测链 | 退出码 0；独立输出目录，无模型推理 |
| SFT --print-command | 1000/100 行数据准备与命令构建通过 |
| GRPO --dry-run | 1000/100 行数据准备与命令构建通过 |
| SFT、GRPO 完整启动参数交给 Hydra --cfg job | 均退出码 0 |
| 五份实际 JSONL 和 Pillow 图片读取 | 2400 行，6960 个唯一路径，通过 |
| 五个分区 source_product_id 两两交集 | 全部为 0 |
| 四份 runtime JSONL 与历史 run_manifest 的 SHA256 | 全部匹配 |
| 服务器当前测试套件 | 82 项通过 |
| 从 Git 暂存区导出的独立仓库快照 | 63 项通过，19 项安装环境专用测试因缺少 .venv 资产跳过 |

官方 wheel 的 SHA256、每个原始/补丁后源文件 SHA256 见
../patches/verl-0.8.0-server.json。补丁测试使用临时目录，没有重装或覆盖正在使用的环境。

## 修正的入口问题

1. setup 的旧分项补丁哈希失效：改为官方 wheel 可重放的完整补丁及哈希清单。
2. baseline 默认与 SFT 测试共用 outputs/test：新增 baseline.yaml 使用独立目录。
3. baseline 只按 sample ID 复用可能混入其他模型预测：默认重新生成，显式 FORCE=0 才复用。
4. baseline 增加安全的 --dry-run/--help 参数处理。
5. SFT 空跑从其他目录调用时 cwd 不正确：在模块调用前切换仓库根目录。
6. 启动进程检查覆盖 -u 等参数形式及 veRL 主入口。
7. GitHub 保留了服务器已删除的 OPD 入口/测试及四阶段定义，导致干净仓库测试收集失败：同步清理并使用 SFT/GRPO/test 三阶段。
8. README 的 SFT 来源商品数和质量/错图位置统计过期：按实际文件更新。

## 本次未验证

服务器 nvidia-smi 返回 No devices were found。本次不能验证 GPU 推理、反向传播、
rollout 或多卡训练；不能承诺全新机器具备网络/编译器/驱动时的完整依赖安装必然成功。
环境检查验证的是现有安装和补丁重放。数据 JSONL 随 Git 提供，但图片、模型需另外还原。

## 历史运行证据

SFT 指标：100 条 train 与 2 条 val 记录。GRPO：初始化验证与 80 个 step，
原始运行日志末尾记录 training seconds=8407.364，已生成 global_step_80 raw checkpoint
和 hf_exports/global_step_80。这些是之前完成的运行，不是本次重新训练。

检查输出位于服务器 outputs/reproducibility；正式训练原始运行清单已在任何空跑覆盖之前
保存到 materials/runs，runtime 输入保存在 materials/runtime。

所有待上传数据 SHA256 均直接对 Git 暂存字节校验。补丁和输入的 .gitattributes 禁止换行归一化，以保留精确哈希。
