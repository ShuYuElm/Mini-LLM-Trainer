# 可分享的 rank 实验结果

这些文件来自已完成的 seed=42、r=4/8/16/32 实验，由根目录的
[summarize_results.py](../../summarize_results.py) 从 JSON 报告生成。
导出不加载模型或数据集，不训练、不推理，也不读取权重文件。

- [完整对比 CSV](rank_comparison.csv)：保留原始数值精度；Base 的训练耗时和显存为空，表示未测量。
- [验证 loss 图（SVG）](validation_loss.svg) / [PNG](validation_loss.png)。
- [参数量、GPU 显存和训练耗时图（SVG）](training_resources.svg) / [PNG](training_resources.png)。
- [来源及 SHA-256 清单](manifest.json)：记录原始 JSON 与导出副本各自的哈希。
- [实验分析](../rank_comparison.md)：包含实验条件、成功/失败样例和结论限制。

| Rank | 配置 | 训练指标 | 全序列评估及回答 | Answer-only 评估及回答 |
| --- | --- | --- | --- | --- |
| 4 | [config](runs/lora_r4_seed42_baseline_v1/config.json) | [metrics](runs/lora_r4_seed42_baseline_v1/metrics.json) | [evaluation](runs/lora_r4_seed42_baseline_v1/evaluation.json) | [answer-only](runs/lora_r4_seed42_baseline_v1/evaluation_answer_only.json) |
| 8 | [config](runs/lora_r8_seed42_baseline_v1/config.json) | [metrics](runs/lora_r8_seed42_baseline_v1/metrics.json) | [evaluation](runs/lora_r8_seed42_baseline_v1/evaluation.json) | [answer-only](runs/lora_r8_seed42_baseline_v1/evaluation_answer_only.json) |
| 16 | [config](runs/lora_r16_seed42_baseline_v1/config.json) | [metrics](runs/lora_r16_seed42_baseline_v1/metrics.json) | [evaluation](runs/lora_r16_seed42_baseline_v1/evaluation.json) | [answer-only](runs/lora_r16_seed42_baseline_v1/evaluation_answer_only.json) |
| 32 | [config](runs/lora_r32_seed42_baseline_v1/config.json) | [metrics](runs/lora_r32_seed42_baseline_v1/metrics.json) | [evaluation](runs/lora_r32_seed42_baseline_v1/evaluation.json) | [answer-only](runs/lora_r32_seed42_baseline_v1/evaluation_answer_only.json) |

## 重新生成

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe summarize_results.py
```

默认读取 `checkpoint/lora_r{4,8,16,32}_seed42_baseline_v1`，写入本目录。
这是当前固定四组实验的汇总工具，不是任意模型或训练设置的通用结果解析器。
重复执行会更新生成的 JSON、CSV、SVG 和 PNG；本说明文件保持不变。
`--no-plots` 仅导出报告和 CSV，无需 matplotlib。

没有原始 checkpoint 目录时，可仅用这里的报告副本重新汇总：

```powershell
.\.venv\Scripts\python.exe summarize_results.py --source-dir experiments/results/runs --output-dir experiments/results_rebuilt
```

源目录和输出目录不得相同或互相包含。工具先核对全部报告，再写结果；
检查完成状态、步数、实验控制设置、adapter 元数据、两种评估策略、
loss/PPL 关系、训练与重新加载评估的一致性，以及样例 ID 和生成文本。

## 证据范围

副本保留指标、配置值、六个提示词和完整回答，仅将 `data_dir`、`adapter_path`、
`checkpoint_path` 改为项目相对路径。原始文件不变。副本中的权重路径只是来源记录，
这里不含 `.pt` 文件；保存的有效配置也不是 `train.py --config` 的输入格式。
来源清单中的 `source` 相对于运行时的 `--source-dir`；源文件哈希针对原始字节，
导出哈希针对规范化后的副本，二者可能不同。

所有 adapter 均用全序列监督训练。两种 loss 使用不同的目标 token 集合，应分别比较。
一次 seed、200 个验证样本和六个生成样例不足以证明普遍的回答质量提升。
训练计时不含加载、验证和保存，且未控制单次运行时的机器负载；
显存是训练峰值，不是 adapter 文件大小。图表纵轴从零开始，不显示不存在的误差区间。
模型和数据 revision 未固定，报告副本不能替代完整训练复现。
