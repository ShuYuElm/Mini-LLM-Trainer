# 回答质量评分汇总

评分来源：用户完成的匿名评分手册；原始分数和理由保持不变。

每个维度为 0–2 分；总分为四维等权相加，满分 8 分，仅作辅助描述。

| 模型 | 类别 | 回答数 | 正确性 | 完整性 | 格式 | 无无据添加 | 总分均值 /8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | all | 40 | 1.400 | 1.450 | 0.950 | 1.175 | 4.975 |
| base | summarization | 8 | 1.875 | 1.875 | 1.375 | 1.500 | 6.625 |
| base | rewriting | 8 | 1.125 | 1.250 | 1.375 | 0.625 | 4.375 |
| base | extraction | 8 | 1.750 | 1.625 | 0.375 | 1.500 | 5.250 |
| base | format_following | 8 | 1.125 | 1.500 | 0.625 | 1.500 | 4.750 |
| base | reasoning | 8 | 1.125 | 1.000 | 1.000 | 0.750 | 3.875 |
| lora_r8_seed42_baseline_v1 | all | 40 | 1.475 | 1.550 | 1.325 | 1.150 | 5.500 |
| lora_r8_seed42_baseline_v1 | summarization | 8 | 1.875 | 2.000 | 1.375 | 1.000 | 6.250 |
| lora_r8_seed42_baseline_v1 | rewriting | 8 | 1.250 | 1.250 | 1.875 | 0.750 | 5.125 |
| lora_r8_seed42_baseline_v1 | extraction | 8 | 1.875 | 1.750 | 1.500 | 1.500 | 6.625 |
| lora_r8_seed42_baseline_v1 | format_following | 8 | 1.500 | 2.000 | 0.750 | 1.875 | 6.125 |
| lora_r8_seed42_baseline_v1 | reasoning | 8 | 0.875 | 0.750 | 1.125 | 0.625 | 3.375 |
| lora_r32_seed42_baseline_v1 | all | 40 | 1.600 | 1.650 | 1.300 | 1.275 | 5.825 |
| lora_r32_seed42_baseline_v1 | summarization | 8 | 1.875 | 2.000 | 1.125 | 1.000 | 6.000 |
| lora_r32_seed42_baseline_v1 | rewriting | 8 | 1.625 | 1.375 | 1.750 | 1.125 | 5.875 |
| lora_r32_seed42_baseline_v1 | extraction | 8 | 1.875 | 1.875 | 1.625 | 1.500 | 6.875 |
| lora_r32_seed42_baseline_v1 | format_following | 8 | 1.500 | 2.000 | 0.750 | 1.875 | 6.125 |
| lora_r32_seed42_baseline_v1 | reasoning | 8 | 1.125 | 1.000 | 1.250 | 0.875 | 4.250 |

同一题上的总分比较：数字按左侧模型总分较高 / 相同 / 较低列出，不代表人工两两偏好投票。

- base vs lora_r8_seed42_baseline_v1: 9 / 15 / 16（共 40 题）
- base vs lora_r32_seed42_baseline_v1: 9 / 13 / 18（共 40 题）
- lora_r8_seed42_baseline_v1 vs lora_r32_seed42_baseline_v1: 4 / 31 / 5（共 40 题）

相同回答但不同评分的组数：0。

本结果仅覆盖一名评分者、40 道合成题和每模型每题一次生成。没有多评分者一致性或统计显著性证据，不能推断一般能力优劣。

- [分模型与类别 CSV](quality_score_summary.csv)
- [逐题分数与理由 CSV](quality_score_details.csv)
- [汇总 JSON 与来源哈希](quality_score_summary.json)
