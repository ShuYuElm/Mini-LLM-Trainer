# 40 题回答质量评估归档

Base、LoRA r=8、LoRA r=32 各生成 40 条回答；用户完成全部 120 条匿名评分后，
于 2026-09-24 导入评分并揭示模型映射。这里保留完整回答、原始评分和汇总结果。

建议先看 [评分汇总](quality_score_summary.md) 和 [实验协议及分析](../quality_eval_v1.md)。

| 文件 | 内容 |
| --- | --- |
| [quality_generation.json](quality_generation.json) | 三组全部回答、模型标识、生成设置和自动检查 |
| [quality_review_blind.json](quality_review_blind.json) | 按题匿名排列的回答、参考答案及评分标准 |
| [quality_review_workbook.md](quality_review_workbook.md) | 用户填写的原始评分手册 |
| [quality_review_scores.json](quality_review_scores.json) | 从手册导入的 120 条四维评分及理由 |
| [quality_review_mapping.json](quality_review_mapping.json) | 已揭示的候选标签与模型对应关系 |
| [quality_score_summary.json](quality_score_summary.json) | 分维度、类别和同题比较，含来源规范化哈希 |
| [quality_score_summary.csv](quality_score_summary.csv) | 便于表格软件查看的分模型/类别均分 |
| [quality_score_details.csv](quality_score_details.csv) | 逐题分数、理由和模型标识 |

每维 0–2 分，总分均值为 Base 4.975/8、r=8 5.500/8、r=32 5.825/8。
这是一名评分者对 40 道项目合成题的描述性结果，不证明普遍能力或统计显著性。
r=8 和 r=32 有 31/40 题总分相同；LoRA 在部分任务类别改善，在摘要等类别也有退步。

在项目根目录运行以下命令可重新计算汇总，不需要模型、checkpoint 或数据集：

```powershell
.\.venv\Scripts\python.exe summarize_quality_scores.py
```

命令会更新四份 `quality_score_summary.*` / `quality_score_details.csv` 汇总文件，
不会改动评分手册、已导入评分、回答或映射。CSV 使用 UTF-8 BOM，方便表格软件读取中文理由。
原始生成报告中的本机路径仅记录运行来源，重算评分不需要这些路径可访问。

本轮评分已结束，因此映射与回答一起归档。若新增独立评分者，应先从已有报告导出到
另一个目录，并只提供匿名手册；不要将映射或按模型分类的回答同时提供给评分者。
