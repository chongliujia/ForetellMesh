# 事件字段与未知项：第二轮接口实验

首个研究／工具 LoRA 在受限角色任务中改善了工具和证据表现，却使完整流程从 12/12 降为 11/12。本轮固定同一基座与候选权重，仅修改可显式选择的提示和修复反馈，以区分接口问题与参数训练问题。新协议为 `grounded_json_v2`，既有默认和 `plain_json_v1` 保留。

## 修复内容

失败样本的完整问题已正确进入 Forecast；模型生成 `event` 时省略了问题末尾的参数 JSON。新提示要求原样保留整个问题，包括换行、引号和参数。若第一次回答校验失败，修复请求的 `required_input_copies` 只重申原输入中的事件字符串和观察时间。模型仍须自己返回完整 JSON；校验器不会补字、改概率、删除字段或从多个回答中挑一个。事件不一致、非法概率、未知证据引用和未来信息仍会被拒绝。

针对候选把已知概率称为未知的问题，新提示要求区分给定参数、在模型假设下可推导的量、真正缺失的信息，以及尚未发生的结果。同时保留有限样本与总体概率的区别，并要求下游检查上游的未知项判断。它不规定“总是有未知项”或“总是没有未知项”，也没有把具体样本答案写进提示。

## 冻结对照

配置：[research_tool_evaluation_v2.json](../configs/research_tool_evaluation_v2.json)。训练权重继续使用 `runs/qwen3_8b_research_tool_sft_r16_v1/adapter`，没有新训练、权重合并或默认升级。

- 重跑原有 24 条角色任务和 12 条完整流程，Base / 候选各一次，共 72 项。两组都使用新协议，与已归档的旧协议运行比较；此前运行的失败记录保持不变。
- 新增 32 条未知项诊断，覆盖 4 个合成事件组、8 种条件、中英文和 Research / Risk。每条分别在 Base / 候选及新 / 旧协议下生成，共 128 项。对照顺序轮换，使用相同 token 上限。
- 完整流程中仅 Research / Risk 加载候选，Forecast / Critic 明确使用 Base；模型参数全程冻结。上下文超限不会静默截断，失败样本计入分母。

诊断配置：[uncertainty_diagnostic_v1.json](../configs/uncertainty_diagnostic_v1.json)。成对条件包括：混合权重给定或缺失、贝叶斯先验给定或缺失、互补概率可推导且有无未来试验、仅有抽样频率或额外明确给定总体概率。样本预先固定，不读取最终测试集，不进入本轮训练。

为了避免使用不可靠的关键词评分器，诊断要求 `unknowns` 只使用问题中定义的字段名；评价集合必须与独立计算的预期集合完全一致，漏项、多报或自由改写都不能通过。因此它是**受限信息状态分类诊断**，不是通用自由文本真实性测量。中英文、角色和反事实变体并不独立，不能把 32 条当作 32 个独立事件。

## 复现

输出目录必须是新的，避免覆盖原实验。

```bash
conda activate lab
PYTHONPATH=src python -m foretellmesh.uncertainty_diagnostic \
  --config configs/uncertainty_diagnostic_v1.json \
  --output data/processed/uncertainty_diagnostic_next

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m foretellmesh.capability_evaluation \
  --bundle data/processed/research_tool_synthetic_v1 \
  --training-run runs/qwen3_8b_research_tool_sft_r16_v1 \
  --system-cohort data/processed/agent_behavior_validation_cohort_v3 \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --config configs/research_tool_evaluation_v2.json \
  --uncertainty-cohort data/processed/uncertainty_diagnostic_next \
  --output runs/qwen3_8b_research_tool_contract_validation_next
```

审计脚本 `runs/agent_contract_verification_v2/reproduce.py` 用于原始 v2 运行完成后的 CPU 回放，核对同一权重、数据与源码哈希、请求边界、适配器选择及全部指标。它还重算上一轮指标，确保新评估逻辑没有追溯改变旧结果。实际生成与资源数据应以完整评估报告为准；脚本回放不能代替模型实验。

## v2 协议的完整结果

200 项生成对照和离线回放均完成，报告位于 `runs/qwen3_8b_research_tool_contract_validation_v2/report.json`，审计位于 `runs/agent_contract_verification_v2/report.json`。

完整流程两组均为 **12/12**，且全部首轮通过；旧候选在英文混合概率题上遗漏事件参数的失败消失。两组 Brier / Log Loss / ECE 均为 **0.209357 / 0.630340 / 0.280578**，报告给定数学概率的误差为 0。平均耗时 Base **17.66 秒**、候选 **18.20 秒**。这些分数没有显示真实预测增益。

原有角色任务仍为：直接工具 Base 4/8、候选 8/8；错误反馈工具 4/8、8/8；Research 0/4、3/4；Risk 摘录 0/4、4/4。候选的一个 Research 回答由漏选证据变为重复引用，仍是错误；没有放宽引用校验。

| 新诊断：正确数 / 32 | Base | 第一版 LoRA |
| --- | ---: | ---: |
| `plain_json_v1` | 10/32 | 6/32 |
| `grounded_json_v2` | 10/32 | 5/32 |

错误包含格式和信息判断两部分。旧协议下 schema 通过数为 Base 16/32、候选 14/32；v2 下分别为 16/32、9/32。大量英文回答将 `unknowns` 生成字典，而旧 Research / Risk 提示没有像 Forecast 那样明确声明该字段是字符串数组。无效回答全部计错，不能仅依据已通过 schema 的子集声称语义能力较好。

完整流程中还保留了英文 Bayes 的语义错误：候选 Risk 声称给定概率未知，下游继续引用该说法。**v2 只解决了本次事件字段回归，尚未解决未知项问题，未晋级默认。** 后续 v3 明确数组类型，并在相同 v3 提示下比较 Base、第一版与第二版 LoRA；第二版使用另外构建的分组训练样本，原诊断保持验证用途。
