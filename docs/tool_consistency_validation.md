# Quant 缺参校验与新情景对照

2026-09-18，在 `lab` / RTX 3090 上完成 128 个 LangGraph 任务。使用既有 Qwen3-8B-Base 和 v3 research/tool LoRA，未训练新权重。

## 结果与决策

64 个任务来自 **8 个新数值组**，覆盖 mixture / Bayes / complement / sampling、中英文与 Research / Risk。训练数据、既有验证/测试数据及上轮 Quant 诊断的可见数值组合均列入排除名单。任务模板仍复用，64 行不是 64 个独立事件。

| 指标 | 原流程，含显式字段范围 | 同输入，启用缺参校验 |
| --- | ---: | ---: |
| 首次 JSON + 完整 unknowns 集合正确 | 64/64 | 64/64 |
| 修正后正确 | 64/64 | 64/64 |
| 修正调用 | 0 | 0 |
| 模型调用 | 64 | 64 |
| 平均任务耗时 | 3.497 秒 | 3.478 秒 |
| 输入 / 输出 tokens | 69,536 / 3,095 | 69,536 / 3,095 |
| 峰值分配显存 | 15.80 GiB | 15.80 GiB |

64 对首次请求和原始输出逐条相同；无退步、无输出截断，基座仅加载一次且权重全部冻结。生成合计 443.70 秒。两组耗时的微小差异不作为速度提升结论。

**没有测出校验的增量收益。** 两组都增加了相同的显式字段范围，本轮不能分离这项输入变化与新数值组的作用，也不能把上轮 59/64 与本轮 64/64 直接解释成能力提升。未触发真实模型修正；修正恢复与拒绝路径由故障注入测试验证。校验保留为 opt-in，不更改默认 checkpoint，不开始 RL。

## 运行时约束

`AgentRunner` / `LangGraphRunner` 新增可选 `tool_consistency="requested_inputs_v1"`，只支持 `quant_research` / `quant_risk`。默认 `None` 保持原行为。

调用方须提供一个有时间来源的证据项，`source` 为 `structured://information-scope/v1`，内容例如：

```json
{"schema_version":"1","requested_fields":["weight","success_probability","future_outcome"]}
```

它描述任务要求检查的字段，不能包含正确答案或缺失字段标签。运行时在模型调用前检查范围、重复项和时间；在模型返回后检查：

- 用户要求检查、且 Quant 明确报告缺失的输入参数，必须在 `unknowns` 中承认。
- 已提供或确定性计算出的量不能被列为未知。
- `unknowns` 不能超出显式范围。

校验不自动补写答案，只通过原有最多一次修正反馈让模型重新生成；仍不合法则失败终止。首次输出和修正后的输出分别评分。反馈仅来自模型可见输入及工具计算，不读取 judge、结算结果或教师答案。

这是一项局部一致性检查，不是完整的不确定性判断器：它不自动要求 `future_outcome`，不把计算依赖缺失推断为所有退化情形下的结果不可识别，也不把零条件概率等数学未定义情形称为缺参。

## 复现和归档

配置：`configs/tool_consistency_diagnostic_v1.json`、`configs/tool_consistency_evaluation_v1.json`。

正式数据：`data/processed/tool_consistency_diagnostic_v2`；运行：`runs/qwen3_8b_tool_consistency_validation_v2`；审计：`runs/tool_guard_verification_v1`。

```bash
conda activate lab
PYTHONPATH=src python -m foretellmesh.tool_consistency_diagnostic \
  --config configs/tool_consistency_diagnostic_v1.json \
  --output data/processed/tool_consistency_diagnostic_next

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m foretellmesh.tool_consistency_evaluation run \
  --cohort data/processed/tool_consistency_diagnostic_next \
  --config configs/tool_consistency_evaluation_v1.json \
  --agent-config configs/capability_agents_v1.json \
  --training-run runs/qwen3_8b_research_tool_sft_r16_v3 \
  --bundle data/processed/research_tool_counterfactual_v3 \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --output runs/qwen3_8b_tool_consistency_validation_next

PYTHONPATH=runs/qwen3_8b_tool_consistency_validation_v2/source_snapshot \
python -m foretellmesh.tool_consistency_evaluation audit \
  --run runs/qwen3_8b_tool_consistency_validation_v2
```

构建前需原始合成数据、v3 数据包和上轮 Quant 诊断数据。运行会冻结输入、配置、源码、token 预算和权重哈希；源码变化后用对应 `source_snapshot` 审计旧运行。

报告 SHA-256：`b1f386eaae163db08ddbfadfae8ad58f8e3f1911ed3807d76a04b683553e667a`。128 个任务、128 次调用已在 CPU 上逐条重放并精确重算分数。原有 1,832 个任务的解析和指标，以及 504 个图任务的请求/结果回归保持一致。

正式运行前有两次启动失败（`...validation_v1`、`...validation_v1_retry1`）：英文范围解析误收录 `meanings`，各完成一个 control 任务后，在 guarded 的输入校验阶段停止。失败数据、源码和报告均保留，不计入正式分数。修复后在构建和生成预检中检查范围，保持种子和数值组不变，创建新的不可覆盖数据目录；没有根据分数筛掉任务。

全套测试 280 项：279 通过，1 项可选 GPU 测试跳过；上述真实 GPU 对照另行完成。
