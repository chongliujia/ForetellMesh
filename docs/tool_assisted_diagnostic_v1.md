# 确定性 Quant 辅助的 Research / Risk 对照

此前[成对诊断](paired_information_diagnostic_v1.md)发现，第三版研究 LoRA 仍会把可计算量列为未知。本轮将计算和输入依赖交给确定性 Quant 步骤，让 Research / Risk 根据任务范围输出剩余未知项；比较 Base 与既有第三版 LoRA，不训练新权重。

## 接口与边界

新增显式工作流程 `quant_research`、`quant_risk`。工具读取唯一一个 `source="structured://probability-model/v1"` 的证据，正文格式为：

```json
{
  "schema_version": "1",
  "model": "sampling",
  "values": {"successes": 17, "trials": 43}
}
```

证据仍须提供 ID、发布时间和可用时间，满足观察时点限制。支持混合概率、二元贝叶斯、互补概率、样本频率四类显式模型；非法数值、额外字段、相互矛盾的互补概率和多个结构化模型来源会被拒绝。缺参不填默认值；空样本、零概率条件事件单列为 `undefined`。

工具返回输入、计算依赖、缺参列表、计算状态、数值及精确分数，并带来源 ID、时间和内容哈希。它不输出标准答案的 `unknowns` 列表，也不读取结算标签或隐藏参数。当前只是基于函数输入依赖的计算器，不是完整符号推理器。只给样本计数时不会估造总体真概率，也不把未来实现结果变成已知。

结构化参数在本实验中由情景规格提供，**没有验证模型从任意新闻提取参数的能力**。声明的时间只能通过元数据校验，不能自动证明来源真实或正文没有被修改。

## 冻结设计

从固定种子抽取每类两个新数值组，共 **8 个事件组、64 个角色任务**。各组覆盖两种信息条件、中英文、Research / Risk。数值排除集合覆盖第三版数据池全部分区、原数学预热集及旧诊断；测试分区只用于数值身份排除，没有选择其提示进行生成、评分或调参。

每个任务有三种匹配输入：

| 条件 | 模型可见内容 |
| --- | --- |
| prose | 原样叙述和字段释义 |
| structured | 相同叙述，加上只包含已提供参数的 JSON 规格 |
| tool | 与 structured 完全相同的输入，另由运行时计算 `upstream.quant` |

缺失先验、缺失权重和未提供的总体概率在三种条件中均保持缺失。标签单独保存。新数值不代表新任务模板，64 个角色变体不能当成 64 个独立事件。

Base 与第三版各运行三种条件，合计 **384 个任务**。固定 `grounded_json_v3`、完整 JSON 代码块解包、2048 token 上下文、384 token 输出预算、BF16 和贪心生成。用真实 `AgentRunner` 执行，最多一次 schema 修复。主指标是**首次回答 schema 有效且未知字段集合完全正确**，修复后结果作为次指标。它不是开放式事实正确性或真实市场预测指标。

预先规定比较 prose→structured、structured→tool、prose→tool 三组配对的改善和退步数。structured→tool 才能隔离工具计算在结构化输入之上的增量。生成前冻结数据、配置、请求、token 预检、基座与适配器哈希、源码；所有回答包括失败均归档。

## 结果

384 个任务全部完成，共 **385 次模型调用**。只有候选 prose 条件发生一次格式修复；修复后该条仍不满足未知项集合要求，所以首次与最终正确数相同。所有条件最终 schema 均为 64/64，输出 token 上限触发为零。

| 输入条件 | Base 首次正确 | v3 LoRA 首次正确 | Base / v3 首次 schema 有效 |
| --- | ---: | ---: | ---: |
| prose | 30/64 | 45/64 | 64/64 / 63/64 |
| structured | 25/64 | 47/64 | 64/64 / 64/64 |
| tool | 22/64 | 59/64 | 64/64 / 64/64 |

| 配对方向 | Base 改善 / 退步 | v3 LoRA 改善 / 退步 |
| --- | ---: | ---: |
| prose->structured | 3 / 8 | 8 / 6 |
| structured->tool | 5 / 8 | 17 / 5 |
| prose->tool | 4 / 12 | 19 / 5 |

**候选使用工具后的净提升伴随明确退步。** structured→tool 有 17 条由错变对、5 条由对变错；可计算量误报未知从 16 条降到 0，但 5 条工具条件失败全部是 `mixture_missing_weight` 中漏掉了 `weight`。工具已正确返回 `missing_inputs=["weight"]`，模型却只报告 `future_outcome` 和 `success_probability`；缺参状态没有被完整传递到角色结论。两组数值中共有四条英文角色回答、一条中文 Risk 回答出现这一问题。

Base 的工具条件可计算量误报同样降为零，但遗漏真正未知项增至 42 条，其中 38 条遗漏 `future_outcome`。因此不能只看误报减少，也不能声称工具对未适配的 Base 普遍有效。

每种信息条件有八条角色/语言变体，第三版各条件首次正确数如下：

| 信息条件 | prose | structured | tool |
| --- | ---: | ---: | ---: |
| bayes_known | 8/8 | 3/8 | 8/8 |
| bayes_missing_prior | 7/8 | 7/8 | 8/8 |
| complement_future | 8/8 | 8/8 | 8/8 |
| complement_parameters | 8/8 | 8/8 | 8/8 |
| mixture_known | 0/8 | 0/8 | 8/8 |
| mixture_missing_weight | 8/8 | 8/8 | 3/8 |
| sampling_empirical | 2/8 | 6/8 | 8/8 |
| sampling_population | 4/8 | 7/8 | 8/8 |

## 成本、审计与处置

| 条件 | Base 平均秒 / 峰值分配 GiB | v3 平均秒 / 峰值分配 GiB |
| --- | ---: | ---: |
| prose | 3.012 / 15.57 | 3.236 / 15.62 |
| structured | 3.093 / 15.60 | 3.321 / 15.66 |
| tool | 3.125 / 15.67 | 3.329 / 15.76 |

生成阶段约 20.40 分钟，基座只加载一次，权重全部冻结。工具自身平均约 0.13 毫秒；tool 条件输入为 860–1035 tokens，structured 为 596–717，prose 为 475–571。第三版 structured / tool 的输出速度约 14.60 / 14.22 token/s。以上是本机串行 PEFT 测量，不能当作 vLLM 并发性能。

原始响应、实际修复请求、工具结果及全部指标由归档源码重放一致。当前重构后的运行器还通过了此前 **1,832 个历史任务**的原始响应与指标回归；LangGraph 又重放本轮全部 384 个任务以及旧完整流程 120 个任务，请求和结果一致。独立 Fraction 核查覆盖 48 个计算状态和 16 个缺参状态，全部正确。

**保留为显式可选研究流程，不更改默认提示或 checkpoint，不启动 RL。** 下一步应验证角色输出如何完整保留缺参依赖，再逐步增加具有可审计数值输入的预测市场情景；不根据这批开发诊断宣称 Brier、Log Loss、ECE 或真实市场表现提升。真实 Polymarket / Kalshi 历史证据和规则核验继续独立保留。

本地归档：

- 数据：`data/processed/tool_assisted_diagnostic_v1`。
- 推理、配置、源码、原始响应：`runs/qwen3_8b_tool_assisted_validation_v1`。
- 重放、失败明细及历史回归：`runs/tool_assisted_verification_v1`。
- 报告 SHA-256：`e411a08791c745e073e26c9206693aeb7f155c9ce27bc505479d3757d143681b`。


## 复现

```bash
conda activate lab
PYTHONPATH=src python -m foretellmesh.tool_assisted_diagnostic \
  --config configs/tool_assisted_diagnostic_v1.json \
  --output data/processed/tool_assisted_diagnostic_next

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m foretellmesh.tool_assisted_evaluation run \
  --cohort data/processed/tool_assisted_diagnostic_next \
  --config configs/tool_assisted_evaluation_v1.json \
  --agent-config configs/capability_agents_v1.json \
  --training-run runs/qwen3_8b_research_tool_sft_r16_v3 \
  --bundle data/processed/research_tool_counterfactual_v3 \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --output runs/qwen3_8b_tool_assisted_validation_next

PYTHONPATH=src python -m foretellmesh.tool_assisted_evaluation audit \
  --run runs/qwen3_8b_tool_assisted_validation_next
```

原实验在 LangGraph 重构前冻结并启动，使用归档源码重放：

```bash
PYTHONPATH=runs/qwen3_8b_tool_assisted_validation_v1/source_snapshot \
python -m foretellmesh.tool_assisted_evaluation audit \
  --run runs/qwen3_8b_tool_assisted_validation_v1
```

图编排另按[LangGraph 接入说明](langgraph_orchestration.md)进行请求、结果与成本验证，不混入本轮工具效果的比较。
