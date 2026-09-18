# ForetellMesh

面向金融事件预测与预测市场研究的框架。采用 **Qwen/Qwen3-8B-Base + 任务型 Agent + 能力型 LoRA + 分阶段 RL**，以独立评估验证系统增益；本地训练目标为 RTX 3090 24GB。真实事件 SFT 是可选路线，正式 RL 前须先验证格式、工具等基础行为能力。

当前已实现第一个里程碑：**统一 JSONL 数据导入 → 时间和事件校验 → 时间切分 → 确定性基线 → 评估报告**。此阶段仅使用 Python 标准库，不需要 GPU、网络或模型权重。

第二阶段已增加真实来源注册、固定版本下载、Prophet Arena / ForecastBench 原始格式适配，以及本地 PMA Kalshi 市场快照 Parquet 适配。真实数据通过独立导入审计进入待核验区；历史时间证明与事件分组未完成前，不会自动成为训练或评估样本。

已进一步跑通 **6 条真实 ForecastBench / Manifold 样本**的历史快照核对、精确结算时间检查和保留集基线评估。该小集合只验证数据链路，不用于训练、调参或模型能力结论；见 [首批真实样本复现说明](docs/verified_subset.md)。

已在 `conda lab` 环境的 RTX 3090 上跑通 **1024 / 2048 token 的普通 BF16 LoRA 短程训练**（r=16、micro batch 1）；峰值分配显存分别为 18.25 / 20.58 GiB。配置、结果与复现命令见 [BF16 LoRA 显存测试](docs/lora_resource_probe.md)。该测试使用合成输入，不保存训练 checkpoint，也不代表正式 forecasting SFT 已完成。

已准备 **432 条合成 SFT 预热样本，共 216 个事件组**，中英文表达共享事件，按组划分为 288 / 72 / 72 条；固定 Qwen tokenizer 检查通过，最长 815 token。仅用于格式、证据引用和明确概率任务，见 [SFT 数据说明](docs/sft_warmup.md)。

真实数据路线以**已结算预测市场的历史回放**为主：已归档 **Polymarket 10 个合约 + Kalshi 22 个合约**，在两次 FOMC 会议前 7 天、前 1 天构建 **64 条观察**；结果与官方声明交叉核验，事前证据、历史报价与结果账本分离。精确结算标签已补齐至 **64/64**，Polymarket 规则更新核验已完成；Kalshi 历史规则、基准关联事件审核及教师答案仍待补齐，尚无可直接微调的真实 SFT 样本，见 [最新链上补证](docs/polygon_settlement_proofs.md) 与 [历史数据方案](docs/historical_prediction_markets.md)。

另保留 Kalshi 42 个合约和 Polymarket 14 个合约的未结算前瞻快照、独立结算轮询和质量审计，用于未来检验，见 [前瞻采集说明](docs/prediction_market_dataset.md)。Prophet 核验流程保留，ForecastBench 继续仅供评估。

已加入**能力路由、串行 Agent 编排、PEFT 请求级切换和 Brier 奖励**。首版规划共享 `research_tool_lora` 与 `forecast_lora` 两类能力；研究/工具能力已训练并评估三版候选，预测能力尚未训练，均未晋级默认模型。使用方式、实际验证边界和系统消融计划见 [多 Agent / LoRA / RL 说明](docs/capability_agents.md)。

已扩展 **Base 多 Agent 行为评测**：冻结合成验证集的 6 个事件组、12 条中英问题，比较三条流程并另测 4 条工具任务。提示与解析策略分开对照，记录失败率、概率误差、调用与 token 成本、显存；最终测试集不参与调试。复现与实测结果见 [行为基线报告](docs/agent_behavior_baseline.md)。

三轮共 120 项实测完成：严格解包 JSON 代码块后，三条预测流程均完成 12/12，给定数学概率的报告误差为 0；工具参数正确 2/4，混合概率任务暴露权重与成功率混淆。四角色流程平均 19.36 秒，单预测 6.52 秒，峰值分配显存 15.51 GiB；这些合成结果不代表真实市场预测提升。

已完成首个 **`research_tool_lora` 合成行为训练及 72 项对照评估**：288 条训练样本，普通 BF16 LoRA r16、两轮，峰值分配显存 17.74 GiB。固定开发验证中，工具参数正确数 4/8 → 8/8、Research 证据集合 0/4 → 3/4、Risk 原句与来源配对 0/4 → 4/4；但该轮完整流程完成数从 12/12 降至 11/12，共同完成样本的预测分数没有改善，暴露了未知项误判和跨角色输出契约问题。权重、原始输出、可重放审计和局限见 [首次能力训练报告](docs/research_tool_lora.md)。

后续已完成**接口修复、第二版训练与两轮共 404 项生成对照**。新提示明确完整事件复制和未知项数组类型；第二版用 432 条合成样本训练，峰值分配显存 18.10 GiB。相同 v3 提示下，Base／旧 LoRA／新 LoRA 的完整流程均完成 12/12，两版工具均保持 8/8；但新 LoRA 的 Research 从旧版 4/4 降至 3/4，指定字段名的未知项诊断从 10/32 降至 1/32，表现出较强模板依赖。**第二版不替换旧版，不进入 RL**；详见 [第二版训练及三组评估](docs/research_tool_lora_v2.md) 和 [先行接口对照](docs/agent_contract_v2.md)。

已补充**缺失参数与任务范围的反事实候选数据池**：训练分区 2,336 条、验证 840 条、测试 840 条，同一数值情景的参数删减、范围、字段别名、中英文与角色变体按组隔离。确定性回放和本地 tokenizer 预检通过，训练最长 957 token；数据设计与检查结果见 [第三版候选数据说明](docs/research_tool_counterfactual_v3.md)。

第三版已从该数据池固定选取 **544 条平衡样本完成普通 BF16 LoRA 训练及 588 项三组对照**。新信息状态验证中，Base／第一版／第三版正确数为 **23/128、21/128、127/128**；旧诊断为 **16/32、10/32、18/32**。旧工具保持 8/8，Research 与 Risk 均为 4/4，完整流程保持 12/12，预测分数与第一版相同。新验证共享训练模板，旧诊断仍有三条格式失败及混合概率、经验频率子项退步；**第三版继续作为研究候选，不设为默认，不启动 RL**。全部原始输出重放通过，详见 [第三版训练与评估报告](docs/research_tool_lora_v3.md)。

后续完成了 **768 次同情景成对诊断**，固定答案，交叉改变指令措辞、显式公式和字段释义呈现。Base／第一版／第三版正确数为 **127/256、122/256、166/256**；第三版增加公式有 39 对由错变对、3 对由对变错，但改变措辞或改用 JSON 释义均无净收益。第三版仅 9/32 条原题的八种表达全部正确，样本计数情景仅 4/32（Base 为 32/32）。96 条原题锚点的原始输出与上一轮完全一致，重放核验通过。仍不更改默认提示或 checkpoint；该轮诊断详见 [成对诊断报告](docs/paired_information_diagnostic_v1.md)。

已完成 **384 项确定性 Quant 辅助对照**：相同结构化输入下，第三版 LoRA 的未知项集合正确数从 **47/64 提升到 59/64**，有 17 条改善、5 条退步；退步均为漏报缺失权重。Base 从 25/64 降为 22/64，工具收益依赖后续角色如何使用结果。原始响应与评分重放通过，保留可选研究流程，见 [Quant 对照报告](docs/tool_assisted_diagnostic_v1.md)。

编排层已接入 **LangGraph 1.2.11**：各角色及确定性 Quant 是独立节点，复用原有校验、修复上限和能力路由，模型节点串行共享一份 Qwen3-8B。504 个归档任务重放及 6 个真实 GPU 工作流复核均通过。原运行器保留作回归参照，`check-agent-workflow --engine langgraph` 可运行本地流程检查；安装和实测边界见 [LangGraph 接入说明](docs/langgraph_orchestration.md)。

已增加可选的 **Quant 缺参一致性校验**，只使用输入范围和工具事实提供一次修正机会。新 8 个数值组、128 次 GPU 对照中，两组首次输出均 **64/64 正确**、原始输出完全一致，没有触发修正，尚未测出校验的额外收益；峰值分配显存 15.80 GiB。校验默认关闭，详情见 [新情景对照报告](docs/tool_consistency_validation.md)。

历史数据支持**用途分开的准入预检**和可离线重建的链上补证。10 个 Polymarket 合约已核对 UMA 报告→NegRisk 映射→最终 CTF 赔付，结算时间取最终赔付区块；规则历史核验明确记录区块浏览器和公开 RPC 的信任范围。缺教师答案只阻塞 SFT；跨基准关联事件、Kalshi 规则、独立事件数量等要求继续保留，64 条历史候选正式评分准入仍为 0。三条 Agent 流程的真实对照策略已版本化，见 [历史准入策略](docs/historical_replay_admission.md) 与 [本次补证报告](docs/polygon_settlement_proofs.md)。

## 快速运行

要求 Python 3.11 或更高版本。在仓库根目录运行：

```bash
PYTHONPATH=src python -m foretellmesh evaluate \
  --data examples/synthetic_forecasts.jsonl \
  --config configs/synthetic_baselines_v1.json \
  --output runs/synthetic_baselines_v1
```

输出目录必须不存在；重复运行请使用新目录，以保留已有实验结果。

也可以安装到自己的开发环境后使用 `foretellmesh evaluate` 命令：

```bash
python -m pip install -e .
```

**示例中的 17 条记录全部为人工构造，分数不代表模型能力或真实预测表现。** 示例故意包含跨时间边界的事件、延迟公布标签、重复记录和未结算事件；预期保留 train / validation / test 各 3 / 2 / 3 条，剔除 9 条。

## 生成的文件

| 文件 | 用途 |
| --- | --- |
| `report.md` | 可阅读的基线分数、覆盖率、市场同样本比较和剔除统计 |
| `report.json` | 完整指标、校准曲线数据、数据与代码哈希、数据源版本和实验参数 |
| `predictions.jsonl` | 验证集与测试集的逐样本预测和评估标签；仅供评估使用 |
| `split_manifest.json` | 每条样本的最终分组，或剔除原因 |
| `config.json` | 本次运行的原始配置副本 |

默认三条基线分别为固定 `0.5`、仅在训练集拟合的经验基准概率、观测时刻可用的市场概率。配置可显式选择 `baselines`；仅评估保留集时使用 `["constant_0_5", "market"]`，不拟合经验概率，也不要求把基准数据放进训练集。省略该选项时行为保持不变。

评估包含 Brier Score、Log Loss、固定等宽分箱的 ECE、校准曲线数据和覆盖率。市场价格缺失时，额外在有市场价格的同一批样本上计算全部基线，报告 `baseline - market` 的差值；负值表示该项分数更低。无可评分样本时返回 `null`，不伪造零分。

## 数据约束

- 每条证据必须满足 `published_at <= available_at <= observation_time`；市场快照遵守同样的可用时间约束。未知时间、未来证据、非法概率或不一致身份会使整次运行失败。
- 结算标签独立保存。模型输入只允许通过 `ForecastInput.to_payload()` 构造，不包含标签、结算时间、数据源身份或切分信息。
- 时间切分由配置固定。跨越 train / validation / test 边界的事件组整体剔除，避免相同或高度相关事件跨集合。
- `eval_only` 数据源所在的事件组不能进入训练集。跨数据源的等价事件必须使用统一 `event_group_id`。
- 规范化后完全相同的问题必须属于同一事件组；相同问题与观测时间的副本只计一次，优先保留 `eval_only` 来源。
- 训练标签须在验证开始时可用，验证标签须在测试开始时可用，测试标签须在评估截止时可用。缺失或延迟标签会被记录并剔除。

完整字段、切分边界、分数定义和局限见 [数据与评估契约](docs/data_contract.md)。时间校验检查的是已声明元数据；它无法证明来源内容从未修改，也不会自动完成语义去重。真实基准报告前仍需审核历史内容快照和事件分组。

## 测试

无需安装测试依赖：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

测试覆盖时间泄漏、标签隔离、事件组完整性、跨数据源重复、标签可用时间、概率与指标边界、市场同样本比较、CLI 失败路径及回放确定性。

## 真实数据导入

数据源版本和文件 SHA-256 已登记在 [来源配置](configs/data_sources_v1.json)。以下命令从上游固定 revision 下载约 9 MB 的 Prophet CSV 及其数据卡，不执行上游脚本：

```bash
PYTHONPATH=src python -m foretellmesh fetch \
  --source prophet_arena --output data/raw/prophet_arena_v1

PYTHONPATH=src python -m foretellmesh ingest \
  --source prophet_arena \
  --data data/raw/prophet_arena_v1/subset_data_1200.csv \
  --output runs/prophet_source_audit
```

ForecastBench 目前固定一个轮次用于**格式接入审计**，不是已批准的最终评估集：

```bash
PYTHONPATH=src python -m foretellmesh fetch \
  --source forecastbench --output data/raw/forecastbench_v1

PYTHONPATH=src python -m foretellmesh ingest \
  --source forecastbench \
  --data data/raw/forecastbench_v1/2026-01-04-llm.json \
  --resolutions data/raw/forecastbench_v1/2026-01-04_resolution_set.json \
  --output runs/forecastbench_source_audit
```

`ingest` 默认生成审计、候选数据、逐样本状态与时间证明模板。未补齐元数据时，`records.jsonl` 为空；这是预期的隔离行为，不是成功完成真实基准评估。只有附带可追溯时间证明、通过事件分组和标签检查的记录才会写入 `records.jsonl`。

来源核查结果、PMA 的支持范围，以及 `--annotations` 用法见 [真实数据接入说明](docs/source_ingestion.md)。标准库测试无需联网；Parquet 适配与相应测试需要可选依赖 `python -m pip install -e '.[parquet]'`。

## 后续里程碑

1. 完善已接入的 Polymarket / Kalshi 历史回放：继续补 Kalshi 历史规则版本，扩充事前证据与独立事件，审核关联事件及基准重叠。Polymarket 首批精确结算和规则补证已完成；前瞻快照继续保留，既有 6 条保留集样本继续用于管线检查。
2. 在已完成的事件字段修复上，改善任务约束遵守、缺失参数的反事实训练和正反证据判断；保留已发现的模板依赖与退步样本，扩充独立验证。真实事件 SFT 可选，教师答案缺失不自动阻止独立评分或后续 RL。
3. 逐个训练和验证能力 LoRA，再消融 Risk / Critic 的增益；冻结输入与事件切分，测量 RL 的资源需求。只有通过行为与回归检查的候选才进入后续预测训练和 RL。

目前已有数据管线、确定性基线评估、LangGraph / 串行多 Agent 编排、PEFT 推理后端、LoRA 资源测试及合成能力训练/对照评估入口。完整在线检索、真实事件模型评测、GRPO 训练器和 vLLM 服务尚未接通；已有真实样本上的确定性基线分数，尚无真实事件上的模型预测性能结论。项目约束见 [AGENTS.md](AGENTS.md)。
