# Base 的多 Agent 行为基线

本轮使用 `conda lab`、RTX 3090 和固定 revision 的 `Qwen/Qwen3-8B-Base`，检验概率 JSON、证据引用和工具调用。权重保持冻结，不加载诊断 LoRA，不使用真实市场结果训练。起始配置见 [agent_behavior_baseline_v1.json](../configs/agent_behavior_baseline_v1.json)。

## 样本与比较方法

从既有合成集的 **validation** 部分，按预先声明的字典序规则为每个数学任务族选 1 个事件组，保留中英文两种表达。共 6 个独立事件组、12 条问题。最终 test 部分不参与此次选择或推理。预处理重放数学证明，并校验输入、目标、抽样结果和原始归档哈希；来源被篡改则拒绝构建。

三个流程使用同一组输入：`single_forecast`、`research_forecast`、`reviewed_forecast`。后者是 Research → Risk → Forecast → Critic。另将混合概率与贝叶斯信号的 4 条问题用于 `calculate`，总计 40 项流程测试。

预测任务的证据包含确定性工具已经算出的概率，用于检查读取、引用和报告计算结果的能力。工具任务仅提供情景设定，移除现成计算答案，要求模型生成正确工具名与参数。输入文件与评分文件分开保存；运行时模型只接收 `ForecastInput`，没有结算标签或评分目标。

生成使用 BF16、无量化、greedy decoding，每次调用最多 384 个新 token，总上下文上限 2048，不截断输入。每个阶段至多修复一次，随后明确返回失败。为了减少总是先执行同一流程的顺序偏差，每个问题轮换三个流程的执行顺序。每次调用预算一致，**总计算预算并不一致**，因此结果不能单独归因于 Agent 分工。

三轮都保留原始请求和回答，使用哈希完全相同的样本、评分文件与 Agent 配置：

| 配置 | 提示 | 回答解析 |
| --- | --- | --- |
| [v1](../configs/agent_behavior_baseline_v1.json) | 既有角色提示 | 只接受纯 JSON |
| [v2](../configs/agent_behavior_baseline_v2.json) | 增加禁止围栏及后续文本的明确要求 | 只接受纯 JSON |
| [v3](../configs/agent_behavior_baseline_v3.json) | 同 v2 | 额外允许完整、单一的 `json` 代码块 |

v2/v3 是看到开发验证集的失败后进行的诊断，不是新的独立测试。v3 只移除完整代码块的外部围栏，不从文字中搜索答案、不选择多个候选、不修补内容；重复 JSON 键、越界概率、未知引用、错误时间及额外字段仍被拒绝。运行器默认仍使用 `strict_json`，实验通过 `response_transport="single_json_fence"` 显式启用解包。`decoded_transport` 和对应计数区分原生纯 JSON 与解包后通过的回答，不把接口修复称为模型学会了格式。

## 如何解读指标

- 首先看最终预测覆盖率、首次完成数、修复次数、非法输出和工具参数匹配。格式错误或后端异常的样本保留为缺失，不补成 0.5，不从分母中删除。
- 数学概率的 MAE 和容差内完成率检验数值报告。容差内完成率以全部问题为分母；覆盖子集上的 MAE 必须同时看覆盖率。
- Brier、Log Loss、ECE 使用合成抽样结果，只是评分实现和行为诊断。中英文共享结果，12 条问题不能当成 12 个独立事件；本轮样本不能支持真实预测或校准能力结论。
- 每条流程报告同一覆盖子集的 0.5 与数学 oracle 分数；跨流程另报共同完成子集。没有共同完成样本时分数为 `null`，不能宣称某流程更准确。
- `failure_penalized_brier_diagnostic` 给失败项记损失 1，用于显式展示失败代价；它不是对缺失概率计算的正规 Brier Score。
- 证据召回只检查预期 ID 是否被引用，不能证明自由文本得到事实支持。工具同时比较调用参数和计算结果，避免“碰巧得到正确数值”被记为参数正确。
- Critic 前后的比较只在相同完成样本上进行。预测已成功但 Critic 失败时，原预测留在阶段日志中，最终输出仍为失败。
- 每项记录模型调用次数、输入/输出 token、生成耗时、总延迟及峰值分配显存。没有单独预热，首项可能包含首次推理开销；本轮是串行本地诊断，不是服务吞吐测试。

## 复现

输出目录必须不存在；以下使用新目录保留既有实验。

```bash
conda activate lab
PYTHONPATH=src python -m foretellmesh prepare-agent-baseline \
  --raw-dataset data/processed/synthetic_sft_raw_v1 \
  --split-config configs/synthetic_sft_splits_v1.json \
  --agent-config configs/capability_agents_v1.json \
  --evaluation-config configs/agent_behavior_baseline_v3.json \
  --output data/processed/agent_behavior_validation_cohort_next

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m foretellmesh.agent_baseline \
  --cohort data/processed/agent_behavior_validation_cohort_next \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --output runs/qwen3_8b_base_agent_behavior_validation_next
```

固定样本包为 `data/processed/agent_behavior_validation_cohort_v{1,2,3}`，包含模型输入、独立评分文件、配置副本与哈希清单。`runs/qwen3_8b_base_agent_behavior_validation_v{1,2,3}` 保存完整 `results.jsonl`、`report.json` 和运行时源码快照 `source_snapshot/foretellmesh`。逐项结果持续落盘；若评测中断或显存不足，只保留部分结果，不发布完整指标。

离线核验脚本位于 `runs/agent_behavior_validation_comparison_v1/reproduce.py`。它验证样本、请求、输出日志及源码哈希，重算全部指标，并检查三轮使用同样的样本与 Agent 配置：

```bash
PYTHONPATH=src python runs/agent_behavior_validation_comparison_v1/reproduce.py
```

需要复现旧代码时，将 `PYTHONPATH` 指向对应运行的 `source_snapshot`，再调用相同模块入口，输出到新目录。不要覆盖已有记录。

## 严格纯 JSON 对照结果

2026-09-18 的前两轮已完整执行，每轮 40 项流程测试。

| 配置 | 单预测完成 | 研究→预测完成 | 研究→风险→预测→批评完成 | 工具参数正确 |
| --- | ---: | ---: | ---: | ---: |
| v1 原提示 | 9/12 | 0/12 | 0/12 | 0/4 |
| v2 明确纯 JSON | 8/12 | 0/12 | 0/12 | 2/4 |

v1 的 80 次调用中，71 次被纯 JSON 解析拒绝，全部是围栏输出；仅在离线诊断中移除完整围栏后，62 次的内容可以通过相应角色校验，其余仍存在字段问题。**这 62 次没有计入 v1 的成功数。** v1/v2 两种多 Agent 流程均停在 Research，不能据此断言 Risk、Critic 或多 Agent 分工没有价值。

两轮中成功的单预测均正确报告了输入证据中的数学概率，概率 MAE 为 0；没有调用触及 384 token 输出上限。v2 未稳定提高预测覆盖率，因此没有把提示变体直接升级为默认行为。后续 v3 单独检验严格解包的工程效果，并保留前两轮结果。

## v3：解包后的实际能力与成本

最终报告：`runs/qwen3_8b_base_agent_behavior_validation_v3/report.json`。40 项全部执行完毕，36 项预测流程完成，2 项工具任务完成、2 项工具任务明确失败。下表完成率允许每阶段一次修复，首次完成单独报告。

| 流程 | 最终完成 | 无需修复完成 | 概率 MAE | 总模型调用 | 总输入 / 输出 token | 平均耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forecast | 12/12 | 10/12 | 0 | 14 | 8,653 / 2,179 | 6.52 秒 |
| Research → Forecast | 12/12 | 9/12 | 0 | 27 | 16,050 / 2,840 | 8.57 秒 |
| Research → Risk → Forecast → Critic | 12/12 | 9/12 | 0 | 51 | 32,981 / 6,367 | 19.36 秒 |

三条流程分别有 10、20、45 次有效调用依赖围栏解包；这不是原生纯 JSON 格式通过率。三条流程的预期证据 ID 平均召回均约 0.611，多数回答只引用计算而未同时引用设定。所有调用均未触及输出上限，也没有后端异常；峰值分配显存 **15.51 GiB**。上述资源只代表本次串行 BF16 推理，不代表 RL 或并发服务显存。

三条流程最终给出的概率完全相同。Critic 的 12 次配对检查均接受原概率，没有修改。相同覆盖子集上的合成结果 Brier 为 **0.209357**、Log Loss 为 **0.630340**、ECE 为 **0.280578**；数学 oracle 本身在这批抽样结果上也是这些分数。小样本 ECE 不能被解读为模型校准结论。增加角色没有提高这批工具结果读取任务的概率精度，四角色流程平均耗时约为单预测的 2.97 倍。

工具任务中，贝叶斯信号的中英文两条均选择了正确参数；混合概率的两条均把含义填反：

```json
{"name":"weighted_probability","arguments":{"probabilities":[0.56,0.44],"weights":[0.94,0.15]}}
```

正确的含义是情景权重 `[0.56, 0.44]`，对应成功率 `[0.94, 0.15]`。模型输出的权重和为 1.09，确定性工具拒绝执行；返回错误后模型仍重复同一错误。因此最终工具参数与结果正确数均为 **2/4**。这里两种语言仅对应 2 个独立工具事件，不能据此估计广泛工具准确率，但已经揭示了一个可复现的参数映射问题。

另做了两项具名案例的人工抽查，不将其泛化为完整事实支持率：

- 混合情景英文样本 `synthetic:0068d947004cf0b1e00aad76:en`：Risk 引入“不可预见的技术故障”，并引用 `setup`，而原证据仅规定合成概率情景，没有技术故障信息。引用存在无法证明引用支持该判断。
- Beta 情景英文样本 `synthetic:07f697ffa921ada80e59a350:en`：输入给定 Beta(1,1) 先验，Forecast 的 `base_rate` 仍为 `null`，Critic 接受该预测。算对最终概率不能代替完整使用先验和有效检查。

## 后续训练优先级

首个能力适配器仍优先 **`research_tool_lora`**：补工具参数语义、证据支持与假设区分、遗漏信息检测，以及出错后的有效修复。当前 432 条合成预热集主要是 Forecast 回答，不能直接当作 Research/Risk/Quant 的合格训练答案；须另构造带确定性工具判据和明确证据支持标注的角色任务，并继续按事件组隔离。

之后再评估 `forecast_lora` 的先验使用、不确定性和真实历史预测任务。在满足行为要求、历史规则与标签核验后接入 Brier 奖励 RL；真实事件教师 SFT 保持可选。风险和批评暂时按任务需要调用，继续增加包含受控错误预测的 Critic 测试，验证它是否会发现错误，而不仅是接受已经正确的概率。

本轮没有训练或发布正式 LoRA，没有启动 RL，也没有获得真实市场预测能力提升的证据。Polymarket / Kalshi 历史数据的规则与结算证明审核、ForecastBench 保留策略均继续沿用原流程。

核验记录在 `runs/agent_behavior_validation_comparison_v1/report.json`：三轮共 **120 项流程测试**，源码快照、输入和请求哈希通过核验，全部指标可从日志精确重算。全量测试 **177 项：176 通过、1 项可选 GPU 测试跳过**；上述三轮完整 8B 推理另行在 3090 上执行完成。
