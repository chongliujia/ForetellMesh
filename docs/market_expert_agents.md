# 原始 Base：量化专家与博弈论 Agent 对照

本轮按用户要求，在离线 Qwen3-8B-Base 系统中加入两个任务角色。继续使用 `conda lab`、RTX 3090、BF16、单份常驻基座，不加载适配器、不训练。新增流程通过 LangGraph 串行执行；每次请求传递结构化结果，不传递完整对话。

## 实测结果（2026-09-19）

**已完成 13 个验证事件组、四组共 52 条流程的真实 GPU 对照，尚未看到预测或收益改善。** 第三版共调用模型 155 次，生成约 20 分钟；50 条流程有效，2 条在量化阶段失败并停止下游。峰值分配显存 15.85 GiB、预留显存 16.33 GiB，模型请求全部使用原始 Base。

| 组别 | 有效流程 | 平均每条决策耗时 | 主成本场景期末资金 | 模拟成交 |
|---|---:|---:|---:|---:|
| 仅增加相同历史特征 | 13/13 | 16.17 秒 | $100 | 0 |
| 加量化专家 | 12/13 | 23.78 秒 | $100 | 0 |
| 加博弈论专家 | 13/13 | 22.77 秒 | $100 | 0 |
| 两位专家都加 | 12/13 | 29.85 秒 | $100 | 0 |

每个账户初始 $100。零成本与高成本场景也均为零成交、零盈亏；原 Base 缓存、市场概率及现金对照同样保持 $100。耗时包含该条已记录的特征计算时间；失败样本计入平均耗时和覆盖率，不从账户回放中删除。

除了博弈论组的一条预测，所有有效预测与市场概率之差都未超过 `1e-6`。在各模型共同有效的 12 条观察上，仅加特征、加量化、两位都加以及原 Base 的 Brier 均约 **0.166915**；博弈论组为 **0.175248**，更差。对应 Log Loss 约 0.451510 与 0.468232，ECE 约 0.220878 与 0.304212。这里使用相同样本比较，不因量化组缺失一个观察点而把不同分母的分数直接排名；13 个事件也不足以得出稳定能力结论。

唯一明显改概率的样本是 `pma:550432:2025-06-26T12:30:00Z`：合约询问 2025 年 6 月失业率是否**恰好为 4.3%**，市场概率为 0.45，博弈论组 Forecast 给出 0.55，结算标签为 No。主成本场景产生 $2 的 Yes 买入意图，但在决策后 300 秒内没有后续成交参考，按 `no_post_decision_print` 取消；没有把信号当作已成交，也未扩大成交窗口。

该样本的博弈论文本把 0.45 解释成交易者相信失业率“低于 4.3%”。这一推论缺少支持：恰好等于 4.3% 的互补事件还包括高于 4.3%。最终 0.55 又恰好是市场 Yes 概率的互补数，存在合约方向或概率含义混淆的迹象，不能把这次变化记为有依据的独立信号。这个检查只说明文本的问题，不能确定模型内部改变概率的原因。

两条失败都来自 `pma:551588:2025-07-08T12:30:00Z` 的量化输出：首次缺少顶层字段，修复后改错观察时间，被校验拦截。第三版共两次格式修复，没有生成达到 token 上限的调用。25 次有效博弈论输出均选择 `information_aggregation`，实际尚未展现丰富的机制区分。各角色生成速度约 25.5–27.9 token/秒；更详细的角色资源和失败原因见运行产物。

历史特征读取了 27,070 条成交记录。原始模型输出、特征重建、21 个独立策略场景的账本和指标全部重放通过；完整测试 437 项，436 项通过、1 项可选测试跳过。新增角色保持可选，不晋级默认流程。下一步应先改善可验证的量化信号与合约语义一致性；本轮不支持直接启动微调或继续增加角色。

本地文件（`runs/` 不入 Git）：

- [评估报告](../runs/qwen3_8b_base_market_experts_pilot_v3/evaluation/report.json)、[审计结果](../runs/qwen3_8b_base_market_experts_pilot_v3/evaluation/audit.json)、[对照图](../runs/qwen3_8b_base_market_experts_pilot_v3/evaluation/market_experts.png)。报告 SHA-256：`1aa57c050f61c6a7ed87cc0f748c22aa657117926ba2cd476d3825d5f8b3dcac`。
- [原始响应](../runs/qwen3_8b_base_market_experts_pilot_v3/results.jsonl)、[角色诊断](../runs/qwen3_8b_base_market_experts_pilot_v3/role_diagnostics.json)、[生成报告](../runs/qwen3_8b_base_market_experts_pilot_v3/generation_report.json)。生成报告 SHA-256：`3de53b03d4f0db63e27b1fc5b0914a75689b81865c1643faa589c4c5b13a7f47`。

## 两个角色

`market_quant` 是市场量化分析专家，与原先只选择 Bayes / 加权概率工具的 `quant` 角色分开。Python/SQLite 先计算截至观察时刻的特征：最近成交参考价、报价年龄，1 小时 / 24 小时 / 7 天的概率点变化、成交记录数、成交区块数、区块均价范围和标准差。模型解释至多三个已有数值特征，不负责算数，也不能引用不存在或值为 null 的指标。

特征只读取区块时间不晚于观察时刻的记录。窗口是 `(T-window, T]`；变化的起点采用不晚于窗口起点、且最多陈旧三小时的最近区块均价，缺少合格起点则为 null。区块均价按成交记录等权，离散度按不规则区块采样计算；没有把它当成规则时间收益波动率、预测误差或事件发生概率。成交笔数不等于资金成交量，价格库没有历史盘口容量和交易者身份。

`game_theory` 分析信息聚合、可能的对冲需求、流动性约束、注意力和政策参与者激励，每次选择一个主要机制，输出一个条件情景。`assumption` 必须以 `If ` 开头，并明确条件成立时的 `implication`；已有证据只是情景背景，不能证明假设成立。提示明确区分市场交易者和政策制定者，不允许从价格和笔数推断内幕信息、仓位、操纵或交易者身份，也不把交易者行为当成宏观结果的成因。

两位专家输出方向判断（支持 Yes、支持 No、混合、没有独立优势）及未知项，最终概率仍由 Forecast 生成。方向性文字不是经过校准的预测，也不会直接绕过交易风控。当前结构校验能验证字段、时间、证据 ID 和特征存在性；自然语言假设是否合理仍需分析，不能把 schema 有效等同于经济推理正确。

## 冻结的开发集试跑

从已有 91 条验证观测、13 个宏观事件组中，每组选一个观察点：先取最早观察时刻，再取当时概率最接近 0.5 的合约，最后以 sample_id 排序打破平局。选择过程不读取标签，最终测试集继续封存。这样得到 13 个事件组的接口与信号诊断，不能替代全部合约回放或独立盈利评测。

| 组别 | 模型调用顺序 | 输入特征 |
|---|---|---|
| features_only | Research → Forecast | 原证据 + 相同历史价格特征 |
| quant | Research → Market Quant → Forecast | 同上 |
| game | Research → Game Theory → Forecast | 同上 |
| quant_game | Research → Market Quant → Game Theory → Forecast | 同上 |

四组 Research 使用相同输入；两位专家各自读取原输入，互不复制对方结论。Forecast 接收相同完整证据与特征，以及该组的简短结构化角色输出，因此增加角色不会暗中增加信息源。保留原始 Base 上一轮相同 13 条观测的缓存结果作额外参照；该缓存没有新特征，且旧版不修复格式，与本轮不是完全相同配置。

新四组统一采用贪心生成、768 个新增 token、8192 上下文上限，并在生成内容恰好构成一个完整 JSON 对象（或完整 JSON 代码块）后停止；每个角色最多一次 schema 修复，最多八次模型调用。没有静默截断；预算不足或修复后仍失败，停止该条下游流程并以缺失预测计入覆盖率，交易层观望。四组总计 52 条工作流，未修复时为 156 次模型调用，按照样本轮换各组执行顺序。

仍使用已冻结的 [100 美元规则](offline_paper_trading.md)：单笔最多 $2、同事件最多 $5、总持仓加在途资金最多 $20，成本后至少每份 $0.03 优势，持有至核验结算。每个策略与成本场景独立从 $100 开始。记录每组实际模型延迟及特征计算时间，决策后才能寻找成交参考；角色增多带来的延迟是系统成本的一部分。市场概率与现金参照使用原 Base 的延迟。

三种费用与溢价仍是假设场景；历史成交区块均价不能证明当时存在可成交盘口。规则、样本、原始输入、特征摘要和源行哈希、代码、模型版本均在生成前冻结。评估阶段才解析结算标签。

## 输出边界诊断

首版生成诊断在 11/52 条工作流后中止，原始输出及中止原因保存在 `runs/qwen3_8b_base_market_experts_pilot_v1`，未对该轮计算收益或预测分数。多个博弈论输出在完整 JSON 后重复生成，导致整条响应校验失败。第二版只增加显式 `stop_on_json_object` 生成停止条件；角色提示、样本规则、seed 和交易策略不变，四组从头重跑。

停止条件只接受当前整个 completion 是一个严格 JSON 对象，支持嵌套、字符串转义和完整 JSON 围栏；不从已经生成的杂乱文本中抽取某个答案，不接受重复键、NaN、前后说明文字或多个对象。字段语义仍由原有校验器检查，错误对象仍需修复或失败。该功能默认关闭，仅通过新实验配置显式启用。

第二版又发现 Base 把 `market_view` 等顶层字段塞入情景数组，6/52 条后中止，未进行收益或预测评分。第三版将博弈论契约简化为七个顶层字段，只分析一个条件机制；保留同样的严格字段、证据、时间校验及一次修复上限，再次冻结全部输入和四组配置。第二版原始记录保存在 `runs/qwen3_8b_base_market_experts_pilot_v2`。两次调整仅依据接口失败，不使用结算答案优化提示。

## 验证与复现

新增测试覆盖未来价格不影响特征、窗口边界、陈旧起点、无标签样本选择、缺失数值和伪造证据拒绝、串行与 LangGraph 路由等价、相同 Forecast 输入、修复上限和失败停止、缺失预测计入覆盖率、逐笔账本与指标重放、配置和数据篡改检测。

输出目录须不存在，在仓库根目录运行：

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
conda run --no-capture-output -n lab python -m foretellmesh.market_experiment run \
  --dataset data/processed/pma_macro_research_20260919_v1 \
  --trade-store data/processed/pma_macro_trade_store_20260919_v4 \
  --baseline runs/qwen3_8b_base_paper_forecasts_v1 \
  --config configs/market_expert_experiment_v3.json \
  --agent-config configs/market_expert_agents_v1.json \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --paper-config configs/paper_trading_100usd_v1.json \
  --output runs/qwen3_8b_base_market_experts_pilot_v3

PYTHONPATH=runs/qwen3_8b_base_market_experts_pilot_v3/source_snapshot \
conda run --no-capture-output -n lab python -m foretellmesh.market_experiment evaluate \
  --run runs/qwen3_8b_base_market_experts_pilot_v3

PYTHONPATH=runs/qwen3_8b_base_market_experts_pilot_v3/source_snapshot \
conda run --no-capture-output -n lab python -m foretellmesh.market_experiment audit \
  --run runs/qwen3_8b_base_market_experts_pilot_v3

PYTHONPATH=runs/qwen3_8b_base_market_experts_pilot_v3/source_snapshot \
conda run --no-capture-output -n lab python scripts/plot_market_experts.py \
  --run runs/qwen3_8b_base_market_experts_pilot_v3

PYTHONPATH=runs/qwen3_8b_base_market_experts_pilot_v3/source_snapshot \
conda run --no-capture-output -n lab python scripts/summarize_market_experts.py \
  --run runs/qwen3_8b_base_market_experts_pilot_v3
```

审计重新构建特征、逐条重放原始模型输出、再计算全部账户账本和指标。不会自动把新增角色或某组结果设为默认，也不因试跑产生交易而启动微调。
