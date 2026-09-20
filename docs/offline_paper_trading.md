# 原始 Base 的 100 美元历史模拟交易

用户于 2026-09-19 明确：最终目标是多 Agent 自动模拟交易获得净收益；先部署离线 Qwen3-8B-Base 做模拟，再判断微调是否必要。100 美元指每个独立策略账户的初始虚拟本金，持仓占用资金，结算后资金才能重新使用。

## 首轮实测结果（2026-09-19）

**原始 Base 的离线推理与模拟交易闭环已跑通，尚未观察到盈利能力。** 本轮不加载 LoRA、不更新参数；91 条验证观测覆盖 55 个合约、13 次独立宏观发布，全部生成有效预测。Research 与 Forecast 共调用模型 182 次，生成流程累计约 23 分钟，单次流程平均 15.14 秒，峰值分配显存 15.62 GiB、峰值预留 15.76 GiB。最终测试集仍封存。

主成本场景 `cost_assumption`（每份买入溢价 $0.01，加买入名义金额的 1% 假设费用）：

| 策略 | 初始本金 | 期末现金 | 净盈亏 | 模拟成交数 |
|---|---:|---:|---:|---:|
| 原始 Base 多 Agent | $100.00 | $100.00 | $0.00 | 0 |
| 市场概率基线 | $100.00 | $100.00 | $0.00 | 0 |
| 持有现金 | $100.00 | $100.00 | $0.00 | 0 |
| 训练事件固定基率 | $100.00 | $78.96 | -$21.04 | 15 |

Base 的 91 条概率几乎复制输入市场价格，全部触发 `no_cost_adjusted_edge`，所以账户保持空仓。零成本和高成本场景中也均为零交易、零收益。Base 的 Brier 为 0.090393737，市场基线为 0.090393736，未显示新增预测信号；不能把零亏损解释为交易成功，也不能据此判定 Base 在其他输入和策略下不可能盈利。

固定基率基线在零成本、主成本、高成本场景分别成交 14、15、15 笔，期末现金分别为 $79.04、$78.96、$78.77。主成本场景费用合计 $0.247510，收益率 -21.039584%，事件时点权益代理值最大回撤约 21.04%（有 2 个陈旧估值点）。成交数随成本变化并非单调关系，因为成交时机、资金占用和后续风险限制会共同变化。该对照表明模拟器能执行有价差的信号并记录亏损；当前 Base 的零成交来自信号判定。

原始模型响应重放、91/91 决策参考价与历史库核对、12 个策略场景的逐笔账本和指标精确复算均通过。完整测试共 422 项：421 项通过，1 项可选 GPU 测试跳过。

本地运行产物（`runs/` 不纳入 Git）：

- [模型运行报告](../runs/qwen3_8b_base_paper_forecasts_v1/report.json)、[资源统计](../runs/qwen3_8b_base_paper_forecasts_v1/resource_summary.json)、[原始输出审计](../runs/qwen3_8b_base_paper_forecasts_v1/audit.json)。模型报告 SHA-256：`f65447eec9848676e8e686bcbca584240126875d7e850552282131f1710dac36`。
- [模拟交易报告](../runs/qwen3_8b_base_paper_100usd_v1/report.json)、[账本审计](../runs/qwen3_8b_base_paper_100usd_v1/audit.json)、[账户结果与权益曲线](../runs/qwen3_8b_base_paper_100usd_v1/paper_trading.png)。交易报告 SHA-256：`8568405488475dd6f17c8df946f156d5ab9b8650be7a1a36149a791d62233683`。

下一步仍优先使用原始 Base，补充严格截至观测时刻的证据和可计算特征，再按预先冻结的交易规则比较新增信号。本轮尚不能证明微调有必要，不启动新的训练。

## 首版流程

```text
经过时间核验的历史证据与市场价格
    → Research Agent（原始 Base）
    → Forecast Agent（同一个原始 Base）
    → 成本后的价差判定
    → 现金 / 单笔 / 事件组 / 总持仓限制
    → 决策之后的成交记录驱动模拟成交
    → 持仓至已核验结算
    → 账户账本、净收益与回撤报告
```

模型在 `conda lab`、RTX 3090 上离线批处理部署。LangGraph 编排两个模型角色，共享一份 BF16 基座，固定 revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`；不构建或加载 PEFT 适配器，不进行参数更新。仓位、成本、成交和记账用确定性代码执行。本入口是本地历史批处理，不是常驻 HTTP 服务。

首轮使用 `pma_macro_research_20260919_v1` 全部 91 条 validation 观测、55 个合约、13 次宏观发布，按时间回放；最终 test 不打开、不调参。训练分区标签仅用于独立固定基率基线，不微调模型。新生成的模型请求不含 outcome、未来结算信息或今日检索内容。

已有原生价格库保存 779 个候选市场的 5,651,134 笔去重成交，首轮先使用规则和结算已核验的上述子集。大规模归档和已合格回测样本是不同层次，不把缺少规则或结算核验的合约自动并入收益统计。

## 冻结交易规则

| 参数 | 首版设置 |
|---|---:|
| 初始模拟本金 | $100 |
| 单笔含成本预算上限 | $2 |
| 同一宏观事件组的在途订单 + 持仓成本上限 | $5 |
| 总在途订单 + 持仓成本上限 | $20 |
| 最小分配预算 | $1 |
| 计入假设成本后的最小预期优势 | 每份 $0.03 |
| 决策参考价最大年龄 | 3 小时，另计模型延迟 |
| 等待后续成交参考的最长时间 | 300 秒 |
| 退出规则 | 持有到已核验结算 |

这是固定小额研究配置，未按当前回测结果优化，不代表最优仓位或平台最小订单要求。相同合约已有订单或持仓时，不因另一条观测重复加仓；同一次发布的相关合约共享事件风险上限。账户不借款、不补充本金。

Forecast 给出 Yes 概率 p，观测时的 Yes 成交参考为 q。候选 Yes 的预测兑付为 p、No 为 1-p；成本参考分别为 q 和 1-q，再加价格溢价和费用。只在成本后的优势达到阈值时提交模拟意图，否则记为 HOLD。模型照抄市场时，在此规则下不会为了产生交易记录而强制买入。

## 历史成交与成本假设

当前 PMA SQLite 库提供成交记录和区块时间，没有完整历史订单簿、可成交深度和队列。因此首版是**历史成交价情景仿真**，不是历史订单簿撮合重建。

决策时刻为 observation_time 加该次真实模型流程的耗时（向上取整秒）。下单时先预留预算；仿真只看决策之后 300 秒内的第一个成交区块均价。这个参考价加上场景溢价后仍须满足事先确定的价格限制，否则取消；不跳过不利成交去寻找后面的有利价格。未取得后续参考、已结算或资金不足时，均不假设成交。

满足限制时，首版假设小额订单可以全部成交；未建模盘口容量、队列、最小订单量、平台 token 精度规则或我们的市场冲击。模拟份额、买入名义金额和费用各自向下保留六位小数，避免循环小数成交均价造成记账尾差；这是本模拟器的数值规则，不是平台实际费用取整规则。No 的参考用 1 减 Yes 成交价构造，也属于仿真假设，不是已观察到的 No 卖价。

三种场景各自从 $100 开始，全部在模型生成前冻结：

| 场景 | 买入每份价格溢价 | 买入名义金额费用 |
|---|---:|---:|
| frictionless_reference | $0 | 0% |
| cost_assumption | $0.01 | 1% |
| cost_stress | $0.03 | 2% |

这些费用是敏感性测试假设，**不是 Polymarket 历史实际费率**。未来接入真实报价时，应按市场及当时版本记录费用参数，不能把今天的费用表回填历史。官方订单簿数据含价格和数量，费用文档另有市场参数说明：[实时订单簿](https://docs.polymarket.com/market-data/realtime-data)、[费用](https://docs.polymarket.com/trading/fees)（查阅于 2026-09-19）。

## 对照与账户指标

每个成本场景比较四个独立账户：

- `base_multi_agent`：本轮离线 Research → Forecast Base 输出。
- `market_implied`：预测等于输入市场概率，检验零信息增益策略的行为。
- `training_event_prior`：只用训练分区结算标签按事件组等权计算的固定概率；不使用验证标签拟合。
- `cash`：不交易，余额保持 $100。

为了对照预测信号本身，基线使用相同的决策时刻、延迟和资金约束。报告期末现金、净盈亏、收益率、实际模拟成交数、费用、胜率、HOLD 原因和逐笔资金流。没有成交时胜率为 null，不能把不交易包装成成功交易。

回撤来自回放事件时点的**权益代理值**：现金 + 预留资金 + 按当时最新历史成交价并扣除假设退出成本估值的持仓。明确记录陈旧估值的点数；没有逐 tick 全量盯市，因此它不是完整盘中最大回撤。持有到结算的实际账本只收取上述买入费用，不再凭空收取卖出费用。

已审核结算的链上 `resolution_time` 驱动历史兑付。标签的 `available_at` 是此次归档采集时间，不伪装成历史采集时间；二者的信任范围与数据集审计保持一致。结算标签只进入环境结算和事后评分，不进入 Agent 或下单函数。

## 复现

输出目录须不存在。推理和账本分别保留代码、配置及数据哈希。运行在仓库根目录：

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
conda run --no-capture-output -n lab python -m foretellmesh.offline_base run \
  --dataset data/processed/pma_macro_research_20260919_v1 \
  --config configs/offline_base_forecasts_v1.json \
  --agent-config configs/capability_agents_v1.json \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --paper-config configs/paper_trading_100usd_v1.json \
  --output runs/qwen3_8b_base_paper_forecasts_v1

PYTHONPATH=src conda run --no-capture-output -n lab python -m foretellmesh.paper_backtest run \
  --forecast-run runs/qwen3_8b_base_paper_forecasts_v1 \
  --dataset data/processed/pma_macro_research_20260919_v1 \
  --trade-store data/processed/pma_macro_trade_store_20260919_v4 \
  --config configs/paper_trading_100usd_v1.json \
  --output runs/qwen3_8b_base_paper_100usd_v1

PYTHONPATH=runs/qwen3_8b_base_paper_forecasts_v1/source_snapshot \
conda run --no-capture-output -n lab python -m foretellmesh.offline_base audit \
  --run runs/qwen3_8b_base_paper_forecasts_v1

PYTHONPATH=runs/qwen3_8b_base_paper_100usd_v1/source_snapshot \
conda run --no-capture-output -n lab python -m foretellmesh.paper_backtest audit \
  --run runs/qwen3_8b_base_paper_100usd_v1
```

若推理后修改交易配置，回放入口拒绝沿用原运行：成本与风险规则须在生成前冻结。每个场景生成每种策略的 `.ledger.jsonl` 和 `.equity_curve.jsonl`；`audit` 重建信号、复算资金流和指标，逐条核对账本。

## 怎样判断是否微调

先用这套闭环确定是证据覆盖不足、Base 不会使用证据、交易规则不足，还是成本和成交约束消除了优势。输入和工具改进也应先在原始 Base 上比较。只有出现明确且可验证的参数适配需求，再设计 LoRA；当前不启动额外 SFT、CPT 或 RL。

首轮属于开发集诊断。市场选择、合格数据覆盖和历史结果可能进入底模预训练等限制仍然存在。即使某个场景盈利，也须在新时间段和更多独立事件验证，不能仅凭本轮将策略晋级。
