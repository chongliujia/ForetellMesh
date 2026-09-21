# 预测信号与模拟交易准入分离

这次修正针对已经确认的机制错误：把缺少独立依据的旧预测概率当作估值，价格移动后据此建仓。它不产生新的预测能力，也不把空仓记作盈利策略。

## 已完成结果（2026-09-20）

六种成本下，新准入策略均为 **$100 → $100、零订单、零成交、零费用、零回撤**，与现金基线一致。旧策略的全部决策、账本和账户曲线与原 v2 完全一致，主成本仍为 **$90.549037**，其余五个场景也保持原值。

这不是一次收益优化或盈利突破：本轮没有合格方法，零成交是预期机制行为。它证明了未验证的预测不会再自动产生价格差交易；预测层的价值仍待建立。

84 份历史输出的资格检查为：5 份失败或过期，31 份没有引用准入的外部事件证据，48 份引用了证据但方法未通过资格验证。31 份指没有实际引用外部证据，不等同于输入包里是否存在材料。所有原始预测、失败及时间戳均保留。

**33 项相关测试通过**，包括 10 项新准入测试；18 组回放从冻结源码独立复算一致，核对 **553,571 条决策记录**。首次生成及复算均重建原历史输入、校验原始 Agent 响应，未新增模型调用或训练。最终测试封存。

产物：[`report.json`](../runs/pma_forecast_admission_mechanism_v1/report.json)、[`audit.json`](../runs/pma_forecast_admission_mechanism_v1/audit.json)、[`admissions.jsonl`](../runs/pma_forecast_admission_mechanism_v1/admissions.jsonl)。报告 SHA-256：`8c962549df0f629591bbb836f01a58586ced8d2840e7badc02155bf61716e86f`。

## 行为变化

`event_allocation.simulate` 的新调用默认要求 `forecast_admission`。没有准入信息的预测仍可保留为研究记录，但不能触发新的买入或基于概率的普通卖出。小时价格止损、显式持仓上限对照、结算与账户记账继续执行；准入失效本身不强制平仓。

每份准入记录绑定原信号的完整哈希、方法版本、信息来源类型及被引用的外部证据时间。方法是否经过验证与单条预测数值分开：

- 市场参考只是参考，不作为独立事件估值。
- 依赖市场价的预测不因为偏离市场就自动获准。
- 隐藏价格的预测不因为形式上独立就自动获准。
- 外部证据引用、时间检查和方法资格全部通过后，仍须满足原有成本后优势、现金、仓位及事件风险限制。
- 提交买单和模拟成交时都重新检查资格，避免资格过期但挂单继续成交。
- 方法资格的可用时间必须不晚于预测观察时间，且成交时尚未过期；不能将后来的评测结论回填为历史已知能力。

该准入器是执行边界，不是自动统计检验器。`qualifications` 是由单独研究审核提供的可信方法登记表，引用评测和审计哈希及有效时间；它不能来自 LLM 的自我声明。本轮登记表为空，实验入口也拒绝非空登记表，因此没有向任何真实方法发放资格。后续签发资格需要另外完成可复现、隔离事件的评测和报告核验，不能仅填入一个哈希字符串便宣称模型已验证。

旧 v2 实验入口显式使用 `legacy_unvalidated_research=True`，用于保持原来的研究协议和轨迹。该绕过不能与新准入器同时传入，也不是新实验的默认值。原有运行目录和冻结源码均保留。

## 本轮验证范围

配置：[forecast_admission_mechanism_v1.json](../configs/forecast_admission_mechanism_v1.json)。重建原始 84 份 Agent 输入与响应，在相同训练市场、历史价格、结算、风险参数和六种费用假设下比较现金、旧未准入策略、新准入策略，共 18 组回放。

没有新训练、模型调用、验证集回放或最终测试访问。原始研究信号不会被删除或改成失败；新的决策日志明确记录“没有外部引用”“方法未验证”等拒绝原因。交易数据不足和事件预测能力不足仍是后续研究问题。

合成测试包含正向准入样例，仅用于验证执行分支能够买入、按费用记账和结算；该夹具的资格与收益均不代表任何历史模型通过评测。测试还覆盖旧价格下跌制造表面优势、夸大置信度、证据越过截止、信号哈希变更、未来资格回填、挂单期间资格过期和资格过期后的价格止损。

## 复现

```bash
PYTHONPATH=src:/tmp/foretellmesh-rl-deps MPLCONFIGDIR=/tmp/foretellmesh-mpl OMP_NUM_THREADS=4 \
python -m foretellmesh.forecast_admission_experiment \
  --config configs/forecast_admission_mechanism_v1.json \
  --output runs/pma_forecast_admission_mechanism_v1

PYTHONPATH=runs/pma_forecast_admission_mechanism_v1/source_snapshot:/tmp/foretellmesh-rl-deps \
MPLCONFIGDIR=/tmp/foretellmesh-mpl OMP_NUM_THREADS=4 \
python -m foretellmesh.forecast_admission_experiment \
  --config runs/pma_forecast_admission_mechanism_v1/config.json \
  --output runs/pma_forecast_admission_mechanism_v1 --reproduce
```

首次运行要求输出目录不存在。复算逐项比较全部决策、账本、权益曲线和汇总；旧策略还逐项与此前 v2 的原始轨迹比较。报告必须将本实验标为回顾性机制检查，不能用它声称已实现样本外盈利。
