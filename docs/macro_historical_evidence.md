# 宏观预测市场历史补证

2026-09-19，本轮为固定目录的 408 个合约构建 816 个历史观测点。**480 个报价通过本地时间与价格检查，覆盖 22 个候选事件组；新增正式评分和 SFT 准入仍为 0。** 这些是证据覆盖进展，不是预测能力提升。本轮模型调用为 0，未产生 Brier / Log Loss / ECE 分数。

## 范围与结果

沿用 [固定目录](macro_evaluation_scope.md)：24 个计划槽位，March FOMC 因 ForecastBench 重合整组排除，February 发布的美国失业率缺少候选，留下 22 个有合约的事件组。408 个合约中 Polymarket 183 个、Kalshi 225 个；每个合约在官方发布前 7 天、前 1 天各取一个观察点。相邻区间、跨平台和两个时点不增加事件组数。保留所有缺失槽位和排除记录。

| 项目 | 结果 | 解释 |
|---|---:|---|
| 有日期的官方发布 | 27 | 20 份 BLS 标题/首段摘录，7 份 Fed 原始网页 |
| 有事前官方发布事实的观察 | 816 / 816 | 仅选发布时间不晚于观察点的前次发布；BLS 发布版本仍待审核 |
| Polymarket 初始化文本 | 179 / 183 合约 | 初始化日志与当前题目、描述匹配；不代表完整规则变更历史 |
| 通过本地检查的价格 | 480 / 816 | Polymarket 366，Kalshi 114；仅候选价格，尚未正式准入 |
| 有候选价格的事件组 | 22 / 22 | 不是统计独立性或样本功效保证 |
| 官方结果与 API 结果交叉核验 | 816 / 816 | 位于独立结果账本，不构成训练概率答案 |
| 原生 API 精确结算标签 | 450 / 816 | Kalshi 225 个合约 × 2；新增 Polymarket CTF 证明尚未接入 |

旧批次的 10 个 Polymarket 合约已有链上证明、20 条观察通过逐条检查，本轮广范围暂存构建尚未复用该链上证明包，不能把两个批次相加计算。旧 64 条暂存数据、旧逐条准入报告均完成回归核对。

## 两个价格接口问题

Kalshi 的历史端点在 `yes_bid/yes_ask.close` 中返回四位小数的美元字符串，实时端点原解析路径使用 `close_dollars`。修正为按端点显式解析后，恢复 91 个候选价格；不把整数美分猜成美元，继续检查价差、新鲜度和时间边界。依据：[官方历史蜡烛图接口](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks)。

Polymarket Data API 的 30 分钟和 3 小时序列各有 220 个请求为空。将整批改采文档所述长期保留的 3 小时序列没有恢复早期覆盖，不能仅用保留期限解释缺失。另行探测 12 小时序列确有较老记录，但出现零宽度结算点，且粒度无法普遍满足三小时新鲜度要求。所有零宽度点继续排除。依据：[Data API 时间粒度与结算点说明](https://docs.polymarket.com/api-reference/markets/get-a-tokens-price-history)。

官方 [CLOB 历史价格接口](https://docs.polymarket.com/api-reference/markets/get-prices-history) 对早期事件返回非空序列。因此 v4 为全部 183 个 Polymarket 合约统一指定 CLOB 来源，`fidelity=1` 分钟；保留 Data API 两种粒度供诊断，不按结果或逐行可用性混用来源。候选价格从 260 增至 480，事件组覆盖从 16 增至 22，策略修改均发生在模型生成之前。

CLOB 只明确提供样本时间和价格，未给底层成交新鲜度、买卖价差和流动性保证。本地检查排除晚于观察点减 60 秒的样本，按原样本时间计算年龄，拒绝边界价格和重复时间戳；这个 60 秒规则是保守处理，不是声称服务端定义了桶结束时间。全部 CLOB 观察保留 `clob_price_semantics_review_required`，需核查来源语义后才可正式评分。`usable_historical_quotes` 计数仅表示通过这些本地检查。

## 尚未通过的质量门槛

- **历史规则**：450 条 Kalshi 观察没有合约级历史规则版本证明。已归档官方 YOYCPI 合约条款和产品认证 PDF，但当前文件和系列更新时间不能证明当时逐合约规则。Polymarket 366 条观察仍需规则更新历史；其中 56 个有初始化证明的合约使用旧 adapter，不能沿用已审核的新 adapter 源码结论。
- **初始化定位**：4 个 April FOMC 合约的 API 交易定位符指向 `QuestionReset`，不是初始化交易；8 条观察明确保留失败原因，不能拿重置时间或创建时间代替初始化。
- **结算**：新增 Polymarket 范围仍需最终 CTF 兑付及时间证明；API oracle 状态仅用于结果交叉核验。
- **BLS 发布版本**：660 条 CPI / U-3 观察使用有日期的官方存档标题和首段。直接 HTTP 曾返回 403，因此归档的是 web 工具提取文本，不是完整原始网页或首次发布快照；尚未做全页更正审核。只提取当期首段指标，不读取后来修订的历史数据库值。发布时间依据 embargo 头部、星期、URL 日期及美国东部时区校验。
- **报价质量**：336 条 Kalshi 观察缺报价或不满足价格、新鲜度、价差要求，仍在分母中；不填充、不用结算值替代。
- **最终用途审核**：新增范围仍需语义去重审查、冻结实际事件组/时间切分与 checkpoint manifest。全部候选保留 evaluation-only，不能用于提示、奖励或 checkpoint 调参。

`observations.jsonl` 是审计暂存记录，包含质量标志和发布时间核验信息，**不得直接送入模型**。当前模块不导出模型输入或 SFT 目标；`outcomes.jsonl` 单独保存标签与结果核验。下一步先核验 CLOB 样本语义并分批复用/扩展 Polymarket 链上证明，再解决 Kalshi 历史规则和 BLS 发布版本，达标后才能冻结评分集运行对照。

## 归档与复现

生效配置为 `configs/macro_historical_evidence_v4.json`。v1–v3 配置保留用于对应已生成的不可变归档；v3 仅修正 v2 文档链接，v4 新增统一 CLOB 来源，请求上限从 1600 调至 2200，其他候选、时点及新鲜度要求不变。

v4 归档含 1914 个响应引用，全部 HTTP 200，其中 1548 个通过哈希校验后复用原始响应和原始采集时间，新采 366 个 CLOB 响应。接口返回成功不等于有可用证据。另有诊断归档保留错误和失败：首次 12 小时探测误用 `market` 参数返回 400，修正为 `token_id` 后归档到独立 v2 目录；没有把 400 当作数据缺失证据。

```bash
conda activate lab
PYTHONPATH=src python -m foretellmesh.macro_history \
  --config configs/macro_historical_evidence_v4.json \
  --capture data/raw/macro_history_capture_20260919_v4 \
  --output data/processed/macro_history_audit_next

PYTHONPATH=src python -m foretellmesh.macro_history_capture \
  --config configs/macro_historical_evidence_v4.json \
  --reuse data/raw/macro_history_capture_20260919_v4 \
  --output data/raw/macro_history_capture_next

PYTHONPATH=src python -m unittest discover -s tests -v
```

命令依赖本机哈希绑定的原始归档；输出目录必须尚不存在。原始数据和运行记录按仓库策略不提交 Git。主产物：`data/processed/macro_history_audit_20260919_v4`；验证记录：`runs/macro_historical_evidence_v1`。完整测试 326 项，325 通过，1 项可选 GPU 测试跳过。新增 20 项测试覆盖官方发布时间/指标口径、历史价格字段、CLOB 边界、结果隔离、请求覆盖、归档篡改和复用。新产物逐字节离线重建一致；旧暂存产物不变，旧报告仅代码来源标记变化。
