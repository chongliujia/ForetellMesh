# 已结算预测市场的历史回放

历史已结算事件是主要训练数据建设路线。Polymarket 与 Kalshi 的未结算快照继续保留，用于未来记录真实预测、回填结果和检查泛化；不再等待这些事件结算才启动历史数据建设。独立合成预热集和 ForecastBench 保留集的用途不变。

训练路线已明确为 [多 Agent、能力 LoRA 与分阶段 RL](capability_agents.md)：真实事件 SFT 可选。下文“教师答案缺失”只属于 SFT 准入缺口，不是独立预测评分或 RL 必须补齐的条件；这些用途仍须通过历史输入、标签和切分审核。既有暂存报告汇总了不同用途的阻塞项，不能把它的全部阻塞项直接当作统一的 RL 准入规则。

当前完成的是一个可离线复现的历史数据试点，尚未发布真实 SFT 训练集。最终结果可以验证事件标签，但不能当作观察时点的“正确预测概率”，也不能自动生成可信的事前分析。

## 首批实际数据

配置：[historical_fomc_collection_v1.json](../configs/historical_fomc_collection_v1.json)。预先固定两次 FOMC 会议，对每个返回的二元合约在官方声明前 7 天和前 1 天各取一次观察，不按最终 Yes/No 筛选合约。

| 事件组 | 官方声明时间 UTC | Polymarket 父事件 | Kalshi 父事件 |
| --- | --- | --- | --- |
| `macro:us:fomc:2026-07` | 2026-07-29 18:00 | `287395` | `KXFED-26JUL` |
| `macro:us:fomc:2026-09` | 2026-09-16 18:00 | `481717` | `KXFED-26SEP` |

2026-09-18 已归档 97 次公开 GET，全部成功。原始响应、URL、请求时间、HTTP Date、SHA-256 和采集配置都保存在不可覆盖的版本目录。

| 项目 | Polymarket | Kalshi | 合计 |
| --- | ---: | ---: | ---: |
| 已结算市场合约 | 10 | 22 | 32 |
| 历史观察样本 | 20 | 44 | 64 |
| 与官方声明交叉核验的结果 | 20 | 44 | 64 |
| 同时具备精确结算时间的标签 | 0 | 44 | 44 |
| 通过当前历史报价门槛的观察 | 20 | 5 | 25 |
| 有初始化原文证明的观察 | 20 | 0 | 20 |
| 可直接用于真实 SFT | 0 | 0 | 0 |

64 条观察只有 **2 个底层事件组**。同一事件下的不同阈值、两个平台的合约以及两个观察时点高度相关，不能当作 64 个独立预测事件，也不足以建立有说服力的训练/验证/测试划分。

两个唯一的事前证据文档是美联储 [2026-06-17 声明](https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm)和 [2026-07-29 声明](https://www.federalreserve.gov/newsevents/pressreleases/monetary20260729a.htm)。后者只作为九月事件的事前证据；对七月事件，它属于结果核验材料。九月事件的结果单独依据 [2026-09-16 声明](https://www.federalreserve.gov/newsevents/pressreleases/monetary20260916a.htm)。这只是证据链的最小验证，尚未覆盖就业、通胀、政策讲话等完整信息集。

## 防止事后信息进入输入

每条记录具有固定观察时点 `T`。导出的预测输入只包含问题、`T`、事前证据和合格的市场价格。结果、结算时间、抓取到的当前状态和核验记录分开保存。

1. **证据时间**：核对美联储历史页面的发布日期、发布时间、时区和 URL 日期；页面若标记更晚修订则拒绝。只提取声明正文，排除导航和当前关联文章；只允许发布时间不晚于 `T` 的证据。实际抓取时间仍按真实时间记录，不回拨到过去。这里依赖官方历史页面的发布声明，不声称拥有当时本地保存的完整网页快照。
2. **问题原文**：Polymarket 使用对应公开交易回执中的 `QuestionInitialized` 日志，核对交易、成功状态、适配器、请求 ID、区块时间和 ABI 解码后的原始问题。初始化必须早于 `T`，当前问题与规则必须匹配初始化原文。这是区块浏览器归档证明，尚未通过独立 RPC 核验。
3. **规则变更**：初始化日志只证明最初版本。该批 Polymarket 状态带有 `new_version_q`，后续补充说明是否在 `T` 前生效尚未核验，所以仍禁止进入训练集。Kalshi 当前规则及公开规则文件不能证明完整条款在历史 `T` 时的版本，44 条观察均保留此阻塞项。
4. **报价时间**：Polymarket 聚合价格的时间戳是时间桶起点，必须满足 `timestamp + resolution_seconds <= T`。历史接口的零宽度点可能混合真实逐笔价格与合成结算点，无法可靠区分时全部排除；本批排除了 25 个零宽度点，不把它们全部声称为泄漏数据。分页未完成时不接受报价。接口语义见 [官方历史价格文档](https://docs.polymarket.com/api-reference/markets/get-a-tokens-price-history)。
5. **报价质量**：只查 `T` 前 6 小时，价格最旧不得超过 3 小时。Polymarket 请求 30 分钟粒度；Kalshi 用小时蜡烛的收盘 bid/ask 中点，蜡烛结束时间不得晚于 `T`，价差不得超过 0.10。边界、单边、缺失或过期报价置空，不补 0.5。历史聚合报价并未通过实时订单簿深度审核，不能用于声称当时可成交的回测。
6. **结果身份**：Polymarket 核对 Gamma/CLOB 有序 outcome/token 映射和 condition ID；Kalshi 核对事件、二元合约、阈值、原生结果及结算字段。结果按精确分数运算与官方利率变化/水平核验；冲突、不完整、争议和非二元结算不作为合格标签。

## 为什么部分已知结果仍没有正式标签

本批 Polymarket 的旧 UMA 结算记录可以交叉核验最终结果，但尚未提供满足本项目标签契约的精确结算时间证明。记录中的 `transaction_hash` 在本批对应初始化交易，不能误当作结算交易；`last_update_timestamp` 也不替代结算时间。

因此 `outcomes.jsonl` 保留已核验结果及证据状态，统一记录中的 `label` 仍为 `null`。只有完整二元 payout、确切结算时间、区块及来源一致时才生成正式标签。字段语义对照 [官方结算状态接口](https://docs.polymarket.com/api-reference/markets/get-resolution-state)。Kalshi 的 44 条观察有精确原生结算时间，但问题历史版本与其他质量条件仍未达标。

## 产物与复现

已保存：

- `data/raw/historical_fomc_capture_20260918_v1/`：97 次原始请求及 `manifest.json`。
- `data/processed/historical_fomc_replay_20260918_v2/`：最终代码生成的历史暂存数据；此前 v1 保留。
- `runs/historical_fomc_verification_20260918_v1/`：离线逐文件复现结果及测试记录。

验证结果：7 个派生文件离线重建后逐字节一致；`conda lab` 全套 149 项测试中 148 项通过，1 项可选 GPU 测试跳过。历史模块的 16 项测试覆盖价格时间桶与零宽度点、结果/输入隔离、来源修订、结算冲突、归档篡改、保留集精确匹配和回放确定性。

派生目录内容：

| 文件 | 用途 |
| --- | --- |
| `candidates.jsonl` | 含嵌套候选记录、原始问题、来源证明、报价质量和阻塞项；不是训练入口 |
| `outcomes.jsonl` | 独立结果账本，区分已核验结果与精确结算标签 |
| `review_inputs.jsonl` | 20 条有初始化证明、事前证据和合格报价的审阅输入；不含结果，仍待规则变更核验 |
| `evidence.jsonl` | 历史声明正文、发布日期及原始请求引用 |
| `exclusions.json` | 解析、身份和来源失败的排除记录 |
| `report.json` | 数量、阻塞项、限制、数据版本、配置/代码/产物哈希 |
| `plan.json` | 此次回放使用的固定配置 |

使用 `conda lab`；以下离线重建不需要网络。输出必须是新目录，命令不会覆盖既有版本：

```bash
conda activate lab
PYTHONPATH=src python -m foretellmesh build-historical-markets \
  --capture data/raw/historical_fomc_capture_20260918_v1 \
  --heldout-index data/processed/forecastbench_sft_exclusion_index_v1.json \
  --output data/processed/historical_fomc_replay_next
```

重新抓取会生成独立的原始版本；上游内容或请求时间改变时，哈希改变是预期行为：

```bash
PYTHONPATH=src python -m foretellmesh capture-historical-markets \
  --config configs/historical_fomc_collection_v1.json \
  --output data/raw/historical_fomc_capture_next

PYTHONPATH=src python -m unittest discover -s tests -q
```

原始归档与派生数据不进入 Git，配置、处理代码和测试进入版本管理。离线重建需要保留完整原始归档和固定 ForecastBench 排除索引。改变处理代码后，报告的代码哈希随之变化。

## 支持范围及训练发布条件

当前是 **FOMC 试点**：只支持本配置使用的标准利率变化与阈值合约，不能直接套用于所有体育、选举或加密市场。市场发现要求原生父事件响应仍包含合约；更旧、仅存在历史归档的事件会明确排除，尚未实现历史市场列表的完整发现和合并。Kalshi 价格查询已依据官方历史切点选择接口，见 [历史数据路由](https://docs.kalshi.com/getting_started/historical_data)。

以下工作完成前，暂存工具固定输出 `ready_for_sft=false` 和 `ready_for_benchmark=false`：

- 补齐合约在各观察时点的有效规则与修订历史；为 Polymarket 补确切结算证明。缺报价的记录不能进入市场相对评估，若另作无市场输入任务，需制定独立准入策略。
- 扩充同一时点以前可验证的证据。数值宏观数据须保存当时发布版本，不能把后来修订的历史序列直接作为事前特征。
- 完成 ForecastBench 语义重叠审查，并跨平台合并同一事件组。当前索引精确匹配只是筛查，不能证明语义无重叠。
- 扩展更多独立历史事件后，按时间与事件组固定训练、验证和测试边界；不随机拆分当前 64 条观察，不用这两个试点事件反复调参后再报告最终测试结果。
- 另行构建仅能读取已审核预测输入的教师任务，记录生成模型、生成时间、证据引用和审核结果。事后生成的回放回答必须使用明确的回溯类型，不能伪装成当时发布的预测。现有 SFT 导入器尚不接受这种教师类型，本次未生成教师答案。

结果标签可用于独立评分和后续 proper-scoring reward；不能把最终 Yes/No 写成 SFT 的 1/0 概率答案，也不能直接把市场价格当成真实预测概率答案。隐藏结果字段仍无法排除基础模型或教师预训练时记住历史结果的风险，所以前瞻预测留档仍是必要补充。

本次未训练新 LoRA、未生成模型预测或 Brier/Log Loss/ECE 报告，也不据此声称预测能力提升。
