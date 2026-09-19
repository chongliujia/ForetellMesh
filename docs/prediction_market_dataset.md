# 自建预测市场数据集 v1

目标：给定时点 T 的市场合约、规则、报价及可验证证据，预测合约最终兑现的概率，并与同一时点的市场报价比较。用户已明确允许自建数据集；数据选型围绕 Kalshi / Polymarket 的实际事件合约。

当前已接入 Kalshi 和 Polymarket 的公开只读 API，建立原始快照、后续结算回填和跨平台质量审计流程。**主要训练数据建设已转向历史已结算事件**，首批 32 个合约、64 条历史观察见 [历史回放说明](historical_prediction_markets.md)；本页记录的前瞻快照继续用于未来检验。外部新闻检索和 SFT 教师答案尚未接入。合成格式预热数据独立保存，见 [SFT 数据说明](sft_warmup.md)。

## 首次实际采集

配置：[kalshi_collection_v1.json](../configs/kalshi_collection_v1.json)。2026-09-18 06:39:08–06:39:19 UTC 完成 9 次公开 GET 请求；每个系列选择 90 天范围内预定关盘时间最近的一个父事件，不按事后结果筛选。

| 父事件 | 内容 | 合约数 | API 声明的预定关盘时间（UTC） |
| --- | --- | ---: | --- |
| `KXCPIYOY-26SEP` | 美国 2026 年 9 月 CPI 同比 | 21 | 2026-10-14 12:29 |
| `KXFED-26OCT` | 2026 年 10 月议息会议后的联邦基金利率 | 11 | 2026-10-28 17:55 |
| `KXUE-AUS26AUG` | 澳大利亚 8 月失业率 | 10 | 2026-09-24 01:25 |

共 42 条未结算观测、3 个父事件，23 条有可用双边报价。其余报价设为 `null`，不填 0.5。预定关盘时间不是结算时间，也不证明统计结果尚未发布；正式发布数据集前要核对统计机构发布时间及提前关盘情况。此次采集是市场数据层的起点，不是可直接训练或报告预测能力的完整数据集。

本地文件：

- `data/raw/kalshi_capture_20260918_v1/`：原始 API 响应、请求 URL、开始/完成时间、HTTP Date、SHA-256，以及可重建的观测预览。
- `data/raw/kalshi_labels_20260918_v1/`：针对同一批合约单独保存的后续状态轮询。
- `data/processed/foretellmesh_kalshi_20260918_v1/`：统一 `records.jsonl`、独立审计 metadata、状态清单和 manifest。

这些数据目录不进入 Git；采集配置、处理代码和测试纳入版本管理。保存原始目录才能离线复现派生数据。

## 运行方式

在仓库根目录使用 `conda lab`；所有输出必须是新目录。没有安装定时任务或后台采集服务。

```bash
conda activate lab

# 新建观测快照，时间由实际响应完成时刻确定，不能通过参数回拨。
PYTHONPATH=src python -m foretellmesh collect-markets \
  --config configs/kalshi_collection_v1.json \
  --output data/raw/kalshi_capture_20260918_v1

# 在后续日期重跑，换一个输出目录；读取原观测中的同一组 ticker。
PYTHONPATH=src python -m foretellmesh poll-market-labels \
  --capture data/raw/kalshi_capture_20260918_v1 \
  --output data/raw/kalshi_labels_20260918_v1

# 离线构建：可以给多个轮询目录；尚无轮询时省略 --labels。
PYTHONPATH=src python -m foretellmesh build-market-dataset \
  --capture data/raw/kalshi_capture_20260918_v1 \
  --labels data/raw/kalshi_labels_20260918_v1 \
  --output data/processed/foretellmesh_kalshi_20260918_v1
```

采集使用 [Kalshi 官方公开市场接口](https://docs.kalshi.com/getting_started/quick_start_market_data)；[市场对象](https://docs.kalshi.com/api-reference/market/get-market)提供规则、美元报价、状态与结算字段。不需要账户凭据，也不调用下单、资金或账户接口。分页与数量上限显式写入配置；达到分页上限的系列整体跳过，避免把不完整事件组误称完整采集。这里的事件范围仅指配置内、当前开放的合约，不声称包含父事件所有历史合约。

## 时间与标签约束

1. `observation_time` 取市场响应和父事件响应都已收到的时刻。市场报价有自己的观测时间；原始规则在该观测时点归档，不倒推其首次发布时间。
2. 仅接受尚未给出结果的开放二元、每份兑现 1 美元的合约，并检查开盘/关盘区间。规则保存在问题上下文；外部 `evidence` 当前为空。
3. 使用美元字符串解析报价，不与旧版美分字段混用。只有 `0 < bid <= ask < 1` 的报价中点进入基线；0/1 边界可能表示缺少一侧，因此初版保守地留空。
4. 轮询另存原始响应。只有最终状态、明确 yes/no、明确 `settlement_ts` 且 `T < settlement_ts <= 本次收到标签的时间` 时才追加标签。标签可用时间采用本地实际收到它的保守时点，不声称它是全网首次可用时间。
5. 规则/合约含义改变、非二元结算、冲突结果或缺失精确结算时间均保留为待核验，不替换成关盘时间。后来的内容不会进入原先的模型输入。
6. 父事件下的阈值合约保留相同原生 `event_group_id`。下文的质量审计另行导出跨平台分组的暂存记录；更广泛的语义关联和 ForecastBench 重叠仍需审核，不创建随机训练/测试切分。

## Polymarket 首批采集与回填

配置：[polymarket_collection_v1.json](../configs/polymarket_collection_v1.json)。2026-09-18 07:08:02–07:08:24 UTC，58 次公开 GET 归档了两个事先选定的 Gamma 父事件：

| Gamma 父事件 ID | 合约内容 | 合约数 |
| --- | --- | ---: |
| `606422` | 2026 年 10 月美联储利率变化幅度 | 5 |
| `1005014` | 美国 2026 年 9 月 CPI 环比 | 9 |

14 条观测均通过身份和时间校验，无排除项；全部获得通过当前门槛的双边报价。实际订单簿距观测时点为 0.77–41.16 秒，价差为 0.001–0.100，每侧最优价挂单量的最小值为 13.3 份合约。报价合格仅表示适合作为这一时点的市场中点基线，不保证预测正确或交易可执行。这是一个小型宏观事件试点，尚未覆盖平台全部主题。

数据链路使用 [Gamma 市场详情](https://docs.polymarket.com/market-data/market-details)、[CLOB 订单簿](https://docs.polymarket.com/api-reference/market-data/get-order-book)与 [Data API 结算状态](https://docs.polymarket.com/api-reference/markets/get-resolution-state)。归档 URL、请求时间、HTTP Date、响应原文及 SHA-256；本地时间由实际请求生成，不能通过配置回拨。

- **身份一致性**：核对父事件、market ID、condition ID，以及 Gamma/CLOB 的有序 outcome/token 映射。按标签寻找 YES token；不假定列表首项总是 YES，也不将价格数组当作结算结果。订单簿必须匹配同一 condition 和 YES token。
- **报价门槛**：取最大 bid、最小 ask，不依赖返回数组排序。允许报价年龄至多 120 秒、单合约采集窗口至多 180 秒、价差至多 0.10、每侧最优价至少 5 份合约。边界价、单边空簿、过期、价差过宽或深度不足时保留合约、报价设为 `null`；交叉盘口、非有限价格、未来时间、身份冲突则隔离观测并记录原因。阈值是首版数据策略，可以版本化调整。
- **观测时状态**：要求开放且可交易，结算状态只能处于尚无提案的已识别阶段；已有结算提案的合约不进入这批前瞻观测。事件关盘时间仍不能证明现实答案未知，必须另查官方发布日历。
- **结果回填**：单独轮询当前规则和 condition 结算状态。只接受明确 `resolved`、无未核验争议、稳定规则/结算上下文、二元兑现 `[1000000, 0]` 或其反向、明确 `resolved_at` 和区块来源，且 `T < resolved_at <= 本次收到标签时间`。按已核对的 outcome 索引读取 YES 兑现结果。多次轮询相互冲突、结算后重开或证明消失都会隔离标签。
- **证明不足时留空**：部分 UMA 旧格式响应只有结算价格与更新时间，缺少兑现向量或精确结算时间。当前实现会保留待核验证据，不把 `closedTime`、预定结束日期或最后更新时间冒充结算时间。补齐这类样本需要后续链上证明适配或独立核验。负风险市场的 UMA question ID 可能不同于 Gamma question ID；按 condition 绑定，并保留两个来源，不强行视为相同。
- **规则澄清边界**：API 的 `new_version_q`、争议和负风险上下文保留在审计 metadata。尚未独立归档链上规则公告板，故完整结算上下文审查仍是发布门槛。

本地原始目录为 `data/raw/polymarket_capture_20260918_v1/` 和 `data/raw/polymarket_labels_20260918_v1/`。经补强 outcome 顺序校验后的派生目录为 `data/processed/foretellmesh_polymarket_20260918_v2/`；早期 v1 派生目录保留，不覆盖。初次轮询 14 条均为 `pending_resolution`，实际尚未验证已结算市场的在线回填；完整兑现、反向 outcome、争议和冲突路径已由固定测试样例验证。

```bash
conda activate lab
PYTHONPATH=src python -m foretellmesh collect-polymarket \
  --config configs/polymarket_collection_v1.json \
  --output data/raw/polymarket_capture_next

PYTHONPATH=src python -m foretellmesh poll-polymarket-labels \
  --capture data/raw/polymarket_capture_20260918_v1 \
  --output data/raw/polymarket_labels_next

PYTHONPATH=src python -m foretellmesh build-polymarket-dataset \
  --capture data/raw/polymarket_capture_20260918_v1 \
  --labels data/raw/polymarket_labels_20260918_v1 data/raw/polymarket_labels_next \
  --output data/processed/foretellmesh_polymarket_next
```

网络错误会中止本次原子归档，不发布半份成功结果；接口恢复后使用新目录重新采集。当前没有后台轮询任务。

## 跨平台质量报告

运行审计前验证 bundle 文件哈希、记录 schema、时间关系、元数据与标签状态、重复身份及标签一致性。分组方案 [market_event_groups_v1.json](../configs/market_event_groups_v1.json) 将同一次美联储会议的“利率水平/利率变化”、同次 CPI 发布的“同比/环比”各自归为一个拆分组。原始合约 ID、问题及 outcome 保持独立；同组不表示两个合约等价，也不强制它们结果相同。

```bash
PYTHONPATH=src python -m foretellmesh audit-market-quality \
  --datasets data/processed/foretellmesh_kalshi_20260918_v1 \
             data/processed/foretellmesh_polymarket_20260918_v2 \
  --groups configs/market_event_groups_v1.json \
  --heldout-index data/processed/forecastbench_sft_exclusion_index_v1.json \
  --output runs/prediction_market_quality_next
```

已生成 `runs/prediction_market_quality_20260918_v2/`：

| 指标 | 数量 |
| --- | ---: |
| Kalshi / Polymarket 观测 | 42 / 14 |
| 平台原生父事件 / 底层拆分组 | 5 / 3 |
| 非空市场中点报价 | 37 |
| 已通过新鲜度、价差、深度门槛的报价 | 14（Polymarket） |
| 已核验最终 outcome | 0 |
| 带外部时点证据的观测 | 0 |
| 可直接用于真实 SFT / 正式 benchmark | 0 / 0 |

Kalshi 旧版采集的 23 条中点报价没有独立订单簿深度和报价生成时间证明，不因合并数据而被自动升级；另有 19 条报价为空。56 条合约观测只有 3 个底层事件组，不能按 56 个独立事件估计统计可信度。

`report.json` 保存计数、输入与配置哈希、代码版本；`review_queue.jsonl` 列出每条记录的阻塞项及 ForecastBench 词面近似候选；`grouped_staging_records.jsonl` 是使用统一事件组的暂存数据。审计对照固定的 2,244 条 ForecastBench 索引，目前 ID/完整问题精确命中数为 0；词面相似搜索不是语义去重证明，不会自动批准训练。所有事件仍须完成 benchmark 语义重叠审查。

该命令是审计和暂存工具，不是训练数据发布器：即使之后补上 outcome，也不会仅据此把 `ready_for_sft` 改为 true。外部证据、现实结果首次公开时间、教师回答、按时间和事件组切分仍需逐项验证。结果标签只用于评分或适当训练奖励，不作为“正确概率”的 SFT 回答。

离线验证已确认：Polymarket 派生文件和质量审计文件均可逐字节重建；原 Kalshi records/metadata/disposition 重建后逐字节一致。记录见同目录 `replay_verification.json`。不复用后来收到的规则或新闻改写原始模型输入。

## 数据如何变成训练材料

- **格式 SFT**：独立的合成预热集，学习 JSON、引用已有证据 ID、读取工具计算结果和表达不确定性。
- **真实预测任务**：市场快照加截至 T 的证据，结算后用真实 outcome 评分。市场报价是输入/基线，outcome 是监督标签，不把任一单次 outcome 写成“正确预测概率”。
- **真实 SFT**：需要另行构建、审核且显式标记的教师回答。回溯生成回答时必须隐藏结果，记录教师模型、生成时间及其潜在历史知识污染；不能把事后生成的文本称为当时实际发布的预测。
- **后续 RL**：在格式与工具能力达标后，使用已冻结训练事件的结算结果计算 proper scoring reward；不从 raw Base 直接开始。

该前瞻批次仍为 `ready_for_sft=false`、`ready_for_benchmark=false`。不再等待其结算才建设训练数据：历史已结算事件已开始独立回放和结果核验。前瞻批次后续补官方发布时间、预测前证据与实际模型预测留档，结算后检查回填。两条流程都须审核规则、教师答案及时间/事件组切分，不能直接用于声称预测能力提升。

## 现有来源的角色

- Prophet Arena 保留已完成的核验流程；缺少 SFT 答案并不等于市场数据无用。
- Prediction Market Analysis 作为批量历史市场与成交数据来源；原先只有 schema 与 Kalshi 快照适配，现增加[原生 Polymarket 历史库流程](pma_historical_archive.md)。成交与区块时间用于重建事前价格；最终快照不直接生成历史证据或预测概率答案。
- ForecastBench 保持仅评估用途。
- 已实读 `LightningRodLabs/outcome-rl-test-dataset` 固定版本 `f200d538760aa94848f2bc803d5307f5552fa73f`：1,265 条 Polymarket 预测任务，带 prompt、结果及模型预测。其 [论文](https://arxiv.org/html/2505.17989v4)将它定义为测试集；即使 Hugging Face 分片名为 `train`，也不自动转为本项目训练数据。新闻摘要、教师预测的历史时间证明还需审查，目前仅存来源研究目录。

Autocast/GJP 不作为这次预测市场训练数据的替代路线。SWM-Bench 的核心标签是未来市场价格，与本项目当前的最终事件兑现概率任务不同，未接入。
