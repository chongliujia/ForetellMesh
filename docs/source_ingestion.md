# 真实数据接入与来源审计

核查日期：2026-09-19。固定版本和文件哈希见 [data_sources_v1.json](../configs/data_sources_v1.json)。原始数据下载到被 Git 忽略的 `data/raw/`，不将第三方数据内容合入本项目源码。

## 来源与当前接入范围

| 项目中的名称 | 核查来源 | 固定 revision | 当前能力 |
| --- | --- | --- | --- |
| Prophet Arena Subset 1200 | [作者 Hugging Face 仓库](https://huggingface.co/datasets/prophetarena/Prophet-Arena-Subset-1200) | `c94b6f450d7fe3b03688799cce1c8b29838b5d96` | 下载 CSV；逐合约展开；保留父事件和 submission 身份；审计并按时间证明转换 |
| ForecastBench | [官方数据入口](https://www.forecastbench.org/datasets/)、[官方仓库](https://github.com/forecastingresearch/forecastbench-datasets)的 Hugging Face 镜像 | `a11ac3a9ba8812cdedab2b79ab3181a9c0825d62` | 固定 2026-01-04 轮次的问题/结算文件；按来源、问题、期限匹配；始终仅供评估 |
| Prediction Market Analysis | [Jonathan Becker 原仓库](https://github.com/Jon-Becker/prediction-market-analysis) | `2276382cb616107db8c8647803bffa4a0d7091f8` | 固定 schema 与许可证；本地 Kalshi 快照适配；独立的 Polymarket 全归档目录与成交/区块关联流程 |

前两个 revision 标识数据仓库版本。PMA revision **只标识上游代码与 schema**，不是外部托管数据归档的版本；本地导入分片使用 `sha256:<文件内容哈希>` 作为 dataset_version。

上游声明：Prophet 数据卡标注 MIT，ForecastBench 数据为 CC-BY-SA-4.0，PMA 仓库为 MIT。`fetch` 保留相应数据卡/许可证；数据内容与本项目的 Apache-2.0 代码许可证分别记录。这里记录上游声明，不将其解释为所有底层新闻、统计数据或市场内容都具有相同授权。

PMA 上游 README 提供完整压缩数据归档；本次固定 HTTP 对象实测为 36,020,641,508 字节（约 33.55 GiB）。[历史库流程](pma_historical_archive.md)单独管理下载、完整压缩帧验证、逐分片哈希、原生 Polymarket 目录及历史成交重建。原有 `fetch --source prediction_market_analysis` 仍只获取 schema 和许可证；不能用该命令的成功状态代替完整数据接入。未执行 `make setup` 或上游 Python 代码。

## 实测结果

下表来自固定文件的本地解析与统计，不使用数据卡数量代替实际结果：

| 数据 | 原始记录 | 展开后的候选 | 当前可进入评估的记录 |
| --- | ---: | ---: | ---: |
| Prophet Arena CSV | 1,200 submissions | 7,022 二元合约观测 | 0 |
| ForecastBench 2026-01-04 | 500 问题，另有 1,176 条结算文件记录 | 2,244 问题/期限 | 默认导入为 0；专用证明流程已验证 6 条 |

默认 `ingest` 不凭空填写缺失时间，因此不提供时间证明时可评估记录为零。后续 [专用验证流程](verified_subset.md) 已通过原始发布提交与平台结算字段验证 6 条 ForecastBench 样本，运行了保留集确定性基线；未训练模型，也不据此声称预测能力提升。

### Prophet Arena

实测 CSV 有 869 个不同 `event_ticker`；上游数据卡写 897 个 unique events，两者定义或版本可能不同，因此保留原始 ID 与实际统计，不擅自修改身份。

- 实际文件包含 `snapshot_time`，没有数据卡表格中提到的 `submission_created_at`。
- `markets` 列为 Python 列表字面量，其余嵌套列为 JSON。适配器只用 `ast.literal_eval` 解析列表，不使用 `eval`。
- 7,022 个合约中，6,380 个有 bid/ask 数据，642 个无可用市场快照。初版采用 `(yes_bid + yes_ask) / 200` 转为概率；这是以美分表示的报价中点，并非成交价格。
- 12,004 条 source entries 都没有独立的发布时间/版本可用时间。原始文件保留这些内容，导出的模型输入不使用其摘要。
- 不将 `close_time` 当作 `resolution_time`，也不将数据下载时间当作当年的标签可用时间。
- `augmented_title` 为上游生成字段，初版只使用原始 title 与合约名称。未建立历史版本证明的 rules 和增强标题不进入输入。
- 894/1,200 submissions 属于 Sports（74.5%）。这份子集不能直接被描述为金融专用训练集。

### ForecastBench

- 数据类问题按 `resolution_dates` 展开，并替换问题中的预测日和结算日占位符；event_id 包含预测轮次和期限，避免把不同目标误认为同一事件。
- 相关期限仍建议放在同一事件组。相同 id 在不同来源下必须分别匹配，禁止仅以 id 连接。
- 实测 2,244 个候选中，1,016 个具有 `resolved=true` 的二元结果，另外 1,228 个尚未结算或没有对应结算记录。
- `resolved=false` 时的 `resolved_to` 可能是参考概率，甚至恰好为 0 或 1；一律不视为 outcome。
- 只有 Polymarket / Manifold 的冻结概率进入市场候选。Metaculus / INFER 的群众预测，以及 FRED、股价等时间序列值，不混入市场概率基线。
- `forecast_due_date` 和 `resolution_date` 只有日期，不能据此编造精确时刻。时间证明要显式给出观测时刻、标签结算与可用时刻；当前适配约定这些日期按 UTC 校验。
- 固定今天可下载的 revision 只能证明文件可复现，不能单独证明其中每段文本在历史预测时点已公开。因此问题/规则的历史内容版本仍需证据。background 没有单独时间证明时不进入输入。

这些适配规则不是官方 ForecastBench 完整评分流程的复刻；特别是未结算参考概率和非二元标签的处理不同。当前只服务于本项目严格二元、已结算事件的内部评估管线。

### PMA Kalshi 子集

对应 [上游 schema](https://github.com/Jon-Becker/prediction-market-analysis/blob/2276382cb616107db8c8647803bffa4a0d7091f8/docs/SCHEMAS.md)。使用方法：

```bash
python -m pip install -e '.[parquet]'
PYTHONPATH=src python -m foretellmesh ingest \
  --source prediction_market_analysis \
  --data /path/to/kalshi_market_snapshots.parquet \
  --output runs/pma_kalshi_source_audit
```

一份分片内须同时有事件的早期 open 快照与稍后 finalized 结果快照，才能产生可进一步审核的有标签候选。观测时点采用早期快照 `_fetched_at`，不是把后来抓取的 metadata 倒填到过去。后来的结果只进入标签，标签可用时间不得早于本文件中相应 finalized 快照的时间。实际结算时间仍要单独证明。

仅含最终状态的市场表会被隔离。无时区时间戳直接报错。该 Kalshi 单分片适配器用合成 Parquet 测试；读取以 batch 进行，但仍在内存中按 ticker 汇总，不适合把完整归档合成一个巨型文件。原生 Polymarket 的 token/outcome 映射、区块时间连接及多分片历史成交重建由独立的 `foretellmesh.pma_*` 模块实现，见[命令与实测报告](pma_historical_archive.md)。

## 导入产物与时间证明

每次 `ingest` 使用全新输出目录，生成：

| 文件 | 用途 |
| --- | --- |
| `audit.json` / `audit.md` | 统计、阻塞原因、输入/输出哈希、来源角色和代码版本 |
| `source.json` | 本次固定来源配置 |
| `candidates.jsonl` | 原始身份、候选问题/市场/结果、源文件定位和风险说明；不允许直接用于模型 |
| `disposition.json` | 每个候选的 accepted / quarantined / excluded 状态 |
| `annotations.template.json` | 绑定原始输入哈希的逐样本时间证明模板，初始全部 pending |
| `records.jsonl` | 仅含通过全部检查的统一格式记录；可能为空 |
| `annotations.applied.json` | 若提供时间证明文件，保留其完整副本 |

时间证明可以来自核验过的源元数据或后续专用历史检索流程，不要求用户逐条手工填写；但工具不会凭空生成缺失事实。`provenance` 应引用历史快照、带版本的事件映射清单、结算公告或标签首次可用记录，不能只写“已审核”。同一 native 父事件可并入更大的关联组，不能拆成互相独立的小组。

模板中每条 entry 的字段：

```json
{
  "decision": "pending",
  "event_group_id": "source-native-group-to-review",
  "observation_time": null,
  "question_available_at": null,
  "resolution_time": null,
  "label_available_at": null,
  "provenance": null
}
```

- `pending`：隔离；缺省 entry 同样隔离。
- `exclude`：主动剔除，`provenance` 记录原因。
- `include`：必须补齐以上字段，确认问题及所用规则的内容版本在观测时点可获取，分组已核验，结算与标签可用时间有来源。

运行示例，路径指向已建立证明的本地文件：

```bash
PYTHONPATH=src python -m foretellmesh ingest \
  --source prophet_arena \
  --data data/raw/prophet_arena_v1/subset_data_1200.csv \
  --annotations data/processed/prophet_annotations_v1.json \
  --output runs/prophet_reviewed_v1
```

输入哈希必须完全匹配；不能给未知样本加标签，不能移动源文件明确给出的观测时间，不能通过 include 绕过未结算、非二元或冲突标签。时间证明只提供元数据，不替换源 outcome。源文件的实际 outcome 仍与模型输入分离。

**当前验证能检查时间大小关系和元数据一致性，不能自动证明注释内容真实。** 即使 accepted，也需要将跨来源语义关联清单一起审查后再冻结正式 split；数据加载器继续执行一致性、去重和时间切分检查。

## 下一步的数据工作

首批 6 条真实样本已建立历史问题版本证明、精确结算时间与局部事件组映射，完整命令见 [复现说明](verified_subset.md)。接下来扩大可验证样本，并单独整理训练数据；小集合用于链路检查，正式评估配置与保留策略仍需冻结。基准数据不会因为缺少训练样本而被移动到训练集。
