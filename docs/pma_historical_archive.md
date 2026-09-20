# Prediction Market Analysis 历史库接入

本流程直接使用 Jonathan Becker 的 [Prediction Market Analysis](https://github.com/Jon-Becker/prediction-market-analysis) 公开历史归档，先批量读取市场、成交和区块数据，再用原生 API 与已有证明模块补缺项。上游程序不在本机执行。

后续[历史补证试跑](pma_proof_pilot.md)已扩为[宏观研究数据集](pma_macro_research_dataset.md)：159 条观察覆盖 91 个合约、22 次不同发布，训练/验证/保留测试分区已冻结。保留测试仍只有 3 次 FOMC，正式双平台评分尚未准入。下文统计仍对应历史库接入和价格重建阶段。

## 来源与版本

- Schema / 上游代码 revision：`2276382cb616107db8c8647803bffa4a0d7091f8`。
- 归档 URL：`https://s3.jbecker.dev/data.tar.zst`；HTTP 对象大小 **36,020,641,508 字节**，约 33.55 GiB。
- 下载配置：`configs/pma_archive_v1.json`，固定大小及 ETag；数据版本使用完整归档的 SHA-256，不能用代码 revision 代替。
- 本次完整归档 SHA-256：`0be77ff1eae2e8c0fa962bbb1fdf7c26522a7bf19cb627cfb19d26388b71a920`。这是本地完整文件校验值，并非上游数字签名。
- 上游对象 Last-Modified 为 2026-02-05，这不是事件覆盖截止日；实际覆盖以原生行和区块时间统计为准。
- 上游仓库许可证声明为 MIT；这不等于每项底层内容均获得相同授权。

下载保留原始压缩包、分段范围、哈希及传输失败记录。断点必须返回准确的 `Content-Range` 和匹配 ETag；错误、截断及缺失段不能进入完整归档。经逐段核验合并后计算全文件 SHA-256，解包时在同一次读取中重新核验归档 SHA-256，并用 `zstd` 命令验证完整压缩帧；校验全部通过后才发布解包目录。Python 流式解压本身可能接受缺失的尾部校验字节，已用回归测试覆盖。

完整解包实测选择 **41,280 个原生 Parquet 分片**：41 个市场分片、40,454 个成交分片和 785 个区块分片，共 49,186,627,297 字节（约 45.8 GiB）。归档成员目录还记录 Apple 元数据及未选中的数据表，不能把所有 tar 成员数当作有效 Parquet 分片数。

## 读取和关联

1. **解包**：仅写入 `polymarket/markets`、`polymarket/trades`、`polymarket/blocks` 的 Parquet，保留全归档成员目录。跳过 Apple 元数据文件及其他数据表；拒绝路径穿越、软链接、硬链接、重复成员和超限文件。完整压缩包仍包含上游 Kalshi 与旧 FPMM 数据，后续可另行读取。
2. **目录审计**：校验每个选中分片的 SHA-256、统计实际行数和 schema，流式读取全部市场记录；创建时间、计划截止日期、抓取时间分开保留。`end_date` 不是正式发布时间或结算时间，`_fetched_at` 不是历史成交时间。
3. **候选选择**：当前首批筛选固定为 `closed=true` 且题目包含 Fed / FOMC / inflation / CPI / unemployment 关键词。这个检索范围包含国际数据、提及次数及衍生事件，不能直接作为“美国宏观发布”分类，更不能自动分配训练或测试角色。
4. **原生身份补充**：通过显式 `closed=true` 的 Gamma 批量接口补充父事件和 token 身份。当前描述仅供规则审核；不同父事件仍可能对应同一次经济数据发布，必须进一步合并关联事件。
5. **历史成交重建**：原始 `OrderFilled` 记录中的大整数 token ID 全程作为字符串处理；USDC 与 outcome token 数量使用整数分数相除，不先转换为浮点。No token 成交映射为对应 Yes 概率。token-token、零数量、边界/越界价格及未知 exchange 明确记录原因。
6. **时间关联**：使用 `block_number` 连接区块表，不以抓取时间或合约截止时间填补缺失时间。SQLite 保存筛选后的成交，避免把整个交易库加载到内存。成交扫描遵循经过哈希绑定的归档成员顺序，减少磁盘随机寻道；相同交易哈希与 log index 的记录去重，价格相关字段冲突直接失败。

SQLite 构建使用 WAL，允许进度读取与写入同时进行；发布前执行完整 checkpoint，再对主数据库文件计算哈希。读者阻止最终 checkpoint 时构建失败，不发布依赖未归档 WAL 的报告。

原生归档的负风险交易所名称是 `NegRisk CTF Exchange`，上游 `SCHEMAS.md` 简称为 `NegRisk`。适配器按[固定版本索引源码](https://github.com/Jon-Becker/prediction-market-analysis/blob/2276382cb616107db8c8647803bffa4a0d7091f8/src/indexers/polymarket/trades.py)接受完整名称；未知名称仍隔离。处理百万级候选成交时 SQLite 缓存预算为 1 GiB，关闭自动 checkpoint，在构建末尾显式执行 TRUNCATE checkpoint，避免反复回写增长中的索引。构建期间 WAL 会占用额外空间；最终报告仅绑定已完整落盘的主数据库，实际进程峰值内存单独记录。

`pma_trades.quote_at` 查询观察点之前最近的有效成交区块，对该区块内价格取算术均值，并检查新鲜度。本版有效成交要求 `0 < p < 1`，边界价和越界价都进入过滤审计；该筛选不等于断言原始链上记录无效。这是明确定义的历史成交价格基线，不代表可执行买卖报价、深度或成交后的收益。存在未关联时间的成交时，该市场报价查询失败，不跳过未知时间后偷偷选择其他记录。

## 本次实测目录（2026-09-19）

| 项目 | 实测数量 |
| --- | ---: |
| 原生市场记录 | 408,863 |
| 原生成交日志行 | 404,540,000 |
| 原生区块记录 | 78,468,431 |
| 市场快照为 closed | 383,707 |
| 当前严格 Yes/No token 映射器支持的市场 | 161,945 |
| 全目录精确命中既有保留身份或题目的市场 | 121 |
| 已关闭宏观关键词候选 | 906 |
| 候选原生 API 返回 | 906 |
| condition 身份与 outcome/token 对同时匹配 | 904 |
| 匹配且支持 Yes/No 的候选 | 898 |
| 原生父事件 | 303 |

三张表各有一种实际 schema，在成功映射的市场中未发现 token 跨市场碰撞。全目录的 Yes/No 映射数量不能解释为其他市场全部损坏：当前适配器只处理这两种 outcome 标签。906 个候选中，6 个使用其他二元标签，另有 2 个缺少 token 映射。原生父事件仍须合并同一次发布的相关合约。

区块表的时间范围为 `2020-09-03T04:33:11Z` 至 `2026-02-02T18:27:46Z`；成交表的区块号范围为 40,000,176 至 82,120,997。区块表覆盖范围不代表每个市场都有同期成交。旧 FPMM 数据仍保留在原始压缩包中，本版价格重建只处理 CTF Exchange / NegRisk 成交。

### 历史价格结果

当前有效价格库为 `data/processed/pma_macro_trade_store_20260919_v4`。

| 项目 | 实测数量 |
| --- | ---: |
| 完整扫描的原生成交行 | 404,540,000 |
| 906 个候选对应的成交行 | 5,651,134 |
| 解析、去重后有效成交 | 5,651,134 |
| 成功关联区块时间的成交 | 5,651,134 |
| 对应不同区块及成功关联数 | 1,716,126 / 1,716,126 |
| 具有历史成交的候选市场 | 779 / 906 |
| 有历史成交且原生身份/token 对匹配 | 779 |
| 本轮过滤成交或无效区块时间 | 0 |
| 新增准入训练 / 评测样本 | 0 / 0 |

选中成交的时间范围为 **2023-03-05 18:55:40 UTC 至 2026-01-25 17:23:54 UTC**。这是归档中成功重建的成交范围，尚未独立认证链上历史的完整性，也不代表每个计划观察点都有新鲜报价。

127 个没有本版可重建成交的市场仍保留在覆盖表中：8 个缺少本版支持的 Yes/No 映射，119 个有映射但归档中没有对应的可用成交。它们的 `end_date_hint` 年份分布是 2021：12、2022：64、2023：14、2025：4、2026：33；该日期仅用于定位缺口，不能替代成交时间或证明市场从未交易。906 个候选中有 15 个精确命中既有保留范围，继续保留排除标记。

完整核验已通过：扫描行数与原始目录一致、数据库 `integrity_check=ok`、先前 423,332 条 CTF 成交逐字段保留、原生身份审计离线重建一致、预览选择与完整归档一致，以及两类交易所覆盖市场的历史截止查询检查。完整测试 **347 项：346 通过，1 项可选 GPU 测试跳过**；本次新增 21 项测试覆盖归档截断/哈希、字段适配、时间截断、保留清单接口、数据库并发和可重复构建。

本机成功阶段耗时：解包 1,031.546 秒、目录审计 1,565.490 秒、最终价格构建 1,048.546 秒。价格构建峰值 RSS 为 1,903,184 KiB，最终主数据库 2,070,466,560 字节。未使用 GPU，模型调用为零。

当前产物：

- 原始归档：`data/raw/pma_archive_20260919_v1`。
- 完整解包：`data/raw/pma_polymarket_extracted_20260919_v1`。
- 市场目录：`data/processed/pma_polymarket_catalog_20260919_v1`。
- 原生身份审计：`data/processed/pma_macro_enrichment_20260919_v2`。
- 有效价格库：`data/processed/pma_macro_trade_store_20260919_v4`。
- 运行与核验：`runs/pma_historical_ingestion_v1/{active_artifacts,verification_full_exchange,negrisk_quote_checks,pipeline_stages_full_exchange,environment}.json`。

较早的价格库 v1–v3 保留作诊断，当前产物以 `active_artifacts.json` 为准。正式训练与评测仍须完成历史规则、结算标签、语义去重和分区核验；这些价格记录不直接生成 SFT 概率答案。

## 用途边界

市场快照通常晚于被研究的历史观察点。快照中的最终价格、关闭状态、成交总量、当前规则不能回填进旧时点的模型输入。候选市场的最终价格提示单独存入 `snapshot_outcome_hints.jsonl`，不当作正式结算标签或 SFT 概率答案。

现有 ForecastBench 索引、此前保留为评测用途的宏观市场继续执行身份和题目排除。非精确匹配不代表已通过语义去重；原生父事件是最小分组单位，还需合并相关发布和跨平台合约。目录和价格仓库均不自动准入训练或评测，也不修改旧 2026 年评测集合的用途。

接下来的训练候选可从历史库选择较早事件，现有较晚评测范围继续保留。正式冻结分区之前，须完成事件分组、信息截断、结算标签和任务目标检查；不能从最终输赢直接制造事前概率答案。

## 复现

使用 `conda lab`。需安装 `pyarrow`、`zstandard`，并能在 PATH 中找到 `zstd` 命令。可选 Python 依赖组为 `.[archive]`。

目录审计使用已有的 [ForecastBench 排除索引](sft_warmup.md)和[宏观评测保留清单](macro_evaluation_scope.md)；首次从空数据目录复现时，先按相应说明构建这两个输入。目录报告绑定它们的内容哈希。

```bash
conda activate lab

# 首次传输；所有下载均校验固定 HTTP 对象身份。
PYTHONPATH=src python -m foretellmesh.pma_archive download \
  --config configs/pma_archive_v1.json \
  --output data/raw/pma_archive_20260919_v1

# 若已停止首次传输，可保留前缀/完整分段后续传；不得与上一个进程同时运行。
PYTHONPATH=src python -m foretellmesh.pma_archive download-parallel \
  --config configs/pma_archive_v1.json \
  --output data/raw/pma_archive_20260919_v1

PYTHONPATH=src python -m foretellmesh.pma_archive extract \
  --config configs/pma_archive_v1.json \
  --archive data/raw/pma_archive_20260919_v1 \
  --output data/raw/pma_polymarket_extracted_next

PYTHONPATH=src python -m foretellmesh.pma_catalog \
  --extraction data/raw/pma_polymarket_extracted_next \
  --heldout-index data/processed/forecastbench_sft_exclusion_index_v1.json \
  --reserved-inventory data/processed/macro_evaluation_inventory_20260919_v3/inventory.json \
  --output data/processed/pma_polymarket_catalog_next

PYTHONPATH=src python -m foretellmesh.pma_trades \
  --extraction data/raw/pma_polymarket_extracted_next \
  --catalog data/processed/pma_polymarket_catalog_next \
  --output data/processed/pma_macro_trade_store_next

PYTHONPATH=src python -m foretellmesh.pma_enrichment capture \
  --input data/raw/pma_polymarket_extracted_next/polymarket/markets \
  --output data/raw/pma_macro_enrichment_next

PYTHONPATH=src python -m foretellmesh.pma_enrichment build \
  --input data/raw/pma_macro_enrichment_next \
  --output data/processed/pma_macro_enrichment_next
```

输出目录须尚不存在，下载目录支持经验证的续传。原始数据、数据库和报告按仓库策略保存在忽略目录；代码、配置和复现说明进入 Git。完整归档下载后会保留副本，分段缓存占用额外空间。
