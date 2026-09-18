# 历史评测范围审核与宏观候选目录

2026-09-19，本轮完成现有 64 条 FOMC 观察与固定 ForecastBench 索引的用途审核，并归档更广的宏观市场目录。**20 条旧观察通过逐条 held-out 回放检查；正式评分准入仍为 0。** 新目录的 **22 个事件组、408 个合约**仅为排除明显身份问题后的补证队列，不是合格评测样本。本轮没有调用模型、训练或产生 Brier / Log Loss / ECE 分数。

## 现有样本的用途审核

`benchmark_review` 把原生 Polymarket 数字 market ID 与 condition hash 绑定到同一个合约，避免因 ForecastBench 使用另一种 ID 而漏检。一个合约命中 held-out 索引时，整组相关合约跨平台一起排除。

审核策略 `configs/historical_benchmark_review_v1.json` 固定完整索引文件哈希及模型输入、原题、原生身份别名、事件组的投影哈希。修改索引、问题、输入、事件组或身份后必须重新审核；结果标签不参与语义审核投影，也不会进入模型输入。

固定索引包含 2,244 条条目：250 条原生问题和 1,994 条数据序列预测时域。对标题及索引元数据的 Codex 辅助审核未发现 7 月、9 月 2026 FOMC 的同合约或同次会议条目；保守记录 823 条可能共享宏观驱动因素的关联条目。**这不是通用语义去重证明，也不是对所有源站合约规则的重新审核，更不代表已测得这些问题之间的相关性。** 原 ForecastBench 全索引继续保留为训练排除来源。

本轮 64 条记录全部标记 `evaluation_only_reserved`；禁止训练、提示词调优、奖励调优和 checkpoint 选择。只有显式 `heldout_replay` 且禁用上述用途的策略可忽略此保留标记。旧 `development_replay` 路径不能把这些数据变成调参集。

| 当前检查结果 | 数量 |
| --- | ---: |
| 精确结算标签 | 64/64 |
| 模型输入 payload 保持不变 | 64/64 |
| 通过逐条 held-out 回放检查 | 20，均为 Polymarket |
| 上述合格观察对应的事件组 | 2 |
| Kalshi 历史规则版本仍缺失 | 44 条观察 |
| 历史报价不可用 | 39 条观察 |
| 正式评分 / SFT 准入 | 0 / 0 |

批次门槛仍保留：至少 20 个事件组、两个平台均有通过检查的样本、冻结实际时间/事件组切分和 checkpoint manifest。事件组数量不是统计独立性或统计功效的保证。教师答案缺失只阻止 SFT，不阻止用独立的结果标签评分。

## 固定扩充范围

`configs/macro_evaluation_discovery_v1.json` 固定 2026 年 1—9 月官方发布窗口：6 次 FOMC、9 次 CPI 同比、9 次美国 U-3 失业率发布，共 24 个日历槽位。CPI / 就业按数据所属月份分组，因此包含 2025 年 12 月数据在 2026 年 1 月的发布。高低区间、不同平台、重复事件页和两个计划观察时点不增加事件组数量。

参考来源为 [Federal Reserve 日历](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) 和 [BLS 2026 发布日历](https://www.bls.gov/schedule/2026/home.htm)。当前日历仅用于发现与候选观察时间规划，**不能证明当时已知日程，也不能代替实际首次发布和修订审核**。BLS 直连返回 403，因此保留浏览工具抽取的官方正文，并标注 `web_tool_extracted_text`，没有冒充原始 HTTP HTML。

采集器只访问计划中明确列出的公开只读接口，总计 80 个请求，均成功并保存原文、URL、状态、请求时间及 SHA-256。Kalshi 同时读取当前事件的嵌套 markets 与历史 markets；按 ticker 去重。端点失败、分页未完成、空事件与未找到月份均保留，不能从分母中删除。

| 目录范围 | Polymarket | Kalshi |
| --- | ---: | ---: |
| 原生事件目录条目 | 25 | 15 |
| 归档合约 | 192 | 236 |
| 有合约覆盖的日历事件组 | 22 | 15 |

已处理的质量问题：

- **3 月议息基准重合**：Polymarket 事件 `67284` 的 condition `0x257b18205f908aef01ef2d1d50e6fea7d29cf5486fe04031d8b101394476faed` 命中 ForecastBench。整次发布的 4 个 Polymarket 和 11 个 Kalshi 合约一起排除出本次新增队列。
- **重复 9 月议息事件**：旧事件 `432066` 的 5 个合约元数据显示 5 月已关闭/报告 resolved，远早于 9 月计划观察时间，全部隔离。当前已完成链上核验的是另一个原生事件 `481717`。API 的状态更新时间只用于发现异常，不作为最终结算时间或标签。
- **日期冲突**：1 月 CPI 的两组区间事件、1 月失业率、6 月 CPI，共 4 个目录条目的市场 endDate 与官方日历日期不同，保留明确标记；不能用市场 endDate 代替真实首次发布时间。
- **未找到的月份**：本轮有界搜索未找到 2026 年 2 月 CPI 和 U-3 的 Polymarket 事件；2 月 CPI 有 Kalshi 候选，2 月 U-3 整槽仍缺失。不能据此断言市场从未存在。
- **国家/系列身份**：Kalshi `KXUE` 目录属于国际失业率，不能直接映射为美国 U-3；9 个美国就业槽位的 Kalshi 映射继续留空。

因此，428 个原生合约减去 15 个基准重合和 5 个异常旧合约，剩 **408 个、22 个有待补证事件组**。这是原生 ID 数量；相邻区间、平台之间仍可能高度相关。Gamma 搜索召回和分页并不完备，计划明确保留这一限制。

目录不会输出 forecast input、监督答案或结算标签，所有 `ready_for_scoring`、`ready_for_training` 均为 false。下一阶段按固定队列补齐实际首次发布及修订记录、历史规则、最终结算、事前价格与证据，再做整个新范围的语义审核和实际样本冻结。未完成前不启动模型对照，也不放宽准入门槛。生成长度、实际输入 token 预算和 checkpoint 内容也须在运行前核验；当前 admission 配置不是已冻结的可执行 GPU run manifest。

## 复现

以下离线命令依赖本机归档；原始数据和运行产物按现有策略不提交 Git，所有输出目录必须尚不存在。

```bash
conda activate lab
PYTHONPATH=src python -m foretellmesh.historical_replay_gate \
  --staging data/processed/historical_fomc_replay_20260918_v2 \
  --capture data/raw/historical_fomc_capture_20260918_v1 \
  --heldout-index data/processed/forecastbench_sft_exclusion_index_v1.json \
  --config configs/historical_heldout_admission_v1.json \
  --chain-bundle configs/historical_chain_bundle_v1.json \
  --benchmark-review configs/historical_benchmark_review_v1.json \
  --output runs/historical_scope_admission_next.json

PYTHONPATH=src python -m foretellmesh.macro_discovery build \
  --capture data/raw/macro_evaluation_capture_20260919_v1 \
  --heldout-index data/processed/forecastbench_sft_exclusion_index_v1.json \
  --output data/processed/macro_evaluation_inventory_next

# 可选新采集；保留已冻结归档，不覆盖。需配置引用的搜索/日历原文。
PYTHONPATH=src python -m foretellmesh.macro_discovery capture \
  --config configs/macro_evaluation_discovery_v1.json \
  --output data/raw/macro_evaluation_capture_next
```

本轮正式目录：`data/processed/macro_evaluation_inventory_20260919_v3`；早期 v1/v2 构建保留，仅 v3 与下列报告对应。运行记录：`runs/historical_scope_review_v1/{admission,inventory_report,verification}.json`。完整 306 项测试：305 通过，1 项可选 GPU 测试跳过；新增 12 项测试覆盖别名跨平台整组排除、评测用途限制、索引/输入变更、缺失分母、国家映射、嵌套市场、原文篡改与结果隔离。目录文件离线重建逐字节一致，旧 64 条准入报告重建完全一致。

```text
Capture manifest SHA-256:
c8691f25e0392b44da699dcdfbbfdc850a63343cfd2169ae6346784c08b1e5ff
Inventory SHA-256:
53f9e8d2b7e43f51e800e856cb290ecaa5180aa988cb4b7003ab8219f25800b1
Admission SHA-256:
7341c73980f770bc4fb35565cad1c7813095b15594b7cfb54fef18a2ee316364
```
