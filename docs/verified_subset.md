# 首批真实样本：历史证明与保留集基线

本实验验证数据链路，没有运行或训练语言模型，也不是正式 ForecastBench 排名。6 条样本过少且按标签可用性筛选，不能支持模型能力、市场效率或校准效果的结论。

## 选择规则与结果

固定使用 2026-01-04 ForecastBench 轮次内的全部 **74 个 Manifold 问题**作为审计范围，检查固定结算文件中全部 7 个已结算问题；不根据基线得分选择样本。选择规则、全部 7 个平台结果摘要哈希和事件组映射保存在 [证明配置](../configs/forecastbench_manifold_provenance_v1.json)。

| 检查结果 | 数量 |
| --- | ---: |
| Manifold 原始问题 | 74 |
| 固定结算文件中已结算 | 7 |
| 问题版本、二元结果和结算日期均通过交叉检查 | 6 |
| 平台精确 UTC 时间与基准结算日期冲突 | 1 |
| 未结算 | 67 |

通过的事件涉及台北 101 攀登、2026 世界杯、美国 2025 GDP 衰退条件、Monopoly 电影和 Musk 财富条件。两个世界杯问题放在同一事件组，最终 6 条样本属于 5 个组。未通过的 Messi 世界杯出场合约也归于同一世界杯组，继续隔离。

当前分组只覆盖这个小集合；它不是所有 Prophet Arena / ForecastBench / PMA 事件已完成跨来源语义去重的证明。没有训练记录，因此本次不存在已导入训练样本与测试样本交叉的情况。

## 历史题目与市场快照

原始问题发布提交为 [`f5dd46a24eb7a1e58c4d04715385aaf869088d0d`](https://huggingface.co/datasets/forecastingresearch/forecastbench-datasets/commit/f5dd46a24eb7a1e58c4d04715385aaf869088d0d)。官方提交元数据记录时间为 **2026-01-04 00:05:20 UTC**。

该提交中的 [题目文件](https://huggingface.co/datasets/forecastingresearch/forecastbench-datasets/resolve/f5dd46a24eb7a1e58c4d04715385aaf869088d0d/datasets/question_sets/2026-01-04-llm.json) 与项目固定的当前版本逐字节一致，SHA-256 为：

```text
6e2a23bdb27d3449e80752e82db4d37ae74680145c188033ff5ae9b1a02de5e1
```

本实验将 **2026-01-04 23:59:59 UTC** 明确选为内部回放观测时点。这是版本化实验定义，不声称它是官方参赛截止时刻。它晚于原始问题发布提交。

市场值来自历史文件中的 `freeze_datetime_value`，报价时刻为 **2025-12-25 00:00:00 UTC**。它是旧的冻结快照，不是 1 月 4 日的新报价。`observed_at` 保留冻结时间，`available_at` 保守设为该已存档文件的发布提交时间。初版只使用历史问题、历史规则引用和这份冻结快照，不使用当前网页或平台返回的题目、背景、价格。

这里的公开时间证据依赖官方仓库的发布提交元数据，未宣称拥有独立的第三方时间公证。验证器同时检查真实提交 ID、提交时间、原始文件哈希及实验截止时点，防止仅填写一个更早时间便放行。

## 精确结算与标签可用性

[Manifold 官方 API 文档](https://docs.manifold.markets/api) 区分 `resolutionTime` 与 `closeTime`。验证器只从 `/v0/market/{id}` 提取以下字段：

```text
id, outcomeType, isResolved, resolution, resolutionTime
```

只有 BINARY、已结算、YES/NO 的记录可用。毫秒时间戳按整数转换成 UTC，避免浮点精度损失。结果必须与固定 ForecastBench 标签相符，UTC 日期也必须与其 `resolution_date` 相符；任何冲突都隔离。

平台返回的其他字段，包括当前 probability、question、description、评论和用户信息，都不进入模型 payload。当前接口的作用仅限于评估端核对标签，历史预测输入仍由已证明的历史题目文件构造。

结算文件固定在 [`a11ac3a9ba8812cdedab2b79ab3181a9c0825d62`](https://huggingface.co/datasets/forecastingresearch/forecastbench-datasets/commit/a11ac3a9ba8812cdedab2b79ab3181a9c0825d62)，提交时间为 **2026-09-17 06:41:31 UTC**。`label_available_at` 使用这一有存档证据的“截至此时已可用”上界，**不是首次公开时间**。该时间只用于本次保留集评分，不能据此允许标签提前进入历史训练。

具体冲突：`lgxNmHNmrYvowjnSyQAM` 的基准日期为 `2026-06-30`，平台精确时间为 `2026-07-01T00:08:42.943Z`。没有把它改写成午夜或使用关闭时间来消除差异。

## 复现

如尚未获取固定原始数据，先执行 README 中的 `fetch --source forecastbench`。以下输出目录必须不存在；重复实验使用新目录。

```bash
PYTHONPATH=src python -m foretellmesh fetch-provenance \
  --plan configs/forecastbench_manifold_provenance_v1.json \
  --output data/raw/forecastbench_manifold_proofs_v1

PYTHONPATH=src python -m foretellmesh verify-forecastbench \
  --plan configs/forecastbench_manifold_provenance_v1.json \
  --data data/raw/forecastbench_v1/2026-01-04-llm.json \
  --resolutions data/raw/forecastbench_v1/2026-01-04_resolution_set.json \
  --proofs data/raw/forecastbench_manifold_proofs_v1 \
  --output runs/forecastbench_manifold_verified_v1

PYTHONPATH=src python -m foretellmesh evaluate \
  --data runs/forecastbench_manifold_verified_v1/import/records.jsonl \
  --config configs/forecastbench_manifold_smoke_v1.json \
  --output runs/forecastbench_manifold_smoke_v1
```

下载器只访问配置中的官方历史文件、提交元数据和公开结算接口，限制响应大小，并在全部验证后发布目录。题目文件/提交元数据校验完整字节哈希；平台接口校验上述五个标签字段的规范化哈希。无关的当前价格变动不影响验证，如果结算字段改变则报错，不能静默重写旧实验。原始响应另存原始字节哈希，便于追查。

完成首次下载后，`verify-forecastbench` 与 `evaluate` 均可离线复现。若未来上游更正结算结果、接口不可访问或元数据表示改变，应保留旧证明目录，并以新的版本化计划审查变动；旧文件不会被覆盖。

验证输出包含：

- `verification.json`：74 个问题的逐项检查、证明 URL/哈希、选择规则、时间和限制。
- `annotations.json`：自动从通过校验的来源生成的 6 条注释，绑定输入哈希。
- `import/records.jsonl`：可供评估的 6 条统一记录。
- `import/audit.json` 与其他导入产物：来源版本、注释哈希和未放行候选的状态。

## 本次基线结果

[评估配置](../configs/forecastbench_manifold_smoke_v1.json) 显式选择固定 0.5 与市场基线，不进行经验概率拟合。train / validation / test 数量为 **0 / 0 / 6**，ForecastBench 保持 `eval_only`。

| 基线 | Brier | Log Loss | ECE（5 箱） | 可评分样本 |
| --- | ---: | ---: | ---: | ---: |
| 固定 0.5 | 0.250000 | 0.693147 | 0.333333 | 6 / 6 |
| 历史市场快照 | 0.180628 | 0.518205 | 0.310721 | 6 / 6 |

指标仅用于核验管线。报告覆盖率的分母是通过资格筛选的 6 条记录；74 个源问题中还有 68 个未放行，这一筛选过程由 `verification.json` 单独完整记录。样本小、事件相关、类别混合、快照陈旧且只选已结算记录，不应把上述差异解读为普遍效果。

下一步需要独立整理可追溯训练数据，扩大预先定义的评估范围，并冻结更有代表性的测试集。这 6 条记录继续作为保留集数据链路检查样本。
