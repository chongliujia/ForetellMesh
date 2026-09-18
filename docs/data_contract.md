# 数据与评估契约 v1

本契约服务于二元事件的离线历史回放。输入为 UTF-8 JSONL，每个非空行是一条记录；未知字段、重复 JSON 键和非标准 NaN / Infinity 均被拒绝。所有时间必须是带时区的 ISO 8601 字符串，读取时统一为 UTC。

## 一条统一记录

```json
{
  "sample_id": "synthetic:example:2024-01-10",
  "dataset_source": "synthetic_training",
  "dataset_version": "1",
  "event_id": "example",
  "event_group_id": "synthetic-example-group",
  "question": "Will the synthetic indicator exceed its threshold?",
  "observation_time": "2024-01-10T00:00:00Z",
  "evidence": [
    {
      "evidence_id": "synthetic-e1",
      "text": "Artificial fixture evidence.",
      "source": "synthetic://example",
      "published_at": "2024-01-09T00:00:00Z",
      "available_at": "2024-01-09T00:00:00Z"
    }
  ],
  "market": {
    "probability": 0.6,
    "observed_at": "2024-01-09T23:59:00Z",
    "available_at": "2024-01-09T23:59:00Z"
  },
  "label": {
    "outcome": 1,
    "resolution_time": "2024-01-20T00:00:00Z",
    "available_at": "2024-01-21T00:00:00Z"
  }
}
```

所有顶层字段均必填。`evidence` 可以是空数组；没有市场快照时 `market` 为 `null`；标签未知时 `label` 为 `null`，不能填入猜测的 outcome。outcome 只接受整数 `0` 或 `1`，不接受布尔值。取消、作废、多结果或部分结算事件暂不支持，源适配器需要另行留存并说明处理策略。

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 整个导入文件中唯一的样本 ID；推荐包含来源与观测时点 |
| `dataset_source` / `dataset_version` | 与实验配置中的数据源注册信息严格匹配 |
| `event_id` | 原数据源的事件身份，同一来源跨版本保持稳定 |
| `event_group_id` | 人工或审核过的规则生成的跨来源事件组，相关合约放在同组 |
| `observation_time` | 本次预测允许看到信息的最晚时间 |
| `evidence.available_at` | 有来源支持的当前证据版本可用时间；修订内容不能沿用原文章的旧时间 |
| `market.observed_at` / `available_at` | 该概率对应的报价时间和可用时间 |
| `label.resolution_time` / `available_at` | 事件结算时间和标签可用于拟合或评估的时间 |

`available_at` 与原始下载时间的含义不同。历史回放需要能证明当时可获取的内容版本；原始下载文件、URL、抓取时间和内容哈希应由源适配器保存在原始数据清单中。不要从当前网页内容反推一个未经证实的历史时间。

若只知道“截至某个已存档时点，该版本已经可用”，可使用这个有证据支持的保守时间上界，并在 provenance 中明确记录。它不代表首次公开时间，不能据此把标签或内容提前到更早的训练/观测截止点。源导入会将市场快照的可用时间取为源声明时间与已证明的问题快照可用时间两者中的较晚者；报价对应时刻 `observed_at` 保持不变。

## 输入与标签分离

加载后，`ForecastRecord` 保存来源元数据、`ForecastInput` 与独立 `Label`。

```python
from pathlib import Path
from foretellmesh.data import load_records

records, dataset_hash = load_records(Path("examples/synthetic_forecasts.jsonl"))
model_payload = records[0].forecast_input.to_payload()
```

模型 payload 只含问题、观测时点、证据及可选市场快照。不要将原始记录、`dataclasses.asdict(record)`、评估报告或 `predictions.jsonl` 拼入模型输入。提供市场概率的模型实验与隐藏市场概率的实验应分别命名；本阶段尚无模型实验。

## 身份、重复与事件分组

导入阶段执行以下检查：

1. `sample_id` 唯一，同一来源、同一事件、同一观测时间不能重复，即使来自不同数据集版本。
2. 相同来源事件的全部观测必须使用同一事件组和一致的结算标签。
3. 经 Unicode NFKC、大小写和空白规范化后相同的问题必须使用同一事件组和一致的结算标签。
4. 完成时间和标签资格判断后，同一问题与观测时点仅保留一条。按 `eval_only` 优先，再按来源、版本、样本 ID 排序选择，其他记录写入剔除清单。

这些规则只覆盖明确可检测的重复。不同措辞、翻译、不同平台合约以及同一宏观事件的关联问题仍需人工或经过验证的分组流程。定期重复的问题必须在问题文本中明确事件周期，否则可能被判为身份冲突。报告中的 `semantic_event_grouping` 明确记录这一限制。

## 切分与标签资格

设配置中的验证开始时间为 V，测试开始时间为 S，评估截止时间为 E，要求 `V < S <= E`：

| 分组 | 观测窗口 | 标签可用时间上界 |
| --- | --- | --- |
| train | `T < V` | `label.available_at <= V` |
| validation | `V <= T < S` | `label.available_at <= S` |
| test | `S <= T <= E` | `label.available_at <= E` |

同一事件组的全部记录先一起检查所属时间窗口。只要跨越 train / validation / test 边界，整组剔除；即使部分记录尚未结算，也不会先移除它们再放行同组记录。此策略保守，会牺牲数据量，剔除清单需随实验保留。

配置里的数据源角色：

- `train_eval`：允许按时间窗口进入三类集合。
- `eval_only`：只允许进入验证和测试。若位于训练窗口，该事件组全部剔除，包含同组的其他来源记录。

初版不移动跨边界事件、不做随机切分、不自动把早期基准数据调整到测试窗口。真实 ForecastBench 用作保留基准时应注册为 `eval_only`，同时冻结版本、事件分组和最终测试策略。

标签必须满足 `observation_time < resolution_time <= label.available_at`。观测晚于 E、未知标签或标签过晚可用的记录会被剔除，并保留具体原因。这些资格规则会改变评估样本构成，不能把已结算子集的表现直接外推到全部开放事件。

## 指标和基线

- **固定基线**：所有概率为 0.5。
- **经验基准概率**：最终训练集 outcome 的均值；不使用验证或测试标签。拟合与评分均按保留的观测样本等权，多个不同观测时点会分别计权。
- **市场基线**：该样本观测时点已可用的市场快照；缺失时不补为 0.5。

配置的可选 `baselines` 是以上名称组成的非空、无重复列表：`constant_0_5`、`empirical_base_rate`、`market`。省略时保留全部三条。明确省略 `empirical_base_rate` 时，既不拟合也不报告经验概率，可在没有训练记录的情况下评估保留集；不会用测试标签估计该概率。省略市场基线时不进行市场同样本对比。

对于有有效概率的 n 个样本：

```text
Brier = mean((p - y)^2)
Log Loss = mean(-log(max(epsilon, p if y == 1 else 1 - p)))
ECE = sum((bin_count / n) * abs(bin_mean_probability - bin_event_frequency))
coverage = prediction_count / eligible_count
```

Log Loss 使用自然对数，并对真实结果的概率做下界截断；p=0/1 时分数依然有限。epsilon 记录在配置和报告中。概率中的 NaN、无穷或越界数值是错误，不能作为缺失值静默丢弃。

ECE 使用固定的等宽概率区间，每个区间左闭右开，最后一个包含 1。它比较事件概率和事件发生频率，不将概率转换为分类置信度。空箱保留 count=0，均值与频率为 null。JSON 报告提供完整曲线数据，暂不生成图片。

市场基线可能只覆盖部分样本，因此 `market_matched` 对所有基线使用同一批有市场快照的样本。全量基线与部分市场分数不应直接用于优劣结论。没有可用市场快照时，同样本对比返回 null。

覆盖率的分母是通过切分与标签资格筛选后的样本数；它不包含被剔除的开放事件。原始总量、最终分组数量和各类剔除数量另行记录。没有测试样本时运行失败；启用经验基线但没有训练样本时也失败。验证集允许为空，并明确报告空指标。

## 可复现性与当前范围

配置版本控制在 `configs/`。每次运行保存原始配置副本、输入文件 SHA-256、配置 SHA-256、切分清单 SHA-256、Python 源码 SHA-256、包版本、Python 版本以及可获取的 Git commit / dirty 状态。相同输入、配置、代码和环境重复运行产生相同指标与清单；改变输入行序仍保持相同切分，但原始文件哈希会变化。

JSONL 统一格式是内部契约，不是现有三个真实数据集的原始格式。已提供原始格式适配与一小组历史来源验证，详见 [来源接入](source_ingestion.md) 和 [真实样本复现](verified_subset.md)。全局语义去重、按事件组的不确定性估计、模型输出有效率和训练 / 推理资源指标仍待实现。当前报告仅验证确定性基线与数据管线，不能据此声称模型改进。
