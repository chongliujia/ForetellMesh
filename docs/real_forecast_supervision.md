# 真实预测监督数据：第一批来源与准入状态

用户选择先补真实预测监督，再训练 Forecast LoRA。本轮新增的是**可追溯的历史教师概率候选库**，不是完成微调，也不是已经合格的 SFT 数据集。没有用合成题替代这项工作。

## 已完成的补充

固定采集费城联储 Survey of Professional Forecasters（SPF）2019–2024 年四个季度的报告，共 24 份。每份提取本期调查对当季及随后四季实际 GDP 环比负增长的五个概率，共 **120 条历史专家预测，覆盖 28 个不同目标季度**。

这是真实受访预测者的均值，由费城联储发布；不能称为联储自身的预测、已校准的真概率或最终结果。目标也不是 NBER 经济衰退，更不是 FOMC 利率、CPI 合约。变量定义及调查时点见 [官方 FAQ](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/spf-faqs)。

新增 [targets.jsonl](../data/processed/spf_forecast_supervision_20260920_v1/targets.jsonl) 保存原始百分数、转换后的概率、目标季度、预测期限、来源 URL、原始文件哈希、实际下载时间和发布日期。完整 [manifest.json](../data/processed/spf_forecast_supervision_20260920_v1/manifest.json) 绑定源文件与配置。所有记录的 `student_input`、`outcome`、`forecast_schema_completion` 均保持 `null`，`split=unassigned`、`ready_for_sft=false`。

不把报告发布时间冒充每位受访者提交预测的时刻。只有日期的发布记录保留 `publication_precision=day`，另记录纽约当地次日零时的保守公开可用界限；这一界限不是精确发布时间，也不是教师的原始信息截止时间。

## 原始来源核对

首轮 HTML 抓取包含三次传输失败；只重试失败请求，成功字节不变。2019 年的部分 URL 返回 HTTP 200，但页面实际为 `Error - 404`，标题校验将其拒绝。2020 年两份网页的概率表有目标年份冲突：

| 报告 | HTML 中的冲突 | 官方 PDF 中对应目标 | 处理 |
|---|---|---|---|
| 2020 Q3 | 第三行写作 2020 Q1 | 2021 Q1 | 采用逐页目视核对的 PDF 表格 |
| 2020 Q4 | 第三行写作 2020 Q2 | 2021 Q2 | 采用逐页目视核对的 PDF 表格 |

没有直接修改抓取内容或按相邻行猜测年份。四份 2019 年报告及这两份 2020 年报告，共六份采用官方 PDF 替代来源；解析先核对 PDF 标题、发布日期，再定位 `Risk of a Negative Quarter (%) / Survey Means`，只取 `New` 列。六个完整表格页已渲染并目视核对，审核记录见 [spf_pdf_supervision_review_v1.json](../configs/spf_pdf_supervision_review_v1.json)。正式库最终没有未解决的报告缺口；旧抓取错误和预检失败仍保留。

原始 HTML 保存在 `data/raw/spf_forecast_supervision_20260920_v1/` 与 `v2/`，六份报告 PDF、官方文档和勘误保存在 `data/raw/spf_forecast_pdf_review_20260920_v1/`。勘误文本中未检索到 2019–2024 年条目，这不是证明历史页面从未被改动。当前下载的官方归档不等于独立保存的首次发布快照；原始版本信任范围仍需在未来的具体训练方案中明确。

## 为什么还没有开始微调

目前补上的是监督概率这一部分。要检验“根据当时的信息预测未来”，还缺以下配套工作：

1. **同一事件的学生输入。** 需要独立的、按历史截止时间过滤的证据。教师报告的答案表和叙述不能直接放入输入，否则模型学到的是答案提取。官方 FAQ 明确不记录每个预测者作出预测的具体日期；问卷日期和提交截止日期可以界定一个区间，但不能据此宣称已复原相同信息集。
2. **事件评分口径。** 实际 GDP 有首发值及多次修订。需要先规定目标按哪一版数据结算，再补对应标签与可用时间，才能计算有意义的 Brier / Log Loss。贴近专家均值的 MAE 仅衡量蒸馏一致性，不能替代事件预测评分。
3. **训练与新开发事件隔离。** 同一目标季度会在多份调查中重复出现，一份调查又含五个期限，120 条不是 120 个独立事件。应按时间及关联事件切分并清除边界重叠，完成与既有基准的语义重合审核；不能随机拆行，也不能将现有验证或最终测试改作训练。
4. **监督字段范围。** SPF 概率不能自动提供本项目所需的证据引用、置信度和推理摘要。后续可以设计仅监督已核验概率字段的协议；不能补写历史推理后宣称它来自专家。

这些候选没有自动接入训练入口。当前 `ready_for_sft_count=0`；这是对完整训练样本的判断，不否定 120 条真实概率记录的价值。下一步应围绕这些目标补时点证据和评分数据，并固定开发分区，而非仅继续增加没有配套输入的教师数字。

## 其他已核对来源

| 来源 | 本轮判断 |
|---|---|
| 本地 Prophet Arena Subset 1200 | 既有审计 7,022 个候选缺历史预测目标，12,004 条来源项缺独立时间证明；此前保存的公开预测 API 示例为空数组，不能据此新增监督。 |
| 已准入 PMA 训练覆盖 | 64 个观察点、37 个合约、12 个事件可以支持已有回放，但当前没有匹配的独立教师回答；SPF GDP 概率不会填进这些不同含义的合约。 |
| ForecastBench | 继续保持评估用途，本轮没有读取其问题或结算标签。 |
| [OpenForesight](https://huggingface.co/datasets/nikhilchandak/OpenForesight/blob/main/README.md) | 官方卡片提供新闻生成问题、答案与检索提示，列出的字段未提供可直接采用的历史教师概率。没有把事后答案转成 SFT 概率，也没有下载或读取其测试分区。 |

## 验证与复现

39 项相关测试通过，覆盖百分数转换、当前/上期列混淆、概率边界、目标季度错位、日期格式、伪 HTTP 200 错误页、导航污染、原始文件篡改、PDF 表格及候选库拒绝作为训练包。独立审核重新解析源文件与六份 PDF，逐条核对全部 120 条概率及季度运算；原结果字节一致。

```bash
PYTHONPATH=src python -m foretellmesh.forecast_supervision \
  --config configs/forecast_supervision_sources_v1.json \
  --capture data/raw/spf_forecast_supervision_20260920_v2 \
  --pdf-review configs/spf_pdf_supervision_review_v1.json \
  --output data/processed/spf_forecast_supervision_20260920_v1

PYTHONPATH=src python scripts/audit_forecast_supervision.py \
  --config configs/forecast_supervision_sources_v1.json \
  --capture data/raw/spf_forecast_supervision_20260920_v2 \
  --pdf-review configs/spf_pdf_supervision_review_v1.json \
  --bundle data/processed/spf_forecast_supervision_20260920_v1 \
  --output runs/real_forecast_supervision_inventory_v1/audit.json

PYTHONPATH=src:tests python -m unittest \
  test_forecast_supervision test_sft_data test_timely_macro_evidence
```

构建命令在目录已存在时逐项复算比较，不覆盖旧结果；审计输出必须使用新路径。源采集配置见 [forecast_supervision_sources_v1.json](../configs/forecast_supervision_sources_v1.json)。本轮未加载 GPU 模型、未训练或晋级 LoRA、未运行 PPO、未导出交易信号；最终测试保持封存。
