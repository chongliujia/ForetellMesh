# 临近预测时点的官方宏观证据

本轮沿用先前固定的 12 个训练事件、19 个观察点，不依据收益或预测误差重新选样。目标是修复历史研究只携带上一份旧材料的问题，再检验原始 Base 能否利用更及时的信息。

## 数据与时间边界

固定采集清单包含 24 份 BLS 历史发布：12 份 CPI 和 12 份就业报告。按 URL 日期、官方禁发时间、星期和报告月份交叉核验；每份保存完整网页工具返回、检索时间和哈希。直接 HTTP 访问返回 403，正式证据使用网页工具提取的官方归档文本。

只提取当期正文中的四个数值字段：CPI 同比、剔除食品能源的 CPI 同比、当月非农就业变化、U-3 失业率。不给模型整张历史表格、后来修订的前月值或网页导航里的当前信息。尤其区分“上升多少个百分点”与“失业率水平”，并为“基本不变但括号中给出 +12,000”这样的官方表述提供明确解析。

每个预测时点仅从固定清单中选取已可用且不超过 60 天的最近一份 CPI 和一份就业报告，按发布时间排序。19 个观察点全部取得这两类证据；新增材料中最新证据的年龄中位数约 **13.23 天**，此前输入为 **41 天**。该清单不是完整历史新闻库，不包含全部工资、PCE、央行讲话或银行业事件信息。

2024 年 7 月 11 日 CPI 发布注明当天重发。因缺乏重发的具体时刻，将此版本保守地视为纽约当地次日零时，即 **2024-07-12 04:00 UTC** 才可用。没有假定重发前的版本等于当前版本，也没有把它放回初次发布时刻。该来源见 [BLS 当日 CPI 归档](https://www.bls.gov/news.release/archives/cpi_07112024.htm)。

其他 23 份发布未在官方页面检索到 `corrected`、`correction` 或 `reissued` 标记。**这是有日期的官方归档正文及更正检索证据，不是独立保存的首次发布快照。** 此信任范围与取回时间分开记录；不能据此声称已排除所有未注明的历史网页修改。

正式证据包：[bundle.json](../data/processed/timely_macro_evidence_20260920_v1/bundle.json)。其中包含 24 份解析结果、19 个按时点选择的证据包、原始抓取路径及哈希。解析调试阶段的失败报告单独保留，未进入模型实验。配置见 [timely_macro_evidence_v1.json](../configs/timely_macro_evidence_v1.json)。

## 配对预测实验

配置：[timely_evidence_forecast_pilot_v1.json](../configs/timely_evidence_forecast_pilot_v1.json)。模型仍为同一 Qwen3-8B-Base，Research → Forecast 使用相同 LangGraph、提示、种子、解码和格式修复设置。无 SFT、LoRA 或 RL 更新。

38 组旧对照来自已审计的前一轮原始输出：原证据＋市场价 19 条，原证据隐藏市场价 19 条。复用前核对完整输入、方法配置、来源哈希和原始响应重放；不是重新生成这 38 条。新增 19 条是在原补充证据组的基础上加入本轮时间匹配的官方指标，市场字段仍为空。同一发布 URL 的新字段替换旧窄摘要，避免将同一信息重复计为两份证据。

标签只进入评分，不进入请求；最后仍同时报告样本分数、事件等权分数、有效覆盖率和原始输出。不能仅凭概率变化或训练内得分改变为交易方法签发资格。全部输出继续作为研究数据，最终测试封存。

## 实测结果与判断

运行目录：[qwen3_8b_train_timely_evidence_pilot_v1](../runs/qwen3_8b_train_timely_evidence_pilot_v1)。19 条新增工作流全部有效，真实新增模型调用 **41 次**，包含 3 次格式修复调用；38 条旧对照直接复用，没有重复计为新增调用。新增工作流合计 435.99 秒，平均 22.95 秒；峰值分配显存约 15.89 GiB，生成速度约 26.44 token/s。加载和预检另计，完整运行约 9.14 分钟。

下表均为相同 19 个观察点、12 个训练事件，覆盖率均为 100%。误差指标越低越好；“上一轮补充旧材料”来自前一轮已冻结结果，其他行见本轮 [scores.json](../runs/qwen3_8b_train_timely_evidence_pilot_v1/scores.json)。

| 输入 / 对照 | Brier | 事件等权 Brier | Log Loss | ECE |
|---|---:|---:|---:|---:|
| 同期市场参考 | 0.014462 | 0.020185 | 0.066472 | 0.056532 |
| 原证据＋市场价（缓存） | 0.014462 | 0.020185 | 0.066472 | 0.056532 |
| 原证据、隐藏市场价（缓存） | 0.250526 | 0.250938 | 0.697400 | 0.373684 |
| 上一轮补充旧材料、隐藏市场价 | 0.276316 | 0.271354 | 0.749926 | 0.384211 |
| 本轮补充及时指标、隐藏市场价 | **0.245789** | **0.266875** | **0.679090** | **0.357895** |
| 固定 0.5 | 0.250000 | 0.250000 | 0.693147 | 0.394737 |

资料更新后，比上一轮补充旧材料的样本 Brier 改善 0.030526；8/19 条概率改变，按事件比较为 4 个改善、2 个退步、6 个不变。但与最初隐藏价格组相比，样本 Brier 仅改善 0.004737，事件等权 Brier 反而变差 0.015938（4 个事件改善、3 个退步、5 个不变）。固定 0.5 对照也出现相同的样本权重与事件权重判断差异。因此，不能把这次小幅样本改善当成稳定能力提升。

新增资料确实进入请求，最新证据年龄中位数从 41 天降至 13.23 天。新组 15/19 条最终输出引用了至少两份证据，6/19 条仍输出 0.5；资料并非全部被忽略，但引用数量本身不能验证推断质量。有价格对照仍然 19/19 条几乎复制市场概率。完整时间、输入与响应诊断见 [diagnostic.json](../runs/qwen3_8b_train_timely_evidence_pilot_v1/diagnostic.json)。

**本轮定位：时效不足是可修复的信息问题，但补齐这些指标仍未形成超过市场参考的独立预测优势。** 尚不能将剩余差距全部归因于模型能力，因为证据范围依然有限，输入摘要方式与推断方法也未分别消融。样本只有 12 个训练事件，结算标签为 17 个 No、2 个 Yes，不能据此推断全部市场的表现；历史事件还存在基础模型预训练记忆这一局限。

本轮不重跑交易、不导出信号、不签发方法资格。此前阻止未经验证信号建仓的默认准入机制保持有效；预测分数改善不等于扣费后盈利。

39 项相关测试通过；冻结源码审计从原始抓取重建证据包，复核 38 条对照来源，重放全部 57 条工作流并重新计算分数通过。审计未发现已记录证据的时间越界；官方归档版本的信任边界仍按上文保留。生成报告 SHA-256：`1de2a13137148dacece4504478430c4a872aea358ef6487d855de6e717969f0d`。

## 复现

```bash
PYTHONPATH=src:/tmp/foretellmesh-rl-deps python -m foretellmesh.timely_macro_evidence \
  --config configs/timely_macro_evidence_v1.json \
  --raw data/raw/timely_macro_evidence_20260920_v1 \
  --output data/processed/timely_macro_evidence_20260920_v1/bundle.json

PYTHONPATH=src:/tmp/foretellmesh-rl-deps HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/foretellmesh-mpl \
conda run --no-capture-output -n lab python -m foretellmesh.evidence_forecast run \
  --config configs/timely_evidence_forecast_pilot_v1.json \
  --agent-config configs/capability_agents_v1.json \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --output runs/qwen3_8b_train_timely_evidence_pilot_v1

PYTHONPATH=runs/qwen3_8b_train_timely_evidence_pilot_v1/source_snapshot:/tmp/foretellmesh-rl-deps \
python scripts/summarize_evidence_forecast.py \
  --run runs/qwen3_8b_train_timely_evidence_pilot_v1
```

证据包已存在时，构建命令重新解析原始抓取并逐项比较，不覆盖原结果。推理输出目录必须不存在。最终审计会重建新证据包、重放 57 条原始响应、核对 38 条缓存控制来源并重新计算分数。

## 可复核来源清单

以下均为对应发布正文中的当期数值，不是今天下载的修订后时间序列。

| 发布日期 | 报告月份 | 当期字段 | 官方来源 |
|---|---|---|---|
| 2023-03-10 | 2023-02 | 非农变化 +311,000；U-3 3.6% | [就业报告](https://www.bls.gov/news.release/archives/empsit_03102023.htm) |
| 2023-03-14 | 2023-02 | CPI 同比 6.0%；核心同比 5.5% | [CPI](https://www.bls.gov/news.release/archives/cpi_03142023.htm) |
| 2023-04-07 | 2023-03 | 非农变化 +236,000；U-3 3.5% | [就业报告](https://www.bls.gov/news.release/archives/empsit_04072023.htm) |
| 2023-04-12 | 2023-03 | CPI 同比 5.0%；核心同比 5.6% | [CPI](https://www.bls.gov/news.release/archives/cpi_04122023.htm) |
| 2023-05-10 | 2023-04 | CPI 同比 4.9%；核心同比 5.5% | [CPI](https://www.bls.gov/news.release/archives/cpi_05102023.htm) |
| 2023-06-02 | 2023-05 | 非农变化 +339,000；U-3 3.7% | [就业报告](https://www.bls.gov/news.release/archives/empsit_06022023.htm) |
| 2023-06-13 | 2023-05 | CPI 同比 4.0%；核心同比 5.3% | [CPI](https://www.bls.gov/news.release/archives/cpi_06132023.htm) |
| 2023-07-07 | 2023-06 | 非农变化 +209,000；U-3 3.6% | [就业报告](https://www.bls.gov/news.release/archives/empsit_07072023.htm) |
| 2023-07-12 | 2023-06 | CPI 同比 3.0%；核心同比 4.8% | [CPI](https://www.bls.gov/news.release/archives/cpi_07122023.htm) |
| 2023-10-06 | 2023-09 | 非农变化 +336,000；U-3 3.8% | [就业报告](https://www.bls.gov/news.release/archives/empsit_10062023.htm) |
| 2023-10-12 | 2023-09 | CPI 同比 3.7%；核心同比 4.1% | [CPI](https://www.bls.gov/news.release/archives/cpi_10122023.htm) |
| 2024-01-05 | 2023-12 | 非农变化 +216,000；U-3 3.7% | [就业报告](https://www.bls.gov/news.release/archives/empsit_01052024.htm) |
| 2024-01-11 | 2023-12 | CPI 同比 3.4%；核心同比 3.9% | [CPI](https://www.bls.gov/news.release/archives/cpi_01112024.htm) |
| 2024-05-03 | 2024-04 | 非农变化 +175,000；U-3 3.9% | [就业报告](https://www.bls.gov/news.release/archives/empsit_05032024.htm) |
| 2024-05-15 | 2024-04 | CPI 同比 3.4%；核心同比 3.6% | [CPI](https://www.bls.gov/news.release/archives/cpi_05152024.htm) |
| 2024-06-07 | 2024-05 | 非农变化 +272,000；U-3 4.0% | [就业报告](https://www.bls.gov/news.release/archives/empsit_06072024.htm) |
| 2024-07-05 | 2024-06 | 非农变化 +206,000；U-3 4.1% | [就业报告](https://www.bls.gov/news.release/archives/empsit_07052024.htm) |
| 2024-07-11 | 2024-06 | CPI 同比 3.0%；核心同比 3.3% | [CPI](https://www.bls.gov/news.release/archives/cpi_07112024.htm) |
| 2024-10-04 | 2024-09 | 非农变化 +254,000；U-3 4.1% | [就业报告](https://www.bls.gov/news.release/archives/empsit_10042024.htm) |
| 2024-10-10 | 2024-09 | CPI 同比 2.4%；核心同比 3.3% | [CPI](https://www.bls.gov/news.release/archives/cpi_10102024.htm) |
| 2024-11-01 | 2024-10 | 非农变化 +12,000；U-3 4.1% | [就业报告](https://www.bls.gov/news.release/archives/empsit_11012024.htm) |
| 2024-11-13 | 2024-10 | CPI 同比 2.6%；核心同比 3.3% | [CPI](https://www.bls.gov/news.release/archives/cpi_11132024.htm) |
| 2024-12-06 | 2024-11 | 非农变化 +227,000；U-3 4.2% | [就业报告](https://www.bls.gov/news.release/archives/empsit_12062024.htm) |
| 2024-12-11 | 2024-11 | CPI 同比 2.7%；核心同比 3.3% | [CPI](https://www.bls.gov/news.release/archives/cpi_12112024.htm) |
