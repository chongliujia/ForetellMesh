# 事前证据与市场价格依赖的配对诊断

本轮针对事件持仓基线暴露的信号问题：旧 Agent 概率几乎照搬研究时的市场价，数日后的价格变化却可能被解释为预测优势。先检验事件预测的信息来源，不调整交易阈值，不启动新的 RL 或 LoRA 训练。

## 实测结果（2026-09-20）

57/57 份真实 GPU 工作流全部完成，三组共同成功样本均为 19 条，覆盖 12 个训练事件。原始请求和响应独立重放、训练输入重建及评分复算全部通过。

| 方案 | Brier ↓ | Log Loss ↓ | ECE ↓ | 事件等权 Brier ↓ |
|---|---:|---:|---:|---:|
| 同观察点市场参考 | 0.014462 | 0.066472 | 0.056532 | 0.020185 |
| 原始证据＋市场价 | 0.014462 | 0.066472 | 0.056532 | 0.020185 |
| 原始证据、隐藏市场价 | 0.250526 | 0.697400 | 0.373684 | 0.250937 |
| 补充事前官方证据、隐藏市场价 | 0.276316 | 0.749926 | 0.384211 | 0.271354 |
| 常数 0.5 | 0.250000 | 0.693147 | 0.394737 | 0.250000 |

有价格组 19/19 条概率与市场价的差均小于 0.00001，平均绝对差约 `9.34e-9`。隐藏价格后，9/19 条输出 0.5，其余仅为 0.3、0.65、0.75。补充组仍有 7/19 条输出 0.5；更多材料并未形成超越市场的预测优势，也没有优于常数 0.5。

补充材料使 4/19 条预测改变，只涉及两次 FOMC 事件：2024-07 的事件平均 Brier 降低 0.1475，2024-12 增加 0.3925，其他 10 个事件不变。这里的变化证据很少，不应将一次变差推广为“更多证据总是有害”。12 个实际增加材料的观察点中，并非每一条都产生新的概率。

Research 和 Forecast 在原始两组均保留／引用一份证据；补充组中 8 条引用一份、8 条引用两份、3 条引用四份。引用已传递到下游，但引用数量不证明有效利用。7 组输入完全相同的隐藏价格重复对照，全部请求及原始响应逐字一致。

按固定合约 ID 选样后，19 个结算标签为 17 个 No、2 个 Yes，类别分布偏斜；市场参考在这一小样本表现很好，不能据此外推全部市场，也不能将以上训练内指标当作样本外证据。模型自报置信度和 ECE 同样受小样本限制。

本轮说明当前系统依赖市场价，而现有少量、较旧的官方证据尚不能支持独立优势。下一阶段应补充临近观察点、首次发布版本明确的经济指标和事件信息，并用有依据的数值／情景分析检验概率形成；不能强迫模型偏离市场来制造买入机会。保留现金基线和原交易策略，不晋级本轮预测。

运行共 123 次模型调用，含 9 次事件问题未逐字复制导致的格式修复；没有最终失败。工作流推理合计约 18.71 分钟，单工作流平均约 19.69 秒，峰值分配显存 15.83 GiB。单个 Base 实例、全部参数冻结、无适配器更新。31 项相关测试通过。

产物：[`scores.json`](../runs/qwen3_8b_train_evidence_forecast_pilot_v1/scores.json)、[`diagnostic.json`](../runs/qwen3_8b_train_evidence_forecast_pilot_v1/diagnostic.json)、[`audit.json`](../runs/qwen3_8b_train_evidence_forecast_pilot_v1/audit.json)。评分文件 SHA-256：`e48b998a0af40ca8fa5597b97601ae33e984e1ee045aec7e6942b9e00d926897`。

## 冻结协议

配置：[evidence_forecast_pilot_v1.json](../configs/evidence_forecast_pilot_v1.json)。仅使用已准入训练分区；12 个事件每组按字典序选第一个合约，保留它的全部准入观察点，共 19 条。选择不读取结算标签，不按此前盈利与否、价格或预测误差筛选。此小样本不能代表全部合约。

三个配对组使用完全相同的原始 Qwen3-8B-Base、Research → Forecast LangGraph 工作流、提示及解码配置：

- `original_market`：原始证据和同一观察时点的市场报价。
- `original_blind`：相同原始证据，市场字段为空。
- `refreshed_blind`：市场字段为空；从全部训练输入中去重得到的 12 份官方发布中，按历史截止补充证据。

补充规则事先固定：`published_at <= available_at <= observation_time`，新增证据发布时间在此前 120 天内；FOMC、CPI、就业报告各保留最近两份，原始证据始终保留。12 个观察点增加材料，7 个没有增加；补充后 7 条有一份、7 条有两份、5 条有四份。它只是复用已核验的本地官方证据池，不能声称获得完整新闻流，也没有把今天的信息带回历史。

输入侧检查发现，原始组最新证据年龄中位数为 41 天，补充组仍为 41 天；最短年龄从 21 天降至约 5.27 天，但更多材料常是更早的同类发布。当前 BLS 准入仅包含已审核的 CPI 同比和 U-3 失业率头条字段，不能将今天归档页面上的全部表格直接扩成历史证据：原有版本审核明确排除了可能修订的表格和整篇发布。补充更多经济指标时仍需逐项核对首次发布版本，不能绕过该限制。

模型共享单个 BF16 实例，贪心解码，最多一次格式修复；记录全部原始请求、响应、失败、耗时和显存。无市场价的组不运行市场 Quant 或 Game Theory，避免价格通过派生特征进入。三个组都使用同一两角色流程，因此本轮有价格对照也不是此前四角色交易信号的逐字复刻。

57 份工作流全部冻结后，才读取训练标签评分。失败保持缺失，报告覆盖率，并在三组共同成功的同一批样本上比较 Brier、Log Loss、ECE、校准分箱和事件等权 Brier。不同合约和重复观察不能冒充新的独立事件。

## 解释边界

隐藏价格只能检验价格依赖，不能自动创造更好的预测。增加材料的组也没有新训练，材料更多不代表新增有效信息。偏离市场、引用更多来源或语言更自信均不构成收益证据。

这是训练内诊断，不是样本外检验；预训练模型可能记忆历史事件，输入时间过滤不能排除模型记忆。概率质量与可执行利润仍分开评估。本轮不导出交易信号，不替换默认策略，不改写此前 $100 → $90.55 的事件持仓结果。新鲜度和实际推理延迟已保留在原始记录中，后续接入交易仍需独立处理信号可用与过期时刻。

## 复现

```bash
PYTHONPATH=src:/tmp/foretellmesh-rl-deps HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/foretellmesh-mpl \
conda run --no-capture-output -n lab python -m foretellmesh.evidence_forecast run \
  --config configs/evidence_forecast_pilot_v1.json \
  --agent-config configs/capability_agents_v1.json \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --output runs/qwen3_8b_train_evidence_forecast_pilot_v1

PYTHONPATH=runs/qwen3_8b_train_evidence_forecast_pilot_v1/source_snapshot:/tmp/foretellmesh-rl-deps \
conda run --no-capture-output -n lab python -m foretellmesh.evidence_forecast audit \
  --run runs/qwen3_8b_train_evidence_forecast_pilot_v1

PYTHONPATH=runs/qwen3_8b_train_evidence_forecast_pilot_v1/source_snapshot:/tmp/foretellmesh-rl-deps \
python scripts/summarize_evidence_forecast.py \
  --run runs/qwen3_8b_train_evidence_forecast_pilot_v1
```

生成要求输出目录不存在；审计从原始训练输入重建选样和证据过滤，重放全部原始响应并重新评分，不进行 GPU 再生成。
