# 市场概率依赖与训练候选诊断

本实验回答两个问题：移除市场概率后，Forecast Base 会输出什么；同题随机采样能否给 Brier 奖励提供有效的概率差异。它不是新的模型训练，也不是正式保留集评测。

## 已完成结果（2026-09-19）

190 次新生成全部完成，原始响应重放和指标复算通过。3 条有价格重复对照的请求、原始响应与概率全部精确一致。

| 贪心验证，91 条 / 13 个事件组 | 有效数 | Brier ↓ | Log Loss ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| 历史市场 | 91 | 0.090394 | 0.297964 | 0.079178 |
| Base 有价格（缓存对照） | 91 | 0.090394 | 0.297964 | 0.079178 |
| Base 隐藏价格 | 91 | 0.261538 | 0.717847 | 0.284615 |
| 常数 0.5 | 91 | 0.250000 | 0.693147 | 0.280220 |

隐藏价格后 63/91 条输出 0.5；另外只出现 0.3、0.35、0.65、0.75。没有取得超越市场的分数，整体还略差于常数 0.5。提示和解析器没有缺价时自动返回 0.5 的规则。

| 训练采样：12 输入 × 4 候选 | 有价格 | 无价格 |
|---|---:|---:|
| 完整契约有效数 | 39/48 | 40/48 |
| 有效候选间存在奖励差异的输入 | 7/12 | 12/12 |
| 有效概率范围大于 0.01 的输入 | 5/12 | 12/12 |
| 仅格式惩罚带来奖励差异的输入 | 3/12 | 0/12 |

有价格的 39 个有效候选中，28 个仍在市场概率 `1e-6` 以内。失败共 17 个：13 个事件文本复制不一致，4 个非完整可解析 JSON。部分失败只漏掉 `q: ` 前缀，部分缩短为标题、丢失规则；本轮全部按失败计入覆盖率，没有事后补字段。采样多样性存在，但不能等同于预测信息或可盈利信号。

生成约 43.4 分钟，峰值分配显存约 15.77 GiB。报告 SHA-256：`0d8850231cf2b7f5c37e8bc64cec6739c14a6501a449d969da944501c009097d`。产物：[报告](../runs/qwen3_8b_forecast_price_rollouts_v1/report.json)、[审计](../runs/qwen3_8b_forecast_price_rollouts_v1/audit.json)、[原始输出](../runs/qwen3_8b_forecast_price_rollouts_v1/results.jsonl)、[对照图](../runs/qwen3_8b_forecast_price_rollouts_v1/forecast_diagnostic.png)。

用户随后明确优先做 **100 美元原始 Base 模拟交易，再判断微调必要性**，因此不依据本诊断直接启动 SFT 或 RL。下一条工作线见 [离线模拟交易](offline_paper_trading.md)。

## 冻结设计

- 使用 `pma_macro_research_20260919_v1`，数据报告 SHA-256 为 `5b0f109cf0a1ed00ea25bdd9bbbebb7658215f8f3409c45d132714086537264e`。
- 全部 91 条 validation、13 个事件组，仅将输入 `market` 改为 `null`，题目、合约规则、证据、观测时间和提示保持一致。使用贪心解码。
- 有价格对照复用上一轮全部 91 条 `single_base` 原始输出，其报告 SHA-256 为 `ca6f26c280fe4dff40cf474ab1d201702281a78a3191ac718e5389e147995fff`。生成前检查文件哈希与原始请求重放；生成后复算原报告。另按固定哈希选择 3 条有价格请求重新生成，检查请求、原始输出及概率一致性。缓存对照不会计入本轮新生成耗时。
- 训练候选仅来自 train 的 6 个事件组。每组按 `hash(seed, sample_id)` 排序，优先选择不同合约，再补剩余观测；最多取 2 条。实际选中 12 条、12 个合约，没有按结算结果或模型表现筛选。
- 每条训练输入分别有价格、无价格，各采样 4 次；温度 0.8、top-p 0.95、top-k 50。两种价格条件使用相同的逐候选种子；这不保证其 token 轨迹相同。
- 合计新生成 190 次：3 次重复对照 + 91 次价格消融 + 96 个训练候选。一次请求一个候选，单基座串行推理，最多 768 输出 tokens、4096 上下文、零重试，禁止截断。
- 基座固定 `Qwen/Qwen3-8B-Base` revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`。普通 BF16、SDPA。为保持与缓存对照的运行路径一致，仍加载上一轮 PEFT 包装，但每次 Forecast 请求均禁用适配器，所有参数冻结。
- 最终 test 文件不打开、不生成、不评分。训练与验证标签在新生成全部完成后才解析评分；请求中只有经过校验的历史输入。

完整计划、任务清单、逐候选种子、预检 token 长度、代码快照与数据哈希在生成前冻结。配置见 [forecast_diagnostic_v1.json](../configs/forecast_diagnostic_v1.json)。

## 怎样解读奖励差异

采用现有 `binary_brier_v1`：有效预测为 `-(p-y)^2`，无效输出为 -1。奖励不进入生成请求，也不拿实现后的结算 0/1 伪造事前概率教师答案。

每个输入的四个候选分别统计：

1. 完整契约有效率；不从文本中截取概率、不事后补字段。
2. 有效概率的范围、总体标准差、不同值数量。
3. **仅有效候选**的奖励范围和总体标准差。
4. 包括无效输出惩罚的奖励差异，以及是否只有格式错误带来差异。

浮点差异阈值固定为 `1e-6`，另报告概率范围超过 `0.01` 的组数。它们是诊断口径，不是效果提升或生产 RL 的准入证明。如果所有候选都复制市场，或概率相同，其预测奖励无法提供有意义的组内区分。格式错误制造的差异只能说明格式有待修复。候选差异较大同样不等于预测正确；不能按实现后的奖励挑选最佳候选再报告为预测成绩。

价格消融按完整覆盖率、共同有效观测及事件组等权指标报告 Brier、Log Loss 和 ECE，附市场、0.5 与训练事件等权经验概率基线。多合约、多观测相关，91 条观测不是 91 个独立事件。

## 复现

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
conda run --no-capture-output -n lab python -m foretellmesh.forecast_diagnostic run \
  --dataset data/processed/pma_macro_research_20260919_v1 \
  --config configs/forecast_diagnostic_v1.json \
  --control-run runs/qwen3_8b_pma_macro_development_v1 \
  --agent-config configs/capability_agents_v1.json \
  --training-run runs/qwen3_8b_research_tool_sft_r16_v3 \
  --bundle data/processed/research_tool_counterfactual_v3 \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --output runs/qwen3_8b_forecast_price_rollouts_v1

PYTHONPATH=runs/qwen3_8b_forecast_price_rollouts_v1/source_snapshot \
conda run --no-capture-output -n lab python -m foretellmesh.forecast_diagnostic audit \
  --run runs/qwen3_8b_forecast_price_rollouts_v1

PYTHONPATH=runs/qwen3_8b_forecast_price_rollouts_v1/source_snapshot \
MPLCONFIGDIR=/tmp/foretellmesh-matplotlib XDG_CACHE_HOME=/tmp/foretellmesh-cache \
conda run --no-capture-output -n lab python scripts/summarize_forecast_diagnostic.py \
  --run runs/qwen3_8b_forecast_price_rollouts_v1
```

输出目录必须不存在。`audit` 从原始响应重新执行输入变换、路由、严格解析和评分，检查缓存对照以及当前实验文件的哈希。它验证保存记录和指标的一致性，不等于在 GPU 上重新生成全部候选。

## 解释边界与后续选择

当前历史输入主要是上一期官方发布。隐藏市场概率同时移除了一个很强的信息来源，因此下降不单独证明模型推理失败；保留价格时照抄也不证明已经学会分析。底模预训练可能包含历史结果，本实验不能证明前瞻预测能力。

本轮另对上一轮原始记录做了输入与研究输出核查：[input_and_research_audit.json](../runs/qwen3_8b_forecast_price_rollouts_v1/input_and_research_audit.json)。全部 91 条验证输入都只有一份证据，发布时间距离观测点 20～48 天。这说明覆盖范围窄，不说明这些历史官方发布不可信。在 88 条两组共同成功的流程中，研究 LoRA 相比 Base 有 87 条只改变了 `unknowns`，其余 1 条完全相同；证据选择改变为 0、最终概率改变为 0。适配器没有在这批输入上给下游增加新的证据。

若未来模拟结果证实需要训练，优先评估单个 `forecast_lora`，不把研究 LoRA 的合成任务提升当作预测能力提升。接口修正、事前证据和工具改进先与原始 Base 对照；行为预热和一步反向传播资源测试均作为后续可选项。预热应对齐实际 Agent 提示、使用有明确数学答案和事前证据的合成任务，不能把真实结算标签当成概率答案，也不能训练模型无理由偏离市场。现有旧版 Forecast 合成包的提示序列化与当前运行时不同，不能直接复用旧 token 文件。

训练集目前只有 6 个宏观事件组。任何小实验都只用于检验训练机制和行为迁移，真实预测增益仍需要更多独立事件和丰富的事前信息验证。CPT 不是上述小实验的前置条件；GRPO 训练器和资源可行性不能由本轮推理结果代替验证。

训练分区的 46 条观测中，38 条来自 4 次 FOMC 发布，6 条来自 1 次 CPI 发布，2 条来自 1 次失业率发布。不能把重复观测视为增加了独立结算事件。Brier 在期望意义下鼓励真实概率：若事件概率是 q，则期望奖励为 `-(p-q)^2-q(1-q)`；但只在一个已知结算事件上反复优化，会偏好 p 接近该次的 0/1 结果。采用 proper scoring rule 不能替代事件数量、时间切分和泛化验证。
