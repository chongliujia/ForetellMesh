# 真实宏观开发集模型对照

本轮使用 `pma_macro_research_20260919_v1` 的全部 **91 条 validation 观测、13 次宏观发布**。同一次发布的多个合约与观测点相互关联，不能视为 91 个独立事件。训练分区和最终测试分区不用于生成请求，最终测试不评分。

数据报告 SHA-256：`5b0f109cf0a1ed00ea25bdd9bbbebb7658215f8f3409c45d132714086537264e`。

## 2026-09-19 实测结果

三组共 273 次流程运行已完成，全部原始响应重放通过，指标精确复算一致。**没有观察到超越输入市场概率的预测改善；研究 LoRA 降低了流程覆盖率。** 所有有效概率与对应市场概率之差均不超过 `4.43e-7`，主要是数字舍入。多 Agent 没有在这批输入上形成有意义的概率修正。

以下三个分数与平均耗时均使用三组**共同有效的 88 条观测、13 个事件组**；覆盖率以原定 91 条为分母。

| 组 | 有效预测 | Brier ↓ | Log Loss ↓ | ECE ↓ | 共同样本平均耗时 |
|---|---:|---:|---:|---:|---:|
| 历史市场概率 | 91/91 | 0.093453 | 0.307292 | 0.082696 | — |
| 单 Agent Base | 91/91 | 0.093453 | 0.307292 | 0.082696 | 13.65 秒 |
| 多 Agent Base | 91/91 | 0.093453 | 0.307292 | 0.082696 | 16.25 秒 |
| 多 Agent + 研究 LoRA v3 | 88/91 | 0.093453 | 0.307292 | 0.082696 | 19.13 秒 |

上述一致性是保留六位小数的结果。共同样本上，多 Agent Base 与研究 LoRA 的三个原始分数也完全相同；单 Agent 和市场仅有舍入量级差异。共同样本按事件组等权 Brier 均为 `0.082452`。完整 91 条上，市场与两组 Base 的 Brier / Log Loss / ECE 均为 `0.090394 / 0.297964 / 0.079178`（六位小数）。研究 LoRA 的自身 88 条分数不能直接与其他组的完整 91 条分数比较。

研究 LoRA 有 3 次把同一 ID 同时列入 `evidence_ids` 和 `counter_evidence_ids`，Research 阶段即失败，没有继续调用 Forecast：

- `pma:522186:2025-03-11T12:30:00Z`：2 月 CPI 同比是否至少 3.1%。
- `pma:534879:2025-05-01T12:30:00Z`：4 月失业率是否至少 4.6%。
- `pma:550434:2025-07-02T12:30:00Z`：6 月失业率是否至少 4.5%。

这是证据引用契约失败，不是 JSON 语法错误。本轮零重试，未对原始响应做去重修补。Base 的 Research 输出为 91/91 有效，研究 LoRA 为 88/91；实际执行的所有 Forecast 调用均有效。这里只检验格式、时间和引用集合，尚不能据此断言证据方向、未知项和置信度都正确。

RTX 3090、`conda lab`、普通 BF16、一个共享基座：三组峰值分配显存分别为 **15.77 / 15.79 / 15.84 GiB**，峰值保留显存 16.29 GiB。生成期间约 73.6 分钟，另有启动校验和加载时间；无 OOM、无触及生成 token 上限的调用。模型调用数分别为 91 / 182 / 179；LoRA 组较少是因为 3 次上游失败，不能解释为效率优化。

本轮说明基本输出接口可在真实合约上工作；没有证明多 Agent 或现有研究适配器提升预测能力。输入证据较薄，主要是上一期官方发布，市场概率本身已包含信息；本轮不能分离证据不足、模型推理不足和对市场输入的依赖。现有研究 LoRA 不晋级默认，最终测试继续封存，RL 尚未启动。

原始结果与审计：

- [运行报告](../runs/qwen3_8b_pma_macro_development_v1/report.json)，SHA-256：`ca6f26c280fe4dff40cf474ab1d201702281a78a3191ac718e5389e147995fff`。
- [冻结计划](../runs/qwen3_8b_pma_macro_development_v1/plan.json)、[原始请求与响应](../runs/qwen3_8b_pma_macro_development_v1/results.jsonl)、[CPU 重放审计](../runs/qwen3_8b_pma_macro_development_v1/audit.json)。
- [与市场概率的差值](../runs/qwen3_8b_pma_macro_development_v1/market_agreement.json)、[对照图 PNG](../runs/qwen3_8b_pma_macro_development_v1/development_scores.png)、[PDF](../runs/qwen3_8b_pma_macro_development_v1/development_scores.pdf)。

## 固定实验

| 组 | 流程 | 适配器 |
|---|---|---|
| `single_base` | Forecast | Base |
| `multi_base` | Research → Forecast | 全部 Base |
| `multi_research_lora` | Research → Forecast | Research 使用第三版 `research_tool_lora`；Forecast 使用 Base |

第三版候选来自 `qwen3_8b_research_tool_sft_r16_v3`，不是预测能力 LoRA。基座固定为 `Qwen/Qwen3-8B-Base` revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`。

使用普通 BF16 单驻留基座与 PEFT 请求级切换，不量化。固定贪心解码、每次请求最多 768 输出 tokens、4096 上下文、零修复重试。保留既有 `grounded_json_v3` 提示和严格完整对象/单 JSON 代码块解析；不从任意文字中截取概率，不替模型补齐字段。三个组的单次请求预算相同；多 Agent 的总调用预算更大，本轮不是相同总算力消融。每个样本轮换三组执行顺序。

所有输入使用原始历史题目与规则、观测时点前的证据和市场成交概率。Research 仅能选择已经供应的证据；本轮没有在线检索。结果日志逐次保存实际请求、所用适配器、完整原始响应、有效性、耗时、token 数与显存。

## 评分和验证

生成结束后，独立评分函数才解析标签。报告包含 Brier、Log Loss、10 个等宽分箱 ECE、完整校准曲线和覆盖率，以及按宏观事件等权平均的 Brier/Log Loss。失败观测留在覆盖率分母，分数本身只对有效预测计算。每组同时报告其有效预测对应的相同观测上的市场分数，另报告三组共同成功样本，不能直接把不同覆盖子集的分数用于排名。

基线包括历史市场概率、常数 0.5、仅用训练分区标签计算的事件等权经验基准概率。市场概率也作为模型输入，模型照抄市场并不能证明新增预测信息。

`audit` 不调用模型：逐步重放原始响应，重新生成路由、上下文筛选和请求，检查适配器、严格解析、失败路径、输入和代码快照哈希，重新计算全部指标。计划在生成前冻结，数据和代码分别留有快照。

这是开发诊断，原有正式评测准入条件保持不变。历史结果可能已进入基座预训练语料，官方日期页也不是独立首发快照；本轮不能证明前瞻预测能力。schema 有效只说明接口通过检查，不等于语义正确、证据充分或 LoRA 可以晋级默认。

## 复现

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
conda run --no-capture-output -n lab python -m foretellmesh.market_development run \
  --dataset data/processed/pma_macro_research_20260919_v1 \
  --config configs/pma_model_development_v1.json \
  --agent-config configs/capability_agents_v1.json \
  --training-run runs/qwen3_8b_research_tool_sft_r16_v3 \
  --bundle data/processed/research_tool_counterfactual_v3 \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --output runs/qwen3_8b_pma_macro_development_v1

PYTHONPATH=runs/qwen3_8b_pma_macro_development_v1/source_snapshot \
conda run --no-capture-output -n lab python -m foretellmesh.market_development audit \
  --run runs/qwen3_8b_pma_macro_development_v1
```

输出目录必须不存在，避免覆盖旧实验。原始数据、权重和 `runs/` 不提交 Git；配置、评测器、测试与本说明提交版本管理。

本轮评测入口增加 7 个专项测试，覆盖未来证据、标签混入、哈希篡改、跨分区事件重叠、缺失/重复观测、失败覆盖率、同观测市场对照及原始输出/适配器重放。全仓测试共 399 项：398 通过，1 个可选 GPU 项跳过。

## RL 连接顺序

本轮没有显示必须先补真实事件 SFT 的证据。下一步先接通训练分区上的候选采样与奖励计算，检查随机采样的输出有效率，以及概率和 Brier 奖励是否有差异。贪心输出通过不能代替随机采样检查；如果同题候选概率都照抄市场，组内优势可能为零，不能盲目进入 GRPO。

首个 RL 原型只使用训练分区的 46 条观测、6 个事件，先采用稳定的 Base 上游配置，只训练一个 `forecast_lora`，以已有 `binary_brier_v1` 为主奖励。根据采样结果决定是否需要补充独立的 Forecast 行为预热，再测一小步反向传播的显存，最后跑固定小预算实验。正式实验仍要求经过行为验证的起点；本轮没有给原始 Base 发放生产 RL 准入，也不代表研究 LoRA 可用于所有任务。GRPO 训练器尚未实现；现有 BF16 SFT 可运行不代表 GRPO 显存已验证。

真实结算 0/1 是评分/RL 标签，不作为事前概率的 SFT 教师答案。严格的金融语料继续预训练（CPT）需要另备语料，不是这条 RL 原型路线的前置条件。

可以另做输出接口消融：输入仍保留完整合约规则，由运行时绑定不变的 `event` 和 `observation_time`，模型只生成概率、置信度、证据引用和未知项，最终公共 JSON schema 保持一致。这需要独立版本和模型对照，不能把当前不合格输出事后补齐后计为成功。本轮仍要求模型完整复制事件文本，没有采用该优化。

生成结束且审计通过后，可导出科学绘图：

```bash
PYTHONPATH=runs/qwen3_8b_pma_macro_development_v1/source_snapshot \
MPLCONFIGDIR=/tmp/foretellmesh-matplotlib XDG_CACHE_HOME=/tmp/foretellmesh-cache \
conda run --no-capture-output -n lab \
  python scripts/plot_market_development.py \
  --run runs/qwen3_8b_pma_macro_development_v1
```

图仅使用三组共同有效观测比较分数，另列全体观测覆盖率；输出 PNG、PDF、市场概率差值与共同样本耗时，以及绑定报告/绘图脚本哈希的清单。
