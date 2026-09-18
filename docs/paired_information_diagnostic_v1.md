# 信息状态迁移的成对诊断

本实验针对[第三版 LoRA](research_tool_lora_v3.md)出现的差异：新模板上的信息状态任务达到 127/128，冻结旧诊断仍只有 18/32，并存在可推导量误报未知及无效角色字段。这里固定同一数值情景、角色和标准答案，系统改变输入表达，检验哪一种明确的提示干预影响回答。不训练新权重，不选择默认提示或 checkpoint。

## 冻结的设计

来源为原有 `uncertainty_diagnostic_v1` 的全部 32 条开发诊断，涉及四个独立数值情景、八种信息条件、中英文和 Research / Risk 两个角色。每条展开为完整的 2×2×2 组合，得到 256 条；Base、第一版、第三版各运行全部组合，总计 **768 次模型调用**。

| 因子 | 第一个水平 | 第二个水平 | 保持不变 |
| --- | --- | --- | --- |
| 指令措辞 | 原诊断说明 | 审核式说明，重复“只列名称、不输出计算值或额外字段” | 已有角色契约、审核范围、数学问题 |
| 公式呈现 | 原叙述隐含数学关系 | 在同一证据后补充等价恒等式 | 已知数值、缺失参数、目标答案 |
| 字段释义呈现 | 原行内 `字段名: 含义` 列表 | JSON 对象数组 `name / meaning` | 名称、释义、顺序 |

“原措辞 + 隐含公式 + 行内字段”的输入与旧诊断逐字相同，作为锚点。显式公式只包含符号关系；缺失先验、缺失权重不会被填入，未来实际结果仍未观察。采样中的公式仅将经验频率表述为已观察成功次数除以历史抽样次数，不将样本频率当作总体概率。

指令措辞的第二个水平重复了系统指令已有的约束，因此这里测量的是这一具体措辞干预，包括约束在问题中的再次出现；不能将其称为任意同义改写的纯效果。字段呈现因子只改变释义列表，不改变证据的叙述格式，也不重命名字段。未完整复制第三版训练模板，避免把多个变化误当作单个因子的贡献。

所有变体保留来源事件组和 observation time，证据发布时间、可用时间均早于 observation time。诊断来源与第三版候选数据登记的排除来源绑定；诊断及其答案不进入训练。最终测试不用于本实验。数据构建和读取会从归档来源确定性重建，重新计算哈希也不能让篡改后的答案或时间进入评估。

数据配置见 [paired_information_diagnostic_v1.json](../configs/paired_information_diagnostic_v1.json)，推理配置见 [paired_information_evaluation_v1.json](../configs/paired_information_evaluation_v1.json)。统一使用原有 `grounded_json_v3`、贪心解码、384 token 生成预算、2048 token 上下文预算、严格角色校验；不修复或丢弃错误输出。一个 BF16 Qwen3-8B 基座驻留，两个既有适配器按请求切换，不叠加。版本为 `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`，环境为 `conda lab` 和 RTX 3090。

请求顺序按样本轮换三个模型。生成前归档全部请求、token 预检、配置、源码与 checkpoint 哈希；每次生成保留原始文本、解码结果、请求、token 数、耗时和峰值显存。

## 预先规定的评分

主指标沿用旧诊断：角色 schema 有效且 `unknowns` 名称集合与标准答案精确相同。单列 schema、词表、已知量误报未知和遗漏未知项。额外报告完整角色契约：Research 正确引用规格证据且反证据为空，Risk 返回预定的空风险列表。后者是本诊断的有限输出契约，不能据此判断任意自然语言风险陈述的真实性；该附加指标不改变旧诊断主指标定义。

每个因子有每模型 **128 对**匹配比较：其余两个因子、数值情景、条件、语言和角色完全相同。方向预先固定为原措辞→审核措辞、隐含公式→显式公式、行内字段→JSON 字段，分别报告“两者正确、两者错误、由错变对、由对变错”。同时报告全部八个组合、按条件/角色/语言/事件组拆分的分数，以及一条原题的八种变体是否全部正确。总增益不替代退步计数。

256 条只对应四个独立数值情景；不能把相关变体当成独立事件做显著性推断。原诊断已用于开发方向判断，本实验用于定位当前失败，不是未见事件泛化的最终检验。这里没有新的预测概率或市场结果，因此不计算或宣称 Brier、Log Loss、ECE 的改善。

## 结果

768 次生成全部完成，离线原始输出重放及指标核验通过。96 条原题锚点的请求、原始文本和校验后输出均与上一轮逐条相同。原题锚点的 Base／第一版／第三版仍为 16/32、10/32、18/32，评估入口没有改变这一结果。

| 指标 | Base | 第一版 | 第三版 |
| --- | ---: | ---: | ---: |
| 未知项集合正确（主指标） | 127/256 | 122/256 | **166/256** |
| 角色 schema 有效 | 256/256 | 207/256 | 250/256 |
| 字段词表有效（无效 schema 计失败） | 256/256 | 187/256 | 250/256 |
| 未知项正确且满足附加角色契约 | 63/256 | 122/256 | 166/256 |
| 一条原题的八个变体全部正确 | 7/32 | 2/32 | **9/32** |
| 一条原题的八个变体 schema 均有效 | 32/32 | 13/32 | 27/32 |

Base 的 Risk 输出全部包含非空风险列表，因此其附加角色契约分数低于主指标；这里的空列表约定不能作为开放式风险分析能力结论。第三版有 90 条主指标失败，其中 79 条多报已知或可推导量、5 条遗漏真正未知项、6 条违反 schema。250 条有效输出均使用允许的字段名称，剩余问题主要发生在可推导性判断和输出结构，而非字段名称词表。

### 哪个因子有效

下表每格为“由错变对 / 由对变错”，每模型、每因子均为 128 对，其他条件相同。不能把同一事件的这些对比当作 128 个独立事件。

| 匹配变化 | Base | 第一版 | 第三版 |
| --- | ---: | ---: | ---: |
| 原措辞 → 审核式措辞 | 13 / 10 | 17 / 11 | **8 / 12** |
| 隐含公式 → 显式公式 | 6 / 17 | 21 / 9 | **39 / 3** |
| 行内字段释义 → JSON 释义 | 18 / 9 | 24 / 8 | **7 / 21** |

第三版的显式公式水平正确数为 101/128，隐含公式为 65/128，净增 36；六条 schema 失败全部发生在隐含公式水平。其余两个因子分别净减 4 和 14。因子效果依赖 checkpoint 和任务：Base 的显式公式反而净减 11；第三版的三个公式配对退步均发生在缺参情景，其中两条漏报 `weight`，一条多报已给定的 `false_positive_rate`。

完整八组合如下，每格分母均为 32：

| 指令 / 公式 / 字段释义 | Base | 第一版 | 第三版 |
| --- | ---: | ---: | ---: |
| 原措辞 / 隐含 / 行内（原题锚点） | 16 | 10 | 18 |
| 原措辞 / 隐含 / JSON | 19 | 14 | 16 |
| 原措辞 / 显式 / 行内 | 11 | 17 | **29** |
| 原措辞 / 显式 / JSON | 16 | 17 | 22 |
| 审核式 / 隐含 / 行内 | 17 | 12 | 15 |
| 审核式 / 隐含 / JSON | 17 | 19 | 16 |
| 审核式 / 显式 / 行内 | 15 | 14 | 28 |
| 审核式 / 显式 / JSON | 16 | 19 | 22 |

29/32 是在已经用于开发的同一组问题上观察到的组合分数，不能把从本表选出的最好组合当作独立验证通过。该组合仍有三条中文错误：Risk 漏报缺失权重，Research 和 Risk 各一次将经验频率多报为未知。当前没有更改默认提示。

### 不能被总分遮盖的退步

| 情景（各 32 个变体） | Base | 第一版 | 第三版 |
| --- | ---: | ---: | ---: |
| 混合模型，参数齐全 | 7 | 27 | **11** |
| 混合模型，缺少权重 | 14 | 4 | 28 |
| Bayes，参数齐全 | 25 | 12 | 20 |
| Bayes，缺少先验 | 0 | 2 | 28 |
| 互补关系，参数审核 | 21 | 4 | 32 |
| 互补关系，未来结果审核 | 4 | 22 | 26 |
| 抽样，只给样本计数 | 32 | 24 | **4** |
| 抽样，另给总体概率 | 24 | 27 | **17** |

第三版在缺参和审核范围上有收益，但“能从现有信息算出什么”仍明显不稳。多报的可推导量主要为 `empirical_frequency`、`success_probability` 和 `posterior_probability`。六条格式失败均来自 Research：三条 Bayes 参数齐全、两条 Bayes 缺少先验、一条抽样；它们是有效 JSON，却添加了未经允许的顶层字段，有些还漏掉 `unknowns`。本轮没有通过删除字段或修复提示改写这些输出。

### 资源与验证

| 指标 | Base | 第一版 | 第三版 |
| --- | ---: | ---: | ---: |
| 平均每次调用耗时（秒） | 3.05 | 3.71 | 3.23 |
| 输出 token / 生成秒 | 26.89 | 14.81 | 14.75 |
| 峰值分配显存（GiB） | 15.74 | 15.80 | 15.80 |
| 碰到 384 token 上限的调用 | 0 | 3 | 0 |

每模型均处理相同的 143,256 个输入 token；不同输出长度会影响耗时，这不是等长吞吐基准。全部生成约 42 分 39 秒；一个基座和两个适配器全程保持冻结。预检输入长度为 473–627 token，加上生成预算最多 1,011 token，无截断、无最终测试生成。PyTorch 分配显存不包含所有驱动和桌面占用。

完整测试 **233 项，232 通过、1 项可选 GPU 检查跳过**；实际 GPU 推理已完成。新增测试覆盖答案与时间不变、原题锚点一致、完整因子网格、缺参值不泄漏、重新计算哈希后的篡改拒绝、匹配对计数及严格 schema。另重算此前四轮共 **1,064 项**历史输出，原有指标全部保持一致。

本地归档：

- [冻结实验计划](../runs/qwen3_8b_paired_information_validation_v1/plan.json) 与 [完整评估报告](../runs/qwen3_8b_paired_information_validation_v1/report.json)
- [离线重放审计](../runs/paired_information_verification_v1/audit.json) 与 [96 条原题锚点比较](../runs/paired_information_verification_v1/anchor_comparison.json)
- [指标、资源与配对汇总](../runs/paired_information_verification_v1/summary.json)
- [第三版全部 90 条失败](../runs/paired_information_verification_v1/candidate_failures.json) 与 [历史指标回归](../runs/paired_information_verification_v1/historical_metric_regression.json)
- [候选处置记录](../runs/paired_information_verification_v1/decision.json)

## 处置与下一步

本轮定位到明确的表达敏感性：对第三版，显式数学依赖关系能纠正一部分判断，但重复格式要求和把字段释义换成 JSON 都不能稳定修复；计算可得性仍是主要弱点。这里没有改变证据数值的呈现方式，因此还不能断言“将原始计数或参数改成结构化键值”会产生何种效果，也不能用这四个旧情景解释第三版在全部新验证情景上的表现。

**保留第三版为研究候选，不更改默认 checkpoint 或提示，不启动 RL。** 下一步优先验证工具辅助的流程：由 Quant / 确定性工具返回计算结果、所需参数和缺参状态，再让 Research / Risk 处理证据与剩余不确定性。工具只能使用 observation time 前的真实输入，不能读标准答案、补填缺失参数或把经验频率当作真实总体概率。先在未见数值情景和多种表达上冻结对照，再决定是否继续做参数训练；本报告对应的实验未包含该改动；后续实现及固定对照见 [Quant 辅助诊断](tool_assisted_diagnostic_v1.md)。

## 复现

原始诊断、两版训练权重及数据按已有报告准备。输出目录须不存在，权重和 `runs/` 本地保留、不纳入 Git。

```bash
conda activate lab
PYTHONPATH=src python -m foretellmesh.paired_diagnostic \
  --source data/processed/uncertainty_diagnostic_v1 \
  --config configs/paired_information_diagnostic_v1.json \
  --output data/processed/paired_information_diagnostic_next

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m foretellmesh.paired_evaluation run \
  --cohort data/processed/paired_information_diagnostic_next \
  --config configs/paired_information_evaluation_v1.json \
  --agent-config configs/capability_agents_v1.json \
  --training-run runs/qwen3_8b_research_tool_sft_r16_v3 \
  --bundle data/processed/research_tool_counterfactual_v3 \
  --previous-training-run runs/qwen3_8b_research_tool_sft_r16_v1 \
  --previous-bundle data/processed/research_tool_synthetic_v1 \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --output runs/qwen3_8b_paired_information_validation_next

PYTHONPATH=src python -m foretellmesh.paired_evaluation audit \
  --run runs/qwen3_8b_paired_information_validation_next
```

离线审计不加载模型，要求使用归档实验的同一源码版本，重新验证请求、适配器身份、原始文本解析、时间约束和全部评分。真实 Polymarket / Kalshi 数据的历史规则与结算证明核验继续独立进行，此诊断不改变真实数据的就绪状态。
