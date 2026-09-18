# 第三版 LoRA：平衡任务的反事实训练

本轮采用原基座 `Qwen/Qwen3-8B-Base`（revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`），在 `conda lab` 与单张 RTX 3090 上训练新的 `research_tool_lora`，检验缺参判断和任务约束遵守，同时回归工具、证据和完整流程。所有输入均为合成开发数据，不能据此认定真实预测市场能力提升。

## 冻结的训练与评估计划

训练启动前保存了 [实验计划](../runs/research_tool_experiment_plan_v3/plan.json)、训练样本 ID、新增验证样本 ID 和配置副本。源码、配置、数据和名单均以哈希绑定；选样不查看目标、损失或模型回答。

训练数据来自 [第三版候选数据池](research_tool_counterfactual_v3.md)，固定取 **544 条**：

| 任务 | 训练记录 |
| --- | ---: |
| 原直接工具调用 | 96 |
| 原预构造反馈工具调用 | 96 |
| 原 Research 证据选择 | 48 |
| 原 Risk 证据提取 | 48 |
| 新 Research 信息状态任务 | 128 |
| 新 Risk 信息状态任务 | 128 |

新任务包含 32 个数值情景，每组固定一个语言、角色和命名方式，保留四种信息条件、两种审核范围，共 8 条。每个任务家族的八组恰好覆盖全部语言 × 角色 × 命名组合；旧任务全部保留，不重复过采样。整体有 104 个事件组，新增任务约占记录数的 47.1%。所有对应变体仍继承来源的事件分区。

训练从 Base 初始化全新适配器，不继续第二版权重。配置见 [第三版训练配置](../configs/qwen3_8b_research_tool_sft_v3.json)：普通 BF16 LoRA、FP32 适配器、r16 / alpha32 / dropout0.05、七类投影模块、AdamW、学习率 1e-4、micro batch 1、梯度累积 16、序列上限 1024、两轮共 68 次更新。仅答案参与损失。validation 的 840 条只报告损失，固定选择最后一轮 checkpoint，不选最低验证损失。

第一版共 36 次更新、第二版 54 次，本轮 68 次。数据组成和优化预算均改变，因此版本间比较不能隔离“反事实样本本身”的因果贡献；若要判断数据方法优劣，还需要匹配更新或 token 预算的消融。

预检在真实 tokenizer 上完成：544 条训练样本最长 957 token，840 条验证最长 954 token；无截断，编码与训练共享实现。最终测试没有参与分词、生成或模型选择。[本轮预检报告](../runs/research_tool_balanced_preflight_v3/report.json)。

## 生成对照

配置：[第四轮三组对照](../configs/research_tool_evaluation_v4.json)。三组为 Base、第一版 LoRA、第三版候选；第二版已暴露退步，本轮不将其作为主要旧 checkpoint 对照。三个组统一使用 `grounded_json_v3`、384 token 生成上限、相同 JSON 解析与严格角色校验。

| 项目 | 每组任务数 | 三组任务数 |
| --- | ---: | ---: |
| 原工具与证据任务 | 24 | 72 |
| 原四角色完整流程 | 12 | 36 |
| 新信息状态验证 | 128 | 384 |
| 冻结的旧未知项诊断 | 32 | 96 |
| 合计 | 196 | 588 |

新信息状态验证来自八个未参加训练的数值情景，每个家族取字典序前两组：一组英文、一组中文，均覆盖四种信息条件、两个角色和两个范围。字段名 / 别名与角色在两种语言间对调，整体平衡，但同一数值组内并未完整交叉所有语言和命名组合。128 条是相关变体，不是 128 个独立事件；模板和任务家族与训练共享，测的是数值情景迁移，而非未见任务家族泛化。

信息状态评分要求：角色 schema 有效、未知名称集合精确相同、Research 引用正确规格证据或 Risk 不编造额外风险。另列未知项正确、词表正确、把已知量误报未知、遗漏未知项，以及 64 组范围对照和 32 组四条件集合的全部正确数。失效输出记错，不从分母剔除。这是有限任务契约评分，不是开放文本真实性评分。

完整流程仍仅让 Research / Risk 使用对应适配器，Forecast / Critic 使用 Base。整个对照只驻留一个基座、两个适配器，每次激活一个；不叠加 LoRA。工具反馈项是预先构造的反馈任务，不能解释为真实失败后的自适应恢复率。

预先规定：改进未知项任务的同时，应保留旧版工具、证据、完整流程覆盖和数学概率报告精度；同时公开 Brier、Log Loss、ECE、格式率、延迟、显存和生成速度。本轮不自动晋级默认模型或开始 RL。

## 结果

训练已经完成，适配器已保存，基座梯度检查通过。可训练参数为 43,646,976，冻结参数为 8,190,735,360。68 次优化更新用时 **916.77 秒**，不包含三次验证、模型校验、加载及保存。

| 轮次 | 同一 840 条验证记录上的答案 token NLL |
| --- | ---: |
| 训练前 | 0.355114 |
| 第一轮后 | 0.004645 |
| 第二轮后 | 0.000927 |

峰值分配显存 **18.10 GiB**，峰值保留显存 **22.37 GiB**。日志出现一次申请 583,008,256 bytes 失败的分配器 OOM 警告，缓存重试后继续；最终 `allocator_retries=1`、`allocator_ooms=0`，没有未恢复的 OOM，训练正常完成。不能将重试成功解释为显存余量充裕。PEFT 加载时将基座名称改成了本地路径，保存前已恢复官方模型 ID 和固定 revision，评估入口再核验权重配置与哈希。

[训练报告与候选权重目录](../runs/qwen3_8b_research_tool_sft_r16_v3/report.json)。损失是同一验证集合上的教师强制结果，不能替代生成能力评估。

588 项生成对照和原始输出重放均已完成。下表中的旧版为第一版 LoRA，新版为第三版；所有版本均使用同一评估提示。

| 能力检查 | Base | 第一版 | 第三版 |
| --- | ---: | ---: | ---: |
| 原直接工具调用 | 4/8 | 8/8 | 8/8 |
| 原反馈工具调用 | 4/8 | 8/8 | 8/8 |
| Research 证据集合 | 0/4 | 4/4 | 4/4 |
| Risk 原句与来源配对 | 0/4 | 4/4 | 4/4 |
| 新信息状态任务正确 | 23/128 | 21/128 | **127/128** |
| 新任务 schema 有效 | 128/128 | 128/128 | 128/128 |
| 新任务名称词表有效 | 98/128 | 57/128 | 128/128 |
| 两种审核范围均正确 | 7/64 | 3/64 | **63/64** |
| 四种信息条件均正确 | 0/32 | 0/32 | **31/32** |
| 新任务：字段名 | 9/64 | 16/64 | 63/64 |
| 新任务：别名 | 14/64 | 5/64 | 64/64 |
| 冻结旧诊断正确 | 16/32 | 10/32 | **18/32** |
| 冻结旧诊断 schema 有效 | 32/32 | 22/32 | 29/32 |

新任务唯一错误来自中文 Risk 的完整混合概率情景。输入给出全部参数和 `success_probability = weight * high_rate + (1 - weight) * low_rate`，第三版仍将可以推导的 `success_probability` 与尚未发生的 `future_outcome` 一起列为未知。不能将 127/128 解读为所有反事实条件已掌握。

旧诊断只有四个独立数值情景，32 条包含相关的语言、角色和条件变体。其总分高于第一版，但仅比 Base 多两条，而且部分条件发生退步：

| 冻结诊断条件（每项 4 条） | Base | 第一版 | 第三版 |
| --- | ---: | ---: | ---: |
| 混合模型，参数齐全 | 1 | 2 | **0** |
| 混合模型，缺少权重 | 3 | 0 | 4 |
| Bayes，参数齐全 | 2 | 0 | 2 |
| Bayes，缺少先验 | 0 | 0 | 2 |
| 互补关系，参数审核 | 3 | 1 | 4 |
| 互补关系，未来结果审核 | 0 | 2 | 4 |
| 抽样，给出样本计数 | 4 | 2 | **0** |
| 抽样，另给总体概率 | 3 | 3 | **2** |

第三版在旧诊断上共有 14 条失败：11 条格式有效但多报了可推导的量，包括四条混合概率、六条经验频率和一条后验概率；另外三条 Research 输出违反 schema。后者将 `posterior_probability`、`future_outcome` 等名称放到顶层字段，其中两条还漏掉 `unknowns` 数组。它们是有效 JSON，但不是有效的角色输出；评分没有删除多余字段或自动补全。第三版这些任务没有碰到生成长度上限。

新验证共享训练模板，旧诊断在措辞、结构和数值上都不同，当前对照不能单独归因于措辞。结果说明在训练模板上的数值迁移明显改善，跨表达方式的稳定性仍未建立。下一项验证应固定同一数值情景和答案，只改变叙述方式、显式公式与字段呈现，形成成对诊断；在此之前不继续扩大训练或启动 RL。

### 完整流程与资源

| 指标 | Base | 第一版 | 第三版 |
| --- | ---: | ---: | ---: |
| 完整流程覆盖 / 首轮完成 | 12/12 | 12/12 | 12/12 |
| 相对给定数学概率的 MAE | 0.00222225 | 0 | 0 |
| Brier Score | 0.210187 | 0.209357 | 0.209357 |
| Log Loss | 0.633072 | 0.630340 | 0.630340 |
| ECE（5 个等宽分箱） | 0.311689 | 0.280578 | 0.280578 |
| 平均完整流程耗时（秒） | 16.14 | 16.66 | 17.19 |
| 完整流程输出 token / 生成秒 | 27.17 | 23.15 | 22.99 |
| 完整流程峰值分配显存（GiB） | 15.84 | 15.84 | 15.84 |
| 新信息任务平均耗时（秒） | 2.95 | 3.14 | 3.21 |
| 新信息任务输出 token / 生成秒 | 27.15 | 14.86 | 14.88 |

完整流程共 12 条、六个独立事件组，中英文变体共享结果。输入已给定数学计算证据，第三版与第一版的概率和分数完全相同。恒定 0.5 基线的 Brier 为 0.25、Log Loss 为 0.693147、ECE 为 0；这个小集合的正负标签恰好各半，因此其 ECE 为零不代表更好的概率预测。校准曲线已保存在完整报告中；当前样本量不足以宣称校准改善，也没有真实市场概率基线。

第三版完整流程平均耗时比第一版多约 3.2%。这是单次串行 PEFT 实测，输出长度不完全相同，不能解释为严格的吞吐基准或 vLLM 性能。推理显存为 PyTorch 峰值分配量，不含所有驱动和桌面占用。588 项任务实际调用模型 696 次：完整流程每项四次，其余每项一次；完整流程没有触发修复调用。

### 验证与候选处置

预先规定的总量比较检查全部通过：旧工具、证据、流程覆盖和数学概率报告精度保持，两个信息状态集合的总正确数提高。但总量检查没有消除上述子项退步和三条格式失败。**第三版保留为研究候选，不晋级默认，不启动 RL。** 真实预测市场能力仍需合格历史数据上的独立评估。

测试共 223 项，222 通过、1 项可选 GPU 检查跳过；本轮训练、适配器选择和生成评估另在真实 GPU 上完成。离线审计核验了冻结名单、训练编码、数据分组与时间约束、适配器哈希和逐请求选择，并从原始输出重放全部 588 项，指标精确一致。另对此前 72 + 200 + 204 = 476 项历史输出重新评分，原有指标均未变化。最终测试仍未用于生成、调参或选择模型。

本地归档（权重、数据和 `runs/` 不随 Git 提交）：

- [完整生成评估报告](../runs/qwen3_8b_research_tool_three_arm_validation_v4/report.json)
- [离线重放审计](../runs/research_tool_verification_v3/report.json) 与 [历史指标回归](../runs/research_tool_verification_v3/historical_metric_regression.json)
- [三组指标与基线汇总](../runs/research_tool_verification_v3/summary.json)
- [全部 15 条候选失败记录](../runs/research_tool_verification_v3/failure_review.json)（新任务 1 条、旧诊断 14 条）
- [预定检查与候选处置](../runs/research_tool_verification_v3/decision.json)

## 复现

数据池和已固定的验证来源须先按各自文档构建。输出目录必须不存在；旧版对照权重来自已归档的第一版训练。

```bash
conda activate lab
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m foretellmesh.capability_training \
  --bundle data/processed/research_tool_counterfactual_v3 \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --config configs/qwen3_8b_research_tool_sft_v3.json \
  --output runs/qwen3_8b_research_tool_sft_r16_next

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m foretellmesh.capability_evaluation \
  --bundle data/processed/research_tool_counterfactual_v3 \
  --training-run runs/qwen3_8b_research_tool_sft_r16_next \
  --previous-bundle data/processed/research_tool_synthetic_v1 \
  --previous-training-run runs/qwen3_8b_research_tool_sft_r16_v1 \
  --system-cohort data/processed/agent_behavior_validation_cohort_v3 \
  --uncertainty-cohort data/processed/uncertainty_diagnostic_v1 \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --config configs/research_tool_evaluation_v4.json \
  --output runs/qwen3_8b_research_tool_three_arm_validation_next
```

已有本地产物可以直接运行离线重放，不加载 GPU 模型：

```bash
conda activate lab
PYTHONPATH=src python runs/research_tool_verification_v3/reproduce.py
```

真实 Polymarket / Kalshi 历史数据的来源、规则版本和结算证明核验继续独立进行，本实验不会提升其就绪状态。
