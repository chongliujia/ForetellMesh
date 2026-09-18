# 第三版候选数据：缺失参数与任务范围

本轮完成**数据生成、确定性回放、防泄漏校验和分词预检**，没有训练第三版 LoRA，也没有模型性能结论。上一轮代码、配置与实验文档已提交为 `9fc4a8f`；本页描述此后新增的数据准备工作。

## 为什么调整

[第二版实验](research_tool_lora_v2.md)的训练损失下降，但在指定字段名的未知项任务中退步。旧数学训练题基本都提供完整参数，追加目标又多为固定自然语言句子，缺少真正无法计算的情景以及按任务改变输出范围的练习。

新数据以原始 v1 的工具、证据任务为底，加入独立生成的信息状态任务；不复制第二版追加的固定句子目标，不改变 Agent 提示版本 `grounded_json_v3` 或输出校验器。

## 样本设计

| 模型 | 成对信息条件 |
| --- | --- |
| Bayes 阳性信号 | 全部给定、缺先验、缺灵敏度、同时缺两者 |
| 两情景混合 | 全部给定、缺权重、缺高情景成功率、同时缺两者 |
| 互补事件 | 只给命中率、只给未命中率、两者均给定、只给非退化区间 |
| 有限 IID 伯努利样本 | 只给样本计数、另给总体概率、缺成功数、缺试验总数 |

每个数值情景有 `4 种信息条件 × 2 种审核范围 × 2 种命名方式 × 2 种语言 × 2 个角色 = 64` 条记录，全部留在同一个事件组。

- **审核范围**：计算审核只问列出的计算量；预测审核还包括未观察结果，采样任务另包括真实总体概率。参数齐全不等于未来结果已知，样本频率也不等于总体概率。
- **命名方式**：原字段名，或逐情景打乱映射的 `item_1` 等别名。输入提供名称到具体量的映射，答案必须遵守本次映射。
- **可确定性**：用确定性依赖规则标注直接给定、可推导和无法确定的量。概率限制在非退化内部区间；不声称这是一般符号求解器。
- **输出**：Research 引用模型规格证据；Risk 不编造数学情景以外的风险。两者的未知项根据实际可见信息与请求范围生成。

例如，Bayes 参数齐全时，计算审核的未知项是 `[]`；预测审核包含尚未观察的结果。删去先验后，先验和后验概率都无法唯一确定。使用别名时返回这两个量对应的名称，而不是背诵先前训练中的句子。

## 分组与质量检查

| 分区 | 总记录 | 总事件组 | 新记录 | 新数值情景 | 新记录中未知项为空 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 2,336 | 104 | 2,048 | 32 | 448 |
| validation | 840 | 30 | 768 | 12 | 168 |
| test | 840 | 30 | 768 | 12 | 168 |

原 v1 的 288 / 72 / 72 条输入与目标保留；固定 24 条角色验证选择保持不变。新数值情景分别使用 2024-01-15、02-15、03-15 的模拟观察时间，证据发布时间和可用时间均在观察点之前。这些是合成时间，并非历史网页取证。

生成器按**可见参数与约束**去重，忽略日期、语言、别名和隐藏参数；避免仅因未展示的先验不同，就把相同缺参题分到训练和验证。所有新分区互斥，并排除原 v1 工具情景及冻结诊断的数值情景。诊断只提供排除参数，不提供训练记录或目标；它已经影响开发方向，仍属于开发诊断，不能充当最终测试。

本次生成还另外对原始数学源中 108 个 Bayes、混合、互补数值情景做了可见参数核对，56 个新情景没有冲突；结果保存在 [原始来源重复核对](../runs/research_tool_counterfactual_preflight_v3/source_overlap.json)。这是当前产物的额外审计；后续重新配置数据时也须重做。IID 经验频率任务与原源带 Beta 先验的任务不视为等价模型。

模型提示只由 `request` 渲染。隐藏参数、答案集合、分区、样本标识和回放 case 均留在训练账本，不进入提示。读取器校验文件哈希后，从配置与来源 case 重新生成整个数据包，检查目标、时间、完整变体及选择名单；篡改内容并重算文件哈希仍会被拒绝。

最终测试仅生成和核验完整性，没有参与分词、生成或模型选择。64 个变体高度相关，不能把记录数解释为独立样本数。

## 实测预检

在 `conda lab` 中，固定 `Qwen/Qwen3-8B-Base` revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`，复用第二版训练配置的 1024 token 长度限制，调用实际训练编码器：

- train 最长 **957 token**，validation 最长 **954 token**，没有截断。
- 逐条检查仅答案参与损失、提示全部屏蔽、答案尾部保留 EOS。
- 本地模型清单及文件哈希核验通过；预检只加载 tokenizer，不把模型权重载入 GPU。
- **216 项测试：215 通过，1 项可选 GPU 测试跳过。** 本轮没有执行 GPU 训练或生成。

[分词与版本报告](../runs/research_tool_counterfactual_preflight_v3/report.json)记录数据哈希、token 编码流哈希、环境与源码身份；[数据清单](../data/processed/research_tool_counterfactual_v3/manifest.json)记录逐条件数量和来源。

## 复现与下一步

输出目录必须不存在。

```bash
conda activate lab
PYTHONPATH=src python -m foretellmesh.research_tool_counterfactual \
  --legacy-bundle data/processed/research_tool_synthetic_v1 \
  --diagnostic-cohort data/processed/uncertainty_diagnostic_v1 \
  --config configs/research_tool_dataset_v3.json \
  --output data/processed/research_tool_counterfactual_next

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m foretellmesh.capability_preflight \
  --bundle data/processed/research_tool_counterfactual_next \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --training-config configs/qwen3_8b_research_tool_sft_v2.json \
  --output runs/research_tool_counterfactual_preflight_next

PYTHONPATH=src python -m unittest discover -s tests -v
```

这里的训练配置仅用于检查现有编码接口和长度预算，不表示执行了第二版或第三版训练。完整数据池中新增未知项任务占比很高，下一轮应先固定按任务与事件组平衡的训练预算，避免工具任务被稀释；再从新验证情景冻结生成对照，连同原有工具、证据、完整流程和旧诊断一起评估。原 24 条角色验证不覆盖新增能力，训练损失不足以完成验收。正反证据判断仍是独立待改善项。

真实 Polymarket / Kalshi 的历史规则、结算证明和来源核验流程继续保留；本轮不改变真实数据就绪状态，不晋级默认 LoRA，不启动 RL。
