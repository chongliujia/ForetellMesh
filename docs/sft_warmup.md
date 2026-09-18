# 合成 SFT 预热数据与真实来源核验

用户已批准先构建独立合成预热集，同时保留真实数据核验。当前完成数据生成、分组切分、输出验证和本地 Qwen tokenizer 检查；没有进行正式 SFT 训练。

## 已生成的数据

固定种子 `20260918`，216 个独立参数组合，各有中英两个版本，共 432 条：训练 288、验证 72、测试 72。六类任务为补集、至少一次成功、不放回抽样、混合概率、Beta-Bernoulli 后验预测和贝叶斯信号更新。相同参数组合的语言变体归入同一个事件组，组间不跨 split。

概率由 Python `Fraction` 精确计算后保留六位小数；输入包含模拟设定和可引用的计算工具结果。目标用于学习结构、引用和读取计算结果，不要求模型替代计算器。`confidence` 表示估计的不确定性，而非事件发生概率的大小。

观测时间、证据时间和结算标签全部是显式模拟值。模拟 outcome 独立抽样，不用于计算 SFT 概率答案。各 split 使用同类模板，因此验证/测试只能检查模板内的格式与参数泛化，不能证明真实事件预测能力。

实际固定 Qwen3-8B-Base tokenizer 检查：

| split | 样本数 | 最长 token |
| --- | ---: | ---: |
| train | 288 | 815 |
| validation | 72 | 813 |
| test | 72 | 812 |

最大长度限制 2048；超长样本报错，不静默截断。损失只覆盖 completion 和 EOS，prompt 的 labels 为 `-100`。Base 模型采用版本化纯文本格式，不假设已经存在可用的 instruction chat template。

## 复现

在仓库根目录使用 `conda activate lab`。输出目录必须不存在，重复运行换新名称。

```bash
PYTHONPATH=src python -m foretellmesh generate-sft-warmup \
  --config configs/synthetic_sft_generator_v1.json \
  --output data/processed/synthetic_sft_raw_v1

PYTHONPATH=src python -m foretellmesh index-heldout \
  --data data/raw/forecastbench_v1/2026-01-04-llm.json \
  --resolutions data/raw/forecastbench_v1/2026-01-04_resolution_set.json \
  --output data/processed/forecastbench_sft_exclusion_index_v1.json

PYTHONPATH=src python -m foretellmesh build-sft \
  --data data/processed/synthetic_sft_raw_v1/records.jsonl \
  --targets data/processed/synthetic_sft_raw_v1/targets.jsonl \
  --config configs/synthetic_sft_splits_v1.json \
  --heldout-index data/processed/forecastbench_sft_exclusion_index_v1.json \
  --output data/processed/synthetic_sft_bundle_v1

PYTHONPATH=src python -m foretellmesh tokenize-sft \
  --bundle data/processed/synthetic_sft_bundle_v1 \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --config configs/sft_tokenization_qwen3_v1.json \
  --output data/processed/synthetic_sft_tokens_v1

PYTHONPATH=src python -m foretellmesh audit-prophet-sft \
  --data data/raw/prophet_arena_v1/subset_data_1200.csv \
  --output runs/prophet_sft_readiness_v1
```

`train/validation/test.jsonl` 文本文件只有 prompt/completion；标签、来源和分组保存在独立 metadata 中。原始 oracle 证明、输入/输出哈希、配置与代码版本随 bundle 保留；builder 会重算 oracle，拒绝伪造答案或不匹配的输入。

固定 ForecastBench 索引包含 2,244 个问题/期限，包括未结算候选。builder 拒绝 eval-only 来源和重叠事件；字面/词汇检查不能证明语义去重，真实事件组必须提交绑定索引哈希的审核记录。

## 真实数据不会因缺少 SFT 答案而被弃用

Prophet 当前 7,022 个候选没有可直接采用的历史预测目标，12,004 个 source entries 缺少独立时间证明，故此次进入 SFT 的真实样本为 0。这个结论只针对当前 SFT 准入要求，不是说它们不能用于后续市场研究、结算评分或 RL。

历史教师目标需保存原始预测及可追溯时间，绑定输入与目标哈希；本地文件哈希只能证明字节一致，不能单独证明历史真实性。现有 `historical_forecast` 通道只接受可验证的历史预测，不支持把今天生成的教师回答伪装成历史发布记录。回溯教师生成需单独设计来源类型和审核流程。

自建真实数据流程见 [预测市场数据集](prediction_market_dataset.md)。合成预热、真实观测和 held-out 评估三类数据分别保留。
