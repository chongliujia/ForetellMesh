# 首个研究／工具能力 LoRA

这是针对已观察到的工具参数混淆、证据选择及风险假设问题的**合成行为预热实验**。基座仍为固定 revision 的 `Qwen/Qwen3-8B-Base`，没有把真实历史事件的事后结果改写成 SFT 概率答案。Polymarket、Kalshi 的历史核验流程保持独立。

## 数据与质量判据

数据配置：[research_tool_dataset_v1.json](../configs/research_tool_dataset_v1.json)。样本包为 `data/processed/research_tool_synthetic_v1`。

| 划分 | 事件组 | 样本 | 直接工具 / 错误反馈工具 / Research / Risk |
| --- | ---: | ---: | --- |
| train | 72 | 288 | 96 / 96 / 48 / 48 |
| validation | 18 | 72 | 24 / 24 / 12 / 12 |
| test | 18 | 72 | 24 / 24 / 12 / 12 |

工具任务沿用既有数学合成集的事件分组与时间划分，移除现成计算结果，要求构造 `weighted_probability` 或 `bayes_binary` 调用。错误反馈变体只添加真实校验器产生的错误信息，与实际 Agent 修复接口一致；不向模型提供正确参数或事后结果。同一事件的中英文、直接与反馈变体均在同一划分。

证据任务使用明确标记的合成预测合约：审核后的产量是否达到门槛。模型需区分当前项目的规则、运营记录和审核信息，与其他项目记录、未经核实的评论。证据 ID 不承载正反分类标签，证据顺序确定性打乱；含风险与无已记录风险两类案例均保留。Risk 的目标是提取有来源支持的风险，不凭空引入技术故障。

所有训练答案均从版本化 case 重放获得，构建和读取时重新核对输入、目标、时间、身份及哈希。跨划分事件组、缺失语言/任务变体、改写目标和未来证据不能通过校验。模型提示只读取 `request`；目标、oracle 和划分元数据留在训练监督或评分侧。测试集只生成并做完整性检查，不参与训练、NLL 选择或模型生成。

这是较窄的合成任务族，模板可能成为捷径，不能代表开放式新闻研究能力。证据评分使用确定性的预期引用集合；风险评分使用“提取原句并正确引用”的契约，允许省略末尾句号，但不会自动判定任意改写的语义真实性。该指标不能被称为通用事实支持率。

## 训练配置

配置：[qwen3_8b_research_tool_sft_v1.json](../configs/qwen3_8b_research_tool_sft_v1.json)。使用 `conda lab`、RTX 3090，普通 BF16 LoRA，r=16、alpha=32、dropout=0.05，覆盖七类投影。基座 BF16 冻结，适配器 FP32；没有量化或基座合并。

- 两轮固定训练，micro batch 1、梯度累积 16、AdamW、学习率 `1e-4`、梯度裁剪 1.0；不根据验证分数选择训练轮次。
- 最大序列 1024，实际训练最长 800、验证最长 797 token，无截断。训练和推理使用同一 `render_agent_prompt`；只对回答及 EOS 计算监督损失。
- 每个样本的 completion loss 取均值，再在累积批次中取均值；最后不足一个批次时按实际样本数缩放。验证 NLL 按 completion token 加权，仅用于记录。
- 每步检查损失和梯度，确认只有 LoRA 参数可训练、基座没有梯度。保存候选适配器的 safetensors、优化器状态、配置、样本与分词哈希、源码快照、GPU/软件信息及显存/耗时。

当前训练入口是固定预算的小实验，不提供中途恢复命令；保存优化器状态方便审计与后续恢复功能开发。训练失败不会生成“已通过评估”的权重状态。

## 评估与路由

配置：[research_tool_evaluation_v1.json](../configs/research_tool_evaluation_v1.json)。在训练前固定 24 条验证任务，涉及 6 个事件组：每个数学工具族 2 组，证据任务 2 组且覆盖两种风险条件。分别以 Base 和候选 LoRA 生成一次回答，直接任务和预构造错误反馈任务分开报告。后者不是对真实错误链条的自适应修复率。

工具评分同时检查参数语义与计算结果。例如对调“成功率”和“情景权重”有时仍可能得到相同乘积和，但参数仍判错。Research 检查支持/反证集合；Risk 检查提取与引用。非法输出计入分母，不能只统计通过 schema 的子集。

系统回归沿用先前冻结的 12 条合成验证问题，对比完整的 Research → Risk → Forecast → Critic 流程。候选组**只有 Research 与 Risk 使用 `research_tool_lora`，Forecast 与 Critic 明确使用 Base**。通过显式 `capability_scope={"research_tool_lora"}` 指定，未训练 `forecast_lora` 不会被假装成已加载适配器；请求的研究/工具适配器缺失时仍报错。

两组在同一个驻留基座上串行执行、按问题轮换执行顺序，记录每次角色、适配器、请求、原始回答、错误、token 与延迟。对比 Brier、Log Loss、ECE、覆盖率、数学答案误差和资源成本；这些预测任务已提供数学计算证据，不能解释为真实市场预测表现。训练后的能力验证不会自动升级默认模型。

## 复现

输出目录必须不存在；保留旧结果并使用新的运行目录。

```bash
conda activate lab
PYTHONPATH=src python -m foretellmesh build-research-tool-data \
  --raw-dataset data/processed/synthetic_sft_raw_v1 \
  --split-config configs/synthetic_sft_splits_v1.json \
  --config configs/research_tool_dataset_v1.json \
  --output data/processed/research_tool_synthetic_next

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m foretellmesh.capability_training \
  --bundle data/processed/research_tool_synthetic_next \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --config configs/qwen3_8b_research_tool_sft_v1.json \
  --output runs/qwen3_8b_research_tool_sft_r16_next

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m foretellmesh.capability_evaluation \
  --bundle data/processed/research_tool_synthetic_next \
  --training-run runs/qwen3_8b_research_tool_sft_r16_next \
  --system-cohort data/processed/agent_behavior_validation_cohort_v3 \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --config configs/research_tool_evaluation_v1.json \
  --output runs/qwen3_8b_research_tool_validation_next
```

入口不会下载模型或安装训练框架。使用运行时源码快照复现时，将 `PYTHONPATH` 指向对应运行目录的 `source_snapshot`。配置中的能力注册表继续表示训练计划，不是 checkpoint 晋级注册表；候选权重的状态以训练和评估报告为准。

## 已完成的首次训练

运行目录：`runs/qwen3_8b_research_tool_sft_r16_v1`。候选权重在 `adapter/adapter_model.safetensors`，约 166.6 MiB；`adapter_config.json` 绑定原始基座和 revision。只有 **43,646,976** 个 LoRA 参数参与训练，**8,190,735,360** 个基座参数冻结，无基座梯度，没有合并权重或更改默认模型。

两轮、36 次优化更新均完成，优化步骤合计约 **473 秒**，不含校验、模型加载、验证 NLL 计算与保存。峰值分配显存 **17.74 GiB**；PyTorch 缓存分配器的峰值保留显存约 **22.24 GiB**，不能把两者混为一谈，也不能据分配量简单推断可并行训练多少任务。

| 时点 | 验证 completion token NLL |
| --- | ---: |
| 训练前 | 0.716866 |
| 第一轮后 | 0.024397 |
| 第二轮后 | 0.006761 |

保存的是预先固定的最后一轮；这些 NLL 没有参与 checkpoint 选择。测试集没有参与分词训练、loss 计算或生成。检查记录位于 `runs/research_tool_verification_v1/preflight.json`，确认 train / validation / test 事件组互斥，且训练组与原有行为基线的六个验证事件无重叠。

## 角色验证

固定的 24 条开发验证任务分别由 Base 与候选回答一次。以下正确数把无效输出算作错误；Research 和 Risk 使用上述受限、确定性的标注契约。

| 任务 | Base | 候选 LoRA |
| --- | ---: | ---: |
| 直接工具调用：参数语义正确 | 4/8 | 8/8 |
| 预构造错误反馈后的工具调用 | 4/8 | 8/8 |
| Research：预期正反证据集合 | 0/4 | 3/4 |
| Risk：风险原句与来源配对 | 0/4 | 4/4 |

候选仍有一条 Research 漏选证据，不能称为完美检索。两个数学工具族、两类风险条件、中英文及错误反馈变体共享仅 **6 个事件组**；这一小集合不足以估计开放域能力。角色分数改善还必须经过完整工作流程回归，不能据此把该适配器直接推广到所有角色或进入正式预测 RL。

## 完整流程回归与处理决定

评估运行：`runs/qwen3_8b_research_tool_validation_v1`。全部 72 项生成任务已完成，包括 48 项角色对照和 24 项完整流程对照；任务执行完成不等于所有模型回答通过校验。

| 完整流程指标 | Base | Research / Risk 使用候选 |
| --- | ---: | ---: |
| 完成数 / 请求数 | 12/12 | 11/12 |
| 无需修复即完成 | 9/12 | 9/12 |
| 共同完成的 11 条：Brier | 0.196486 | 0.196486 |
| 共同完成的 11 条：Log Loss | 0.606055 | 0.606055 |
| 共同完成的 11 条：ECE | 0.252230 | 0.252230 |
| 有效输出相对给定数学答案的平均绝对误差 | 0 | 0 |
| 全部 12 次请求平均耗时 | 20.17 秒 | 18.59 秒 |
| 全部模型调用数 | 51 | 50 |
| 输出 token / 生成秒 | 26.43 | 22.49 |

Base 全部 12 条的 Brier / Log Loss / ECE 为 0.209357 / 0.630340 / 0.280578。候选缺失了一条预测，不能把其 11 条上的较低分数解释为提升；上述相同样本比较没有显示数值增益。耗时包含失败流程，且输出长度、修复次数不同，不能据平均耗时宣称效率提升。完整流程峰值分配显存两组均约 15.6–15.7 GiB；这不包含训练优化器或 RL 采样成本。

失败样本是 `synthetic:0068d947004cf0b1e00aad76:en`。在候选 Research / Risk 输出进入下游后，仍使用 Base 的 Forecast 连续两次省略原问题末尾的参数 JSON，触发 `forecast event must exactly match the input question`。原始回答虽给出了正确的 `0.5924`，仍按接口失败记录为空预测，没有用评估端修补答案或放宽原有校验。这表明单角色改善尚未转化为可靠的流程组合。

另外，逐条检查原始输出时发现候选 Risk 在多个数学问题上把已经给出的概率或先验称为未知，例如 Bayes 英文样本中声称先验和概率未知。未来结果尚未发生，与模型输入中没有概率参数，是不同情况；现有结构校验无法发现这种语义错误。这是回归诊断，不是新增的通用事实支持率指标。

**候选保留为实验权重，不晋级默认配置，不启动正式 RL。** 下一轮先增加“已知参数、未知结果、真正缺失信息”的对照训练与独立验证，检查事件身份在跨角色传递时是否完整，再评估 Research / Risk 的使用范围。工具参数映射值得继续验证，但当前小样本成绩不足以支持广泛启用。任何接口修改或后续训练都使用新的版本和运行目录，保留本次失败记录。

## 审计与回放

训练报告、候选权重和原始生成分别保存在：

- [训练报告](../runs/qwen3_8b_research_tool_sft_r16_v1/report.json)及同目录 `adapter/`。
- [评估报告](../runs/qwen3_8b_research_tool_validation_v1/report.json)及同目录 `results.jsonl`。
- [离线复核报告](../runs/research_tool_verification_v1/report.json)及同目录 `reproduce.py`、`tests.txt`、`preflight.json`。

离线复核已验证数据、适配器、配置、源码快照及生成文件哈希；核对训练与验证事件组隔离、模型输入不含结算标签、证据不晚于观察时点，以及每次调用的适配器选择。保存的原始回答能够重放全部角色和系统结果，指标逐项一致。该回放验证评估逻辑和已保存输出，不等同于保证跨硬件重新生成的文本逐位相同。

```bash
conda activate lab
PYTHONPATH=src python runs/research_tool_verification_v1/reproduce.py
```

回归测试共 188 项，187 项通过，1 项可选 GPU 测试跳过；本次完整 8B 训练和 72 项生成对照另行实际执行。测试覆盖目标重放、分组隔离、提示监督掩码、参数语义、部分能力路由、证据契约和失败样本计分。真实预测市场数据仍走独立核验流程，本实验没有产生真实事件预测能力提升的证据。
