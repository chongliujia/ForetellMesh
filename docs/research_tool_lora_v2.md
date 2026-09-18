# 第二版研究／工具候选：信息状态与三组对照

第一版 LoRA 的小规模工具任务已有改善，但在数学任务中将已给定概率称为未知。仅修改提示的 [v2 接口实验](agent_contract_v2.md) 恢复了完整流程覆盖率，却没有修复这一语义错误；新的未知项诊断还暴露了数组类型声明不足的问题。

本轮的 `grounded_json_v3` 在 v2 上明确：`unknowns` 必须是字符串数组，不能是字典或对象数组。输出校验、标签隔离、时间边界及每个角色最多一次修复保持不变。默认协议仍未切换。

## 新增训练监督

配置：[research_tool_dataset_v2.json](../configs/research_tool_dataset_v2.json)。数据包为 `data/processed/research_tool_synthetic_v2`。原有工具、错误反馈、合成市场证据与风险任务全部保留；新的 Research / Risk 任务来自原数学合成源的六类问题，各变体继承原事件组和时间划分。

- 预测视角：区分未观察到的随机结果／潜在状态，与已经给定或计算出的概率；Beta-Binomial 样本保留真实率的后验不确定性。
- 计算输入审核视角：问题明确只审核模型计算需要的输入是否齐全，不审核随机结果或潜在状态。其给定输入足够，`unknowns=[]`。
- 所有监督由版本化 case 重放。概率计算证据来自原先固定的数学 oracle，未来结果标签不进入请求。增加的是行为监督，不是真实预测市场事后概率答案。

| 划分 | 样本数 | 事件组 | 原有 / 新增 |
| --- | ---: | ---: | ---: |
| train | 432 | 88 | 288 / 144 |
| validation | 144 | 26 | 72 / 72 |
| test | 144 | 26 | 72 / 72 |

训练最长 957、验证最长 954 token，均未截断。测试集只做结构、分组和重放检查，不做 loss 或模型生成。固定的 32 条未知项诊断继续留在验证侧；额外核对其数学参数未与本轮训练的对应情景重复。诊断与训练仍共享数学任务类别，不能视为跨领域泛化评估。

## 训练和比较方法

训练配置：[qwen3_8b_research_tool_sft_v2.json](../configs/qwen3_8b_research_tool_sft_v2.json)。固定 Qwen3-8B-Base revision，从同一 Base 重新初始化 r16 LoRA；不是续训第一版，也不叠加适配器。普通 BF16、micro batch 1、累积 16、两轮、学习率 `1e-4`，最后一轮固定保存。样本增加后共有 54 次更新，第一版为 36 次；两轮一致不等于计算预算一致。

评估配置：[research_tool_evaluation_v3.json](../configs/research_tool_evaluation_v3.json)。Base、第一版、第二版全部使用同一 v3 提示、同一输入和 token 上限：24 条原有角色任务、12 条完整流程、32 条未知项诊断，每组 68 项，共 204 项。第一版加载名为 `research_tool_previous_lora`，第二版为 `research_tool_lora`；名字只是同一能力的 checkpoint 别名。每次请求只启用一个适配器，Forecast / Critic 均使用 Base。

这样可以分别检查旧候选在 v2 → v3 提示后的变化，以及相同 v3 提示下新旧权重的差异。原有角色评价的模型输入和目标逐条核对相等。比较缺失预测时报告覆盖率，并使用所有组共同完成的样本计算同样本分数，不把筛掉失败样本造成的分数降低算作提升。

本轮仍是受限合成验证。Risk 摘录契约、字段名分类与真实市场预测是不同指标，不能相互替代。权重训练完成也不会自动改变默认模型或开启 RL。

## 第二版训练实测

运行目录：`runs/qwen3_8b_research_tool_sft_r16_v2`，候选保存在 `adapter/`。54 次更新全部完成，优化步骤合计 **734.01 秒**，不包括校验、加载、验证和保存。43,646,976 个 LoRA 参数更新，基座无梯度；未合并权重。

| 时点 | 当前验证集 completion token NLL |
| --- | ---: |
| 训练前 | 0.816178 |
| 第一轮后 | 0.024135 |
| 第二轮后 | 0.006389 |

新旧验证集的组成不同，不能用两次训练的绝对 NLL 直接判定哪个模型更好。保存最后一轮是预先固定的策略，NLL 没有参与选择。

峰值分配显存 **18.10 GiB**、峰值保留显存 **22.37 GiB**。第 5 与第 6 次更新之间记录到一次约 556 MiB 的分配不足提示，缓存分配器回收缓存后继续，训练未中断；见 `runs/research_tool_verification_v2/memory_note.json`。这是恢复成功的显存压力，不能描述为完全没有 OOM 提示，也不能据分配量假设还可提高批量。

## 三组评估结果与决定

204 项生成对照已完成，三组都使用 `grounded_json_v3`。

| 角色／诊断指标 | Base | 第一版 LoRA | 第二版 LoRA |
| --- | ---: | ---: | ---: |
| 直接工具参数正确 | 4/8 | 8/8 | 8/8 |
| 预构造错误反馈工具正确 | 4/8 | 8/8 | 8/8 |
| Research 正反证据集合正确 | 0/4 | 4/4 | 3/4 |
| Risk 原句与来源配对正确 | 0/4 | 4/4 | 4/4 |
| 未知项诊断：schema 合格 | 32/32 | 22/32 | 28/32 |
| 未知项诊断：遵守字段名词表 | 32/32 | 20/32 | 2/32 |
| 未知项诊断：预期字段集合完全正确 | 16/32 | 10/32 | 1/32 |
| 诊断中触及输出 token 上限的调用 | 0 | 1 | 3 |

第二版在原数学回归中能正确表述未观察的结果、潜在状态与真实率不确定性，Risk 的未知项文字与该组参考答案逐条匹配。但新诊断要求用字段名时，它仍倾向于复制自然语言句子；还有额外生成 `success_probability` 等角色 schema 之外字段的情况。因此 **1/32 是受限任务契约得分，不能解释为只有一条自由文本事实正确**。这种不遵守明确输出要求的行为，仍会使后续解析和奖励计算失败。

Research 的退步出现在 `evidence:0a9ab39621b03eabae196b65:zh:research`：第二版将“没有报告复核积压”的审核记录列入反证，第一版在相同提示下分类正确。新增训练并未丢掉已有工具能力，却没有全面保持其他任务的表现。

| 完整流程指标，全部相同的 12 条输入 | Base | 第一版 LoRA | 第二版 LoRA |
| --- | ---: | ---: | ---: |
| 完成数，且无需修复 | 12/12 | 12/12 | 12/12 |
| Brier | 0.210187 | 0.209357 | 0.209357 |
| Log Loss | 0.633072 | 0.630340 | 0.630340 |
| ECE | 0.311689 | 0.280578 | 0.280578 |
| 相对给定数学概率的平均绝对误差 | 0.002222 | 0 | 0 |
| 每条流程平均耗时 | 16.75 秒 | 17.06 秒 | 18.71 秒 |
| 输出 token / 生成秒 | 26.17 | 22.60 | 22.09 |
| 模型调用总数 | 48 | 48 | 48 |

三组完整流程的峰值分配显存均约 **15.84 GiB**，此时同一个基座上驻留两个适配器，逐请求切换。比较用相同生成上限，实际输出长度不相同；没有声称完成匹配总 token 预算的消融。

Base 的数值偏差来自中文 Bayes 样本：Research 只传递 `setup`，漏掉 `calculation`，Forecast 报出 0.2 而非给定的约 0.173333，Critic 仍接受。两版 LoRA 保留计算证据。这里的差异反映对已给定计算结果的使用，不是真实市场预测能力；第二版相对第一版没有预测分数增益。

**决定：第二版仅保留为实验权重，不替换第一版，也不晋级默认或进入 RL。** v3 接口在本次原有回归中解决了事件参数遗漏和已给定概率误报；第一版在 v3 下也不再出现原英文 Bayes 的说法，不能将修复全部归因于追加训练。下一轮优先改善对任务约束的遵守、真正缺失参数的反事实样本和正反证据判断，扩大独立验证；不能靠继续降低模板任务的训练损失来宣告泛化提升。Base 与第一版继续作为对照组，均未获得全域部署批准。

## 结果文件与核验

- [第二版训练报告](../runs/qwen3_8b_research_tool_sft_r16_v2/report.json)，同目录 `adapter/` 保存候选权重。
- [三组评估报告](../runs/qwen3_8b_research_tool_three_arm_validation_v3/report.json)，同目录 `results.jsonl` 保存每次请求、原始回答、错误和资源记录。
- [第二版离线核验](../runs/research_tool_verification_v2/report.json)，同目录 `reproduce.py`、`tests.txt`、`token_preflight.json` 和 `memory_note.json`。
- [先行 v2 提示对照的核验](../runs/agent_contract_verification_v2/report.json)，该轮 200 项结果保持原样。

两轮共 404 项生成对照均完成离线回放，重算指标与保存报告一致。第二版核验检查新旧安全权重、数据和源码快照哈希，按事件组隔离、输入不含结果标签、证据可用时间、适配器选择、训练分词身份和独立诊断的数值情景排除。参考未知项文字匹配仅作为描述性检查，不能替代语义真实性评估。代码测试为 **202 项：201 通过、1 项可选 GPU 测试跳过**；完整 GPU 训练和生成对照已另行实际执行。

```bash
conda activate lab
PYTHONPATH=src python runs/research_tool_verification_v2/reproduce.py
PYTHONPATH=src python runs/agent_contract_verification_v2/reproduce.py
```

本轮没有将真实 Polymarket / Kalshi 暂存样本升级为训练数据，也没有使用最终测试集调参。真实市场的来源、历史规则和结算证明仍须沿独立核验流程补齐。

## 复现

输出目录必须不存在。

```bash
conda activate lab
PYTHONPATH=src python -m foretellmesh.research_tool_augmentation \
  --legacy-bundle data/processed/research_tool_synthetic_v1 \
  --raw-dataset data/processed/synthetic_sft_raw_v1 \
  --split-config configs/synthetic_sft_splits_v1.json \
  --config configs/research_tool_dataset_v2.json \
  --output data/processed/research_tool_synthetic_next

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m foretellmesh.capability_training \
  --bundle data/processed/research_tool_synthetic_next \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --config configs/qwen3_8b_research_tool_sft_v2.json \
  --output runs/qwen3_8b_research_tool_sft_r16_next

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m foretellmesh.capability_evaluation \
  --bundle data/processed/research_tool_synthetic_next \
  --training-run runs/qwen3_8b_research_tool_sft_r16_next \
  --previous-bundle data/processed/research_tool_synthetic_v1 \
  --previous-training-run runs/qwen3_8b_research_tool_sft_r16_v1 \
  --system-cohort data/processed/agent_behavior_validation_cohort_v3 \
  --uncertainty-cohort data/processed/uncertainty_diagnostic_v1 \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --config configs/research_tool_evaluation_v3.json \
  --output runs/qwen3_8b_research_tool_three_arm_validation_next
```
