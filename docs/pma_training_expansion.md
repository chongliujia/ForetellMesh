# Polymarket 历史训练集扩充与分配算法对照

2026-09-20。现有下载量很大，但原来的分配实验只用了 6 个独立事件。此次从已有归档新增 6 个训练事件，训练集成为 **37 个合约、12 个事件、64 条观察**。原来的 46 条训练输入和标签逐条保留，验证与最终测试仍在原数据集，未复制、未重放，也未用于训练。

## 归档库存与实际准入

[库存配置](../configs/pma_training_inventory_v1.json)绑定已有宏观目录、成交价格、市场原始身份、父数据集和保留基准索引。盘点 906 个候选合约，其中 779 个有已重建价格。排除已有分区、受保护的同父事件、基准命中和无价格记录后，得到 141 个较早候选，分属 72 个原生父事件。这个数字是待补证库存，不是合格训练集；父事件标题关键词只帮助排查，不能证明语义独立或历史规则。

本轮提前固定 8 个旧 FOMC 父事件的全部 24 个原生档位，不按标签或收益挑选合约。[扩充配置](../configs/pma_legacy_fomc_extension_v1.json)与[范围审核](../configs/pma_legacy_scope_review_v1.json)固定选择、官方发布日期、原有分区和基准保留索引。新增内容全部早于 2025-01-01，和原验证／测试的事件身份及事件组保持隔离。

| 阶段 | 数量 |
|---|---:|
| 固定选择的事件／合约 | 8 / 24 |
| 完成规则与结算证明的合约 | 22 |
| 计划 T−7 / T−1 观察 | 48 |
| 新增合格观察 | 18 |
| 指定时点价格超过 3 小时而排除 | 26 |
| 初始化回执未取得而排除 | 4 |
| 最终新增事件／合约 | 6 / 12 |

新增事件为 2023-03、05、06、07、11 和 2024-01。2023-09、12 没有观察通过本轮全部准入检查。合约 251824、252289 在定向重试后仍缺初始化回执，失败原样保留；没有用当前描述代替历史证明。没有放宽原来的 3 小时价格新鲜度门槛。准入依赖证据完整性和价格可用性，仍会产生覆盖偏差，不能视为整个 Polymarket 的代表性样本。

产物：`data/processed/pma_training_inventory_20260920_v1/` 和 `data/processed/pma_macro_training_extension_20260920_v2/`。v1 扩充产物与三轮原始补证目录保留，后一次构建不覆盖旧记录。

## 旧适配器规则与结算审核

历史合约使用 `0x6a9d222616c90fca5754cd1333cfd9b7fb6a4f74`，与此前支持的 NegRisk 适配器结构不同。[固定源码审核](../configs/pma_legacy_rules_review_v1.json)绑定 [Polygonscan 已验证源码](https://polygonscan.com/address/0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74#code)、所有源码文件及运行代码的 SHA-256。

人工检查的范围是：初始化 ancillary 内容没有后续写入路径；公告板按创建者追加更新；无代理、delegatecall 或 selfdestruct 替换入口；reset 会改变请求状态并被排除；紧急处理状态被排除。源码中的正常结算路径直接向 Conditional Tokens 报告赔付，并在同一交易中发出正常 QuestionResolved。

程序逐合约要求：

- 原生身份、Yes/No token、完整问题与描述和初始化日志一致；不能截取当前描述中的有利片段冒充历史规则。
- 旧版 10 字段 ABI 正确；初始化时间与当前请求时间一致，无 reset、paused、紧急状态，创建者更新为空。
- 初始化块及固定 finalized 块运行代码均等于审核代码。
- 正常 QuestionResolved 和直接 CTF ConditionResolution 属于同一成功交易；condition、oracle、question 映射及二元赔付一致。
- 相邻区块赔付分母由 0 变 1，分子与事件一致；官方历史发布的结果与标签一致。
- 同一公共 RPC 的链 ID、请求参数、响应哈希、固定区块及采集时间相互绑定。

信任边界：依赖区块浏览器源码验证和每份证明的单一 RPC 供应者，并未本地重编译或独立验证链共识。初始化及最新代码相等，配合已审核的不可替换路径提供规则连续性的证据；它不证明订单簿深度、实际可成交性或基础模型无预训练污染。旧版公告没有新版特有的公告板措辞，因此不把该措辞设成旧版必要条件，但仍要求公告板无任何创建者更新。

修复了官方文本解析兼容性：旧声明的 `raise ... to` 也能解析；继续限定委员会决定句并排除异议者意见。来源均为配置中的官方历史原文，当前抓取时间仅作为采集信息，不回填新的新闻。

## 训练对照设计

[冻结配置](../configs/allocation_algorithm_comparison_v1.json)使用同一训练集、88 维价格／账户／费用输入、25 个动作和相同交易执行环境，对照 Maskable PPO 与自实现 Masked Double DQN。只训练小型分配网络；Qwen3-8B-Base 及适配器保持冻结，本轮没有新增 LLM 调用。

- 每种方法种子 7 / 17 / 27，各 131,072 个环境步，固定最后检查点；6 个模型全部训练结束后再重放。
- 两者 gamma=1，以已经扣费的净清算权益增量为奖励，无活跃度惩罚、最低交易次数或强迫交易。
- 每个完整训练 episode 按 1% / 0% / 2% 手续费轮换，费用在决策前可见；另有价格溢价。按 6 种固定成本假设重放，保留 $100 现金和固定均值回归控制组。
- PPO 保持 entropy=0.01、GAE=0.95、1024 步 rollout；Double DQN 使用合法动作内 epsilon 探索、在线网络选动作、目标网络评价，并在 bootstrap 中屏蔽非法动作。
- 相同环境步数不等于相同梯度更新数或参数数目；本轮不是完整超参数搜索，也不能归因成“所有 PPO 都不如所有 DQN”。
- Double DQN 的 replay buffer 来自策略与历史模拟器的交互。它不是直接把原生成交记录当成账户状态／动作／奖励轨迹；历史模拟仍缺完整订单簿和反事实执行证明。

扩充回放覆盖 2023-03-15 至 2024-12-17，共 15,433 个小时步、548,126 条原生成交记录。131,072 步约为 8.49 次全日历遍历；有大量等待时段，这不等于 131,072 个独立市场事件。研究信息和跨事件多样性仍有限。

Double DQN 检查点仅支持推理，不承诺恢复 optimizer／随机数／replay buffer 后继续训练。训练与复算命令及结果放在本页后续章节。

## 可复现命令

```bash
PYTHONPATH=runs/pma_allocation_algorithm_comparison_v1/source_snapshot \
  python -m foretellmesh.pma_training_extension \
  --config configs/pma_legacy_fomc_extension_v1.json \
  --capture data/raw/pma_legacy_fomc_capture_20260920_v3 \
  --review configs/pma_legacy_scope_review_v1.json \
  --output /tmp/pma_training_overlay_rebuild

PYTHONPATH=src:/tmp/foretellmesh-rl-deps python -m foretellmesh.allocation_algorithm_comparison \
  --config configs/allocation_algorithm_comparison_v1.json \
  --output runs/pma_allocation_algorithm_comparison_v1

PYTHONPATH=runs/pma_allocation_algorithm_comparison_v1/evaluation_source_snapshot:/tmp/foretellmesh-rl-deps \
  python -m foretellmesh.allocation_algorithm_comparison \
  --config runs/pma_allocation_algorithm_comparison_v1/config.json \
  --output runs/pma_allocation_algorithm_comparison_v1 --reproduce
```

输出目录必须不存在，不能覆盖旧实验。可选 RL 依赖按配置锁定；当前 `sb3-contrib==2.7.1` 位于临时依赖目录，其余依赖来自本地环境。长期复现需按配置安装相同版本。数据构建独立重跑已实现报告及全部产物字节一致，并检查旧训练输入／标签逐条保留。实验 source_snapshot 保存此次数据构建和训练代码；以后代码变化时应使用该快照复建。


## 加载兼容性修复

六次训练完成并冻结后，首次回放加载 Double DQN 时失败：Gym 的动作数量保存成 `numpy.int64`，默认 `weights_only=True` 加载器不接受该 NumPy 元数据。保留全部原始检查点、训练报告及训练代码，没有追加训练或选择不同权重。

修复仅将新检查点维度保存成 Python 整数，并在读取旧检查点时显式允许 NumPy scalar、dtype 和 Int64DType 三种类型，继续使用 `weights_only=True`。新增旧格式检查点回归用例通过。`evaluation_fix.json` 记录变更，`evaluation_source_snapshot` 保留评估代码，原 `source_snapshot` 保留训练／数据构建代码。完成脚本为 `scripts/complete_allocation_algorithm_comparison.py`；重放前仍验证所有原检查点哈希。

本轮价格输入对照与此前多 Agent 实验的市场集合、时间跨度和预算不同；不能直接把两轮资金差额归因于换算法。算法比较仅在本轮同一数据、输入和执行条件内进行。新增事件尚未生成冻结的多 Agent 研究结果，因此本轮不声称提升了语言模型或多 Agent 预测能力。


## 固定训练集回放结果

所有金额为从 $100 虚拟本金出发的期末资金；下表 PPO / Double DQN 为三个预定种子的均值。都是训练内诊断，不是样本外盈利证据。

| 成本场景（每份价格溢价 / 手续费） | PPO 均值 | Double DQN 均值 | 固定均值回归 | 现金 |
|---|---:|---:|---:|---:|
| frictionless_reference（$0.00 / 0%） | $102.37 | $99.33 | $100.21 | $100.00 |
| fee_zero（$0.01 / 0%） | $89.40 | $91.82 | $88.33 | $100.00 |
| cost_assumption（$0.01 / 1%） | $87.96 | $88.70 | $85.91 | $100.00 |
| fee_interpolation（$0.01 / 1.5%） | $85.80 | $87.13 | $85.03 | $100.00 |
| fee_high（$0.01 / 2%） | $84.09 | $85.43 | $87.77 | $100.00 |
| cost_stress（$0.03 / 2%） | $65.28 | $64.74 | $89.37 | $100.00 |

主成本为每份价格溢价 $0.01、成交金额手续费 1%。三个种子均完整保留：

| 方法 / 种子 | 期末资金 | 最大回撤 | 买入次数 | 手续费合计 | 有非等待动作可选时主动等待率 |
|---|---:|---:|---:|---:|---:|
| ppo / 7 | $91.06 | 9.48% | 151 | $3.75 | 71.29% |
| ppo / 17 | $85.61 | 14.40% | 178 | $3.41 | 0.00% |
| ppo / 27 | $87.22 | 12.99% | 161 | $3.09 | 0.37% |
| double_dqn / 7 | $88.95 | 11.82% | 163 | $5.01 | 74.98% |
| double_dqn / 17 | $89.33 | 11.95% | 140 | $3.77 | 73.37% |
| double_dqn / 27 | $87.81 | 13.33% | 137 | $3.36 | 83.02% |

本轮主成本下 Double DQN 均值比 PPO 高约 $0.74，但六个训练结果都未超过现金。无任何费用和价格溢价时，PPO 均值为 $102.37，Double DQN 为 $99.33；一旦加入价格溢价，即使手续费为零，两种方法的三个种子也全部低于 $100。不同成本输入可能改变动作，因此这些差额不能简单当成固定交易路径上的费用归因。

这次结果支持继续保留 PPO，并把 Double DQN 留作对照；不足以确立替换默认算法的理由。后续优先扩充独立事件类型、补足有历史时点的研究证据，并检验信号是否形成成本后优势。候选数据不能仅靠重复训练同一批成交变成新的独立样本。


优化器计数说明：本轮 PPO 每个种子有 21,402 个参数、4,096 次 minibatch 更新；Double DQN 在线网络有 11,481 个参数、32,257 次更新（另有同形状目标网络）。固定训练报告中 PPO 的 `gradient_updates=512` 实际来自 SB3 的 epoch 计数。原始报告保持不变，独立算术审计会明确该字段语义和正确步数；当前代码已分别记录 epochs 与 optimizer steps，并通过直接统计 Adam.step 调用次数的合成回归测试。没有因此追加历史训练。

## 完成核验

- 113 项相关测试通过；包括旧适配器语义篡改、时间／分区隔离、Double DQN 合法动作与延迟奖励、执行和奖励回归，以及优化器计数的直接验证。
- 数据集独立重建，报告及全部产物字节一致；46 条原训练输入／标签完整保留。记录：`runs/pma_training_extension_audit_v1/`。
- 从冻结评估代码和原检查点完成全部 **48 组**训练内回放复算，决策、订单、成交、资金曲线和汇总一致。记录：`runs/pma_allocation_algorithm_comparison_v1/audit.json`。
- 独立核对 **740,784 个决策点、12,409 笔成交**：合法动作、无强制交易、每笔手续费、现金、持仓余额、最大回撤、费用轮换暴露及现金对照均通过。记录：同目录 `arithmetic_audit.json`。
- 原始训练报告和检查点保持不变，Qwen / LoRA 未更新，没有真实订单，最终测试保持封存，候选不晋级默认。

算术复核命令：

```bash
PYTHONPATH=src python scripts/audit_allocation_algorithm_comparison.py \
  runs/pma_allocation_algorithm_comparison_v1
```
