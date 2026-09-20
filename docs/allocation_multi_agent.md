# 多 Agent 分析接入交易策略

本轮按用户要求，将真实 Qwen3-8B 多 Agent 输出接入 PPO 的交易决策状态。仅训练分区、模拟执行、初始现金 $100；没有每七天必须交易的约束。此前纯价格策略见 [费用变量与净收益 PPO](allocation_profit_fee.md)。

## 实测结果

已完成 **84 份真实 GPU 多 Agent 工作流、九次 PPO 训练和 84 组模拟回放**。其中 79 份工作流有效、5 份失败；总计 348 次模型调用，覆盖 17 个合约，失败全部保留为缺失信号。GPU 生成约 48 分钟，平均每份 34.23 秒，峰值分配显存 15.85 GiB，始终只加载一份原始 Base。

主成本情景假设每次买卖按成交额收取 1% 手续费，并对每份额买入加价 / 卖出减价各 $0.01。三种子结果如下：

| 策略 | 种子 7 期末资金 | 种子 17 | 种子 27 | 平均期末资金 |
| --- | ---: | ---: | ---: | ---: |
| 价格特征对照 | $92.951907 | $84.774020 | $84.709110 | $87.478346 |
| 同时点历史价格参考 | $93.451636 | $82.862531 | $84.709110 | $87.007759 |
| 多 Agent 分析 | $94.136952 | $88.705912 | $90.727226 | **$91.190030** |
| 多 Agent 权重、推理时屏蔽分析 | $94.136952 | $88.014126 | $88.533972 | $90.228350 |
| 始终持有现金 | $100 | $100 | $100 | **$100** |

多 Agent 组相对价格特征对照，三个种子分别多 $1.185045 / $3.931892 / $6.018116，均值多 **$3.711684**；相对历史价格参考组，均值多 $4.182271。主成本平均买入次数从 144 降至 129.67，平均手续费从 $2.893596 降至 $2.544403，最差种子最大回撤由 15.29% 降至 11.29%。**这是训练内少亏的证据，三个种子仍全部亏损，没有超过现金基线。**

按主情景的**相同成交时间和份额**逐笔拆分，三种子均值如下；这不是让策略在零成本下重新选单：

| 策略 | 原始价格盈亏 | 加减价损耗 | 手续费 | 净盈亏 |
| --- | ---: | ---: | ---: | ---: |
| 价格特征对照 | +$0.034496 | $9.662554 | $2.893596 | −$12.521654 |
| 历史价格参考 | +$0.269570 | $10.212944 | $3.048867 | −$12.992241 |
| 多 Agent | −$0.608645 | $5.656922 | $2.544403 | −$8.809970 |

改善主要来自价格加减价损耗减少，原始价格盈亏没有提高。不能把少亏直接解释为更准确的预测。

同一个多 Agent checkpoint 在推理时屏蔽分析，种子 17 / 27 的资金分别减少 $0.691786 / $2.193254，种子 7 不变。这个干预表明部分策略确实依赖分析字段；它有输入分布变化，不能把整个配对训练差值都解释为某一角色的因果收益。

| 费用情景 | 价格特征对照均值 | 历史价格参考均值 | 多 Agent 均值 |
| --- | ---: | ---: | ---: |
| 无手续费、无价格加减价 | $98.938450 | $99.605644 | $99.460651 |
| 0% 手续费、每份额 ±$0.01 | $89.431023 | $89.121868 | $93.579805 |
| 1% 手续费、每份额 ±$0.01 | $87.478346 | $87.007759 | $91.190030 |
| 1.5% 手续费、每份额 ±$0.01 | $85.255254 | $85.911971 | $89.783193 |
| 2% 手续费、每份额 ±$0.01 | $84.135672 | $85.144363 | $88.739378 |
| 2% 手续费、每份额 ±$0.03 | $69.863196 | $70.345603 | $79.382085 |

多 Agent 在有成本的五种情景中均比两组对照少亏；无成本下没有一致优势，且所有情景的平均资金均低于 $100。不同费用情景沿用同一批训练事件，不能当作六个独立样本外测试。

另一个关键发现：**79/79 份有效 Forecast 概率与生成时的市场概率之差都不超过 0.00001，最大差仅 0.0000046501**，基本是复制和数值舍入。不能据此声称提高了预测准确性或发现了独立概率优势。量化和博弈论观点虽被接入，但其增益仍需在新的事件上验证；本轮不晋级策略，不训练或更换 LoRA。

每个策略为 29,594 参数，训练 32,768 步、7 个完整历史周期，共 294,912 步；九次 CPU 训练合计 271.50 秒。交易阶段记录 554.91 秒，包含价格准备、训练与首轮回放，另有前置 Agent 核验及独立审计。增加回放或 Agent 调用没有增加独立事件数量。

- [完整收益、回撤、费用和所有种子](../runs/pma_allocation_multi_agent_v1/report.json)
- [训练参数、实际费率步数和权重校验](../runs/pma_allocation_multi_agent_v1/training_report.json)
- [84 组独立重放审计](../runs/pma_allocation_multi_agent_v1/audit.json)
- [信号时序、覆盖率与逐笔手续费核对](../runs/pma_allocation_multi_agent_v1/agent_checks.json)
- [相同成交的成本拆分](../runs/pma_allocation_multi_agent_v1/cost_decomposition.json)
- [实际 GPU 生成记录](../runs/qwen3_8b_base_train_allocation_agents_v1/generation_report.json)
- [原始 Agent 请求和响应](../runs/qwen3_8b_base_train_allocation_agents_v1/results.jsonl)

结果报告 SHA-256：`d47d14bc8ce712f770655eb2d3f0af37553096044a962a6e480acc20d8d73e06`。

最终 49 项回归测试通过；84 组账本、原始 Agent 响应和历史输入全部复算一致，另独立检查 **393,204 个决策点**的等待合法性、信号生成/过期时间和 **20,170 笔买卖成交**的实际手续费，84 组成本恒等式全部成立。主成本三个多 Agent 策略的候选位信号覆盖率约为 60%–70%，不是每个时点都有新分析。

训练后补充了非法动作和终态调用的提前拒绝，防止失败调用修改信号计数；当前实现已重放上述全部有效轨迹，与冻结训练源码的原始结果完全一致。冻结源码仍保留，变更与验证哈希见[验证来源记录](../runs/pma_allocation_multi_agent_v1/verification_provenance.json)。

## 冻结的实验协议

Research → market Quant → Game Theory → Forecast 使用现有 LangGraph 工作流，串行共享一个 BF16 Qwen3-8B-Base。原始模型与 LoRA 均不训练，实际更新的是交易决策小网络。配置见 [Agent 生成](../configs/allocation_agent_signals_v1.json) 和 [PPO 对照](../configs/allocation_multi_agent_v1.json)。

只读取训练分区中的 25 个合约、6 个事件组。每个历史决策点只能选择当时已经存在且价格可用的市场，按过去 24 小时成交笔数排序；每个事件组最多选择两个市场，每七天刷新研究，信号最长有效七天。预计共 84 份分析、17 个合约。这里的七天是**研究缓存刷新间隔**，不要求交易。

历史问题须与已核验的不可变合约规则一致。研究证据来自已有训练分区的证据池，`published_at` 和 `available_at` 都必须不晚于当前观察时点。价格统计由 Python / SQL 截断至该时点，不给 Agent 结算标签、未来价格或之后的新闻。此证据池并非完整历史新闻流。

每份分析在观察时点加上实际特征计算、模型推理时间后才可用。同一时点的任务计算串行排队延迟，符合单模型部署。失败或过期的分析没有概率值，不能用结算答案、较晚输出或 0.5 回填。模型本身可能记忆历史事件结果，时间过滤无法排除预训练污染。

## 配对比较

- `price_control`：88 个现有价格/账户/费用特征，加相同的 Agent 可用性与年龄元数据；分析内容全部置零。
- `market_anchor`：在相同可用性和年龄条件下，只加入生成分析时的历史市场概率及其与当前价格的差；没有 Agent 的观点、置信度和未知项字段。这用于识别额外价格参考本身的作用。
- `multi_agent`：相同结构，额外提供预测概率、相对当前价格的概率差、自报置信度、量化与博弈论观点的独热编码、未知项数量。
- 每个候选新增 16 个值，共 152 维输入。三组相同种子的初始权重逐张量相同，均训练 32,768 步；种子为 7 / 17 / 27。
- 三组均按完整周期切换 1% → 0% → 2% 手续费，手续费率提前可见。候选排序、25 个动作、执行规则、熵系数和训练预算保持相同，只比较分析内容的贡献。
- 九个 checkpoint 全部冻结后，运行六种成本场景；每种包含现金、固定均值回归、九个 PPO，以及三个多 Agent checkpoint 在推理时屏蔽内容的干预对照，共 84 组回放。

历史价格参考对照在读取任何本轮交易收益、训练任何本轮历史策略之前加入：生成阶段发现 Forecast 概率几乎复制历史市场价，因此不能把增加一个价格参考所带来的变化归因于 Agent 推理。配置在历史 PPO 训练前冻结，不按收益选样本或调参。

屏蔽内容的干预用于核验策略是否依赖这个输入通道，存在输入分布变化，不能当成独立训练的基线。主要比较仍然是配对训练的 `price_control` 与 `multi_agent`。

预测概率对应事件最终结算，**不等于未来 48 小时的交易价格预测**。本轮只把它作为可选状态，不用它伪造成交价、可实现收益或奖励。自报置信度也不是已验证的概率校准。

奖励仍为扣费后净清算权益变化，整段累计奖励等于期末现金减 $100，`gamma=1`；等待始终可选。现金/仓位上限、止损、最长持仓、冷却期和每日进场上限沿用原协议。

## 复现命令

使用 `conda lab`，已有 `/tmp/foretellmesh-rl-deps` 提供 `sb3-contrib==2.7.1`。GPU 阶段完全使用本地模型，无网络请求。

```bash
export PYTHONPATH=src:tests:/tmp/foretellmesh-rl-deps
export MPLCONFIGDIR=/tmp/foretellmesh-mpl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4

conda run --no-capture-output -n lab python -m foretellmesh.trading_agent_signals generate \
  --dataset data/processed/pma_macro_research_20260919_v1 \
  --store data/processed/pma_macro_trade_store_20260919_v4 \
  --reference runs/pma_allocation_profit_fee_v1 \
  --config configs/allocation_agent_signals_v1.json \
  --agent-config configs/market_expert_agents_v1.json \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --output runs/qwen3_8b_base_train_allocation_agents_v1

conda run --no-capture-output -n lab python -m foretellmesh.trading_agent_signals audit \
  --run runs/qwen3_8b_base_train_allocation_agents_v1

conda run --no-capture-output -n lab python -m foretellmesh.trading_rl_agents run \
  --signals runs/qwen3_8b_base_train_allocation_agents_v1 \
  --config configs/allocation_multi_agent_v1.json \
  --output runs/pma_allocation_multi_agent_v1

PYTHONPATH=runs/pma_allocation_multi_agent_v1/source_snapshot:/tmp/foretellmesh-rl-deps \
conda run --no-capture-output -n lab python -m foretellmesh.trading_rl_agents audit \
  --run runs/pma_allocation_multi_agent_v1
```

输出目录不得已存在。配置、数据与模型哈希、原始模型请求/输出、152 维观测哈希、每次决策引用的信号 ID、成交账本、训练日志和权重均保留。审计重建历史输入、重新校验全部原始 Agent 响应，再逐笔重放所有策略。

## 简单环境的学习检查

另行完成了[无优势学习检查](../runs/allocation_no_edge_learnability_v1/report.json)：四个市场的价格始终为 0.5，手续费与加减价均为正，没有任何可盈利的交易。两组各三个种子、每个 32,768 步，**六个冻结策略均学会零交易，期末现金均为 $100**，独立进程重新加载权重后复算通过。

此检查的中性 Agent 字段是显式合成夹具，**没有模型调用，也不是历史市场收益实验**。它只证明当前优化器和环境能够在简单任务中学会空仓；不证明已经能识别真实市场机会、正确处理长时序信用分配或产生样本外利润。配置见 [allocation_no_edge_check_v1.json](../configs/allocation_no_edge_check_v1.json)，检查脚本为 `scripts/check_allocation_no_edge.py`。

## 解释范围

这是一轮训练内接入与方法对照，不是样本外盈利评测。旧候选排序和动作探索偏向仍未改变，不能把这轮结果用来证明 PPO 已收敛。手续费是已知的模拟假设，历史首笔成交代理不证明盘口深度和真实可成交量；合约集合和回放终点仍有事后研究选择限制。最终测试不打开，策略不自动晋级。
