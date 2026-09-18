# Polymarket 历史结算与规则补证

后续用途审核和 24 次宏观发布扩充目录见 [最新范围审核](macro_evaluation_scope.md)：现有 20 条 Polymarket 观察已通过逐条 held-out 检查，整批正式评分仍未准入。以下为链上补证阶段的记录。

2026-09-19，补齐了既有 **10 个 Polymarket 合约、20 条观察记录**的最终结算与规则更新核验。整批 Polymarket / Kalshi 数据的精确结算标签从 **44/64 增至 64/64**。正式评分与 SFT 准入仍为 **0**；本轮没有运行模型、训练 LoRA 或计算真实预测分数。

| 数据检查 | 补证前 | 补证后 |
| --- | ---: | ---: |
| 精确结算标签 | 44/64 | 64/64 |
| Polymarket 缺最终结算证明 | 20 | 0 |
| Polymarket 规则更新语义待核验 | 20 | 0 |
| Kalshi 缺历史规则版本 | 44 | 44 |
| 不可用历史报价 | 39 | 39 |
| 跨基准关联事件审核待完成 | 64 | 64 |
| 独立事件组 | 2 | 2 |

数量按观察记录计，同一合约有两个观察时点，各阻塞项可重叠。20 条 Polymarket 观察来自两次 FOMC 会议，不能当成 20 个独立事件。既有准入要求仍是至少 20 个独立事件组、两个平台均有合格样本，以及冻结实际时间切分和 checkpoint manifest。

## 最终结算时间

这批市场使用 NegRisk 路径。UMA adapter 的 `QuestionResolved` 先向 NegRiskOperator 报告结果；CTF 的 `ConditionResolution` 才确立最终条件赔付。两步之间实测相隔 **28～47 秒**，不能把第一步的时间当作最终结算时间。

| FOMC 事件 | UMA 报告时间（UTC） | 最终 CTF 结算时间（UTC） | 合约数 |
| --- | --- | --- | ---: |
| 2026-07，No 分支 | 07-29 20:00:57 | 07-29 20:01:42 | 4 |
| 2026-07，Yes 分支 | 07-29 20:05:57 | 07-29 20:06:41 | 1 |
| 2026-09，部分 No 分支 | 09-16 20:02:12 | 09-16 20:02:43 | 3 |
| 2026-09，另一个 No 分支 | 09-16 20:02:15 | 09-16 20:02:43 | 1 |
| 2026-09，Yes 分支 | 09-16 21:39:40 | 09-16 21:40:27 | 1 |

`polygon_settlement_audit` 核验：

1. 从原始市场归档绑定 market ID、UMA request ID、NegRisk question ID、condition ID 和 Yes/No 顺序，并要求既有 CLOB token 映射核验通过。
2. UMA 成功回执中的报告结果必须与同笔交易的 NegRiskOperator `QuestionReported` 一致。
3. CTF 成功回执必须包含匹配的 `ConditionResolution`，oracle 地址、原生 question ID、condition ID 和 `[Yes, No]` 赔付均须吻合。
4. 交易必须在指定区块的交易列表中；日志、回执、区块的哈希、编号、交易序号与日志序号必须一致，removed 日志不接受。
5. 最终区块前一块 payout denominator 为 0，最终区块为 1，两个 payout numerators 与事件日志相同。
6. 结果与原有官方 FOMC 声明交叉核验；观察时间早于首次公开结果，首次公开结果不晚于 UMA 报告，最终结算不早于报告，并处于已确认区块范围内。

仅支持已审核的二元 NegRisk 路径。未知结果、50/50 赔付、尚未支持解码的 UMA 人工结算事件、重复日志、身份或时间冲突均阻止补证。原始 API 的 `last_update_timestamp` 只曾用于定位区块，初始化 transaction hash 不作为结算证明。

初次区块范围查询失败的响应仍留在旧归档。改用指定 `blockHash` 查询后，已归档 29 个 UMA 查询响应和 110 个最终 CTF 查询响应，均为成功请求。新增采集器可按版本化区块定位表重取 UMA 报告，并在 finalized 范围内以有限次数二分定位最终赔付；每次最多 20 个问题、1500 个请求，无交易提交。

## 规则历史的审核依据与信任范围

本次审核限定 UMA adapter `0x69c47de9d4d3dad79590d61b9e05918e03775f24`。源文件与部署代码取自其 [Polygonscan 验证页面](https://polygonscan.com/address/0x69c47De9D4D3Dad79590d61b9e05918E03775f24#code)，页面原始哈希、17 个 Solidity 文件与 settings.json 的哈希固定在 `configs/polygon_rules_review_v1.json`。

源码审核结论：

- `initialize` 根据包含 initializer 的 ancillary text 建立问题，拒绝重复初始化；`_saveQuestion` 只由初始化路径调用。后续 reset、暂停、授权与结算路径没有替换 ancillary text 或 creator 的操作。
- `BulletinBoard.postUpdate` 仅向由 question ID 与调用者地址共同索引的数组追加记录；`getUpdates` 返回整个数组。继承与内部库中没有清空、删除该数组的路径。
- 部署合约不是代理；所审核源码没有 delegatecall、自毁或升级入口。外部 token 转移使用普通 CALL，不能直接改写本合约存储。此处核验的是规则存储语义，不是完整智能合约安全审计。
- 10 个原始问题均明确指定应考虑该公告板上的创建者更新，并带匹配的 initializer 后缀。

公开 RPC 在最早初始化区块 **84,422,682** 与规则状态查询固定区块 **94,026,740** 返回相同的 **16,183 字节** runtime；也与验证页面的部署 runtime 逐字节相同：

```text
SHA-256 879273f998eb21bd7a96121ed7413d7249fc0202b12202c72ab3cdcf74f0d4dd
```

在上述源码与服务商证明的信任前提下，数组只能追加且固定区块上创建者数组为空，可以推断较早观察时点不存在创建者更新。规则审计同时要求观察时点不晚于该固定区块，而不是把一次当前状态查询无限延伸到未来。

这是**Polygonscan 源码验证证明 + 单一公开 RPC 区块/回执/状态证明**。没有声称本地独立重新编译、轻客户端验证或独立共识证明。哈希固定的是这次具体源码审核；程序不是通用 Solidity 语义证明器。换 adapter、非空更新或不同源码均需要新的解码和审核。

NegRisk 两步流程可对照官方 [NegRiskOperator](https://github.com/Polymarket/neg-risk-ctf-adapter/blob/main/src/NegRiskOperator.sol) 与 [NegRiskAdapter](https://github.com/Polymarket/neg-risk-ctf-adapter/blob/main/src/NegRiskAdapter.sol)；正式审计使用归档响应和固定事件 ABI，不在历史回放时读取当前网页作为模型证据。

## 补证发布和标签隔离

`historical_chain_supplement` 先调用原始 staging 重建检查，再执行两项链上审计。它只解除已证明的结算、规则更新阻塞，并派生新的 dataset version；跨基准语义审核、教师监督与其它源数据阻塞继续保留。

新结果在 `data/processed/historical_fomc_chain_20260919_v2`，包含 `candidates.jsonl`、独立 `outcomes.jsonl`、`review_inputs.jsonl`、`proofs.json` 和报告。原始 staging 与初版补证结果保留。所有 **64 条模型可见 payload** 都与原始 staging 相同；其中 20 条 Polymarket 观察可供输入检查，其状态仍是 review-only。结果标签与新取到的事后链上状态不会加入模型输入。

`historical_replay_gate --chain-bundle ...` 会重新从归档核验补证，不接受手工修改 ready 标志后放行。当前报告为 `not_admitted`、`model_calls=0`、`score_metrics=null`。缺教师答案只阻塞 SFT；不能用已结算 0/1 标签冒充事前概率教师答案。

## 复现与验证

以下离线命令依赖本机已归档文件；原始响应与训练产物遵循仓库既有策略，不提交到 Git。输出路径必须尚不存在。

```bash
conda activate lab
PYTHONPATH=src python -m foretellmesh.historical_chain_supplement \
  --staging data/processed/historical_fomc_replay_20260918_v2 \
  --capture data/raw/historical_fomc_capture_20260918_v1 \
  --heldout-index data/processed/forecastbench_sft_exclusion_index_v1.json \
  --bundle configs/historical_chain_bundle_v1.json \
  --output data/processed/historical_fomc_chain_replay_next

PYTHONPATH=src python -m foretellmesh.historical_replay_gate \
  --staging data/processed/historical_fomc_replay_20260918_v2 \
  --capture data/raw/historical_fomc_capture_20260918_v1 \
  --heldout-index data/processed/forecastbench_sft_exclusion_index_v1.json \
  --config configs/historical_replay_admission_v1.json \
  --chain-bundle configs/historical_chain_bundle_v1.json \
  --output runs/historical_chain_admission_next.json

# 可选公开 RPC 重采集，保留每个原始响应；不会自动替换已冻结归档。
PYTHONPATH=src python -m foretellmesh.polygon_settlement_capture \
  --oracle-plan configs/polygon_settlement_capture_v1.json \
  --output data/raw/polygon_oracle_next
PYTHONPATH=src python -m foretellmesh.polygon_settlement_capture \
  --oracle-archive data/raw/polygon_oracle_next \
  --output data/raw/polygon_ctf_next

PYTHONPATH=src python -m unittest discover -s tests
```

本轮全量测试 294 项，293 项通过、1 项可选 GPU 测试跳过。新测试覆盖完整模拟采集→离线审计→补证→准入路径，以及最终结算/UMA 时间区分、身份替换、失败回执、赔付边界、非二元结果、源码与字节码篡改、未来观察时点和原始模型输入不变。真实 10 个合约的全部归档通过离线审计；输出文件哈希也完成了重复重建核对。

原始补证位置：

- `data/raw/polygon_settlement_capture_20260918_v1`
- `data/raw/polygon_ctf_settlement_20260918_v1`
- `data/raw/polygon_proof_capture_20260918_v1`
- `data/raw/polygon_adapter_code_20260918_v2`
- `data/raw/settlement_research_20260918_v2`
- 准入与复核：`runs/historical_chain_verification_v1`

下一项数据工作是关联事件/ForecastBench 保留集审核，以及扩充独立事件与 Kalshi 历史规则证据。优先完成这些，再冻结真实模型对照集合；当前两次 FOMC 会议不足以判断多 Agent 或新 LoRA 的预测增益。
