# 历史市场数据准入与链上补证

**2026-09-19 更新：**后续已补齐 10 个 Polymarket 合约的最终 CTF 结算和固定源码审核下的规则历史核验；精确结算标签增至 64/64。现有样本的跨基准用途审核随后完成，20 条观察通过逐条 held-out 检查；Kalshi 历史规则和事件组覆盖仍未完成，正式准入仍为 0，见 [最新范围审核](macro_evaluation_scope.md)。最新结果、信任范围和复现命令见 [结算与规则补证](polygon_settlement_proofs.md)。以下保留 2026-09-18 首次核验记录；旧 `polygon_proof_audit` 单独运行仍只表示补充状态证明。

2026-09-18，本轮重建并核查了既有 Polymarket / Kalshi 历史 FOMC 数据。64 条观察记录来自 32 个合约、**2 个独立事件组**；正式回放和 SFT 准入仍均为 **0 条**。没有计算真实事件的模型排名，也没有将候选数据用于训练。

## 用途分离

新增 `historical_replay_gate`，先核对文件哈希，再从原始 HTTP 归档完整重建候选、时间、标签及阻塞项。修改 `ready_for_benchmark` 或删除阻塞项后重新计算哈希，仍不能通过重建核验。

`historical_teacher_target_missing` 只阻止 SFT，不阻止独立的预测评分。结算标签始终留在评估侧；不能把已发生事件的 0/1 结果作为事前概率教师答案。

目前评分侧阻塞项如下，数量可重叠：

| 阻塞项 | 观察记录数 |
| --- | ---: |
| Polymarket 精确结算证明缺失 | 20 |
| Polymarket 完整规则更新语义待核验 | 20 |
| Kalshi 观察时点规则版本缺失 | 44 |
| 历史报价不可用 | 39 |
| 跨基准语义重复审核待完成 | 64 |

准入失败时报告 `model_calls=0`、`score_metrics=null`，保留全部失败分母，不输出伪造的零分或完整覆盖率。

## Polymarket 补证结果

已保存公开 Polygon RPC 的 10 笔初始化交易回执、10 个 `getQuestion` 状态及对应创建者的 10 个 `getUpdates` 返回值。所有状态查询绑定 finalized 区块 **94,026,740**，区块哈希为 `0x170eb1c583744b9553ff69533f8533ad7f0ac020b82106d07996fefcfc8bfc6b`。

审计逐条核对 question ID、adapter、成功交易、创建者、初始化时间、原始 ancillary text、ABI 布局、调用参数和原始文件哈希。10 个创建者更新数组在该区块均为空，当前状态均显示 resolved。

这些是**补充的服务商状态证明**：尚未核验部署字节码与规则更新语义的完整对应，也没有独立共识验证。`resolved=true` 无法证明精确结算时间和 payout，因此本轮没有据此解除旧准入阻塞。

结算日志查询未完成：dRPC 对实际仅 361 个区块的范围返回免费计划范围错误，另一个公开 RPC 返回 HTTP 403。失败响应已保存；不能把查询失败解释为没有结算日志。没有把初始化交易哈希或 API 的 `last_update_timestamp` 冒充结算证明。

ABI 来自官方 [IUmaCtfAdapter.sol](https://github.com/Polymarket/uma-ctf-adapter/blob/main/src/interfaces/IUmaCtfAdapter.sol) 与 [BulletinBoard.sol](https://github.com/Polymarket/uma-ctf-adapter/blob/main/src/mixins/BulletinBoard.sol)；本轮读取的原文分别以 SHA-256 `07527436d57054ca5e86e9989df3c2e0c8f1721206f2f3b0afa517341d853cf0`、`6450d4d8eb0754a92b6c440a92b84dff75e2b3c57cab8739840bcd0322881a92` 保存在探测归档。仓库代码一致性不等同于部署代码证明。

Kalshi 单合约公开接口已补查，仍未取得能证明历史规则版本的记录。当前规则不能回填为观察时点已知规则。

## 下一次模型对照的冻结策略

`configs/historical_replay_admission_v1.json` 固定以下设计：

- 同一 Qwen3-8B revision，比较 single Forecast、Research→Forecast、Research→Risk→Forecast→Critic。
- Base 与 research/tool v3 两种模型路由分别比较；后者只给 Research/Risk 使用已验证研究适配器，Forecast/Critic 仍为 Base。尚无训练好的 forecast LoRA。
- 主比较使用同样本共同覆盖，并同时报告全部样本覆盖；记录 Brier、Log Loss、ECE、0.5/市场概率基线及调用、tokens、耗时、显存。
- 经验基准率只有在存在独立更早数据时才使用，不从本次评估结果计算。
- 至少 20 个独立事件组作为开发准入下限，双方平台均须有通过检查的样本。这个下限不代表统计功效保证；变体不能充当独立事件。
- 必须继续冻结实际时间/事件组切分、checkpoint manifest 和合格输入，完成基准重叠与历史知识污染审查。

当前新增的是可执行准入预检和对照策略；真实模型对照 runner 尚未在这批不合格数据上启动。下一步先取得可验证的结算回执及规则版本，并扩充不同事件组，再冻结实际数据发布。若历史规则长期无法补齐，可并行收集带原文哈希和抓取时间的前瞻快照，待事件结算后进入评分流程。

## 复现

```bash
conda activate lab
PYTHONPATH=src python -m foretellmesh.historical_replay_gate \
  --staging data/processed/historical_fomc_replay_20260918_v2 \
  --capture data/raw/historical_fomc_capture_20260918_v1 \
  --heldout-index data/processed/forecastbench_sft_exclusion_index_v1.json \
  --config configs/historical_replay_admission_v1.json \
  --output runs/historical_admission_next.json

# 可选的新联网补证：仅公开 RPC 读取，无钱包或交易提交。
PYTHONPATH=src python -m foretellmesh.polygon_capture \
  --staging data/processed/historical_fomc_replay_20260918_v2 \
  --output data/raw/polygon_updates_next

# 对本轮已归档响应进行离线审计。
PYTHONPATH=src python -m foretellmesh.polygon_proof_audit \
  --archive data/raw/polygon_proof_capture_20260918_v1 \
  --staging data/processed/historical_fomc_replay_20260918_v2 \
  --output runs/polygon_proof_audit_next.json
```

补证采集器每次最多处理 20 个已暂存问题，固定 finalized 区块，逐请求保存原始响应和哈希；非空规则更新需另行解码与历史审核，不会自动放行。

本地证据与报告位于 `data/raw/historical_proof_probe_20260918_v1`、`data/raw/polygon_proof_capture_20260918_v1`、`data/raw/polygon_log_fallback_20260918_v1`、`runs/tool_guard_verification_v1/{historical_admission,polygon_proof_audit}.json`。
