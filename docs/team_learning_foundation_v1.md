# 团队自主学习：首批实现与实测

本文保留基础阶段的当时状态；后续历史参数实验见 [首轮结果](team_rsi_cycle_v1.md)，当前以 [先验证效果、再可选微调](team_recursive_learning.md) 为准。

本轮已实现不按题材筛选的历史入口、团队 episode、反馈/记忆接口和候选经验导出。**尚未完成经验筛选、8B 微调和独立评估，尚无 RSI 能力提升或盈利结论。** 总体目标见 [团队递归学习协议](team_recursive_learning.md)。

## 广泛市场入口

在 408,863 条市场记录中，按 2023–2024 年创建时间提示、当前 Yes/No token 支持范围及既有身份排除，得到 13,595 个发现候选。固定种子 20260920 按 market ID 哈希选取 128 个，不使用题材、最终价格、成交量、获胜状态或 closed 状态筛选。所选集合与扩展后既有训练合约身份交集也为零。

| 项目 | 实测结果 |
| --- | ---: |
| 选定候选 | 128 |
| 扫描原生成交行 | 404,540,000 |
| 重建成交 | 689,849 |
| 有重建成交的候选 | 125 / 128 |
| 所需区块及时间关联成功 | 286,120 / 286,120 |
| 原生身份/token 对匹配 | 128 / 128 |
| 原生父事件 | 127 |
| 首批固定 8 个合约的历史规则/结算核验 | 7 / 8 通过 |
| 新增正式训练准入 | 0 |

价格库约 246 MiB，保留全部候选，包括缺价者。创建日期、当前题目和描述仍是回溯发现材料；只有后续历史核验通过的内容才能作为历史输入。未覆盖其他 outcome 标签和旧 FPMM；成交不是盘口、深度或保证可成交价格。PMA 旧入口仍默认宏观候选，新的 `--selection` 是独立显式入口。

原生身份抓取 v1 使用 `/markets/{id}`，该端点没有父事件；v2 使用列表端点，分别请求 closed=true/false，补齐父事件。v2 共 131 次请求、3 次失败，其中两个结算定位请求失败，原始响应全部保留。当前 API 的结算字段只用于定位链上证明。

证明 v1 曾误将 NegRisk loader 的 `(policy, runtime)` 元组传给只接受 runtime 字符串的核验器，导致三项误报。已修复，增加实际 loader 类型回归检查，从同一批归档响应离线重建 v2；**这三项是接口错误，不是数据缺陷**。v2 七项通过，一项当前规则与初始化文本不一致而保留隔离。v1 诊断没有覆盖或删除。

证明通过不代表事件分组、语义去重、基准隔离已完成，也不证明预测能力。

## 团队 episode 原型

`team_learning.py` 用 LangGraph 串行复用一个 8B：Research 自选历史/跨合约变化查询并提出假设 → 确定性工具 → Forecast 给概率及是否考虑交易 → Risk 否决 → 模拟执行反馈 → Reflection 复盘。

工具仅返回时点以前的价格，保留缺失/陈旧状态；相关系数标记为描述统计。实际生成延迟进入决策时间。复盘引用独立反馈事实 ID，记忆带来源、可用时间与内容哈希，并始终标为待验证假设。候选导出区分事前决策和事后复盘，不自动把盈利或格式有效解释成正确教师答案。

原型是独立 100 美元 episode 账户、固定风险上限、持有至结算的执行器，**不是连续组合、多轮自主研究或自主中途卖出的完整交易团队**。未放宽旧入口的默认信号准入，无真实交易。

## 真实 8B 检查

为验证接口，使用已有合格训练分区的四个旧事件，不把宏观样本当成新的广泛市场学习。新 128 个候选未进入模型训练。源码冻结后启动，Qwen3-8B-Base 权重与 LoRA 均未更新。

| 项目 | 实测结果 |
| --- | ---: |
| 真实模型调用 | 21 |
| episode 记录 | 4 |
| 完成团队决策 | 3 / 4 |
| 完成复盘 | 2 / 4 |
| 通过结构校验的候选经验 | 13 |
| 保存的候选记忆 / 后续历史决策实际使用 | 2 / 0 |
| 合格 SFT 样本 / 新微调 | 0 / 未启动 |
| 峰值分配 / 保留显存 | 16.42 / 18.00 GiB |

21 次调用中 8 次校验失败，包括可修复的工具参数错误、风险 JSON 截断、复盘使用不存在的 `fact_1`。修复上限后失败保持缺失。完整决策生成约 16–47 秒。

四个独立账户均保持 100 美元、零成交。两组 Brier 劣于同期市场，一组相同，一组因团队失败缺失预测。不是能力或收益改善。

原标签的 `available_at` 保留在 2026 年核验时点，没有倒填到 2023 年，所以两条复盘记忆未进入更早的历史决策。测试覆盖了“反馈可用后，不同事件才可读取记忆”的机制；**实测没有证明跨 episode 记忆学习**。离线学习轮次与历史时间的契约仍需补齐，不能为让记忆生效而改写原时间。

四个 episode 的原始请求、输出、工具结果、评分与账本已从冻结源码离线重放一致。

## 产物与复现

- 配置：`configs/team_market_discovery_v1.json`、`configs/team_learning_interface_smoke_v1.json`。
- 候选：`data/processed/team_market_discovery_20260920_v1`。
- 价格：`data/processed/team_market_trade_store_20260920_v1`。
- 身份：`data/raw/team_market_identity_20260920_v2`。
- 证明：`data/raw/team_market_proofs_20260920_v2`。
- 冻结源码、模型清单、原始调用、候选经验：`runs/qwen3_8b_team_learning_interface_smoke_v1`。
- 数据审计：`runs/team_learning_foundation_audit_v1/audit.json`。

在 `conda lab` 中运行，输出目录须不存在：

```bash
PYTHONPATH=src python -m foretellmesh.team_market_scope \
  --config configs/team_market_discovery_v1.json \
  --catalog data/processed/pma_polymarket_catalog_20260919_v1 \
  --membership data/processed/pma_macro_research_20260919_v1/membership.jsonl \
  --output data/processed/team_market_discovery_next

PYTHONPATH=src python -m foretellmesh.pma_trades \
  --extraction data/raw/pma_polymarket_extracted_20260919_v1 \
  --catalog data/processed/pma_polymarket_catalog_20260919_v1 \
  --selection data/processed/team_market_discovery_next \
  --output data/processed/team_market_trade_store_next

# 离线复算，不调用 GPU、不打开最终测试。
PYTHONPATH=runs/qwen3_8b_team_learning_interface_smoke_v1/source_snapshot \
python -m foretellmesh.team_learning_experiment audit \
  --output runs/qwen3_8b_team_learning_interface_smoke_v1

PYTHONPATH=src:tests python -m unittest \
  test_team_learning test_pma_data test_pma_proof_pilot \
  test_pma_legacy_proof test_paper_trading test_langgraph_runtime -q
```

剩余工作：完成新事件分组/基准隔离和独立开发清单，扩大真实团队经验生成，实现任务级经验筛选和训练导出，再训练候选能力 LoRA 并运行四组对照。当前只有候选导出器，没有合格经验训练器；不以自我评价、格式通过或偶然盈利替代训练质量核验。
