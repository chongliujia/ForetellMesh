# PMA 历史合约补证试跑

后续扩充已形成[首版 FOMC 研究数据集](pma_fomc_research_dataset.md)，完成用途审阅与时间分区；本页保留最初 16 合约试跑的结果和当时的限制。原始证明被新流程重新核验，旧派生产物与代码哈希仍保留作阶段记录。

2026-09-19，在已完成价格重建的 PMA 历史库上，固定 2025 年三月、五月、六月、七月四次 FOMC 会议的全部 16 个候选合约。每个合约计划取官方声明前 7 天、前 1 天两个观察点。选择不依赖最终输赢或报价可用性；这是工程试跑，不能视为有代表性的预测基准。

## 实测结果

| 检查 | 结果 |
| --- | ---: |
| 选定父事件 / 合约 | 4 / 16 |
| 完整归档链上补证的合约 | 16 |
| 同时通过历史规则与最终结算核验 | 12 |
| 计划历史观察 | 32 |
| 通过逐条数据检查的观察 | 24 |
| 这些合格观察对应的事件组 | 3 |
| 被隔离的合约 / 观察 | 4 / 8 |
| 正式训练 / 评测准入 | 0 / 0 |

29 次 HTTP 请求、1,097 次 RPC 只读请求均成功。实际网络采集窗口约 **183 秒**；不包括代码开发、源码审阅、测试和离线重建，也不能外推为全部市场的补证耗时。16 个合约的 32 个预设观察点都有符合三小时新鲜度门槛的归档成交报价；8 条观察因规则差异继续隔离。

五月、六月、七月的 12 个合约已核验初始化原文、观察时点以前的规则、最终 CTF 赔付、官方结果以及事前价格。每条可审阅输入只使用上次 FOMC 声明作为最小证据集，尚未覆盖当时的就业、通胀和其他新闻。

## 实际发现的规则差异

三月的 4 个合约 `516008`、`516009`、`516010`、`516011`，初始化日志中的会议日期为 **March 19 - 20, 2025**，而目前 Gamma 描述为 **March 18 - 19, 2025**。日期修正何时生效、是否存在其他补充说明尚未审核，程序没有把当前描述回填到历史输入。

该批仍保留全部原始响应和计划观察点。修复解析器不会自动消除这个真实的数据差异；后续需要单独的规则版本审阅，或继续排除这些合约。API 中 `new_version_q` 本身也不能代替规则修改记录。

## 旧版部署与规则审核

本批使用适配器 `0x2f5e3684cb1f318ec51b00edba38d79ac2c0aa9d`，以及报告合约 `0x71523d0f655b41e805cec45b17163f528b59b820`。它们与先前 2026 年试点的部署不同；地址可对照 [Polymarket 官方部署表](https://github.com/Polymarket/neg-risk-ctf-adapter/blob/main/addresses.json)。本次取得的原文 SHA-256 为 `e5f0031146f7376f02dc106111ea3183dba4e6b0b336a92cdb3cbaa47835b163`。

历史适配器的 [Polygonscan 验证源码](https://polygonscan.com/address/0x2F5e3684cb1F318ec51b00Edba38d79Ac2c0aA9d#code)、18 个源码/settings 文件及部署 runtime 均已归档，审核绑定在 [pma_rules_review_v1.json](../configs/pma_rules_review_v1.json)。runtime 为 16,185 字节，SHA-256 为 `792f55bdb962310a3136c551f83929574ebb104606b3dd9dd808f5cb619434ad`。

与此前已审阅源码逐文件比较：除 `UmaCtfAdapter.sol`、`IUmaCtfAdapter.sol` 和 `settings.json` 外，其余文件哈希相同。差异包括两天的紧急结算等待期、通过 Finder 确定 oracle、紧急结算方法/事件命名；本批只接受普通 `QuestionResolved` 流程，不据此支持紧急结算。

规则存储相关路径另行检查：

- `initialize` 追加 initializer，按 ancillary text 生成 question ID，并拒绝重复初始化；`_saveQuestion` 仅由此调用。
- reset、暂停、授权和结算不会替换 ancillary text 或 creator。reset 会改变请求时间，本试跑对这类状态不一致保守拒绝。
- `BulletinBoard.postUpdate` 只追加由 question ID 和创建者地址索引的记录；源码中无清空或删除该数组的路径。
- 审阅部署无代理、delegatecall、自毁或升级入口。runtime 中固定的 `ctf` 地址与上述旧版报告合约一致。

每个合约分别核对初始化区块和明确 finalized 区块上的 runtime，均须与审核版本完全一致。初始化文本、当前 question 状态和创建者身份必须一致；公告板创建者数组须为空。基于已审核的不可改写原文和只追加数组，可以推断固定区块之前没有创建者更新，不能把证明延伸到固定区块之后。

信任范围为 **Polygonscan 源码验证 + 单一公开 RPC 的区块、回执、状态证明**，未声称独立编译验证或独立链上共识证明。不同部署、非空更新、不同规则或不支持的结算路径均不自动放行。

## 最终标签与时间隔离

结算 API 的更新时间只用于定位候选区块。正式核验绑定：

1. UMA 的普通结算日志及成功回执，与同笔交易的 NegRisk `QuestionReported` 对应的 native question ID、request ID、结果一致。
2. 最终 CTF `ConditionResolution` 的 condition ID、oracle 地址、native question ID、二元赔付与 UMA 报告一致。
3. 日志、回执、区块、交易序号和日志序号相互匹配，交易须存在于所声明区块。
4. 最终区块前一块 denominator 为 0，最终区块为 1，两个 numerators 与日志相同。
5. 初始化时间 ≤ 观察时点 < 官方结果发布时间 ≤ UMA 报告时间 ≤ 最终 CTF 结算时间。

旧版报告与最终结算之间存在延迟，仍使用 CTF 区块时间作为 `resolution_time`。当前 API 状态、终值价格和官方结果正文不会进入模型输入。

官方结果依据对应 [五月](https://www.federalreserve.gov/newsevents/pressreleases/monetary20250507a.htm)、[六月](https://www.federalreserve.gov/newsevents/pressreleases/monetary20250618a.htm)、[七月](https://www.federalreserve.gov/newsevents/pressreleases/monetary20250730a.htm)声明，解析发布日期、发布时间、正文和修订标记。官方历史页面的日期是来源的历史发表声明，不冒充当时已保存的本地网页快照。

市场价格沿用 PMA 的“观察时点以前最新区块内平均成交价”，要求不早于 T−3 小时。它是市场基线，不是可执行 bid/ask。每条输入通过既有 typed schema；结果标签保存在评估侧，`review_inputs.jsonl` 通过显式字段白名单导出。

## 产物、复现与发布边界

- 冻结试跑配置：`configs/pma_proof_pilot_v1.json`。
- 原始采集：`data/raw/pma_proof_pilot_20260919_v1`。
- 有效派生版本：`data/processed/pma_proof_pilot_20260919_v3`；v1/v2 为适配检查记录。
- 源码归档：`data/raw/pma_adapter_source_20260919_v1`。
- 官方部署表：`data/raw/pma_deployment_reference_20260919_v1`。
- 运行、测试、离线复现：`runs/pma_proof_pilot_v1`。

输出包含合约证明、逐观察候选、隔离的标签、只供审核的模型输入、事件分组和计数报告。配置固定上游目录、原生身份、价格仓库、基准索引及源码审阅的哈希；重建核对原始响应与完整价格数据库哈希，并将处理代码哈希纳入数据版本。失败记录保留在分母中。

```bash
conda activate lab

# 公共只读重采集；使用新的输出目录，不覆盖原始证明。
PYTHONPATH=src python -m foretellmesh.pma_proof_pilot capture \
  --config configs/pma_proof_pilot_v1.json \
  --output data/raw/pma_proof_pilot_next

# 从本次原始归档离线重建。
PYTHONPATH=src python -m foretellmesh.pma_proof_pilot build \
  --config configs/pma_proof_pilot_v1.json \
  --capture data/raw/pma_proof_pilot_20260919_v1 \
  --output data/processed/pma_proof_pilot_rebuild_next

PYTHONPATH=src python -m unittest discover -s tests -q
```

新增 15 项测试覆盖完整采集→重建、离线确定性、标签隔离、当前规则变化、非空更新、runtime 替换、错误报告合约、失败回执、赔付边界与状态冲突、未来证据/价格、官方结果冲突、缺价和采集失败保留分母等路径。

全量测试 362 项：361 通过、1 项可选 GPU 测试跳过。真实归档的 6 个派生文件（包括报告）离线重建后逐字节一致，24 条模型输入与 typed schema 的白名单导出完全一致，全部处理代码哈希与运行报告吻合，见 `runs/pma_proof_pilot_v1/verification.json`。

**这 24 条观察仍是数据检查通过的审核样本，尚非正式发布的训练或评测集。** 四个父事件已按各自 FOMC 发布分组，合格记录只覆盖其中三组。还需扩充独立事件、审核跨父事件关联与 ForecastBench 语义重叠、冻结实际时间切分及用途。当前精确基准匹配为零不代表语义去重通过，也未更改旧评测范围或双平台对照的准入要求。预训练模型可能记住历史结果的风险须在最终评测设计中单独处理。

本轮未调用模型、未训练 LoRA，也未报告预测能力提升。真实事件 SFT 另需事前教师答案；独立评分与后续 RL 不因缺教师答案而受阻。
