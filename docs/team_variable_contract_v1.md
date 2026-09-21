# 研究变量、单位和来源的绑定协议

日期：2026-09-21。上一轮分步研究能检查数据共同覆盖，但模型仍将票房假设映射为合约价格。本轮新增显式变量协议，让研究角色先声明需要测量什么，数据角色再绑定来源。

## 交接方式

研究角色自行提出关系、合约和预测目标，并为每个输入声明：

- `variable_id`：本次研究中唯一的变量编号。
- `definition`：测量含义。
- `quantity`、`unit`：量的身份及单位。
- `role`、`lag_days`：目标/参照合约，以及过去多少天。

数据角色只能返回每个变量对应的 `source_id`，或 `null` 与缺失原因。它不能改写变量、单位、合约或时点。变量必须逐一覆盖，未知来源、遗漏和重复绑定均拒绝；来源登记由程序提供，模型不能自行添加。

当前实现先让研究角色声明变量，**不向它展示来源登记、来源类型标签和价格覆盖表**；来源目录只在后续交给数据角色。这样可以检查模型是否根据假设本身命名变量，而非为了匹配现有数据复制来源标签。它仍接收合约规则与原探索候选；这不是新数据集或独立盲测。第三版还将研究目录限制为可调用的合约编号、规则和初始化时间，事件分组继续由评估程序使用，避免把分组编号混入工具参数。

当前唯一已接入的数值来源是 `pma_yes_trade_price`：

| 属性 | 定义 |
| --- | --- |
| 测量的量 | `contract_yes_trade_price` |
| 单位 | `USD_per_YES_share` |
| 数值 | 历史 Yes 成交的区块均价 |
| 时间 | `source_time`，通过历史 as-of 读取和新鲜度检查 |
| 来源 | 已核验的价格 SQLite 文件及来源报告，各自记录 SHA-256 |

程序要求变量的量和单位同时与来源严格一致。相同单位不能让不同量互换；相同量也不能静默转换单位。不做币种换算、不将美分当美元、不把价格当票房、发帖数量或实测事件结果。外部变量仍可由团队自由提出，但没有确切来源就保留为缺失；不会以价格代理偷偷替代。

另有一条显式传输兼容规则：模型若把 `source_id` 的 JSON `null` 写成字符串 `"null"`，且同时提供非空缺失原因，验证器只将其解析为“缺失”。它绝不会因此产生可用来源。原始模型字节保留在调用档案中，规范化结果另存于工作流；缺失原因为空仍拒绝。

所有变量绑定通过后，才机械转换为既有解释器的价格读取请求，继续执行共同覆盖、公式引用校验、冻结和评分。完整变量协议、绑定和来源登记的哈希进入输入检查结果；冻结时的前序调用哈希保留这条来源链。原始研究声明始终归档，转换只适配已核验的读取接口。

## 可保证的范围

该协议能阻止**已明确声明的不同量或单位被错误绑定**，也能防止数据角色在交接时悄悄改变含义。它不能自动证明研究角色最初写下的 `definition`、`quantity` 和自然语言假设彼此完全一致；研究角色仍可能从一开始就错误命名变量。这类语义错误需要单独复核，不能因为格式和来源一致就宣称研究正确。

这里的单位核验针对数据来源绑定，不是对任意预测公式进行完整量纲证明。预测误差改善、净收益和独立泛化仍需分别检验，不由变量协议证明。

本轮沿用同一冻结 Base、原有训练分区 20 个合约和评价门槛。没有新增外部数据、重新划分训练集、开启最终测试或微调。旧流程保留；只有 `method_workflow=typed_staged_team` 启用该协议。

## 复现

```bash
PYTHONPATH=src:tests /home/jiachongliu/anaconda3/envs/lab/bin/python \
  -m unittest test_team_variable_contracts test_team_staged_methods test_team_executable_methods -q

PYTHONPATH=src /home/jiachongliu/anaconda3/envs/lab/bin/python \
  -m foretellmesh.team_executable_experiment run \
  --config configs/team_variable_contract_v3.json \
  --output runs/qwen3_8b_team_variable_contract_v3

PYTHONPATH=runs/qwen3_8b_team_variable_contract_v3/source_snapshot \
/home/jiachongliu/anaconda3/envs/lab/bin/python \
  -m foretellmesh.team_executable_experiment audit \
  --output runs/qwen3_8b_team_variable_contract_v3
```

原始角色输出、每次校验错误、明确缺失和停止状态均保留。每个角色至多一次结构修复；不根据评分反复生成计划。

## 首轮实测与协议修订

`qwen3_8b_team_variable_contract_v1` 完成两次冻结模型调用，格式均有效，却未解决最初的语义问题：研究角色将变量命名为 `spiderman_gross`、`transformers_gross`，定义明确写首周票房、以 USD 衡量，但 `quantity` 填了 `contract_yes_trade_price`，`unit` 填了 `USD_per_YES_share`。数据角色随后匹配价格来源，字段核验通过。

这次仍因 90 个预定日期没有共同可用输入而停止，没有进入计算或评分。**不能把这种停止解释成变量语义已被核验。** 首轮报告和失败原样保留，报告 SHA-256 为 `8b28baa3e51793760e65484c4b696427d4ef8a6870874c098d7f97ad51901820`。

第二版只调整研究声明与来源展示的顺序，以及与此相关的提示文字；不改模型、合约集合、数据源或评价门槛。第一版没有预测分数，不存在按收益挑选该修订的情况。所有历史回放均使用各自冻结的源码。

`qwen3_8b_team_variable_contract_v2` 的原始输出已将量写为 `Opening Weekend Gross`，单位写为 `Millions of Dollars`，而非合约价格。但它将 `team:native_market_251132` 等事件组编号当作合约编号，两次输出均被拒绝，因此数据角色没有运行。不能把局部变量声明改善描述为整个流程成功。报告 SHA-256：`5bb4dc8025ad718df9f73bf33d77d07566ab007ec0f40ee92d5e3649e0dffaa5`。

第三版将研究目录中的路由标识简化为 `market_id`，并让编号错误明确返回允许的合约编号；程序不自动改写或猜测模型原始编号。前两版均没有进入预测评分，这些修订针对语义交接和接口缺陷，不是根据效果挑选结果。

## 第三轮实测及兼容处理验证

第三轮 `qwen3_8b_team_variable_contract_v3` 的研究角色成功声明了合约编号 251132、251306，两个变量均为 `opening_weekend_gross`、单位 `USD_million`，描述也明确是电影首周票房。数据角色报告没有匹配来源，但将缺失来源写成了字符串 `"null"`。原版本校验器因此两次拒绝它，运行状态保留为 `data_selection_failed`，不能事后改称成功。其报告 SHA-256 为 `c621b552d554da4f91e3ebee975afedcf600ad8d91203f209ba92c01a9168d33`。

加入上述保守兼容规则后，使用第三轮**原始前两次输出**进行 CPU 重处理，不再次调用模型，也不修改原始实验。请求与记录逐项一致，唯独缺失值的传输解析更新；原第三次修复调用不再需要，但仍保留在旧档案中。结果为 `unavailable_variables`，明确列出两项百万美元单位的首周票房需求，计算、冻结和评分均未启动。

这说明本例可以准确保留声明并报告来源缺失，不说明团队已经能提出有效策略。原假设仍包含未经证实的盈利断言，单个例子的变量改正也不能证明所有自然语言映射可靠。当前缺口仅针对本假设和已登记来源，不能概括为整个 Polymarket 历史档案不足。

兼容处理记录：`runs/team_variable_null_transport_check_v1`。其中 `price_reads=0` 指工作流未进入价格覆盖读取；准备步骤仍会核验文件并重建原始输入上下文。该记录是**新校验器的离线兼容检验**，不是第四次 GPU 实验，也不是把第三次原运行重审为成功。

```bash
PYTHONPATH=src /home/jiachongliu/anaconda3/envs/lab/bin/python \
  scripts/replay_team_variable_null_transport.py \
  --source runs/qwen3_8b_team_variable_contract_v3 \
  --output /tmp/team_variable_null_replay
```

新增十四项变量与来源协议测试，相关模块共 **156 项测试通过**。三次 GPU 运行的冻结源码审计全部通过，共七次模型调用，始终为零个真实预测评分观测。兼容处理另验证两次既有输出，零新增模型调用。全部失败、源码、配置和数据哈希保留；未开启新开发事件或最终测试，未微调，未产生新交易收益。
