# RTX 3090 上的 BF16 LoRA 资源测试

此测试回答：在本机 RTX 3090 上，固定配置的 Qwen3-8B-Base 普通 LoRA 能否完成前向、反向和优化器更新，以及峰值显存和吞吐量是多少。它不保存适配器，不产生可部署 checkpoint，不评估预测能力。

使用 `conda` 的 `lab` 环境。基座模型固定为 `Qwen/Qwen3-8B-Base` 的 `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` 版本。先下载官方文件，核对权重的上游 LFS SHA-256，再在每次测试前重新核对本地文件。模型代码使用已安装的 Transformers，`trust_remote_code=False`，训练期间只读取本地权重。

## 本机实测结果（2026-09-18）

RTX 3090 24GB、驱动 580.178.04，`lab` 中 Python 3.11.14、PyTorch 2.13.0（CUDA 13.0）、Transformers 5.17.0、PEFT 0.20.0、Accelerate 1.12.0。以下两种长度均完成 48 个微步、3 次 AdamW 更新，无 OOM，梯度和损失有限，适配器实际更新，基座保持冻结。

| 序列长度 | 峰值分配显存 | 峰值预留显存 | 预热后输入 token/s | 状态 |
| --- | ---: | ---: | ---: | --- |
| 1024 | 18.25 GiB | 18.83 GiB | 856.6 | 通过 |
| 2048 | 20.58 GiB | 21.67 GiB | 896.4 | 通过 |

冻结的 BF16 基座有 8,190,735,360 个参数，FP32 LoRA 有 43,646,976 个可训练参数。两次测试的完整配置、输入数据和代码哈希均已核对，使用相同的代码与模型版本。

2048 测试结束前、模型和优化器仍驻留时，设备可用显存约 0.97 GiB；这不是训练期间的连续最小余量。该长度已经可用，但余量有限，首轮正式 SFT 应先保持 `r=16`、micro batch 1，评估和保存阶段仍需实测。1024 的对应余量约 3.81 GiB。无需为这套已测配置强制使用 4-bit QLoRA。

原始报告保存在本机忽略目录：

- [1024 测试报告](../runs/qwen3_8b_bf16_lora_probe_seq1024_v1/report.json)
- [2048 测试报告](../runs/qwen3_8b_bf16_lora_probe_seq2048_v1/report.json)

报告中的合成交叉熵只验证训练流程，不是预测指标。未保存模型 checkpoint，未进行 ForecastBench 模型评测，也未将任何新模型设为默认。普通回归测试 71 项通过；另有 1 项需显式开启的 GPU 流程测试，已单独执行通过。

## 方法

版本化配置：[qwen3_8b_bf16_lora_probe_v1.json](../configs/qwen3_8b_bf16_lora_probe_v1.json)。

- 基座 BF16、不量化、全部放在单张 GPU；只训练 FP32 LoRA 参数。
- `r=16`、`alpha=32`、`dropout=0.05`，覆盖 q/k/v/o/gate/up/down 七类投影。
- micro batch 1，梯度累积 16，AdamW `lr=1e-4`、`weight_decay=0`、`foreach=False`，梯度裁剪 1.0。
- SDPA、梯度检查点（非 reentrant）、关闭 KV cache，未启用 CPU offload。
- 每种长度独立加载模型并重新初始化适配器。先跑 1024，再依据显存实测运行 2048。
- 每种长度 3 次优化器更新，共 48 个微步；第一轮作为预热，后两轮统计吞吐量。
- 输入为公平硬币的合成场景，重复上下文填满序列，无 padding；仅对 JSON 回答及 EOS 计算损失。0.5 来自合成设定，不来自真实事件的事后标签或市场价格。
- 检查基座不接收梯度、只有 LoRA 可训练、梯度非零、损失有限、适配器参数实际改变。

训练前后评估同一合成输入的交叉熵，只用于检查数值流程。它不是验证集损失或预测 Log Loss；重复合成输入上的下降不能说明泛化。Prophet Arena、ForecastBench 和 PMA 都不参与此测试。

报告记录 PyTorch `max_memory_allocated` 和 `max_memory_reserved`，覆盖加载、训练前后检查及优化器更新。两者都不包含桌面进程和部分驱动显存；设备空闲显存是在检查点读取，不能当作连续测量的最小值。吞吐量包含完整累积与优化器更新，报告同时区分输入 token/s 和参与监督的 token/s。

## 复现

在仓库根目录执行。下载阶段需要网络和约 16GB 权重空间，后续 probe 只需本地文件与 GPU。

```bash
conda activate lab
PYTHONPATH=src python -m foretellmesh.fetch_probe_model \
  --config configs/qwen3_8b_bf16_lora_probe_v1.json

PYTHONPATH=src python -m foretellmesh.lora_probe \
  --config configs/qwen3_8b_bf16_lora_probe_v1.json \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --sequence-length 1024 \
  --output runs/qwen3_8b_bf16_lora_probe_seq1024_v1

# 先检查 1024 的报告和显存余量，再运行：
PYTHONPATH=src python -m foretellmesh.lora_probe \
  --config configs/qwen3_8b_bf16_lora_probe_v1.json \
  --model-manifest checkpoints/qwen3_8b_base_manifest.json \
  --sequence-length 2048 \
  --output runs/qwen3_8b_bf16_lora_probe_seq2048_v1
```

输出目录必须不存在，已有模型 manifest 可以直接复用。下载器若报 `Unknown scheme for proxy URL ... socks://`，可对该次下载命令使用 `env -u ALL_PROXY -u all_proxy`，保留可用的 HTTP/HTTPS 代理；不修改 shell 的全局配置。

本机 Xet 传输曾停滞，实际改用普通 HTTP 下载：在下载命令前设置 `HF_HUB_DISABLE_XET=1`，并用 `--cache-dir checkpoints/hf_http_cache` 保存独立缓存。下载方式不改变权重内容，完整文件仍需通过相同的 SHA-256 核对。

`report.json` 保存版本、配置、数据与代码哈希、训练步骤、吞吐量、显存、前后合成损失及错误状态。`synthetic_batch.json` 保存确切的输入 token 和监督 mask；`config.json` 保存原始配置。OOM/训练错误会保留失败报告，不标记为成功。浮点结果和速度可能受硬件、驱动及计算内核影响，不保证逐位相同。

普通测试不导入 GPU 依赖。可选 GPU 流程测试使用极小的随机 Qwen3，在临时目录检查相同的训练入口，不能替代完整 8B 实测：

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
FORETELLMESH_GPU_TEST=1 PYTHONPATH=src python -m unittest tests.test_lora_probe_gpu -v
```

模型规格见 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-8B-Base)，LoRA 接口见 [PEFT 文档](https://huggingface.co/docs/peft/package_reference/lora)。正式 SFT 仍需要合格的历史训练数据，以及独立的预测质量评估。
