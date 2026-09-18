# AGENTS.md

## 1. Project Goal

This repository builds a financial forecasting and research system based on **Qwen3-8B**, with:

- Multi-Agent orchestration
- Multi-LoRA capability specialization
- Financial and prediction-market datasets
- SFT / preference optimization / reinforcement learning
- RAG, search, Python, SQL, and external data tools
- Probability forecasting and calibration evaluation
- Local development on a single RTX 3090 24GB where practical

The primary objective is **not generic chat quality**. The system should become a reliable financial research and event-forecasting engine that can:

1. gather evidence,
2. reason over time-bounded information,
3. use tools correctly,
4. produce calibrated probabilities,
5. expose uncertainty,
6. support reproducible evaluation.

---

## 2. Foundation Model

Default foundation model:

```text
Qwen/Qwen3-8B-Base
```

Do not replace the foundation model without an explicit benchmark showing a meaningful improvement on this repository's forecasting and agent tasks.

Model-selection decisions must prioritize:

1. forecasting quality,
2. probability calibration,
3. Chinese + English financial understanding,
4. LoRA / QLoRA compatibility,
5. RL training compatibility,
6. vLLM / serving compatibility,
7. RTX 3090 development feasibility.

Generic benchmark scores alone are not sufficient justification for changing the base model.

---

## 3. Core Architecture

Use capability-oriented LoRAs and task-oriented Agents.

Do **not** assume:

```text
1 Agent = 1 LoRA
```

Preferred structure:

```text
Qwen3-8B Base
│
├── research_lora
├── fundamental_lora
├── valuation_lora
├── risk_lora
├── quant_lora
├── tool_lora
├── forecast_lora
└── calibration_lora
```

Agents may share the same LoRA.

Example:

```text
Orchestrator
│
├── Research Agent
│   └── research_lora
│
├── Fundamental Agent
│   └── fundamental_lora
│
├── Quant Agent
│   └── quant_lora / tool_lora
│
├── Risk Agent
│   └── risk_lora
│
├── Forecast Agent
│   └── forecast_lora
│
└── Calibration / Critic Agent
    └── calibration_lora
```

An Agent is:

```text
Agent =
system prompt
+ workflow
+ tools
+ context
+ memory/state
+ optional LoRA
```

A LoRA represents a reusable **capability**, not an identity.

---

## 4. Recommended Agent Responsibilities

### 4.1 Orchestrator

Responsibilities:

- classify the user/task intent,
- decide which Agents are required,
- control execution order,
- limit redundant Agent calls,
- merge structured outputs,
- resolve conflicting evidence,
- pass only relevant context downstream.

The Orchestrator should remain lightweight.

Do not train domain knowledge into the Orchestrator unless necessary.

---

### 4.2 Research Agent

Responsibilities:

- retrieve news, filings, reports, and historical evidence,
- enforce observation-time constraints,
- extract facts with timestamps,
- attach source metadata,
- separate facts from commentary,
- detect potentially stale evidence.

Preferred output:

```json
{
  "evidence": [],
  "counter_evidence": [],
  "unknowns": [],
  "source_times": [],
  "observation_time": ""
}
```

---

### 4.3 Fundamental Agent

Responsibilities:

- financial-statement interpretation,
- business-quality analysis,
- earnings-quality analysis,
- cash-flow analysis,
- balance-sheet analysis,
- industry and competitive analysis.

Do not use the Fundamental Agent as a source of real-time market data.

---

### 4.4 Quant Agent

Responsibilities:

- numerical analysis,
- time-series transforms,
- statistical features,
- scenario analysis,
- Python calculations,
- structured market-data processing.

Prefer tools for arithmetic and data processing.

Do not rely on LLM mental arithmetic when deterministic computation is available.

---

### 4.5 Risk Agent

Responsibilities:

- identify downside scenarios,
- detect financial / operational / liquidity / credit risks,
- challenge assumptions,
- enumerate failure modes,
- inspect tail risks and missing variables.

The Risk Agent must not merely repeat the main forecast.

---

### 4.6 Forecast Agent

Responsibilities:

- predict event probabilities,
- combine evidence from upstream Agents,
- distinguish base rates from case-specific evidence,
- report uncertainty,
- avoid false precision,
- produce machine-readable probabilities.

Required core output:

```json
{
  "event": "...",
  "probability": 0.0,
  "confidence": "low|medium|high",
  "base_rate": null,
  "key_evidence": [],
  "counter_evidence": [],
  "unknowns": [],
  "observation_time": ""
}
```

`probability` must be within `[0, 1]`.

---

### 4.7 Calibration / Critic Agent

Responsibilities:

- inspect Forecast Agent overconfidence,
- compare forecasts to historical calibration,
- identify ignored base rates,
- identify duplicated or correlated evidence,
- detect reasoning inconsistencies,
- propose a revised probability only when justified.

It should not automatically push probabilities toward 0.5.

---

## 5. Multi-LoRA Rules

Use LoRA for capabilities that benefit from parameter adaptation.

Good LoRA targets:

- financial reasoning style,
- forecasting behavior,
- valuation reasoning,
- risk analysis,
- structured tool calling,
- probability calibration,
- domain-specific response formats.

Do not use LoRA to memorize:

- current prices,
- latest financial statements,
- live market probabilities,
- frequently changing news,
- ephemeral facts.

Those belong in:

- RAG,
- databases,
- APIs,
- tools.

Prefer consistent LoRA configuration across adapters when possible.

Default starting point:

```yaml
r: 16-32
lora_alpha: 32-64
lora_dropout: 0.05
```

Typical target modules:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Do not stack multiple independently trained LoRAs online without evaluation.

Prefer one active domain/capability LoRA per Agent request unless a specific composition has been validated.

---

## 6. Local Training Target

Primary local hardware assumption:

```text
GPU: RTX 3090
VRAM: 24 GB
```

Default local fine-tuning method:

```text
4-bit QLoRA
NF4
BF16 compute
gradient checkpointing enabled
```

Starting configuration:

```yaml
model: Qwen/Qwen3-8B-Base

quantization:
  load_in_4bit: true
  quant_type: nf4
  double_quant: true
  compute_dtype: bfloat16

training:
  max_seq_length: 2048
  micro_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 1e-4
  epochs: 2

lora:
  r: 32
  alpha: 64
  dropout: 0.05
```

Treat these as defaults, not immutable constants.

Measure actual VRAM before increasing sequence length or batch size.

---

## 7. Training Stages

Preferred training order:

```text
Qwen3-8B-Base
    ↓
Optional Domain Continued Pretraining
    ↓
Financial / Agent SFT
    ↓
Optional Preference Optimization
    ↓
Forecast SFT
    ↓
GRPO / RLVR
    ↓
Calibration Evaluation
```

Do not start RL from a raw Base checkpoint for production experiments.

RL should begin from a model that already:

- follows the output schema,
- understands the task,
- generates valid probabilities,
- uses tools correctly enough for reward computation.

---

## 8. Prediction-Market Data Schema


## 8.1 Core Datasets

The current core dataset plan includes:

```text
Prophet Arena Subset 1200
ForecastBench
Prediction Market Analysis
```

These datasets should be treated as distinct sources with explicit provenance and split policies.

### Prophet Arena Subset 1200

Primary intended use:

- forecasting SFT,
- compact supervised experiments,
- prompt/schema validation,
- early GRPO/RLVR experiments,
- fast ablation studies.

Guidelines:

- preserve the original question/event identity,
- retain all available timestamps,
- retain resolution labels,
- avoid splitting near-duplicate or same-event questions across train/test,
- use as a fast iteration subset rather than the sole production benchmark.

### ForecastBench

Primary intended use:

- forecasting evaluation,
- out-of-sample benchmarking,
- probability calibration measurement,
- model-to-model comparison,
- checkpoint regression testing.

Guidelines:

- prefer keeping ForecastBench held out from training when used as a benchmark,
- do not tune prompts or reward weights repeatedly against the final test portion,
- report Brier Score, Log Loss, ECE, and coverage where applicable,
- preserve chronological ordering and event grouping,
- record the exact ForecastBench version/revision used in each experiment.

### Prediction Market Analysis

Primary intended use:

- prediction-market-specific SFT,
- market-implied probability analysis,
- forecast-vs-market comparison,
- calibration studies,
- RL reward construction,
- historical market replay.

Recommended fields where available:

```json
{
  "market_id": "",
  "question": "",
  "platform": "",
  "created_at": "",
  "observation_time": "",
  "market_probability": null,
  "market_price_history": [],
  "available_information": [],
  "resolution_time": "",
  "outcome": null
}
```

For this dataset, temporal integrity is mandatory.

At any historical replay point `T`:

```text
all visible evidence timestamp <= T
all market prices used as features timestamp <= T
resolution information > T must remain hidden
```

Use market probability as a baseline, not as ground truth.

The primary comparison should be:

```text
model forecast quality
vs.
market-implied forecast quality
```

using proper scoring rules.

### Dataset Role Separation

Default role assignment:

```text
Prophet Arena Subset 1200
    -> SFT / fast experiments / RL prototypes

ForecastBench
    -> held-out evaluation / calibration benchmark

Prediction Market Analysis
    -> prediction-market training + market-relative evaluation + RL
```

Do not merge all three datasets blindly into one random split.

Each example must retain:

```text
dataset_source
dataset_version
event_id
observation_time
resolution_time
split
```

Recommended unified metadata:

```json
{
  "dataset_source": "",
  "dataset_version": "",
  "event_id": "",
  "market_id": null,
  "question": "",
  "observation_time": "",
  "resolution_time": "",
  "outcome": null,
  "split": ""
}
```

Cross-dataset deduplication is required before benchmark reporting.

If the same or near-equivalent event appears in multiple datasets, ensure it does not leak from training into held-out evaluation.


Every forecasting example should preserve temporal provenance.

Preferred schema:

```json
{
  "market_id": "",
  "question": "",
  "created_at": "",
  "observation_time": "",
  "market_probability": null,
  "available_information": [],
  "resolution_time": "",
  "outcome": null
}
```

Important fields:

- `observation_time`
- `resolution_time`
- `outcome`

For training examples, the model must never receive information published after `observation_time`.

---

## 9. No Temporal Leakage

Temporal leakage is a critical failure.

For a sample with:

```text
observation_time = T
```

all model-visible evidence must satisfy:

```text
source_timestamp <= T
```

This applies to:

- news,
- filings,
- reports,
- web results,
- market prices,
- prediction-market prices,
- RAG chunks,
- social posts,
- derived features,
- annotations.

Never use current web/RAG results when replaying a historical forecast unless the retrieval system supports strict historical cutoff filtering.

If the timestamp provenance is unknown, treat the evidence as unsafe for historical evaluation.

---

## 10. Dataset Splitting

Do not use naive random train/test splitting for prediction-market data.

Preferred split dimensions:

### Time split

Example:

```text
Train: earliest period
Validation: later period
Test: latest held-out period
```

### Event-group split

Closely related contracts/questions should remain in the same split.

Examples:

```text
US Election 2024
├── candidate wins
├── party wins
├── electoral vote threshold
├── state-level outcomes
└── popular-vote outcomes
```

Do not distribute highly correlated questions across train and test sets.

Deduplicate semantically equivalent markets.

---

## 11. Forecast Evaluation

Accuracy alone is insufficient.

Primary metrics:

```text
Brier Score
Log Loss
Expected Calibration Error (ECE)
Calibration Curve
```

Optional metrics:

```text
AUC
accuracy at explicit thresholds
sharpness
coverage
confidence-stratified performance
market-relative score
```

For binary events:

```text
Brier = (p - y)^2
```

where:

```text
p = predicted probability
y ∈ {0, 1}
```

Always compare against meaningful baselines:

- 0.5 constant baseline,
- empirical base-rate baseline,
- prediction-market implied probability where available,
- previous model checkpoint.

Do not claim forecasting improvement based only on answer fluency.

---

## 12. RL / GRPO Design

Prediction markets are suitable for verifiable rewards.

Prefer proper scoring rules.

Example reward:

```text
R_forecast = -(p - y)^2
```

or use a bounded/log-safe log-score implementation.

Composite reward may include:

```text
R =
w_forecast * R_forecast
+ w_format * R_format
+ w_tool * R_tool
+ w_evidence * R_evidence
```

Start simple.

Do not introduce many reward terms before verifying that the base reward works.

Monitor for reward hacking.

Never optimize primarily for verbosity or chain-of-thought length.

---

## 13. PnL Is Not the Primary Forecast Reward

Do not use trading or betting PnL as the first-stage reward for forecasting quality.

PnL mixes forecasting skill with:

- position sizing,
- transaction costs,
- spread,
- liquidity,
- timing,
- execution,
- market impact.

Train forecasting quality first.

Only evaluate decision/allocation policies in a separate layer.

---

## 14. Tool Use

Use deterministic tools whenever appropriate.

Preferred tools:

```text
Python
SQL
calculator
financial-data APIs
market APIs
RAG
web/search
document parsers
```

Rules:

- calculations -> Python/calculator,
- structured retrieval -> SQL/API,
- fresh facts -> API/RAG/search,
- long documents -> document retrieval/parsing,
- model parameters -> reasoning and synthesis.

Do not fine-tune the model to replace a deterministic tool.

---

## 15. Structured Outputs

All inter-Agent communication should prefer JSON or typed schemas.

Avoid free-form prose between Agents when downstream parsing is required.

Validate:

- schema,
- probability bounds,
- timestamps,
- required fields,
- tool arguments.

If validation fails, retry with a repair prompt before continuing the pipeline.

---

## 16. Reproducibility

Every training run must record:

```text
base model
base model revision
dataset version
dataset hash
train/validation/test split version
LoRA configuration
optimizer
learning rate
batch size
gradient accumulation
max sequence length
seed
training framework version
CUDA version
GPU model
reward implementation version
evaluation code version
```

Store experiment configuration in version-controlled YAML/JSON.

Do not rely on undocumented CLI history.

---

## 17. Experiment Naming

Use explicit run names.

Example:

```text
qwen3-8b_forecast-sft_r32_lr1e-4_v3
qwen3-8b_forecast-grpo_brier_v2
qwen3-8b_risk-lora_r16_v1
```

Avoid names such as:

```text
test1
final
final2
new_final
```

---

## 18. Evaluation Before Merging

A new LoRA/checkpoint should not become the default without evaluation.

At minimum compare:

```text
Brier Score
Log Loss
ECE
format validity
tool-call validity
latency
VRAM
tokens/sec
```

For domain LoRAs also include task-specific evaluation.

A gain in one metric must not silently hide severe regression in another.

---

## 19. Serving

Preferred serving architecture:

```text
Qwen3-8B
    +
vLLM
    +
Multi-LoRA
```

Keep one base model resident where possible.

Load/switch capability LoRAs per request.

Do not start a separate full 8B model process for every Agent unless isolation is explicitly required.

Track:

- active LoRA,
- KV-cache usage,
- request concurrency,
- context length,
- GPU memory headroom.

---

## 20. Context Discipline

Do not pass entire Agent transcripts downstream by default.

Pass:

- final structured result,
- evidence references,
- uncertainty,
- essential reasoning summary,
- timestamps.

Long unfiltered context increases:

- latency,
- KV-cache usage,
- hallucination risk,
- evidence duplication,
- prompt contamination.

---

## 21. Financial Data Principles

Separate:

```text
static/domain knowledge
```

from:

```text
dynamic/current facts
```

Static/domain knowledge may be learned through SFT/LoRA.

Dynamic facts should normally be retrieved.

Examples of dynamic facts:

- market price,
- latest earnings,
- current prediction-market odds,
- breaking news,
- current macro releases.

---

## 22. Prediction-Market Safety Boundary

The core research objective is forecasting and decision-support quality.

Do not add automatic real-money wagering, deposit, withdrawal, or autonomous execution logic as an incidental feature.

Keep:

```text
forecasting
```

separate from:

```text
capital allocation / execution
```

unless the latter is an explicitly reviewed subsystem.

Backtests and simulated execution should be clearly labeled as simulations.

---

## 23. Code Quality

Before changing training or evaluation code:

1. inspect the existing implementation,
2. preserve public interfaces where possible,
3. add or update tests,
4. avoid unrelated refactors,
5. document behavior-changing config changes.

Prefer:

- small modules,
- typed interfaces,
- deterministic preprocessing,
- explicit schemas,
- reproducible commands.

Avoid hidden global state.

---

## 24. Testing Priorities

High-priority tests:

```text
dataset timestamp filtering
temporal leakage prevention
event-group split integrity
probability range validation
reward correctness
Brier/log-loss correctness
LoRA loading/switching
tool-call schema validation
Agent routing
historical replay determinism
```

Any bug affecting timestamp filtering or evaluation validity is release-blocking.

---

## 25. Definition of Done

A feature is not complete merely because it runs.

For model/training changes, "done" means:

```text
training completes
+
evaluation completes
+
metrics are recorded
+
no temporal leakage is detected
+
results are reproducible
+
config is versioned
```

For Agent changes:

```text
routing works
+
output schema validates
+
tool use is correct
+
failure cases are handled
+
latency/resource impact is measured
```

---

## 26. Default Decision Rules

When uncertain:

- Prefer **Qwen3-8B** over adding another foundation model.
- Prefer **RAG/API** over memorizing changing facts.
- Prefer **tools** over mental arithmetic.
- Prefer **capability LoRAs** over one-LoRA-per-Agent.
- Prefer **SFT before RL**.
- Prefer **proper scoring rules** over binary correctness rewards.
- Prefer **time-based evaluation** over random splits.
- Prefer **reproducibility** over one-off benchmark gains.
- Prefer **calibration** over unjustified confidence.
- Prefer **simple reward functions** before complex reward shaping.
- Prefer **one validated change at a time**.

---

## 27. Current Technical Direction

Unless explicitly changed, assume:

```text
Foundation:
Qwen/Qwen3-8B-Base

Local GPU:
RTX 3090 24GB

Fine-tuning:
4-bit QLoRA

Initial LoRA rank:
r = 16 or 32

Serving:
vLLM + Multi-LoRA

Core domain:
finance + event forecasting + prediction markets

Core datasets:
Prophet Arena Subset 1200
ForecastBench
Prediction Market Analysis

Core training:
SFT → optional preference optimization → GRPO/RLVR

Primary forecasting metrics:
Brier Score + Log Loss + ECE

Primary architectural principle:
task-oriented Agents + capability-oriented LoRAs
```

When proposing changes, optimize for this architecture rather than introducing unrelated infrastructure.
