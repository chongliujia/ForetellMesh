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

User-confirmed product objective (2026-09-19): build multi-agent automated **simulated trading** for prediction markets, starting with **USD 100 of virtual capital**, and evaluate net profitability and drawdown after explicit execution costs. Begin with an offline, unmodified Qwen3-8B-Base baseline; do not start further fine-tuning or RL until its simulation results identify a need. SFT, LoRAs, and RL are optional improvements, not prerequisites for the first trading simulation.

Historical user requirements (2026-09-19, activity requirement superseded below): include a price mean-reversion strategy and require at least one actual simulated buy/sell fill every rolling 7 days. Preserve earlier experiments and their original activity audits; never reinterpret them using new requirements. Mean-reversion price targets are not event-resolution probabilities.

User-authorized RL experiment (2026-09-20): a small allocation-policy PPO prototype may train on the admitted training partition while Qwen3-8B and capability adapters remain frozen. Keep forecast scoring separate from net-equity trading rewards. Freeze all seeds/checkpoints before development evaluation; do not train on the already evaluated validation contracts or open the final test. The first prototype uses deterministic price features and is not evidence of improved LLM/multi-agent forecasting.

Latest user direction (2026-09-20): remove the requirement to trade every seven days and remove all cadence action guards, activity penalties and cadence-based promotion gates from new allocation experiments. Allow indefinite cash holding when trading has no net advantage. Explicitly expose transaction fees as a policy variable and evaluate different fee assumptions. Optimize after-cost net terminal equity under the existing cash/exposure/risk limits, with a $100 cash-only baseline; do not double-count fees already charged in the ledger. This is allocation-policy learning, not a replacement for proper forecast scoring. Historical cadence configurations remain for reproducibility only. Simulated execution only; never fabricate profitable fills or promote in-sample gains as generalization.

User-confirmed team-learning extension (2026-09-20): borrow testable ideas from futures into the prediction-market team, with autonomous relationship discovery and flexible holding/review horizons. Support early sale or redemption at resolution; redemption is not a fabricated exchange fill. Every filled trade, winning, losing or flat, must become an auditable team-learning experience; open trades retain their process without premature outcome grading. Learn only after feedback is available. Keep model-generated lessons as unverified hypotheses, freeze methods during independent evaluation, and consider fine-tuning only after independently verified effectiveness. No automatic fine-tuning after each trade. The opt-in lifecycle implementation and bounded smoke protocol are documented in `docs/team_lifecycle_trade_experience_v1.md`; existing experimental ledgers remain immutable.

User-authorized experience verification (2026-09-21): before more model training, enforce ledger-owned trade facts and check proposed procedural changes before decision-memory admission. Preserve every trade and failed reflection, but do not admit format-only legacy reflections or free-form assessments as decision knowledge. Check side, PnL class, exit type, event-probability meaning, cited facts, and an explicit comparison protocol; retain rejected/no-lesson records. A separate Base critic can miss errors and is not a truth oracle. Checked methods remain exploratory, and reused development comparisons must not be relabeled independent validation. Current protocol: `docs/team_fact_gate_v1.md`.

User-authorized executable method research (2026-09-21): have the team translate its candidate methods into concrete tool-executable experiments before further independent validation. Freeze the proposed computation and comparison criteria before tool scoring, preserve missing data and failed plans, and keep forecast targets distinct from net-profit claims. Strategy content remains team-generated. The initial bounded interpreter uses admitted training contract prices and rules; unsupported external evidence must remain missing. Reused training screens are not independent effectiveness evidence and do not authorize fine-tuning or strategy promotion. Protocol: `docs/team_executable_method_v1.md`.

User-authorized staged team research (2026-09-21): split relation design, required-data selection and calculation into separate bounded role calls, with deterministic availability and binding checks between them. Keep unsupported external variables explicit, require calculations to use the declared inputs without silently changing the hypothesis or contracts, and preserve early stops and failed repairs. Freeze any assembled method before scoring; structural success is not predictive or trading effectiveness. Protocol: `docs/team_staged_method_v1.md`.

User-authorized variable/source contracts (2026-09-21): have the research role declare each variable's identity, meaning, quantity, unit, contract role and historical lag before data selection. Bind each variable to a program-owned source with matching quantity and unit, backed by data hashes, or explicitly report it missing. Do not silently convert units, substitute price proxies or rewrite research declarations to fit the available source. This checks declared metadata and handoffs, not arbitrary natural-language truth. Keep old runs and protocols reproducible. Protocol: `docs/team_variable_contract_v1.md`.

User-authorized autonomous research cycle (2026-09-21): expand the research catalogue to all proof-admitted training records, without inheriting old fixed trade-entry observation restrictions. Run at most ten retained attempts on the frozen shared Base: model-selected contract retrieval, typed relation design, source/coverage checks, frozen executable testing, and chronological feedback including worthwhile missing-data requests. Failures, abstentions and duplicates consume budget; do not select only winners or restart based on scores. Within-cycle adaptation is training, not independent validation. Freeze a qualified method before a new independent simulated comparison; no candidate means no efficacy claim or fine-tuning. Protocol: `docs/team_autonomous_research_v1.md`.

User-authorized research handoff repair (2026-09-21): turn retrospective next-focus text into an explicit next research task with admitted contract IDs and an objective, consume it directly in the next attempt, validate contract routing, include concrete failure feedback, and allow bounded duplicate repair. Keep semantic correctness unproven when only routing validates. Diagnose lifecycle-aware, actual-trade observation clocks separately from the frozen-model handoff comparison, preserving the original evaluator and simulation rules until a new protocol is explicitly tested. Do not infer trading cadence from observation spacing. Protocol: `docs/team_research_handoff_v1.md`.

User-authorized multi-contract research expression (2026-09-21): permit one prediction target to consume multiple explicitly named variables from different admitted contracts. Bind every variable to a market ID, measured quantity, unit, lag and checked source; preserve unavailable external observations. Produce one forecast per target/time, never count inputs or repeated target bindings as independent observations, and preserve event-group scoring, temporal cutoffs and freeze-before-score. Keep old single-peer protocols reproducible. Protocol: `docs/team_multi_input_v1.md`.

User-authorized semantic consistency diagnostic (2026-09-21): check whether the proposed hypothesis, declared measurements and bound target describe the same test before data selection. Do not ban cross-topic associations or confuse missing evidence with semantic contradiction. Use the same frozen Base only as a fallible reviewer, check fixed known-mismatch and explicit cross-topic controls before permitting this bounded run to proceed, and allow at most one recorded substantive revision. Keep controls out of researcher inputs and scoring; preserve rejected proposals, abstentions and errors. A passed semantic screen is neither truth nor predictive effectiveness. Actual v1 controls matched 3/3, but two live attempts failed task binding before semantic review; do not claim a demonstrated model revision loop. Protocol: `docs/team_semantic_consistency_v1.md`.

User-authorized task negotiation (2026-09-21): before relation design, allow the research member to account for every assigned contract and explicitly propose a narrower, replaced or clarified task, or abstain. The coordinator must accept the proposal, retain the original task, or abstain; never silently discard unused IDs or fabricate a relation to meet routing constraints. Preserve original/proposed/effective tasks, per-contract reasons, decisions and errors. Retrieve full rules for effective IDs and retain exact task binding, semantic/source checks and freeze-before-score. One negotiation per attempt, fixed budget, frozen Base; recorded task changes are not evidence of learning effectiveness. The first diagnostic starts from the registered prior training binding failure. Protocol: `docs/team_task_negotiation_v1.md`.

User-authorized review authority correction (2026-09-21): in new `executable_review_boundary_v1` research, same-Base semantic review is advisory only. Neither topic-distance opinions, uncertain/invalid reviews nor fixed probe results may veto empirical investigation or force proposal rewrites. Deterministic task binding, target type/horizon, exact source quantity/unit, causal input times, calculation bounds and freeze-before-score retain authority. Record a program-owned execution contract and preserve all critic opinions as unverified; arbitrary prose truth and predictive effectiveness remain unproven. Keep raw failed outputs and old gate protocols reproducible. Do not use these opinions or routing improvements as fine-tuning supervision or promotion evidence. Protocol: `docs/team_review_boundary_v1.md`.

User-authorized data-catalogue discovery (2026-09-21): new `executable_data_catalogue_v1` research may query program-owned source measurement/unit/time metadata and cutoff-filtered marginal input coverage before task negotiation and variable design. Keep queries bounded and training-only; expose no price values, outcome labels or future records. Coverage is offline design metadata, never a feature at earlier replay times. Let the team choose a new hypothesis about available measurements, request a precisely defined missing source, or abstain; never silently relabel external variables as prices. Preserve task/source/time checks, advisory critic status, all failures, old source-hidden protocols and sealed holdouts. No automatic fine-tuning. Protocol: `docs/team_data_catalogue_v1.md`.

User-authorized context comparison (2026-09-21): diagnose the archived two-contract task being answered with an old four-contract assessment. Hold the current task, directory, instruction, schema, model and greedy generation fixed; compare full failure history with deterministic structured failure receipts and experiment fingerprints. Preserve full history and every output in audit artifacts. Use a fixed ABBA role-call budget, count initial and repaired responses separately, and do not treat repeats as independent cases. Only past records may build the compact input; do not inject the current known failure or other arms' outputs. This is an isolated interface diagnostic, not a production-loop change, forecast evaluation or fine-tuning trigger. Protocol: `docs/team_context_comparison_v1.md`.

User-authorized next experiment (2026-09-20): connect actual multi-agent analysis to the allocation policy and retry the simulation. Generate frozen Research / market Quant / Game Theory / Forecast outputs through LangGraph on the single shared Qwen3-8B-Base, then compare paired price-control and Agent-informed PPO policies on training data only. Agent outputs become visible after measured generation latency, expire explicitly, and are never backfilled into earlier states. Keep failed outputs missing. Preserve fees, cash-only baseline and absence of required trade frequency. Periodic research refresh does not imply periodic trading. Freeze all policies before evaluation; final test remains sealed.

User direction on turnover (2026-09-20): prioritize low-turnover prediction-market strategies. Research refresh must not itself trigger trading. Review the short-horizon hourly re-entry and 48-hour holding assumptions against an event-driven holding design before further allocation experiments. Preserve cash holding, explicit costs, risk limits, historical runs and sealed holdouts. No new hard trade-frequency requirement was specified. Design note: `docs/event_driven_allocation_design.md`.

User clarification on trading horizon (2026-09-20): normal prediction-market trading should operate over several days to several weeks. Separate routine allocation review from hourly risk monitoring; research refresh alone must not imply trading. Use explicit risk exceptions, permit long cash holding, and do not infer a mandatory minimum number of trades or an unconditional minimum holding period.

User-authorized signal diagnosis (2026-09-20): prioritize independent, time-bounded event evidence and forecast quality against the contemporaneous market baseline before another allocation algorithm change. A frozen Base train-only paired evidence/price pilot is authorized. Do not force forecasts to deviate from market prices, promote in-sample gains, or reopen sealed holdouts. Preserve the days-to-weeks trading baseline and its original results.

User-authorized signal/allocation correction (2026-09-20): forecasts without demonstrated independent value must not automatically authorize entries. Separate immutable signal provenance from method qualification; market-reference prices are not independent event valuations, and price disagreement or self-reported confidence is not qualification. New event-allocation calls default to no signal-driven entry without admission. Explicit legacy bypasses exist only for historical research reproduction. Cash preservation from rejecting unqualified signals is a mechanism check, not evidence of profitable forecasting; the underlying evidence and forecasting gaps remain to be addressed.

User-authorized timely evidence work (2026-09-20): supplement the same fixed training pilot with dated, time-filtered official macro releases, then compare frozen Base forecasts. Preserve original controls and sealed holdouts. Current-period narrative fields and explicitly reviewed reissues are separate from revised historical tables; retain actual retrieval times and bounded source-vintage claims. Fresher evidence alone does not grant trading qualification. Protocol and sources: `docs/timely_macro_evidence.md`.

User-authorized forecast SFT direction (2026-09-20): prepare qualified real prediction supervision, then run a small capability LoRA comparison on Qwen3-8B-Base. The user explicitly selected real-data supplementation before training, rather than synthetic probability warm-up. Historical expert probabilities are candidate teacher judgments, not true event probabilities or realized outcomes. Pair targets with independently time-bounded student inputs, document teacher/source provenance, and freeze event/time-separated development evaluation before training. Do not copy teacher reports into the student prompt, fabricate historical rationales, relabel existing holdouts, or start further allocation PPO as a substitute. Candidate source inventory is not training admission; final test remains sealed.

Latest user clarification (2026-09-20): RSI means Recursive Self-Improvement, not the Relative Strength Index. The multi-agent system is a simulated research/trading team. The intended loop is broad prediction-market historical exploration → team decisions and simulated execution → externally checked feedback and retrospectives → qualified experience-derived training data → capability fine-tuning of the shared Qwen3-8B → controlled evaluation of the updated team. Do not substitute macro-only expert-probability collection, manually prescribed stock-analysis relationships, or training only a small allocation PPO for this objective. Agents should propose and test cross-contract relationships and research methods. Generated analyses are candidate experience, not automatically true supervision. Never present retrospective information as an original ex-ante decision. Separate memory/prompt improvements from parameter improvements, retain failures and cash decisions, and preserve event/time-separated evaluation and the sealed final test. Historical macro experiments and SPF collection remain reproducible auxiliary artifacts. Begin with a bounded iteration, not indefinite self-modification, automatic checkpoint promotion, or real-money execution. Implementation gaps and the proposed protocol are recorded in `docs/team_recursive_learning.md`.

Latest user correction (2026-09-20): do NOT fine-tune after every team-learning round. Keep the shared model and adapters frozen while the team explores, tests relationships, improves its methods/memory and collects feedback. Fine-tuning is optional and conditional on independently verified effectiveness of the team-discovered improvement on fresh, event/time-separated cases. Experience count, valid JSON, a profitable episode, or a lower fitting loss is not sufficient. First freeze and compare the learned team method against the unchanged team under matched evidence, fees and risk constraints; retain negative results, assess costs/drawdown and relevant forecasting diagnostics, and keep the final test sealed. Only after reproducible effectiveness is established may a separate bounded capability fine-tuning experiment consolidate the qualified experience. The current exploration entry point must never automatically launch training. The already completed small LoRA pilot remains a historical artifact, not evidence satisfying this condition or a default checkpoint. This supersedes the automatic experience-to-LoRA step above.

Latest user target (2026-09-20): the first milestone is 10% cumulative after-cost simulated return: one continuous USD 100 virtual account reaching USD 110, superseding the floated USD 200 target. No deadline or annualized/monthly return was specified. Preserve exposure limits and optional cash holding; do not force trades, increase risk or repeatedly tune on evaluation data to hit the target. Do not aggregate independently reset episode accounts as one portfolio. Report terminal settled cash, costs and drawdown; reaching the milestone alone does not demonstrate a reproducible team-discovered improvement or authorize automatic fine-tuning.

Completed frozen method-memory pilot (2026-09-20): `docs/team_method_memory_v1.md` records 6 learning cases followed by 6 fresh-event paired cases. All 18 episodes replayed; control ended at $100.600698 and memory at $96.000002, with decision coverage 6/6 versus 4/6 and worse common-covered forecast scores. No effectiveness or fine-tuning admission. The 9 development catalogue event groups in `runs/team_method_memory_comparison_v1/consumed_development.json` have now been exposed and must not be reused as fresh confirmation, including catalogue peers. Preserve the negative result and actual 2026 memory creation timestamps. Future independent comparisons need newly frozen, unexposed events.

Completed preserved-reference fresh recheck (2026-09-20): `docs/team_reference_recheck_v1.md` records the unchanged v5 workflow versus the existing interface candidate on 3 previously boundary-purged, newly reviewed nonpolitical event groups. Original cohorts remain unchanged. Both primary one-hour accounts stayed at $100 with zero fills; post-hoc fixed 4/24-hour order diagnostics ended at about $96/$94 and do not change the default policy. No profitability or RSI improvement established, no fine tuning. The three groups in `runs/team_reference_recheck_diagnostic_v1/consumed_development.json` are now used and cannot serve as fresh confirmation. Preserve the $106.523346 original result and its source snapshot as a candidate reference, not a promoted default.

User-authorized structured-history integration (2026-09-21): after the preserved one-case context comparison, run a bounded two-attempt full research workflow with frozen Qwen3-8B-Base. The opt-in `executable_structured_history_v1` projects past failures, experiment fingerprints and tool constraints into model context; current tasks remain authoritative and complete raw records remain in audit artifacts. Preserve current-role bounded repair, deterministic task/source/time checks, duplicate rejection and pre-score freezing. This is context management, not parameter learning or independent effectiveness evidence. No new holdouts, automatic fine-tuning, strategy promotion or real execution. Protocol: `docs/team_structured_history_v1.md`.

User-authorized grounded research revision (2026-09-21): continue the preserved unscored joint-input failure with two frozen attempts. Refresh coverage for the exact negotiated task, separate deterministic tool facts from unverified hypotheses and reviewer advice, and require reflection citations to concrete failures plus a proposed change. Preserve autonomous contract/method selection, historical causality, missing sources, duplicate rejection and pre-score freezing. Citation validity is not semantic correctness or effectiveness; no fine-tuning, new holdouts or trading. Opt-in protocol: `docs/team_grounded_revision_v1.md`.

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

At the system level, simulated net PnL, capital usage, drawdown, and execution feasibility are primary product outcomes. Keep Brier/calibration as forecasting diagnostics. Compare the unmodified Base system with cash/no-trade and fixed rule baselines before training. Do not equate better forecast metrics with profitable trading, or a profitable small backtest with established out-of-sample skill.

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

The product objective includes explicitly authorized automated **simulated** trading with USD 100 initial virtual capital. Forecasting and decision quality support this objective.

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
