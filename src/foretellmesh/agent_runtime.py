"""Bounded serial orchestration over supplied, time-validated evidence.

The backend receives only role instructions and a ForecastInput payload. It
never receives a ForecastRecord or outcome. Retrieval and production serving
are separate integrations; this module does not claim source authenticity.
"""
from copy import deepcopy
from types import SimpleNamespace
import json
import re
import time
from typing import Protocol, TypedDict

from .capabilities import route_plan
from .data import strict_json
from .probability_tools import execute_probability_tool
from .quant_state import build_quant_state
from .schema import ForecastInput, ValidationError, fields, iso, nonempty, parse_record, probability, timestamp
from .sft_data import INSTRUCTION, validate_response
from .synthetic_sft import canonical_hash

PROMPTS = {
    "research": "Select relevant supplied evidence. Return JSON with evidence_ids, counter_evidence_ids, unknowns, observation_time. ID arrays must be disjoint and refer to supplied evidence. Do not invent facts or fetch current information.",
    "risk": "Identify distinct downside scenarios and missing variables from supplied evidence. Return JSON with risks (array of {scenario, evidence_ids}), unknowns, observation_time. Evidence IDs must exist; an empty list denotes an explicitly hypothetical risk.",
    "forecast": INSTRUCTION,
    "critic": "Review the supplied forecast for ignored evidence, base rates, correlated evidence and unjustified certainty. Do not automatically shrink toward 0.5. Return JSON with accept (boolean), revised_probability (null if accepted), rationale, evidence_ids, unknowns, observation_time. A revision requires a rationale and supporting evidence IDs. Do not invent sources.",
    "quant": "Select one deterministic probability tool using only supplied numerical inputs. Return JSON with name and arguments. Tools: bayes_binary(prior, sensitivity, false_positive_rate), weighted_probability(probabilities, weights). All values are probabilities; mixture weights sum to one. Never supply Python or shell code.",
}

OUTPUT_PROTOCOLS = {
    "baseline_v1": "",
    "plain_json_v1": (
        "\nOutput protocol: Return exactly one JSON object, beginning with { and ending with }. "
        "Do not use Markdown or code fences. Do not add any text before or after the object. "
        "Stop after that object; do not continue with another task, input, or output."
    ),
}

# Opt-in prompt revision. The v1 prompts and all output validators stay intact.
OUTPUT_PROTOCOLS["grounded_json_v2"] = OUTPUT_PROTOCOLS["plain_json_v1"]
OUTPUT_PROTOCOLS["grounded_json_v3"] = OUTPUT_PROTOCOLS["plain_json_v1"]
GROUNDING_INSTRUCTION = (
    "\nDistinguish supplied facts and parameters, quantities derivable under the stated assumptions, "
    "genuinely missing inputs, and unresolved future outcomes. A future outcome being unknown does not "
    "make its supplied probability or prior unknown. Do not call an explicitly supplied or derivable "
    "quantity missing. Preserve actual sampling uncertainty and unverified assumptions; an empirical "
    "frequency need not equal the population probability. Check upstream claims against the original "
    "input instead of copying their unknowns blindly. List only relevant, specific unknowns; [] is valid "
    "when none are identified. Do not invent unsupported hazards."
)


def agent_instruction(role: str, output_protocol: str) -> str:
    if role not in PROMPTS or output_protocol not in OUTPUT_PROTOCOLS:
        raise ValidationError("unknown role or output protocol")
    instruction = PROMPTS[role] + OUTPUT_PROTOCOLS[output_protocol]
    if output_protocol in ("grounded_json_v2", "grounded_json_v3"):
        if role != "quant":
            instruction += GROUNDING_INSTRUCTION
            if output_protocol == "grounded_json_v3":
                instruction += (
                    "\nunknowns must be a JSON array of strings, never an object, dictionary, "
                    "key-value mapping, or array of objects. If the question specifies a vocabulary, "
                    "put only applicable names as strings in that array. Keep the role's exact output fields."
                )
        if role == "forecast":
            instruction += (
                "\nThe event field must copy the ENTIRE input.question string verbatim, including every "
                "newline and any appended numbers or JSON parameters. Escape newlines and quotes as JSON "
                "requires. Do not summarize, shorten, or drop the parameter suffix."
            )
    return instruction


def repair_feedback(role: str, output_protocol: str, context: ForecastInput, error: str) -> dict:
    feedback = {"validation_error": error,
                "instruction": "Return a complete corrected JSON object matching the requested schema."}
    if output_protocol in ("grounded_json_v2", "grounded_json_v3") and role == "forecast":
        # Only immutable model-visible metadata is repeated. No outcome, oracle
        # probability, or modified model answer is introduced into the retry.
        feedback["required_input_copies"] = {"event": context.question,
                                             "observation_time": iso(context.observation_time)}
        feedback["instruction"] += " Copy required_input_copies exactly; generate the other fields from the supplied evidence."
    return feedback

RESPONSE_TRANSPORTS = {"strict_json", "single_json_fence"}


def decode_agent_response(encoded: str, transport: str) -> tuple[dict, str]:
    """Accept only the entire object or an entire explicitly JSON fenced block.

    Never extract an object from surrounding prose or select among candidates.
    Parsing and semantic/schema validation remain separate, mandatory steps.
    """
    if transport not in RESPONSE_TRANSPORTS:
        raise ValidationError("unknown response transport")
    if transport == "single_json_fence":
        match = re.fullmatch(r"```json[ \t]*\r?\n([\s\S]*?)\r?\n```", encoded.strip())
        if match:
            return strict_json(match.group(1)), "json_fence"
    return strict_json(encoded), "raw_json"


class AgentBackend(Protocol):
    """Implementations must honor the requested adapter, or raise an error."""
    available_adapters: set[str]

    def generate(self, request: dict) -> dict | str: ...


def validated_input(value: ForecastInput) -> ForecastInput:
    if not isinstance(value, ForecastInput):
        raise ValidationError("AgentRunner accepts ForecastInput only; labels must remain outside")
    return parse_record({"sample_id": "input", "dataset_source": "input", "dataset_version": "1",
                         "event_id": "input", "event_group_id": "input", "label": None,
                         **value.to_payload()}).forecast_input


def string_list(value, context: str) -> list[str]:
    if not isinstance(value, list) or len(value) > 50:
        raise ValidationError(f"invalid {context}")
    for item in value:
        nonempty(item, context)
    if len(set(value)) != len(value):
        raise ValidationError(f"duplicate {context}")
    return value


def validate_agent_output(role: str, value: dict, context: ForecastInput) -> dict:
    known = {e.evidence_id for e in context.evidence}
    def refs(items):
        if set(string_list(items, "evidence references")) - known:
            raise ValidationError("unknown evidence reference")
    if role == "forecast":
        return validate_response(value, SimpleNamespace(forecast_input=context))
    if role == "quant":
        execute_probability_tool(value)
        return value
    required = {
        "research": {"evidence_ids", "counter_evidence_ids", "unknowns", "observation_time"},
        "risk": {"risks", "unknowns", "observation_time"},
        "critic": {"accept", "revised_probability", "rationale", "evidence_ids", "unknowns", "observation_time"},
    }
    fields(value, required[role], role + " output")
    if timestamp(value["observation_time"], "agent observation") != context.observation_time:
        raise ValidationError("agent observation time changed")
    string_list(value["unknowns"], "unknowns")
    if role == "research":
        refs(value["evidence_ids"])
        refs(value["counter_evidence_ids"])
        if set(value["evidence_ids"]) & set(value["counter_evidence_ids"]):
            raise ValidationError("research evidence roles overlap")
    elif role == "risk":
        if not isinstance(value["risks"], list) or len(value["risks"]) > 20:
            raise ValidationError("invalid risk list")
        for risk in value["risks"]:
            fields(risk, {"scenario", "evidence_ids"}, "risk")
            nonempty(risk["scenario"], "risk scenario")
            refs(risk["evidence_ids"])
    elif role == "critic":
        if type(value["accept"]) is not bool:
            raise ValidationError("critic accept must be boolean")
        refs(value["evidence_ids"])
        nonempty(value["rationale"], "critic rationale")
        if value["accept"]:
            if value["revised_probability"] is not None:
                raise ValidationError("accepting critic cannot also revise")
        else:
            probability(value["revised_probability"])
            if not value["evidence_ids"]:
                raise ValidationError("critic revision requires evidence")
    return deepcopy(value)


class ExecutionState(TypedDict):
    """Per-invocation state; contains model-visible inputs and validated results only."""
    context: ForecastInput
    current: ForecastInput
    plan: dict
    summaries: dict
    trace: list[dict]
    prediction: dict | None
    quant_state: dict | None
    tool_trace: list[dict]
    result: dict | None


class AgentRunner:
    def __init__(self, config: dict, backend: AgentBackend, *, output_protocol: str = "baseline_v1",
                 response_transport: str = "strict_json"):
        if not isinstance(output_protocol, str) or output_protocol not in OUTPUT_PROTOCOLS:
            raise ValidationError("unknown output protocol")
        self.config, self.backend = deepcopy(config), backend
        self.output_protocol = output_protocol
        if not isinstance(response_transport, str) or response_transport not in RESPONSE_TRANSPORTS:
            raise ValidationError("unknown response transport")
        self.response_transport = response_transport

    def run(self, context: ForecastInput, *, workflow: str = "research_forecast", mode: str = "base",
            capability_scope: set[str] | None = None) -> dict:
        state = self._initialize(context, workflow, mode, capability_scope)
        self._execute_quant(state)
        for step in state['plan']['steps']:
            if state['result'] is not None:break
            self._execute_step(state, step)
        return self._finish(state)

    def _initialize(self, context: ForecastInput, workflow: str, mode: str,
                    capability_scope: set[str] | None) -> ExecutionState:
        context = validated_input(context)
        plan = route_plan(self.config, workflow, mode, capability_scope=capability_scope)
        missing = set(plan["requires_loaded_adapters"]) - set(self.backend.available_adapters)
        if missing:
            raise ValidationError("requested adapters are not loaded: " + ", ".join(sorted(missing)))
        limits = self.config["limits"]
        if plan["max_model_calls"] > limits["max_model_calls"]:
            raise ValidationError("workflow exceeds model-call budget")
        return {'context': context, 'current': context, 'plan': plan, 'summaries': {}, 'trace': [],
                'prediction': None, 'quant_state': None, 'tool_trace': [], 'result': None}

    def _execute_quant(self, state: ExecutionState) -> None:
        if state['plan'].get('deterministic_steps'):
            started = time.perf_counter()
            try:
                quant_state = build_quant_state(state["context"])
            except (ValueError, TypeError) as exc:
                state['result'] = {'status': 'failed', 'stage': 'quant', 'error': 'invalid_tool_evidence',
                        'prediction': None, 'stages': {}, 'trace': [], 'model_calls': 0,
                        'tool_trace': [{'tool': 'probability_state_v1', 'status': 'failed',
                                        'error': str(exc), 'seconds': time.perf_counter() - started}]}
                return
            state['quant_state'] = quant_state
            state['tool_trace'].append({'tool': 'probability_state_v1', 'status': 'completed',
                               'seconds': time.perf_counter() - started, 'output_sha256': canonical_hash(quant_state)})

    def _fail(self, state: ExecutionState, role: str, error: str) -> None:
        result = {"status": "failed", "stage": role, "error": error, "prediction": None,
                  "stages": deepcopy(state['summaries']), "trace": state['trace']}
        if state['quant_state'] is not None:
            result.update(tool_result=state['quant_state'], tool_trace=state['tool_trace'], model_calls=len(state['trace']))
        state['result'] = result

    def _execute_step(self, state: ExecutionState, step: dict) -> None:
        if state['result'] is not None:return
        context, current = state['context'], state['current']
        summaries, trace = state['summaries'], state['trace']
        prediction, quant_state = state['prediction'], state['quant_state']
        limits = self.config['limits']
        role = step["agent"]
        # Risk and critic inspect the original evidence, including evidence
        # omitted by Research. Forecast sees only the selected union.
        visible = current if role == "forecast" else context
        upstream = {}
        if quant_state is not None:
            upstream['quant'] = quant_state
        if role == "forecast":
            upstream = {k: summaries[k] for k in ("research", "risk") if k in summaries}
        elif role == "critic":
            upstream = {"forecast": prediction}
            if "risk" in summaries:
                upstream["risk"] = summaries["risk"]
        request = {"agent": role, "adapter": step["adapter"], "instruction": agent_instruction(role, self.output_protocol),
                   "input": visible.to_payload(), "upstream": upstream}
        error = None
        for attempt in range(limits["max_repairs"] + 1):
            attempt_request = deepcopy(request)
            if error:
                attempt_request["repair"] = repair_feedback(role, self.output_protocol, visible, error)
            serialized = json.dumps(attempt_request, ensure_ascii=False, allow_nan=False)
            if len(serialized) > limits["max_input_chars"]:
                return self._fail(state, role, "input_budget_exceeded")
            started = time.perf_counter()
            try:
                output = self.backend.generate(deepcopy(attempt_request))
            except Exception as exc:
                trace.append({"agent": role, "adapter": step["adapter"], "attempt": attempt,
                              "seconds": time.perf_counter() - started, "status": "backend_error", "error_type": type(exc).__name__})
                return self._fail(state, role, "backend_error")
            elapsed = time.perf_counter() - started
            try:
                encoded = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, allow_nan=False)
                if len(encoded) > limits["max_output_chars"]:
                    raise ValidationError("output_budget_exceeded")
                parsed, decoded_transport = decode_agent_response(encoded, self.response_transport)
                value = validate_agent_output(role, parsed, visible)
            except (ValueError, TypeError, OverflowError) as exc:
                error = str(exc)
                trace.append({"agent": role, "adapter": step["adapter"], "attempt": attempt, "seconds": elapsed,
                              "status": "invalid_output", "validation_error": error,
                              "input_sha256": canonical_hash(attempt_request)})
                continue
            trace.append({"agent": role, "adapter": step["adapter"], "attempt": attempt, "seconds": elapsed,
                          "status": "valid", "decoded_transport": decoded_transport,
                          "input_sha256": canonical_hash(attempt_request), "output_sha256": canonical_hash(value)})
            break
        else:
            return self._fail(state, role, "output_schema_failed")
        summaries[role] = value
        if role in ("research", "risk"):
            selected = set(summaries.get("research", {}).get("evidence_ids", [])) | set(summaries.get("research", {}).get("counter_evidence_ids", []))
            if "research" not in summaries:
                selected = {e.evidence_id for e in context.evidence}
            for risk in summaries.get("risk", {}).get("risks", []):
                selected.update(risk["evidence_ids"])
            current = ForecastInput(context.question, context.observation_time,
                                    tuple(e for e in context.evidence if e.evidence_id in selected), context.market)
        elif role == "forecast":
            prediction = value
        elif role == "critic" and not value["accept"]:
            # Keep the original forecast and explicit critique separate; no
            # synthetic evidence list is invented for the revised number.
            prediction = {**prediction, "probability": value["revised_probability"],
                          "unknowns": list(dict.fromkeys(prediction["unknowns"] + value["unknowns"]))}
        state.update(current=current, prediction=prediction)

    def _finish(self, state: ExecutionState) -> dict:
        if state['result'] is not None:return state['result']
        summaries, trace = state['summaries'], state['trace']
        prediction, quant_state = state['prediction'], state['quant_state']
        result = {"status": "completed", "workflow": state["plan"]["workflow"], "mode": state["plan"]["mode"], "prediction": prediction,
                "stages": summaries, "tool_result": quant_state if quant_state is not None else execute_probability_tool(summaries["quant"]) if "quant" in summaries else None,
                "trace": trace, "model_calls": len(trace), "checkpoint_quality_verified": False}
        if quant_state is not None:result['tool_trace'] = state['tool_trace']
        return result
