"""Explicitly scripted, offline integration checks; no model quality claims."""
from copy import deepcopy
from pathlib import Path
import tempfile

from .agent_runtime import AgentRunner
from .capabilities import load_capabilities, route_plan
from .data import sha256_bytes, strict_json
from .evaluation import code_provenance, json_text
from .schema import ValidationError, fields, parse_record


class ScriptedBackend:
    def __init__(self, responses: dict, adapters=()):
        self.responses, self.available_adapters = deepcopy(responses), set(adapters)
        self.calls = []

    def generate(self, request: dict):
        self.calls.append(deepcopy(request))
        values = self.responses.get(request["agent"], [])
        if not values:
            raise ValidationError("scripted fixture exhausted")
        return values.pop(0)


def check_agent_workflow(config_path: Path, fixture_path: Path, output: Path, workflow="reviewed_forecast") -> dict:
    if output.exists():
        raise ValidationError("agent check output already exists")
    config, config_hash = load_capabilities(config_path)
    raw = fixture_path.read_bytes()
    fixture = fields(strict_json(raw.decode()), {"schema_version", "kind", "input", "responses"}, "agent fixture")
    if fixture["schema_version"] != "1" or fixture["kind"] != "scripted_agent_test_only":
        raise ValidationError("only explicitly scripted test fixtures are accepted")
    fields(fixture["input"], {"question", "observation_time", "evidence", "market"}, "fixture input")
    record = parse_record({**fixture["input"], "sample_id": "fixture", "dataset_source": "synthetic",
                           "dataset_version": "1", "event_id": "fixture", "event_group_id": "fixture", "label": None})
    backend = ScriptedBackend(fixture["responses"], config["capabilities"])
    result = AgentRunner(config, backend).run(record.forecast_input, workflow=workflow, mode="capability")
    report = {"schema_version": "1", "kind": "scripted_agent_integration_check", "result": result,
              "plan": route_plan(config, workflow, "capability"), "fixture_sha256": sha256_bytes(raw),
              "config_sha256": config_hash, "code": code_provenance(), "forecasting_metrics": None,
              "limitations": "Scripted responses test plumbing only. No trained LoRA, model inference, retrieval, or forecasting evaluation."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".agent-check-", dir=output.parent) as tmp:
        stage = Path(tmp) / "check"
        stage.mkdir()
        (stage / "report.json").write_text(json_text(report))
        (stage / "requests.json").write_text(json_text(backend.calls))
        stage.rename(output)
    return report
