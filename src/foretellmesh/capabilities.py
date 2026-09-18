"""Task routes are independent of reusable capability adapters."""
from pathlib import Path
import re

from .data import sha256_bytes, strict_json
from .schema import ValidationError, fields, probability

WORKFLOWS = {
    "single_forecast": ("forecast",),
    "research_forecast": ("research", "forecast"),
    "reviewed_forecast": ("research", "risk", "forecast", "critic"),
    "research": ("research",),
    "risk": ("risk",),
    "calculate": ("quant",),
}


def load_capabilities(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = fields(strict_json(raw.decode()), {"schema_version", "base_model", "base_revision", "lora",
                   "capabilities", "agents", "limits", "execution"}, "capability configuration")
    if value["schema_version"] != "1" or value["base_model"] != "Qwen/Qwen3-8B-Base":
        raise ValidationError("unsupported capability configuration/base model")
    if not isinstance(value["base_revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", value["base_revision"]):
        raise ValidationError("base revision must be immutable")
    lora = fields(value["lora"], {"r", "alpha", "dropout", "target_modules"}, "capability LoRA")
    for name in ("r", "alpha"):
        if type(lora[name]) is not int or not 1 <= lora[name] <= 256:
            raise ValidationError("invalid LoRA rank/alpha")
    if not 0 <= probability(lora["dropout"], "dropout") < 1:
        raise ValidationError("dropout must be below one")
    modules = lora["target_modules"]
    if (not isinstance(modules, list) or not modules or any(not isinstance(m, str) for m in modules)
            or len(modules) != len(set(modules))
            or set(modules) - {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}):
        raise ValidationError("invalid target modules")
    caps = value["capabilities"]
    if not isinstance(caps, dict) or not caps:
        raise ValidationError("capabilities must be a nonempty mapping")
    for name, spec in caps.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*_lora", name):
            raise ValidationError("invalid capability name")
        fields(spec, {"skills", "training_routes", "status"}, "capability")
        # This registry is a training plan, not a checkpoint promotion mechanism.
        if spec["status"] != "planned":
            raise ValidationError("checkpoint approval cannot be asserted in the training plan")
        for key in ("skills", "training_routes"):
            if not isinstance(spec[key], list) or not spec[key] or any(not isinstance(x, str) or not x.strip() for x in spec[key]):
                raise ValidationError("capability skills/routes must be nonempty string lists")
    fields(value["agents"], {"research", "quant", "risk", "forecast", "critic"}, "agent routes")
    if any(not isinstance(cap, str) or cap not in caps for cap in value["agents"].values()):
        raise ValidationError("each agent must select one registered capability")
    limits = fields(value["limits"], {"max_repairs", "max_model_calls", "max_input_chars", "max_output_chars"}, "agent limits")
    for key, low, high in (("max_repairs", 0, 2), ("max_model_calls", 1, 12),
                           ("max_input_chars", 100, 100000), ("max_output_chars", 100, 30000)):
        if type(limits[key]) is not int or not low <= limits[key] <= high:
            raise ValidationError("invalid agent budget")
    expected = {"base_model_instances": 1, "concurrent_requests": 1, "adapters_per_request": 1}
    if value["execution"] != expected or any(type(x) is not int for x in value["execution"].values()):
        raise ValidationError("local execution requires one shared base and one adapter per serial request")
    return value, sha256_bytes(raw)


def route_plan(config: dict, workflow: str, mode: str = "base", *, capability_scope: set[str] | None = None) -> dict:
    if workflow not in WORKFLOWS or mode not in ("base", "capability"):
        raise ValidationError("unknown workflow/adapter mode")
    if capability_scope is not None:
        if (mode != "capability" or not isinstance(capability_scope, set) or not capability_scope
                or any(not isinstance(x, str) for x in capability_scope) or capability_scope - config["capabilities"].keys()):
            raise ValidationError("invalid explicit capability scope")
    steps = [{"agent": agent, "capability": config["agents"][agent],
              "adapter": None if mode == "base" or (capability_scope is not None and config["agents"][agent] not in capability_scope) else config["agents"][agent]}
             for agent in WORKFLOWS[workflow]]
    return {"workflow": workflow, "mode": mode, "base_model": config["base_model"],
            "base_revision": config["base_revision"], "steps": steps,
            "max_model_calls": len(steps) * (config["limits"]["max_repairs"] + 1),
            "requires_loaded_adapters": sorted({s["adapter"] for s in steps if s["adapter"]}),
            "checkpoint_quality_verified": False}
