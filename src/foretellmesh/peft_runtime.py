"""One resident PEFT model, one active adapter, serialized inference requests."""
from contextlib import nullcontext
import json
import math
from threading import Lock
import time

from .schema import ValidationError
from .data import strict_json


def complete_json_object(text: str) -> bool:
    """A generation boundary, never extraction from surrounding text.

    Stop only when the entire completion is one strict JSON object or one
    complete JSON fence. Semantic validation/repair still runs downstream.
    """
    import re
    match = re.fullmatch(r'```json[ \t]*\r?\n([\s\S]*?)\r?\n```', text.strip())
    try:
        return isinstance(strict_json(match.group(1) if match else text), dict)
    except (ValueError, TypeError):
        return False


def render_agent_prompt(request: dict) -> str:
    """Shared byte-for-byte serialization for inference and capability training."""
    body = {"input": request["input"], "upstream": request["upstream"]}
    if "repair" in request:
        body["repair"] = request["repair"]
    return "Task:\n" + request["instruction"] + "\nInput JSON:\n" + json.dumps(body, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\nOutput JSON:\n"


class SharedPeftExecutor:
    """All accesses to a mutable PEFT model must go through this one instance.

    The lock covers selection, the full operation, and restoration. Separate
    executors over the same model would not provide request isolation.
    """
    def __init__(self, model):
        self._model, self._lock = model, Lock()
        self.available_adapters = set(getattr(model, "peft_config", {}))
        if self.available_adapters:
            if not isinstance(model.active_adapter, str):
                raise ValidationError("adapter composition is not supported")
            if any(layer.merged_adapters for layer in model.get_layer_status()):
                raise ValidationError("merged adapters cannot provide an isolated base baseline")
        self._model.eval()
        self._model.requires_grad_(False)

    def execute(self, adapter: str | None, operation):
        if adapter is not None and (not isinstance(adapter, str) or adapter not in self.available_adapters):
            raise ValidationError("adapter not loaded; refusing silent base fallback")
        with self._lock:
            previous = self._model.active_adapter if self.available_adapters else None
            try:
                if adapter is not None:
                    self._model.set_adapter(adapter, inference_mode=True)
                self._model.eval()
                self._model.requires_grad_(False)
                scope = self._model.disable_adapter() if adapter is None and self.available_adapters else nullcontext()
                with scope:
                    return operation(self._model)
            finally:
                if previous is not None:
                    self._model.set_adapter(previous, inference_mode=True)
                self._model.eval()
                self._model.requires_grad_(False)


class PeftTextBackend:
    """Offline Base-model prompt rendering; no tools or retrieval inside generate."""
    def __init__(self, executor: SharedPeftExecutor, tokenizer, *, max_context_tokens=2048, max_new_tokens=512,
                 sampling: dict | None = None, stop_on_json_object: bool = False):
        if (type(max_context_tokens) is not int or type(max_new_tokens) is not int
                or not 1 <= max_new_tokens < max_context_tokens):
            raise ValidationError("invalid generation token budget")
        self.executor, self.tokenizer = executor, tokenizer
        self.max_context_tokens, self.max_new_tokens = max_context_tokens, max_new_tokens
        if type(stop_on_json_object) is not bool:
            raise ValidationError('invalid JSON stopping policy')
        self.stop_on_json_object = stop_on_json_object
        if sampling is not None:
            if (not isinstance(sampling, dict) or set(sampling) != {'temperature', 'top_p', 'top_k'}
                    or any(type(sampling[k]) not in (int, float) or not math.isfinite(sampling[k])
                           for k in ('temperature', 'top_p'))
                    or not 0 < sampling['temperature'] <= 2 or not 0 < sampling['top_p'] <= 1
                    or type(sampling['top_k']) is not int or sampling['top_k'] < 0):
                raise ValidationError('invalid sampling configuration')
        self.sampling = dict(sampling) if sampling is not None else None
        self.available_adapters = set(executor.available_adapters)
        self.last_usage = None

    def generate(self, request: dict) -> str:
        import torch
        self.last_usage = None
        started = time.perf_counter()
        prompt = render_agent_prompt(request)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) + self.max_new_tokens > self.max_context_tokens:
            raise ValidationError("context budget exceeded; silent truncation is forbidden")

        def generate(model):
            device = next(model.parameters()).device
            tokens = torch.tensor([ids], device=device)
            with torch.inference_mode():
                decoding = {'do_sample': False} if self.sampling is None else {'do_sample': True, **self.sampling}
                if self.stop_on_json_object:
                    from transformers import StoppingCriteria, StoppingCriteriaList
                    tokenizer, prompt_length = self.tokenizer, len(ids)
                    class CompleteObject(StoppingCriteria):
                        def __call__(self, input_ids, scores, **kwargs):
                            return torch.tensor([
                                complete_json_object(tokenizer.decode(row[prompt_length:], skip_special_tokens=True))
                                for row in input_ids], dtype=torch.bool, device=input_ids.device)
                    decoding['stopping_criteria'] = StoppingCriteriaList([CompleteObject()])
                result = model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                        **decoding, max_new_tokens=self.max_new_tokens,
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
            completion = result[0, len(ids):]
            text = self.tokenizer.decode(completion, skip_special_tokens=True)
            self.last_usage = {"input_tokens": len(ids), "output_tokens": len(completion),
                               "output_reached_token_limit": len(completion) == self.max_new_tokens,
                               "seconds": time.perf_counter() - started}
            return text
        return self.executor.execute(request["adapter"], generate)
