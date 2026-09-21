"""Outcome-likelihood readout of the SAME Qwen base and forecast LoRA.

Train two-token conditional log loss against genuine binary outcomes, never
imitate a model's probability or turn outcomes into 100%-confidence JSON targets.
"""
from copy import deepcopy
import json
import math
import statistics
import time

from .schema import probability, timestamp, fields
from .synthetic_sft import canonical_hash
from .team_learning import require, validate_context


def binary_prompt(request, market_id):
    context = request['input']; mids = validate_context(context)
    require(market_id in mids and request['agent'] == 'forecast', 'invalid probability readout request')
    target = next(m for m in context['markets'] if m['market_id'] == market_id)
    peers = [{'market_id': m['market_id'], 'question': m['input']['question'],
              'market': m['input']['market']} for m in context['markets'] if m['market_id'] != market_id]
    retrieved = request['upstream'].get('retrieved_markets', [])
    require(isinstance(retrieved, list) and len(retrieved) <= 4, 'unbounded retrieved market scope')
    allowed = set(mids)
    for row in retrieved:
        fields(row, {'market_id','event_group_id','question','initialized_at','record_sha256'}, 'retrieved market')
        require(row['record_sha256'] == canonical_hash({k:v for k,v in row.items() if k != 'record_sha256'})
                and timestamp(row['initialized_at'], 'market initialization') <= timestamp(context['observation_time'], 'cutoff'),
                'unbound or future retrieved market')
        require(isinstance(row['question'], str) and row['question'], 'missing retrieved rules')
        allowed.add(row['market_id'])
    # Deterministic numeric summaries bound to the archived complete tool output;
    # no hidden truncation of rule text, evidence, or supervised answers.
    summaries = []
    for result in request['upstream'].get('tools', []):
        require(result['result_sha256'] == canonical_hash({k: v for k, v in result.items() if k != 'result_sha256'}),
                'tool result changed')
        require(timestamp(result['as_of'], 'tool') <= timestamp(context['observation_time'], 'cutoff'), 'future tool result')
        summary = {'query': result['query'], 'source_sha256': result['result_sha256'], 'series': {}}
        for mid, rows in result['series'].items():
            require(mid in allowed, 'tool contains unknown market')
            for r in rows:
                require(timestamp(r['source_time'], 'quote') <= timestamp(r['at'], 'sample')
                        <= timestamp(context['observation_time'], 'cutoff'), 'future summarized quote')
                probability(r['price'])
            prices = [r['price'] for r in rows]
            summary['series'][mid] = {'samples': len(rows), 'first': rows[0] if rows else None,
                'last': rows[-1] if rows else None, 'mean': statistics.fmean(prices) if prices else None,
                'min': min(prices) if prices else None, 'max': max(prices) if prices else None,
                'change': prices[-1]-prices[0] if len(prices) >= 2 else None}
        for key in ('paired_daily_changes', 'pearson_change_correlation', 'interpretation'):
            if key in result: summary[key] = result[key]
        if 'rule_records' in result:
            require(result['query']['tool']=='rules_snapshot','unexpected rule evidence')
            require({r['market_id'] for r in result['rule_records']} == set(result['query']['market_ids']), 'rules scope differs')
            for row in result['rule_records']:
                fields(row, {'market_id','event_group_id','question','initialized_at','record_sha256'}, 'tool rules')
                require(row['market_id'] in allowed and row['record_sha256']==canonical_hash({k:v for k,v in row.items() if k!='record_sha256'})
                    and timestamp(row['initialized_at'],'rules')<=timestamp(context['observation_time'],'cutoff'),'unbound or future rule evidence')
            summary['rule_records']=result['rule_records']
            require(result['relationship_verified'] is False,'unverified semantic claim promoted')
            summary['relationship_verified']=False
        summaries.append(summary)
    data = {'observation_time': context['observation_time'], 'target': target['input'], 'peer_markets': peers,
            'research_hypotheses_not_facts': request['upstream'].get('research', {}).get('hypotheses', []),
            'historical_tools': summaries, 'online_memory': context['memory'],
            'offline_training_lessons_not_evidence': request['upstream'].get('offline_learning_artifacts', [])}
    if 'offline_team_memory' in request['upstream']:
        from .team_method_memory import validate_memory
        data['offline_team_memory_not_evidence']=validate_memory(request['upstream']['offline_team_memory'],context)
    if 'online_trade_lessons' in request['upstream']:
        from .team_trade_experience import validate_online_lessons
        data['online_trade_lessons_not_evidence']=validate_online_lessons(
            request['upstream']['online_trade_lessons'],context['observation_time'])
    if retrieved: data['retrieved_historical_market_rules'] = retrieved
    if 'research_review' in request['upstream']:
        data['research_review_not_independent_validation'] = request['upstream']['research_review']
    return ('Task: Predict whether the target contract resolves Yes using only the supplied time-bounded information. '
            'Research statements and learned lessons are hypotheses, not proven facts. Answer exactly Yes or No. '
            'The probabilities of those answer tokens will be scored; do not explain.\nInput JSON:\n'
            + json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\nAnswer:')


def answer_tokens(tokenizer):
    ids = [tokenizer.encode(s, add_special_tokens=False) for s in ('No', 'Yes')]
    require(all(len(x) == 1 for x in ids) and ids[0] != ids[1], 'single distinct No/Yes tokens required')
    return [x[0] for x in ids]


def binary_probability(logits):
    """CPU audit of the two-class normalized likelihood, stable at extremes."""
    require(len(logits) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in logits), 'invalid binary logits')
    high = max(logits); z = [math.exp(v-high) for v in logits]
    return z[1]/sum(z)


class OutcomeBackend:
    def __init__(self, text_backend, *, forecast_adapter=None, max_scoring_tokens=2048):
        self.text_backend = text_backend; self.forecast_adapter = forecast_adapter
        self.tokenizer = text_backend.tokenizer; self.executor = text_backend.executor
        self.token_ids = answer_tokens(self.tokenizer); self.max_scoring_tokens = max_scoring_tokens
        self.readouts = []

    def generate(self, request):
        if request['agent'] != 'forecast':
            request = deepcopy(request); request['adapter'] = None
            return self.text_backend.generate(request)
        import torch
        forecasts = []
        for market in request['input']['markets']:
            mid = market['market_id']; prompt = binary_prompt(request, mid)
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            require(0 < len(ids) <= self.max_scoring_tokens, 'binary forecast context exceeds fixed budget')
            started = time.perf_counter()
            def score(model):
                tokens = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
                with torch.inference_mode():
                    result = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), logits_to_keep=1, use_cache=False)
                return result.logits[0, -1, self.token_ids].float().cpu().tolist()
            logits = self.executor.execute(self.forecast_adapter, score); p = binary_probability(logits)
            self.readouts.append({'market_id': mid, 'request_sha256': canonical_hash(request),
                'prompt': prompt, 'prompt_sha256': canonical_hash(prompt), 'tokens': len(ids),
                'answer_token_ids_no_yes': self.token_ids, 'logits_no_yes': logits, 'probability': p,
                'adapter': self.forecast_adapter, 'seconds': time.perf_counter()-started})
            forecasts.append({'market_id': mid, 'probability': p, 'consider_trade': True})
        return json.dumps({'forecasts': forecasts,
                           'unknowns': ['Two-answer likelihood readout; calibration and trading edge are unproven.']}, sort_keys=True)
