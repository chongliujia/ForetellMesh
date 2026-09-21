"""Team-authored forecast experiments, evaluated by a bounded arithmetic interpreter.

No eval/exec, orders, parameter fitting or model-written ground truth. A training
screen is only a prerequisite for a later independent, frozen-method experiment.
"""
import ast
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import timedelta
import math
import statistics

from .schema import fields, iso, timestamp, ValidationError
from .synthetic_sft import canonical_hash
from .team_learning import require


PLANNER = '''Turn the supplied tentative method into ONE executable forecast experiment, or explicitly abstain.
Choose the relationship, contracts and calculation yourself; no strategy category is prescribed.
Return ONLY one valid JSON object with exactly two fields: plan and no_plan_reason.
plan must be an actual JSON object or null, never a schema placeholder. no_plan_reason is null when a plan exists.
PLAN has exactly hypothesis, forecast_target, horizon_days, expression, bindings, min_observations, min_improvement.
forecast_target is future_yes_price or resolves_yes. They are DIFFERENT quantities.
For future_yes_price choose horizon_days as an integer 1..14; for resolves_yes use null.
expression is a bounded arithmetic expression that returns your predicted Yes price or event probability in [0,1].
Available inputs: p('target',0) is the latest fresh Yes trade price at observation time; p('target',N) is the
fresh Yes price N calendar days earlier. p('peer',N) works only if a peer is bound. N must be a literal integer 0..30.
Allowed arithmetic: + - * /, parentheses, abs(x), min(x,y), max(x,y), finite numeric constants.
There are no other variables/functions, news, tweet counts, external asset prices, future prices or labels.
The expression is interpreted as data, never run as Python. Division by zero, missing/stale inputs and out-of-range
predictions are failed observations; they are not filled with guessed values or silently clipped.
bindings is a list of 1..4 objects with exactly target and peer, using supplied market IDs; peer may be null.
Every target must differ; each peer must differ from its target. Choose peers based on your hypothesis, not a fixed category.
If expression never reads p('peer',N), ALL bindings must have peer=null. If it does, ALL must supply a peer.
Do not claim a cross-contract relation if your expression reads only the target. Do not pair adjacent catalogue entries arbitrarily.
Use the SAME expression across all bindings. min_observations is an integer 8..100 for future_yes_price,
or 3..100 for resolves_yes (one observation per target). min_improvement is a number >0 and <=1.
The baseline is the current target Yes price. The program compares mean squared future-price error or event Brier,
averaging within each event group and then equally across groups. Positive improvement means lower error than baseline.
The screen also requires at least 3 distinct target event groups and 80% valid coverage; no single lucky trade can pass it.
Prediction windows do not overlap within a target; event resolutions are scored once per target.
This is reused TRAINING data for mechanism diagnosis, NOT independent validation or proof of profit.
The formula must explicitly implement the proposed relationship; do not merely write 'validate correlation'.
If the supplied tools cannot express the required method or needed data is absent, plan=null and explain the gap.
Keep hypothesis/no_plan_reason under 600 characters and expression under 400 characters. Do not invent observations.'''

REVIEWER = '''Review the actual executable experiment. Return ONLY an object with exactly
result_sha256, status, eligible_for_independent_validation, limitation.
Copy the first three fields EXACTLY from the supplied deterministic result. No changing its verdict based on your opinion.
limitation is one short string (1..600 characters) explaining a real evidence gap or next check.
A training error reduction is not independent effectiveness, a future trade-price forecast is not an event probability,
and neither metric is net trading profit. No strategy promotion or model training is authorized by this review.'''


def bounded_text(value, limit=600):
    require(isinstance(value, str) and 0 < len(value.strip()) <= limit, 'missing/bounded experiment text')


def compile_expression(expression):
    """Parse a tiny expression language; reject every unlisted AST node."""
    bounded_text(expression, 400)
    try:
        root = ast.parse(expression, mode='eval').body
    except (SyntaxError, RecursionError) as exc:
        raise ValidationError('invalid arithmetic expression') from exc
    require(sum(1 for _ in ast.walk(root)) <= 96, 'expression exceeds node budget')
    refs = set()

    def visit(node, depth=0):
        require(depth <= 12, 'expression exceeds depth budget')
        if isinstance(node, ast.Constant):
            require(type(node.value) in (int, float) and abs(node.value) <= 1000
                    and math.isfinite(node.value), 'invalid numeric literal')
        elif isinstance(node, ast.UnaryOp):
            require(isinstance(node.op, (ast.UAdd, ast.USub)), 'unsupported unary operation')
            visit(node.operand, depth+1)
        elif isinstance(node, ast.BinOp):
            require(isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)), 'unsupported arithmetic operation')
            visit(node.left, depth+1); visit(node.right, depth+1)
        elif isinstance(node, ast.Call):
            require(isinstance(node.func, ast.Name) and not node.keywords, 'unsupported function call')
            name = node.func.id
            if name == 'p':
                require(len(node.args) == 2 and all(isinstance(n, ast.Constant) for n in node.args),
                        'price requires literal role and lag')
                role, lag = (n.value for n in node.args)
                require(role in ('target', 'peer') and type(lag) is int and 0 <= lag <= 30,
                        'unknown price role or future/unbounded lag')
                refs.add((role, lag))
            else:
                require(name in ('abs', 'min', 'max') and len(node.args) == (1 if name == 'abs' else 2),
                        'unsupported function/arity')
                for arg in node.args: visit(arg, depth+1)
        else:
            raise ValidationError('unsupported expression syntax')
    visit(root)
    require(refs, 'expression must use observable prices')
    return root, sorted(refs)


def calculate(root, values):
    def go(node):
        if isinstance(node, ast.Constant): result = float(node.value)
        elif isinstance(node, ast.UnaryOp):
            result = go(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp):
            a, b = go(node.left), go(node.right)
            if isinstance(node.op, ast.Add): result = a+b
            elif isinstance(node.op, ast.Sub): result = a-b
            elif isinstance(node.op, ast.Mult): result = a*b
            else:
                require(b != 0, 'division by zero')
                result = a/b
        elif node.func.id == 'p': result = values[(node.args[0].value, node.args[1].value)]
        else:
            args = [go(n) for n in node.args]
            result = {'abs': abs, 'min': min, 'max': max}[node.func.id](*args)
        require(math.isfinite(result) and abs(result) <= 1e12, 'nonfinite/unbounded expression result')
        return result
    value = go(root)
    require(0 <= value <= 1, 'prediction outside [0,1]')
    return value


def validate_plan(value, catalogue):
    fields(value, {'plan', 'no_plan_reason'}, 'executable method')
    plan = value['plan']
    if plan is None:
        bounded_text(value['no_plan_reason']); return deepcopy(value)
    require(value['no_plan_reason'] is None, 'plan cannot also abstain')
    fields(plan, {'hypothesis', 'forecast_target', 'horizon_days', 'expression', 'bindings',
                  'min_observations', 'min_improvement'}, 'method plan')
    bounded_text(plan['hypothesis'])
    require(plan['forecast_target'] in ('future_yes_price', 'resolves_yes'), 'unknown forecast target')
    horizon = plan['horizon_days']
    require((type(horizon) is int and 1 <= horizon <= 14) if plan['forecast_target'] == 'future_yes_price'
            else horizon is None, 'forecast horizon/target mismatch')
    _, refs = compile_expression(plan['expression'])
    require(isinstance(plan['bindings'], list) and 1 <= len(plan['bindings']) <= 4, 'unbounded experiment bindings')
    seen = set(); needs_peer = any(role == 'peer' for role, _ in refs)
    for binding in plan['bindings']:
        fields(binding, {'target', 'peer'}, 'method binding')
        target, peer = binding['target'], binding['peer']
        require(isinstance(target, str) and target in catalogue and target not in seen, 'unknown/duplicate target')
        require(peer is None or isinstance(peer, str) and peer in catalogue and peer != target, 'unknown/self peer')
        require((peer is not None) == needs_peer, 'peer binding differs from expression inputs')
        seen.add(target)
    minimum = 3 if plan['forecast_target'] == 'resolves_yes' else 8
    require(type(plan['min_observations']) is int and minimum <= plan['min_observations'] <= 100,
            'minimum observations outside target-specific bounds')
    gain = plan['min_improvement']
    require(type(gain) in (int, float) and math.isfinite(gain) and 0 < gain <= 1, 'invalid minimum improvement')
    return deepcopy(value)


def fresh_price(feed, mid, at, max_age_seconds):
    quote = feed.latest(mid, at)
    if quote is None: return None, 'missing_quote'
    price, source = quote
    require(source <= at and 0 < float(price) < 1, 'future/invalid price returned by tool')
    if (at-source).total_seconds() > max_age_seconds: return None, 'stale_quote'
    return {'market_id': mid, 'sample_time': iso(at), 'source_time': iso(source), 'price': float(price)}, None


def forecast_at(plan, binding, catalogue, feed, at, max_age_seconds):
    """Strictly causal interface: never accepts outcomes or future target prices."""
    root, refs = compile_expression(plan['expression']); values = {}; evidence = []
    for role, lag in sorted(set(refs) | {('target', 0)}):
        mid = binding[role]; point = at-timedelta(days=lag)
        if point < timestamp(catalogue[mid]['initialized_at'], 'initialization'):
            return None, evidence, 'before_initialization'
        row, error = fresh_price(feed, mid, point, max_age_seconds)
        if error: return None, evidence, role+':'+error
        evidence.append({**row, 'role': role, 'lag_days': lag}); values[(role, lag)] = row['price']
    try: prediction = calculate(root, values)
    except ValidationError as exc: return None, evidence, str(exc)
    return {'prediction': prediction, 'baseline': values[('target', 0)]}, evidence, None


def validate_protocol(protocol):
    fields(protocol, {'as_of', 'feedback_cutoff', 'max_days_per_target', 'max_quote_age_seconds',
                      'min_event_groups', 'min_coverage', 'partition', 'independent_validation'}, 'method protocol')
    require(protocol['partition'] == 'train' and protocol['independent_validation'] is False,
            'this executor only admits reused training diagnostics')
    require(timestamp(protocol['as_of'], 'as of') <= timestamp(protocol['feedback_cutoff'], 'feedback'),
            'feedback cutoff precedes features')
    require(type(protocol['max_days_per_target']) is int and 1 <= protocol['max_days_per_target'] <= 120,
            'unbounded observation grid')
    require(type(protocol['max_quote_age_seconds']) is int and 1 <= protocol['max_quote_age_seconds'] <= 10800,
            'invalid price freshness')
    require(type(protocol['min_event_groups']) is int and protocol['min_event_groups'] >= 3
            and type(protocol['min_coverage']) in (int, float) and .8 <= protocol['min_coverage'] <= 1,
            'insufficient screen requirements')


def execute_plan(proposal, catalogue, feed, labels, protocol):
    """Freeze a plan before calling this evaluator; labels stay outside forecast_at."""
    validate_protocol(protocol); validate_plan(proposal, catalogue)
    plan = proposal['plan']; rows = []
    cutoff = timestamp(protocol['as_of'], 'as of'); feedback = timestamp(protocol['feedback_cutoff'], 'feedback')
    if plan is not None:
        for binding in plan['bindings']:
            mid = binding['target']; label = labels[mid]
            require(type(label['outcome']) is int and label['outcome'] in (0, 1), 'invalid event outcome')
            resolution = timestamp(label['resolution_time'], 'resolution')
            available = timestamp(label['available_at'], 'label availability')
            require(resolution <= available, 'label precedes resolution')
            init = timestamp(catalogue[mid]['initialized_at'], 'initialization')
            # Fixed daily UTC grid from initialization, not selected by observed return.
            start = init.replace(hour=0, minute=0, second=0, microsecond=0)+timedelta(days=1)
            horizon = plan['horizon_days']; step = horizon or 1
            for offset in range(0, protocol['max_days_per_target'], step):
                at = start+timedelta(days=offset)
                if at >= resolution or at > cutoff: break
                end = at+timedelta(days=horizon) if horizon else None
                if end is not None and (end >= resolution or end > cutoff): break
                prediction, evidence, error = forecast_at(plan, binding, catalogue, feed, at,
                                                          protocol['max_quote_age_seconds'])
                row = {'binding': deepcopy(binding), 'event_group_id': catalogue[mid]['event_group_id'],
                       'observation_time': iso(at), 'forecast': prediction, 'feature_evidence': evidence,
                       'error': error, 'target_evidence': None, 'model_loss': None, 'baseline_loss': None}
                if end is not None:
                    target, target_error = fresh_price(feed, mid, end, protocol['max_quote_age_seconds'])
                    if target_error: row['error'] = row['error'] or 'target:'+target_error
                    elif timestamp(target['source_time'], 'target') <= at:
                        row['error'] = row['error'] or 'target:not_after_observation'
                    else: row['target_evidence'] = target
                    y = None if target is None else target['price']
                else:
                    y = label['outcome']
                    if available > feedback: row['error'] = row['error'] or 'target:feedback_not_available'
                    else: row['target_evidence'] = {'outcome': y, 'resolution_time': iso(resolution), 'available_at': iso(available)}
                if row['error'] is None:
                    row['model_loss'] = (prediction['prediction']-y)**2
                    row['baseline_loss'] = (prediction['baseline']-y)**2
                row['observation_sha256'] = canonical_hash(row); rows.append(row)
                # One predeclared date per binary resolution, even if inputs fail.
                if horizon is None: break
    valid = [r for r in rows if r['error'] is None]; groups = defaultdict(list)
    for row in valid: groups[row['event_group_id']].append(row)
    group_scores = {g: {'observations': len(rs), 'model_loss': statistics.fmean(r['model_loss'] for r in rs),
                         'baseline_loss': statistics.fmean(r['baseline_loss'] for r in rs)} for g, rs in sorted(groups.items())}
    improvement = statistics.fmean(s['baseline_loss']-s['model_loss'] for s in group_scores.values()) if groups else None
    coverage = len(valid)/len(rows) if rows else 0
    sufficient = bool(plan and len(valid) >= plan['min_observations'] and len(groups) >= protocol['min_event_groups']
                      and coverage >= protocol['min_coverage'])
    status = ('no_plan' if plan is None else 'insufficient_data' if not sufficient
              else 'training_screen_passed' if improvement >= plan['min_improvement'] else 'training_screen_failed')
    result = {'kind': 'executable_method_result_v1', 'proposal_sha256': canonical_hash(proposal),
              'protocol_sha256': canonical_hash(protocol), 'metric': None if plan is None else
                  'brier' if plan['forecast_target'] == 'resolves_yes' else 'future_yes_price_mse',
              'status': status, 'observations_attempted': len(rows), 'valid_observations': len(valid),
              'coverage': coverage, 'target_event_groups': len(groups), 'group_scores': group_scores,
              'mean_group_improvement': improvement, 'failures': dict(Counter(r['error'] for r in rows if r['error'])),
              'eligible_for_independent_validation': status == 'training_screen_passed',
              'independent_validation': False, 'effectiveness_verified': False, 'fine_tuning_admitted': False,
              'net_profit_evaluated': False, 'observations': rows}
    result['result_sha256'] = canonical_hash(result)
    return result


def validate_review(value, result):
    fields(value, {'result_sha256', 'status', 'eligible_for_independent_validation', 'limitation'}, 'executable method review')
    for key in ('result_sha256', 'status', 'eligible_for_independent_validation'):
        require(type(value[key]) is type(result[key]) and value[key] == result[key], 'review contradicts tool verdict: '+key)
    bounded_text(value['limitation']); return deepcopy(value)


def run_workflow(runner, context, catalogue, feed, labels, protocol, freeze):
    """Researcher -> immutable registration -> deterministic tool -> critic."""
    from langgraph.graph import StateGraph, START, END
    def propose(state):
        try:
            proposal = runner.structured('experiment_planner', PLANNER, context, {}, lambda v: validate_plan(v, catalogue))
            freeze(proposal)  # Caller persists this before the evaluator reads targets.
            return {**state, 'proposal': proposal, 'proposal_error': None}
        except (ValueError, TypeError, KeyError) as exc:
            return {**state, 'proposal': None, 'proposal_error': str(exc)}
    def execute(state):
        result = None if state['proposal'] is None else execute_plan(state['proposal'], catalogue, feed, labels, protocol)
        return {**state, 'result': result}
    def review(state):
        review = None; error = None
        if state['result'] is not None:
            summary = {k: v for k, v in state['result'].items() if k != 'observations'}
            try:
                review = runner.structured('experiment_auditor', REVIEWER,
                    {'phase': 'after_training_tool_feedback'}, {'result': summary}, lambda v: validate_review(v, state['result']))
            except (ValueError, TypeError, KeyError) as exc: error = str(exc)
        return {**state, 'review': review, 'review_error': error}
    graph = StateGraph(dict)
    for name, fn in [('propose', propose), ('execute', execute), ('review', review)]: graph.add_node(name, fn)
    for a, b in [(START, 'propose'), ('propose', 'execute'), ('execute', 'review'), ('review', END)]: graph.add_edge(a, b)
    return graph.compile().invoke({}, {'max_concurrency': 1, 'recursion_limit': 8})
