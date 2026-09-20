"""Label-free role and resource census from the frozen specialist raw responses."""
import argparse
from collections import Counter
from pathlib import Path

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.market_development import rows
from foretellmesh.schema import ValidationError


def summarize(root):
    report=strict_json((root/'generation_report.json').read_text())
    if report['status']!='completed' or report['results_sha256']!=sha256_file(root/'results.jsonl'):
        raise ValidationError('completed, hash-matched raw responses required')
    rr=rows(root/'results.jsonl');calls=[c for r in rr for c in r['calls']];roles={}
    for role in sorted({c['request']['agent'] for c in calls}):
        cc=[c for c in calls if c['request']['agent']==role]
        seconds=sum((c['usage'] or {}).get('seconds',0) for c in cc)
        tokens=sum((c['usage'] or {}).get('output_tokens',0) for c in cc)
        roles[role]={'calls':len(cc),'repairs':sum('repair' in c['request'] for c in cc),
            'output_tokens':tokens,'generation_seconds':seconds,
            'output_tokens_per_generation_second':tokens/seconds if seconds else None}
    changes=[]
    for r in rr:
        pred=r['result']['prediction'];q=r['calls'][0]['request']['input']['market']['probability']
        if pred is not None and abs(pred['probability']-q)>1e-6:
            changes.append({'sample_id':r['sample_id'],'arm':r['arm'],
                'market_probability':q,'forecast_probability':pred['probability']})
    result={'results_sha256':sha256_file(root/'results.jsonl'),
        'generation_report_sha256':sha256_file(root/'generation_report.json'),'script_sha256':sha256_file(Path(__file__)),
        'role_resources':roles,'probability_changes_over_1e_6':changes,
        'game_mechanisms':dict(Counter(r['result']['stages']['game_theory']['mechanism'] for r in rr if 'game_theory' in r['result']['stages'])),
        'failed_workflows':[{'sample_id':r['sample_id'],'arm':r['arm'],'stage':r['result']['stage'],
            'errors':[t.get('validation_error',t.get('error_type')) for t in r['result']['trace'] if t['status']!='valid']}
            for r in rr if r['result']['status']!='completed'],
        'all_model_requests_adapter_none':all(c['request']['adapter'] is None for c in calls),
        'total_model_calls':len(calls),
        'output_limit_calls':sum((c['usage'] or {}).get('output_reached_token_limit',False) for c in calls)}
    (root/'role_diagnostics.json').write_text(json_text(result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    print(json_text(summarize(p.parse_args().run)))
