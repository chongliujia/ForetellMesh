"""Export audited diagnostic summaries and a standalone scientific figure."""
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics

from foretellmesh.agent_runtime import decode_agent_response
from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.forecast_diagnostic import audit, prediction
from foretellmesh.market_development import rows


def summarize(run: Path):
    checked = audit(run)
    report = strict_json((run/'report.json').read_text())
    config = strict_json((run/'config.json').read_text())
    results = rows(run/'results.jsonl')
    inputs = {r['sample_id']:r['input'] for part in ('train','validation')
              for r in rows(run/f'data/partitions/{part}.inputs.jsonl')}
    grouped = defaultdict(list)
    for r in results:grouped[r['phase']+':'+r['price']].append(r)
    resources = {}
    for name, rr in grouped.items():
        usage = [c['usage'] for r in rr for c in r['calls'] if c['usage'] is not None]
        seconds = math.fsum(u['seconds'] for u in usage)
        probabilities = [prediction(r) for r in rr]
        valid = [r for r in rr if prediction(r) is not None]
        resources[name] = {
            'calls':len(rr), 'valid':len(valid),
            'mean_workflow_seconds':statistics.fmean(r['seconds'] for r in rr),
            'generation_seconds':seconds,
            'input_tokens':sum(u['input_tokens'] for u in usage),
            'output_tokens':sum(u['output_tokens'] for u in usage),
            'output_tokens_per_second':sum(u['output_tokens'] for u in usage)/seconds if seconds else None,
            'output_token_limit_calls':sum(u['output_reached_token_limit'] for u in usage),
            'peak_allocated_GiB':max(r['memory']['peak_allocated_bytes'] for r in rr)/2**30,
            'peak_reserved_GiB':max(r['memory']['peak_reserved_bytes'] for r in rr)/2**30,
            'valid_within_1e_6_of_original_market':sum(abs(prediction(r)-inputs[r['sample_id']]['market']['probability'])<=config['probability_tolerance'] for r in valid),
            'probability_counts':dict(Counter(str(p) for p in probabilities)),
            'errors':dict(Counter(t.get('validation_error',t.get('error_type','')) for r in rr for t in r['result']['trace'] if t['status']!='valid')),
        }
    summary = {'report_sha256':checked['report_sha256'], 'phase_resources':resources,
               'generation_seconds':math.fsum(r['seconds'] for r in results),
               'scope':{'common_validation_observations':len(report['metrics']['price_ablation']['common_sample_ids']),
                        'validation_denominator':len([r for r in results if r['phase']=='price_ablation']),
                        'train_selected_observations':report['metrics']['training_candidates']['visible']['sample_groups'],
                        'train_event_groups':report['metrics']['training_candidates']['visible']['event_groups']},
               'training_performed':False, 'final_test_scored':False}
    failures = []
    for r in results:
        if r['result']['status'] == 'completed':continue
        call = r['calls'][0]
        item = {k:r[k] for k in ('sample_id','phase','price','candidate')}
        item['validation_errors'] = [t.get('validation_error',t.get('error_type','')) for t in r['result']['trace'] if t['status']!='valid']
        try:
            value,_ = decode_agent_response(call['output'],config['response_transport'])
            actual = value.get('event') if isinstance(value,dict) else None
            expected = call['request']['input']['question']
            item['expected_event_chars'] = len(expected)
            item['actual_event_chars'] = len(actual) if isinstance(actual,str) else None
            item['event_differs_only_by_q_prefix'] = isinstance(actual,str) and expected.startswith('q: ') and actual == expected[3:]
            item['actual_event_excerpt'] = actual[:160] if isinstance(actual,str) else None
        except (ValueError,TypeError):
            item['transport_valid'] = False
        failures.append(item)
    summary['failure_details'] = failures
    (run/'summary.json').write_text(json_text(summary))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,3,figsize=(16,5),layout='constrained')
    ablation = report['metrics']['price_ablation']
    names = ['market','visible_cached','hidden','constant_0_5']
    scores = [ablation['common'][name]['brier'] for name in names]
    axes[0].bar(['Market','Visible\n(cached)','Hidden','0.5'],scores,color=['#777777','#2676b5','#d16b31','#b5b5b5'])
    for i,s in enumerate(scores):
        if s is not None:axes[0].text(i,s+.008,f'{s:.4f}',ha='center')
    axes[0].set_ylim(0,max(s for s in scores if s is not None)*1.22)
    axes[0].set_ylabel('Brier score (lower is better)')
    axes[0].set_title(f"Paired validation: {len(ablation['common_sample_ids'])} observations")
    hidden = [r for r in results if r['phase']=='price_ablation' and prediction(r) is not None]
    axes[1].scatter([inputs[r['sample_id']]['market']['probability'] for r in hidden],
                    [prediction(r) for r in hidden],s=30,alpha=.65,color='#d16b31')
    axes[1].plot([0,1],[0,1],ls='--',color='#999999',lw=1)
    axes[1].set(xlim=(-.03,1.03),ylim=(-.03,1.03),xlabel='Historical market probability',
                ylabel='Model probability with market hidden',title='Dependence on the price input')
    candidate = report['metrics']['training_candidates']
    order = [g['sample_id'] for g in candidate['visible']['groups']]
    for arm,offset,color in [('visible',-.15,'#2676b5'),('hidden',.15,'#d16b31')]:
        ix = {g['sample_id']:g for g in candidate[arm]['groups']}
        yy = [ix[s]['probability_spread']['range'] for s in order]
        axes[2].scatter([i+offset for i in range(len(order))],yy,s=34,color=color,label=arm.capitalize())
    axes[2].axhline(config['meaningful_probability_range'],ls=':',color='#999999',lw=1)
    axes[2].set_xticks(range(len(order)),[s.split(':')[1] for s in order],rotation=70)
    axes[2].set(ylabel='Probability range among valid candidates',xlabel='Selected training contract ID',
                title='Four samples per input / price condition')
    axes[2].legend(frameon=False)
    fig.suptitle('Qwen3-8B Base: price ablation and training rollout diagnostics',fontsize=14)
    for ext in ('png','pdf'):fig.savefig(run/f'forecast_diagnostic.{ext}',dpi=180)
    plt.close(fig)
    artifacts=['summary.json','forecast_diagnostic.png','forecast_diagnostic.pdf']
    (run/'figure_manifest.json').write_text(json_text({'report_sha256':checked['report_sha256'],
        'script_sha256':sha256_file(Path(__file__)), 'artifacts':{n:sha256_file(run/n) for n in artifacts}}))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    print(json_text(summarize(p.parse_args().run)))
