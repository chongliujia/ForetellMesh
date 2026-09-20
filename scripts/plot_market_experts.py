"""Visualize a fully audited Base specialist pilot; no profitability promotion."""
import argparse
from pathlib import Path

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.market_experiment import audit


def plot(root):
    checked=audit(root);dest=root/'evaluation';(dest/'audit.json').write_text(json_text(checked))
    r=strict_json((dest/'report.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms=['original_base','features_only','quant','game','quant_game','market_implied','cash']
    labels=['Original Base (cached)','Features only','+ Quant','+ Game theory','+ Quant + Game','Market baseline','Cash']
    scenarios=['frictionless_reference','cost_assumption','cost_stress'];colors=['#699d9b','#2b6777','#bd7951']
    fig,ax=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for i,s in enumerate(scenarios):
        ax[0].barh([j+(i-1)*.23 for j in range(len(arms))],
            [float(r['scenarios'][s][a]['final_cash']) for a in arms],height=.22,color=colors[i],label=s.replace('_',' '))
    ax[0].axvline(100,color='#444',ls='--',lw=1);ax[0].set_yticks(range(len(arms)),labels);ax[0].invert_yaxis()
    ax[0].set(title='Final account value (initial $100)',xlabel='USD; historical trade-print fill assumptions')
    ax[0].set_ylim(7.2,-.6)
    ax[0].legend(frameon=False,fontsize=8,loc='lower center',ncol=3)
    aa=arms[:-1];common=r['common_valid_scores']
    values=[common[a]['brier'] for a in aa] if common else [0]*len(aa)
    ax[1].barh(range(len(aa)),values,color=['#999999','#699d9b','#2b6777','#bd7951','#634b78','#999999'])
    ax[1].set_yticks(range(len(aa)),labels[:-1]);ax[1].invert_yaxis()
    ax[1].set(title=f"Brier on {r['common_valid_count']} commonly valid observations",xlabel='Brier score (lower is better)')
    for i,(a,v) in enumerate(zip(aa,values)):
        coverage=r['forecast_scores'][a]['prediction_count']
        ax[1].text(v+.004,i,f"{v:.4f} | valid {coverage}/{r['observations']}",va='center',fontsize=9)
    if values:ax[1].set_xlim(0,max(.1,max(values)*1.5))
    fig.suptitle(f"Qwen3-8B Base: market specialist pilot / {r['event_groups']} event groups",fontsize=14)
    fig.supxlabel('Development pilot. Hypothetical fills/costs; original cache has different input and repair settings. No training.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(dest/f'market_experts.{ext}',dpi=180)
    plt.close(fig)
    (dest/'figure_manifest.json').write_text(json_text({'report_sha256':checked['report_sha256'],
        'script_sha256':sha256_file(Path(__file__)),
        'artifacts':{name:sha256_file(dest/name) for name in ('market_experts.png','market_experts.pdf')}}))
    return checked


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    print(json_text(plot(p.parse_args().run)))
