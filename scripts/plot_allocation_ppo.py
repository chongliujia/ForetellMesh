"""Plot all prespecified PPO seeds and controls from an audited simulation."""
import argparse
from datetime import datetime
from pathlib import Path

from foretellmesh.data import sha256_file,strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.market_development import rows


def plot(root):
    audit=strict_json((root/'audit.json').read_text());report=strict_json((root/'report.json').read_text())
    if audit['status']!='passed' or audit['report_sha256']!=sha256_file(root/'report.json'):raise ValueError('matching audit required')
    for name,digest in report['artifact_hashes'].items():
        if sha256_file(root/name)!=digest:raise ValueError('audited artifact changed')
    c=strict_json((root/'config.json').read_text());seeds=c['seeds'];primary=c['primary_seed']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    colors=['#217b85','#c47b41','#75629a'];basecolors={'corrected_mean_reversion':'#303846','weekly_cost_only':'#9aa1aa'}
    names={'corrected_mean_reversion':'Corrected mean reversion','weekly_cost_only':'Weekly cost control'}
    arms=list(names)+[f'{stage}_{seed}' for seed in seeds for stage in ('initial','ppo')]
    names.update({f'{stage}_{seed}':f'{"PPO" if stage=="ppo" else "Untrained"} seed {seed}' for seed in seeds for stage in ('initial','ppo')})
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    end=datetime.fromisoformat(report['validation_data']['end'])
    curves=list(basecolors)+[f'ppo_{seed}' for seed in seeds]
    for arm in curves:
        rr=[r for r in rows(root/'evaluation/cost_assumption'/f'{arm}.equity_curve.jsonl') if datetime.fromisoformat(r['time'])<=end]
        color=basecolors[arm] if arm in basecolors else colors[seeds.index(int(arm.split('_')[-1]))%len(colors)]
        axes[0,0].plot([datetime.fromisoformat(r['time']) for r in rr],[float(r['equity_proxy']) for r in rr],
            label=names[arm],color=color,lw=1.5 if arm==f'ppo_{primary}' else 1,alpha=.9)
    axes[0,0].axhline(100,color='#aaa',ls='--',lw=.8)
    axes[0,0].set(title='Main costs: $100 account path',ylabel='Liquidation-value proxy (USD)')
    axes[0,0].xaxis.set_major_locator(mdates.MonthLocator());axes[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[0,0].legend(fontsize=8,frameon=False,loc='best')
    main=report['scenarios']['cost_assumption'];palette=[]
    for arm in arms:
        palette.append(basecolors[arm] if arm in basecolors else colors[seeds.index(int(arm.split('_')[-1]))%len(colors)])
    values=[float(main[a]['net_pnl']) for a in arms]
    bars=axes[0,1].barh(range(len(arms)),values,color=palette)
    for bar,arm in zip(bars,arms):
        if arm.startswith('initial'):bar.set_alpha(.4)
    axes[0,1].set_yticks(range(len(arms)),[names[a] for a in arms]);axes[0,1].invert_yaxis();axes[0,1].axvline(0,color='#888',lw=.8)
    axes[0,1].set(title='Main costs: paired initial / trained policies',xlabel='Net PnL (USD); all seeds reported')
    axes[0,1].margins(x=.22)
    for i,v in enumerate(values):axes[0,1].text(v+(.12 if v>=0 else -.12),i,f'{v:+.2f}',ha='left' if v>=0 else 'right',va='center',fontsize=8)
    scenarios=['frictionless_reference','cost_assumption','cost_stress'];scenario_names=['No costs','Main costs','Stress costs']
    for i,arm in enumerate(curves):
        v=[float(report['scenarios'][s][arm]['net_pnl']) for s in scenarios]
        color=basecolors[arm] if arm in basecolors else colors[seeds.index(int(arm.split('_')[-1]))%len(colors)]
        axes[1,0].plot(range(3),v,marker='o',color=color,label=names[arm])
    axes[1,0].axhline(0,color='#888',ls='--',lw=.8);axes[1,0].set_xticks(range(3),scenario_names)
    axes[1,0].set(title='Cost sensitivity: same frozen policies',ylabel='Net PnL (USD)')
    axes[1,0].legend(fontsize=8,frameon=False,loc='best')
    gaps=[main[a]['cadence']['actual_max_gap_hours']/24 for a in arms]
    axes[1,1].barh(range(len(arms)),gaps,color=palette)
    axes[1,1].set_yticks(range(len(arms)),[names[a] for a in arms]);axes[1,1].invert_yaxis()
    axes[1,1].axvline(7,color='#b24c40',ls='--',lw=1,label='7-day requirement')
    axes[1,1].set(title='Main costs: longest actual-fill gap',xlabel='Days (calendar boundaries included)')
    axes[1,1].set_xlim(0,max(7,max(gaps))*1.3)
    for i,(gap,arm) in enumerate(zip(gaps,arms)):
        axes[1,1].text(gap+.12,i,f'{gap:.2f} / {"pass" if main[arm]["policy_requirements_met"] else "FAIL"}',va='center',fontsize=8)
    axes[1,1].legend(fontsize=8,frameon=False,loc='best')
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Allocation PPO prototype | 6 training groups -> 13 development validation groups',fontsize=15)
    fig.supxlabel('Historical simulation with hypothetical fills/costs. Terminal PnL includes residual settlement; no LLM updates or live orders.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(root/f'allocation_ppo.{ext}',dpi=180)
    plt.close(fig)
    (root/'figure_manifest.json').write_text(json_text({'report_sha256':audit['report_sha256'],'script_sha256':sha256_file(Path(__file__)),
        'artifacts':{name:sha256_file(root/name) for name in ('allocation_ppo.png','allocation_ppo.pdf')}}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);plot(p.parse_args().run)
