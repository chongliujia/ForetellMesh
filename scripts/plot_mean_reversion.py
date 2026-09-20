"""Audit and plot a frozen historical mean-reversion development simulation."""
import argparse
from datetime import datetime
from pathlib import Path

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.market_development import rows
from foretellmesh.reversion_experiment import audit


def plot(root):
    checked=audit(root)
    (root/'audit.json').write_text(json_text(checked))
    r=strict_json((root/'report.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    arms=['mean_reversion_only','weekly_liquidity_control','mean_reversion_weekly']
    labels=['Mean reversion only','Weekly participation only','Mean reversion + weekly']
    scenarios=['frictionless_reference','cost_assumption','cost_stress']
    scenario_labels=['No costs','Main costs','Stress costs']
    colors=['#247b86','#86929f','#d17b37']
    fig,axes=plt.subplots(1,3,figsize=(17,5.8),layout='constrained')
    start=datetime.fromisoformat(r['start']);end=datetime.fromisoformat(r['end'])
    for arm,label,color in zip(arms,labels,colors):
        points=[x for x in rows(root/'cost_assumption'/f'{arm}.equity_curve.jsonl') if start<=datetime.fromisoformat(x['time'])<=end]
        axes[0].plot([datetime.fromisoformat(x['time']) for x in points],
            [float(x['equity_proxy']) for x in points],label=label,color=color,lw=1.5)
    axes[0].axhline(100,color='#777',ls='--',lw=.9)
    axes[0].set(title='Main cost assumption: account path',ylabel='Liquidation-value proxy (USD)')
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[0].legend(loc='lower left',fontsize=8,frameon=False)
    for i,(scenario,label) in enumerate(zip(scenarios,scenario_labels)):
        values=[float(r['scenarios'][scenario][a]['final_cash'])-100 for a in arms]
        axes[1].barh([j+(i-1)*.25 for j in range(3)],values,height=.23,label=label,
            color=['#98bdc0','#247b86','#d17b37'][i])
        for j,v in enumerate(values):
            axes[1].text(v+(.07 if v>=0 else -.07),j+(i-1)*.25,f'{v:+.2f}',
                ha='left' if v>=0 else 'right',va='center',fontsize=8)
    axes[1].set_yticks(range(3),['Reversion','Weekly only','Combined'])
    axes[1].invert_yaxis();axes[1].axvline(0,color='#777',lw=.9)
    axes[1].set(title='Net PnL from $100 initial cash',xlabel='USD; terminal after any settlements')
    axes[1].margins(x=.2);axes[1].legend(loc='upper left',fontsize=8,frameon=False)
    for i,(a,label,color) in enumerate(zip(arms,labels,colors)):
        v=r['scenarios']['cost_assumption'][a];gap=v['cadence']['actual_max_gap_hours']/24
        axes[2].barh(i,gap,color=color,height=.5)
        axes[2].text(gap+.2,i,f"{gap:.2f} d / {v['entry_count']+v['exit_count']} fills",va='center',fontsize=9)
    axes[2].axvline(7,color='#ad483d',ls='--',lw=1,label='7-day maximum')
    axes[2].set_yticks(range(3),['Reversion','Weekly only','Combined']);axes[2].invert_yaxis()
    axes[2].set(title='Longest gap between fills: main costs',xlabel='Days, including calendar start/end')
    axes[2].set_xlim(0,max(r['scenarios']['cost_assumption'][a]['cadence']['actual_max_gap_hours']/24 for a in arms)*1.4)
    axes[2].legend(loc='lower right',fontsize=8,frameon=False)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f"Mean-reversion simulation | {r['market_count']} Polymarket contracts / {r['event_groups']} macro events",fontsize=15)
    fig.supxlabel('Feb–Jul 2025 development replay. Hypothetical full fills and costs; no order-book depth. No LLM calls or training.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(root/f'mean_reversion.{ext}',dpi=180)
    plt.close(fig)
    (root/'figure_manifest.json').write_text(json_text({'report_sha256':checked['report_sha256'],
        'script_sha256':sha256_file(Path(__file__)),
        'artifacts':{name:sha256_file(root/name) for name in ('mean_reversion.png','mean_reversion.pdf')}}))
    return checked


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    print(json_text(plot(p.parse_args().run)))
