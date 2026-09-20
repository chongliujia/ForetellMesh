"""Plot an audited historical simulation; never imply executable fills."""
import argparse
from datetime import datetime
from pathlib import Path

from foretellmesh.data import sha256_file,strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.market_development import rows
from foretellmesh.paper_backtest import audit


def plot(root):
    checked=audit(root);(root/'audit.json').write_text(json_text(checked))
    report=strict_json((root/'report.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    scenarios=list(report['scenarios']);arms=['base_multi_agent','market_implied','training_event_prior','cash']
    labels=['Base agents','Market signal','Train prior','Cash']
    colors=['#767676','#2676b5','#c45d2d']
    fig,ax=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    width=.23
    for j,scenario in enumerate(scenarios):
        values=[float(report['scenarios'][scenario][a]['final_cash']) for a in arms]
        xx=[i+(j-1)*width for i in range(len(arms))]
        ax[0].bar(xx,values,width=width,label=scenario.replace('_',' '),color=colors[j%len(colors)])
        for x,v in zip(xx,values):ax[0].text(x,v+.8,f'{v:.2f}',ha='center',rotation=90,fontsize=8)
    ax[0].axhline(100,color='#333333',lw=1,ls='--')
    ax[0].set_xticks(range(len(arms)),labels)
    top=max(float(v['final_cash']) for arms_data in report['scenarios'].values() for v in arms_data.values())
    ax[0].set(ylim=(0,max(115,top*1.15)),ylabel='Final account balance (USD)',title='Each account starts with $100')
    ax[0].legend(frameon=False,fontsize=8)
    for j,scenario in enumerate(scenarios):
        rr=rows(root/scenario/'training_event_prior.equity_curve.jsonl')
        ax[1].plot([datetime.fromisoformat(r['time']) for r in rr],[float(r['equity_proxy']) for r in rr],
                   label='Train prior / '+scenario.replace('_',' '),color=colors[j%len(colors)],alpha=.85)
    rr=rows(root/'cost_assumption/base_multi_agent.equity_curve.jsonl')
    ax[1].plot([datetime.fromisoformat(r['time']) for r in rr],[float(r['equity_proxy']) for r in rr],
               color='#186e43',lw=2,label='Base agents / cost assumption')
    ax[1].axhline(100,color='#333333',lw=1,ls='--',label='Initial capital')
    locator=mdates.AutoDateLocator();ax[1].xaxis.set_major_locator(locator);ax[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax[1].set(ylabel='Event-sampled equity proxy (USD)',title='Assumed liquidation values; stale marks retained')
    ax[1].legend(frameon=False,fontsize=8)
    fig.suptitle(f"Qwen3-8B Base paper replay: {report['observation_count']} observations / {report['event_groups']} release groups",fontsize=14)
    fig.supxlabel('Historical trade-print simulation. Full fills and costs are assumptions; development results only.',fontsize=10)
    for ext in ('png','pdf'):fig.savefig(root/f'paper_trading.{ext}',dpi=180)
    plt.close(fig)
    (root/'figure_manifest.json').write_text(json_text({'report_sha256':checked['report_sha256'],
        'script_sha256':sha256_file(Path(__file__)),
        'artifacts':{name:sha256_file(root/name) for name in ('paper_trading.png','paper_trading.pdf')}}))
    return checked


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    print(json_text(plot(p.parse_args().run)))
