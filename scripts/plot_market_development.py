"""Render development coverage and common-observation calibration after audit."""
import argparse
from importlib.metadata import version
import math
from pathlib import Path

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.market_development import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    verified = audit(args.run)
    report = strict_json((args.run/'report.json').read_text())
    metrics = report['metrics']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = {'single_base': 'Single Base', 'multi_base': 'Multi Base',
             'multi_research_lora': 'Multi + research LoRA', 'market': 'Market'}
    colors = {'single_base': '#2563eb', 'multi_base': '#7c3aed',
              'multi_research_lora': '#059669', 'market': '#475569'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    arms = ['single_base', 'multi_base', 'multi_research_lora']
    inputs = {r['sample_id']: r['input'] for r in
              (strict_json(s) for s in (args.run/'data/partitions/validation.inputs.jsonl').read_text().splitlines())}
    records = [strict_json(s) for s in (args.run/'results.jsonl').read_text().splitlines()]
    common_ids = set(metrics['common_coverage']['sample_ids'])
    diagnostics = {'market_agreement_tolerance': 1e-6, 'arms': {}}
    for arm in arms:
        valid = [r for r in records if r['arm'] == arm and r['result']['status'] == 'completed']
        differences = [abs(r['result']['prediction']['probability']-inputs[r['sample_id']]['market']['probability'])
                       for r in valid if inputs[r['sample_id']]['market'] is not None]
        matched = [r for r in valid if r['sample_id'] in common_ids]
        diagnostics['arms'][arm] = {
            'valid_with_market': len(differences), 'exact_market_probability': sum(d == 0 for d in differences),
            'within_tolerance': sum(d <= diagnostics['market_agreement_tolerance'] for d in differences),
            'max_absolute_probability_change': max(differences) if differences else None,
            'mean_absolute_probability_change': math.fsum(differences)/len(differences) if differences else None,
            'common_success_count': len(matched),
            'common_success_mean_workflow_seconds': math.fsum(r['seconds'] for r in matched)/len(matched) if matched else None}
    (args.run/'market_agreement.json').write_text(json_text(diagnostics))
    coverage = [metrics['arms'][a]['scores'] for a in arms]
    ax = axes[0]
    bars = ax.bar(range(len(arms)), [s['coverage'] for s in coverage], color=[colors[a] for a in arms])
    ax.set_xticks(range(len(arms)), [names[a].replace(' ', '\n', 1) for a in arms], fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel('Fraction of all observations')
    ax.set_title('Valid forecast coverage')
    for bar, s in zip(bars, coverage):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.025,
                f"{s['prediction_count']}/{s['eligible_count']}", ha='center', fontsize=10)
    common = metrics['common_coverage']['scores']
    count = len(metrics['common_coverage']['sample_ids'])
    comparisons = arms+['market']
    ax = axes[1]
    ax.set_title(f'Brier score: same {count} observations')
    ax.set_ylabel('Lower is better')
    if count:
        bars = ax.bar(range(len(comparisons)), [common[a]['brier'] for a in comparisons],
                      color=[colors[a] for a in comparisons])
        ax.set_xticks(range(len(comparisons)), [names[a].replace(' ', '\n', 1) for a in comparisons], fontsize=9)
        for bar, a in zip(bars, comparisons):
            ax.annotate(f"{common[a]['brier']:.4f}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
        ax.margins(y=.2)
    else:
        ax.text(.5, .5, 'No common valid forecasts', transform=ax.transAxes, ha='center')
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color='#cbd5e1', linestyle='--', label='Ideal')
    if count:
        for a in comparisons:
            bins = [b for b in common[a]['calibration_curve'] if b['count']]
            ax.plot([b['mean_probability'] for b in bins], [b['event_frequency'] for b in bins],
                    marker='o', markersize=4, linewidth=1.3, alpha=.8, color=colors[a], label=names[a])
    ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), xlabel='Mean predicted probability',
           ylabel='Observed event frequency', title='Calibration on common observations')
    ax.legend(loc='best', fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f"Historical development diagnostic: {metrics['observation_count']} observations / "
                 f"{metrics['event_group_count']} macro releases\n"
                 'Correlated contracts; market supplied as input; final test remains sealed', fontsize=12)
    outputs = [args.run/'market_agreement.json']
    for extension in ('png', 'pdf'):
        path = args.run/f'development_scores.{extension}'
        fig.savefig(path, dpi=180); outputs.append(path)
    plt.close(fig)
    (args.run/'figure_manifest.json').write_text(json_text({
        'audit': verified, 'report_sha256': sha256_file(args.run/'report.json'),
        'script_sha256': sha256_file(Path(__file__)), 'matplotlib': version('matplotlib'),
        'artifacts': {p.name: sha256_file(p) for p in outputs}}))


if __name__ == '__main__':main()
