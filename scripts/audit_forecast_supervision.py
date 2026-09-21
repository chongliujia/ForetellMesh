"""Rebuild the candidate archive and independently check target arithmetic."""
import argparse
from collections import Counter
from decimal import Decimal
from pathlib import Path

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.forecast_supervision import build, require_trainable
from foretellmesh.schema import ValidationError


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'capture', 'pdf-review', 'bundle', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValidationError('audit output exists')
    report = build(a.config, a.capture, a.bundle, a.pdf_review)
    rows = [strict_json(line) for line in (a.bundle/'targets.jsonl').read_text().splitlines()]
    groups = Counter()
    for row in rows:
        if Decimal(str(row['teacher_probability'])) != Decimal(row['teacher_published_percent'])/100:
            raise ValidationError('probability conversion differs')
        key = row['source_release'].split('-')
        year, quarter = int(key[2]), int(key[1][1:])
        target_year, target_quarter = row['target_quarter'].split(':Q')
        if (int(target_year)-year)*4+int(target_quarter)-quarter != row['horizon_quarters']:
            raise ValidationError('horizon arithmetic differs')
        if (row['student_input'] is not None or row['outcome'] is not None
                or row['forecast_schema_completion'] is not None or row['ready_for_sft']
                or row['split'] != 'unassigned'):
            raise ValidationError('candidate has manufactured training/evaluation state')
        if row['provenance']['publication_precision'] != 'day':
            raise ValidationError('invented publication precision')
        groups[row['event_group_id']] += 1
    try:
        require_trainable(report)
    except ValidationError:
        candidate_rejected = True
    else:
        raise ValidationError('candidate accepted as a training bundle')
    result = {'status': 'passed', 'manifest_sha256': sha256_file(a.bundle/'manifest.json'),
              'targets_sha256': sha256_file(a.bundle/'targets.jsonl'),
              'parsed_releases': report['parsed_releases'], 'targets_checked': len(rows),
              'unique_target_quarters': len(groups), 'source_and_pdf_extraction_rebuilt': True,
              'candidate_rejected_as_training_bundle': candidate_rejected,
              'training_performed': False, 'holdouts_opened': False,
              'source_script_sha256': sha256_file(Path(__file__))}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json_text(result))
    print(json_text(result))
