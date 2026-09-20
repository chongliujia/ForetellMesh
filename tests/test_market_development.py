from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foretellmesh.agent_baseline_data import input_context
from foretellmesh.capabilities import load_capabilities
from foretellmesh.data import sha256_file
from foretellmesh.evaluation import json_text
from foretellmesh.market_development import ARMS, DATA_FILES, load_config, make_jobs, read_inputs, replay, runner, score
from foretellmesh.schema import ValidationError
from foretellmesh.sft_data import jsonl
from foretellmesh.synthetic_sft import canonical_hash

ROOT = Path(__file__).resolve().parents[1]


class MarketDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = load_config(ROOT/'configs/pma_model_development_v1.json')
        self.agents, _ = load_capabilities(ROOT/'configs/capability_agents_v1.json')
        self.inputs = [{'sample_id': 'v'+str(i), 'input': {
            'question': 'Future event '+str(i)+'?', 'observation_time': '2025-02-01T00:00:00Z',
            'evidence': [{'evidence_id': 'e', 'text': 'Past fact', 'source': 'https://example.org/archive',
                          'published_at': '2025-01-01T00:00:00Z', 'available_at': '2025-01-01T00:00:00Z'}],
            'market': {'probability': .2, 'observed_at': '2025-02-01T00:00:00Z', 'available_at': '2025-02-01T00:00:00Z'}}} for i in range(3)]
        self.membership = [{'sample_id': r['sample_id'], 'event_id': r['sample_id'], 'event_group_id': 'g'+str(i//2),
                            'split': 'validation', 'dataset_version': 'fixture'} for i, r in enumerate(self.inputs)]
        self.membership += [{'sample_id': sid, 'event_id': sid, 'event_group_id': sid, 'split': split,
                             'dataset_version': 'fixture'} for sid, split in [('train', 'train'), ('test', 'test')]]
        label = lambda y: {'outcome': y, 'resolution_time': '2025-02-02T00:00:00Z', 'available_at': '2025-02-02T00:00:00Z'}
        self.write('membership.jsonl', self.membership)
        self.write('partitions/validation.inputs.jsonl', self.inputs)
        self.write('partitions/validation.labels.jsonl', [{'sample_id': r['sample_id'], 'label': label(i%2)} for i,r in enumerate(self.inputs)])
        self.write('partitions/train.labels.jsonl', [{'sample_id': 'train', 'label': label(0)}])
        self.freeze()

    def write(self, name, value):
        path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(jsonl(value))

    def freeze(self):
        (self.root/'report.json').write_text(json_text({'dataset_version': 'fixture',
            'artifact_hashes': {name: sha256_file(self.root/name) for name in DATA_FILES}}))
        self.config['dataset_report_sha256'] = sha256_file(self.root/'report.json')

    def fake_results(self, invalid=None):
        result = []
        for job in make_jobs(self.inputs):
            calls = []
            class Backend:
                available_adapters = {'research_tool_lora'}
                def generate(backend, request):
                    value = {'unknowns': [], 'observation_time': request['input']['observation_time']}
                    if request['agent'] == 'research':
                        value.update(evidence_ids=['e'], counter_evidence_ids=[])
                    else:
                        value.update(event=request['input']['question'], probability=.3, confidence='low',
                                     base_rate=None, key_evidence=['e'], counter_evidence=[])
                    output = 'invalid' if (job['arm'], job['sample_id']) == invalid else json_text(value)
                    calls.append({'request': deepcopy(request), 'request_sha256': canonical_hash(request), 'output': output,
                                  'usage': {'input_tokens': 200, 'output_tokens': 50, 'seconds': 1., 'output_reached_token_limit': False},
                                  'error_type': None, 'error': None})
                    return output
            payload = next(r['input'] for r in self.inputs if r['sample_id'] == job['sample_id'])
            r = runner(self.agents, Backend(), self.config).run(input_context(payload), **ARMS[job['arm']])
            result.append({**job, 'result': r, 'calls': calls, 'seconds': 1.,
                           'memory': {'peak_allocated_bytes': 1, 'peak_reserved_bytes': 2}})
        return result

    def test_only_validation_inputs_are_parsed_and_test_files_are_unnecessary(self):
        original = Path.read_text
        def guard(path, *args, **kwargs):
            self.assertNotIn('test.', path.name)
            self.assertNotIn('labels', path.name)
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', guard):
            _, inputs, _ = read_inputs(self.root, self.config)
        self.assertEqual(inputs, self.inputs)

    def test_reject_future_evidence_and_label_in_input(self):
        for mutation in ('future', 'label'):
            inputs = deepcopy(self.inputs)
            if mutation == 'future':inputs[0]['input']['evidence'][0]['available_at'] = '2025-03-01T00:00:00Z'
            else:inputs[0]['input']['label'] = 0
            self.write('partitions/validation.inputs.jsonl', inputs); self.freeze()
            with self.assertRaises(ValidationError):read_inputs(self.root, self.config)

    def test_hash_split_overlap_missing_and_duplicate_samples_rejected(self):
        (self.root/'partitions/validation.inputs.jsonl').write_text('changed')
        with self.assertRaises(ValidationError):read_inputs(self.root, self.config)
        for inputs in (self.inputs[:-1], self.inputs+[self.inputs[0]]):
            self.write('partitions/validation.inputs.jsonl', inputs); self.freeze()
            with self.assertRaises(ValidationError):read_inputs(self.root, self.config)
        self.write('partitions/validation.inputs.jsonl', self.inputs)
        m = deepcopy(self.membership); m[-1]['event_group_id'] = 'g0'
        self.write('membership.jsonl', m); self.freeze()
        with self.assertRaises(ValidationError):read_inputs(self.root, self.config)

    def test_raw_replay_and_capability_routing(self):
        rr = self.fake_results(invalid=('multi_base', 'v0'))
        replay(self.inputs, rr, self.agents, self.config)
        for row in rr:
            for call in row['calls']:
                req = call['request']
                self.assertNotIn('label', req['input'])
                expected = 'research_tool_lora' if row['arm'] == 'multi_research_lora' and req['agent'] == 'research' else None
                self.assertEqual(req['adapter'], expected)
        for field in ('output', 'request'):
            changed = deepcopy(rr)
            if field == 'output':changed[0]['calls'][0]['output'] = '{}'
            else:changed[0]['calls'][0]['request']['adapter'] = 'research_tool_lora'
            with self.assertRaises(ValidationError):replay(self.inputs, changed, self.agents, self.config)

    def test_failed_predictions_count_in_coverage_and_matched_market_uses_same_rows(self):
        rr = self.fake_results(invalid=('single_base', 'v0'))
        membership = {m['sample_id']: m for m in self.membership}
        s = score(self.root, self.inputs, membership, rr, self.config)
        a = s['arms']['single_base']
        self.assertEqual(a['scores']['coverage'], 2/3)
        self.assertAlmostEqual(a['scores']['brier'], (.7**2+.3**2)/2)
        self.assertAlmostEqual(a['market_matched']['market']['brier'], (.8**2+.2**2)/2)
        self.assertEqual(s['common_coverage']['sample_ids'], ['v1', 'v2'])
        # g0 has two observations and g1 has one: event averaging differs.
        full = s['arms']['multi_base']['scores']
        self.assertAlmostEqual(full['event_mean']['brier'], ((.3**2+.7**2)/2+.3**2)/2)
        for changed in (rr[:-1], rr+[rr[0]], [rr[1],rr[0]]+rr[2:]):
            with self.assertRaises(ValidationError):score(self.root, self.inputs, membership, changed, self.config)

    def test_out_of_range_probability_and_invalid_label_rejected(self):
        rr = self.fake_results(); rr[0]['result']['prediction']['probability'] = 1.2
        with self.assertRaises(ValidationError):score(self.root, self.inputs, {m['sample_id']: m for m in self.membership}, rr, self.config)
        with self.assertRaises(ValidationError):replay(self.inputs, rr, self.agents, self.config)

    def test_no_test_partition_or_sampling_override(self):
        for key, value in [('partition', 'test'), ('do_sample', True), ('training', True), ('max_repairs', 1)]:
            c = deepcopy(self.config); c[key] = value; p = self.root/'config.json'; p.write_text(json_text(c))
            with self.assertRaises(ValidationError):load_config(p)


if __name__ == '__main__':unittest.main()
