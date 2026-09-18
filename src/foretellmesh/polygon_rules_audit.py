"""Replay the specifically reviewed adapter's rule-history attestations.

Source hashes freeze a documented review, not an automatic Solidity analyzer.
Source-to-bytecode correspondence relies on Polygonscan's verification; no local
recompilation or independent chain consensus verification is claimed.
"""
import re

from .data import sha256_bytes, sha256_file, strict_json
from .polygon_proof_audit import audit as audit_initial_questions
from .polygon_settlement_audit import ProofArchive, UMA
from .schema import ValidationError, timestamp


def extract_source(page):
    # Parse only this JSON-valued JS string; never evaluate page JavaScript.
    matches = re.findall(r"var editor_contractJsonData = '([^\n]+)'", page)
    if len(matches) != 1:raise ValidationError('ambiguous verified source payload')
    encoded = re.sub(r"(?<!\\)\\([`'])", r'\1', matches[0])
    try:sources = strict_json(strict_json('"'+encoded+'"'))['sources']
    except (ValueError, KeyError, TypeError) as exc:raise ValidationError('invalid verified source JSON') from exc
    section = page.split('Deployed Bytecode', 1)
    match = re.search(r'<div>(0x[0-9a-fA-F]+)</div>', section[1]) if len(section) == 2 else None
    if match is None:raise ValidationError('missing explorer deployed bytecode')
    return sources, bytes.fromhex(match[1][2:])


def audit_rules(staging, candidates, initial_root, code_root, explorer_root, policy_path):
    policy = strict_json(policy_path.read_text())
    if (policy.get('schema_version') != '1' or policy.get('adapter') != UMA
            or policy.get('review_id') != 'polymarket_uma_negrisk_v4_rules_20260919_v1'
            or policy.get('trust_model') != 'polygonscan_source_verification_plus_single_public_rpc'
            or policy.get('reviewed_properties') != ['immutable_initialized_ancillary', 'append_only_creator_updates', 'no_proxy_delegatecall_or_selfdestruct']):
        raise ValidationError('unsupported rule-history source review')
    manifest = strict_json((explorer_root/'manifest.json').read_text())
    refs = [r for r in manifest['requests'] if r['url'] == policy['source_url']]
    if len(refs) != 1 or refs[0]['status'] != 200 or refs[0]['request'] is not None:
        raise ValidationError('missing unique verified-source page')
    ref = refs[0]; path = (explorer_root/ref['file']).resolve()
    if not path.is_relative_to(explorer_root.resolve()):raise ValidationError('unsafe verified source artifact')
    raw = path.read_bytes()
    if sha256_bytes(raw) != ref['sha256'] or ref['sha256'] != policy['source_page_sha256']:
        raise ValidationError('reviewed source page changed')
    if timestamp(ref['started_at'], 'source start') > timestamp(ref['completed_at'], 'source end'):
        raise ValidationError('invalid source capture clock')
    sources, runtime = extract_source(raw.decode())
    if ({name: sha256_bytes(value['content'].encode()) for name, value in sources.items()} != policy['source_sha256']
            or sha256_bytes(runtime) != policy['runtime_sha256']):
        raise ValidationError('source review or deployed bytecode changed')
    initial = audit_initial_questions(initial_root, staging)
    selected = [c for c in candidates if c['question_proof']]
    minimum = min(c['question_proof']['block_number'] for c in selected)
    pins = {r['finalized_block'] for r in initial['rows']}
    if len(pins) != 1:raise ValidationError('rule states use different finalized blocks')
    pinned = next(iter(pins)); code = ProofArchive(code_root, require_chain=False)
    # Chain identity is already attested by the initial-state archive, using the
    # same endpoint. Bytecode calls use its explicitly pinned numeric block.
    state_refs = ProofArchive(initial_root)
    pin_ref = next(iter(state_refs.index().values()))['pinned_block']
    pin = state_refs.get(pin_ref, 'eth_getBlockByNumber', ['finalized', False])
    pin_time = int(pin['timestamp'], 16)
    endpoints = {r['url'] for r, _ in state_refs.refs.values() if r['request']['method'] == 'eth_chainId'}
    code_blocks = set()
    for code_ref, _ in code.refs.values():
        if code_ref['request']['method'] != 'eth_getCode':continue
        params = code_ref['request']['params']
        if len(params) != 2 or params[0] != UMA or not re.fullmatch(r'0x[0-9a-f]+', params[1]):
            raise ValidationError('bytecode address or block not pinned')
        if code_ref['url'] not in endpoints:raise ValidationError('bytecode chain identity unbound')
        value = code.get(code_ref, 'eth_getCode', params)
        if value != '0x'+runtime.hex():raise ValidationError('RPC bytecode differs from reviewed explorer code')
        if int(params[1], 16) == pinned and pin_time > timestamp(code_ref['completed_at'], 'code capture').timestamp():
            raise ValidationError('bytecode block after capture')
        code_blocks.add(int(params[1], 16))
    if pinned not in code_blocks or minimum not in code_blocks:
        raise ValidationError('missing initialization/finalized bytecode coverage')
    by_qid = {r['question_id']: r for r in initial['rows']}
    for c in selected:
        proof = c['question_proof']; state = by_qid[proof['question_id']]
        required_clause = f'Updates made by the question creator via the bulletin board at {UMA}'
        if (proof['adapter'] != UMA or required_clause.lower() not in proof['ancillary_data'].lower()
                or not proof['ancillary_data'].endswith(',initializer:'+state['creator'][2:])
                or proof['block_number'] < minimum
                or not timestamp(proof['published_at'], 'initialization') <= timestamp(c['record']['observation_time'], 'observation')
                or timestamp(c['record']['observation_time'], 'observation').timestamp() > pin_time):
            raise ValidationError('question does not match reviewed creator-update semantics')
    return {'kind': 'polygon_rule_history_audit', 'status': 'passed_reviewed_source_attestations',
            'contracts': len(by_qid), 'observation_rows': len(selected),
            'question_ids': sorted(by_qid), 'creator_updates': 0, 'finalized_block': pinned,
            'runtime_bytes': len(runtime), 'runtime_sha256': policy['runtime_sha256'],
            'source_files': len(sources), 'policy_sha256': sha256_file(policy_path),
            'initial_proof': initial, 'code_manifest_sha256': sha256_file(code_root/'manifest.json'),
            'source_manifest_sha256': sha256_file(explorer_root/'manifest.json'),
            'trust_scope': 'Frozen manual source review + Polygonscan source verification + public RPC; no local recompilation or consensus proof.',
            'historical_inference': 'For the reviewed immutable append-only bulletin board, an empty creator array at the pinned block implies no earlier creator updates.'}
