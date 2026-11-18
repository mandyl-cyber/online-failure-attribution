"""Mechanically audit the returned Qwen32 paired claim run without rewriting it."""
import hashlib
import json
from pathlib import Path
import subprocess

from segmenter import stage_claim_summary as core
from scripts.download_qwen32_awq import WEIGHT_SHA256, REVISION


RETURN = Path('adjacent_claim_pair_v1')
PACK = Path('pilot/adjacent_claim_pair_v1')
OUT = Path('pilot/adjacent_claim_pair_return_v1_audit')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return core.strict_json(path.read_text(encoding='utf-8'))


def tracked_digest(path):
    data = subprocess.check_output(['git','show','HEAD:'+path.as_posix()])
    return hashlib.sha256(data).hexdigest()


def main():
    for name in ('requests.jsonl','groups.json','protocol.json','reference_points.json'):
        if digest(RETURN/name) != digest(PACK/name):
            raise ValueError('Returned input drift: '+name)
    config, protocol = read(RETURN/'config.json'), read(PACK/'protocol.json')
    if config['protocol_sha256'] != digest(PACK/'protocol.json'):
        raise ValueError('Protocol hash mismatch')
    if config['runner_sha256'] != tracked_digest(Path('remote_pilots/openhands_smoke/adjacent_claim_pair.py')):
        raise ValueError('Runner differs from committed version')
    if config['core_sha256'] != tracked_digest(Path('segmenter/stage_claim_summary.py')):
        raise ValueError('Claim core differs from committed version')
    if config['pinned_model_revision'] != REVISION or read(RETURN/'weight_sha256.json') != WEIGHT_SHA256:
        raise ValueError('Model provenance mismatch')
    rows = {r['job_id']:r for r in
            [core.strict_json(line) for line in (PACK/'requests.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]}
    groups, preflight = read(PACK/'groups.json'), read(RETURN/'preflight.json')
    assert preflight['eligible_groups']+preflight['excluded_groups'] == len(groups)
    seen, status = set(), {'ready':0,'ready_empty':0,'failed':0,'excluded':0}
    failure_details, group_results = [], []
    for group in groups:
        eligible = preflight['groups'][group['group_id']]['eligible']
        part = dict(group_id=group['group_id'], task=group['task'], eligible=eligible, arms={})
        for arm in ('original','adjacent'):
            part['arms'][arm] = []
            for job in group['arms'][arm]:
                assert job not in seen
                seen.add(job)
                path = RETURN/(job+'.json')
                if not eligible:
                    if path.exists():
                        raise ValueError('Excluded job was run: '+job)
                    status['excluded'] += 1
                    continue
                if not path.exists():
                    raise ValueError('Missing eligible result: '+job)
                result = read(path)
                if result['job_id'] != job or result['input_sha256'] != rows[job]['input_sha256']:
                    raise ValueError('Result identity mismatch: '+job)
                if result['status'] in ('ready','ready_empty'):
                    response = core.strict_json(result['raw_response']['choices'][0]['message']['content'])
                    if core.validate(response, rows[job]) != result['claims']:
                        raise ValueError('Validated claims drift: '+job)
                elif result['status'] == 'failed':
                    failure_details.append(dict(group_id=group['group_id'],task=group['task'],
                                                arm=arm,job_id=job,error=result.get('error'),
                                                finish_reason=result.get('raw_response',{}).get('choices',[{}])[0].get('finish_reason')))
                else:
                    raise ValueError('Unknown result status: '+result['status'])
                status[result['status']] += 1
                part['arms'][arm].append(dict(job_id=job,status=result['status'],
                                              claims=len(result.get('claims',[])),
                                              seconds=result.get('seconds'),usage=result.get('usage')))
        group_results.append(part)
    if seen != set(rows):
        raise ValueError('Not all requests accounted for')
    returned_results = sorted(RETURN.glob('*.json'))
    report = dict(version=1,run_dir=str(RETURN), input_hashes_match=True,
                  runner_core_model_hashes_match=True, request_status=status,
                  eligible_groups=preflight['eligible_groups'],
                  excluded_groups=preflight['excluded_groups'],
                  failures=failure_details, groups=group_results,
                  returned_json_sha256={p.name:digest(p) for p in returned_results},
                  semantic_quality='not mechanically evaluated',
                  caveat='A ready result proves schema and reference ID validity only, not factual support.')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'mechanical_report.json').write_bytes(core.encoded(report)+b'\n')
    print(json.dumps({k:report[k] for k in ('request_status','eligible_groups','excluded_groups','failures')},
                     ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
