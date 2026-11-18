"""Create a compact, source-linked v3/v4 review packet for returned Qwen outputs."""
import json
from pathlib import Path

from segmenter import stage_claim_summary as core
from segmenter import stage_claim_summary_v4 as v4


ROOT = Path('pilot/adjacent_claim_pair_v1')
NEW = Path('stage_claim_facts_v1')
OLD = Path('adjacent_claim_pair_v1')
OUT = Path('pilot/stage_claim_facts_v1_return')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    groups = load(ROOT/'groups.json')
    references = {x['group_id']:x for x in load(ROOT/'reference_points.json')}
    originals = {x['job_id']:x for x in (core.strict_json(line) for line in
        (ROOT/'requests.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())}
    mapping = load(NEW/'mapping.json')
    preflight = load(NEW/'preflight.json')
    packet = []
    totals = dict(eligible_groups=0, old_ready=0, new_ready=0, old_failed=0,
                  new_failed=0, old_claims=0, new_claims=0, new_missing_critical=0,
                  new_exact_fact_copies=0,
                  old_seconds=0., new_seconds=0., old_prompt_tokens=0,
                  new_prompt_tokens=0., old_completion_tokens=0,
                  new_completion_tokens=0.)
    for group in groups:
        gid = group['group_id']
        eligible = preflight['groups'][gid]['eligible']
        item = dict(group_id=gid, task=group['task'], eligible=eligible,
                    reference_points=references[gid]['points'],
                    forbidden_inferences=references[gid]['forbidden_inferences'],
                    arms={})
        if eligible:
            totals['eligible_groups'] += 1
        for arm in ('original','adjacent'):
            results = []
            if eligible:
                for old_id in group['arms'][arm]:
                    new_id = mapping[old_id]
                    old = load(OLD/(old_id+'.json'))
                    new = load(NEW/(new_id+'.json'))
                    row = v4.build(originals[old_id]['source_snapshot'],
                                   originals[old_id]['chunk_chars'])
                    assert row['job_id'] == new_id
                    assert new['source_job_id'] == old_id
                    assert new['input_sha256'] == row['input_sha256']
                    one = dict(old_job_id=old_id, new_job_id=new_id,
                               facts=[dict(type=f['type'],step=f['step_id'],
                                           test_count=f['test_count'],exit_code=f['exit_code'])
                                      for f in row['observed_facts']],
                               old_status=old['status'], new_status=new['status'],
                               old_claims=[x['text'] for x in old.get('claims',[])],
                               new_claims=[x['text'] for x in new.get('claims',[])],
                               new_error=new.get('error'),
                               new_missing_critical=new.get('missing_critical_fact_ids',[]),
                               old_seconds=round(old.get('seconds',0),3),
                               new_seconds=round(new.get('seconds',0),3),
                               old_usage=old.get('usage'), new_usage=new.get('usage'))
                    fact_sentences = {fact['summary'] for fact in row['observed_facts']}
                    one['new_exact_fact_copies'] = sum(
                        claim['text'] in fact_sentences for claim in new.get('claims',[]))
                    totals['new_exact_fact_copies'] += one['new_exact_fact_copies']
                    if new['status'] == 'failed' and new.get('raw_response'):
                        one['failed_raw_claims'] = [x['text'] for x in core.strict_json(
                            new['raw_response']['choices'][0]['message']['content']).get('claims',[])]
                    results.append(one)
                    for label, value in [('old',old),('new',new)]:
                        totals[label+'_ready'] += value['status'] in ('ready','ready_empty')
                        totals[label+'_failed'] += value['status'] == 'failed'
                        totals[label+'_claims'] += len(value.get('claims',[]))
                        totals[label+'_seconds'] += value.get('seconds',0)
                        usage = value.get('usage') or {}
                        totals[label+'_prompt_tokens'] += usage.get('prompt_tokens',0)
                        totals[label+'_completion_tokens'] += usage.get('completion_tokens',0)
                    totals['new_missing_critical'] += bool(new.get('missing_critical_fact_ids'))
            item['arms'][arm] = results
        packet.append(item)
    for key in ('old_seconds','new_seconds'):
        totals[key] = round(totals[key],3)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'paired_review_packet.json').write_bytes(core.encoded(packet)+b'\n')
    (OUT/'mechanical_summary.json').write_bytes(core.encoded(totals)+b'\n')
    print(json.dumps(totals,ensure_ascii=False,indent=2))
    print('Failed:',[(x['task'],arm,r['new_error']) for x in packet
                     for arm,rs in x['arms'].items() for r in rs if r['new_status']=='failed'])


if __name__=='__main__':
    main()
