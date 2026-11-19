"""Freeze dev labels and replace only the observed-fact sidecar in one condition."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from segmenter import stage_claim_facts_v2

SOURCE=Path('pilot/claim_sft_reviewed_v2/dev.jsonl')
OUT=Path('pilot/claim_sidecar_input_ablation_v1')


def sha(blob):return hashlib.sha256(blob).hexdigest()


def replace_sidecar(row):
    result=deepcopy(row)
    request=json.loads(row['prompt'][1]['content'])
    evidence=[]
    mapping={}
    for unit in request['units']:
        for field in ('action','observation'):
            for record in unit[field]:
                evidence.append(dict(record_id=record['record_id'],source_step_id=unit['step_id'],field=field,
                    text=''.join(c['text'] for c in record['chunks'])))
                for chunk in record['chunks']:
                    mapping[chunk['id']]=dict(record_id=record['record_id'],source_step_id=unit['step_id'],field=field)
    facts=stage_claim_facts_v2.extract(dict(evidence=evidence),mapping)
    changed=deepcopy(request)
    changed['observed_facts_version']=stage_claim_facts_v2.VERSION
    changed['observed_facts']=[{k:v for k,v in f.items() if k!='summary'} for f in facts]
    assert {k:v for k,v in request.items() if k not in ('observed_facts','observed_facts_version')}=={k:v for k,v in changed.items() if k not in ('observed_facts','observed_facts_version')}
    result['prompt'][1]['content']=json.dumps(changed,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    assert result['completion']==row['completion'] and result['prompt'][0]==row['prompt'][0]
    return result


def main():
    raw=SOURCE.read_bytes()
    assert sha(raw)=='dba4154cd69cf1beb200d0a075fab42a1a36b27e016c728ee7a051a86cb70012'
    rows=[json.loads(x) for x in raw.decode('utf-8').splitlines()]
    changed=[replace_sidecar(r) for r in rows]
    for name,blob in [('original',raw),('sidecar_v2',('\n'.join(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':')) for r in changed)+'\n').encode())]:
        p=OUT/name/'dev.jsonl';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob)
    hashes={n:sha((OUT/n/'dev.jsonl').read_bytes()) for n in ('original','sidecar_v2')}
    protocol=dict(protocol='claim-sidecar-input-ablation-v1',examples=len(rows),parents=len({r['parent_stage_key'] for r in rows}),task_groups=len({r['group'] for r in rows}),
        source=SOURCE.as_posix(),file_hashes=hashes,reference_labels_identical=True,system_prompt_identical=True,raw_units_identical=True,
        only_request_changes=['observed_facts','observed_facts_version'],
        model='Same frozen trained7B NF4+LoRA adapter in both conditions; do not retrain',
        labels='Original v2 dev, not v4_scopefix targets',semantic_review='pending_generation_and_review',
        scope='Development input comparison; not independent acceptance or training. Native runtime defaults unchanged.')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'protocol.json').write_bytes((json.dumps(protocol,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(protocol,indent=2))


if __name__=='__main__':main()
