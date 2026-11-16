"""Read-only analysis of returned Qwen outputs; never repairs or accepts claims."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from .run_ledger_sensitivity import check_schema


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='swe_ledger_sensitivity_v1')
    args=ap.parse_args();source=Path(args.results);base=Path('pilot/swe_ledger_sensitivity_v1')
    out=Path('pilot/swe_ledger_return_audit_v1');out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Audit already exists')
    requests={r['job_id']:r['request'] for r in map(json.loads,(base/'requests.jsonl').read_text(encoding='utf-8').splitlines())}
    manifest=read(base/'remote_manifest.json');jobs={r['job_id']:r for r in manifest['jobs']}
    config=read(source/'config.json')
    lf_sha=hashlib.sha256((base/'requests.jsonl').read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    assert config['input_sha256'] in (sha(base/'requests.jsonl'),lf_sha)
    counts=Counter();refs=Counter();arms=defaultdict(Counter);details=[];enums=Counter();tokens=Counter()
    for jid,request in requests.items():
        path=source/(jid+'.json');r=read(path);assert r['job_id']==jid
        arm=jobs[jid]['arm'];error=r.get('error','none');counts[error]+=1;arms[arm][error]+=1
        reply=r.get('raw_response');detail=dict(job_id=jid,arm=arm,status=r['status'],error=error,source_sha256=sha(path))
        if reply:
            counts['has_model_response']+=1
            for k,v in reply.get('usage',{}).items():
                if isinstance(v,int):tokens[k]+=v
            choice=reply['choices'][0];detail['finish_reason']=choice['finish_reason']
            content=choice['message'].get('content') or ''
            detail['response_chars']=len(content)
            if choice['finish_reason']!='stop':
                details.append(detail);continue  # never parse/recover truncated JSON
            try:parsed=json.loads(content)
            except json.JSONDecodeError:
                detail['leading_text']=content[:160];details.append(detail);continue
            counts['complete_json']+=1
            try:check_schema(parsed,request['response_schema']);detail['schema_valid']=True;counts['schema_valid']+=1
            except ValueError as exc:detail['schema_valid']=False;detail['schema_error']=str(exc)
            known={e['record_id']:e['text'] for e in request['evidence']}
            for c in parsed.get('claims',[]):
                counts['claims_in_complete_json']+=1
                enums['kind:'+str(c.get('kind'))]+=1;enums['basis:'+str(c.get('basis'))]+=1
                for ref in c.get('evidence_refs',[]):
                    refs['total']+=1
                    text=known.get(ref.get('evidence_id'));quote=ref.get('quote');a=ref.get('start');b=ref.get('end')
                    if text is None or not isinstance(quote,str) or not quote:
                        refs['unknown_id_or_invalid_quote']+=1;continue
                    if type(a)==int and type(b)==int and 0<=a<b<=len(text) and text[a:b]==quote:
                        refs['original_exact_span']+=1
                    else:
                        matches=[];offset=0
                        while True:
                            found=text.find(quote,offset)
                            if found<0:break
                            matches.append(found);offset=found+1
                        if len(matches)==1:refs['wrong_span_but_unique_literal_quote']+=1
                        elif matches:refs['wrong_span_ambiguous_literal_quote']+=1
                        else:refs['quote_not_literal_in_evidence']+=1
            details.append(detail)
    assert len(details)==48
    summary=dict(requests=48,config=config,input_lf_sha256=lf_sha,counts=dict(counts),by_arm={a:dict(c) for a,c in arms.items()},
        references_in_complete_json=dict(refs),enums=dict(enums),reported_token_usage=dict(tokens),
        original_valid=0,repaired_claims=0,semantic_quality='not_evaluated',
        limitation='First-error counts overlap underlying problems. Reference counts inspect only complete stop+JSON outputs; not an accuracy score.',
        inputs={str(base/'requests.jsonl'):sha(base/'requests.jsonl'),str(source/'summary.json'):sha(source/'summary.json')})
    for name,value in [('summary.json',summary),('per_job.json',details)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
