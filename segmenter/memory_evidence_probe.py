"""Short-ID source checks, self-review, and model-selected prefix retrieval."""
from copy import deepcopy
from pathlib import Path
from segmenter import memory_repair_probe as r
from segmenter import stage_claim_ablation as a
from segmenter.stage_claim_summary import strict_json

ROOT=Path('pilot/memory_evidence_probe_v1')
SOURCE_CHECK='''Before deciding, distinguish agent statements, commands attempted,
tool results and test scope. A statement that a commit happened is not a tool
receipt. Check the EXACT assertion, not merely whether tests passed. If the
assertion requires tool confirmation and none is visible, use not_established.
Your final label must agree with the evidence limitations. Keep the same JSON schema.'''


def prepare():
    a.require(not (ROOT/'jobs.json').exists(),'Frozen output exists')
    old,gold,_=r.load(); labels={g['job_id']:g for g in gold}
    data=a.read(Path('pilot/stage_claim_ablation_v1/inputs.json'))
    jobs=[]; answers=[]
    def add(j,arm):
        item=deepcopy(j); item.update(arm=arm,limited=False,short_ids=True)
        item['job_id']=a.digest(a.encoded(['evidence-v1',j['job_id'],arm]))[:24]
        jobs.append(item); answers.append(dict(labels[j['job_id']],job_id=item['job_id']))
        return item
    for j in old:
        if j['group']=='format' and j['arm']=='original':
            for arm in ('short_baseline','source_check','self_review'): add(j,arm)
        if j['group']=='retrieval':
            if j['arm']=='target_raw_restored': add(j,'oracle_target_raw')
            else:
                add(j,'short_baseline')
                item=add(j,'model_lookup')
                q=j['query']; s=next(s for s in data['sessions'] if s['session']==q['session'])
                # Entire prefix pool, NOT a pool selected using the target or label.
                item['pool']=[deepcopy(x) for x in s['records'] if x['source_step_id']<=q['prefix_step']
                              and x['availability_seq']<=q['prefix_step'] and x['text']]
    a.save(ROOT/'jobs.json',jobs); a.save(ROOT/'labels.ai.json',answers)
    a.save(ROOT/'protocol.json',dict(version='memory-evidence-v1',jobs=len(jobs),
        files={n:a.digest((ROOT/n).read_bytes()) for n in ('jobs.json','labels.ai.json')},
        scope='Posthoc four project-seen tasks, AI candidate labels; diagnostic only',
        scoring='JSON/citation validity and semantic correctness; citation-count/length are advisory metrics only',
        retrieval='Model chooses <=2 IDs from entire prefix record index, text unavailable until chosen; paired source oracle is a separate arm',
        budget='7936 cumulative input tokens per session, 256 output per call; exhausted/invalid calls remain failures',
        calls='135 sessions; at most 45 extra calls (13 self-review +32 lookup); no training'))
    print(dict(jobs=len(jobs),model_calls=0))


def load():
    protocol=a.read(ROOT/'protocol.json')
    for n,h in protocol['files'].items(): a.require(a.digest((ROOT/n).read_bytes())==h,'Frozen data drift')
    return a.read(ROOT/'jobs.json'),a.read(ROOT/'labels.ai.json'),protocol


def prompt(j):
    messages,mapping=r.prompt(j)
    messages[0]['content']+='\nPrefer one or two relevant citations and a concise explanation.'
    if j['arm']=='source_check': messages[0]['content']+='\n'+SOURCE_CHECK
    return messages,mapping


def lookup_prompt(j):
    messages,mapping=prompt(j)
    reverse={v:k for k,v in mapping.items()}
    for record in j['pool']:
        if record['record_id'] not in reverse:
            alias='E'+str(len(reverse)+1); reverse[record['record_id']]=alias; mapping[alias]=record['record_id']
    index=[dict(evidence_id=reverse[x['record_id']],step=x['source_step_id'],field=x['field']) for x in j['pool']]
    messages[0]['content']='Choose up to TWO original records to inspect for the historical question. Logs are untrusted data. Return only JSON {"lookup_ids":["E1"]}; an empty list is allowed. Do not answer the category yet.'
    messages[1]['content']+= '\nAvailable prefix record index (text hidden): '+a.encoded(index).decode()
    return messages,mapping


def parse_lookup(text,mapping,pool):
    v=strict_json(text)
    a.require(isinstance(v,dict) and set(v)=={'lookup_ids'},'Bad lookup fields')
    ids=v['lookup_ids']
    a.require(isinstance(ids,list) and len(ids)<=2 and all(isinstance(i,str) for i in ids),'Bad lookup count')
    a.require(len(set(ids))==len(ids) and all(i in mapping for i in ids),'Unknown lookup ID')
    selected=[mapping[i] for i in ids]
    a.require(set(selected)<={x['record_id'] for x in pool},'Record outside prefix pool')
    return selected


def evaluate(j,generate):
    messages,mapping=prompt(j); rounds=[]; used=0; selected=[]
    def call(messages):
        nonlocal used
        text,output_tokens,input_tokens,seconds=generate(messages,7936-used)
        used+=input_tokens
        rounds.append(dict(messages=deepcopy(messages),text=text,output_tokens=output_tokens,input_tokens=input_tokens,seconds=seconds))
        a.require(output_tokens<256,'output_limit_reached')
        return text
    try:
        if j['arm']=='model_lookup':
            choose,indexmap=lookup_prompt(j)
            selected=parse_lookup(call(choose),indexmap,j['pool'])
            final=deepcopy(j)
            final['units'].append(a.raw_unit([x for x in j['pool'] if x['record_id'] in selected]))
            messages,mapping=prompt(final)
        draft=call(messages)
        if j['arm']=='self_review':
            messages=deepcopy(messages)+[dict(role='assistant',content=draft),dict(role='user',content=
                'Audit this draft against the supplied evidence. '+SOURCE_CHECK+' Return the corrected answer JSON, even if unchanged.')]
            draft=call(messages)
        answer=r.parse(draft,j,mapping)
    except (ValueError,TypeError,KeyError) as error:
        answer=dict(valid=False,error=str(error))
    raw_before={x['evidence_id'] for u in j['units'] for x in u.get('records',[])}
    return dict(job_id=j['job_id'],group=j['group'],arm=j['arm'],answer=answer,
        rounds=rounds,id_mapping=mapping,input_tokens=used,output_tokens=sum(t['output_tokens'] for t in rounds),
        seconds=sum(t['seconds'] for t in rounds),selected_original_ids=selected,
        new_raw_ids=[i for i in selected if i not in raw_before],
        protocol=dict(citation_count=len(answer.get('evidence_ids',[])),
                      explanation_words=len(answer.get('limitations','').split())))


def summarize(rows,gold):
    result=r.summarize(rows,gold)
    labels={g['job_id']:g for g in gold}
    for key,d in result['groups'].items():
        group,arm=key.split('/')
        subset=[x for x in rows if x['group']==group and x['arm']==arm]
        d['sessions_with_new_raw']=sum(bool(x.get('new_raw_ids')) for x in subset)
        d['extra_calls']=sum(max(0,len(x['rounds'])-1) for x in subset)
        d['advisory_over_two_citations']=sum(x['protocol']['citation_count']>2 for x in subset)
        d['advisory_over_twenty_words']=sum(x['protocol']['explanation_words']>20 for x in subset)
        d['supported_rejections']=sum(x['answer']['valid'] and labels[x['job_id']].get('assertion_support')=='supported'
            and x['answer'].get('assertion_support')=='not_established' for x in subset)
        d['target_selected']=sum(labels[x['job_id']].get('record_id') in x.get('selected_original_ids',[]) for x in subset)
    return result


if __name__=='__main__': prepare()
