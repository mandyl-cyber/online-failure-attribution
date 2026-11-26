"""Read-only native-event shadow replay/follow. Never executes logged actions."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
import re
from pathlib import Path
import time
from segmenter.adapters.openhands import OpenHandsStreamAdapter
from segmenter.evidence_state import EvidenceState,native_fact,short_view
from segmenter.stage_claim_ablation import read,save,require,digest,encoded


class Shadow:
    def __init__(self,session,bindings=(),claims=()):
        self.session=session;self.adapter=OpenHandsStreamAdapter();self.state=EvidenceState()
        self.events={};self.seq=0;self.snapshots=[];self.bindings={};self.claims=list(claims)
        for b in bindings:
            require(b.get('provenance')=='trusted_runtime','Bindings must declare trusted_runtime provenance')
            require(isinstance(b.get('available_event_seq'),int) and b['available_event_seq']>0,'Binding visibility required')
            require(b['action_id'] not in self.bindings,'Duplicate binding')
            self.bindings[b['action_id']]=deepcopy(b)

    def feed(self,event):
        require(event.get('id') is not None,'Native event ID required for audited provenance')
        self.seq+=1
        previous=self.events.get(event.get('id'))
        if previous is not None: require(previous==event,'Conflicting native event ID')
        if event.get('id') is not None: self.events[event['id']]=deepcopy(event)
        rows=[]
        for step in self.adapter.feed(event):
            ref=step.raw_ref;action=self.events[ref['action_id']]
            obs=self.events.get(ref.get('observation_id'))
            b=self.bindings.get(ref['action_id'],{})
            if b and b['available_event_seq']>self.seq: b={}
            ids=[self.session+':native:'+str(ref['action_id'])]
            if obs is not None:ids.append(self.session+':native:'+str(ref['observation_id']))
            fact=native_fact(action,obs,step.idx,ids,b)
            issues=step.attrs['input_integrity']['issues']
            if issues: fact['status']='unknown'
            self.state.append(fact)
            latest=self.state.query(step.idx,kind='test_run',limit=1)
            decisions=[]
            for c in self.claims:
                if c['prefix_step']!=step.idx: continue
                if not c.get('object_id'):
                    decisions.append(dict(claim=c,state='pending_binding',missing=['object_id']))
                else: decisions.append(dict(claim=c,state='assessed',decision=self.state.assess(c)))
            row=dict(session=self.session,step=step.idx,received_event_seq=self.seq,
                native_action_timestamp=action.get('timestamp'),native_observation_timestamp=obs.get('timestamp') if obs else None,
                input_integrity=deepcopy(step.attrs['input_integrity']),fact=fact,
                latest_observed_test_status=latest[0]['status'] if latest else 'unknown',
                evidence_view=short_view(self.state.query(step.idx,limit=4)),claim_decisions=decisions,
                timing=dict(claim_publication_time=None,alarm_latency=None,
                    reason='No claim-publication or alarm timestamps are supplied by native event input'))
            self.snapshots.append(row);rows.append(row)
        return rows

    def summary(self):
        facts=self.state.facts;n=len(facts)
        return dict(session=self.session,events=self.seq,steps=n,
            kind_counts=dict(Counter(f['kind'] for f in facts)),status_counts=dict(Counter(f['status'] for f in facts)),
            unclassified=sum(f['kind']=='unclassified' for f in facts),
            unknown_objects=sum(f['object_id'] is None for f in facts),unknown_versions=sum(f['version'] is None for f in facts),
            pending_actions=len(self.adapter._pending),adapter_diagnostics=self.adapter.diagnostics,
            scope='Read-only engineering shadow; event arrival order, not measured stage/claim availability or alarm effectiveness')


def compare_models(shadows,root):
    rows=[]
    frozen=read(Path('pilot/stage_claim_ablation_v1/inputs.json'))
    refs={(s['session'],r['source_step_id']):r['raw_ref']['action_id'] for s in frozen['sessions'] for r in s['records']}
    for path in sorted((root/'answers').glob('*.json')):
        x=read(path)
        if not isinstance(x.get('case_id'),str):continue
        parts=x['case_id'].split(':')
        if len(parts)<2 or parts[0] not in shadows:continue
        prefix=int(parts[1][6:]); shadow=shadows[parts[0]]
        snapshot=next((s for s in shadow.snapshots if s['step']==prefix),None)
        if snapshot is None:continue
        require(refs.get((parts[0],prefix))==snapshot['fact']['native_action_id'],'Model/shadow native step mapping mismatch')
        answer=x['answer']; status=answer.get('latest_run_status')
        rows.append(dict(case_id=x['case_id'],arm=x['arm'],model_valid=answer['valid'],model_status=status,
            shadow_status=snapshot['latest_observed_test_status'],
            status_disagrees=bool(answer['valid'] and status!=snapshot['latest_observed_test_status']),
            support_comparison='unavailable: model free-text assertion lacks trusted object/version binding',
            scope='Disagreement diagnostic, not ground-truth accuracy'))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('pilot','replay','follow'))
    p.add_argument('--events',type=Path);p.add_argument('--session',default='shadow')
    p.add_argument('--bindings',type=Path);p.add_argument('--claims',type=Path)
    p.add_argument('--model-returns',type=Path)
    p.add_argument('--out',type=Path,default=Path('out/evidence_shadow_v1'))
    args=p.parse_args();require(not args.out.exists(),'Use a new --out; input file is always read-only')
    bindings=read(args.bindings) if args.bindings else [];claims=read(args.claims) if args.claims else []
    sources=[]
    if args.command=='pilot':
        require(not bindings and not claims,'For per-session bindings/claims use replay/follow explicitly')
        base=Path('pilot/claim_prospective_eval_v1');protocol=read(base/'protocol.json')
        for s in protocol['sessions']:
            path=base/s['path'];require(digest(path.read_bytes())==s['sha256'],'Native source drift')
            sources.append((s['session_id'],path))
    else:
        require(args.events is not None,'--events required');sources=[(args.session,args.events)]
    shadows={}
    require(all(re.fullmatch(r'[A-Za-z0-9_-]+',s) for s,_ in sources),'Session must be a simple identifier')
    args.out.mkdir(parents=True)
    save(args.out/'config.json',dict(command=args.command,sources=[(s,str(p.resolve())) for s,p in sources],
        code=digest(Path(__file__).read_bytes()),evidence_code=digest(Path(__file__).with_name('evidence_state.py').read_bytes()),
        bindings=bindings,claims=claims,no_model_inference=True,no_trajectory_execution=True))
    for session,path in sources:
        shadow=Shadow(session,bindings,claims);shadows[session]=shadow
        # Text mode buffers partial lines; only newline-terminated events enter adapter.
        pending=''
        with path.open(encoding='utf-8-sig') as stream:
            while True:
                chunk=stream.readline()
                if not chunk:
                    if args.command!='follow':break
                    opened=os.fstat(stream.fileno());current=path.stat()
                    require((opened.st_dev,opened.st_ino)==(current.st_dev,current.st_ino),'Input rotated; restart in a new output directory')
                    require(path.stat().st_size>=stream.tell(),'Input file truncated; restart in a new output directory')
                    save(args.out/session/'summary.json',shadow.summary());time.sleep(1);continue
                pending+=chunk
                if not pending.endswith('\n'):continue
                event=json.loads(pending);pending=''
                for row in shadow.feed(event):
                    save(args.out/session/'steps'/(str(row['step'])+'.json'),row)
                if args.command=='follow':save(args.out/session/'summary.json',shadow.summary())
        # EOF never flushes unfinished actions into fabricated observations.
        save(args.out/session/'summary.json',dict(shadow.summary(),incomplete_trailing_line=bool(pending)))
    comparisons=compare_models(shadows,args.model_returns) if args.model_returns else []
    if args.model_returns:require(bool(comparisons),'No compatible case_id model answers found under --model-returns/answers')
    save(args.out/'model_comparisons.json',comparisons)
    result=dict(sessions=[s.summary() for s in shadows.values()],model_comparisons=len(comparisons),
                model_status_disagreements=sum(r['status_disagrees'] for r in comparisons),
                claim_timing='unavailable; no publication timeline supplied',
                scope='Seen native replay engineering diagnostic; not held-out online alarm evaluation')
    save(args.out/'summary.json',result);print(result)


if __name__=='__main__':main()
