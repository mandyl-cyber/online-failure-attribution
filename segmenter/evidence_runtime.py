"""Runtime instrumentation seam: trusted file hashes, typed probes and real publication times.

The host must call these methods; agent log text cannot register trusted bindings.
Publication hook records a genuinely published evidence batch, not stage inference.
"""
from pathlib import Path
from copy import deepcopy
import re
import time
from segmenter.evidence_shadow import Shadow
from segmenter.evidence_state import short_view
from segmenter.stage_claim_ablation import digest,encoded,save,require


class RuntimeEvidence:
    def __init__(self,session,workspace,out,clock=time.monotonic_ns):
        self.workspace=Path(workspace).resolve();self.out=Path(out)
        require(not self.out.exists(),'Use a new runtime output directory')
        self.out.mkdir(parents=True);self.clock=clock;self.shadow=Shadow(session)
        self.records=[];self.stages={};self.claims=[];self.publications={};self.event_ids=set()

    def record(self,kind,**values):
        row=dict(sequence=len(self.records)+1,monotonic_ns=self.clock(),kind=kind,**deepcopy(values))
        self.records.append(row)
        with (self.out/'runtime.jsonl').open('ab') as f:f.write(encoded(row)+b'\n')
        return row

    def snapshot_file(self,path):
        path=Path(path).resolve()
        require(path.is_relative_to(self.workspace),'Snapshot outside declared workspace')
        require(path.is_file(),'Snapshot requires an existing regular file')
        return dict(object_id=str(path),version=digest(path.read_bytes()))

    def bind(self,action_id,path):
        require(action_id not in self.event_ids,'Bind before native action arrives')
        b=dict(self.snapshot_file(path),action_id=action_id,provenance='trusted_runtime',
               available_event_seq=self.shadow.seq+1)
        require(action_id not in self.shadow.bindings,'Duplicate runtime binding')
        self.shadow.bindings[action_id]=b
        self.record('binding',binding=b)
        return deepcopy(b)

    def feed(self,event):
        self.event_ids.add(event['id'])
        received=self.record('native_received',event=event)
        rows=self.shadow.feed(event)
        for row in rows:
            # Unchanged file before/after a command is a narrow version guarantee,
            # not proof that dependencies/test files stayed unchanged.
            f=row['fact'];b=self.shadow.bindings.get(f['native_action_id'])
            if b and f['version'] is not None:
                path=Path(b['object_id'])
                if not path.is_file() or digest(path.read_bytes())!=b['version']:
                    f['version']=None
                    next(x for x in self.shadow.state.facts if x['fact_id']==f['fact_id'])['version']=None
            row['evidence_view']=short_view(self.shadow.state.query(row['step'],limit=4))
            self.record('step_ready',step=row['step'],native_received_sequence=received['sequence'],fact=f)
            save(self.out/'steps'/(str(row['step'])+'.json'),row)
        return rows

    def seal(self,stage_id,through_step):
        require(stage_id not in self.stages,'Duplicate stage seal')
        require(0<through_step<=len(self.shadow.state.facts),'Stage exceeds visible prefix')
        self.stages[stage_id]=self.record('stage_sealed',stage_id=stage_id,through_step=through_step)

    def publish(self,stage_id,fact_ids):
        require(stage_id in self.stages and stage_id not in self.publications,'Stage must seal exactly once before publishing')
        seal=self.stages[stage_id]
        visible={f['fact_id']:f for f in self.shadow.state.facts if f['step']<=seal['through_step']}
        require(bool(fact_ids) and set(fact_ids)<=set(visible),'Publication includes invisible evidence')
        payload=dict(stage_id=stage_id,backend='structured_evidence_sidecar',facts=[visible[i] for i in fact_ids])
        # Actual atomic persistence precedes the publication timestamp.
        save(self.out/'batches'/(digest(stage_id.encode())[:24]+'.json'),payload)
        self.publications[stage_id]=self.record('evidence_published',stage_id=stage_id,fact_ids=fact_ids,
            sealed_at_monotonic_ns=seal['monotonic_ns'],backend='structured_evidence_sidecar')

    def statement(self,action_id,text,path):
        require(action_id in self.event_ids,'Statement must first arrive as a native action')
        native=self.shadow.events[action_id]
        require(native.get('action') in ('finish','message'),'Only native agent statements can be decomposed')
        args=native.get('args',{})
        observed=args.get('final_thought',args.get('content',native.get('message','')))
        require(text==observed,'Statement/source text mismatch')
        binding=self.snapshot_file(path)
        # These exact sentences are controlled probes. Arbitrary prose remains
        # unresolved; no semantic object/scope is guessed.
        templates={'All tests passed.':'current_test_pass',
                   'The changes have been committed.':'commit_confirmed'}
        row=self.record('statement_received',action_id=action_id,text=text,binding=binding)
        item=dict(claim_id=digest(encoded([action_id,text,binding]))[:24],received=row,
                  decomposition='exact_template' if text in templates else 'unresolved')
        if text in templates:
            item['claim']=dict(kind=templates[text],prefix_step=len(self.shadow.state.facts),**binding)
        self.claims.append(item)
        self.record('claim_registered',item=item)
        return deepcopy(item)

    def decide(self,item):
        if item['decomposition']=='unresolved':
            decision=dict(state='pending_decomposition',truth_status='not_evaluated')
        else:
            claim=item['claim']; result=self.shadow.state.assess(claim)
            ids={f['fact_id'] for f in result['inspected_facts']}
            ready=set(i for p in self.publications.values() for i in p['fact_ids'])
            eligible=[p for p in self.publications.values() if ids.intersection(p['fact_ids'])]
            if ids and not ids<=ready:
                decision=dict(state='pending_publication',assertion_support=None,required_fact_ids=sorted(ids))
            else:
                decision=dict(state='assessed',**result)
            decision['timing']=dict(statement_to_decision_ns=self.clock()-item['received']['monotonic_ns'],
                latest_required_publication_ns=max((p['monotonic_ns'] for p in eligible),default=None),
                scope='Instrumentation overhead in this process; not model extraction latency or alarm lead time')
        return self.record('claim_decision',claim_id=item['claim_id'],decision=decision)

    def close(self):
        decisions=[r for r in self.records if r['kind']=='claim_decision']
        summary=dict(native=self.shadow.summary(),claims=len(self.claims),decision_attempts=len(decisions),
            assessed=sum(r['decision']['state']=='assessed' for r in decisions),
            pending_publication=sum(r['decision']['state']=='pending_publication' for r in decisions),
            pending_decomposition=sum(r['decision']['state']=='pending_decomposition' for r in decisions),
            published_batches=len(self.publications),
            scope='Runtime sidecar wiring. Host supplies bindings and actual stage callbacks; no autonomous-agent or failure-alarm efficacy claim.')
        save(self.out/'summary.json',summary)
        return summary
