"""Conservative evidence sidecar. Native provenance, explicit bindings, no log execution."""
import argparse
from copy import deepcopy
from pathlib import Path
import re
from segmenter.stage_claim_ablation import encoded, digest, read, save, require

KINDS=('test_run','commit','edit','agent_statement','unclassified')


def observed_test_status(text):
    counts=re.findall(r'^Ran (\d+) tests? in .+$',text,re.M)
    if len(counts)>1: return 'unknown'  # Combined commands need separate records.
    if counts:
        if int(counts[0])==0: return 'zero_tests'
        if re.search(r'^FAILED\b',text,re.M): return 'failed'
        if re.search(r'^OK(?:\s*\([^\n]*\))?\s*$',text,re.M): return 'passed'
    if re.search(r'^(?:IndentationError|SyntaxError|ModuleNotFoundError|ImportError):',text,re.M):
        return 'blocked'
    return 'unknown'


def native_fact(action, observation, step, evidence_ids, binding=None):
    """Action/observation must be paired by native IDs before calling this seam.

    Object/version bindings are supplied by the caller, never inferred from prose.
    The native observation is retained as source; no command is executed.
    """
    binding=binding or {}
    command=action.get('args',{}).get('command','')
    source='agent_assertion' if action.get('action') in ('finish','message','think') else 'unknown'
    kind='agent_statement' if source=='agent_assertion' else 'unclassified'
    status='unknown'
    content=observation.get('content','') if observation else ''
    if not isinstance(content,str): content=''
    is_tool=action.get('action')=='run' and observation is not None and observation.get('observation')=='run'
    if is_tool:
        source='tool_observation'
        # Only a single unittest command or a single git commit is supported.
        simple=isinstance(command,str) and not re.search(r'[\n;&|`<>]',command)
        if simple and re.match(r'^\s*python(?:3(?:\.\d+)?)?\s+-m\s+unittest\b',command):
            kind='test_run'; status=observed_test_status(content)
        if simple and re.match(r'^\s*git\s+commit(?:\s|$)',command):
            kind='commit'
            exit_code=observation.get('extras',{}).get('exit_code')
            status='confirmed' if exit_code==0 and re.search(r'^\[[^\]\n]+ [0-9a-f]{7,40}\]',content,re.M) else 'unknown'
    text=content if observation else action.get('args',{}).get('final_thought',action.get('message',''))
    return dict(fact_id=digest(encoded([step,evidence_ids,kind,text,binding]))[:24],step=step,
        available_step=step,kind=kind,status=status,source=source,
        object_id=binding.get('object_id'),version=binding.get('version'),
        scope=binding.get('scope','displayed native command only'),evidence_ids=list(evidence_ids),
        native_action_id=action.get('id'),native_observation_id=observation.get('id') if observation else None,
        command=command,text=text)


class EvidenceState:
    def __init__(self):
        self.facts=[]

    def append(self,fact):
        f=deepcopy(fact)
        require(f['kind'] in KINDS,'Unknown fact kind')
        require(isinstance(f['step'],int) and f['available_step']>=f['step'],'Invalid causal time')
        require(bool(f['evidence_ids']),'Evidence IDs required')
        require(not any(x['fact_id']==f['fact_id'] for x in self.facts),'Duplicate fact')
        self.facts.append(f)

    def query(self,prefix,object_id=None,kind=None,version=None,limit=4):
        require(limit>0,'Positive retrieval limit required')
        selected=[f for f in self.facts if f['step']<=prefix and f['available_step']<=prefix
            and (object_id is None or f['object_id']==object_id)
            and (kind is None or f['kind']==kind) and (version is None or f['version']==version)]
        return deepcopy(sorted(selected,key=lambda f:(f['step'],f['available_step']),reverse=True)[:limit])

    def assess(self,claim):
        """Typed claim, not a free-prose truth classifier. Unknown is not false."""
        kind=claim['kind']; prefix=claim['prefix_step']
        require(kind in ('observed_test_pass','current_test_pass','commit_confirmed'),'Unsupported obligation')
        require(bool(claim.get('object_id')),'Claim needs an explicit object binding')
        required='commit' if kind=='commit_confirmed' else 'test_run'
        candidates=self.query(prefix,claim['object_id'],required,limit=max(1,len(self.facts)))
        inspected=[]; missing=[]; supported=False
        if kind=='current_test_pass' and claim.get('version') is None:
            missing.append('current_version_binding')
        else:
            if kind=='current_test_pass':
                candidates=[f for f in candidates if f['version']==claim['version']]
            if candidates:
                # Never cherry-pick an earlier pass over a newer failure/unknown run.
                latest_step=candidates[0]['step']
                inspected=[f for f in candidates if f['step']==latest_step]
                expected='confirmed' if kind=='commit_confirmed' else 'passed'
                supported=all(f['source']=='tool_observation' and f['status']==expected for f in inspected)
                if not supported: missing.append('matching_successful_tool_result')
            else: missing.append('matching_tool_evidence')
        if kind=='current_test_pass' and supported:
            edits=self.query(prefix,claim['object_id'],'edit',limit=max(1,len(self.facts)))
            if any(f['step']>=inspected[0]['step'] for f in edits):
                supported=False; missing.append('reverification_after_edit')
        return dict(assertion_support='supported' if supported else 'not_established',
            truth_status='not_evaluated',obligation=kind,missing=missing,
            evidence_ids=list(dict.fromkeys(i for f in inspected for i in f['evidence_ids'])),
            inspected_facts=inspected,scope='Limited to typed obligation and trusted upstream bindings; not full task correctness')


def replay(events,claims):
    state=EvidenceState()
    for f in events: state.append(f)
    return [dict(claim=c,decision=state.assess(c)) for c in claims]


def short_view(facts):
    """Stable reversible IDs for a bounded query; never rewrite original text."""
    ids=sorted({i for f in facts for i in f['evidence_ids']})
    mapping={f'E{n+1}':i for n,i in enumerate(ids)}
    reverse={v:k for k,v in mapping.items()}
    view=deepcopy(facts)
    for f in view: f['evidence_ids']=[reverse[i] for i in f['evidence_ids']]
    return dict(facts=view,id_mapping=mapping)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--facts',type=Path,required=True)
    p.add_argument('--claims',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('out/evidence_state_replay.json'))
    args=p.parse_args()
    save(args.out,dict(results=replay(read(args.facts),read(args.claims)),scope='Sidecar obligation check; no model inference or trajectory execution'))


if __name__=='__main__': main()
