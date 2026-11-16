"""Bounded prefix evidence, not a purpose classifier or a boundary decision."""
from collections import deque
from copy import deepcopy
import re

VERSION = 'prefix-relations-v2'


def clipped(text, limit):
    text = str(text or '')
    if len(text) <= limit:
        return text
    return text[:limit//2] + '\n[clipped]\n' + text[-limit//2:]


def task_text(value):
    if isinstance(value, list):
        value = '\n'.join(v.get('text','') for v in value if isinstance(v,dict) and v.get('type')=='text')
    value = str(value or '')
    issue = re.search(r'<issue_description>(.*?)</issue_description>', value, re.S)
    return (issue.group(1) if issue else value)[:12000]


def file_refs(text):
    # Lexical references only; no filesystem lookup or cwd inference.
    return list(dict.fromkeys(re.findall(r'[\w./-]+\.(?:py|js|ts|json|toml|yaml|yml|rst|md)\b', text)))[:32]


def path_related(a, b):
    a, b = a.replace('\\','/'), b.replace('\\','/')
    return a == b or (not a.startswith('/') and b.endswith('/'+a)) or (not b.startswith('/') and a.endswith('/'+b))


class RelationEvidence:
    def __init__(self, task):
        self.task = task_text(task)
        self.entities = list(dict.fromkeys(re.findall(r'[A-Za-z_][\w.]*(?:\([^\n)]{0,60}\))?',self.task)))[:256]
        self.errors = deque(maxlen=8)
        self.history = deque(maxlen=8)
        self.last_edit = None
        self.plan = None
        self.last_idx = None

    def feed(self, step):
        idx = step['idx']
        if self.last_idx is not None and idx <= self.last_idx:
            raise ValueError('Increasing steps required')
        self.last_idx = idx
        action = str(step.get('action') or '')[:8192]
        obs = clipped(step.get('observation'), 16384)
        tool = step.get('tool_class','')
        paths = file_refs(action)
        runtime = tool not in ('READ','SEARCH','EDIT','PLAN') and (
            step.get('tool') in ('run','run_ipython','execute_bash') or
            tool in ('EXEC','EXEC_WRITE','VERIFY','INSTALL'))
        signature = ' '.join(action.split())
        zero = bool(re.search(r'\b(?:Ran|Found)\s+0\s+tests?\b|NO TESTS RAN',obs,re.I))
        code = step.get('exit_code')
        if code is None:
            matches = re.findall(r'(?:completed|finished) with exit code (-?\d+)',obs)
            code = int(matches[-1]) if matches else None
        failed = runtime and (zero or (code is not None and code != 0) or
                  bool(re.search(r'Traceback \(most recent call last\)|ModuleNotFoundError:|ImportPathMismatchError:',obs)))
        self.errors = deque([e for e in self.errors if idx-e['step_id']<=32], maxlen=8)
        # Exact successful retry only; success elsewhere never erases prior failure.
        if runtime and code == 0 and not failed:
            self.errors = deque([e for e in self.errors if e['signature'] != signature], maxlen=8)
        if failed:
            reason = 'zero_tests' if zero else 'execution_failed'
            self.errors.append(dict(step_id=idx, kind=reason, signature=signature,
                                   action=clipped(action,1200), observation=clipped(obs,2400), paths=paths))
        if tool=='EDIT' and ('has been edited' in obs or 'File created successfully' in obs) and not obs.lstrip().startswith('ERROR:'):
            self.last_edit=dict(step_id=idx, paths=paths, evidence='tool_reported_edit',action=clipped(action,1200))
        if self.last_edit and idx-self.last_edit['step_id']>32:
            self.last_edit=None
        raw_plan=step.get('public_plan')
        previous_plan=self.plan
        if isinstance(raw_plan,dict) and isinstance(raw_plan.get('task_list'),list):
            tasks=raw_plan['task_list']
            self.plan=dict(step_id=idx, tasks=[{k:str(t.get(k,''))[:400] for k in ('id','title','status')}
                      for t in tasks[:32] if isinstance(t,dict)],truncated=len(tasks)>32,
                      interpretation='agent declaration, not verified execution')
        if self.plan and idx-self.plan['step_id']>32:
            self.plan=None
        links=[]
        for err in self.errors:
            if err['step_id']>=idx:
                continue
            if tool in ('READ','SEARCH') and any(path_related(p,q) for p in paths for q in err['paths']):
                links.append(dict(kind='reads_object_named_in_prior_failed_action',from_step=err['step_id'],
                                  to_step=idx,certainty='lexical_candidate_not_same_goal_proof'))
        completed=[]
        if raw_plan and self.plan:
            before={t['id']:t['status'] for t in (previous_plan or {}).get('tasks',[])}
            completed=[t for t in self.plan['tasks'] if t['status'] in ('done','completed') and
                       before.get(t['id']) not in ('done','completed') and
                       re.search(r'test|verif',t['title'],re.I)]
        tension=bool(completed and self.errors)
        self.history = deque([h for h in self.history if idx-h['step_id']<=32], maxlen=8)
        for h in self.history:
            if any(path_related(p,q) for p in paths for q in h['paths']):
                links.append(dict(kind='shared_lexical_object', from_step=h['step_id'],
                                  to_step=idx, certainty='not_same_goal_proof'))
        outcome = ('zero_tests' if zero else 'execution_failed') if failed else (
            'exit_zero_not_goal_success' if code == 0 else 'unknown')
        # History includes successful reads/commands, failures and declarations alike.
        # It excludes the current step, which is already represented separately.
        history = deepcopy(list(self.history))
        self.history.append(dict(step_id=idx,tool_class=tool,paths=paths,
            action=clipped(action,1200),observation=clipped(obs,1600),
            exit_code=code,outcome=outcome,
            public_plan=deepcopy(self.plan) if self.plan and self.plan['step_id']==idx else None))
        evidence=dict(version=VERSION, observed_through_step=idx,
            task_issue_excerpt=clipped(self.task,2000),
            relevant_task_entities=[x for x in self.entities if len(x)>=4 and x in action][:24],
            current=dict(step_id=idx,tool_class=tool,action=clipped(action,2400),observation=clipped(obs,3200),exit_code=code),
            recent_failure_candidates=[{k:v for k,v in e.items() if k!='signature'} for e in self.errors],
            recent_action_history=history,
            recent_edit=self.last_edit, public_plan=self.plan, relation_candidates=links,
            declaration_evidence_tension_candidate=dict(present=tension,completed_items=completed,
                reason='co-occurrence only, not contradiction; failure may be the intended reproduction outcome'),
            purpose_relation='undetermined', boundary_decision=None,
            limitations=['Lexical evidence only; no goal equivalence inferred.',
                         'Recent failures may have been resolved by a different command.',
                         'Bounded last 32 steps, 8 failures, 32 plan items; no raw future or gold input.'])
        return deepcopy(evidence)
