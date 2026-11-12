"""Opt-in public plan-state features; no task titles/IDs or annotation input."""
import math
from collections import Counter,deque


def enrich(rows,payloads):
    history=deque(maxlen=4);known=False;previous={};age=1000;result=[]
    for row in rows:
        payload=payloads.get(row['idx'])
        age=min(1000,age+1)
        state=previous
        if payload is not None:
            # IDs only reconcile updates; their values never become features.
            state={x.get('id',str(i)):x.get('status','unknown').lower() for i,x in enumerate(payload['task_list'])}
        old_active={k for k,v in previous.items() if v in ('in_progress','doing','started')}
        active={k for k,v in state.items() if v in ('in_progress','doing','started')}
        done=lambda s:sum(v in ('done','completed') for v in s.values())
        d=dict(plan_update=int(payload is not None),plan_known=int(known or payload is not None),
               plan_active_changed=int(known and active!=old_active),
               plan_completed_delta=done(state)-done(previous),
               plan_changed_items=sum(previous.get(k)!=v for k,v in state.items()) if known and payload is not None else 0)
        if payload is not None:age=0;known=True;previous=state
        d.update(plan_age=math.log1p(age),plan_tasks=math.log1p(len(state)),
                 plan_active=math.log1p(len(active)),plan_completed=math.log1p(done(state)))
        history.append(d)
        extra={f'lag{lag}:{k}':v for lag,item in enumerate(reversed(history)) for k,v in item.items()}
        result.append(dict(row,features={**row['features'],**extra}))
    return result
