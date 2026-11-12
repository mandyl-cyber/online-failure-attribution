import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from segmenter.audit_plan_payloads import public_payload,find_calls
from segmenter.plan_features import enrich


def test_native_call_isolation():
    task={'id':'p1','title':'Run tests','status':'in_progress','thought':'hidden'}
    args={'command':'plan','task_list':[task],'gold_stage':9,'thought':'hidden'}
    cleaned=public_payload(args)
    assert cleaned=={'command':'plan','task_list':[{'id':'p1','title':'Run tests','status':'in_progress'}]}
    call={'id':'current','function':{'name':'task_tracker','arguments':args}}
    future={'id':'future','function':{'name':'task_tracker','arguments':{'task_list':[{'title':'future'}]}}}
    assert find_calls([call,future],{'current'})=={'current':cleaned}
    assert find_calls([call,call],{'current'})=={'current':cleaned}
    future['id']='current'
    try:find_calls([call,future],{'current'})
    except ValueError:pass
    else:raise AssertionError('Conflicting native IDs accepted')


def test_plan_prefix():
    rows=[dict(idx=i,features={'lag0:tool':'PLAN'}) for i in range(1,5)]
    payloads={1:{'task_list':[{'id':'private-id','title':'hidden-title','status':'in_progress'}]},
              3:{'task_list':[{'id':'private-id','title':'hidden-title','status':'done'}]}}
    result=enrich(rows,payloads)
    for n in range(1,5):assert enrich(rows[:n],payloads)==result[:n]
    assert result[2]['features']['lag0:plan_completed_delta']==1
    assert result[2]['features']['lag0:plan_active_changed']==1
    assert 'private-id' not in str(result) and 'hidden-title' not in str(result)
    assert enrich(rows,{})[0]['features']['lag0:plan_known']==0


if __name__=='__main__':
    test_native_call_isolation()
    test_plan_prefix()
    print('plan payload whitelist, native-call isolation and duplicate tests passed')
