"""Two-turn tool protocol probe. No external tool or shell command is executed."""
import argparse
import json
import urllib.request
import urllib.error


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-url',default='http://127.0.0.1:30001/v1')
    a=p.parse_args()
    def ask(messages,tools,choice):
        body=dict(model='qwen-local',messages=messages,tools=tools,tool_choice=choice,temperature=0,max_tokens=512)
        req=urllib.request.Request(a.base_url+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=180) as response:result=json.load(response)
        except urllib.error.HTTPError as e:
            raise SystemExit(e.read().decode(errors='replace'))
        print(json.dumps(result,ensure_ascii=False,indent=2));return result['choices'][0]
    tools=[dict(type='function',function=dict(name='get_probe_status',description='Return diagnostic probe status.',
           parameters=dict(type='object',properties=dict(probe_id=dict(type='string')),required=['probe_id'])))]
    messages=[dict(role='user',content="Call get_probe_status with probe_id='swe-online-001'. After receiving its result, repeat its receipt exactly.")]
    first=ask(messages,tools,'auto')
    calls=first['message'].get('tool_calls',[])
    if first['finish_reason']!='tool_calls' or len(calls)!=1:raise SystemExit('FAIL: expected one structured tool call')
    call=calls[0]
    if call['function']['name']!='get_probe_status' or json.loads(call['function']['arguments'])!={'probe_id':'swe-online-001'}:
        raise SystemExit('FAIL: wrong tool or arguments')
    receipt='PROBE_RESULT_73921'
    messages += [first['message'],dict(role='tool',tool_call_id=call['id'],content=json.dumps(dict(status='ok',receipt=receipt)))]
    second=ask(messages,tools,'none')
    if second['finish_reason']!='stop' or receipt not in (second['message'].get('content') or ''):
        raise SystemExit('FAIL: final response did not consume tool receipt')
    print('PASS: structured tool request and tool-result round trip; not a full OpenHands compatibility test')


if __name__=='__main__':main()
