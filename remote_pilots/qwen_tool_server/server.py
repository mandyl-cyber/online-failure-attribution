"""Local Qwen2.5 non-streaming tool-call probe server. Executes no tools."""
import argparse
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import re
import time
import uuid


def output_budget(request):
    values=[request[k] for k in ('max_tokens','max_completion_tokens') if request.get(k) is not None]
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in values):
        raise ValueError('Output token budget must be a positive integer')
    if len(set(values))>1:raise ValueError('Conflicting max_tokens and max_completion_tokens')
    return values[0] if values else 1024


def prepare(request):
    if request.get('stream'):
        raise ValueError('Streaming is not supported by this probe server')
    if request.get('n',1)!=1 or request.get('stop'):
        raise ValueError('Only n=1 and no custom stop supported')
    choice=request.get('tool_choice','auto')
    if choice not in ('auto','none'):
        raise ValueError('Only auto/none tool_choice supported; forced calls are not implemented')
    tools=request.get('tools') or []
    if not isinstance(tools,list):raise ValueError('tools must be a list')
    names=[]
    for t in tools:
        if t.get('type')!='function' or not isinstance(t.get('function'),dict):raise ValueError('Invalid tool')
        name=t['function'].get('name')
        if not isinstance(name,str) or not name or name in names:raise ValueError('Invalid/duplicate tool name')
        names.append(name)
    messages=deepcopy(request['messages'])
    if not isinstance(messages,list) or not messages:raise ValueError('messages must be nonempty')
    for m in messages:
        # Qwen's template serializes tool arguments itself: normalize API JSON strings.
        for call in m.get('tool_calls') or []:
            fn=call['function'];args=fn['arguments']
            if isinstance(args,str):args=json.loads(args)
            if not isinstance(args,dict):raise ValueError('Tool arguments must be an object')
            fn['arguments']=args
        if m.get('content') is None:m['content']=''
        if isinstance(m['content'],list):
            parts=m['content']
            if any(not isinstance(x,dict) or x.get('type')!='text' or not isinstance(x.get('text'),str) for x in parts):
                raise ValueError('Only text content parts supported; images are not accepted')
            m['content']='\n'.join(x['text'] for x in parts)
        if not isinstance(m['content'],str):raise ValueError('Only string message content supported')
    if choice=='none':tools=[]
    if request.get('response_format'):
        raise ValueError('response_format is not implemented by this tool probe server')
    return messages,tools


def parse_completion(text,tools,ended):
    if not ended:
        return {'role':'assistant','content':text},'length'
    blocks=list(re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S))
    residue=re.sub(r'<tool_call>\s*.*?\s*</tool_call>','',text,flags=re.S).strip()
    if '<tool_call>' in residue or '</tool_call>' in residue:raise ValueError('Malformed tool-call tags')
    if not blocks:return {'role':'assistant','content':text},'stop'
    names={t['function']['name'] for t in tools};calls=[]
    for b in blocks:
        obj=json.loads(b.group(1))
        if not isinstance(obj,dict) or set(obj)!={'name','arguments'}:raise ValueError('Malformed generated tool call')
        if obj['name'] not in names or not isinstance(obj['arguments'],dict):raise ValueError('Unknown tool or invalid arguments')
        calls.append(dict(id='call_'+uuid.uuid4().hex,type='function',function=dict(
            name=obj['name'],arguments=json.dumps(obj['arguments'],ensure_ascii=False))))
    return dict(role='assistant',content=residue or None,tool_calls=calls),'tool_calls'


def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True);p.add_argument('--model-name',default='qwen-local')
    p.add_argument('--port',type=int,default=30001);p.add_argument('--context-limit',type=int,default=16384)
    a=p.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True,trust_remote_code=False)
    if '<tool_call>' not in str(tokenizer.chat_template):raise ValueError('Expected Qwen tool-aware chat template')
    model=AutoModelForCausalLM.from_pretrained(a.model_path,dtype=torch.bfloat16,
          local_files_only=True,trust_remote_code=False).to('cuda').eval()
    limit=min(a.context_limit,int(model.config.max_position_embeddings))
    eos=model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    class Handler(BaseHTTPRequestHandler):
        def send_json(self,status,payload):
            body=json.dumps(payload,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            self.send_json(200,dict(object='list',data=[dict(id=a.model_name,object='model')])) if self.path=='/v1/models' else self.send_json(404,dict(error='Unknown endpoint'))
        def do_POST(self):
            if self.path!='/v1/chat/completions':self.send_json(404,dict(error='Unknown endpoint'));return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=8*1024*1024:raise ValueError('Invalid body size')
                r=json.loads(self.rfile.read(size))
                if r.get('model')!=a.model_name:raise ValueError('Unexpected model ID')
                messages,tools=prepare(r)
                maximum=output_budget(r);temp=float(r.get('temperature',0))
                if maximum<1 or temp<0:raise ValueError('Invalid generation parameters')
                inputs=tokenizer.apply_chat_template(messages,tools=tools or None,add_generation_prompt=True,
                       return_dict=True,return_tensors='pt').to('cuda')
                count=inputs['input_ids'].shape[-1]
                if count+maximum>limit:raise ValueError(f'Input {count} + output {maximum} exceeds {limit}; no silent truncation')
                kwargs=dict(max_new_tokens=maximum,do_sample=temp>0)
                if temp>0:kwargs['temperature']=temp
                with torch.inference_mode():generated=model.generate(**inputs,**kwargs)
                completion=generated[0,count:];text=tokenizer.decode(completion,skip_special_tokens=True)
                ended=bool(len(completion) and int(completion[-1]) in eos)
                message,reason=parse_completion(text,tools,ended)
                self.send_json(200,dict(id='chatcmpl-'+uuid.uuid4().hex,object='chat.completion',created=int(time.time()),
                     model=a.model_name,choices=[dict(index=0,message=message,finish_reason=reason)],
                     usage=dict(prompt_tokens=count,completion_tokens=len(completion),total_tokens=count+len(completion))))
            except (ValueError,KeyError,TypeError,AttributeError) as e:
                self.send_json(400,dict(error=dict(message=str(e),type='request_or_generation_validation_error')))
    print(f'Serving tool probe at http://127.0.0.1:{a.port}/v1 '
          f'(non-streaming only; effective context limit={limit}, including output)',flush=True)
    HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()


if __name__=='__main__':main()
