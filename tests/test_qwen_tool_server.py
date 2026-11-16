import unittest
from remote_pilots.qwen_tool_server.server import prepare,parse_completion

TOOLS=[dict(type='function',function=dict(name='probe',parameters={'type':'object'}))]


class ToolServerTests(unittest.TestCase):
    def test_call(self):
        m,r=parse_completion('<tool_call>{"name":"probe","arguments":{"id":"x"}}</tool_call>',TOOLS,True)
        self.assertEqual(r,'tool_calls');self.assertEqual(m['tool_calls'][0]['function']['arguments'],'{"id": "x"}')
    def test_truncation_never_emits_call(self):
        m,r=parse_completion('<tool_call>{"name":"probe","arguments":{}}</tool_call>',TOOLS,False)
        self.assertEqual(r,'length');self.assertNotIn('tool_calls',m)
    def test_history_normalized_without_mutation(self):
        original=dict(messages=[dict(role='assistant',content=None,tool_calls=[dict(function=dict(name='probe',arguments='{}'))]),dict(role='tool',content='ok')],tools=TOOLS)
        messages,tools=prepare(original)
        self.assertEqual(messages[0]['tool_calls'][0]['function']['arguments'],{})
        self.assertEqual(original['messages'][0]['tool_calls'][0]['function']['arguments'],'{}')
        self.assertEqual(tools,TOOLS)
    def test_invalid_calls(self):
        for text in ['<tool_call>{}</tool_call>','<tool_call>{"name":"other","arguments":{}}</tool_call>','<tool_call>{']:
            with self.assertRaises(ValueError):parse_completion(text,TOOLS,True)
    def test_unsupported_modes(self):
        for extra in [dict(stream=True),dict(tool_choice='required'),dict(tool_choice={'type':'function'}),dict(response_format={'type':'json_object'})]:
            with self.assertRaises(ValueError):prepare(dict(messages=[dict(role='user',content='hi')],**extra))


if __name__=='__main__':unittest.main()
