"""Conservative, prefix-only evidence; reported writes are not filesystem proof."""
import re

from .schema import ToolClass


class ExecutionContext:
    def __init__(self):
        self.prompts = {}

    def apply(self, step):
        if step.tool not in ('run', 'run_ipython'):
            return
        a = step.attrs
        session = str(a.get('terminal_session', 'default'))
        previous = self.prompts.get(session)
        # Only known waiting prompts; ordinary output mentioning a password is insufficient.
        tail = step.observation.strip().splitlines()[-1:] or ['']
        prompt = bool(re.fullmatch(r'(?:\S+\s+login:|[Pp]assword:|\(qemu\))', tail[0].strip()))
        self.prompts[session] = prompt and step.exit_code is None
        explicit = a.get('is_input') is True
        inferred = previous and step.exit_code is None
        if step.tool == 'run' and (explicit or inferred):
            a['terminal_context'] = 'explicit_input' if explicit else 'previous_waiting_prompt'
            a['original_tool_class'] = step.tool_class
            a['original_cmd_classes'] = a.get('cmd_classes', [])
            a['cmd_classes'] = ['interactive_input']
            a.pop('irreversible', None)
            a.pop('undo', None)
            step.tool_class = ToolClass.OTHER
            step.written_paths = []
            step.touched_paths = []
            return
        classes = set(a.get('cmd_classes', []))
        if 'probe' not in classes:
            return
        # Retain exact provenance for human audit. Do not infer writes from a script name.
        reports = []
        for m in re.finditer(r'(?im)^\s*(?:wrote(?:\s+[^\n:]{0,60})?\s+to|(?:[\w ]{0,40}\s+)?saved\s+to)\s+(/[^\s:]+)', step.observation):
            reports.append({'path': m.group(1).rstrip('.'), 'text': m.group(0),
                            'span': [m.start(), m.end()], 'source': 'tool_output'})
        if reports:
            a['reported_writes'] = reports
            a['probe_purpose'] = 'possible_generation'
            # A generator exiting 0 is not evidence of successful verification.
            a['suppress_probe_verify'] = True
