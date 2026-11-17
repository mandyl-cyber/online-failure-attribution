"""Lossless source-role annotations for a diagnostic input ablation.

Hints are deterministic format detection, not semantic truth labels. Original
text, chunks and evidence IDs stay unchanged; ambiguous records stay unclassified.
"""
from copy import deepcopy
from . import stage_claim_summary as core


def render(row):
    core.verify_row(row)
    request=deepcopy(row['request'])
    request['source_policy']={
        'task_context':'User-provided background, not observed stage evidence; do not report its examples as executed results.',
        'agent_plan':'Agent intentions and self-reported status/notes. done is a reported status, not independently verified success.',
        'plan_update_receipt':'Confirms only the update and item count. Does not verify the plan contents, test outcomes or implementation.',
        'action':'Shows a requested/attempted operation, not successful execution.',
        'observation':'Tool-returned text may contain echoed commands, source code or agent-authored messages. Its transport does not make every clause a verified fact.',
        'commit_message':'Agent-authored description embedded in command/output; commit receipt proves commit occurrence, not behavioral correctness.',
        'summary_focus':'Retain specific changes, observed verification scope and unresolved blockers; avoid replacing them with generic task-list updates.',
        'classification_limit':'Format-based hints only. Unclassified records require inspecting the original text. No hindsight or model labels are used.'}
    for unit in request['units']:
        action=''.join(c['text'] for p in unit['action'] for c in p['chunks'])
        try: parsed=core.strict_json(action)
        except (ValueError,TypeError): parsed=None
        plan=isinstance(parsed,dict) and parsed.get('command')=='plan' and isinstance(parsed.get('task_list'),list)
        # This textual signature is a hint, never an execution/commit-success detector.
        commit='git commit' in action
        unit['source_roles']={'action':'agent_plan' if plan else 'action',
                              'observation':'observation',
                              'detection_basis':'public action shape and receipt signature; no native tool identity in this pack'}
        obs=''.join(c['text'] for p in unit['observation'] for c in p['chunks'])
        if plan and obs.startswith('Task list has been updated with '):
            unit['source_roles']['observation']='plan_update_receipt'
        if commit:
            unit['source_roles']['embedded_authored_text']='commit_message'
    return request
