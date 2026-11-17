"""Persist manual AI judgments for the inspected retry1 outputs; not an automatic judge."""
from collections import Counter
from pathlib import Path
import statistics
from remote_pilots.openhands_smoke import seal_claim_diagnostic as pilot
from remote_pilots.openhands_smoke import claims_v3 as io
from segmenter import stage_claim_summary as core

# Citation-grounded judgment, including source attribution. 'partial' includes
# partially supported clauses, status loss, or misattribution; not full correctness.
JUDGMENTS={
0:{'baseline':[
 ('partial','Update receipts show eight items, not the phase contents; action citations are missing.'),
 ('unsupported','Empty citation list; details are not supported by a cited stage record.'),
 ('unsupported','S1 observation only says no task list; issue description is not a tool finding.'),
 ('unsupported','Same irrelevant no-task-list citation; no subgroup execution in stage.'),
 ('unsupported','Empty citations; restates user instructions as an agent claim.')],
 'evidence_first':[
 ('supported','No-task-list receipt supports initial absence.'),
 ('partial','Creation is supported, but an observed tool update is labeled agent_assertion.'),
 ('supported','Latest action shows phase1 in_progress and remaining phases todo; not verified task success.')]},
3:{'baseline':[
 ('unsupported','Receipt only reports two task items; passing tests and no-regression are agent notes, not tool observations.'),
 ('unsupported','Receipt does not establish committed file scope; assertion in notes was upgraded to tool fact.')],
 'evidence_first':[
 ('supported','Commit output supports occurrence and quoted commit message, not correctness of the claimed fix.'),
 ('unsupported','Same wrong update receipt; upgrades agent-reported tests/no-regression to tool verification.'),
 ('unsupported','Update receipt does not verify final review or absence of extraneous committed files.')]},
6:{'baseline':[
 ('supported','Action and edit receipt show added record_unapplied call for the replacement migration.'),
 ('unsupported','Cited snippet does not name enclosing method; unapply is not the unapply_migration named in plan.'),
 ('partial','Update receipt supports an update, not specific phase/next-step information.')],
 'evidence_first':[
 ('partial','Edit is supported; ensuring consistency with requirement is stronger than edit evidence alone.'),
 ('supported','Public plan explicitly names unapply_migration and intended change; accurately labels it as plan content.')]},
9:{'baseline':[
 ('partial','Receipts show item count; BytesWarning/pytest context is not supported by those citations.'),
 ('supported','Plan says restate and analyze requirements; phrased as a plan rather than completed work.')],
 'evidence_first':[
 ('supported','Receipt supports initial absence.'),
 ('partial','Creation observed in tool receipt, but labeled agent_assertion.'),
 ('unsupported','Contradicted: only phase1 is in_progress; phases2-8 remain todo, and phase1 changed from todo.')]},
12:{'baseline':[
 ('partial','Byte-conversion plan supports broad topic but cited plan lacks OPeNDAP/pydap detail.'),
 ('partial','Plan proposes _Unsigned=False handling, not the claimed observed data failure.'),
 ('unsupported','Cited plan does not supply engine=pydap reproduction details; borrowed from task context.'),
 ('unsupported','variables.py line311 is not in cited plan; task context cannot be cited via unrelated stage IDs.'),
 ('supported','Plan explicitly proposes unsigned-to-signed conversion for _Unsigned=False.')],
 'evidence_first':[
 ('supported','Action and receipt support the eight-item update.'),
 ('partial','Only signed/unsigned topic and proposed False handling are in citation, not full backend behavior.'),
 ('unsupported','netcdf4/pydap comparison is absent from cited action; task statement is not execution evidence.'),
 ('partial','False handling is planned, but same-location-as-True detail is not in the cited plan.')]},
15:{'baseline':[
 ('unsupported','Task-update receipt does not show materialization of any dataset.'),
 ('supported','Commit message supports claimed intended method changes, labeled agent_assertion.'),
 ('supported','Git output supports commit occurrence and message.'),
 ('supported','Task note explicitly asserts minimality/lazy access; source correctly marks agent assertion, not verification.')],
 'evidence_first':[
 ('partial','Commit occurrence is supported but labeled assertion; intended effect not runtime verification.'),
 ('supported','Commit message supports paths and claimed intended lazy behavior as agent assertion.'),
 ('supported','Explicitly attributes implementation description to commit message.'),
 ('partial','Task is marked done, not simply a future plan; loses the reported completion status.')]},
18:{'baseline':[
 ('partial','Two concrete outputs are supported; wording confirm the issue is imprecise and correct-for-all-strings exceeds the two cases.'),
 ('supported','Agent diagnosis is present in cited notes and labeled assertion.'),
 ('partial','Description is in plan notes but full implementation statement spans the uncited next chunk.'),
 ('unsupported','Cites a plan/update receipt, not the execution result; fix-worked upgraded from agent notes.'),
 ('supported','Successful removal command and git clean status support cleanup and no uncommitted changes.')],
 'evidence_first':[
 ('supported','Script contents and creation receipt support file creation and purpose.'),
 ('partial','Output supports two cases, but action citation is missing and wording generalizes string roles.'),
 ('partial','Notes describe implementation as done; future plan wording loses that status and citation omits next chunk.'),
 ('supported','Receipt reports seven updated items; statement is a coarse description of the plan update.'),
 ('partial','Cleanup and working-tree cleanliness supported; no pending commits is ambiguous, not proof of push/sync status.')]},
21:{'baseline':[
 ('partial','Plan contents supported but tool update occurrence is labeled assertion; omits concrete purpose/status.'),
 ('unsupported','Same item count does not prove consistency; phase1 status changed.')],
 'evidence_first':[
 ('supported','Explicitly lists planned phases, matching action content.'),
 ('supported','Receipts and viewed plan support stored updates and organizational progress, not completed code work.')]}
}

OMISSIONS={
0: {'baseline':['Latest phase1 in_progress / later todo status','Concrete next planned activities'],
    'evidence_first':['Concrete planned reproduction, exploration, implementation and verification']},
3: {'baseline':['Modified/untracked files','Commit identity and change scope','Custom reproduction blocked by app registry'],
    'evidence_first':['Modified/untracked files','Commit hash and line-change scope','Custom reproduction blocked by app registry']},
6: {'baseline':['Testing in_progress and reproduction todo'], 'evidence_first':['Testing in_progress and reproduction todo']},
9: {'baseline':['Current phase status and planned later work'], 'evidence_first':['Correct task statuses','Concrete planned later work']},
12:{'baseline':['Latest phase statuses and verification plan'], 'evidence_first':['Latest phase statuses and reproduction/verification plan']},
15:{'baseline':['Commit hash and line-change counts'], 'evidence_first':['Commit hash and line-change counts','Accurate reported final-review done status']},
18:{'baseline':['Agent-reported full-suite failures and limited verification scope'],
    'evidence_first':['Agent-reported full-suite failures and limited verification scope']},
21:{'baseline':['Link-parsing objective','Latest phase statuses'], 'evidence_first':['Link-parsing objective','Latest phase statuses']}
}


def main():
    returned=Path('seal_claim_run_v3_retry1'); out=Path('pilot/seal_claim_return_v3_retry1_audit')
    _,rows,requests=pilot.load(returned)
    for name in ('inputs.jsonl','requests.jsonl','review.json','protocol.json'):
        if (returned/name).read_bytes()!=(pilot.ROOT/name).read_bytes(): raise ValueError('pack drift')
    references=io.read(returned/'review.json'); pre=io.read(returned/'preflight.json')
    reviews=[]; metrics={}; hashes={}
    for variant in ('baseline','evidence_first'):
        verdicts=Counter(); accepted=Counter(); durations=[];tokens=[];ready=0
        for index in JUDGMENTS:
            ref=references[index]; job=ref['job_id']; path=returned/(job+'_'+variant+'.json')
            record=io.read(path);hashes[path.name]=core.digest(path.read_bytes())
            response=record['raw_response']; choice=response['choices'][0]
            if choice['finish_reason']!='stop': raise ValueError('unexpected incomplete response')
            expected=next(c['prompt_tokens'] for c in pre['checks'] if c['job_id']==job and c['variant']==variant)
            if response['usage']['prompt_tokens']!=expected: raise ValueError('prompt mismatch')
            parsed=core.strict_json(choice['message']['content'])
            # A top-level list remains a rejected output; inspect contents for diagnosis only.
            claims=parsed if isinstance(parsed,list) else parsed['claims']
            decisions=JUDGMENTS[index][variant]
            if len(claims)!=len(decisions): raise ValueError('review count mismatch')
            schema_error=None
            try: core.validate(parsed,rows[job])
            except ValueError as exc: schema_error=str(exc)
            if (schema_error is None)!=(record['status']=='ready'): raise ValueError('status mismatch')
            ready+=record['status']=='ready';durations.append(record['seconds']);tokens.append(response['usage']['completion_tokens'])
            judgments=[]
            for n,(claim,(verdict,note)) in enumerate(zip(claims,decisions),1):
                verdicts[verdict]+=1
                if record['status']=='ready': accepted[verdict]+=1
                judgments.append(dict(claim_index=n,claim=claim,citation_grounded_verdict=verdict,note=note))
            reviews.append(dict(job_id=job,case_index=index,variant=variant,reviewed=True,reviewer_type='AI',
                blinded=False,status=record['status'],schema_error=schema_error,claims=judgments,
                missing_or_inaccurate_key_points=OMISSIONS[index][variant],
                note='Manual qualitative reference comparison; no numeric coverage rate assigned. Invalid outputs remain failures.'))
        metrics[variant]=dict(attempted=8,ready=ready,all_returned_claims=dict(verdicts),ready_output_claims=dict(accepted),
                              median_seconds=statistics.median(durations),total_seconds=sum(durations),
                              output_tokens=sum(tokens))
    out.mkdir(parents=True,exist_ok=False)
    io.save(out/'ai_review.json',reviews)
    io.save(out/'summary.json',dict(metrics=metrics,source_result_hashes=hashes,
        total_selected_stages=24,reviewed_stages=8,paired_both_ready=6,
        scope='Eight short, project-seen stages; manual AI audit with visible variant; not human gold or population accuracy.',
        conclusion='Both arms 7/8 structurally valid. Evidence-first has mixed semantic changes and a new task-status contradiction; no deployment claim.',
        verdict_definition='Supported requires cited evidence and source attribution to support the whole statement; partial includes missing clauses/status or attribution issues. Unsupported may still repeat true user context with a wrong citation.'))
    print(metrics)


if __name__=='__main__': main()
