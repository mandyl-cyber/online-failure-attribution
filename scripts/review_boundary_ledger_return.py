"""Fixed AI review of the returned rollout; never edit frozen references."""
from collections import Counter
from pathlib import Path
import json

from scripts.run_stage_claim_pipeline import read, save
from scripts.audit_neural_ledger_v3_return import require
from scripts.review_claim_prospective_eval_return import CLAIMS, FACTS
from segmenter.ledger import sha, encoded


# Per session, per sealed job, in local claim order. These are explicit readings,
# not a heuristic semantic grader. Re-review is required if outputs change.
NEURAL = {
 'stable_unique': [
  [('supported','Listing and zero-test report are cited; implementation detail is omitted.'),('supported','Zero tests are explicitly reported.')],
  [('supported','The failed replacement receipt says no replacement.'),('supported','The receipt acknowledges the edit without asserting correctness.')]],
 'bounded_chunks': [
  [('supported','Zero tests, source read and empty probe are shown; negative-size condition omitted.')],
  [('supported','Guard and zero tests are supported; remaining loop bound omitted.'),('supported','Preserves agent-report status; the original broader fixed claim is narrowed.')]],
 'prefix_totals': [
  [('partial','One run has three failures; then test_2 fails invents an additional temporal event.'),('unsupported','Only a test command is cited, not a result or agent assertion of three failures.')],
  [('partial','Only the first failed replacement had the wrong signature. The second failure and successful edit receipt are uncited; the cited test failure is reduced to no pass.')],
  [('supported','Invalid new_str is observed; equal-old/new reason omitted.'),('partial','Added comment is supported, but two displays/edits are not established by this single action and receipt.')],
  [('unsupported','One test invocation reports pass/fail/pass. The claim invents fail/pass/fail and repeated failures; the next events are edit attempts.'),('partial','Actions support proposed replacements, not completion of the second edit.'),('partial','The comment is visible; an isolated after snapshot does not establish no function change.')],
  [('supported','The cited test reports one failure; actual/expected vector omitted.'),('supported','All three invalid-new_str observations are cited; detailed reason omitted.'),('partial','Three comment lines are visible, but only one was added by this edit; snapshot is confused with delta.')]],
 'config_overrides': [
  [('supported','Three failures and one success are reported.'),('partial','if v is truthiness filtering, not merely non-None filtering.')],
  [('supported','Edit acknowledgment and later indentation exception are supported by tool results.'),('supported','The two-failure result and following indentation exception are cited.')],
  [('supported','Receipt displays the new if indentation.'),('supported','Traceback identifies the base[k] assignment indentation.')],
  [('supported','Receipt shows conditional update and the test reports failure.'),('unsupported','Failing assertion compares x with its original dictionary, not x with z; the later identity assertion is not the failed assertion.')],
  [('supported','Copy-before-update receipt and four-test success are both cited.')],
  [('partial','Agent-report basis is retained, but a finish statement is still mislabeled as a commit event/receipt.')]],
}

# (coverage, local job:claim, explanation). Same frozen facts as the rule arm.
NF = {
 'stable_unique': {
  1:('full','1:c2','Zero tests retained.'),2:('missing','','Generic listing mention omits sorted/set behavior.'),
  3:('partial','2:c1','Failure retained, mismatch reason omitted.'),4:('partial','2:c2','Generic receipt omits seen-dictionary/input-order behavior.')},
 'bounded_chunks': {
  1:('full','1:c1','Zero tests retained.'),2:('missing','','Loop bound omitted.'),
  3:('partial','1:c1','Empty probe retained, negative-size condition omitted.'),
  4:('partial','2:c1','Guard retained; unchanged loop bound omitted.'),5:('full','2:c1','Zero tests retained.'),
  6:('partial','2:c2','Agent status retained, but broad fixed assertion is narrowed to invalid sizes.')},
 'prefix_totals': {
  1:('full','1:c1','Three failures retained in a clause; chronology error separately scored.'),
  2:('partial','2:c1','First failed replacement retained.'),3:('partial','2:c1','Signature difference retained; actual signature omitted.'),
  4:('missing','','Second failed replacement observation is uncited; proposed common cause is wrong.'),
  5:('missing','','In-place mutation omitted.'),6:('missing','','Successful edit observation is uncited.'),
  7:('missing','','No pass does not preserve the observed failed test outcome and diagnostic.'),
  8:('partial','3:c1','Invalid parameter retained; equal-old/new reason omitted.'),
  9:('full','3:c2','Added comment is a supported subclause with receipt citation.'),
  10:('partial','4:c1','Coarse failure mentioned, but test order/outcomes fabricated; not full coverage.'),
  11:('missing','','Action-only equality description omits observed rejection; other claim mislabels events as tests.'),
  12:('full','4:c3','Added comment subclause supported; no-change inference separately scored.'),
  13:('partial','5:c1','Failure retained, actual/expected vector omitted.'),
  14:('partial','5:c2','Invalid parameter retained; detailed reason omitted.'),
  15:('partial','5:c2','Invalid parameter retained; detailed reason omitted.'),
  16:('partial','5:c2','Invalid parameter retained; detailed reason omitted.'),
  17:('full','5:c3','Comment presence retained; snapshot-versus-delta error separately scored.')},
 'config_overrides': {
  1:('full','1:c1','Observed scope retained.'),3:('partial','1:c2','Truthiness mischaracterized as non-None.'),
  5:('full','2:c1','Edit acknowledged.'),6:('full','2:c2','Two-failure outcome retained.'),
  7:('missing','','Second edit acknowledgment is not described, despite source being cited.'),
  8:('full','2:c2','Indentation exception retained.'),9:('full','3:c1','Edit acknowledged.'),
  10:('full','3:c2','Indentation exception retained.'),11:('full','4:c1','Edit acknowledged.'),
  12:('full','4:c1','Observed suite failure retained; erroneous cause separately scored.'),
  13:('full','5:c1','Copy assignment and conditional update retained.'),14:('full','5:c1','Four passing tests retained.'),
  15:('partial','6:c1','Committed assertion retained as agent report, but finish event still mislabeled commit.')},
}


def main():
    out=Path('out/boundary_ledger_return_audit_v1')
    rows=read(out/'claims.json')
    require(sha(encoded(rows))=='c4c93bb77c3d16f35e531211f77b44c22c74eb6ee09ca408e1840d5478b7b6e6','Outputs changed; fresh review needed')
    pack=Path('pilot/claim_prospective_eval_v1')
    before=(pack/'references.json').read_bytes()
    require(sha(before)==read(pack/'protocol.json')['reference_sha256'],'Reference changed')
    original=[json.loads(x) for x in (pack/'stage_jobs.jsonl').read_text(encoding='utf-8').splitlines()]
    old_index={(r['session'],r['job']['job_id']):f'{i:04d}' for i,r in enumerate(original,1)}
    reviewed=[];known={};old_keys={}
    for row in rows:
        arm,name=row['arm'],row['session']; number=int(Path(row['file']).stem)
        local=row['claim']['local_id']; idx=int(local[1:])-1
        if arm=='rules':
            old=old_index[name,row['job_id']]
            judgment,note=CLAIMS[old][idx]
        else:judgment,note=NEURAL[name][number-1][idx]
        key=f'{arm}/{name}/{number}:{local}'
        reviewed.append(dict(row,claim_key=key,reviewer_kind='AI',reviewed=True,judgment=judgment,review_note=note))
        known[key]=row
        if arm=='rules':old_keys[f'{old}:{local}']=key
    require(len(rows)==sum(map(len,CLAIMS.values()))+sum(len(cs) for jobs in NEURAL.values() for cs in jobs),'Incomplete claim review')
    coverage=[];negative={'zero_tests','edit_failed','verification_failed','runtime_exception'}
    for arm in ('rules','neural'):
        for fact in read(pack/'references.json')['facts']:
            name,step=fact['session'],fact['step_id']
            jobs=[(int(p.stem),read(p)) for p in (Path('boundary_ledger_rollout_v1')/arm/name/'stage_jobs').glob('*.json')]
            matching=[(n,j) for n,j in jobs if j['stage']['start_step_id']<=step<=j['stage']['end_step_id']]
            require(len(matching)==(1 if fact['eligibility']=='sealed_stage' else 0),'Fact sealing eligibility changed')
            keys=[]
            if not matching:status,note='not_generated_open_stage','No forced EOF seal; no generation.'
            else:
                table=FACTS if arm=='rules' else NF
                status,key,note=table[name][step]
                if key:
                    key=old_keys[key] if arm=='rules' else f'{arm}/{name}/{key}'
                    claim=known[key];job=matching[0][1]
                    require(claim['job_id']==job['job_id'],'Cross-stage fact mapping')
                    # Necessary provenance check only, never automatic entailment.
                    source_ids={e['record_id'] for e in job['evidence'] if e.get('step_id')==step and e.get('field')==fact['source_field']}
                    if fact.get('record_id'):source_ids.add(fact['record_id'])
                    require(any(r['evidence_id'] in source_ids for r in claim['claim']['evidence_refs']),'Fact coverage lacks source citation')
                    keys=[key]
            coverage.append(dict(arm=arm,fact_id=fact['fact_id'],importance=fact['importance'],category=fact['category'],coverage=status,
                claim_keys=keys,review_note=note,reviewer_kind='AI',coarse_negative_outcome_retained=(status in ('full','partial') if matching and fact['category'] in negative else None)))
    summary=read(out/'summary.json');semantic={}
    for arm in ('rules','neural'):
        sealed=[r for r in coverage if r['arm']==arm and r['coverage']!='not_generated_open_stage']
        semantic[arm]=dict(claim_judgments=dict(Counter(r['judgment'] for r in reviewed if r['arm']==arm)),
            sealed_fact_coverage=dict(Counter(r['coverage'] for r in sealed)),
            sealed_key_fact_coverage=dict(Counter(r['coverage'] for r in sealed if r['importance']=='key_outcome')),
            coarse_negative_outcomes_retained=sum(r['coarse_negative_outcome_retained'] is True for r in sealed),
            sealed_negative_outcomes=sum(r['coarse_negative_outcome_retained'] is not None for r in sealed))
    summary.update(semantic_review=semantic,decision='retain_rule_default_semantic_gate_failed',review_scope='Explicit non-blind AI judgments, not human accuracy or independent generalization. Correct subclauses can cover facts in partially wrong claims; coarse negative retention does not establish correct cause/order. Four known synthetic sessions have no stage gold; no boundary F1 measured.')
    save(out/'claim_ai_review.json',reviewed)
    save(out/'fact_coverage.json',coverage)
    issues=[r for r in reviewed if r['judgment']!='supported']
    save(out/'semantic_issues.json',issues);save(out/'summary.json',summary)
    destination=Path('pilot/boundary_ledger_rollout_v1')
    for filename,value in [('return_summary.json',summary),('return_semantic_issues.json',issues),('return_fact_coverage.json',coverage)]:
        (destination/filename).write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    require((pack/'references.json').read_bytes()==before,'Frozen reference modified')
    print(semantic)


if __name__=='__main__':main()
