"""Fixed AI judgments: claim support and frozen-fact coverage are separate."""
from collections import Counter
from pathlib import Path

from scripts.run_stage_claim_pipeline import read,save
from scripts.audit_neural_ledger_v3_return import require
from segmenter.ledger import sha,encoded


CLAIMS={
 '0001':[('supported','The cited results report zero tests and display the set-based source.'),('supported','The failed replacement reports no match/no replacement; wording is imprecise but does not establish a successful edit.')],
 '0002':[('supported','The edit receipt reports the file edit and displays the new function; no correctness claim.')],
 '0003':[('supported','Zero tests, a source read and an empty probe result are shown; the summary omits the negative-size condition.')],
 '0004':[('supported','The receipt acknowledges an edit and displays code, but does not summarize the actual guard or loop bound.')],
 '0005':[('supported','The native verification result explicitly reports zero tests.')],
 '0006':[('partial','Three tests fail in one run. The added then test_2 fails wording implies an extra temporal event not established by that single result.'),('unsupported','Only the unittest command is cited; an action request contains no failure count or agent assertion of that count.')],
 '0007':[('partial','Both replacements failed, but the claimed original-version explanation is not what the tool reports: old_str did not appear verbatim.'),('supported','The grep and source output show the actual signature/body; interpret different relative to the failed proposed old_str.')],
 '0008':[('supported','The receipt displays the initial zero and accumulation loop; this does not imply all input values are included.'),('partial','The cited edit action proposes old/new function text. It should explicitly say requested rather than imply a completed replacement.')],
 '0009':[('supported','The cited output shows test_0 passing, test_1 failing with AssertionError, and an invalid new_str response.')],
 '0010':[('supported','The receipt displays an added comment and an already-present initial zero.'),('unsupported','Only an edit action is cited. It is not a receipt acknowledging completion, regardless of agent_assertion label.')],
 '0011':[('partial','One test fails, but the missing expected fourth value 5 is described as an extra fifth element. The error direction/index are wrong; invalid edit parameters are supported.')],
 '0012':[('supported','The receipt displays modified source.'),('partial','The old and new action text both already contain the initial zero. Only the comment is newly added; the change attribution is wrong.')],
 '0013':[('supported','The result reports one failure.'),('unsupported','Only the unittest command is cited, with no solve result. A result in another observation cannot repair this citation.')],
 '0014':[('supported','All three cited replacements report invalid new_str.')],
 '0015':[('supported','The receipt displays the added comment and ends with edit-review text; no behavioral fix is claimed.')],
 '0016':[('supported','The output reports three failures and one passing test.'),('partial','Filtering with if v excludes False, zero and empty strings, not just None. Non-null is an incorrect characterization.'),('unsupported','Both citations are commands, not observations or agent reports of their results.')],
 '0017':[('supported','The receipt acknowledges the edit and displays the update line.')],
 '0018':[('supported','The suite reports two failures.'),('unsupported','The cited action is only the unittest invocation; it does not contain AssertionError.')],
 '0019':[('supported','The receipt displays a loop and conditional assignment; syntax correctness is not asserted.'),('partial','This citation shows a requested old/new replacement. It should preserve that action/request status.')],
 '0020':[('supported','The traceback reports IndentationError.')],
 '0021':[('supported','The receipt shows the updated if line, without claiming the remaining indentation is correct.')],
 '0022':[('supported','The traceback identifies task.py and IndentationError.')],
 '0023':[('supported','The edit observation supports the receipt/source description; the thought-logged citation is unnecessary.'),('supported','The displayed conditional assignment is guarded by v is not None; this does not assert input immutability.')],
 '0024':[('supported','The suite reports one failure.'),('unsupported','The command alone does not show that a failing test was printed. The claim improperly reports an outcome from action-only evidence.')],
 '0025':[('supported','The receipt acknowledges the edit and displays the function, without preserving the copy-before-update detail.')],
 '0026':[('partial','Observed test success is supported within its scope. The other citation is an agent finish statement, not a commit message/receipt; source/event attribution is wrong.')],
}

# Full requires the substantive frozen fact, not exact wording; partial means
# the coarse event is present but a condition, reason or diagnostic is missing
# or wrong. These are explicit AI readings, not citation-overlap scoring.
FACTS={
 'stable_unique':{
  1:('full','0001:c1','Zero tests explicitly retained.'),
  2:('partial','0001:c1','Set-based implementation retained, sorted-return detail omitted.'),
  3:('partial','0001:c2','Failed edit retained, exact old_str mismatch reason not stated.'),
  4:('partial','0002:c1','Edit acknowledged, but the seen dictionary/input-order implementation is not described.'),
 },
 'bounded_chunks':{
  1:('full','0003:c1','Zero tests retained.'),
  2:('missing','','Reading the file does not preserve the loop-bound observation.'),
  3:('partial','0003:c1','Empty result retained, negative-size condition omitted.'),
  4:('partial','0004:c1','Generic edit receipt omits both guard and remaining loop bound.'),
  5:('full','0005:c1','Zero tests retained.'),
  6:('missing','','The contradictory agent fixed/unverified assertion is omitted.'),
 },
 'prefix_totals':{
  1:('full','0006:c1','The three failing tests are represented; the claim has a separate chronology error.'),
  2:('partial','0007:c1','Failed replacement retained, reason is misrepresented.'),
  3:('full','0007:c2','The actual signature is retained.'),
  4:('partial','0007:c1','Failed replacement retained, reason is misrepresented.'),
  5:('missing','','The in-place mutation in the listed source is not summarized.'),
  6:('partial','0008:c1','Initial zero retained; truncating loop-bound detail omitted.'),
  7:('partial','0009:c1','Failure retained; specific actual/expected output is absent.'),
  8:('partial','0009:c1','Invalid replacement retained; equal-old/new reason omitted.'),
  9:('full','0010:c1','Added comment acknowledged in the receipt.'),
 10:('partial','0011:c1','Failure retained, but element direction/index are wrong.'),
 11:('partial','0011:c1','Invalid replacement retained; equal-old/new reason omitted.'),
 12:('partial','0012:c1','A generic modified-source receipt omits the specific comment; the other claim cites only the action and misstates the initial-zero change.'),
 13:('partial','0013:c1','One failure retained, actual/expected vector omitted.'),
 14:('partial','0014:c1','Invalid replacement retained, equal-old/new reason omitted.'),
 15:('partial','0014:c1','Invalid replacement retained, equal-old/new reason omitted.'),
 16:('partial','0014:c1','Invalid replacement retained, equal-old/new reason omitted.'),
 17:('full','0015:c1','Added comment acknowledged.'),
 },
 'config_overrides':{
  1:('full','0016:c1','Three failures and one pass retain the observed scope.'),
  3:('partial','0016:c2','Truthiness is misdescribed as non-null filtering.'),
  5:('full','0017:c1','The generic edit-acknowledgment fact is retained.'),
  6:('full','0018:c1','The observed suite and two-failure outcome are retained; no claim of general success.'),
  7:('full','0019:c1','Edit acknowledgment is retained.'),
  8:('full','0020:c1','IndentationError retained.'),
  9:('full','0021:c1','Edit acknowledgment retained.'),
 10:('full','0022:c1','IndentationError retained.'),
 11:('full','0023:c1','Edit acknowledgment retained.'),
 12:('full','0024:c1','One failure in the observed suite retained.'),
 13:('partial','0025:c1','Edit acknowledged, but copying base and returning result are omitted.'),
 14:('full','0026:c1','Explicit within-shown-scope success is supported by the native four-test output; no exact verbal numeral is required.'),
 15:('missing','','The agent claim that changes were committed is not accurately retained; finish is instead mislabeled as a commit message.'),
 },
}


def main():
    out=Path('out/claim_prospective_eval_return_audit_v1')
    rows=read(out/'claims.json')
    require(sha(encoded(rows))=='fc78d193d58a05e3028f4ddf5ff650ad9927a16749074307ef05578370081162','Output changed; fresh AI review required')
    pack=Path('pilot/claim_prospective_eval_v1')
    reference_bytes=(pack/'references.json').read_bytes()
    require(sha(reference_bytes)==read(pack/'protocol.json')['reference_sha256'],'Reference changed')
    reviewed=[]
    known={}
    for row in rows:
        key=row['file'][:-5]
        index=int(row['claim']['local_id'][1:])-1
        judgment,note=CLAIMS[key][index]
        claim_key=f'{key}:{row["claim"]["local_id"]}'
        reviewed.append(dict(row,claim_key=claim_key,reviewer_kind='AI',judgment=judgment,review_note=note))
        known[claim_key]=row
    require(sum(len(v) for v in CLAIMS.values())==len(rows),'Incomplete claim review')
    coverage=[]
    facts=read(pack/'references.json')['facts']
    negative={'zero_tests','edit_failed','verification_failed','runtime_exception'}
    for fact in facts:
        if fact['eligibility']=='open_stage_not_sealed':
            status,claim_keys,note='not_generated_open_stage',[],'No artificial EOF seal; no model job exists for this fact.'
        else:
            status,keys,note=FACTS[fact['session']][fact['step_id']]
            claim_keys=keys.split(',') if keys else []
            for key in claim_keys:
                claim=known[key]
                require(claim['session']==fact['session'] and claim['job_id']==fact['sealed_job_id'],'Cross-stage coverage citation')
                # Provenance is necessary, never sufficient for coverage.
                require(any(ref['evidence_id']==fact['record_id'] for ref in claim['claim']['evidence_refs']),'Coverage has no source-record citation')
        coverage.append(dict(fact_id=fact['fact_id'],session=fact['session'],category=fact['category'],importance=fact['importance'],
            eligibility=fact['eligibility'],coverage=status,claim_keys=claim_keys,review_note=note,reviewer_kind='AI',
            coarse_negative_outcome_retained=(status in ('full','partial') if fact['category'] in negative and fact['eligibility']=='sealed_stage' else None)))
    require(sum(len(v) for v in FACTS.values())==sum(f['eligibility']=='sealed_stage' for f in facts),'Incomplete fact review')
    sealed=[r for r in coverage if r['eligibility']=='sealed_stage']
    key=[r for r in sealed if r['importance']=='key_outcome']
    negative_rows=[r for r in sealed if r['coarse_negative_outcome_retained'] is not None]
    result=dict(reviewer_kind='AI',blinded=False,claim_judgments=dict(Counter(r['judgment'] for r in reviewed)),
        sealed_fact_coverage=dict(Counter(r['coverage'] for r in sealed)),
        sealed_key_fact_coverage=dict(Counter(r['coverage'] for r in key)),
        all_key_facts=sum(r['importance']=='key_outcome' for r in coverage),
        open_key_facts=sum(r['importance']=='key_outcome' and r['eligibility']!='sealed_stage' for r in coverage),
        sealed_negative_outcomes=len(negative_rows),coarse_negative_outcomes_retained=sum(r['coarse_negative_outcome_retained'] for r in negative_rows),
        scope='Single AI review, not human accuracy. Full/partial/missing are explicit semantic judgments. Correct subclauses may cover a fact while another clause makes that claim partial. Coarse failure retention does not certify its cause, condition or diagnostic details.')
    save(out/'claim_ai_review.json',dict(summary=result,rows=reviewed))
    save(out/'fact_coverage.json',dict(summary=result,rows=coverage))
    save(out/'semantic_issues.json',[r for r in reviewed if r['judgment']!='supported'])
    summary=read(out/'summary.json')
    summary['semantic_review']=result
    save(out/'summary.json',summary)
    require((pack/'references.json').read_bytes()==reference_bytes,'Frozen reference was modified')
    print(result)


if __name__=='__main__':
    main()
