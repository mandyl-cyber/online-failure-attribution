"""Record a conservative AI review of the frozen paired claim return.

This is a targeted, non-exhaustive evidence review. It never converts failed
responses into valid claims and never calls the model again.
"""
import json
from pathlib import Path

from segmenter import stage_claim_summary as core


ROOT = Path('adjacent_claim_pair_v1')
OUT = Path('pilot/adjacent_claim_pair_return_v1_audit')


REVIEWS = {
    'django__django-12406': dict(
        coverage={'original':['full','partial'],'adjacent':['full','partial']},
        verdict='original_better',
        note='Both retain the 406-test result. The adjacent summary says it "Tested specific sections" of a test file, but steps 41–43 only used sed to read code. The original gives low-value command summaries and one action-only assertion that a command was executed.'),
    'django__django-15382': dict(
        coverage={'original':['full','full'],'adjacent':['missed','full']},
        verdict='original_better',
        note='The adjacent summary drops the preceding Ctrl+C/exit-130 fact while listing later search commands. The original preserves it.'),
    'django__django-15563': dict(
        coverage={'original':['full','full'],'adjacent':['partial','full']},
        verdict='mixed',
        note='Both preserve the missing-pytest outcome. The adjacent summary compresses the specific multi-table-inheritance issue into generic task-list items, while using one rather than two requests.'),
    'django__django-16032': dict(
        coverage={'original':['not_scored']*3,'adjacent':['not_scored']*3},
        verdict='excluded',
        note='The paired group was excluded before generation because at least one request exceeded the 8192-token input-plus-output budget.'),
    'django__django-16631': dict(
        coverage={'original':['not_scored']*2,'adjacent':['not_scored']*2},
        verdict='invalid_pair',
        note='Both stage-45 responses fail source/provenance validation: they state that auth_tests passed while citing a task-list action rather than a tool result. The original stage-46 commit response is valid, but the pair is incomplete.'),
    'matplotlib__matplotlib-26342': dict(
        coverage={'original':['full','full','partial'],'adjacent':['full','full','full']},
        verdict='adjacent_better',
        note='Both preserve the edit receipt and NotImplementedError. The adjacent summary also identifies that the observed set_paths method is inherited from Collection, making the unresolved failure easier to understand.'),
    'pydata__xarray-7229': dict(
        coverage={'original':['not_scored']*2,'adjacent':['not_scored']*2},
        verdict='invalid_pair',
        note='The original test-stage response ends with an extra quote after a JSON object and is invalid. The adjacent response is valid and reports the one observed test success, but a complete paired semantic comparison is unavailable.'),
    'microsoft__vscode-160342': dict(
        coverage={'original':['full','full'],'adjacent':['full','partial']},
        verdict='mixed',
        note='Both preserve the missing-loader test failure and subsequent source view. The adjacent summary spends more claims on environment metadata and gives less useful source detail.'),
    'prettier__prettier-6604': dict(
        coverage={'original':['partial','full'],'adjacent':['partial','partial']},
        verdict='original_better',
        note='Both underdescribe the missing-parentheses examples. The adjacent summary misreads sed -n 1,200p after grep as the first 200 source-file lines, and labels the observed search outcome as an agent assertion.'),
}


def main():
    mechanical = core.strict_json((OUT/'mechanical_report.json').read_text(encoding='utf-8'))
    references = {r['task']:r for r in core.strict_json((ROOT/'reference_points.json').read_text(encoding='utf-8'))}
    if set(REVIEWS) != set(references):
        raise ValueError('Review/task mismatch')
    entries = []
    for group in mechanical['groups']:
        task = group['task']
        decision = REVIEWS[task]
        npoints = len(references[task]['points'])
        if any(len(decision['coverage'][arm]) != npoints for arm in ('original','adjacent')):
            raise ValueError('Reference point count mismatch: '+task)
        complete = group['eligible'] and all(
            job['status'] in ('ready','ready_empty')
            for arm in ('original','adjacent') for job in group['arms'][arm])
        if complete != (decision['verdict'] not in ('excluded','invalid_pair')):
            raise ValueError('Completeness/review mismatch: '+task)
        entries.append(dict(group_id=group['group_id'],task=task,complete_pair=complete,
                            reviewer_type='AI',reviewed=True,
                            reference_points_exhaustive=False,
                            point_coverage={arm:{f'p{i}':s for i,s in enumerate(decision['coverage'][arm],1)}
                                            for arm in ('original','adjacent')},
                            verdict=decision['verdict'],note=decision['note']))
    totals = {arm:{key:0 for key in ('full','partial','missed')} for arm in ('original','adjacent')}
    for entry in entries:
        if entry['complete_pair']:
            for arm in totals:
                for label in entry['point_coverage'][arm].values():
                    totals[arm][label] += 1
    report = dict(version=1,reviewer_type='AI',
                  scope='Selective pre-output reference points and targeted claim errors; not human gold or exhaustive claim-level accuracy.',
                  complete_pairs=sum(e['complete_pair'] for e in entries),
                  key_point_coverage_complete_pairs=totals,
                  qualitative_verdict_counts={v:sum(e['verdict']==v for e in entries)
                                              for v in sorted({e['verdict'] for e in entries})},
                  groups=entries,
                  decision='Do not replace the online default boundary policy based on this pilot.')
    (OUT/'ai_review.json').write_bytes(core.encoded(report)+b'\n')
    print(json.dumps({k:report[k] for k in ('complete_pairs','key_point_coverage_complete_pairs',
                                           'qualitative_verdict_counts')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
