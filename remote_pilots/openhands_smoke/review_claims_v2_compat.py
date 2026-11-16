"""Materialize explicit AI review decisions for the fixed six returned jobs.

This is a recorded case review, NOT an automatic semantic correctness classifier.
Raw results remain unchanged. No network calls and no evidence execution.
"""
import json
from collections import Counter
from pathlib import Path
from statistics import median

from remote_pilots.openhands_smoke import claims_v2 as v


# In frozen compat request order; each decision was made by reading the full stage.
# verdict, explanation, proposed evidence IDs (suggestions only; no repairs accepted).
DECISIONS = [
    [
        ('supported', 'Three tool receipts report eight task items; low-information metadata.', []),
        ('supported', 'e0008 explicitly reports the current working directory; not inferred from ls target.', []),
        ('supported', 'e0008 reports exit code zero for ls; this does not imply task success.', []),
    ],
    [
        ('supported', 'The tool receipts report eight items.', []),
        ('supported', 'Task-list entry is a plan, not completed work; wording preserves this distinction.', []),
        ('supported', 'Planned environment/test activity, not evidence of execution.', []),
        ('supported', 'Planned ops.linspace/TensorFlow exploration; objects should be identifiers, not a whole task sentence.', []),
        ('supported', 'Planned reproduction, not a successful reproduction.', []),
        ('supported', 'Planned final review, not evidence of completion.', []),
    ],
    [
        ('incomplete_citation', 'e0001 only says eight items. The debugging/Keras subject requires task-list and repository context.', ['e0001', 'e0003', 'e0008']),
        ('supported', 'e0012 lists backend linspace definitions; e0040 shows JAX. e0016 is irrelevant footer and should be removed.', ['e0012', 'e0040']),
        ('wrong_citation', 'e0031 is einsum/subtract/matmul, not linspace. e0037 shows endpoint branches; describe source branches, not verified runtime correctness.', ['e0036', 'e0037']),
        ('wrong_citation', 'e0041 contains logarithmic/logical functions. Comparing parameter signatures requires TF e0036 and JAX e0040; equal signatures do not prove equal semantics.', ['e0036', 'e0040']),
        ('condition_lost', 'e0043 accepts axis=0 and rejects axis!=0. Saying no axis parameter loses the crucial condition.', ['e0043']),
        ('insufficient_evidence', 'Backend-specific definitions do not establish a generic cross-backend dispatcher. e0015 offers a lead, but do not assert unrestricted cross-backend compatibility.', ['e0015']),
    ],
    [
        ('supported', 'Listing gives CHANGELOG.md size 313645; low relevance to code diagnosis.', []),
        ('supported', 'Listing gives package.json size 10727; low relevance.', []),
        ('supported', 'Listing gives package-lock.json size 2486098; low relevance.', []),
        ('object_type_error', 'The d prefix identifies node_modules as a directory. 53248 is its listed metadata size, not recursive installed dependency size.', ['e0003']),
        ('supported', 'The displayed Tooltip.d.ts declares TooltipProps; preserve its packages/mui-material/src/Tooltip path.', ['e0025', 'e0026']),
        ('scope_needs_qualification', 'The evidence establishes a declared componentsProps interface, not observed runtime behavior. Rephrase as a source/type declaration with concrete types.', ['e0026']),
    ],
    [
        ('supported', 'Old/new action and successful edit receipt establish the exact type change; no validation result shown.', []),
        ('supported', 'Edit receipt confirms modification; subsumed by c1.', []),
        ('supported', 'tooltip declaration remains identical; secondary unchanged context.', []),
        ('supported', 'arrow declaration remains identical; secondary unchanged context.', []),
        ('supported', 'Same popper type change as c1; duplicate fact.', []),
    ],
    [
        ('supported', 'The replaced PopperProps component is now Partial<PopperProps>; intersection with Overrides remains unchanged.', []),
        ('supported', 'Successful edit receipt; subsumed by c1.', []),
        ('incomplete_citation', 'e0003 alone is an unlabeled version string. Add e0002 yarn -v to identify its producer; stage context supports the statement.', ['e0002', 'e0003']),
    ],
]
OMISSIONS = [
    ['e0003: planned ops.linspace/TensorFlow investigation, reproduction and verification are absent; eight-item count alone does not preserve intent.'],
    ['e0003: omitted planned analysis, implementation and post-fix reproduction/tests. Merge plan items rather than spend five claims enumerating a partial plan.'],
    ['e0036-e0037: TF dtype conversion, num<0 guard and endpoint-specific handling are more specific than broad backend summaries.',
     'e0050-e0051: inspected convert_to_tensor logic is not represented. These are candidate useful facts, not an exhaustive recall gold set.'],
    ['e0026 vs e0029: componentsProps.popper uses PopperProps & Overrides whereas top-level PopperProps uses Partial<PopperProps>; omission identifiable from this stage alone.',
     'e0004-e0005: npm 8.18.0 and node v18.8.0 are omitted while generic file sizes occupy four claims.'],
    ['No observed test result in this one-step stage; preserve that limitation without declaring the task untested globally.'],
    ['No observed post-edit test in these two steps; yarn version is not verification of the change.'],
]


def main():
    root = Path('swe_claims_v2_compat')
    rows = [json.loads(x) for x in (v.ROOT / 'compat_requests.jsonl').read_text(encoding='utf-8').splitlines()]
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    assert config['input_sha256'] == v.digest(v.encoded(rows))
    assert config['contract_sha256'] == v.digest(v.encoded(v.SCHEMA))
    assert config['instruction_sha256'] == v.digest(v.INSTRUCTION.encode())
    runner = Path(v.__file__).read_bytes()
    assert config['runner_sha256'] in {v.digest(runner), v.digest(runner.replace(b'\r\n', b'\n'))}
    source = {r['request']['job_id']: r['request'] for r in map(json.loads,
        Path('pilot/swe_ledger_sensitivity_v1/requests.jsonl').read_text(encoding='utf-8').splitlines())}
    manifest = json.loads((v.ROOT / 'protocol.json').read_text(encoding='utf-8'))
    jobs = {j['job_id']: j for j in manifest['jobs']}
    returned_review = json.loads((root / 'semantic_review.json').read_text(encoding='utf-8'))
    assert [r['job_id'] for r in rows] == manifest['compat_job_ids']
    assert len(rows) == len(DECISIONS) == len(returned_review) == 6
    audited, counts, seconds, usage = [], Counter(), [], Counter()
    for i, (row, decisions) in enumerate(zip(rows, DECISIONS)):
        assert v.build(source[row['job_id']]) == row
        path = root / (row['job_id'] + '.json')
        result = json.loads(path.read_text(encoding='utf-8'))
        assert result['job_id'] == row['job_id'] and result['status'] == 'valid'
        reply = result['raw_response']
        assert reply['choices'][0]['finish_reason'] == 'stop'
        claims = v.validate(json.loads(reply['choices'][0]['message']['content']), row)
        assert claims == result['claims'] == returned_review[i]['claims']
        assert returned_review[i]['evidence'] == row['request']['evidence']
        assert len(claims) == len(decisions)
        seconds.append(result['seconds'])
        usage.update({k: value for k, value in reply.get('usage', {}).items() if type(value) == int})
        reviewed = []
        for claim, (verdict, note, proposed_ids) in zip(claims, decisions):
            counts[verdict] += 1
            assert all(eid in row['evidence_map'] for eid in proposed_ids)
            reviewed.append(dict(**claim, reviewed=True, reviewer_type='AI', verdict=verdict,
                supported_by_cited_evidence=verdict == 'supported', review_note=note,
                suggested_evidence_ids=proposed_ids, suggestions_applied=False))
        audited.append(dict(job_id=row['job_id'], run_id=row['run_id'], arm=jobs[row['job_id']]['arm'],
            stage=row['request']['stage'], raw_result_sha256=v.digest(path.read_bytes()),
            reviewed=True, reviewer_type='AI', claims=reviewed, evidence=row['request']['evidence'],
            missing_facts=OMISSIONS[i], complete_recall_gold=False))
    out = Path('pilot/swe_claims_v2_compat_audit')
    out.mkdir(exist_ok=False)
    v.save(out / 'ai_review.json', audited)
    summary = dict(jobs=6, claims=sum(counts.values()), verdicts=dict(counts),
        schema_and_provenance_valid=6, reviewer_type='AI', human_accuracy_validation=False,
        input_and_source_mapping_verified=True, runner_verified=True,
        claims_at_cap_jobs=sum(len(r['claims']) == 6 for r in audited),
        seconds=dict(total=sum(seconds), median=median(seconds), maximum=max(seconds)),
        server_reported_usage=dict(usage),
        decision='Hold full run: revise citation discipline, conditional wording and fact prioritization first.',
        limitations=['Two selected development tasks; stages overlap and differ across arms.',
            'No representative precision or recall estimate; no stage ranking or online alarm score.',
            'Suggestions are review annotations, not corrected model output.',
            'No future trajectory steps or final outcomes used for semantic decisions.'])
    v.save(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
