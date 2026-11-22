"""Persist fixed AI citation judgments for the frozen 42-claim v4 return."""
from collections import Counter
from pathlib import Path

from scripts.run_stage_claim_pipeline import read, save
from scripts.audit_neural_ledger_v3_return import require, content
from segmenter.ledger import sha, encoded


def main():
    out=Path('out/neural_ledger_v4_return_audit')
    rows=read(out/'claims.json')
    require(sha(encoded(rows))=='53389243b912c7692bd49142d85ee0beb204436a16a68a4070bf4dad7f7b73fd',
            'Output snapshot changed; perform a fresh semantic review')
    # These are AI judgments read against actual citations and paired actions,
    # not regex scores, human gold or exhaustive stage-fact recall annotation.
    notes={
        ('swe','00000001.json'):[
            ('supported','Plan actions show reading in progress and reproduction todo; no completion is inferred.'),
            ('supported','The rings.py excerpt includes def as_expr and surrounding implementation/comments.'),
            ('supported','The source includes def ring and related constructors.'),
            ('supported','The agent view action requests exactly the 1-200 range; treated as an action description, not proof that viewing completed.'),
            ('supported','The first page shows PolyRing initialization and methods in rings.py.'),
            ('unsupported','The second page is a source listing around lines 378-599, not an as_expr edit receipt. No before/after evidence establishes no change.'),
            ('supported','The precise supplied-symbol expression is printed in the observation.'),
            ('supported','The polyutils.py excerpt shows its docstring and generator-ordering helpers.'),
            ('partial','The observation displays polyutils.py, but calling it a PolyElement.as_expr method receipt invents an object/event relationship.'),
        ],
        ('swe','00000002.json'):[
            ('supported','The edit receipt shows the nested if symbols / length check and the default-symbol else branch; it does not establish runtime correctness.'),
        ],
        ('swe','00000003.json'):[
            ('supported','The cited output prints expressions for supplied and default symbols.'),
            ('supported','The cited task list marks reproduction and implementation done; this is correctly an agent assertion.'),
            ('supported','The result explicitly says pytest command not found and exit code 127.'),
        ],
        ('swe','00000004.json'):[
            ('supported','The grep result contains multiple as_expr references; this is read-only activity.'),
            ('supported','The listing and paired ls action identify .git contents.'),
            ('supported','Git status shows M sympy/polys/rings.py.'),
            ('supported','The has output is interpreted using its paired test -f .gitignore command; not from that word in isolation.'),
            ('supported','The two git config outputs give the named identity.'),
        ],
        ('swe','00000005.json'):[
            ('partial','The observation is a commit receipt/message describing intended behavior. It does not establish that the code actually respects supplied symbols; source/wording should preserve agent intent.'),
        ],
        ('swe','00000006.json'):[
            ('partial','The task-list receipt only acknowledges an update. Other cited observations include echoed agent narratives and a limited reproduction; they do not provide the cited plan statuses or independently verify the narrative effects.'),
            ('supported','The quoted agent narrative asserts preserved defaults and ValueError behavior. Supported as agent_assertion only, not as independent verification or proof of correctness.'),
        ],
        ('terminal','00000001.json'):[
            ('supported','Find, ls and head outputs show the path, header and five sample rows. Ten is the head request limit, not an observed row count.'),
            ('supported','Python 3.12.3 and pip3 command not found are explicit.'),
            ('partial','The paired command uses which apt-get && which apk && which yum. Failure at apk can skip yum, so absence of yum is not established.'),
        ],
        ('terminal','00000002.json'):[
            ('partial','The result installs python3.12-venv, pip/setuptools wheels. It does not show installation of the Python 3.12.3 interpreter itself; package identity and scope are conflated.'),
            ('partial','The cited output ends mid-download. Download progress is supported, installation success is not. The completion guard did not catch this wording.'),
            ('supported','The other cited observation includes Successfully installed and the package list.'),
            ('supported','The exact interpreter path and version are printed.'),
        ],
        ('terminal','00000003.json'):[
            ('supported','The tool acknowledges creation of the named script; this does not validate its content.'),
        ],
        ('terminal','00000004.json'):[
            ('supported','The conversion tool reports a created Parquet file, size 2276 bytes and conversion completion.'),
            ('supported','The ls output lists both file paths, sizes and times.'),
        ],
        ('terminal','00000005.json'):[
            ('supported','A file-created receipt supports creation of verify_parquet.py.'),
        ],
        ('terminal','00000006.json'):[
            ('supported','The output prints the Parquet shape, columns, dtypes and rows; this does not independently compare all CSV semantics.'),
        ],
        ('live_smoke','00000001.json'):[
            ('supported','The output explicitly reports Ran 0 tests and NO TESTS RAN.'),
            ('supported','The source listing contains def merge_ranges.'),
        ],
        ('live_smoke','00000002.json'):[
            ('supported','The tool acknowledges the edit and displays the modified snippet; no successful fix is inferred.'),
        ],
        ('live_smoke','00000003.json'):[
            ('supported','The traceback names ranges.py and IndentationError.'),
            ('supported','Both replacement observations report multiple occurrences and no replacement.'),
        ],
        ('live_smoke','00000004.json'):[
            ('supported','The tool acknowledges a ranges.py edit.'),
            ('partial','The cited action requests replacing < with <=. Prefer explicitly requested/proposed wording; this reference alone is not an edit completion receipt.'),
        ],
        ('live_smoke','00000005.json'):[
            ('supported','The traceback explicitly reports IndentationError.'),
        ],
        ('live_smoke','00000006.json'):[
            ('supported','The observations report multiple matches, no replacement, and refusal to overwrite an existing file.'),
        ],
    }
    reviewed=[]
    require(sum(len(v) for v in notes.values())==len(rows),'Review coverage mismatch')
    for row in rows:
        index=int(row['claim']['local_id'][1:])-1
        judgment,note=notes[(row['session'],row['job'])][index]
        reviewed.append(dict(row,reviewer='AI',reviewed=True,judgment=judgment,review_note=note,
            claim_sha256=sha(encoded(content(row['claim'])))))
    result=dict(reviewer_kind='AI',blinded=False,
        scope='All 42 published claims. Cited excerpts with paired actions for interpretation. No human accuracy or exhaustive fact-recall claim. Agent assertions are not independent verification.',
        snapshot_sha256=sha(encoded(rows)),judgments=dict(Counter(r['judgment'] for r in reviewed)),rows=reviewed)
    save(out/'ai_review.json',result)
    save(out/'semantic_issues.json',[r for r in reviewed if r['judgment']!='supported'])
    summary=read(out/'summary.json')
    summary['semantic_review']=dict(reviewer_kind='AI',blinded=False,claims=len(reviewed),judgments=result['judgments'],scope=result['scope'])
    save(out/'summary.json',summary)
    print(summary['semantic_review'])


if __name__=='__main__':
    main()
