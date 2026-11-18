"""Freeze conservative AI reference points before paired claim generation."""
import hashlib
import json
from pathlib import Path

from segmenter import stage_claim_summary as core


ROOT = Path('pilot/adjacent_claim_pair_v1')

# These are deliberately limited, observable points, not exhaustive gold claims.
# Each location is (normalized step, action|observation).
POINTS = {
    'django__django-12406': [
        ('The targeted Django test run reports 406 tests, OK, 14 skipped, and exit 0.', [(40,'action'),(40,'observation')]),
        ('The subsequent commands display portions of modelchoicefield tests; no edit or new test run is shown in this window.', [(41,'action'),(42,'action'),(43,'action')])],
    'django__django-15382': [
        ('The preceding command was interrupted with Ctrl+C and reports exit 130.', [(31,'action'),(31,'observation')]),
        ('The following commands search tests for Exists/none() and inspect compiler.py; the window shows exploration rather than a completed fix.', [(32,'action'),(33,'action'),(34,'action')])],
    'django__django-15563': [
        ('The task list describes a proposed multi-table inheritance update fix; it is an agent assertion.', [(44,'action'),(44,'observation')]),
        ('The pytest command fails because pytest is not found, and python -m pytest reports no pytest module.', [(45,'action'),(45,'observation'),(46,'action'),(46,'observation')])],
    'django__django-16032': [
        ('The direct pytest command is unavailable with exit 127.', [(23,'action'),(23,'observation')]),
        ('The alternative Django runtests invocation reports Found 0 test(s) and Ran 0 tests; this is not verification of the target behavior.', [(26,'action'),(26,'observation')]),
        ('Later steps inspect query.py and the annotations model declarations.', [(28,'action'),(29,'action'),(31,'action')])],
    'django__django-16631': [
        ('The task list asserts a SECRET_KEY_FALLBACKS change; the assertion alone does not verify behavior.', [(45,'action'),(45,'observation')]),
        ('Git reports commit b71b475044 affecting two files, with 40 insertions and 11 deletions.', [(46,'action'),(46,'observation')])],
    'matplotlib__matplotlib-26342': [
        ('An edit receipt reports adding set_paths to contour.py.', [(37,'action'),(37,'observation')]),
        ('A smoke invocation of cs.set_paths fails with NotImplementedError and exit 1.', [(38,'action'),(38,'observation')]),
        ('Introspection shows ContourSet.set_paths points to Collection.set_paths in the observed environment.', [(39,'action'),(39,'observation')])],
    'pydata__xarray-7229': [
        ('The task list asserts a where/coordinate-attribute fix; it is an agent assertion.', [(44,'action'),(44,'observation')]),
        ('The specific xarray test_where command shows a passing dot and exit 0; the observed scope is that one invoked test.', [(45,'action'),(45,'observation')])],
    'microsoft__vscode-160342': [
        ('npm run test-node fails because the out/vs/loader module cannot be found.', [(38,'action'),(38,'observation')]),
        ('The following step views explorerView.ts; it does not show a successful retest.', [(39,'action'),(39,'observation')])],
    'prettier__prettier-6604': [
        ('The printed formatting examples show parentheses missing in the labeled Bug variants.', [(31,'action'),(31,'observation')]),
        ('The subsequent searches inspect TSUnionType and TSArrayType locations; this window does not show an edit or passing test.', [(32,'action'),(32,'observation'),(33,'action'),(33,'observation')])],
}


def main():
    target = ROOT/'reference_points.json'
    if target.exists():
        raise FileExistsError(target)
    groups = json.loads((ROOT/'groups.json').read_text(encoding='utf-8'))
    rows = {r['job_id']:r for r in
            [core.strict_json(line) for line in (ROOT/'requests.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]}
    references = []
    for group in groups:
        task = group['task']
        maps = [rows[j]['evidence_map'] for j in group['arms']['adjacent']]
        points = []
        for i, (statement, locations) in enumerate(POINTS[task], 1):
            ids = sorted({eid for mapping in maps for eid, ref in mapping.items()
                          if (ref['source_step_id'],ref['field']) in locations})
            if not ids:
                raise ValueError(f'Missing supporting evidence for {task} point {i}')
            points.append(dict(point_id=f'p{i}', text=statement,
                               source_locations=[dict(step=s,field=f) for s,f in locations],
                               possible_evidence_ids=ids, priority='key',
                               coverage=None, matched_claim_ids=[]))
        references.append(dict(group_id=group['group_id'],task=task,
                               reviewer_type='AI', reviewed_before_model_output=True,
                               exhaustive=False, points=points,
                               forbidden_inferences=['Do not promote a plan or commit message to verified behavior.',
                                                     'Do not treat a zero-test run, import failure or missing test runner as passing verification.',
                                                     'Do not infer a completed fix from a file view or edit receipt.']))
    target.write_bytes(core.encoded(references)+b'\n')
    protocol_path = ROOT/'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    protocol['reference_points_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    protocol['reference_provenance'] = 'AI review of evidence before model output; selective, not human gold'
    protocol_path.write_bytes(json.dumps(protocol,ensure_ascii=False,indent=2).encode('utf-8')+b'\n')
    print(json.dumps({'groups':len(references),
                      'points':sum(len(r['points']) for r in references),
                      'reference_sha256':protocol['reference_points_sha256']}))


if __name__ == '__main__':
    main()
