"""Persist explicitly inspected, pre-output AI reference points; not an automatic judge."""
from remote_pilots.openhands_smoke import seal_claim_diagnostic as pilot
from remote_pilots.openhands_smoke import claims_v3 as io

CASES={
0:[('agent_assertion','The agent plans reading, reproduction, exploration, a minimal fix, verification and final review; reading is marked in progress and the remaining phases todo.',[(3,'action')])],
3:[('tool_observation','Before the commit, git status shows autodetector.py modified and repro_custom_fk.py untracked.',[(27,'action'),(27,'observation')]),
   ('tool_observation','Git reports commit 405e4d218d with one insertion and one deletion in one file; the command stages django/db/migrations/autodetector.py.',[(30,'action'),(30,'observation')]),
   ('agent_assertion','The agent marks verification done and reports autodetector tests passing, but also says the custom reproduction could not run because of app registry constraints.',[(31,'action')])],
6:[('tool_observation','The edit adds record_unapplied(migration.app_label, migration.name) within the replacement-migration branch, after recording the replaced migrations as unapplied.',[(22,'action'),(22,'observation')]),
   ('agent_assertion','The plan marks implementation done, testing in progress, and the minimal reproduction still todo; this stage contains no test outcome.',[(23,'action')])],
9:[('agent_assertion','The plan schedules reproduction, code exploration, a minimal fix, verification and final review; requirement analysis is in progress and later phases remain todo.',[(3,'action')])],
12:[('agent_assertion','The agent plans to locate _Unsigned conversions, reproduce signed/unsigned byte handling, implement _Unsigned=False conversion and verify it; reading is in progress and these later actions remain todo.',[(2,'action')])],
15:[('tool_observation','Git reports commit fc4cf4ac affecting two files with three insertions and two deletions; the staged paths are xarray/core/common.py and xarray/core/variable.py.',[(36,'action'),(36,'observation')]),
    ('agent_assertion','The commit message and task note claim chunk access avoids materializing data and remains lazy; this stage contains no runtime verification of that behavior.',[(36,'action'),(37,'action')])],
18:[('tool_observation','The created verification script is run successfully and prints DependsOn=testKinesisStreamRoleLogicalId for the logical-id case and DependsOn=[] for the ARN case.',[(25,'action'),(25,'observation'),(26,'action'),(26,'observation')]),
    ('agent_assertion','The task notes report full-suite environment-related failures and a switch to targeted verification; these notes do not establish that the full suite passed or independently prove the failure cause.',[(27,'action')]),
    ('tool_observation','After the removal command for the temporary script, git status reports a clean working tree.',[(28,'action'),(28,'observation')])],
21:[('agent_assertion','The agent plans to explore link parsing logic, reproduce, implement a minimal fix and verify it; the latest plan marks understanding the issue in progress and all later phases todo.',[(3,'action')])]
}


def main():
    root=pilot.ROOT; _,rows,_=pilot.load(root)
    ordered=list(rows.values()); reviews=io.read(root/'review.json'); byid={r['job_id']:r for r in reviews}
    for index,points in CASES.items():
        row=ordered[index]; review=byid[row['job_id']]
        if review['reviewed']: raise ValueError('refusing to replace existing review')
        result=[]
        for n,(basis,text,locations) in enumerate(points,1):
            ids=[eid for eid,e in row['evidence_map'].items() if (e['source_step_id'],e['field']) in locations]
            if not ids: raise ValueError('missing evidence')
            result.append(dict(point_id=f'p{n}',text=text,source=basis,evidence_ids=ids,priority='key'))
        review.update(reviewed=True,reviewer_type='AI',input_complete=not pilot.input_issues(row),
            required_points=result,excluded_inferences=[
                'Do not treat task-list status, notes or commit messages as independently verified results.',
                'Do not infer full task success or full-suite success from edits, commits or targeted checks.'],
            reviewed_before_outputs=True,exhaustive=False,
            note='AI inspected all supplied stage evidence and recovered public plan arguments. Completeness means no known blank/missing supplied field; not a guarantee that native observations were never truncated. Reference wording is not a unique gold summary.')
    io.save(root/'review.json',reviews)
    io.save(root/'review_status.json',dict(reviewed=sum(r['reviewed'] for r in reviews),total=len(reviews),
        selection='first eight short stages; descriptive diagnostic, not representative accuracy',
        remaining='16 stages pending full semantic review; no quality label inferred'))
    print(dict(reviewed=sum(r['reviewed'] for r in reviews),total=len(reviews)))


if __name__=='__main__': main()
