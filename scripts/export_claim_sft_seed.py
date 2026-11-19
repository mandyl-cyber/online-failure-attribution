"""Export explicitly inspected short-stage AI references as a small SFT seed."""
import json
from pathlib import Path
from segmenter import stage_claim_summary as base
from segmenter import stage_claim_summary_v5 as core

# Each tuple identifies exact source steps and the fields supporting the reference.
REFERENCES={
"c17f6903":[
("tool_observation","The displayed diff adds recursive mutually-exclusive-group discovery and uses required_dests when building call_command parse_args.",[(27,"observation")]),
("tool_observation","Git reports commit 9a0899518c changing one file with 16 insertions and 1 deletion.",[(28,"observation")]),
("agent_assertion","The commit message describes including kwargs for required mutually exclusive groups.",[(28,"action")])],
"74ac328a":[
("tool_observation","The pytest command fails because pytest is not found with exit code 127; the first runtests.py command rejects -q and exits with code 2.",[(23,"observation"),(24,"observation")]),
("tool_observation","The subsequent runtests.py invocations report 126 and 138 tests respectively, both with OK and exit code 0.",[(25,"action"),(25,"observation"),(26,"action"),(26,"observation")])],
"c01fb31e":[
("tool_observation","Git reports commit 672e66ba23 changing one file with 4 insertions and 33 deletions.",[(19,"observation")]),
("agent_assertion","The commit describes removing undocumented PSRP extras and version gates; the plan reports tests could not run because pytest was unavailable and marks final review done.",[(19,"action"),(20,"action")])],
"7879600a":[
("tool_observation","Git reports commit a024894cfd for modified basic.py with 4 insertions and 7 deletions.",[(52,"observation"),(53,"observation")]),
("agent_assertion","The commit describes a selinuxenabled CLI fallback; task notes report constrained verification and mark the listed tasks done.",[(53,"action"),(54,"action")]),
("tool_observation","The later units.modules.test_apt invocation runs 4 tests and reports 4 NameErrors for undefined expand_pkgspec_from_fnmatches, then exits with code 1.",[(55,"observation")])],
"be9c3455":[
("tool_observation","The perm_groups.py edit receipt replaces in-loop deletion with collecting indices and consistently filtering rep_blocks, num_blocks and blocks.",[(11,"observation")]),
("agent_assertion","The plan marks reading done and repository testing in_progress while later listed phases remain todo.",[(12,"action")]),
("tool_observation","The script reports DihedralGroup(18) order 36 and degree 18, and Sylow-2 subgroup order 4 for DihedralGroup(18) and DihedralGroup(50), then exits with code 0.",[(13,"observation")])],
"9cefea22":[
("tool_observation","The probe prints is_real=False for positive and negative infinity and I, and is_real=None for nan, then exits with code 0.",[(24,"observation")]),
("tool_observation","The point.py edit receipt adds an is_infinite check to the numeric non-real coordinate rejection condition.",[(25,"observation")])],
"eb076a47":[
("tool_observation","The three targeted pytest commands for test_svg, test_polygon and test_do_poly_distance each show a passing dot at 100% and exit code 0.",[(26,"action"),(26,"observation"),(27,"action"),(27,"observation"),(28,"action"),(28,"observation")]),
("tool_observation","The sympy/geometry/tests invocation shows passing dots at 100%, a dimension-change UserWarning in test_point, and exit code 0.",[(29,"action"),(29,"observation")])],
"5ae84e44":[
("tool_observation","The probe prints Point2D(1,2) for both evaluate(False) examples, catches the expected imaginary-coordinate ValueError, prints points with positive and negative infinity, and exits with code 0.",[(30,"observation")]),
("tool_observation","Git reports commit 2dac3a5d31 changing point.py with 2 insertions and 1 deletion.",[(31,"observation")]),
("agent_assertion","The final plan marks all phases done and reports geometry tests passing.",[(32,"action")])],
"f6afc2ce":[
("tool_observation","After removing tmp_repro_monitor.js, git status still lists monitorStack.js modified and Dockerfile untracked; git then reports commit 89c358b8 changing one file with 5 insertions and 5 deletions.",[(26,"observation"),(27,"observation")]),
("agent_assertion","The commit message describes null guards for stack monitoring; the plan marks verification done and final review in_progress.",[(27,"action"),(28,"action")])],
"1d57f599":[
("tool_observation","The mergeIamTemplates.js edit receipt adds a warning for a truthy iamRoleStatements value that is not an array; the existing array branch still concatenates statements.",[(22,"observation")]),
("tool_observation","The search finds iamRoleStatements references in mergeIamTemplates.test.js at lines 107 and 125.",[(24,"observation")]),
("agent_assertion","The plan marks analysis and implementation done and verification in_progress.",[(25,"action")])],
"69a92348":[
("tool_observation","The verification script prints DependsOn=testKinesisStreamRoleLogicalId for the logical-ID case and DependsOn=[] for the ARN case, then exits with code 0.",[(26,"observation")]),
("agent_assertion","The plan reports full-suite failures as environment-related and reports switching to targeted verification; it marks the listed phases done.",[(27,"action")]),
("tool_observation","After the temporary script removal, git status reports a clean working tree.",[(28,"observation")])],
"17e5f634":[
("tool_observation","Git reports commit da8d5eb93 for the modified gettext.py with 8 insertions and 1 deletion.",[(19,"observation"),(20,"observation")]),
("agent_assertion","The commit message describes deduplicating gettext message locations while preserving order.",[(20,"action")]),
("agent_assertion","The task notes claim the intl subset was checked, attribute remaining failures to unrelated causes, and change final review from in_progress to done.",[(21,"action"),(22,"action")])],
"503b46ee":[
("tool_observation","The initial view reports no task list found.",[(1,"observation")]),
("agent_assertion","The newly created eight-phase plan changes reading to in_progress while subsequent reproduction, implementation and verification remain todo.",[(2,"action"),(3,"action")])],
"bcf0f934":[
("agent_assertion","The Serverless repair plan initially marks all eight phases todo, then moves problem reading to in_progress while later phases remain todo.",[(1,"action"),(3,"action")]),
("tool_observation","The repository listing shows lib, tests, package.json and node_modules.",[(2,"observation")])],
"42bd049e":[
("tool_observation","The tool reports creating tmp_repro_monitor.js, and the Node version probe prints v16.20.2.",[(13,"observation"),(14,"observation")]),
("tool_observation","Running the monitor reproduction prints Cannot read properties of null (reading 'endsWith'); despite that error text, the command exits with code 0.",[(15,"observation")])],
"115d4f64":[
("agent_assertion","Successive task-list updates mark repository testing done, then exploration done, then reproduction done and fix analysis in_progress. Implementation, verification and final review remain todo.",[(16,"action"),(17,"action"),(18,"action")])],
"5e0d09d6":[
("tool_observation","Three edit receipts show null guards added before eventStatus.endsWith checks for FAILED, PROGRESS and COMPLETE and before stackStatus.endsWith('ROLLBACK_COMPLETE') in monitorStack.js.",[(19,"observation"),(20,"observation"),(21,"observation")])],
"3ae14aff":[
("tool_observation","Git reports commit 55d97cc5a5 changing config.py with 96 insertions; the subsequent view displays _render_settings handling display values and origins.",[(32,"observation"),(33,"observation")]),
("agent_assertion","The commit message describes adding GALAXY_SERVERS configuration to ansible-config dump output.",[(32,"action")])],
"bee762f1":[
("agent_assertion","The _Unsigned conversion plan changes reading to in_progress while tests, reproduction, implementation and verification remain todo.",[(1,"action"),(2,"action")])],
"7f8c4f58":[
("tool_observation","Git status lists xarray/coding/variables.py modified; commit 1016f7ca records one file changed with 9 insertions and 1 deletion.",[(19,"observation"),(20,"observation")]),
("agent_assertion","The commit message describes converting unsigned integer arrays to signed for _Unsigned=false during CF decoding.",[(20,"action")])],
"712a6670":[
("tool_observation","The displayed test_backends.py source includes NetCDF group and subgroup tests.",[(21,"observation")]),
("agent_assertion","The plan marks all phases done and reports targeted verification while attributing full-suite failures to the environment.",[(22,"action")]),
("tool_observation","The decoding script prints int8 [-1,0,1,127,-128] for the unsigned input with _Unsigned=false and uint8 [255,0,1,127,128] for the signed input with _Unsigned=true, then exits with code 0.",[(23,"observation")])],
"c7e19986":[
("tool_observation","Git reports commit fc4cf4ac changing common.py and variable.py with 3 insertions and 2 deletions.",[(36,"observation")]),
("agent_assertion","The commit message and final task note claim chunk access remains lazy and avoids materializing data.",[(36,"action"),(37,"action")])],
"0ff07ecf":[
("tool_observation","The initial task-list view reports no task list found.",[(1,"observation")]),
("agent_assertion","The agent creates eight planned phases, then marks requirements analysis in_progress while later phases remain todo.",[(2,"action"),(3,"action")])],
"36c62761":[
("tool_observation","The setuponly.py edit receipt adds bytes/bytearray-specific saferepr formatting for cached fixture parameters.",[(31,"observation")]),
("tool_observation","After removal of temporary outputs and test_bytes_param.py, git status lists only setuponly.py modified; commit 6cada262f reports 7 insertions and 1 deletion and a clean working tree.",[(33,"observation"),(34,"observation")]),
("agent_assertion","The commit message describes avoiding BytesWarning under Python -bb with --setup-show.",[(34,"action")])],
"c8c84f21":[
("tool_observation","The executor.py edit receipt adds record_unapplied for the replacement migration itself after recording its replaced migrations as unapplied.",[(22,"observation")]),
("agent_assertion","The plan marks implementation done and describes the additional replacement-migration recorder call.",[(23,"action")])],
"50c3ed3c":[
("tool_observation","Git reports commit acbf03cbea for the modified django/utils/html.py, with 3 insertions and 2 deletions.",[(31,"observation"),(33,"observation")]),
("agent_assertion","The commit message describes changing urlize trailing punctuation handling for HTML-escaped fragments.",[(33,"action")])],
"0f16f1a2":[
("tool_observation","The tool reports creating repro_custom_fk.py.",[(8,"observation")]),
("tool_observation","The subsequent reproduction fails with AppRegistryNotReady: Models aren't loaded yet, and exits with code 1.",[(9,"observation")])],
"1a5bdc98":[
("tool_observation","The displayed autodetector.py initially deletes deconstruction[2]['to']; the edit receipt replaces that deletion with pop('to', None).",[(20,"observation"),(21,"observation")]),
("agent_assertion","The plan marks implementation and fix analysis done, describing the intended avoidance of KeyError.",[(22,"action")])],
"962db112":[
("tool_observation","pytest -q could not run because pytest was not found; the exit code is 127.",[(6,"observation")])],
"760763b0":[
("tool_observation","The psrp.py edit receipt removes AUTH_KWARGS from the pypsrp.wsman import.",[(21,"observation")]),
("tool_observation","Git reports commit 421eef0aa5 with one insertion and one deletion in one file.",[(22,"observation")])],
"e73ff525":[
("tool_observation","pytest -q failed because pytest was not found, with exit code 127.",[(6,"observation")]),
("tool_observation","requirements.txt lists jinja2, PyYAML, cryptography, packaging and resolvelib >=0.5.3,<0.6.0, and the test directory lists units and integration subdirectories.",[(7,"observation"),(8,"observation")])],
"a822e651":[
("tool_observation","The QD FITS self-comparison raised an ambiguous-array ValueError and exited with code 1.",[(15,"observation")]),
("tool_observation","The edit receipt shows TableDataDiff now handles Q alongside P in the row-wise allclose branch.",[(16,"observation")]),
("agent_assertion","The plan marks implementation done while reproduction remains in progress and verification remains todo.",[(17,"action")]),
("tool_observation","The subsequent QD self-comparison prints Identical? True and No differences found, then exits with code 0.",[(18,"observation")])],
"c8f5a6b8":[
("tool_observation","The targeted pytest invocation failed loading conftest with ImportPathMismatchError and exit code 4.",[(19,"observation")]),
("tool_observation","A separate script prints True for PD and QJ self-comparisons and False for the differing QD files; its report identifies one differing table element and it exits with code 0.",[(20,"observation")]),
("agent_assertion","The task list marks phases complete and attributes the pytest failure to import-path issues while reporting validation through targeted scripts.",[(21,"action")])],
"b652bf09":[
("tool_observation","Git reports commit e38a73a0b changing astropy/io/fits/diff.py with one insertion and one deletion.",[(23,"observation")]),
("agent_assertion","The commit message describes including Q alongside P in VLA comparisons.",[(23,"action")]),
("tool_observation","The final git status still lists diff1.fits, diff2.fits, diffbug.fits, pvlaf.fits and qjvla.fits as untracked.",[(24,"observation")])],
"ec14c40a":[
("tool_observation","The edit receipt shows call_command collecting required mutually-exclusive-group destinations, including subparser groups, when constructing parse_args.",[(21,"observation")]),
("tool_observation","The inline command probe prints OK for shop_id=1 and --shop-id=1, prints the expected missing-group CommandError, and exits with code 0.",[(22,"observation")])],
"310be630":[
("tool_observation","The command listed the workspace and Ansible repository directories.",[(1,"observation")]),
("agent_assertion","The plan marks reading in progress and the seven subsequent phases todo.",[(2,"action")]),
("tool_observation","pytest -q could not execute because pytest was not found; the command exited with code 127.",[(3,"observation")])],
"c925001d":[
("tool_observation","Git status lists llms.py modified and Dockerfile untracked before staging.",[(61,"observation")]),
("tool_observation","Git reports commit 8717b839b changing two files with 54 insertions and 21 deletions and creating Dockerfile.",[(64,"observation")]),
("agent_assertion","The commit message describes local-cache support for BaseLLM synchronous and asynchronous generation.",[(64,"action")])],
"13ac042b":[
("agent_assertion","The plan initially lists all eight link-parsing repair phases as todo, then marks understanding the issue in progress while later phases remain todo.",[(1,"action"),(3,"action")])],
"4d46f993":[
("agent_assertion","The plan concerns Variable assignment coercion for objects with a values attribute; reading is changed to in_progress and implementation and verification remain todo.",[(1,"action"),(3,"action")])],
"86c7e0a6":[
("tool_observation","The edit receipt shows variable.py replacing unconditional values extraction with checks limited to pandas Series/DataFrame and xarray DataArray.",[(13,"observation")])],
"1953a96b":[
("agent_assertion","The task list first marks reproduction in progress, then marks it done and fix analysis in progress.",[(14,"action"),(16,"action")]),
("tool_observation","The reproduction prints object-dtype arrays preserving set() and HasValues instances for loc and direct assignment, then exits with code 0.",[(15,"observation")])],
}


def main():
    root=Path("claim_sft_teacher_repaired_v1")
    out=Path("pilot/claim_sft_seed_v1")
    out.mkdir(parents=True,exist_ok=True)
    exported={"train":[],"dev":[]}
    reviews=[]
    for line in (root/"effective_inputs.jsonl").read_text(encoding="utf-8").splitlines():
        entry=json.loads(line)
        row=entry["row"]
        prefix=row["job_id"][:8]
        if prefix not in REFERENCES:
            continue
        current=core.build(row["source_snapshot"],row["chunk_chars"])
        claims=[]
        for basis,text,locations in REFERENCES[prefix]:
            ids=[eid for eid,ref in current["evidence_map"].items()
                 if (ref["source_step_id"],ref["field"]) in locations]
            claims.append(dict(text=text,source=basis,evidence_ids=ids))
        response={"claims":claims}
        base.validate(response,current["base_row"])
        review=dict(job_id=current["job_id"],source_job_id=row["job_id"],reviewed=True,
            reviewer_type="AI",reference_scope="Supplied stage evidence",
            final_response=response,group=entry["metadata"]["group"],split=entry["metadata"]["split"])
        reviews.append(review)
        exported[review["split"]].append(dict(job_id=current["job_id"],group=review["group"],
            parent_stage_key=base.digest(base.encoded(dict(run_id=row["run_id"],
                stage=row["source_snapshot"]["stage"]))),
            reviewer_type="AI",prompt=[dict(role="system",content=core.INSTRUCTION),
                dict(role="user",content=base.encoded(current["request"]).decode("utf-8"))],
            completion=[dict(role="assistant",content=base.encoded(response).decode("utf-8"))]))
    assert len(reviews)==len(REFERENCES)
    assert not ({r["group"] for r in exported["train"]}&{r["group"] for r in exported["dev"]})
    for split,records in exported.items():
        (out/f"{split}.jsonl").write_bytes(b"".join(base.encoded(r)+b"\n" for r in records))
    (out/"ai_review.json").write_bytes(base.encoded(reviews)+b"\n")
    summary=dict(counts={k:len(v) for k,v in exported.items()},reviewer_type="AI",
                 scope="Short-stage seed, selected after inspecting evidence; development pipeline only",
                 pending_parent_review=78-len(reviews),training_started=False)
    (out/"summary.json").write_bytes(base.encoded(summary)+b"\n")
    print(summary)


if __name__=="__main__":
    main()
