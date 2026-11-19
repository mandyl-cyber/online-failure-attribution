"""Persist evidence-checked parent references and their whole-citation child projections."""
import json
from pathlib import Path
from segmenter import stage_claim_summary as base
from segmenter import stage_claim_summary_v5 as core


def observed(text, *steps):
    return ("tool_observation",text,[(step,"observation") for step in steps])


def asserted(text, *steps):
    return ("agent_assertion",text,[(step,"action") for step in steps])


REFERENCES = {
"c9f80cc2": [
    observed("The initial mutually search returned exit 1. Both escaped-parenthesis call_command searches printed grep syntax errors while the pipelines exited with code 0.",6,7,8),
    observed("The mutually-exclusive search returns references in test files.",9),
    observed("The call_command search returns test references, and mutually_exclusive_group is found in showmigrations.py; the tests search prints no matching lines.",10,11,12)],
"d6007474": [
    asserted("The migration repair plan starts with all phases todo, then marks reading in_progress while testing and later phases remain todo.",1,4),
    observed("The repository directory was listed and MigrationExecutor was located at executor.py line 10.",3,5)],
"782513c9": [
    asserted("The plan describes leaving a squashed migration marked applied after unapplying and proposes recording that replacement as unapplied; testing is in_progress.",15),
    observed("The squashed-migration search prints no results, while the test-runner search finds runtests.py references.",16,17),
    observed("The first runtests.py attempt rejects -q with exit code 2.",18),
    observed("The next attempt rejects a file-path test label and requests a dotted module name; another invocation fails importing migrations.tests with one loader error.",19,20),
    observed("The migrations test directory was listed after these failed attempts.",21)],
"88cd7bdc": [
    asserted("The urlize plan changes reading from todo to in_progress, then done, and repository testing to in_progress. Its note describes duplicated entity fragments after URL punctuation trimming.",1,3,5),
    observed("The Django repository directory was listed.",4)],
"c6765676": [
    observed("The html.py edit receipt replaces punctuation slicing with k=len(middle_unescaped)-len(stripped), then slices middle from its end.",25),
    asserted("The task note reports a manual urlize reproduction and marks reproduction done while analysis remains in_progress.",26),
    observed("The urlize probe prints links with trailing punctuation outside the anchor, including a %3C href for the HTML-escaped case, and exits with code 0.",27),
    observed("The targeted urlize filter invocation reports 45 tests, OK and exit code 0.",28),
    asserted("The plan explains the intended slicing change, marks analysis and implementation done and verification in_progress.",29),
    observed("The template-filter probe prints a rel=nofollow link with href containing %3C and visible 1&amp;lt, then exits with code 0.",30)],
"f40d08cd": [
    observed("The command displays autodetector.py lines 780 to 860.",10),
    asserted("The plan identifies deletion of the to key as the suspected KeyError source and proposes pop('to', None).",11),
    observed("The search finds MigrationAutodetector test references, and the beginning of test_autodetector.py is displayed.",12,13),
    observed("README.rst was displayed and the repository listing includes repro_custom_fk.py.",14,15),
    observed("The command displays autodetector.py lines 440 to 520.",16)],
"e227cc8b": [
    observed("Reading scripts/runtests.py fails because the file is absent, with exit code 2; tox.ini is then displayed.",17,18),
    observed("The command successfully displays tests/runtests.py, showing the actual test-runner source.",19)],
"2bd6a46f": [
    observed("Before committing, git status lists autodetector.py modified and repro_custom_fk.py untracked; git reports commit 405e4d218d with one insertion and one deletion.",27,30),
    asserted("The task notes claim autodetector tests passed while explicitly reporting the custom reproduction was blocked by app-registry constraints.",31)],
"8c31b3e0": [
    observed("The initial task-list view reports no task list found.",1),
    asserted("The SAM image-processor plan starts with all phases todo, then marks reading in_progress.",2,3),
    observed("The Transformers repository directory was listed.",4),
    asserted("The plan subsequently marks reading done and repository testing in_progress.",5)],
"a166eb90": [
    observed("The psrp.py edit receipt changes allow_extras from True to False.",8),
    observed("The subsequent edit receipt removes dynamic extras parsing and constructs documented PSRP connection kwargs.",9)],
"00d2b545": [
    observed("The psrp search locates plugin references in integration helpers and test/units/plugins/connection/test_psrp.py.",10),
    observed("The command displays the first 200 lines of test_psrp.py, including parametrized option data.",11),
    asserted("The task list marks reading done and testing in_progress while later phases remain todo.",12),
    observed("pytest cannot be found and exits 127; python -m pytest reports no pytest module and exits 1.",13,14),
    observed("requirements.txt and the remainder of test_psrp.py are displayed, including an assertion about unsupported extras.",15,16),
    observed("Git status lists psrp.py modified. Installing pytest fails after connection-refused retries and ends with no matching distribution, exit code 1.",17,18)],
"1720bd3c": [
    observed("The searches show psrp.py allow_extras=False and locate allow_extras handling in task_executor.py and plugin modules.",23,24),
    observed("The pypsrp.FEATURES search prints no matches, and psrp.py lines 680 to 820 are displayed.",25,26)],
"0032dc46": [
    asserted("The SELinux/respawn plan starts with all phases todo, then marks reading in_progress.",1,3),
    observed("The Ansible repository directory was listed.",4),
    asserted("The plan marks reading done and describes the desired interpreter-respawn and SELinux compatibility work.",5)],
"a4a480ed": [
    observed("dnf.py is displayed and the search returns libselinux-python references in modules, basic.py and integration support.",25,26),
    observed("The basic.py edit receipt replaces the missing-bindings hard failure with selinuxenabled CLI detection, returning rc==0, and uses the Python bindings when available.",27)],
"90eb9b5b": [
    observed("The config.py edit receipt adds _get_galaxy_server_configs with dynamic server settings, values and origins, required markers and timeout defaults.",19),
    observed("A second config.py edit receipt injects GALAXY_SERVERS when dumping base or all configurations.",20)],
"6442e992": [
    asserted("The plan describes adding Galaxy server configurations to ansible-config dump output.",21),
    observed("pytest is not found with exit code 127, and python -m pytest reports no pytest module with exit code 1.",22,23),
    observed("unittest reports zero tests and NO TESTS RAN with exit code 5.",24),
    observed("The temporary config creation/dump command reports a shell syntax error near && and exits 2.",25),
    observed("The next dump attempt reports no ansible module, although the pipeline exits with code 0.",26),
    observed("A base.yml excerpt is displayed, and a further dump attempt raises PackageNotFoundError for jinja2 while the pipeline exits with code 0.",27,28),
    observed("requirements.txt is displayed; dependency installation fails after connection-refused retries for jinja2 and exits 1.",29,30)],
"3cbc5ee2": [
    observed("The variable.py edit receipt replaces self.data.chunks with self._data.chunks.",18),
    observed("The common.py edit receipt changes chunk discovery to inspect v.chunks rather than accessing v.data.",19),
    observed("The get_chunksizes search pipeline prints count 4.",20),
    observed("The search locates two get_chunksizes(self.variables.values()) uses in dataset.py.",21),
    asserted("The plan reports the chunk-access changes and keeps exploration in_progress with reproduction todo.",22)],
"d2b30e30": [
    observed("The commands list the pytest repository and display setuponly.py.",4,5),
    observed("The search locates saferepr in _pytest/_io/saferepr.py and that source file is displayed.",6,7),
    asserted("The plan marks requirements analysis done and test/reproduction execution in_progress.",8),
    observed("The cached_param search locates its assignments and formatting in setuponly.py; setuponly.py and setupplan.py are listed and setupplan.py is displayed.",9,10,11)],
"f1ed2b36": [
    observed("Git initially reports a clean working tree and the repository directory is listed.",12,13),
    observed("The setuponly.py edit receipts import saferepr and apply it when formatting cached_param.",14,15)],
"a497c0a8": [
    asserted("The plan marks reading and exploration done while testing is in_progress.",16),
    observed("The attempted reproduction generator raises SyntaxError and exits 1.",17),
    observed("The heredoc command prints an end-of-file delimiter warning and exits with code 0.",18),
    observed("A pytest attempt reports no pytest module and EXIT:1 despite pipeline exit code 0.",19),
    observed("The next pytest attempt raises ModuleNotFoundError for attr and prints EXIT:1 while the pipeline exits 0.",20),
    observed("The git user.name command prints openhands.",21),
    observed("The git user.email command prints openhands@all-hands.dev.",22),
    observed("The later pytest attempt reports a collection SyntaxError and EXIT:2, while the outer command exits 0.",23),
    observed("The displayed test file contains stray heredoc/shell lines; a subsequent command rewrites it to the parametrized bytes test.",24,25),
    observed("The final pytest invocation reports 2 passed and EXIT:0 with outer exit code 0.",26)],
"7f8a558f": [
    observed("The sed command displays monitorStack.js lines 1 to 160.",4),
    observed("The numbered view displays monitorStack.js, including its monitoring loop and resolve(stackStatus).",5),
    observed("package.json identifies the Serverless package and its Node engine requirement.",6)],
"c3d6f6eb": [
    asserted("The IAM-statement plan initially lists all phases todo, then marks reading in_progress; the later note describes the requirement to warn on non-array iamRoleStatements.",1,3,5),
    observed("The Serverless repository directory was listed.",4)],
"886f082d": [
    observed("Git status lists mergeIamTemplates.js modified and Dockerfile untracked.",30),
    observed("CLI.js and .gitignore are displayed.",31,32),
    observed("Git reports a commit whose title describes warning on non-array iamRoleStatements.",33),
    asserted("The final task note marks review done and claims tests introduced no new failures.",34)],
"b69445ce": [
    observed("The repository and tests directories are listed.",1,2),
    observed("package.json identifies Serverless version 1.11.0, and the AWS plugin directory is listed.",3,4),
    observed("Directory listings follow package/compile/events to stream, where index.js and its test file are present.",5,6,7,8)],
"f2b9221f": [
    observed("The stream/index.js edit receipt adds handling for a non-ARN string role when constructing EventSourceMapping DependsOn.",21),
    asserted("The plan describes targeted role-dependency verification and marks its listed work statuses.",22),
    observed("Git status lists stream/index.js modified; the following command records a commit titled to describe custom-role DependsOn handling.",23,24)],
"3afef39f": [
    observed("The repository listing and minimal_blocks search locate perm_groups.py and related tests.",4,5),
    observed("The commands display perm_groups.py lines 2100 to 2280, including the block-system removal code.",6,7),
    observed("The sylow_subgroup search locates implementation and test references.",8),
    observed("The command displays perm_groups.py lines 4220 to 4400.",9),
    observed("The DihedralGroup(18) probe prints order 36 and degree 18, then raises IndexError while computing sylow_subgroup and exits 1.",10)],
"5b0b0c8d": [
    asserted("The plan marks initial testing done and exploration in_progress.",14),
    observed("The targeted pytest attempt fails because pytest is not found, with exit code 127.",15),
    observed("The group-theory probe prints Sylow subgroup orders 5 for S6 at p=5, 3/3/81 for A3/A5/A9 at p=3, and 8/4 for S4/A4 at p=2; it exits with code 0.",16),
    asserted("The plan marks exploration done and reproduction in_progress.",17),
    observed("The tool creates dev_repro_sylow.py and running it prints order 4 for both DihedralGroup(18) and DihedralGroup(50), exiting 0.",18,19),
    asserted("The task list updates implementation and verification progress after the reproduction run.",20)],
"287a5a10": [
    observed("Git status lists perm_groups.py modified; the requested diff slice prints no output.",21,22),
    observed("Git reports a commit for perm_groups.py with 8 insertions and 6 deletions.",23),
    asserted("The task list marks verification done and final review in_progress.",24),
    observed("After dev_repro_sylow.py removal, git reports a clean working tree.",25),
    asserted("The final task-list update marks final review done.",26)],
"d9351f85": [
    observed("test_console.py is displayed and task_timeout references are found in executor/config code; the broad test search returns test_wait_import_task_timeout in Galaxy tests.",21,22,23),
    observed("Another search returns the Galaxy timeout test, then test_adhoc.py is displayed.",24,25),
    observed("The final targeted grep pipeline prints no matching lines.",26)],
"11b2778e": [
    observed("The command displays lib/ansible/cli/__init__.py lines 1 to 260.",27),
    observed("The search locates _play_prereqs at line 442 and displays the later CLI source.",28),
    observed("The command displays vars/manager.py.",29),
    observed("The include-timeout/console-extra-vars search pipeline prints no matching lines.",30),
    observed("The command displays utils/vars.py.",31)],
"b2ede0fd": [
    observed("The option_helpers.py edit receipt adds the integer --task-timeout argument with default None.",32),
    observed("The Makefile is displayed.",33),
    observed("The adhoc.py edit receipt conditionally adds task_timeout to the task dictionary.",34),
    observed("The console.py edit receipts add add_runtask_options and conditionally add timeout to its task dictionary.",35,36),
    observed("The task_include.py edit receipt adds timeout to VALID_INCLUDE_KEYWORDS.",37),
    observed("The following applypatch attempt fails because the command is missing; a string replacement also fails because its old string is absent.",38,39),
    observed("The subsequent git command records a commit describing timeout and console extra-vars support.",40)],
"098aa6f1": [
    asserted("The task list marks reading done and testing in_progress.",41),
    observed("The first CLI probe raises ModuleNotFoundError for ansible and exits 1.",42),
    observed("Retrying with PYTHONPATH=lib raises ModuleNotFoundError for yaml and exits 1.",43),
    asserted("The updated plan reports inspection-based validation and missing pytest/PyYAML constraints.",44)],
"badd8edf": [
    asserted("The plan marks reading done and test execution in_progress for the LLM cache issue.",15),
    observed("The language-model, libs/core and test directories are listed; pytest fails because the command is not found, exit code 127.",16,17,18,19),
    observed("The unit-test language_models and llms directories are listed.",20,21),
    observed("The first 220 lines of test_base.py are displayed.",22),
    observed("The later test_base.py slice prints no source lines; pyproject.toml is then displayed.",23,24)],
"140f360c": [
    observed("The commands display LLM cache tests and fake.py.",42,43),
    observed("Searching llm_caching_params returns exit code 1; searching cache= returns multiple source and test matches.",44,45),
    observed("The displayed chat-model cache test source includes local/global and synchronous/asynchronous cache cases.",46),
    observed("The llms.py edit receipt introduces llm_cache from self.cache, passes it to aget_prompts and removes the local-cache NotImplementedError branch in that path.",47),
    observed("Further llms.py edit receipts pass llm_cache to update_cache and aupdate_cache.",48,49)],
"eb4d85c8": [
    asserted("The plan describes a draggable-helper canvas reference as the suspected pickling cause and proposes excluding _draggable from saved state.",23,26),
    observed("The __getstate__ search results and figure.py source are displayed.",24,25),
    observed("Legend is found at line 329 and its source is displayed.",27,28),
    observed("The Agg-backend draggable-legend probe prints Pickle OK and exits with code 0.",29)],
"48f0b073": [
    observed("The legend.py edit receipt adds __getstate__ setting _draggable to None.",30),
    observed("The command displays text.py lines 1600 to 1720.",31),
    observed("The text.py edit receipt adds another __getstate__ setting _draggable to None.",32),
    observed("The Agg-backend probe prints serialized lengths 56172 for the legend figure and 56853 for the annotation figure, All good and exit code 0.",33)],
"d1d8e13d": [
    observed("applypatch is not found and exits 127, and the subsequent string replacement reports no replacement performed.",23,24),
    observed("The command displays numbered linkComputer.ts lines 200 to 340.",25),
    observed("Another string replacement fails; the later command displays lines 258 to 270.",26,27),
    observed("A third replacement fails, and a Python inspection reports old in file? False.",28,29),
    observed("The next Python script prints the located character offsets and done after writing the replacement, then exits with code 0.",30),
    asserted("The task list records implementation progress for link parsing after the script.",31)],
}


def make_claims(specs,row):
    claims=[]
    for basis,text,locations in specs:
        ids=[eid for eid,ref in row["evidence_map"].items()
             if (ref["source_step_id"],ref["field"]) in locations]
        wanted=set(locations)
        found={(ref["source_step_id"],ref["field"]) for eid,ref in row["evidence_map"].items()
               if eid in ids}
        if found!=wanted:
            raise ValueError("Reference source location missing")
        claims.append(dict(text=text,source=basis,evidence_ids=ids))
    response={"claims":claims}
    base.validate(response,row["base_row"])
    return response


def main():
    packets=json.loads(Path("pilot/claim_sft_remaining_v1/packets.json").read_text(encoding="utf-8"))
    out=Path("pilot/claim_sft_long_review_v1")
    out.mkdir(parents=True,exist_ok=True)
    reviews=[]
    projected=[]
    uncovered=[]
    for packet in packets:
        specs=REFERENCES[packet["parent_job_id"][:8]]
        if not specs:
            raise ValueError("Pending reference")
        row=core.build(packet["source"])
        response=make_claims(specs,row)
        parent_key=base.digest(base.encoded(dict(run_id=row["run_id"],
            stage=packet["source"]["stage"])))
        reviews.append(dict(parent_job_id=packet["parent_job_id"],job_id=row["job_id"],
            split=packet["metadata"]["split"],group=packet["metadata"]["group"],
            reviewed=True,reviewer_type="AI",scope="Key facts in supplied stage evidence",
            final_response=response,exhaustive_semantic_coverage=False))
        projected_spec_indices=set()
        for child in packet["children"]:
            child_row=core.build(child["row"]["source_snapshot"],child["row"]["chunk_chars"])
            locations={(r["source_step_id"],r["field"]) for r in child_row["evidence_map"].values()}
            subset=[spec for spec in specs if set(spec[2])<=locations]
            projected_spec_indices.update(i for i,spec in enumerate(specs) if set(spec[2])<=locations)
            if not subset:
                uncovered.append(dict(parent_job_id=packet["parent_job_id"],child_job_id=child["row"]["job_id"]))
                continue
            projected.append(dict(parent_job_id=packet["parent_job_id"],job_id=child_row["job_id"],
                parent_stage_key=parent_key,
                group=packet["metadata"]["group"],split=packet["metadata"]["split"],
                reviewer_type="AI",prompt=[dict(role="system",content=core.INSTRUCTION),
                    dict(role="user",content=base.encoded(child_row["request"]).decode())],
                completion=[dict(role="assistant",content=base.encoded(make_claims(subset,child_row)).decode())]))
        if projected_spec_indices!=set(range(len(specs))):
            missing=[specs[i] for i in set(range(len(specs)))-projected_spec_indices]
            raise ValueError(f'Parent {packet["parent_job_id"][:8]} facts lost during projection: {missing}')
    (out/"ai_review.json").write_bytes(base.encoded(reviews)+b"\n")
    (out/"projected_examples.jsonl").write_bytes(b"".join(base.encoded(r)+b"\n" for r in projected))
    summary=dict(reviewed_parents=len(reviews),projected_examples=len(projected),
                 uncovered_children=uncovered,training_started=False)
    if uncovered:
        raise ValueError("Some child inputs still lack reviewed targets")
    (out/"summary.json").write_bytes(base.encoded(summary)+b"\n")
    print(summary)


if __name__=="__main__":
    main()
