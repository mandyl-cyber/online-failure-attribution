"""Structured observed facts for stage summaries, retaining raw evidence citations."""
from copy import deepcopy
from . import stage_claim_summary as v3
from . import stage_claim_summary_v4 as v4

PROTOCOL = "stage-claim-summary-v5"
SCHEMA = v3.SCHEMA
INSTRUCTION = v4.INSTRUCTION + """
Write concrete stage claims with the affected file, command or test scope.
Preserve observed test counts, skipped counts and exit codes when supplied.
Combine repeated reads of the same object into one claim with all relevant citations.
Explain specific results from raw evidence. Include distinct failed attempts and
their causes. Attribute task-list statements explicitly to the task list.
"""


def build(source, chunk_chars=1200):
    row = deepcopy(v4.build(source,chunk_chars))
    row["request"]["protocol"] = PROTOCOL
    row["request"]["observed_facts"] = [
        {key:value for key,value in fact.items() if key != "summary"}
        for fact in row["observed_facts"]]
    row["job_id"] = v3.digest(v3.encoded(row["request"]))
    row["input_sha256"] = v3.digest(v3.encoded(row["request"]))
    return row


def verify_row(row):
    if build(row["source_snapshot"],row["chunk_chars"]) != row:
        raise ValueError("v5 input drift")


def validate(response,row):
    verify_row(row)
    checked = v4.validate(response,v4.build(row["source_snapshot"],row["chunk_chars"]))
    for i,claim in enumerate(checked["claims"],1):
        claim["claim_id"] = f'{row["job_id"]}:c{i}'
    return checked
