"""Partition stage evidence into consecutive whole-step requests under a token budget."""
from copy import deepcopy
from . import stage_claim_summary_v5 as core


def partition(row, fits):
    """Return fitting parts and explicit oversized single steps, retaining parent IDs."""
    source = row["source_snapshot"]
    start = source["stage"]["start_step_id"]
    end = source["stage"]["end_step_id"]
    parts = []
    oversized = []
    def visit(lo, hi):
        snapshot = deepcopy(source)
        snapshot["stage"]["start_step_id"] = lo
        snapshot["stage"]["end_step_id"] = hi
        snapshot["job_id"] = f'{source["job_id"]}:part:{lo}-{hi}'
        snapshot["evidence"] = [r for r in snapshot["evidence"] if lo <= r["source_step_id"] <= hi]
        candidate = core.build(snapshot,row["chunk_chars"])
        if fits(candidate):
            parts.append(dict(parent_job_id=row["job_id"],parent_stage=deepcopy(source["stage"]),
                              row=candidate))
        elif lo == hi:
            oversized.append(dict(parent_job_id=row["job_id"],step_id=lo,
                                  reason="single_step_exceeds_budget"))
        else:
            mid = (lo+hi)//2
            visit(lo,mid)
            visit(mid+1,hi)
    visit(start,end)
    return parts, oversized
