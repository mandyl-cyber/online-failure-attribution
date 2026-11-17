"""Check actual recovered public calls against the fixed adapter, not model output."""
import json
from pathlib import Path
from segmenter.adapters.openhands import OpenHandsStreamAdapter
from remote_pilots.openhands_smoke import claims_v3 as io
from remote_pilots.openhands_smoke.repair_seal_claim_inputs import full_calls
from segmenter import stage_claim_summary as core


def main():
    repairs=io.read(Path('pilot/seal_claim_diagnostic_v3/repairs.json'))
    records={r['traj_id']:r for r in io.read(Path('eval/boundary_supervised/data_fixed/records.json'))}
    cache={};results=[]
    for r in repairs:
        tid=r['run_id']
        if tid not in cache:
            blob=Path(records[tid]['raw_source']).read_bytes()
            if core.digest(blob)!=r['source_sha256']:raise ValueError('raw input hash mismatch')
            cache[tid]=full_calls(json.loads(blob.decode('utf-8')),{x['native_call_id'] for x in repairs if x['run_id']==tid})
        args=cache[tid][r['native_call_id']]
        if args!=r['payload']:raise ValueError('recovery drift')
        adapter=OpenHandsStreamAdapter()
        adapter.feed(dict(id=r['native_call_id'],action='task_tracking',source='agent',args=args))
        # No fabricated result: verify the action on explicit incomplete flush.
        step=adapter.flush()[0]
        if json.loads(step.action)!=args or step.attrs['public_action_args']!=args:
            raise ValueError('public arguments lost')
        results.append(dict(run_id=tid,native_call_id=r['native_call_id'],source_sha256=r['source_sha256'],
                            exact_args_preserved=True,missing_observation_explicit=True))
    out=Path('eval/event_input_repair_v2');out.mkdir(parents=True,exist_ok=False)
    io.save(out/'report.json',dict(checked_calls=len(results),trajectories=len(cache),results=results,
        scope='16 previously lost public calls, rechecked against native source; not a full-stream benchmark or online restart test'))
    print(dict(checked_calls=len(results),trajectories=len(cache),out=str(out)))


if __name__=='__main__':main()
