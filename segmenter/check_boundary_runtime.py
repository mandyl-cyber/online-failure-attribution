"""Check streaming/batch equivalence on held-out folds and timed live-file smoke."""
import json
import tempfile
import threading
import time
from pathlib import Path

from .boundary_model import OnlineBoundaryPredictor, joblib
import numpy as np
from .boundary_features import BoundaryFeatures
from .adapters.openhands import load_trajectory, OpenHandsStreamAdapter
from .bench import iter_bench
from .run_online import OnlineSegmenter, _iter_live_events
from .prepare_downstream import validate_ranges


def main():
    data = Path('eval/boundary_supervised/data_fixed')
    exp = Path('eval/boundary_supervised/experiment')
    records = json.loads((data/'records.json').read_text(encoding='utf-8'))
    chosen = []
    for source in sorted({r['source'] for r in records}):
        chosen += [r for r in records if r['source']==source][:5]
    wanted = {r['traj_id']:r for r in chosen}
    entries = {e['traj_id']:e for e in iter_bench('codetrace_bench/bench_manifest.verified.jsonl','codetrace_bench') if e['traj_id'] in wanted}
    model_ms, feed_ms, checks = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for r in chosen:
            e = entries[r['traj_id']]
            steps, _ = load_trajectory(e['traj_path'],base=e['base'],expect_steps=e['step_count'])
            steps = steps[:r['n_steps']]  # previously audited: exact IDs or FINISH-only suffix
            assert [s.idx for s in steps] == r['ids']
            path = exp/f"fold_{r['fold']}.joblib"
            model = joblib.load(path)
            f = BoundaryFeatures()
            batch = model.probabilities([f.feed(s) for s in steps])
            online = OnlineBoundaryPredictor(model)
            actual = [online.feed(s) for s in steps]
            assert np.allclose(batch,[e['score'] for e in actual],atol=1e-10)
            for n in sorted({1,len(steps)//2,len(steps)-1}):
                prefix = OnlineBoundaryPredictor(model)
                rows = [prefix.feed(s) for s in steps[:n]]
                assert [(x['boundary_step_id'],x['score']) for x in rows] == [(x['boundary_step_id'],x['score']) for x in actual[:n]]
            out = Path(tmp)/r['traj_id']
            seg = OnlineSegmenter(out_dir=out,boundary_model=path)
            for s in steps:
                start = time.perf_counter()
                row = seg.feed(s)
                feed_ms.append((time.perf_counter()-start)*1000)
                model_ms.append(row['supervised_boundary']['ms'])
            seg.finish()
            ranges = json.loads((out/'supervised_stage_ranges.json').read_text())
            validate_ranges(ranges,r['ids'])
            assert all(s['sealed'] for s in ranges)
            checks.append(dict(traj_id=r['traj_id'],steps=len(steps),fold=r['fold'],prefix_consistent=True))
        # Incrementally append a partial JSON line then complete it, through actual live reader.
        log = Path(tmp)/'live.jsonl'
        log.touch()
        events = []
        for i in range(4):
            events += [dict(id=2*i+1,source='agent',action='run',args={'command':'ls' if i<2 else 'pytest -q'}),
                       dict(id=2*i+2,cause=2*i+1,observation='run',content='ok',extras={'metadata':{'exit_code':0}})]
        produced = {}
        def writer():
            time.sleep(.03)
            with log.open('a',encoding='utf-8') as fh:
                for event in events:
                    text = json.dumps(event)
                    cut = len(text)//2
                    fh.write(text[:cut]); fh.flush(); time.sleep(.01)
                    produced[event['id']] = time.perf_counter()
                    fh.write(text[cut:]+'\n'); fh.flush(); time.sleep(.01)
        worker = threading.Thread(target=writer)
        worker.start()
        ad = OpenHandsStreamAdapter()
        live = OnlineSegmenter(out_dir=Path(tmp)/'live_out',boundary_model=exp/'candidate.joblib')
        seen, latencies, nsteps = [], [], 0
        for ev in _iter_live_events(log,.005,.25):
            seen.append(ev['id'])
            for s in ad.feed(ev):
                live.feed(s); nsteps += 1
                latencies.append((time.perf_counter()-produced[ev['id']])*1000)
        for s in ad.flush():
            live.feed(s); nsteps += 1
        live.finish(); worker.join()
        assert seen == [e['id'] for e in events] and nsteps == 4
    def quantiles(xs):
        return dict(zip(('p50','p95','p99'),map(float,np.quantile(xs,[.5,.95,.99]))))
    report = dict(checks=checks,steps=len(feed_ms),model_ms=quantiles(model_ms),
                  feed_with_disk_ms=quantiles(feed_ms),live_file_smoke=dict(events=len(seen),steps=nsteps,
                  poll_seconds=.005,partial_lines=True,post_write_to_feed_ms=quantiles(latencies)),
                  limitation='local sequential replay and synthetic live file; not a deployed real agent or concurrent load test; adapter time excluded from replay feed timing')
    (exp/'runtime.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))


if __name__ == '__main__':
    main()
