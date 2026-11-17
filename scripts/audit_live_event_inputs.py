"""Replay returned native event archives without executing any agent actions."""
from pathlib import Path
from collections import Counter
import json
import zipfile
from segmenter.adapters.openhands import OpenHandsStreamAdapter
from segmenter.ledger import atomic, sha


def main():
    reports=[]
    for path in sorted(Path('.').glob('run-*-diagnostics-*.zip')):
        with zipfile.ZipFile(path) as archive:
            names=sorted([n for n in archive.namelist() if '/events/' in n and n.endswith('.json')],
                         key=lambda n:int(Path(n).stem))
            adapter=OpenHandsStreamAdapter();steps=[];error=None
            for name in names:
                try:steps+=adapter.feed(json.loads(archive.read(name)))
                except Exception as exc:
                    error=dict(file=name,error=type(exc).__name__+': '+str(exc));break
            pending=len(adapter._pending)
            flushed=adapter.flush() if error is None else []
            steps+=flushed
            issues=Counter(i for s in steps for i in s.attrs['input_integrity']['issues'])
            reports.append(dict(archive=path.name,archive_sha256=sha(path.read_bytes()),events=len(names),
                steps=len(steps),pending_before_flush=pending,flushed_incomplete=len(flushed),error=error,
                pairing=dict(Counter(s.attrs['input_integrity']['pairing'] for s in steps)),
                issues=dict(issues),diagnostics=adapter.diagnostics,
                steps_index=[dict(idx=s.idx,tool=s.tool,raw_ref=s.raw_ref,input_integrity=s.attrs['input_integrity']) for s in steps]))
    if not reports:raise ValueError('No returned archives found')
    out=Path('eval/live_input_audit_v1');out.mkdir(parents=True,exist_ok=False)
    atomic(out/'report.json',dict(runs=reports,scope='three historical smoke runs, native event-ID replay order; not live latency or population accuracy'))
    print([{k:r[k] for k in ('archive','events','steps','pending_before_flush','error','pairing','issues')} for r in reports])


if __name__=='__main__':main()
