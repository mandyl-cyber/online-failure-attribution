"""Synthetic live JSONL producer + observer restart; not a real agent benchmark."""
import argparse
import json
import threading
import time
from pathlib import Path
from segmenter.online_ledger import OnlineLedger
from segmenter.run_online import _iter_live_events
from segmenter.durable_input import atomic_json
from segmenter.timeline_report import report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    a=ap.parse_args(); root=Path(a.out)
    root.mkdir(parents=True, exist_ok=False)
    source=root/'source.jsonl'; source.touch()
    errors=[]
    def producer():
        try:
            with source.open('a', encoding='utf-8') as fh:
                for i in range(8):
                    action=dict(id=2*i, source='agent', action='run', args={'command':'pytest' if i%2 else 'cat a.py'})
                    obs=dict(id=2*i+1, observation='run', cause=2*i, content='1 passed', extras={'exit_code':0})
                    for ev in (action,obs):
                        line=json.dumps(ev)+'\n'
                        fh.write(line[:12]); fh.flush(); time.sleep(.005)
                        fh.write(line[12:]); fh.flush(); time.sleep(.02)
                fh.write(json.dumps(dict(id=16, source='agent', action='finish', args={}))+'\n'); fh.flush()
        except Exception as exc: errors.append(str(exc))
    runtime=OnlineLedger(root/'ledger', 'synthetic', background=False)
    writer=threading.Thread(target=producer); writer.start()
    restarted=False
    try:
        for event in _iter_live_events(source, .002, 5):
            runtime.ingest(event)
            if event['id']==4:
                runtime.close()
                runtime=OnlineLedger(root/'ledger', 'synthetic', background=False)
                assert 4 in runtime.adapter._pending
                assert runtime.ingest(event) is False
                restarted=True
            if runtime.finished: break
        runtime.drain()
        assert runtime.finished and len(runtime.segmenter.steps)==9
        assert not runtime.adapter._pending
    finally:
        writer.join(timeout=10); runtime.close()
    assert not errors and not writer.is_alive() and restarted
    result=report(root)
    result.update(validation='synthetic live file, not OpenHands live run', steps=9, restart_with_pending=True)
    atomic_json(root/'report.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
