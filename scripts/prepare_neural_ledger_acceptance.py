"""Freeze three real native-event journals for controlled streaming replay."""
from pathlib import Path
from scripts.run_stage_claim_pipeline import read, save
from segmenter.ledger import encoded,sha


def main():
    root=Path('pilot/neural_ledger_acceptance_v1')
    sources=[('swe','eval/online_ledger_smoke_v1/SWE/native_events'),
             ('terminal','eval/online_ledger_smoke_v1/Terminal-Bench/native_events'),
             ('live_smoke','eval/temporal_boundary_v1/external_candidates/ledger_probe/original/native_events')]
    sessions=[]
    for name,directory in sources:
        records=[read(p) for p in sorted(Path(directory).glob('*.json'))]
        events=[r['event'] for r in records if r.get('kind')=='event']
        if not events:
            raise ValueError('Missing native session '+directory)
        payload=b''.join(encoded(e)+b'\n' for e in events)
        root.mkdir(parents=True,exist_ok=True)
        path=root/(name+'.jsonl')
        path.write_bytes(payload)
        sessions.append(dict(session_id=name,path=path.name,sha256=sha(payload),events=len(events),source=directory,
                             terminal_finish_present=any(e.get('action')=='finish' for e in events)))
    save(root/'protocol.json',dict(sessions=sessions,scope='Three available real event streams, including SWE and Terminal-Bench',
        scenario_labels='Failure/retest/status scenarios require semantic review; session selection is for engineering acceptance.',
        replay='Original event order, configurable replay interval. Open final stage stays open without a terminal event.',
        extractor='Existing 7B SFT adapter bridged to ledger protocol; single worker and shared model',
        checks=['event/step coverage','sealed jobs vs published batches','retries and failures','restart byte identity',
                'controlled replay queue/generation/publication times','GPU allocation','AI semantic review pending']))
    print(sessions)


if __name__=='__main__':
    main()
