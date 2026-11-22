"""Package the preselected fold0 adjacent GRU without refitting or retuning."""
from pathlib import Path

from scripts.run_stage_claim_pipeline import read,save
from scripts.audit_neural_ledger_v3_return import require
from segmenter.ledger import sha


def main():
    source=Path('temporal_boundary_train_v1/run_fold0/fold0/immediate/model_adjacent.json')
    out=Path('artifacts/boundary_gru_adjacent_v1')
    meta=read(source)
    require(meta['policy']==dict(threshold=.55,margin=0.,max_delay=0,adjacent_threshold=.65),'Policy changed')
    weights=(source.parent/meta['weights']).read_bytes()
    require(sha(weights)==meta['weights_sha256'],'Weight checksum mismatch')
    out.mkdir(parents=True,exist_ok=True)
    target=out/'weights.pt'
    if target.exists():require(target.read_bytes()==weights,'Existing candidate weights changed')
    else:target.write_bytes(weights)
    meta['weights']='weights.pt'
    save(out/'model.json',meta)
    from segmenter.temporal_boundary import TemporalPredictor
    model=TemporalPredictor.load(out/'model.json')
    count=sum(p.numel() for p in model.net.parameters())
    require(count==meta['parameter_count'],'Parameter count mismatch')
    save(out/'provenance.json',dict(source=source.as_posix(),source_json_sha256=sha(source.read_bytes()),
        model_json_sha256=sha((out/'model.json').read_bytes()),weights_sha256=sha(weights),
        policy=meta['policy'],parameters=count,bytes=len(weights),
        selection='Preexisting fold0 deployment candidate, not chosen by new end-to-end outputs',
        scope='Exploratory SWE-trained model; not all-data refit and not an independent accuracy result'))
    print(dict(path=str(out/'model.json'),parameters=count,bytes=len(weights),weights_sha256=sha(weights)))


if __name__=='__main__':main()
