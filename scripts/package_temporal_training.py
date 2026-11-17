"""Package audited features for Git sync; omit text unused by structure-only LR."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',default='eval/temporal_boundary_v1/data')
    ap.add_argument('--out',default='remote_pilots/temporal_boundary_v1/data')
    a=ap.parse_args();source=Path(a.source);out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    records=json.loads((source/'records.json').read_text(encoding='utf-8'))
    audit=json.loads((source/'audit.json').read_text(encoding='utf-8'))
    if audit['failures'] or len(records)!=audit['expected']:raise ValueError('Incomplete preparation')
    for record in records:
        for row in record['baseline_rows']:row.pop('text',None)
    payload=json.dumps(records,ensure_ascii=False,separators=(',',':')).encode()
    (out/'records.json.gz').write_bytes(gzip.compress(payload,compresslevel=9,mtime=0))
    shutil.copyfile(source/'vectors.npz',out/'vectors.npz')
    # Stable bytes after Git's Windows/Linux newline conversion.
    (out/'audit.json').write_bytes((source/'audit.json').read_text(encoding='utf-8').encode('utf-8'))
    manifest=dict(version=1,trajectories=len(records),steps=sum(len(r['ids']) for r in records),
        omitted='Only baseline_rows.text; all numeric/categorical LR features and vectors preserved',
        original_records_sha256=hashlib.sha256((source/'records.json').read_bytes()).hexdigest(),
        files={name:dict(sha256=hashlib.sha256((out/name).read_bytes()).hexdigest(),bytes=(out/name).stat().st_size)
               for name in ('records.json.gz','vectors.npz','audit.json')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
