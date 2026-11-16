"""Freeze v2 SWE purpose packages from immutable v1 trajectory selections."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('remote_pilots/swe_purpose_v1'))
    ap.add_argument('--out',type=Path,default=Path('remote_pilots/swe_purpose_v2'))
    args=ap.parse_args()
    if args.out.exists(): raise SystemExit('Refusing overwrite: '+str(args.out))
    assets=Path(__file__).resolve().parent.parent/'remote_pilots'/'swe_purpose_v2_assets'
    client=Path(__file__).resolve().parent/'purpose_pilot_remote_v2.py'
    for split in ('train','dev'):
        src=args.source/split; dst=args.out/split
        dst.mkdir(parents=True)
        for name in ('trajectories.jsonl','ENVIRONMENT_TEMPLATE.txt'):
            shutil.copyfile(src/name,dst/name)
        shutil.copyfile(client,dst/'remote.py')
        for name in ('SYSTEM_PROMPT.txt','OUTPUT_SCHEMA.json','README.md'):
            shutil.copyfile(assets/name,dst/name)
        trajectories=[json.loads(x) for x in (dst/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dst.iterdir())}
        (dst/'package_manifest.json').write_text(json.dumps(dict(protocol='swe-causal-bounded-purpose-v2',
            source_v1_trajectory_sha256=hashlib.sha256((src/'trajectories.jsonl').read_bytes()).hexdigest(),
            trajectories=len(trajectories),requests=sum(len(t['steps']) for t in trajectories),files=files),indent=2)+'\n')
        print(split,len(trajectories),sum(len(t['steps']) for t in trajectories))


if __name__=='__main__': main()
