"""Run the structured-fact v5 prompt with the frozen v1 experiment runner."""
from pathlib import Path
from remote_pilots.openhands_smoke import stage_claim_facts_v1 as runner
from segmenter import stage_claim_summary_v5


if __name__ == "__main__":
    import sys
    runner.v4 = stage_claim_summary_v5
    if "--out" not in sys.argv:
        sys.argv.extend(["--out", str(Path("out/stage_claim_facts_v2"))])
    raise SystemExit(runner.main())
