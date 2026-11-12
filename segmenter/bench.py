"""CodeTraceBench 读取：manifest -> 条目（stage 标注、incorrect 步、轨迹路径）。

数据集（huggingface-cli download NJU-LINK/CodeTraceBench --repo-type dataset --local-dir ./codetrace_bench）
  bench_manifest.{verified,full}.jsonl / .parquet     每行一条轨迹的元数据和标注
  bench_artifacts/full/<traj_id>.tar.zst              原始轨迹

第 1 天要做的"打印一条轨迹的全部字段"：
  python -m segmenter.bench inspect --manifest codetrace_bench/bench_manifest.verified.jsonl --artifacts codetrace_bench
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Iterator, Optional


def read_manifest(path: str | Path) -> list[dict]:
    p = Path(path)
    if p.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("读 parquet 需要 pandas + pyarrow；或者改用同名的 .jsonl manifest") from e
        return pd.read_parquet(p).to_dict("records")
    rows = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _j(x, default):
    if x is None:
        return default
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return default
    return x


def parse_entry(row: dict) -> dict:
    """把 manifest 的一行整理成评测要用的形态。"""
    stages = [s for s in _j(row.get("stages"), []) if isinstance(s, dict) and "start_step_id" in s]
    inc = [s for s in _j(row.get("incorrect_stages"), []) if isinstance(s, dict)]
    incorrect, unuseful, instances = [], [], []
    for s in inc:
        ids = sorted(int(i) for i in (s.get("incorrect_step_ids") or []))
        incorrect += ids
        unuseful += [int(i) for i in (s.get("unuseful_step_ids") or [])]
        if ids:
            instances.append(ids[0])           # 每个出错 stage 的第一个 incorrect 步 = 一个错误实例
    starts = [int(s["start_step_id"]) for s in stages]
    return {
        "traj_id": str(row.get("traj_id")), "agent": row.get("agent"), "model": row.get("model"),
        "solved": row.get("solved"), "step_count": row.get("step_count"), "difficulty": row.get("difficulty"),
        "stages": stages, "incorrect_steps": sorted(set(incorrect)), "unuseful_steps": sorted(set(unuseful)),
        "instances": sorted(set(instances)),
        "base": min(starts) if starts and min(starts) in (0, 1) else 1,
        "source_relpath": row.get("source_relpath"), "artifact_path": row.get("artifact_path"),
    }


_KEEP_FLAT = re.compile(r"(^|/)(steps\.json|[^/]*-\d+\.\d+\.json)$")
_EVENT_FILE = re.compile(r"(^|/)events/(\d+\.json)$")


def _extract_stream(fileobj, dest: Path) -> int:
    """只解出轨迹需要的内容，写到短路径下的少数几个文件里。

    原始目录层级太深（超 Windows 260 字符上限），而且一条轨迹可能有几千个事件小文件，逐个落盘很慢。
         .../events/<n>.json            -> 合并成 dest/events.jsonl（按 n 排序；只取遇到的第一个 events 目录）
         steps.json / <model>-<ts>.json -> dest/<文件名>
    其余（终端录屏、日志、pkl 等）不解。返回读到的轨迹文件数。
    """
    n, ev_parent, events = 0, None, {}
    with tarfile.open(fileobj=fileobj, mode="r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name.replace("\\", "/")
            em = _EVENT_FILE.search(name)
            if not em and not _KEEP_FLAT.search(name):
                continue
            src = tf.extractfile(m)
            if src is None:
                continue
            if em:
                parent = name[: em.start()]
                if ev_parent is None:
                    ev_parent = parent
                if parent != ev_parent:
                    continue
                try:
                    events[int(em.group(2)[:-5])] = json.loads(src.read().decode("utf-8", "replace"))
                except Exception:
                    continue
            else:
                with open(dest / name.rsplit("/", 1)[-1], "wb") as fh:
                    shutil.copyfileobj(src, fh)
            n += 1
    if events:
        with open(dest / "events.jsonl", "w", encoding="utf-8") as fh:
            for k in sorted(events):
                fh.write(json.dumps(events[k], ensure_ascii=False) + "\n")
    return n


def _extract(archive: Path, dest: Path) -> None:
    """解 .tar.zst：python-zstandard > zstd 命令。"""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import zstandard
    except ImportError:
        zstandard = None
    if zstandard is not None:
        with open(archive, "rb") as fh:
            n = _extract_stream(zstandard.ZstdDecompressor().stream_reader(fh), dest)
    elif shutil.which("zstd"):
        proc = subprocess.Popen(["zstd", "-d", "-c", str(archive)], stdout=subprocess.PIPE)
        n = _extract_stream(proc.stdout, dest)
        if proc.wait() != 0:
            raise RuntimeError(f"zstd 解压失败：{archive}")
    else:
        raise RuntimeError("解 .tar.zst 需要 pip install zstandard（或系统里有 zstd 命令）")
    if n == 0:
        raise RuntimeError(f"{archive.name} 里没有找到轨迹文件（events/*.json、steps.json 或 completion 日志）")
    (dest / ".ok2").write_text(str(n), encoding="utf-8")


def locate_trajectory(entry: dict, root: str | Path, cache: str | Path | None = None) -> Path:
    """找到（必要时解包）一条轨迹的目录或文件。"""
    root = Path(root)
    tid = entry["traj_id"]
    cache = Path(cache) if cache else root / "_extracted"
    cands = []
    for rel in (entry.get("source_relpath"), entry.get("artifact_path")):   # 已解开的原始目录优先，省一次解包
        if rel:
            cands += [root / rel, root / Path(rel).name]
    cands += [root / tid, root / "bench_artifacts" / "full" / f"{tid}.tar.zst",
              root / "bench_artifacts" / "verified" / f"{tid}.tar.zst", root / f"{tid}.tar.zst",
              root / f"{tid}.json", root / f"{tid}.jsonl"]
    for c in cands:
        if not c.exists():
            continue
        if c.name.endswith((".tar.zst", ".tzst")):
            dest = cache / tid
            if not (dest / ".ok2").exists():        # 没有完成标记（含上次解到一半的）：重解
                _extract(c, dest)
            return dest
        return c
    raise FileNotFoundError(f"找不到轨迹 {tid}（试过 {len(cands)} 个位置，根目录 {root}）")


def iter_bench(manifest: str | Path, artifacts: str | Path, agent: Optional[str] = "OpenHands",
               cache: str | Path | None = None, limit: Optional[int] = None,
               traj_ids: Optional[set] = None) -> Iterator[dict]:
    n = 0
    for row in read_manifest(manifest):
        if agent and str(row.get("agent", "")).lower() != agent.lower():
            continue
        e = parse_entry(row)
        if traj_ids and e["traj_id"] not in traj_ids:
            continue
        try:
            e["traj_path"] = str(locate_trajectory(e, artifacts, cache))
        except Exception as ex:  # noqa: BLE001
            e["error"] = f"{type(ex).__name__}: {ex}"
        yield e
        n += 1
        if limit and n >= limit:
            return


def cmd_inspect(a) -> None:
    from collections import Counter
    from .adapters.openhands import find_trajectory_file, load_trajectory, _detect, _load_json_any
    rows = read_manifest(a.manifest)
    print(f"manifest: {len(rows)} 行；agent 分布：{dict(Counter(r.get('agent') for r in rows))}")
    row = next((r for r in rows if (a.traj_id and str(r.get("traj_id")) == a.traj_id)
                or (not a.traj_id and str(r.get("agent", "")).lower() == a.agent.lower())), None)
    if row is None:
        sys.exit("没有匹配的轨迹")
    print("\n== manifest 字段 ==")
    for k, v in row.items():
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
        print(f"  {k:<28} {s[:300]}{' ...' if len(s) > 300 else ''}")
    e = parse_entry(row)
    path = locate_trajectory(e, a.artifacts, a.cache)
    print(f"\n== 轨迹位置 == {path}")
    if path.is_dir():
        for p in sorted(path.rglob("*"))[:80]:
            if p.is_file():
                print(f"  {p.relative_to(path)}  ({p.stat().st_size} bytes)")
        f = find_trajectory_file(path)
        print(f"\n== 选中的轨迹文件 == {f}")
    else:
        f = path
    if f and f.is_file():
        fmt, recs = _detect(_load_json_any(f))
        print(f"格式 = {fmt}，{len(recs)} 条记录。前 {a.n} 条的全部字段：")
        for r in recs[: a.n]:
            print(json.dumps(r, ensure_ascii=False, indent=1, default=str)[:3000])
    steps, meta = load_trajectory(path, base=e["base"], expect_steps=e["step_count"])
    print(f"\n== 转成统一 step == {len(steps)} 步（manifest step_count = {e['step_count']}，"
          f"起始步号 {e['base']}，原生步号 = {meta['native_ids']}）")
    if e["step_count"] is not None and meta.get("n_core", len(steps)) != int(e["step_count"]):
        print("  !! 步数与 manifest 不一致：标注的 step_id 对不上，先看上面的原始字段再调 adapters/openhands.py")
    for st in steps[: a.n]:
        print(json.dumps(st.to_dict(), ensure_ascii=False, indent=1)[:1500])
    print(f"\nstages: {e['stages'][:5]}\nincorrect_steps: {e['incorrect_steps']}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="segmenter.bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect", help="打印一条轨迹的全部字段")
    p.add_argument("--manifest", required=True)
    p.add_argument("--artifacts", required=True)
    p.add_argument("--traj-id")
    p.add_argument("--agent", default="OpenHands")
    p.add_argument("--cache")
    p.add_argument("-n", type=int, default=3)
    p.set_defaults(fn=cmd_inspect)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
