"""在线划分器：每来一步就跑 事件检测 -> 树 -> 切分 -> 候选 claim，全部零解码。

  python -m segmenter.run_online replay <轨迹文件或目录> --out out/demo --print
  python -m segmenter.run_online follow <events 目录或 .jsonl> --out out/live      # 接在正在跑的 OpenHands 事件流上
  python -m segmenter.run_online bench  --manifest bench_manifest.verified.jsonl --artifacts <数据集根目录> --out out/bench

每条轨迹写出：steps.jsonl  events.jsonl  claims.jsonl  tree.json  stage_ranges.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from .adapters.openhands import OpenHandsStreamAdapter, load_trajectory
from .claims import ClaimExtractor
from .context import ExecutionContext
from .events import E, Event, EventDetector
from .rules import Rules, default_rules
from .schema import Step
from .segment import Segmenter
from .tree import StateTree


def _atomic_write(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


class OnlineSegmenter:
    """一条轨迹一个实例。feed(step) 返回这一步的事件向量（events.jsonl 的一行）。"""

    def __init__(self, rules: Rules | None = None, out_dir: str | Path | None = None,
                 modes: list[str] | None = None, boundary_model: str | Path | None = None,
                 purpose_model: str | Path | None = None, task: str = ''):
        self.rules = rules or default_rules()
        self.detector = EventDetector(self.rules)
        self.context = ExecutionContext()
        self.tree = StateTree(self.rules)
        modes = modes or [self.rules.segment.get("mode", "node")]
        self.segmenters = {m: Segmenter(self.rules, m) for m in modes}
        self.segmenter = self.segmenters[modes[0]]          # 主模式：写 stage_ranges.json 的那个
        self.claimer = ClaimExtractor(self.rules)
        self.supervised = None
        if boundary_model is not None:
            from .boundary_model import OnlineBoundaryPredictor
            self.supervised = OnlineBoundaryPredictor.load(boundary_model)
        self.purpose = None
        if purpose_model is not None:
            from .purpose_student import PurposeStudent
            self.purpose = PurposeStudent.load(purpose_model)
        self.task = task
        self.steps: list[Step] = []
        self.rows: list[dict] = []
        self.claims: list[dict] = []
        self.out = Path(out_dir) if out_dir else None
        self._fh = {}
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)
            names = ['steps', 'events', 'claims'] + (['boundary_events'] if self.supervised else [])
            if self.purpose:
                names.append('purpose_events')
            for name in names:
                self._fh[name] = open(self.out / f"{name}.jsonl", "w", encoding="utf-8")

    def _append(self, name: str, obj: dict) -> None:
        fh = self._fh.get(name)
        if fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fh.flush()

    def _dump_structs(self) -> None:
        if self.out:
            _atomic_write(self.out / "tree.json", self.tree.to_dict())
            _atomic_write(self.out / "stage_ranges.json", self.segmenter.snapshot())
            if self.supervised:
                _atomic_write(self.out / "supervised_stage_ranges.json", self.supervised.snapshot())

    def feed(self, step: Step) -> dict:
        t0 = time.perf_counter()
        purpose_prediction = None
        if self.purpose:
            if not self.task.strip():
                raise ValueError('Purpose model requires task text; pass --task-file if the stream lacks it')
            prior = [self.steps[-1].to_dict()] if self.steps else []
            purpose_prediction = self.purpose.predict(self.task,prior+[step.to_dict()])
        if self.rules.raw.get('execution_context', {}).get('enabled', False):
            self.context.apply(step)
        events = self.detector.detect(step)                       # 事件检测
        upd = self.tree.update(step, events)                      # 树
        ref = self.detector.detect_references(step, self.tree)    # 引用（要看更新后的树）
        deps = {}
        if ref is not None:
            events.append(ref)
            deps = self.tree.add_references(step, ref)
        sealed = {m: s.update(step, events, upd) for m, s in self.segmenters.items()}   # 切分
        claims = self.claimer.extract(step, events, upd["node"])  # 候选 claim

        types = [e.type for e in events]
        trig = self.rules.trigger
        k = int(trig.get("dep_k", 2))
        row = {
            "idx": step.idx, "tool_class": step.tool_class, "events": types, "node": upd["node"],
            "refs": ref.attrs["refs"] if ref else [],
            "for_idx": next((e.for_idx for e in events if e.type == E.VERIFY_PASS), None),
            "closed_node": [c[0] for c in upd["closed"]], "close_reason": [c[1] for c in upd["closed"]],
            "opened_node": upd["opened"],
            "sealed_stage": [s["stage_id"] for s in sealed[self.segmenter.mode]],
            "dep": deps,
            # 归因器的触发判定（σ -> consequential 那一条要等账本写入器，M4）
            "trigger": bool(set(types) & {E.HANDOFF, E.VERIFY_PASS, E.IRREVERSIBLE}) or any(v >= k for v in deps.values()),
            "sync": bool(set(types) & set(trig.get("sync", []))),
            "n_claims": len(claims),
            "detail": [e.to_dict() for e in events],
            "ms": round((time.perf_counter() - t0) * 1000, 3),
        }
        if purpose_prediction:
            row['purpose_prediction'] = purpose_prediction
            self._append('purpose_events',dict(step_id=step.idx,**purpose_prediction))
        self.steps.append(step)
        self.rows.append(row)
        self.claims += claims
        if self.supervised:
            boundary = self.supervised.feed(step)
            row['supervised_boundary'] = boundary
            self._append('boundary_events', boundary)
        self._append("steps", step.to_dict())
        self._append("events", row)
        for c in claims:
            self._append("claims", c)
        if upd["closed"] or row["sealed_stage"] or (self.supervised and boundary['status'] == 'committed'):
            self._dump_structs()
        return row

    def finish(self) -> dict:
        self.tree.finish()
        for s in self.segmenters.values():
            s.finish()
        self._dump_structs()
        if self.out and self.supervised:
            _atomic_write(self.out / 'supervised_stage_ranges.json', self.supervised.snapshot(final=True))
        for fh in self._fh.values():
            fh.close()
        self._fh = {}
        return self.result()

    def result(self) -> dict:
        stages = {m: s.snapshot() for m, s in self.segmenters.items()}
        if self.supervised:
            stages['supervised'] = self.supervised.snapshot()
        return {"rows": self.rows, "tree": self.tree.to_dict(), "claims": self.claims, "stages": stages}


def run_steps(steps: list[Step], rules: Rules | None = None, out_dir=None, modes=None,
              boundary_model=None,purpose_model=None,task='') -> dict:
    seg = OnlineSegmenter(rules, out_dir, modes, boundary_model,purpose_model,task)
    for st in steps:
        seg.feed(st)
    return seg.finish()


# -------------------------------------------------------------------- CLI

def _fmt_row(step: Step, row: dict) -> str:
    extra = []
    if row["refs"]:
        extra.append("refs=" + ",".join(row["refs"]))
    if row["for_idx"] is not None:
        extra.append(f"for_idx={row['for_idx']}")
    for d in row["detail"]:
        if d["type"] == E.VERIFY_RUN:
            extra.append(f"result={d['result']}")
    if row["closed_node"]:
        extra.append("closed=" + ",".join(f"{n}({r})" for n, r in zip(row["closed_node"], row["close_reason"])))
    if row["sealed_stage"]:
        extra.append(f"stage#{row['sealed_stage'][0]} sealed")
    if row["trigger"]:
        extra.append("TRIGGER" + ("/sync" if row["sync"] else ""))
    act = (step.action or step.thought).replace("\n", " ")[:60]
    return f"{step.idx:>4} {row['node']:>4} {step.tool_class:<10} {'+'.join(row['events']):<38} {' '.join(extra):<40} | {act}"


def cmd_replay(a) -> None:
    rules = Rules(a.rules) if a.rules else default_rules()
    steps, meta = load_trajectory(a.path, rules, base=a.base)
    task=Path(a.task_file).read_text(encoding='utf-8') if a.task_file else meta.get('task','')
    seg = OnlineSegmenter(rules, a.out, [a.mode] if a.mode else None,
                          a.boundary_model,a.purpose_model,task)
    print(f"# {meta['source']}  format={meta['format']}  steps={len(steps)}", file=sys.stderr)
    for st in steps:
        row = seg.feed(st)
        if a.print:
            print(_fmt_row(st, row))
        if a.delay:
            time.sleep(a.delay)
    res = seg.finish()
    ms = sorted(r["ms"] for r in res["rows"]) or [0]
    n_trig = sum(r["trigger"] for r in res["rows"])
    print(f"# nodes={len(res['tree']['nodes'])} stages={len(res['stages'][seg.segmenter.mode])} "
          f"claims={len(res['claims'])} triggers={n_trig} "
          f"per-step ms: median={ms[len(ms) // 2]} max={ms[-1]}" + (f"  -> {a.out}" if a.out else ""), file=sys.stderr)


def _iter_live_events(path: Path, poll: float, idle: float):
    """跟读一个正在增长的事件流：.jsonl 文件，或 events/ 目录（一个事件一个 json）。"""
    last = time.time()
    if path.is_dir():
        seen = set()
        while True:
            files = sorted((p for p in path.glob("*.json") if p.name not in seen),
                           key=lambda x: (int(x.stem) if x.stem.isdigit() else 1 << 60, x.name))
            for p in files:
                try:
                    ev = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    break                       # 还没写完，下一轮再读
                seen.add(p.name)
                last = time.time()
                yield ev
            if time.time() - last > idle:
                return
            time.sleep(poll)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            buf = ""
            while True:
                line = fh.readline()
                if line:
                    buf += line
                    if buf.endswith("\n"):
                        if buf.strip():
                            yield json.loads(buf)
                        buf, last = "", time.time()
                    continue
                if time.time() - last > idle:
                    return
                time.sleep(poll)


def cmd_follow(a) -> None:
    rules = Rules(a.rules) if a.rules else default_rules()
    ad = OpenHandsStreamAdapter(rules, base=a.base)
    explicit_task=Path(a.task_file).read_text(encoding='utf-8') if a.task_file else ''
    seg = OnlineSegmenter(rules, a.out, [a.mode] if a.mode else None,
                          a.boundary_model,a.purpose_model,explicit_task)
    done = False
    try:
        for ev in _iter_live_events(Path(a.path), a.poll, a.idle):
            for st in ad.feed(ev):
                if not explicit_task:
                    seg.task=ad.task
                row = seg.feed(st)
                print(_fmt_row(st, row), flush=True)
                done = done or st.tool == "finish"
            if done:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if a.out:
            diagnostic_dir=Path(a.out)
            diagnostic_dir.mkdir(parents=True,exist_ok=True)
            (diagnostic_dir/'input_diagnostics.json').write_text(
                json.dumps(ad.diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
    for st in ad.flush():
        if not explicit_task:
            seg.task=ad.task
        print(_fmt_row(st, seg.feed(st)), flush=True)
    seg.finish()


def cmd_bench(a) -> None:
    from .bench import iter_bench
    rules = Rules(a.rules) if a.rules else default_rules()
    n = ok = 0
    for entry in iter_bench(a.manifest, a.artifacts, agent=a.agent, cache=a.cache, limit=a.limit):
        n += 1
        if entry.get("error"):
            print(f"[skip] {entry['traj_id']}: {entry['error']}", file=sys.stderr)
            continue
        try:
            steps, meta = load_trajectory(entry["traj_path"], rules, base=entry["base"], expect_steps=entry.get("step_count"))
            task=Path(a.task_file).read_text(encoding='utf-8') if a.task_file else meta.get('task','')
            run_steps(steps, rules, Path(a.out or "out") / entry["traj_id"],
                      [a.mode] if a.mode else None, a.boundary_model,a.purpose_model,task)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {entry['traj_id']}: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"# {ok}/{n} 条轨迹已回放 -> {a.out}", file=sys.stderr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="segmenter.run_online", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", help="输出目录")
        p.add_argument("--rules", help="rules.yaml 路径（默认用包内的）")
        p.add_argument("--mode", choices=["node", "phase"], help="切分模式（默认读 rules.yaml）")
        p.add_argument("--base", type=int, default=1, help="事件流格式的起始步号（steps.json 用原生步号）")
        p.add_argument("--boundary-model", help="本地训练的受信任 joblib 模型；另写 supervised_stage_ranges.json")
        p.add_argument("--purpose-model", help="可选小目的模型；只写 purpose_events.jsonl，不改变默认阶段划分")
        p.add_argument("--task-file", help="小目的模型使用的任务文本；事件流没有任务文本时必需")

    p = sub.add_parser("replay", help="回放一条落盘轨迹")
    p.add_argument("path")
    p.add_argument("--print", action="store_true", help="逐步打印事件")
    p.add_argument("--delay", type=float, default=0.0, help="每步之间停几秒，模拟在线")
    common(p); p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("follow", help="跟读正在运行的 OpenHands 事件流")
    p.add_argument("path")
    p.add_argument("--poll", type=float, default=0.5)
    p.add_argument("--idle", type=float, default=600.0, help="这么多秒没有新事件就结束")
    common(p); p.set_defaults(fn=cmd_follow)

    p = sub.add_parser("bench", help="批量回放 CodeTraceBench 的 OpenHands 子集")
    p.add_argument("--manifest", required=True)
    p.add_argument("--artifacts", required=True, help="数据集下载目录（含 bench_artifacts/）或已解包目录")
    p.add_argument("--agent", default="OpenHands")
    p.add_argument("--cache", help="tar.zst 解包缓存目录")
    p.add_argument("--limit", type=int)
    common(p); p.set_defaults(fn=cmd_bench)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
