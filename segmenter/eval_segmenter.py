"""划分器打分（M1 的四个数字 + 事件召回）。

  # 数字 1-3 + 事件召回（CodeTraceBench，OpenHands 子集）
  python -m segmenter.eval_segmenter run --manifest codetrace_bench/bench_manifest.verified.jsonl \
      --artifacts codetrace_bench --out eval/verified --modes node,phase --export-stages eval/verified/stage_ranges

  # 别的数据集（AgentRx 等）：自备 gold jsonl，每行 {traj_id, traj_path, key_steps: [...], stages?: [...], base?: 1}
  python -m segmenter.eval_segmenter run --gold-jsonl agentrx_gold.jsonl --out eval/agentrx

  # 数字 4（下游影响）：CodeTracer 分别喂人标 stage 和规则 stage 跑完之后，比两份 codetracer_labels.json
  python -m segmenter.eval_segmenter downstream --manifest ... --gold-run outputs_gold/ --rule-run outputs_rule/

定义
  边界 F1      规则切点 vs 人标 stage 切点（除第一个外每个 stage 的起始步），±tol 步内一对一匹配
  错误步落点   人标 incorrect 步里，带 STATE_CHANGE 事件（即落在节点的改状态步上）的比例
  窗口代理     每个错误实例（每个出错 stage 的第一个 incorrect 步）到之后第一个 VERIFY_PASS / REFERENCE 的步数
               any = 任意一个；targeted = 指向该步所在节点的 REFERENCE，或放行了该节点内验证的 VERIFY_PASS
  事件召回     关键步之后 k 步内（s < idx <= s+k）是否有任一结构事件（REFERENCE / HANDOFF / VERIFY_PASS / IRREVERSIBLE）
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

from .adapters.openhands import load_trajectory
from .bench import iter_bench, read_manifest
from .events import E
from .rules import Rules, default_rules
from .run_online import run_steps
from .segment import cut_points

GATE_BOUNDARY_F1 = 0.70
GATE_EVENT_RECALL = 0.60
GATE_DOWNSTREAM = 5.0


# ------------------------------------------------------------------ 指标

def match_boundaries(pred: list[int], gold: list[int], tol: int = 1) -> int:
    """±tol 内的一对一贪心匹配，返回 TP。"""
    pred, gold = sorted(pred), sorted(gold)
    i = j = tp = 0
    while i < len(pred) and j < len(gold):
        if abs(pred[i] - gold[j]) <= tol:
            tp += 1; i += 1; j += 1
        elif pred[i] < gold[j]:
            i += 1
        else:
            j += 1
    return tp


def prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def window_proxy(rows: list[dict], step_node: dict, s: int) -> tuple[Optional[int], Optional[int]]:
    """(any, targeted) 距离；找不到为 None（窗口没有被任何结构事件关上）。"""
    node = step_node.get(s)
    d_any = d_tgt = None
    for r in rows:
        if r["idx"] <= s:
            continue
        ev = r["events"]
        if d_any is None and (E.VERIFY_PASS in ev or E.REFERENCE in ev):
            d_any = r["idx"] - s
        if d_tgt is None and node is not None:
            if (E.REFERENCE in ev and node in r["refs"]) or \
               (E.VERIFY_PASS in ev and step_node.get(r["for_idx"]) == node):
                d_tgt = r["idx"] - s
        if d_any is not None and (d_tgt is not None or node is None):
            break
    return d_any, d_tgt


def dist_summary(ds: list[Optional[int]]) -> dict:
    found = sorted(d for d in ds if d is not None)
    if not ds:
        return {"n": 0}
    out = {"n": len(ds), "closed": len(found), "never_closed_rate": round(1 - len(found) / len(ds), 4)}
    if found:
        q = lambda p: found[min(len(found) - 1, int(p * len(found)))]
        out.update({"median": statistics.median(found), "mean": round(statistics.fmean(found), 2),
                    "p25": q(0.25), "p75": q(0.75), "p90": q(0.90),
                    "share_eq_1": round(sum(d == 1 for d in found) / len(found), 4),
                    "share_ge_3": round(sum(d >= 3 for d in found) / len(found), 4),
                    "hist": {str(k): v for k, v in sorted(Counter(min(d, 21) for d in found).items())}})
    return out


# ------------------------------------------------------------------ gold 来源

def iter_gold_jsonl(path: str) -> Iterator[dict]:
    for row in read_manifest(path):
        keys = sorted(int(x) for x in row.get("key_steps") or row.get("incorrect_steps") or [])
        yield {"traj_id": str(row["traj_id"]), "traj_path": row["traj_path"], "stages": row.get("stages") or [],
               "incorrect_steps": keys, "instances": keys, "base": int(row.get("base", 1)),
               "step_count": row.get("step_count"), "solved": row.get("solved")}


# ------------------------------------------------------------------ 主流程

def evaluate(entries: Iterator[dict], rules: Rules, modes: list[str], tol: int = 1, recall_k: int = 3,
             recall_include_self: bool = False, keep_misaligned: bool = False,
             export_stages: Optional[Path] = None, dump_dir: Optional[Path] = None,
             recall_ks=(3, 6, 10), _grouped=False) -> dict:
    recall_ks = sorted(set([recall_k, *recall_ks]))
    if any(k < 1 for k in recall_ks):
        raise ValueError("recall windows must be positive")
    groups = {}
    curve = {k: {name: [0, 0] for name in ("instances", "incorrect_steps")} for k in recall_ks}
    per_traj, win_rows = [], []
    micro = {m: [0, 0, 0] for m in modes}            # tp, n_pred, n_gold
    uni = [0, 0, 0]                                  # 参照：知道 stage 数的等距切分
    macro = {m: [] for m in modes}
    land = Counter(); land_tc = Counter()
    rec = {"instances": [0, 0], "incorrect_steps": [0, 0]}
    ev_counts = Counter(); n_steps_total = 0; ms_all = []
    skipped = Counter()

    import time
    t_start, n_seen = time.time(), 0
    for e in entries:
        tid = e["traj_id"]
        n_seen += 1
        print(f"\r[{n_seen}] {time.time() - t_start:6.0f}s  {tid[-60:]:<60}", end="", file=sys.stderr, flush=True)
        if e.get("error"):
            skipped["not_found"] += 1
            print(f"[skip] {tid}: {e['error']}", file=sys.stderr)
            continue
        try:
            steps, meta = e.get("_loaded") or load_trajectory(e["traj_path"], rules, base=e["base"], expect_steps=e.get("step_count"))
        except Exception as ex:  # noqa: BLE001
            skipped["load_error"] += 1
            print(f"[skip] {tid}: {type(ex).__name__}: {ex}", file=sys.stderr)
            continue
        n_core = meta.get("n_core", len(steps))
        aligned = e.get("step_count") is None or n_core == int(e["step_count"])
        if not aligned and not keep_misaligned:
            skipped["misaligned"] += 1
            print(f"[skip] {tid}: 转出 {n_core} 步，manifest 说 {e['step_count']} 步（步号对不上）", file=sys.stderr)
            continue
        source = e.get("source") or source_family(e)
        res = e.get("_result") or run_steps(steps, rules, (dump_dir / tid) if dump_dir else None, modes)
        if not _grouped:
            groups.setdefault(source, []).append(dict(e, _loaded=(steps, meta), _result=res))
        rows = res["rows"]
        step_node = {r["idx"]: r["node"] for r in rows}
        by_idx = {r["idx"]: r for r in rows}
        n_steps_total += len(rows)
        ms_all += [r["ms"] for r in rows]
        for r in rows:
            ev_counts.update(r["events"])

        rec_t = {"traj_id": tid, "source": source, "n_steps": len(steps), "format": meta["format"], "aligned": aligned,
                 "solved": e.get("solved"), "n_nodes": len(res["tree"]["nodes"]),
                 "n_incorrect": len(e["incorrect_steps"])}
        # 1. 边界 F1
        if e["stages"]:
            gold = cut_points(e["stages"])
            k = max(1, round(len(steps) / len(e["stages"])))
            upred = list(range(e["base"] + k, e["base"] + len(steps), k))
            uni[0] += match_boundaries(upred, gold, tol); uni[1] += len(upred); uni[2] += len(gold)
            for m in modes:
                pred = cut_points(res["stages"][m])
                tp = match_boundaries(pred, gold, tol)
                micro[m][0] += tp; micro[m][1] += len(pred); micro[m][2] += len(gold)
                p, r_, f = prf(tp, len(pred), len(gold))
                if gold:
                    macro[m].append(f)
                rec_t.update({f"{m}_pred_cuts": len(pred), f"{m}_gold_cuts": len(gold), f"{m}_f1": round(f, 4)})
            if export_stages:
                d = export_stages / tid
                d.mkdir(parents=True, exist_ok=True)
                clean = [{k: s[k] for k in ("stage_id", "start_step_id", "end_step_id")} for s in res["stages"][modes[0]]]
                (d / "stage_ranges.json").write_text(json.dumps(clean, indent=1), encoding="utf-8")
        # 2. 错误步落点
        for s in e["incorrect_steps"]:
            r = by_idx.get(s)
            if r is None:
                land["missing"] += 1
                continue
            land["total"] += 1
            land["state"] += E.STATE_CHANGE in r["events"]
            land["verify"] += E.VERIFY_RUN in r["events"] and E.STATE_CHANGE not in r["events"]
            land_tc[r["tool_class"]] += 1
        # 3. 窗口代理 + 事件召回
        lo = 0 if recall_include_self else 1
        for name, keys in (("instances", e["instances"]), ("incorrect_steps", e["incorrect_steps"])):
            for s in keys:
                if s not in by_idx:
                    continue
                hit = any(set(by_idx[i]["events"]) & E.STRUCTURAL for i in range(s + lo, s + recall_k + 1) if i in by_idx)
                rec[name][0] += hit; rec[name][1] += 1
                for k in recall_ks:
                    khit = any(set(by_idx[i]["events"]) & E.STRUCTURAL
                               for i in range(s + lo, s + k + 1) if i in by_idx)
                    curve[k][name][0] += khit
                    curve[k][name][1] += 1
                if name == "instances":
                    d_any, d_tgt = window_proxy(rows, step_node, s)
                    win_rows.append({"traj_id": tid, "source": source, "step": s, "solved": e.get("solved"),
                                     "d_any": d_any, "d_targeted": d_tgt, "recall_hit": int(hit),
                                     "tool_class": by_idx[s]["tool_class"]})
        per_traj.append(rec_t)

    print(file=sys.stderr)
    summary = {
        "n_traj": len(per_traj), "skipped": dict(skipped), "n_steps": n_steps_total, "tol": tol,
        "boundary": {m: dict(zip(("precision", "recall", "f1"), (round(x, 4) for x in prf(*micro[m]))),
                         macro_f1=round(statistics.fmean(macro[m]), 4) if macro[m] else None,
                         n_pred=micro[m][1], n_gold=micro[m][2]) for m in modes},
        "boundary_uniform_baseline": dict(zip(("precision", "recall", "f1"), (round(x, 4) for x in prf(*uni)))),
        "error_landing": {"n": land["total"], "missing": land["missing"],
                          "state_change_rate": round(land["state"] / land["total"], 4) if land["total"] else None,
                          "verify_rate": round(land["verify"] / land["total"], 4) if land["total"] else None,
                          "by_tool_class": dict(land_tc.most_common())},
        "window_proxy": {"any": dist_summary([w["d_any"] for w in win_rows]),
                         "targeted": dist_summary([w["d_targeted"] for w in win_rows]),
                         "faap_reference": {"median": 1, "share_ge_3": 0.44}},
        "event_recall": {k: {"n": v[1], "recall": round(v[0] / v[1], 4) if v[1] else None} for k, v in rec.items()},
        "recall_k": recall_k,
        "event_rate_per_step": {k: round(v / n_steps_total, 4) for k, v in ev_counts.most_common()} if n_steps_total else {},
        "per_step_ms": {"median": statistics.median(ms_all), "p99": sorted(ms_all)[int(0.99 * (len(ms_all) - 1))]} if ms_all else {},
    }
    best = max((summary["boundary"][m]["f1"] for m in modes), default=0.0)
    r_inst = summary["event_recall"]["instances"]["recall"]
    summary["gates"] = {
        "boundary_f1>=0.70": None if not any(micro[m][2] for m in modes) else bool(best >= GATE_BOUNDARY_F1),
        "event_recall>=0.60": None if r_inst is None else bool(r_inst >= GATE_EVENT_RECALL),
    }
    summary["recall_include_self"] = recall_include_self
    summary["event_recall_curve"] = {str(k): {name: {"hits": v[0], "n": v[1],
        "recall": round(v[0] / v[1], 4) if v[1] else None} for name, v in counts.items()}
        for k, counts in curve.items()}
    if not _grouped:
        summary["by_source"] = {source: evaluate(iter(es), rules, modes, tol, recall_k,
            recall_include_self, keep_misaligned, recall_ks=recall_ks, _grouped=True)["summary"]
            for source, es in sorted(groups.items())}
    return {"summary": summary, "per_traj": per_traj, "window": win_rows}


def source_family(entry: dict) -> str:
    path = str(entry.get("source_relpath") or "").replace("\\", "/").lower()
    if path.startswith("swe_raw/"):
        return "SWE"
    if "terminal" in path or path.startswith("openhands/"):
        return "Terminal-Bench"
    return "unknown"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _figure(win_rows: list[dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, key, title in zip(axes, ("d_any", "d_targeted"), ("any VERIFY_PASS / REFERENCE", "targeted at the error's node")):
        ds = [min(w[key], 21) for w in win_rows if w[key] is not None]
        ax.hist(ds, bins=[x - 0.5 for x in range(1, 23)], color="#4477aa")
        if ds:
            ax.axvline(statistics.median(ds), color="#cc3311", lw=1, label=f"median = {statistics.median(ds):g}")
            ax.legend(frameon=False)
        ax.set_title(f"{title} (n = {len(ds)})", fontsize=9)
        ax.set_xlabel("steps from incorrect step to first acceptance event (21 = 21+)")
    axes[0].set_ylabel("error instances")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def print_summary(s: dict) -> None:
    g = lambda ok: "—" if ok is None else ("PASS" if ok else "FAIL")
    print(f"\n轨迹 {s['n_traj']} 条，{s['n_steps']} 步；跳过 {s['skipped'] or 0}")
    print(f"\n[1] 边界 F1（±{s['tol']} 步；门禁 ≥ {GATE_BOUNDARY_F1}）  {g(s['gates']['boundary_f1>=0.70'])}")
    for m, b in s["boundary"].items():
        print(f"    {m:<6} P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f}  macro-F1={b['macro_f1']}  (pred {b['n_pred']} / gold {b['n_gold']})")
    u = s["boundary_uniform_baseline"]
    print(f"    等距切分参照（已知 stage 数）  P={u['precision']:.3f} R={u['recall']:.3f} F1={u['f1']:.3f}")
    el = s["error_landing"]
    print(f"\n[2] 错误步落点（预期 > 0.70）  改状态步占比 = {el['state_change_rate']}  纯验证步 = {el['verify_rate']}  n = {el['n']}"
          + (f"  (!! {el['missing']} 个标注步号不在轨迹里)" if el["missing"] else ""))
    print(f"    按 tool_class：{el['by_tool_class']}")
    print("\n[3] 窗口代理分布（FaaP：中位 1，44% ≥ 3）")
    for k in ("any", "targeted"):
        w = s["window_proxy"][k]
        if w.get("closed"):
            print(f"    {k:<9} n={w['n']} 中位={w['median']} 均值={w['mean']} =1: {w['share_eq_1']:.0%}  ≥3: {w['share_ge_3']:.0%}  p90={w['p90']}  从未关上: {w['never_closed_rate']:.0%}")
        else:
            print(f"    {k:<9} n={w.get('n', 0)}（没有可统计的实例）")
    er = s["event_recall"]
    print(f"\n[+] 事件召回（关键步后 {s['recall_k']} 步内有结构事件；门禁 ≥ {GATE_EVENT_RECALL}）  {g(s['gates']['event_recall>=0.60'])}")
    print(f"    按错误实例 = {er['instances']['recall']} (n={er['instances']['n']})   按 incorrect 步 = {er['incorrect_steps']['recall']} (n={er['incorrect_steps']['n']})")
    print(f"\n事件密度（每步）：{s['event_rate_per_step']}")
    print(f"每步耗时 ms：{s['per_step_ms']}")
    print("\n[4] 下游影响：见 `eval_segmenter downstream`（需要先用 CodeTracer 各跑一遍）")


def cmd_run(a) -> None:
    overrides = dict(kv.split("=", 1) for kv in a.set or [])
    overrides = {k: json.loads(v) if v[:1] in "[{0123456789tf-" else v for k, v in overrides.items()}
    rules = Rules(a.rules, overrides) if (a.rules or overrides) else default_rules()
    modes = a.modes.split(",")
    if a.gold_jsonl:
        entries = iter_gold_jsonl(a.gold_jsonl)
    elif a.manifest and a.artifacts:
        entries = iter_bench(a.manifest, a.artifacts, agent=a.agent or None, cache=a.cache, limit=a.limit)
    else:
        sys.exit("需要 --manifest + --artifacts，或 --gold-jsonl")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = evaluate(entries, rules, modes, tol=a.tol, recall_k=a.recall_k, recall_include_self=a.recall_include_self,
                   keep_misaligned=a.keep_misaligned,
                   export_stages=Path(a.export_stages) if a.export_stages else None,
                   dump_dir=(out / "traj") if a.dump else None)
    (out / "summary.json").write_text(json.dumps(res["summary"], ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "rules.json").write_text(json.dumps(rules.raw, ensure_ascii=False, indent=2), encoding="utf-8")
    curve_rows = []
    for source, summary in {"all": res["summary"], **res["summary"]["by_source"]}.items():
        for k, counts in summary["event_recall_curve"].items():
            for unit, stats in counts.items():
                curve_rows.append(dict(source=source, k=int(k), unit=unit, **stats))
    _write_csv(out / "event_recall_curve.csv", curve_rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for source in dict.fromkeys(r["source"] for r in curve_rows):
            rs = [r for r in curve_rows if r["source"] == source and r["unit"] == "instances"]
            ax.plot([r["k"] for r in rs], [r["recall"] for r in rs], marker="o", label=source)
        ax.set(xlabel="Steps after error (k)", ylabel="Structural-event proxy recall", ylim=(0, 1))
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "event_recall_curve.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    _write_csv(out / "per_traj.csv", res["per_traj"])
    _write_csv(out / "window_proxy.csv", res["window"])
    print_summary(res["summary"])
    if res["window"] and _figure(res["window"], out / "window_proxy.png"):
        print(f"\n图：{out / 'window_proxy.png'}")
    print(f"结果：{out}/summary.json  per_traj.csv  window_proxy.csv")


def _label_steps(d: Path, tid: str) -> Optional[set]:
    f = d / tid / "codetracer_labels.json"
    if not f.exists():
        return None
    labels = json.loads(f.read_text(encoding="utf-8"))
    return {int(s) for st in labels for s in st.get("incorrect_step_ids", [])}


def cmd_downstream(a) -> None:
    """数字 4：CodeTracer 喂人标 stage_ranges vs 喂规则 stage_ranges，步级 F1 之差。"""
    from .bench import parse_entry
    agg = {"gold": [0, 0, 0, []], "rule": [0, 0, 0, []]}
    n = 0
    missing = []
    expected = 0
    for row in read_manifest(a.manifest):
        if a.agent and str(row.get("agent", "")).lower() != a.agent.lower():
            continue
        e = parse_entry(row)
        expected += 1
        gt = set(e["incorrect_steps"])
        preds = {"gold": _label_steps(Path(a.gold_run), e["traj_id"]), "rule": _label_steps(Path(a.rule_run), e["traj_id"])}
        if preds["gold"] is None or preds["rule"] is None:
            missing.append({"traj_id": e['traj_id'], "missing_arms": [k for k,v in preds.items() if v is None]})
            continue
        n += 1
        for k, pred in preds.items():
            tp = len(pred & gt)
            agg[k][0] += tp; agg[k][1] += len(pred); agg[k][2] += len(gt)
            agg[k][3].append(prf(tp, len(pred), len(gt))[2])
    if not n:
        report = {'expected': expected, 'paired': 0, 'missing': missing, 'gate_pass': None}
        if getattr(a, 'out', None):
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        sys.exit("No paired labels; no downstream score or gate is available.")
    out = {}
    for k, (tp, npred, ngold, f1s) in agg.items():
        p, r, f = prf(tp, npred, ngold)
        out[k] = {"micro_p": round(100 * p, 2), "micro_r": round(100 * r, 2), "micro_f1": round(100 * f, 2),
                  "macro_f1": round(100 * statistics.fmean(f1s), 2)}
        print(f"{k:<5} P={out[k]['micro_p']} R={out[k]['micro_r']} F1={out[k]['micro_f1']}  macro-F1={out[k]['macro_f1']}")
    diff = 100 * (statistics.fmean(agg['gold'][3]) - statistics.fmean(agg['rule'][3]))
    gate = bool(diff <= GATE_DOWNSTREAM) if n == expected else None
    report = dict(expected=expected, paired=n, missing=missing, metrics=out,
                  macro_f1_difference_points=diff, gate_pass=gate)
    if getattr(a, 'out', None):
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Paired {n}/{expected}; macro F1 difference = {diff:.4f} points; gate={gate}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="segmenter.eval_segmenter", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="边界 F1 / 错误步落点 / 窗口代理 / 事件召回")
    p.add_argument("--manifest"); p.add_argument("--artifacts"); p.add_argument("--gold-jsonl")
    p.add_argument("--agent", default="OpenHands")
    p.add_argument("--out", required=True)
    p.add_argument("--rules"); p.add_argument("--cache"); p.add_argument("--limit", type=int)
    p.add_argument("--modes", default="node,phase", help="要比较的切分模式；第一个是主模式")
    p.add_argument("--tol", type=int, default=1)
    p.add_argument("--recall-k", type=int, default=3)
    p.add_argument("--recall-include-self", action="store_true")
    p.add_argument("--keep-misaligned", action="store_true", help="步数与 manifest 不一致的轨迹也算")
    p.add_argument("--export-stages", help="把主模式的 stage_ranges.json 按 <traj_id>/ 导出，喂给 CodeTracer")
    p.add_argument("--dump", action="store_true", help="每条轨迹的 events / tree / claims 也落盘")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="临时覆盖 rules.yaml，如 --set tree.explore_gap=2")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("downstream", help="CodeTracer 下游步级 F1 差")
    p.add_argument("--manifest", required=True); p.add_argument("--gold-run", required=True)
    p.add_argument("--rule-run", required=True); p.add_argument("--agent", default="OpenHands")
    p.add_argument("--out", help="Save paired coverage, missing runs and downstream metrics as JSON")
    p.set_defaults(fn=cmd_downstream)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
