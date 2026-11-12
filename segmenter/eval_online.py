"""Boundary scoring which keeps late predictions in the precision denominator."""
import statistics
from .eval_segmenter import prf


def matched_pairs(events, gold, tol=1, delay=None):
    # Maximum one-to-one matching with both position and emission-time constraints.
    edges = []
    for e in events:
        b, t = e['boundary_step_id'], e['detected_at_step_id']
        if t < b:
            raise ValueError('A boundary cannot precede observation of its own step')
        edges.append([j for j, g in enumerate(gold)
                      if abs(b-g) <= tol and (delay is None or t <= g+delay)])
    owner = {}
    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False
    for i in range(len(events)):
        augment(i, set())
    return [(i, j) for j, i in owner.items()]


def counts(events, gold, tol=1, delay=None):
    return [len(matched_pairs(events, gold, tol, delay)), len(events), len(gold)]


def score(events, gold, tol=1, delay=None):
    c = counts(events, gold, tol, delay)
    return dict(zip(('precision','recall','f1'), prf(*c)), counts=c)


def aggregate(items, labels):
    out = {}
    for tol in (0, 1, 2):
        for delay in (None, 0, 1, 2):
            c, macros = [0,0,0], []
            for tid, events in items.items():
                gold = labels[tid]['boundaries']
                x = counts(events, gold, tol, delay)
                c = [a+b for a,b in zip(c,x)]
                macros.append(prf(*x)[2])
            key = f'tol{tol}_' + ('eventual' if delay is None else f'D{delay}')
            out[key] = dict(zip(('precision','recall','f1'),prf(*c)), counts=c,
                           macro_f1=statistics.fmean(macros) if macros else 0.)
    out['no_gold_boundary_trajectories'] = sum(not labels[t]['boundaries'] for t in items)
    out['predictions_on_no_gold'] = sum(len(e) for t,e in items.items() if not labels[t]['boundaries'])
    return out


def window_diff(pred, gold, ids, window):
    if len(ids) <= window:
        return None
    p, g = set(pred), set(gold)
    errors = 0
    for i in range(len(ids)-window):
        span = ids[i+1:i+window+1]
        errors += sum(x in p for x in span) != sum(x in g for x in span)
    return errors / (len(ids)-window)
