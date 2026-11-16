"""Check frozen prefix provenance and future references; no model evaluation."""
from collections import Counter
from pathlib import Path
from .build_future_diagnostic import read, sha, save


def main():
    root=Path('pilot/swe_future_diagnostic_v1')
    protocol=read(root/'protocol.json')
    for path,digest in protocol['frozen_hashes'].items():
        assert sha(path)==digest,'Original prefix artifact changed'
    before=read(root/'prefix_review_frozen.json')
    assert before==read('pilot/swe_relation_balanced_v2/ai_review.json')
    prior={r['case_id']:r for r in before['cases']}
    packets={r['case_id']:r for r in read(root/'future_packets.json')}
    reviews=read(root/'ai_review.json')['cases']
    assert len(reviews)==len(prior)==len(packets)==16
    assert {r['case_id'] for r in reviews}==set(prior)==set(packets)
    for r in reviews:
        p=packets[r['case_id']];ids=[s['idx'] for s in p['future']]
        assert r['category'] in protocol['categories']
        assert p['offline_only'] and len(ids)<=3 and len(ids)==p['available_future_steps']
        assert all(i>p['target_step'] for i in ids) and ids==sorted(set(ids))
        assert set(r['future_evidence_steps'])<=set(ids)
        assert r['after_suggestion'] in ('undetermined','continue_plausible','switch_plausible')
    result=dict(n=16,checks_passed=True,categories=dict(Counter(r['category'] for r in reviews)),
        prefix_undetermined=sum(r['suggestion']=='undetermined' for r in prior.values()),
        after_undetermined=sum(r['after_suggestion']=='undetermined' for r in reviews),
        fewer_than_three_future_actions=sum(p['available_future_steps']<3 for p in packets.values()),
        caveat='Counts describe selected AI-reviewed excerpts, not online accuracy, a performance ceiling, or proof of future necessity.')
    save(root/'audit.json',result);print(result)


if __name__=='__main__':main()
