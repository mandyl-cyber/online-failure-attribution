"""Join the AI purpose review to held-out boundary predictions for diagnosis.

This is an exploratory AI-to-AI comparison, never a human accuracy estimate.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from .eval_online import matched_pairs
from .ledger import atomic, encoded, sha
from .purpose_probe import LABELS


def analyze(review_dir, evaluation_dir, probe_dir, data_dir, out_dir):
    review_dir, evaluation_dir = Path(review_dir), Path(evaluation_dir)
    probe_dir, data_dir, out_dir = Path(probe_dir), Path(data_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError('Use a fresh output directory')
    draft_path=review_dir/'ai_draft.json'
    draft=json.loads(draft_path.read_text(encoding='utf-8'))
    review=json.loads((review_dir/'ai_review.json').read_text(encoding='utf-8'))
    packets=json.loads((review_dir/'blind_packets.json').read_text(encoding='utf-8'))
    selection=json.loads((review_dir/'selection.json').read_text(encoding='utf-8'))
    if review.get('reviewer_kind')!='ai':
        raise ValueError('This analysis is specifically for the completed AI review')
    if review.get('draft_sha256')!=sha(draft_path.read_bytes()):
        raise ValueError('AI draft changed')
    if selection['blind_packets_sha256']!=sha((review_dir/'blind_packets.json').read_bytes()):
        raise ValueError('Blind packet changed')
    ids=[x['request_id'] for x in packets]
    if len(ids)!=len(set(ids)) or set(ids)!=set(review['labels']) or set(ids)!=set(draft['suggestions']):
        raise ValueError('Review and packet IDs differ')
    probe_identity=json.loads((probe_dir/'manifest.json').read_text())['identity']
    if selection['probe_identity']!=probe_identity:
        raise ValueError('Probe identity changed')
    evaluation=json.loads((evaluation_dir/'report.json').read_text(encoding='utf-8'))
    if evaluation['probe_identity']!=probe_identity:
        raise ValueError('Evaluation uses a different probe')
    mapping=json.loads((evaluation_dir/'mapping.json').read_text(encoding='utf-8'))
    original={m['pilot_id']:m['match']['traj_id'] for m in mapping if m['match']}
    predictions=json.loads((evaluation_dir/'predictions.json').read_text(encoding='utf-8'))
    gold=json.loads((data_dir/'labels.json').read_text(encoding='utf-8'))
    if set(predictions['fusion'])!=set(predictions['structure']) or set(predictions['fusion'])!=set(original.values()):
        raise ValueError('Prediction trajectory set does not match mapping')
    status={}
    global_counts=Counter()
    for tid in predictions['fusion']:
        boundaries=gold[tid]['boundaries']
        fusion=predictions['fusion'][tid];structure=predictions['structure'][tid]
        fset={e['boundary_step_id'] for e in fusion}
        sset={e['boundary_step_id'] for e in structure}
        if len(fset)!=len(fusion) or len(sset)!=len(structure):
            raise ValueError('Duplicate predicted boundary')
        fmatched={fusion[i]['boundary_step_id'] for i,_ in matched_pairs(fusion,boundaries,1,0)}
        smatched={structure[i]['boundary_step_id'] for i,_ in matched_pairs(structure,boundaries,1,0)}
        for step in fset|sset:
            group=('both' if step in fset and step in sset else
                   'fusion_only' if step in fset else 'structure_only')
            global_counts[group]+=1
            if step in fset:
                global_counts[group+'_fusion_matched' if step in fmatched else group+'_fusion_unmatched']+=1
            if step in sset:
                global_counts[group+'_structure_matched' if step in smatched else group+'_structure_unmatched']+=1
            status[(tid,step)]=dict(group=group,fusion_matched=step in fmatched,
                                    structure_matched=step in smatched)
    f=evaluation['methods']['fusion']['tol1_D0']['counts']
    s=evaluation['methods']['structure']['tol1_D0']['counts']
    if [global_counts['both_fusion_matched']+global_counts['fusion_only_fusion_matched'],
        global_counts['both']+global_counts['fusion_only'],f[2]]!=f:
        raise ValueError('Fusion counts differ from evaluation')
    if [global_counts['both_structure_matched']+global_counts['structure_only_structure_matched'],
        global_counts['both']+global_counts['structure_only'],s[2]]!=s:
        raise ValueError('Structure counts differ from evaluation')
    rows=[]
    for packet in packets:
        rid=packet['request_id'];item=review['labels'][rid]
        if item.get('reviewed') is not True or item.get('suggested_label')!=draft['suggestions'][rid]['label']:
            raise ValueError('Incomplete/altered AI review: '+rid)
        label=item.get('final_label')
        if label not in LABELS:
            raise ValueError('Invalid AI review label: '+rid)
        response=json.loads((probe_dir/'responses'/f'{rid}.json').read_text(encoding='utf-8'))
        if response['status']!='scored' or response['input_sha256']!=sha(encoded(packet['packet'])):
            raise ValueError('Probe response mismatch: '+rid)
        pilot,step=rid.split('_s');tid=original[pilot];step=int(step)
        x=status.get((tid,step),dict(group='neither',fusion_matched=False,structure_matched=False))
        rows.append(dict(request_id=rid,source=packet['source'],qwen=response['purpose'],
                         ai_review=label,agree=response['purpose']==label,**x))
    def summarize(items):
        return dict(n=len(items),qwen_ai_disagree=sum(not x['agree'] for x in items),
                    qwen_ai_agree=sum(x['agree'] for x in items),
                    fusion_matched=sum(x['fusion_matched'] for x in items))
    sampled={name:summarize([x for x in rows if x['group']==name])
             for name in ('fusion_only','structure_only','both','neither')}
    new_fp=[x for x in rows if x['group']=='fusion_only' and not x['fusion_matched']]
    sampled_fp=[x for x in rows if x['group'] in ('fusion_only','both') and not x['fusion_matched']]
    report=dict(protocol='purpose-ai-review-boundary-diagnostic-v1',
        reference_provenance='AI review of AI draft, not human gold',
        global_prediction_counts=dict(global_counts),evaluation_fusion_counts=f,
        evaluation_structure_counts=s,sampled=summarize(rows),sampled_groups=sampled,
        sampled_fusion_unmatched=summarize(sampled_fp),
        sampled_fusion_only_unmatched=summarize(new_fp),
        sampled_fusion_only_unmatched_ids=[x['request_id'] for x in new_fp],
        top_disagreements=[dict(ai_review=a,qwen=q,n=n) for (a,q),n in
                           Counter((x['ai_review'],x['qwen']) for x in rows if not x['agree']).most_common(10)],
        review_sha256=sha((review_dir/'ai_review.json').read_bytes()),
        prediction_sha256=sha((evaluation_dir/'predictions.json').read_bytes()),
        limitations=['Sample was stratified by source and Qwen label, not selected for boundary error analysis',
                     'Only sampled steps have AI review labels; three sampled fusion-only false positives cannot explain all 60',
                     'Predicted-boundary matching uses the same one-to-one tolerance ±1 D0 rule as evaluation',
                     'AI-to-AI disagreement cannot establish human purpose accuracy or causal mechanism'])
    out_dir.mkdir(parents=True)
    atomic(out_dir/'report.json',report)
    atomic(out_dir/'sample_rows.json',rows)
    print(json.dumps({k:report[k] for k in ('global_prediction_counts','sampled','sampled_fusion_only_unmatched')},
                     ensure_ascii=False,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--review',type=Path,default=Path('pilot/purpose_human_audit_v1'))
    ap.add_argument('--evaluation',type=Path,default=Path('eval/purpose_probe_evaluation_v2'))
    ap.add_argument('--probe',type=Path,default=Path('eval/purpose_probe_v1'))
    ap.add_argument('--data',type=Path,default=Path('eval/boundary_supervised/data_fixed'))
    ap.add_argument('--out',type=Path,default=Path('eval/purpose_ai_boundary_diagnostic_v1'))
    args=ap.parse_args()
    analyze(args.review,args.evaluation,args.probe,args.data,args.out)


if __name__=='__main__':
    main()
