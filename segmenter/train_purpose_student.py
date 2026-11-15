"""Train a small purpose student on 60 AI-reviewed cases; grouped CV only."""
import argparse
import gzip
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

from .ledger import atomic, sha
from .purpose_probe import LABELS, trajectories
from .purpose_student import FEATURE_VERSION, PurposeStudent, feature_text


def main():
    import numpy as np
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.model_selection import GroupKFold

    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--review',type=Path,default=Path('pilot/purpose_human_audit_v1'))
    ap.add_argument('--out',type=Path,default=Path('eval/purpose_student_v1'))
    args=ap.parse_args()
    if args.out.exists():
        raise SystemExit('Use a fresh output directory')
    packets=json.loads((args.review/'blind_packets.json').read_text(encoding='utf-8'))
    selection=json.loads((args.review/'selection.json').read_text(encoding='utf-8'))
    review=json.loads((args.review/'ai_review.json').read_text(encoding='utf-8'))
    draft_path=args.review/'ai_draft.json'
    draft=json.loads(draft_path.read_text(encoding='utf-8'))
    if selection['blind_packets_sha256']!=sha((args.review/'blind_packets.json').read_bytes()):
        raise ValueError('Blind packet changed')
    if review.get('reviewer_kind')!='ai' or review.get('draft_sha256')!=sha(draft_path.read_bytes()):
        raise ValueError('Expected the audited AI review and unchanged draft')
    ids=[x['request_id'] for x in packets]
    if len(ids)!=len(set(ids)) or set(ids)!=set(review['labels']) or set(ids)!=set(draft['suggestions']):
        raise ValueError('Review/packet ID mismatch')
    for rid in ids:
        row=review['labels'][rid]
        if row.get('reviewed') is not True or row.get('suggested_label')!=draft['suggestions'][rid]['label']:
            raise ValueError('Unreviewed/altered AI suggestion: '+rid)
        if row.get('final_label') not in LABELS:
            raise ValueError('Invalid label: '+rid)
    texts=[feature_text(x['packet']['task'],x['packet']['steps']) for x in packets]
    gold=[review['labels'][rid]['final_label'] for rid in ids]
    groups=[x['packet']['trajectory_id'] for x in packets]
    sources=[x['source'] for x in packets]
    if len(set(groups))<5:
        raise ValueError('Need at least five trajectory groups')
    def model():
        return TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=8000,
                               sublinear_tf=True), LogisticRegression(C=1.,max_iter=500,
                                                                        class_weight='balanced')
    oof=[None]*len(ids)
    folds=[]
    for fold,(train,test) in enumerate(GroupKFold(n_splits=5).split(texts,gold,groups)):
        vectorizer,clf=model()
        xtrain=vectorizer.fit_transform([texts[i] for i in train])
        clf.fit(xtrain,[gold[i] for i in train])
        predicted=clf.predict(vectorizer.transform([texts[i] for i in test]))
        for i,value in zip(test,predicted):
            oof[i]=str(value)
        folds.append(dict(fold=fold,train_trajectories=sorted(set(groups[i] for i in train)),
                          test_trajectories=sorted(set(groups[i] for i in test)),
                          train_n=len(train),test_n=len(test),train_classes=sorted(set(gold[i] for i in train))))
    if any(x is None for x in oof):
        raise AssertionError('Missing out-of-fold prediction')
    present=[x for x in LABELS if x in gold]
    def metrics(indices,predictions):
        y=[gold[i] for i in indices];p=[predictions[i] for i in indices]
        return dict(n=len(y),accuracy=float(accuracy_score(y,p)),
                    macro_f1=float(f1_score(y,p,labels=present,average='macro',zero_division=0)),
                    per_class_recall={label:(sum(a==b==label for a,b in zip(y,p))/sum(a==label for a in y)
                                             if label in y else None) for label in present})
    all_indices=list(range(len(ids)))
    majority=[]
    for fold in folds:
        train_labels=[gold[i] for i in all_indices if groups[i] in fold['train_trajectories']]
        base=Counter(train_labels).most_common(1)[0][0]
        majority.extend((i,base) for i in all_indices if groups[i] in fold['test_trajectories'])
    majority_pred=[None]*len(ids)
    for i,label in majority:majority_pred[i]=label

    args.out.mkdir(parents=True)
    vectorizer,clf=model()
    matrix=vectorizer.fit_transform(texts)
    clf.fit(matrix,gold)
    artifact=dict(feature_version=FEATURE_VERSION,classes=[str(x) for x in clf.classes_],
                  vocabulary={k:int(v) for k,v in vectorizer.vocabulary_.items()},
                  idf=vectorizer.idf_.tolist(),coef=clf.coef_.tolist(),
                  intercept=clf.intercept_.tolist(),ngram_range=[3,5],sublinear_tf=True,
                  label_source='AI review of AI draft; not human gold',
                  review_sha256=sha((args.review/'ai_review.json').read_bytes()),
                  code_sha256=sha(Path(__file__).read_bytes()))
    model_path=args.out/'model.json.gz'
    with gzip.open(model_path,'wt',encoding='utf-8',compresslevel=9) as f:
        json.dump(artifact,f,ensure_ascii=False,separators=(',',':'))
    student=PurposeStudent.load(model_path)
    # Verify standard-library runtime exactly reproduces the trained sklearn model.
    reference=clf.predict_proba(matrix)
    max_error=max(abs(student.probabilities_text(text)[str(label)]-reference[i,j])
                  for i,text in enumerate(texts) for j,label in enumerate(clf.classes_))
    if max_error>1e-9:
        raise AssertionError(f'Exported runtime differs from sklearn: {max_error}')
    ts=trajectories();pilot_predictions={}
    times=[]
    for t in ts:
        output=[]
        for j,step in enumerate(t['steps']):
            row=student.predict(t['task'],t['steps'][:j+1])
            times.append(row['ms'])
            output.append(dict(step_id=step['idx'],purpose=row['purpose']))
        pilot_predictions[t['trajectory_id']]=output
    atomic(args.out/'predictions_735.json',pilot_predictions)
    atomic(args.out/'oof_predictions.json',[dict(request_id=rid,group=group,source=source,
                                                ai_reference=label,prediction=predicted)
                                             for rid,group,source,label,predicted in zip(ids,groups,sources,gold,oof)])
    report=dict(protocol='purpose-student-ai60-v1',label_provenance='AI-reviewed AI draft, no human gold',
                n=len(ids),groups=len(set(groups)),gold_counts=dict(Counter(gold)),
                unlearned_classes=[x for x in LABELS if x not in gold],
                grouped_oof=metrics(all_indices,oof),grouped_majority=metrics(all_indices,majority_pred),
                by_source={s:metrics([i for i,x in enumerate(sources) if x==s],oof)
                           for s in sorted(set(sources))},folds=folds,
                confusion_labels=present,
                confusion_matrix=confusion_matrix(gold,oof,labels=present).tolist(),
                artifact_bytes=model_path.stat().st_size,
                artifact_sha256=sha(model_path.read_bytes()),
                exported_runtime_max_probability_error=max_error,
                inference_ms=dict(n=len(times),median=statistics.median(times),
                                  p95=float(np.quantile(times,.95)),max=max(times)),
                full_pilot_prediction_counts=dict(Counter(x['purpose'] for v in pilot_predictions.values() for x in v)),
                versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__),
                limitations=['60 AI reference labels only; no human accuracy',
                             'Group-held-out purpose evaluation is not a boundary segmentation evaluation',
                             'No DELIVER training examples; model cannot predict that class',
                             'Sample was stratified by source and Qwen predictions, not population-representative',
                             'Final model trained on all 60; 735 predictions are in-sample for those 60'])
    atomic(args.out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('grouped_oof','grouped_majority','artifact_bytes','inference_ms')},
                     ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
