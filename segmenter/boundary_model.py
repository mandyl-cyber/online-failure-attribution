"""Sparse logistic boundary model and zero-lookahead streaming inference."""
import importlib.util
import sys
import time
from pathlib import Path

if importlib.util.find_spec('sklearn') is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'vendor/boundary_deps'))

import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from .boundary_features import BoundaryFeatures, FEATURE_VERSION, select_window


class BoundaryModel:
    def __init__(self, window=4, text=False, C=1.0):
        self.config = dict(window=window, text=text, C=C)
        self.feature_version = FEATURE_VERSION
        self.threshold = .5
        self.dv = DictVectorizer()
        self.scale = MaxAbsScaler()
        self.tv = TfidfVectorizer(max_features=16000, min_df=3, ngram_range=(1,2), sublinear_tf=True) if text else None
        self.clf = LogisticRegression(C=C, solver='liblinear', max_iter=500, random_state=41)

    def matrix(self, rows, fit=False):
        d = [select_window(r,self.config['window']) for r in rows]
        x = self.dv.fit_transform(d) if fit else self.dv.transform(d)
        x = self.scale.fit_transform(x) if fit else self.scale.transform(x)
        if self.tv is not None:
            t = [r['text'] for r in rows]
            text = self.tv.fit_transform(t) if fit else self.tv.transform(t)
            x = hstack([x,text],format='csr')
        return x

    def fit(self, rows, labels):
        self.clf.fit(self.matrix(rows,True),labels)
        return self

    def probabilities(self, rows):
        return self.clf.predict_proba(self.matrix(rows))[:,1]


class OnlineBoundaryPredictor:
    """Append-only committed spans; observations of the current step are allowed."""
    def __init__(self, model):
        self.model = model
        if model.feature_version != FEATURE_VERSION:
            raise ValueError('Feature version mismatch')
        self.features = BoundaryFeatures(4)
        self.stages = []
        self.start = self.previous = None

    @classmethod
    def load(cls, path):
        if Path(path).suffix == '.json':
            from .temporal_boundary import TemporalPredictor
            return TemporalPredictor.load(path)
        return cls(joblib.load(path))

    def feed(self, step):
        t = time.perf_counter()
        x = self.features.feed(step)
        probability = float(self.model.probabilities([x])[0])
        boundary = self.previous is not None and probability >= self.model.threshold
        if self.start is None:
            self.start = step.idx
        if boundary:
            self.stages.append(dict(stage_id=len(self.stages)+1,start_step_id=self.start,
                                    end_step_id=self.previous,sealed=True))
            self.start = step.idx
        self.previous = step.idx
        return dict(boundary_step_id=step.idx if boundary else None, detected_at_step_id=step.idx,
                    emitted_at=time.time(), score=probability, status='committed' if boundary else 'continue',
                    model_version=self.model.feature_version, ms=(time.perf_counter()-t)*1000)

    def snapshot(self, final=False):
        return self.stages + ([dict(stage_id=len(self.stages)+1,start_step_id=self.start,
                                   end_step_id=self.previous,sealed=final)] if self.start is not None else [])
