"""Small, standard-library-only causal purpose classifier runtime.

The artifact is experimental and must be loaded only from a trusted path.
"""
import gzip
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

FEATURE_VERSION = 'purpose-student-text-v1'


def edge(text, head, tail=None):
    text = str(text or '')
    tail = head if tail is None else tail
    if len(text) <= head + tail:
        return text
    return text[:head] + ' [TRUNCATED] ' + text[-tail:]


def feature_text(task, steps):
    """Use only the current completed step and one preceding step."""
    if not steps:
        raise ValueError('At least one completed step is required')
    current = steps[-1]
    previous = steps[-2] if len(steps) > 1 else {}
    return '\n'.join((
        'task: ' + edge(task, 500),
        'previous tool: ' + str(previous.get('tool') or ''),
        'previous action: ' + edge(previous.get('action'), 160),
        'previous observation: ' + edge(previous.get('observation'), 160),
        'current tool: ' + str(current.get('tool') or ''),
        'current action: ' + edge(current.get('action'), 500),
        'current observation: ' + edge(current.get('observation'), 500),
    ))


class PurposeStudent:
    def __init__(self, artifact):
        if artifact.get('feature_version') != FEATURE_VERSION:
            raise ValueError('Purpose feature version mismatch')
        if artifact.get('ngram_range') != [3,5] or artifact.get('sublinear_tf') is not True:
            raise ValueError('Unsupported vectorizer configuration')
        self.artifact = artifact
        self.classes = artifact['classes']
        self.vocab = artifact['vocabulary']
        self.idf = artifact['idf']
        self.coef = artifact['coef']
        self.intercept = artifact['intercept']
        n=len(self.vocab)
        if len(self.idf)!=n or any(len(row)!=n for row in self.coef):
            raise ValueError('Artifact dimensions differ')
        if len(self.classes)!=len(self.coef) or len(self.classes)!=len(self.intercept):
            raise ValueError('Artifact class dimensions differ')

    @classmethod
    def load(cls, path):
        with gzip.open(path,'rt',encoding='utf-8') as f:
            return cls(json.load(f))

    def probabilities_text(self, text):
        text=re.sub(r'\s\s+',' ',text.lower())
        counts=Counter()
        for width in (3,4,5):
            for i in range(len(text)-width+1):
                index=self.vocab.get(text[i:i+width])
                if index is not None:
                    counts[index]+=1
        values={i:(1+math.log(c))*self.idf[i] for i,c in counts.items()}
        norm=math.sqrt(sum(x*x for x in values.values()))
        if norm:
            values={i:x/norm for i,x in values.items()}
        logits=[b+sum(row[i]*x for i,x in values.items())
                for row,b in zip(self.coef,self.intercept)]
        peak=max(logits)
        exp=[math.exp(x-peak) for x in logits]
        total=sum(exp)
        return {label:value/total for label,value in zip(self.classes,exp)}

    def predict(self, task, steps):
        started=time.perf_counter()
        probabilities=self.probabilities_text(feature_text(task,steps))
        label=max(probabilities,key=probabilities.get)
        return dict(purpose=label,probabilities=probabilities,
                    ms=(time.perf_counter()-started)*1000,
                    model_version=FEATURE_VERSION)
