"""Bounded-delay features: candidate boundary b is emitted only at step b+d."""
from collections import deque


class ConfirmationBuffer:
    def __init__(self, delay):
        if delay not in (0,1,2):
            raise ValueError('delay must be 0, 1 or 2')
        self.delay=delay
        self.rows=deque(maxlen=delay+1)
        self.seen=0
        self.last=None

    def feed(self,row):
        if self.last is not None and row['idx']<=self.last:
            raise ValueError('Native step IDs must increase')
        self.last=row['idx'];self.seen+=1;self.rows.append(row)
        if self.seen<self.delay+2: # The first trajectory step is never a boundary.
            return None
        rows=list(self.rows)
        features=dict(rows[0]['features'])
        for offset,r in enumerate(rows[1:],1):
            # The candidate already has four-step history. Only add each newly
            # available step, not a duplicated window or any labels/text.
            features.update({f'confirm{offset}:{k[5:]}':v for k,v in r['features'].items() if k.startswith('lag0:')})
        return dict(boundary_step_id=rows[0]['idx'],detected_at_step_id=row['idx'],features=features)

    # No implicit EOS flush: missing observations cannot be invented. The final
    # d candidates remain unconfirmed and their gold boundaries still count.


def confirmation_rows(rows,delay):
    buffer=ConfirmationBuffer(delay)
    result=[]
    for row in rows:
        item=buffer.feed(row)
        if item is not None: result.append(item)
    return result
