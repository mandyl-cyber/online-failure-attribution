"""Bounded revision of boundary decisions; no model and no implicit EOS lookahead."""


class DelayedSeal:
    """Boundary b starts a new stage; sealing emits [start, b-1].

    Each feed supplies a complete replacement for all still revisable decisions.
    Caller must use only steps observed through t. Native IDs must be contiguous.
    """

    def __init__(self, delay=0, start=1):
        if type(delay) is not int or delay not in (0, 1, 2):
            raise ValueError('delay must be 0, 1 or 2')
        if type(start) is not int or start < 0:
            raise ValueError('invalid start')
        self.delay, self.origin = delay, start
        self.current, self.start = start - 1, start
        self.pending = {}
        self.revisions = 0
        self.closed = False

    def feed(self, t, decisions):
        if self.closed or type(t) is not int or t != self.current + 1:
            raise ValueError('closed stream or noncontiguous step')
        expected = set(range(max(self.origin + 1, t - self.delay), t + 1))
        if (set(decisions) != expected or any(type(k) is not int for k in decisions)
                or any(type(v) is not bool for v in decisions.values())):
            raise ValueError('supply exactly the mutable boundary window')
        self.revisions += sum(self.pending[k] != v for k, v in decisions.items() if k in self.pending)
        self.pending = dict(decisions)
        self.current = t
        boundary = t - self.delay
        event = []
        if boundary in self.pending:
            if self.pending.pop(boundary):
                event.append(dict(start_step_id=self.start, end_step_id=boundary - 1,
                                  boundary_step_id=boundary, sealed_at_step_id=t,
                                  reason='confirmed_boundary'))
                self.start = boundary
        return event

    def finish(self):
        """Explicit stream end seals the residual stage, not provisional boundaries."""
        if self.closed:
            raise ValueError('already closed')
        self.closed = True
        if self.current < self.origin:
            return []
        return [dict(start_step_id=self.start, end_step_id=self.current,
                     boundary_step_id=None, sealed_at_step_id=self.current,
                     reason='stream_end', unconfirmed_boundaries=sorted(k for k,v in self.pending.items() if v))]
