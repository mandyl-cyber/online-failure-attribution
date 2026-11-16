import unittest
from remote_pilots.openhands_smoke.audit_timing import decompose


def event(b,t=None):
    return dict(boundary_step_id=b,detected_at_step_id=b if t is None else t)


class TimingAuditTests(unittest.TestCase):
    def test_late_one_step_and_far_prediction(self):
        events=[event(11),event(20)]
        a=decompose(events,[10,30],0)
        self.assertEqual(a['fp'],{'timing_only':1,'outside_tolerance':1})
        self.assertEqual(a['fn'],{'timing_only':1,'no_nearby_prediction':1})
        self.assertEqual(decompose(events,[10,30],1)['tp'],1)

    def test_competing_predictions_are_not_all_counted(self):
        a=decompose([event(9),event(10)],[10],0)
        self.assertEqual(a['tp'],1)
        self.assertEqual(a['fp'],{'competition':1})

    def test_reassignment_can_increase_matching(self):
        events=[event(10),event(12)]
        self.assertEqual(decompose(events,[10,11],0)['tp'],1)
        self.assertEqual(decompose(events,[10,11],1)['tp'],2)

    def test_no_gold_and_delayed_emission(self):
        self.assertEqual(decompose([event(10)],[],0)['fp'],{'no_gold':1})
        self.assertEqual(decompose([event(10,12)],[10],1)['fp'],{'timing_only':1})


if __name__=='__main__':unittest.main()
