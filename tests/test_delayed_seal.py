import unittest
from segmenter.delayed_seal import DelayedSeal


class SealTests(unittest.TestCase):
    def test_revision_before_claim_dispatch(self):
        s = DelayedSeal(2)
        self.assertEqual(s.feed(1, {}), [])
        self.assertEqual(s.feed(2, {2: True}), [])
        self.assertEqual(s.feed(3, {2: False, 3: True}), [])
        self.assertEqual(s.feed(4, {2: False, 3: True, 4: False}), [])
        event = s.feed(5, {3: True, 4: False, 5: False})[0]
        self.assertEqual((event['start_step_id'], event['end_step_id'], event['sealed_at_step_id']), (1, 2, 5))
        self.assertEqual(s.revisions, 1)
        with self.assertRaises(ValueError):
            s.feed(6, {3: False, 4: False, 5: False, 6: False})

    def test_partition_and_end_policy(self):
        for delay in (0, 1, 2):
            s = DelayedSeal(delay)
            emitted = []
            for t in range(1, 8):
                emitted += s.feed(t, {b: b in (3, 7) for b in range(max(2, t-delay), t+1)})
            emitted += s.finish()
            covered = [i for e in emitted for i in range(e['start_step_id'], e['end_step_id']+1)]
            self.assertEqual(covered, list(range(1, 8)))
            self.assertEqual([e['boundary_step_id'] for e in emitted[:-1]], [3, 7] if delay == 0 else [3])
            with self.assertRaises(ValueError):
                s.finish()

    def test_bad_input_does_not_advance(self):
        s = DelayedSeal(1)
        s.feed(1, {})
        with self.assertRaises(ValueError):
            s.feed(2, {2: 1})
        self.assertEqual(s.current, 1)
        self.assertEqual(s.feed(2, {2: True}), [])


if __name__ == '__main__':
    unittest.main()
