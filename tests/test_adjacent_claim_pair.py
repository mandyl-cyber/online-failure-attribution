import unittest

from remote_pilots.openhands_smoke import adjacent_claim_pair as pair


class FakeTokenizer:
    def apply_chat_template(self, messages, **_):
        # Only the Matplotlib pair is too long; both arms must be excluded.
        return [1] * (7000 if 'ContourSet' in messages[1]['content'] else 100)


class AdjacentClaimPairTests(unittest.TestCase):
    def test_frozen_pairs_and_symmetric_token_exclusion(self):
        protocol, groups, rows = pair.inputs(pair.ROOT)
        self.assertEqual((len(groups), len(rows)), (9, 28))
        check = pair.preflight(pair.ROOT, FakeTokenizer())
        self.assertEqual(check['eligible_groups'], 8)
        excluded = [g for g in groups if not check['groups'][g['group_id']]['eligible']]
        self.assertEqual([g['task'] for g in excluded], ['matplotlib__matplotlib-26342'])
        self.assertEqual(len(excluded[0]['arms']['original']), 2)
        self.assertEqual(len(excluded[0]['arms']['adjacent']), 1)


if __name__ == '__main__':
    unittest.main()
