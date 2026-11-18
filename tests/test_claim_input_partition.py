import unittest
from tests.test_stage_claim_facts import source
from segmenter import stage_claim_summary_v5 as core
from segmenter.claim_input_partition import partition


class PartitionTests(unittest.TestCase):
    def test_whole_steps_and_oversized_are_accounted(self):
        original=source("cat a","a",1)
        other=source("cat b","b",2)
        original["stage"]["end_step_id"]=2
        original["detected_at_seq"]=2
        original["evidence"].extend(other["evidence"])
        row=core.build(original)
        parts,oversized=partition(row,lambda r: r["request"]["stage"]["start_step_id"]==r["request"]["stage"]["end_step_id"])
        self.assertEqual(len(parts),2)
        self.assertEqual(oversized,[])
        records=[e["record_id"] for p in parts for e in p["row"]["source_snapshot"]["evidence"]]
        self.assertEqual(records,[e["record_id"] for e in original["evidence"]])
        for part in parts:
            core.verify_row(part["row"])
        parts,oversized=partition(row,lambda r: False)
        self.assertEqual(parts,[])
        self.assertEqual([e["step_id"] for e in oversized],[1,2])


if __name__=="__main__":
    unittest.main()
