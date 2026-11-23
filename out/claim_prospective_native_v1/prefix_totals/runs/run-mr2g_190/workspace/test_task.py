import unittest
from task import solve

class TaskTests(unittest.TestCase):
    def test_0(self):
        self.assertEqual(solve([]), [0])
    def test_1(self):
        self.assertEqual(solve([2,-1,4]), [0,2,1,5])
    def test_2(self):
        x=[1,2,3]; before=list(x); solve(x); self.assertEqual(x,before)
