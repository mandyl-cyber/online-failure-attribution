import unittest
from task import solve

class TaskTests(unittest.TestCase):
    def test_0(self):
        self.assertEqual(solve([]), [])
    def test_1(self):
        self.assertEqual(solve(['b','a','b','c']), ['b','a','c'])
    def test_2(self):
        x=['b','a','b']; before=list(x); solve(x); self.assertEqual(x,before)
