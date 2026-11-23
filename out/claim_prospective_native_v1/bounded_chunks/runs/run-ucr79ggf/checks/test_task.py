import unittest
from task import solve

class TaskTests(unittest.TestCase):
    def test_0(self):
        self.assertEqual(solve([1,2,3,4,5],2), [[1,2],[3,4],[5]])
    def test_1(self):
        self.assertEqual(solve([],2), [])
    def test_2(self):
        with self.assertRaises(ValueError): solve([1],0)
    def test_3(self):
        with self.assertRaises(ValueError): solve([1],-1)
