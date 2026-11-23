import unittest
from task import solve

class TaskTests(unittest.TestCase):
    def test_0(self):
        self.assertEqual(solve({'a':1},{'a':0}), {'a':0})
    def test_1(self):
        self.assertEqual(solve({'a':True},{'a':False,'b':''}), {'a':False,'b':''})
    def test_2(self):
        self.assertEqual(solve({'a':1},{'a':None}), {'a':1})
    def test_3(self):
        x={'a':1}; y={'b':2}; z=solve(x,y); self.assertEqual(x,{'a':1}); self.assertEqual(y,{'b':2}); self.assertIsNot(x,z)
