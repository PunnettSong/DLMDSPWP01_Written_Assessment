import unittest

class ZeroUnitTest(unittest.TestCase):
    def test_Zero(self, val):
        self.val = val
        self.assertEqual(val, 0, "The value is should zero")
    def test_NotZero(self, val):
        self.val = val
        self.assertNotEqual(val, 0, "The value should not be zero")
if __name__ == '__main__':
    unittest.Train_Data()