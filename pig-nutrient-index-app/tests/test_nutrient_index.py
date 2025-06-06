import unittest
from src.index_calculator.nutrient_index import NutrientIndexCalculator

class TestNutrientIndexCalculator(unittest.TestCase):
    def setUp(self):
        # Initialize the NutrientIndexCalculator before each test
        self.calculator = NutrientIndexCalculator(min_weight=30, max_weight=120)

    def test_index_low(self):
        # Test case for a weight below the minimum threshold
        self.assertEqual(self.calculator.calculate_index(20), 0)

    def test_index_high(self):
        # Test case for a weight above the maximum threshold
        self.assertEqual(self.calculator.calculate_index(130), 100)

    def test_index_mid(self):
        # Test case for a weight in the middle of the range
        self.assertEqual(self.calculator.calculate_index(75), 50)

if __name__ == '__main__':
    unittest.main()