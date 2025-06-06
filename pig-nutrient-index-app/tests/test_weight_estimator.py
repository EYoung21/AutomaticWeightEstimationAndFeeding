import unittest
from src.vision.weight_estimator import WeightEstimator

class TestWeightEstimator(unittest.TestCase):
    def setUp(self):
        # Initialize the WeightEstimator before each test
        self.estimator = WeightEstimator()

    def test_estimate_weight(self):
        # Test with a valid image input
        test_image_path = 'path/to/test/image_with_pig.jpg'  # Replace with actual image path
        weight = self.estimator.estimate_weight(test_image_path)
        print("Estimated weight:", weight)
        self.assertGreater(weight, 0, "Estimated weight should be positive.")

    def test_invalid_image(self):
        # Test with an invalid image input
        with self.assertRaises(ValueError):
            self.estimator.estimate_weight('invalid_path.jpg')

if __name__ == '__main__':
    unittest.main()