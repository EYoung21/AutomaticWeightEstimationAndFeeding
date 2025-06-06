# Test for CalanGateController class

import unittest
from src.feeder_control.calan_gate import CalanGateController

class TestCalanGateController(unittest.TestCase):
    def setUp(self):
        # Initialize the CalanGateController with a mock configuration
        self.controller = CalanGateController(min_feed=1.0, max_feed=5.0)

    def test_open_gate(self):
        # Test that the gate opens correctly for a specific pig
        pig_id = "pig_001"
        result = self.controller.open_gate(pig_id)
        self.assertTrue(result, "Gate should open for the specified pig.")

    def test_close_gate(self):
        # Test that the gate closes correctly for a specific pig
        pig_id = "pig_001"
        self.controller.open_gate(pig_id)  # Ensure the gate is open first
        result = self.controller.close_gate(pig_id)
        self.assertTrue(result, "Gate should close for the specified pig.")

    def test_control_gate_with_nutrient_index(self):
        # Test that the gate control logic works based on nutrient index
        pig_id = "pig_001"
        nutrient_index = 75  # Example nutrient index
        result = self.controller.control_gate(pig_id, nutrient_index)
        self.assertTrue(result, "Gate control should succeed based on nutrient index.")

    def test_invalid_pig_id(self):
        # Test behavior with an invalid pig ID
        invalid_pig_id = "invalid_pig"
        with self.assertRaises(ValueError):
            self.controller.open_gate(invalid_pig_id)

    def test_feed_amount_low_index(self):
        self.assertEqual(self.controller.get_feed_amount(0), 5.0)

    def test_feed_amount_high_index(self):
        self.assertEqual(self.controller.get_feed_amount(100), 1.0)

    def test_feed_amount_mid_index(self):
        self.assertEqual(self.controller.get_feed_amount(50), 3.0)

if __name__ == '__main__':
    unittest.main()