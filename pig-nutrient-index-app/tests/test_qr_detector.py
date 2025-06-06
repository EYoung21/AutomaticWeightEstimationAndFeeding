import unittest
from src.vision.qr_detector import QRDetector, detect_qr_code

class TestQRDetector(unittest.TestCase):
    def setUp(self):
        # Initialize the QRDetector instance before each test
        self.qr_detector = QRDetector()

    def test_qr_code_detection(self):
        # Test if the QR code detection works correctly
        test_image_path = 'path/to/test/image_with_qr_code.jpg'
        detected_qr_codes = self.qr_detector.detect_qr_codes(test_image_path)
        self.assertGreater(len(detected_qr_codes), 0, "No QR codes detected in the image.")

    def test_qr_code_decoding(self):
        # Test if the QR code decoding works correctly
        test_image_path = 'path/to/test/image_with_qr_code.jpg'
        decoded_data = self.qr_detector.decode_qr_code(test_image_path)
        self.assertIsNotNone(decoded_data, "QR code could not be decoded.")

    def test_invalid_image(self):
        # Test the behavior with an invalid image
        invalid_image_path = 'path/to/test/invalid_image.jpg'
        with self.assertRaises(Exception):
            self.qr_detector.detect_qr_codes(invalid_image_path)

    def test_detect_qr_code(self):
        # Use a sample image with a QR code for testing
        image_path = "sample_qr.jpg"
        result = detect_qr_code(image_path)
        print("Detected QR codes:", result)
        # You can add assertions if you have expected data

if __name__ == '__main__':
    unittest.main()