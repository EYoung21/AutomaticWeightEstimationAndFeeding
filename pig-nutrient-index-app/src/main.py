# Contents of /pig-nutrient-index-app/pig-nutrient-index-app/src/main.py

# Import necessary modules
from src.vision.qr_detector import QRDetector
from src.vision.weight_estimator import WeightEstimator
from src.index_calculator.nutrient_index import NutrientIndexCalculator
from src.feeder_control.calan_gate import CalanGateController
import cv2  # OpenCV for image processing
import time  # For timing operations

def process_pig_image(image_path):
    # Step 1: Identify pig by QR code
    qr_detector = QRDetector()
    pig_ids = qr_detector.detect_qr_codes(image_path)
    if not pig_ids:
        print("No pig QR code detected.")
        return
    pig_id = pig_ids[0]  # Assume one pig per image

    # Step 2: Estimate pig weight
    weight_estimator = WeightEstimator()
    weight = weight_estimator.estimate_weight(image_path)
    print(f"Pig {pig_id} estimated weight: {weight:.2f} kg")

    # Step 3: Calculate nutrient index
    index_calculator = NutrientIndexCalculator()
    nutrient_index = index_calculator.calculate_index(weight)
    print(f"Pig {pig_id} nutrient index: {nutrient_index}")

    # Step 4: Control feeder
    feeder = CalanGateController()
    feed_amount = feeder.get_feed_amount(nutrient_index)
    feeder.control_feeder(pig_id, feed_amount)

def main():
    # Initialize components
    weight_estimator = WeightEstimator()
    qr_detector = QRDetector()
    calan_gate_controller = CalanGateController()
    nutrient_index_calculator = NutrientIndexCalculator()

    # Main loop to process images and control feeders
    while True:
        # Capture image from camera (assuming a camera is connected)
        ret, frame = cv2.VideoCapture(0).read()
        if not ret:
            print("Failed to capture image")
            continue

        # Detect QR code to identify the pig
        pig_id = qr_detector.detect_qr_code(frame)
        if pig_id is None:
            print("No QR code detected")
            continue

        # Estimate the weight of the pig
        estimated_weight = weight_estimator.estimate_weight(frame)
        
        # Calculate the nutrient index based on the estimated weight
        nutrient_index = nutrient_index_calculator.calculate_index(estimated_weight)

        # Control the Calan gate feeder based on the nutrient index
        calan_gate_controller.control_feeder(pig_id, nutrient_index)

        # Wait for a short period before the next iteration
        time.sleep(1)

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/test/image_with_pig_and_qr.jpg"
    process_pig_image(image_path)
    main()