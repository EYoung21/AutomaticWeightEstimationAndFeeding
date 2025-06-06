# Contents of /pig-nutrient-index-app/pig-nutrient-index-app/src/main.py

# Import necessary modules
from src.vision.qr_detector import QRDetector
from src.vision.weight_estimator import WeightEstimator
from src.index_calculator.nutrient_index import NutrientIndexCalculator
from src.feeder_control.calan_gate import CalanGateController
from src.vision.yolo_detector import YOLODetector
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

def process_pig_image_with_yolo(image_path):
    # Step 1: Detect pigs using YOLO
    yolo_detector = YOLODetector(model_path='yolov5s.pt')  # Use your trained model path
    pig_boxes = yolo_detector.detect_pigs(image_path)
    if not pig_boxes:
        print("No pigs detected in the image.")
        return

    # Load the image for cropping
    img = cv2.imread(image_path)

    for idx, box in enumerate(pig_boxes):
        x1, y1, x2, y2 = map(int, box)
        pig_crop = img[y1:y2, x1:x2]

        # Optional: Save or process pig_crop as needed
        # cv2.imwrite(f"pig_{idx}.jpg", pig_crop)

        # Step 2: Identify pig by QR code (on the cropped pig region)
        qr_detector = QRDetector()
        pig_ids = qr_detector.detect_qr_codes(pig_crop)
        pig_id = pig_ids[0] if pig_ids else f"unknown_{idx}"

        # Step 3: Estimate pig weight (on the cropped pig region)
        weight_estimator = WeightEstimator()
        # You may need to modify estimate_weight to accept an image array
        weight = weight_estimator.estimate_weight(pig_crop)
        print(f"Pig {pig_id} estimated weight: {weight:.2f} kg")

        # Step 4: Calculate nutrient index
        index_calculator = NutrientIndexCalculator()
        nutrient_index = index_calculator.calculate_index(weight)
        print(f"Pig {pig_id} nutrient index: {nutrient_index}")

        # Step 5: Control feeder
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

def live_camera_monitor():
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    yolo_detector = YOLODetector(model_path='yolov5s.pt')
    qr_detector = QRDetector()
    weight_estimator = WeightEstimator()
    index_calculator = NutrientIndexCalculator()
    feeder = CalanGateController()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        pig_boxes = yolo_detector.detect_pigs(frame)
        for idx, box in enumerate(pig_boxes):
            x1, y1, x2, y2 = map(int, box)
            pig_crop = frame[y1:y2, x1:x2]

            # Detect QR code
            pig_ids = qr_detector.detect_qr_codes(pig_crop)
            pig_id = pig_ids[0] if pig_ids else f"unknown_{idx}"

            # Estimate weight
            weight = weight_estimator.estimate_weight(pig_crop)
            nutrient_index = index_calculator.calculate_index(weight)
            feed_amount = feeder.get_feed_amount(nutrient_index)
            feeder.control_feeder(pig_id, feed_amount)

            # Draw bounding box and info
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{pig_id}: {weight:.1f}kg, idx:{nutrient_index}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("Pig Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/test/image_with_pig_and_qr.jpg"
    process_pig_image(image_path)
    main()
    # Example usage with YOLO
    image_path = "path/to/test/image_with_pigs.jpg"
    process_pig_image_with_yolo(image_path)
    # To run live monitoring, uncomment the line below:
    # live_camera_monitor()