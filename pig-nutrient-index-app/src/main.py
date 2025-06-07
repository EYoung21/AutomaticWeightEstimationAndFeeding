# Contents of /pig-nutrient-index-app/pig-nutrient-index-app/src/main.py

# Import necessary modules
from src.vision.rfid_detector import RFIDDetector
from src.vision.weight_estimator import WeightEstimator
from src.index_calculator.nutrient_index import NutrientIndexCalculator
from src.feeder_control.calan_gate import CalanGateController
from src.vision.yolo_detector import YOLODetector
from src.utils.dataset_loader import PigDatasetLoader
import cv2  # OpenCV for image processing
import time  # For timing operations
import os
import numpy as np

def load_trained_model(model_path="models/pig_weight_model_mlp.joblib"):
    """Load a pre-trained weight estimation model"""
    estimator = WeightEstimator()
    if os.path.exists(model_path):
        try:
            estimator.load_model(model_path)
            print(f"Loaded trained model from {model_path}")
            return estimator
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model - please train first using train_model.py")
            return estimator
    else:
        print(f"Model file {model_path} not found")
        print("Using untrained model - please train first using train_model.py")
        return estimator

def process_pig_image(image_path, use_trained_model=True):
    """Process a single pig image with RFID detection and weight estimation"""
    print(f"Processing image: {image_path}")
    
    # Step 1: Identify pig by RFID tag
    rfid_detector = RFIDDetector()
    pig_info = rfid_detector.detect_pig_in_camera_area()
    
    if not pig_info or not pig_info['pig_id']:
        print("No pig detected in camera area.")
        return
    
    pig_id = pig_info['pig_id']
    confidence = pig_info['confidence']
    print(f"Pig identified: {pig_id} (confidence: {confidence:.2f})")
    print(f"Detection method: {pig_info['detection_method']}")
    
    # Display metadata if available
    if 'metadata' in pig_info and pig_info['metadata']:
        metadata = pig_info['metadata']
        if 'breed' in metadata:
            print(f"Breed: {metadata['breed']}")
        if 'birth_date' in metadata:
            print(f"Birth date: {metadata['birth_date']}")

    # Step 2: Estimate pig weight
    if use_trained_model:
        weight_estimator = load_trained_model()
    else:
        weight_estimator = WeightEstimator()
    
    try:
        if weight_estimator.is_trained:
            weight = weight_estimator.estimate_weight(image_path)
        else:
            print("Warning: Using basic weight estimation - train model for better accuracy")
            # Fallback to basic estimation
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                weight = area * 0.00005  # Basic scaling factor
            else:
                weight = 50.0  # Default weight
        
        print(f"Pig {pig_id} estimated weight: {weight:.2f} kg")
    except Exception as e:
        print(f"Error estimating weight: {e}")
        weight = 50.0  # Default weight

    # Step 3: Calculate nutrient index
    index_calculator = NutrientIndexCalculator()
    nutrient_index = index_calculator.calculate_index(weight)
    print(f"Pig {pig_id} nutrient index: {nutrient_index}")

    # Step 4: Control feeder
    feeder = CalanGateController()
    feed_amount = feeder.get_feed_amount(nutrient_index)
    feeder.control_feeder(pig_id, feed_amount)
    
    # Step 5: Visualize results
    visualize_results(image_path, pig_info, weight, nutrient_index, feed_amount)
    
    return {
        'pig_id': pig_id,
        'weight': weight,
        'nutrient_index': nutrient_index,
        'feed_amount': feed_amount,
        'confidence': confidence
    }

def visualize_results(image_path, pig_info, weight, nutrient_index, feed_amount):
    """Create a visualization of the detection and estimation results"""
    rfid_detector = RFIDDetector()
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return
    
    # Draw ear tag detection results
    result_img = rfid_detector.visualize_detection(img, pig_info)
    
    # Add weight and nutrition information
    y_offset = 90
    cv2.putText(result_img, f"Weight: {weight:.1f} kg", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    y_offset += 30
    cv2.putText(result_img, f"Nutrient Index: {nutrient_index}/100", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    y_offset += 30
    cv2.putText(result_img, f"Feed Amount: {feed_amount} kg", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Show result
    cv2.imshow("Pig Analysis Results", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result image
    output_path = f"results_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, result_img)
    print(f"Result saved to: {output_path}")

def process_pig_image_with_yolo(image_path, use_trained_model=True):
    """Process image with YOLO pig detection first, then ear tag detection"""
    # Step 1: Detect pigs using YOLO
    yolo_detector = YOLODetector(model_path='yolov5s.pt')  # Use your trained model path
    pig_boxes = yolo_detector.detect_pigs(image_path)
    
    if not pig_boxes:
        print("No pigs detected in the image.")
        return

    # Load the image for cropping
    img = cv2.imread(image_path)
    rfid_detector = RFIDDetector()
    
    if use_trained_model:
        weight_estimator = load_trained_model()
    else:
        weight_estimator = WeightEstimator()

    results = []
    for idx, box in enumerate(pig_boxes):
        x1, y1, x2, y2 = map(int, box)
        pig_crop = img[y1:y2, x1:x2]

        # Step 2: Identify pig by ear tags in the cropped region
        pig_info = rfid_detector.detect_and_identify_pig(pig_crop)
        pig_id = pig_info['pig_id'] if pig_info['pig_id'] else f"unknown_{idx}"

        # Step 3: Estimate pig weight on the cropped pig region
        try:
            if weight_estimator.is_trained:
                weight = weight_estimator.estimate_weight(pig_crop)
            else:
                # Basic fallback estimation
                gray = cv2.cvtColor(pig_crop, cv2.COLOR_BGR2GRAY)
                weight = np.sum(gray > 50) * 0.0001  # Very basic estimation
        except Exception as e:
            print(f"Error estimating weight for pig {idx}: {e}")
            weight = 50.0

        print(f"Pig {pig_id} estimated weight: {weight:.2f} kg")

        # Step 4: Calculate nutrient index
        index_calculator = NutrientIndexCalculator()
        nutrient_index = index_calculator.calculate_index(weight)
        print(f"Pig {pig_id} nutrient index: {nutrient_index}")

        # Step 5: Control feeder
        feeder = CalanGateController()
        feed_amount = feeder.get_feed_amount(nutrient_index)
        feeder.control_feeder(pig_id, feed_amount)
        
        results.append({
            'pig_id': pig_id,
            'weight': weight,
            'nutrient_index': nutrient_index,
            'feed_amount': feed_amount,
            'bounding_box': (x1, y1, x2, y2)
        })

    return results

def demo_with_dataset(data_path="data", num_samples=5):
    """Demo the system using images from the PIGRGB-Weight dataset"""
    print("=== Demo with PIGRGB-Weight Dataset ===")
    
    # Load dataset
    loader = PigDatasetLoader(data_path)
    try:
        image_paths, weights = loader.load_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    if not image_paths:
        print("No images found in dataset!")
        return
    
    print(f"Dataset loaded: {len(image_paths)} images")
    
    # Select random samples for demo
    import random
    sample_indices = random.sample(range(len(image_paths)), min(num_samples, len(image_paths)))
    
    results = []
    for i in sample_indices:
        image_path = image_paths[i]
        actual_weight = weights[i]
        
        print(f"\n--- Processing sample {len(results)+1}/{num_samples} ---")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Actual weight: {actual_weight:.2f} kg")
        
        result = process_pig_image(image_path, use_trained_model=True)
        if result:
            result['actual_weight'] = actual_weight
            result['weight_error'] = abs(result['weight'] - actual_weight)
            results.append(result)
            
            print(f"Weight error: {result['weight_error']:.2f} kg")
    
    # Summary statistics
    if results:
        print(f"\n=== DEMO SUMMARY ===")
        avg_error = sum(r['weight_error'] for r in results) / len(results)
        print(f"Average weight estimation error: {avg_error:.2f} kg")
        
        for result in results:
            print(f"Pig {result['pig_id']}: "
                  f"Estimated {result['weight']:.1f}kg "
                  f"(Actual {result['actual_weight']:.1f}kg, "
                  f"Error {result['weight_error']:.1f}kg) "
                  f"Feed: {result['feed_amount']}kg")

def live_camera_monitor():
    """Live camera monitoring with RFID detection"""
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    
    # Initialize components
    yolo_detector = YOLODetector(model_path='yolov5s.pt')
    rfid_detector = RFIDDetector()
    weight_estimator = load_trained_model()
    index_calculator = NutrientIndexCalculator()
    feeder = CalanGateController()

    print("Starting live camera monitoring...")
    print("Press 'q' to quit, 's' to save current frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Detect pigs using YOLO
        pig_boxes = yolo_detector.detect_pigs(frame)
        
        for idx, box in enumerate(pig_boxes):
            x1, y1, x2, y2 = map(int, box)
            pig_crop = frame[y1:y2, x1:x2]

            # Detect RFID and identify pig
            pig_info = rfid_detector.detect_pig_in_camera_area(timeout=1.0)
            pig_id = pig_info['pig_id'] if pig_info and pig_info['pig_id'] else f"unknown_{idx}"

            # Estimate weight
            try:
                if weight_estimator.is_trained:
                    weight = weight_estimator.estimate_weight(pig_crop)
                else:
                    weight = 50.0  # Default
            except:
                weight = 50.0

            # Calculate nutrition and feeding
            nutrient_index = index_calculator.calculate_index(weight)
            feed_amount = feeder.get_feed_amount(nutrient_index)
            
            # Control feeder
            feeder.control_feeder(pig_id, feed_amount)

            # Draw bounding box and info
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            
            # Display information
            info_text = f"{pig_id}: {weight:.1f}kg, idx:{nutrient_index}, feed:{feed_amount}kg"
            cv2.putText(frame, info_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            # Draw RFID metadata if available
            if pig_info and 'metadata' in pig_info and pig_info['metadata']:
                metadata = pig_info['metadata']
                if 'breed' in metadata:
                    breed_text = f"Breed: {metadata['breed']}"
                    cv2.putText(frame, breed_text, (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

        cv2.imshow("Pig Monitoring - Live Feed", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = int(time.time())
            filename = f"live_capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")

    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function with menu system"""
    print("=== Pig Nutrient Index Application ===")
    print("Enhanced with RFID Detection and Machine Learning Weight Estimation")
    
    while True:
        print("\nSelect an option:")
        print("1. Process single image")
        print("2. Demo with dataset samples")
        print("3. Live camera monitoring")
        print("4. Train new model")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                process_pig_image(image_path)
            else:
                print("Image file not found!")
                
        elif choice == '2':
            num_samples = input("Number of samples to demo (default 5): ").strip()
            try:
                num_samples = int(num_samples) if num_samples else 5
            except ValueError:
                num_samples = 5
            demo_with_dataset(num_samples=num_samples)
            
        elif choice == '3':
            live_camera_monitor()
            
        elif choice == '4':
            print("To train a new model, run:")
            print("python src/train_model.py --data_path data --model_type mlp")
            print("or")
            print("python src/train_model.py --compare")
            
        elif choice == '5':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    # Quick demo if run directly
    if len(os.sys.argv) > 1:
        if os.sys.argv[1] == '--demo':
            demo_with_dataset()
        elif os.sys.argv[1] == '--live':
            live_camera_monitor()
        else:
            # Process single image
            image_path = os.sys.argv[1]
            if os.path.exists(image_path):
                process_pig_image(image_path)
            else:
                print(f"Image file not found: {image_path}")
    else:
        main()