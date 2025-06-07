#!/usr/bin/env python3
"""
Demonstration script for the Pig Nutrient Index Application.
This script shows how to train a model and run inference on the PIGRGB-Weight dataset.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def demo_training():
    """Demonstrate training the weight estimation model"""
    print("=== DEMO: Training Weight Estimation Model ===\n")
    
    # Import training functionality
    from train_model import train_and_evaluate
    
    # Train a model with limited samples for quick demo
    print("Training model with 500 samples for quick demonstration...")
    result = train_and_evaluate(
        data_path="data",
        model_type="mlp",  # Multi-layer perceptron
        test_size=0.2,
        max_samples=500  # Limit for faster demo
    )
    
    if result:
        print("✅ Training completed successfully!")
        print(f"Model saved to: {result['model_path']}")
        return True
    else:
        print("❌ Training failed!")
        return False

def demo_inference():
    """Demonstrate inference with the trained model"""
    print("\n=== DEMO: Running Inference ===\n")
    
    from main import demo_with_dataset
    
    # Run inference on a few sample images
    print("Running inference on 3 sample images...")
    demo_with_dataset(data_path="data", num_samples=3)

def demo_rfid_detection():
    """Demonstrate RFID detection functionality"""
    print("\n=== DEMO: RFID Detection ===\n")
    
    from vision.rfid_detector import RFIDDetector
    from utils.dataset_loader import PigDatasetLoader
    import random
    
    # Initialize RFID detector
    detector = RFIDDetector()
    
    print("Testing RFID detection system...")
    
    # Simulate camera area detection
    pig_info = detector.detect_pig_in_camera_area(timeout=2.0)
    
    if pig_info:
        print(f"✅ Pig detected: {pig_info['pig_id']}")
        print(f"Detection method: {pig_info['detection_method']}")
        print(f"Confidence: {pig_info['confidence']:.2f}")
        
        # Display metadata if available
        if 'metadata' in pig_info and pig_info['metadata']:
            metadata = pig_info['metadata']
            if 'breed' in metadata:
                print(f"Breed: {metadata['breed']}")
            if 'birth_date' in metadata:
                print(f"Birth date: {metadata['birth_date']}")
        
        # Test feeder detection
        print("\nTesting feeder access control...")
        feeder_info = detector.detect_pig_at_feeder("FEEDER_01", timeout=2.0)
        
        if feeder_info:
            print(f"Pig at feeder: {feeder_info['pig_id']}")
            print(f"Feeding authorized: {feeder_info['feeding_authorized']}")
            print(f"Reason: {feeder_info['reason']}")
        
        print("✅ RFID detection demo completed!")
    else:
        print("❌ No pig detected in simulated area")
        print("✅ RFID detection demo completed (simulation mode)")

def demo_system_overview():
    """Show an overview of the complete system"""
    print("=== PIG NUTRIENT INDEX APPLICATION DEMO ===\n")
    
    print("🐷 SYSTEM OVERVIEW:")
    print("1. Pig Detection: Uses YOLO or direct image processing")
    print("2. Pig Identification: RFID tags for reliable identification")
    print("3. Weight Estimation: Machine learning with morphological features")
    print("4. Nutrition Calculation: Body weight → nutrient index (0-100)")
    print("5. Feeding Control: Automated Calan gate feeders with RFID access")
    print()
    
    print("📡 RFID SYSTEM:")
    from vision.rfid_detector import RealRFIDHardware
    hardware = RealRFIDHardware.get_recommended_hardware()
    print(f"• Reader: {hardware['reader']['model']} ({hardware['reader']['cost']})")
    print(f"• Tags: {hardware['tags']['model']} ({hardware['tags']['cost']})")
    print(f"• Range: {hardware['reader']['read_range']}")
    print()
    
    print("📊 DATASET:")
    from utils.dataset_loader import PigDatasetLoader
    loader = PigDatasetLoader("data")
    try:
        stats = loader.get_dataset_statistics()
        print(f"• Total images: {stats['total_images']}")
        print(f"• Weight range: {stats['min_weight']:.1f} - {stats['max_weight']:.1f} kg")
        print(f"• Average weight: {stats['mean_weight']:.1f} ± {stats['std_weight']:.1f} kg")
    except Exception as e:
        print(f"• Dataset not found or error: {e}")
    
    print()
    print("🎯 INNOVATION HIGHLIGHTS:")
    print("• Replaces manual weighing with computer vision")
    print("• Individual pig tracking with RFID tags")
    print("• Automated, personalized feeding with access control")
    print("• Real-time monitoring capabilities")
    print("• Cost-effective for medium-scale farms")
    print("• Reliable in harsh farm environments")

def check_dataset():
    """Check if the dataset is properly set up"""
    print("=== CHECKING DATASET SETUP ===\n")
    
    data_path = "data"
    if not os.path.exists(data_path):
        print(f"❌ Dataset directory '{data_path}' not found!")
        print("\nPlease ensure the PIGRGB-Weight dataset is extracted to the 'data' directory.")
        return False
    
    rgb_path = os.path.join(data_path, "RGB_9579")
    if not os.path.exists(rgb_path):
        print(f"❌ RGB_9579 directory not found in {data_path}")
        print("\nThe dataset should have the following structure:")
        print("data/")
        print("└── RGB_9579/")
        print("    ├── fold1/")
        print("    ├── fold2/")
        print("    └── ...")
        return False
    
    # Count images
    image_count = 0
    for root, dirs, files in os.walk(rgb_path):
        image_count += len([f for f in files if f.endswith('.png')])
    
    if image_count == 0:
        print("❌ No PNG images found in dataset!")
        return False
    
    print(f"✅ Dataset found with {image_count} images")
    return True

def run_full_demo():
    """Run the complete demonstration"""
    print("🚀 STARTING FULL DEMONSTRATION\n")
    
    # 1. System overview
    demo_system_overview()
    print("\n" + "="*60 + "\n")
    
    # 2. Check dataset
    if not check_dataset():
        print("\n❌ Demo cannot continue without proper dataset setup.")
        return
    
    print("\n" + "="*60 + "\n")
    
    # 3. RFID detection demo
    demo_rfid_detection()
    print("\n" + "="*60 + "\n")
    
    # 4. Training demo
    training_success = demo_training()
    print("\n" + "="*60 + "\n")
    
    # 5. Inference demo (only if training succeeded)
    if training_success:
        demo_inference()
    else:
        print("⚠️ Skipping inference demo due to training failure")
    
    print("\n" + "="*60 + "\n")
    print("🎉 DEMONSTRATION COMPLETE!")
    print("\nTo run the full application:")
    print("python src/main.py")
    print("\nTo train a model:")
    print("python src/train_model.py --data_path data --model_type mlp")

def main():
    """Main function with menu"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--full':
            run_full_demo()
            return
        elif sys.argv[1] == '--check':
            check_dataset()
            return
        elif sys.argv[1] == '--train':
            demo_training()
            return
        elif sys.argv[1] == '--inference':
            demo_inference()
            return
        elif sys.argv[1] == '--rfid':
            demo_rfid_detection()
            return
    
    print("=== Pig Nutrient Index Application Demo ===\n")
    print("Available options:")
    print("1. --full      : Run complete demonstration")
    print("2. --check     : Check dataset setup")
    print("3. --train     : Demo model training")
    print("4. --inference : Demo inference")
    print("5. --rfid      : Demo RFID detection")
    print("\nExample usage:")
    print("python demo.py --full")
    print("python demo.py --check")

if __name__ == "__main__":
    main() 