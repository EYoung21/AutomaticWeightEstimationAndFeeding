#!/usr/bin/env python3
"""
Incremental training script for pig weight estimation.
Start with a small dataset and progressively add more images.
"""

import os
import sys
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_loader import PigDatasetLoader
from vision.weight_estimator import WeightEstimator

def incremental_training_demo(data_path, initial_samples=1000, increment_size=1000, test_percentage=0.2):
    """
    Demonstrate incremental training by progressively adding more images until ALL dataset is used.
    
    Parameters:
    data_path: Path to the PIGRGB-Weight dataset
    initial_samples: Number of images to start with
    increment_size: Number of images to add in each increment
    test_percentage: Percentage of total dataset to reserve for testing
    """
    print("=== Incremental Training Demo - Full Dataset ===")
    print(f"Starting with: {initial_samples} images")
    print(f"Adding: {increment_size} images per increment")
    print(f"Goal: Use ALL images in the dataset")
    print(f"Test set: {test_percentage*100}% of total dataset")
    
    # Load full dataset
    print("\n1. Loading full dataset...")
    loader = PigDatasetLoader(data_path)
    
    try:
        all_image_paths, all_weights = loader.load_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    if len(all_image_paths) < initial_samples:
        print(f"Dataset too small! Only {len(all_image_paths)} images available.")
        return None
    
    total_images = len(all_image_paths)
    
    # Calculate test set size from total dataset
    test_size = int(total_images * test_percentage)
    training_size = total_images - test_size
    
    # Shuffle the dataset for random sampling
    indices = np.random.permutation(total_images)
    all_image_paths = [all_image_paths[i] for i in indices]
    all_weights = [all_weights[i] for i in indices]
    
    print(f"Dataset shuffled. Total: {total_images} images")
    print(f"Training images available: {training_size}")
    print(f"Test images reserved: {test_size}")
    
    # Split data: first part for training (will be used incrementally), last part for testing
    training_paths = all_image_paths[:training_size]
    training_weights = all_weights[:training_size]
    test_paths = all_image_paths[training_size:]
    test_weights = all_weights[training_size:]
    
    # Initial training data
    initial_paths = training_paths[:initial_samples]
    initial_weights = training_weights[:initial_samples]
    
    print(f"Starting with {len(initial_paths)} images, will grow to {len(training_paths)} total")
    
    # Initial training
    print(f"\n2. Initial training with {len(initial_paths)} images...")
    estimator = WeightEstimator(model_type='mlp')  # Only MLP supports incremental training
    
    try:
        initial_results = estimator.train(initial_paths, initial_weights)
        print(f"Initial training completed!")
        print(f"Initial MAE: {initial_results['mae']:.2f} kg")
        print(f"Initial R²: {initial_results['r2']:.3f}")
        
        # Test initial model
        if test_paths:
            test_results = estimator.evaluate(test_paths, test_weights)
            print(f"Initial test MAE: {test_results['mae']:.2f} kg")
            print(f"Initial test R²: {test_results['r2']:.3f}")
            
    except Exception as e:
        print(f"Error during initial training: {e}")
        return None
    
    # Incremental training
    current_count = initial_samples
    training_history = [{
        'total_samples': current_count,
        'train_mae': initial_results['mae'],
        'train_r2': initial_results['r2'],
        'test_mae': test_results['mae'] if test_paths else None,
        'test_r2': test_results['r2'] if test_paths else None
    }]
    
    while current_count < len(training_paths):
        # Determine increment size (don't exceed available training data)
        actual_increment = min(increment_size, len(training_paths) - current_count)
        
        if actual_increment <= 0:
            break
            
        # Get next batch of images
        next_end = current_count + actual_increment
        additional_paths = training_paths[current_count:next_end]
        additional_weights = training_weights[current_count:next_end]
        
        print(f"\n3. Adding {len(additional_paths)} more images (total: {next_end})...")
        
        try:
            # Continue training with additional images
            incremental_results = estimator.continue_training(additional_paths, additional_weights)
            
            print(f"Incremental training completed!")
            print(f"Additional data MAE: {incremental_results['mae']:.2f} kg")
            print(f"Additional data R²: {incremental_results['r2']:.3f}")
            
            # Test updated model
            if test_paths:
                test_results = estimator.evaluate(test_paths, test_weights)
                print(f"Updated test MAE: {test_results['mae']:.2f} kg")
                print(f"Updated test R²: {test_results['r2']:.3f}")
                
            # Record progress
            training_history.append({
                'total_samples': next_end,
                'incremental_mae': incremental_results['mae'],
                'incremental_r2': incremental_results['r2'],
                'test_mae': test_results['mae'] if test_paths else None,
                'test_r2': test_results['r2'] if test_paths else None
            })
            
            current_count = next_end
            
        except Exception as e:
            print(f"Error during incremental training: {e}")
            break
    
    # Print training history
    print(f"\n=== TRAINING PROGRESS SUMMARY ===")
    print(f"Progress: {current_count}/{len(training_paths)} training images used ({current_count/len(training_paths)*100:.1f}%)")
    print(f"{'Samples':<8} {'Test MAE':<10} {'Test R²':<8} {'Status'}")
    print("-" * 50)
    
    for i, history in enumerate(training_history):
        samples = history['total_samples']
        test_mae = history.get('test_mae', 'N/A')
        test_r2 = history.get('test_r2', 'N/A')
        status = "Initial" if i == 0 else f"Increment {i}"
        
        if test_mae != 'N/A':
            print(f"{samples:<8} {test_mae:<10.2f} {test_r2:<8.3f} {status}")
        else:
            print(f"{samples:<8} {'N/A':<10} {'N/A':<8} {status}")
    
    # Show final dataset utilization
    print(f"\n=== FINAL DATASET UTILIZATION ===")
    print(f"Total dataset: {total_images} images")
    print(f"Used for training: {current_count} images ({current_count/total_images*100:.1f}%)")
    print(f"Used for testing: {len(test_paths)} images ({len(test_paths)/total_images*100:.1f}%)")
    
    if current_count >= len(training_paths):
        print("🎉 SUCCESS: Used ALL available training data!")
    else:
        remaining = len(training_paths) - current_count
        print(f"⚠️  {remaining} training images remaining (stopped early)")
    
    # Save final model
    print(f"\n4. Saving final model...")
    model_filename = f"pig_weight_model_incremental_{current_count}samples.joblib"
    model_path = os.path.join("models", model_filename)
    
    os.makedirs("models", exist_ok=True)
    
    try:
        estimator.save_model(model_path)
        print(f"Final model saved to: {model_path}")
        print(f"Total training samples used: {current_count}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    return {
        'estimator': estimator,
        'training_history': training_history,
        'final_samples': current_count,
        'model_path': model_path
    }

def main():
    parser = argparse.ArgumentParser(description='Incremental training for pig weight estimation')
    parser.add_argument('--data_path', type=str, default='data', 
                       help='Path to PIGRGB-Weight dataset directory')
    parser.add_argument('--initial_samples', type=int, default=1000,
                       help='Number of images to start with')
    parser.add_argument('--increment_size', type=int, default=1000,
                       help='Number of images to add in each increment')
    parser.add_argument('--test_percentage', type=float, default=0.2,
                       help='Percentage of dataset to reserve for testing (0.2 = 20%)')
    
    args = parser.parse_args()
    
    # Check if data path exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data path '{args.data_path}' does not exist!")
        print("Please make sure the PIGRGB-Weight dataset is in the 'data' directory")
        return
    
    incremental_training_demo(
        data_path=args.data_path,
        initial_samples=args.initial_samples,
        increment_size=args.increment_size,
        test_percentage=args.test_percentage
    )

if __name__ == "__main__":
    main() 