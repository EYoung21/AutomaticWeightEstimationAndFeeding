#!/usr/bin/env python3
"""
Model inspection utility - quickly check stats of any saved model.
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

def inspect_model(model_path, data_path=None, test_samples=500):
    """
    Load a model and show its current performance stats.
    
    Parameters:
    model_path: Path to the saved model file
    data_path: Path to dataset (for testing if desired)
    test_samples: Number of test samples to use for evaluation
    """
    print("=== Model Inspection Utility ===")
    print(f"Model file: {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        return None
    
    # Load the model
    print("\n1. Loading model...")
    try:
        estimator = WeightEstimator()
        estimator.load_model(model_path)
        
        print(f"✅ Model loaded successfully!")
        print(f"Model type: {estimator.model_type}")
        print(f"Training status: {'Trained' if estimator.is_trained else 'Not trained'}")
        
        # Show model details
        if hasattr(estimator.model, 'n_estimators'):
            print(f"Random Forest - Trees: {estimator.model.n_estimators}")
        elif hasattr(estimator.model, 'hidden_layer_sizes'):
            print(f"Neural Network - Layers: {estimator.model.hidden_layer_sizes}")
            if hasattr(estimator.model, 'n_iter_'):
                print(f"Training iterations: {estimator.model.n_iter_}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # If dataset provided, test the model
    if data_path and os.path.exists(data_path):
        print(f"\n2. Testing model on dataset...")
        
        try:
            loader = PigDatasetLoader(data_path)
            all_image_paths, all_weights = loader.load_dataset()
            
            print(f"Dataset loaded: {len(all_image_paths)} images")
            
            # Use random sample for quick testing
            if len(all_image_paths) > test_samples:
                print(f"Using random sample of {test_samples} images for quick evaluation...")
                indices = np.random.choice(len(all_image_paths), test_samples, replace=False)
                test_paths = [all_image_paths[i] for i in indices]
                test_weights = [all_weights[i] for i in indices]
            else:
                test_paths = all_image_paths
                test_weights = all_weights
            
            # Evaluate model
            print("Evaluating model performance...")
            results = estimator.evaluate(test_paths, test_weights)
            
            print(f"\n=== MODEL PERFORMANCE ===")
            print(f"Test samples: {len(test_paths)}")
            print(f"Mean Absolute Error: {results['mae']:.2f} kg")
            print(f"R² Score: {results['r2']:.3f}")
            
            if results['predictions']:
                # Additional statistics
                actual = np.array(results['actual'])
                predicted = np.array(results['predictions'])
                
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                
                print(f"Root Mean Square Error: {rmse:.2f} kg")
                print(f"Mean Absolute Percentage Error: {mape:.2f}%")
                
                # Weight range statistics
                print(f"\nWeight range in test set:")
                print(f"  Actual: {np.min(actual):.1f} - {np.max(actual):.1f} kg")
                print(f"  Predicted: {np.min(predicted):.1f} - {np.max(predicted):.1f} kg")
                
                # Error distribution
                errors = np.abs(actual - predicted)
                print(f"\nError distribution:")
                print(f"  Mean error: {np.mean(errors):.2f} kg")
                print(f"  Median error: {np.median(errors):.2f} kg")
                print(f"  90th percentile error: {np.percentile(errors, 90):.2f} kg")
                print(f"  Max error: {np.max(errors):.2f} kg")
                
        except Exception as e:
            print(f"❌ Error testing model: {e}")
    
    else:
        print(f"\n⚠️  No dataset provided for testing")
        print(f"Use --data_path to test model performance")
    
    return estimator

def list_available_models(models_dir="models"):
    """List all available model files"""
    print("=== Available Models ===")
    
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return []
    
    model_files = []
    for file in os.listdir(models_dir):
        if file.endswith('.joblib'):
            model_path = os.path.join(models_dir, file)
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            model_files.append((file, file_size))
    
    if not model_files:
        print(f"❌ No model files found in {models_dir}")
        return []
    
    print(f"Found {len(model_files)} model files:")
    print(f"{'Filename':<50} {'Size (MB)':<10}")
    print("-" * 65)
    
    for filename, size in sorted(model_files):
        print(f"{filename:<50} {size:<10.2f}")
    
    return [f[0] for f in model_files]

def quick_comparison(models_dir="models", data_path=None, test_samples=200):
    """Quickly compare all available models"""
    print("=== Quick Model Comparison ===")
    
    model_files = list_available_models(models_dir)
    if not model_files:
        return
    
    if not data_path or not os.path.exists(data_path):
        print("❌ Need valid dataset path for comparison")
        return
    
    print(f"\nComparing models using {test_samples} test images...")
    print(f"{'Model':<30} {'MAE (kg)':<10} {'R²':<8} {'MAPE (%)':<10}")
    print("-" * 65)
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        
        try:
            # Load model
            estimator = WeightEstimator()
            estimator.load_model(model_path)
            
            # Quick test
            loader = PigDatasetLoader(data_path)
            all_paths, all_weights = loader.load_dataset()
            
            # Random sample
            indices = np.random.choice(len(all_paths), min(test_samples, len(all_paths)), replace=False)
            test_paths = [all_paths[i] for i in indices]
            test_weights = [all_weights[i] for i in indices]
            
            results = estimator.evaluate(test_paths, test_weights)
            
            if results['predictions']:
                actual = np.array(results['actual'])
                predicted = np.array(results['predictions'])
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                
                model_name = model_file.replace('pig_weight_model_', '').replace('.joblib', '')
                print(f"{model_name:<30} {results['mae']:<10.2f} {results['r2']:<8.3f} {mape:<10.2f}")
            
        except Exception as e:
            model_name = model_file.replace('pig_weight_model_', '').replace('.joblib', '')
            print(f"{model_name:<30} {'ERROR':<10} {'ERROR':<8} {'ERROR':<10}")

def main():
    parser = argparse.ArgumentParser(description='Inspect saved pig weight estimation models')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to specific model file to inspect')
    parser.add_argument('--data_path', type=str, default='data',
                       help='Path to PIGRGB-Weight dataset directory')
    parser.add_argument('--test_samples', type=int, default=500,
                       help='Number of test samples for evaluation')
    parser.add_argument('--list_models', action='store_true',
                       help='List all available models')
    parser.add_argument('--compare_all', action='store_true',
                       help='Compare all available models')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory containing model files')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models(args.models_dir)
        
    elif args.compare_all:
        quick_comparison(args.models_dir, args.data_path, args.test_samples)
        
    elif args.model_path:
        inspect_model(args.model_path, args.data_path, args.test_samples)
        
    else:
        # Show available models and prompt for selection
        model_files = list_available_models(args.models_dir)
        
        if model_files:
            print(f"\nTo inspect a specific model, use:")
            print(f"python src/inspect_model.py --model_path models/FILENAME")
            print(f"\nOr compare all models:")
            print(f"python src/inspect_model.py --compare_all")

if __name__ == "__main__":
    main() 