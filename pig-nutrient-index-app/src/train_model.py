#!/usr/bin/env python3
"""
Training script for pig weight estimation model using the PIGRGB-Weight dataset.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_loader import PigDatasetLoader
from vision.weight_estimator import WeightEstimator
import config

def plot_results(actual, predicted, title="Weight Estimation Results"):
    """Plot actual vs predicted weights"""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(actual, predicted, alpha=0.6)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', lw=2)
    plt.xlabel('Actual Weight (kg)')
    plt.ylabel('Predicted Weight (kg)')
    plt.title('Actual vs Predicted Weights')
    plt.grid(True)
    
    # Residuals plot
    plt.subplot(2, 2, 2)
    residuals = np.array(predicted) - np.array(actual)
    plt.scatter(actual, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Weight (kg)')
    plt.ylabel('Residuals (kg)')
    plt.title('Residuals Plot')
    plt.grid(True)
    
    # Error distribution
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel('Residuals (kg)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    
    # Weight distribution
    plt.subplot(2, 2, 4)
    plt.hist(actual, bins=30, alpha=0.7, label='Actual', color='blue')
    plt.hist(predicted, bins=30, alpha=0.7, label='Predicted', color='red')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_results_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_and_evaluate(data_path, model_type='mlp', test_size=0.2, max_samples=None, tune_hyperparams=False):
    """
    Train and evaluate the weight estimation model.
    
    Parameters:
    data_path: Path to the PIGRGB-Weight dataset
    model_type: 'mlp' or 'rf' (Random Forest)
    test_size: Fraction of data to use for testing
    max_samples: Maximum number of samples to use (for faster testing)
    tune_hyperparams: Whether to perform hyperparameter tuning
    """
    print("=== Pig Weight Estimation Model Training ===")
    print(f"Model type: {model_type}")
    print(f"Data path: {data_path}")
    print(f"Test size: {test_size}")
    print(f"Hyperparameter tuning: {tune_hyperparams}")
    
    # Load dataset
    print("\n1. Loading dataset...")
    loader = PigDatasetLoader(data_path)
    
    try:
        image_paths, weights = loader.load_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    if not image_paths:
        print("No images found in dataset!")
        return None
    
    # Print dataset statistics
    stats = loader.get_dataset_statistics()
    print(f"\nDataset Statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"Weight range: {stats['min_weight']:.2f} - {stats['max_weight']:.2f} kg")
    print(f"Mean weight: {stats['mean_weight']:.2f} ± {stats['std_weight']:.2f} kg")
    
    # Limit samples if specified (for faster testing)
    if max_samples and len(image_paths) > max_samples:
        print(f"\nLimiting to {max_samples} samples for faster training...")
        indices = np.random.choice(len(image_paths), max_samples, replace=False)
        image_paths = [image_paths[i] for i in indices]
        weights = [weights[i] for i in indices]
    else:
        print(f"\nUsing full dataset: {len(image_paths)} samples")
    
    # Split dataset
    print(f"\n2. Splitting dataset...")
    train_paths, test_paths, train_weights, test_weights = loader.get_train_test_split(test_size=test_size)
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Test samples: {len(test_paths)}")
    
    # Initialize and train model
    print(f"\n3. Training {model_type.upper()} model...")
    estimator = WeightEstimator(model_type=model_type, tune_hyperparams=tune_hyperparams)
    
    try:
        training_results = estimator.train(train_paths, train_weights)
        print(f"Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return None
    
    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    try:
        test_results = estimator.evaluate(test_paths, test_weights)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Training MAE: {training_results['mae']:.2f} kg")
        print(f"Training R²: {training_results['r2']:.3f}")
        print(f"Test MAE: {test_results['mae']:.2f} kg")
        print(f"Test R²: {test_results['r2']:.3f}")
        
        # Calculate additional metrics
        if test_results['predictions']:
            mape = np.mean(np.abs((np.array(test_results['actual']) - np.array(test_results['predictions'])) / np.array(test_results['actual']))) * 100
            print(f"Test MAPE: {mape:.2f}%")
            
            # Plot results
            plot_results(test_results['actual'], test_results['predictions'], f"{model_type.upper()} Model")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None
    
    # Save model
    print("\n5. Saving model...")
    model_filename = f"pig_weight_model_{model_type}{'_tuned' if tune_hyperparams else ''}.joblib"
    model_path = os.path.join("models", model_filename)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    try:
        estimator.save_model(model_path)
        print(f"Model saved to: {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    return {
        'estimator': estimator,
        'training_results': training_results,
        'test_results': test_results,
        'model_path': model_path
    }

def compare_models(data_path, max_samples=1000):
    """Compare different model types"""
    print("=== Comparing Model Types ===")
    
    results = {}
    
    for model_type in ['mlp', 'rf']:
        print(f"\n--- Training {model_type.upper()} model ---")
        result = train_and_evaluate(data_path, model_type=model_type, max_samples=max_samples)
        if result:
            results[model_type] = result
    
    # Compare results
    if len(results) > 1:
        print("\n=== MODEL COMPARISON ===")
        print(f"{'Model':<10} {'Train MAE':<12} {'Train R²':<10} {'Test MAE':<12} {'Test R²':<10}")
        print("-" * 60)
        
        for model_type, result in results.items():
            train_mae = result['training_results']['mae']
            train_r2 = result['training_results']['r2']
            test_mae = result['test_results']['mae']
            test_r2 = result['test_results']['r2']
            
            print(f"{model_type.upper():<10} {train_mae:<12.2f} {train_r2:<10.3f} {test_mae:<12.2f} {test_r2:<10.3f}")

def main():
    parser = argparse.ArgumentParser(description='Train pig weight estimation model')
    parser.add_argument('--data_path', type=str, default='data', 
                       help='Path to PIGRGB-Weight dataset directory')
    parser.add_argument('--model_type', type=str, choices=['mlp', 'rf'], default='mlp',
                       help='Model type: mlp (Neural Network) or rf (Random Forest)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use (for faster testing)')
    parser.add_argument('--tune_hyperparams', action='store_true',
                       help='Perform hyperparameter tuning (slower but better accuracy)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare different model types')
    
    args = parser.parse_args()
    
    # Check if data path exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data path '{args.data_path}' does not exist!")
        print("Please make sure the PIGRGB-Weight dataset is in the 'data' directory")
        return
    
    if args.compare:
        compare_models(args.data_path, args.max_samples or 1000)
    else:
        train_and_evaluate(
            data_path=args.data_path,
            model_type=args.model_type,
            test_size=args.test_size,
            max_samples=args.max_samples,
            tune_hyperparams=args.tune_hyperparams
        )

if __name__ == "__main__":
    main() 