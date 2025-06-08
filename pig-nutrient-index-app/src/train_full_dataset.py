#!/usr/bin/env python3
"""
Full Dataset Training Script - Train on ALL images at once.
Uses the enhanced WeightEstimator with batch processing and memory optimization.
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

def plot_results(actual, predicted, model_type="Neural Network"):
    """Plot actual vs predicted weights and save the results"""
    plt.figure(figsize=(12, 9))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(actual, predicted, alpha=0.6, s=20)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', lw=2)
    plt.xlabel('Actual Weight (kg)')
    plt.ylabel('Predicted Weight (kg)')
    plt.title('Actual vs Predicted Weights')
    plt.grid(True, alpha=0.3)
    
    # Add R² to the plot
    r2 = r2_score(actual, predicted)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    # Residuals plot
    plt.subplot(2, 2, 2)
    residuals = np.array(predicted) - np.array(actual)
    plt.scatter(actual, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Weight (kg)')
    plt.ylabel('Residuals (kg)')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals (kg)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Weight distribution comparison
    plt.subplot(2, 2, 4)
    plt.hist(actual, bins=30, alpha=0.7, label='Actual', color='blue', edgecolor='black')
    plt.hist(predicted, bins=30, alpha=0.7, label='Predicted', color='red', edgecolor='black')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f'full_dataset_results_{model_type.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Results plot saved as: {filename}")
    plt.show()

def train_full_dataset(data_path="data", model_type='mlp', test_size=0.2, tune_hyperparams=False):
    """
    Train on the complete PIGRGB-Weight dataset.
    
    Parameters:
    data_path: Path to the PIGRGB-Weight dataset
    model_type: 'mlp' for Neural Network or 'rf' for Random Forest
    test_size: Fraction of data to use for testing
    tune_hyperparams: Whether to perform hyperparameter tuning
    """
    print("🚀 === FULL DATASET TRAINING ===")
    print(f"Model type: {model_type.upper()}")
    print(f"Data path: {data_path}")
    print(f"Test size: {test_size*100}%")
    print(f"Hyperparameter tuning: {'ON' if tune_hyperparams else 'OFF'}")
    
    # Load complete dataset
    print("\n📂 1. Loading complete dataset...")
    loader = PigDatasetLoader(data_path)
    
    try:
        image_paths, weights = loader.load_dataset()
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    if not image_paths:
        print("❌ No images found in dataset!")
        return None
    
    # Print dataset statistics
    stats = loader.get_dataset_statistics()
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total images: {stats['total_images']:,}")
    print(f"   Weight range: {stats['min_weight']:.2f} - {stats['max_weight']:.2f} kg")
    print(f"   Mean weight: {stats['mean_weight']:.2f} ± {stats['std_weight']:.2f} kg")
    
    print(f"\n🎯 Using FULL dataset: {len(image_paths):,} images")
    
    # Split dataset
    print(f"\n✂️  2. Splitting dataset...")
    train_paths, test_paths, train_weights, test_weights = loader.get_train_test_split(test_size=test_size)
    
    print(f"   Training samples: {len(train_paths):,}")
    print(f"   Test samples: {len(test_paths):,}")
    
    # Initialize and train model
    print(f"\n🧠 3. Training {model_type.upper()} model on full dataset...")
    print("   This may take some time - grab a coffee! ☕")
    
    estimator = WeightEstimator(model_type=model_type, tune_hyperparams=tune_hyperparams)
    
    try:
        print(f"\n⏳ Starting training with {len(train_paths):,} images...")
        training_results = estimator.train(train_paths, train_weights)
        print(f"✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None
    
    # Evaluate on test set
    print(f"\n📈 4. Evaluating on test set ({len(test_paths):,} images)...")
    try:
        test_results = estimator.evaluate(test_paths, test_weights)
        
        print(f"\n🎉 === FINAL RESULTS ===")
        print(f"Training samples: {len(train_paths):,}")
        print(f"Test samples: {len(test_paths):,}")
        print(f"Features extracted: {len(estimator.extract_pig_features(train_paths[0]).keys())}")
        print(f"")
        print(f"📊 TRAINING PERFORMANCE:")
        print(f"   MAE: {training_results['mae']:.2f} kg")
        print(f"   R²:  {training_results['r2']:.3f}")
        print(f"")
        print(f"🎯 TEST PERFORMANCE:")
        print(f"   MAE:  {test_results['mae']:.2f} kg")
        print(f"   R²:   {test_results['r2']:.3f}")
        
        # Calculate additional metrics
        if test_results['predictions']:
            actual = np.array(test_results['actual'])
            predicted = np.array(test_results['predictions'])
            
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            
            print(f"   RMSE: {rmse:.2f} kg")
            print(f"   MAPE: {mape:.2f}%")
            
            # Error statistics
            errors = np.abs(actual - predicted)
            print(f"")
            print(f"📉 ERROR STATISTICS:")
            print(f"   Mean error:   {np.mean(errors):.2f} kg")
            print(f"   Median error: {np.median(errors):.2f} kg")
            print(f"   90th percentile: {np.percentile(errors, 90):.2f} kg")
            print(f"   Max error:    {np.max(errors):.2f} kg")
            
            # Create and save plots
            print(f"\n📊 5. Generating results visualization...")
            plot_results(test_results['actual'], test_results['predictions'], 
                        f"{model_type.upper()} Full Dataset")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None
    
    # Save model
    print(f"\n💾 6. Saving model...")
    model_filename = f"pig_weight_model_{model_type}_full_dataset{'_tuned' if tune_hyperparams else ''}.joblib"
    model_path = os.path.join("models", model_filename)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    try:
        estimator.save_model(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        # Print file size
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"   File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
    
    # Final summary
    print(f"\n🏆 === TRAINING COMPLETE ===")
    print(f"✅ Successfully trained on {len(train_paths):,} images")
    print(f"✅ Achieved {test_results['mae']:.2f} kg MAE on test set")
    print(f"✅ Model saved and ready for use")
    print(f"")
    print(f"🚀 Ready for your hackathon presentation!")
    
    return {
        'estimator': estimator,
        'training_results': training_results,
        'test_results': test_results,
        'model_path': model_path,
        'total_samples': len(train_paths)
    }

def main():
    parser = argparse.ArgumentParser(description='Train pig weight estimation model on FULL dataset')
    parser.add_argument('--data_path', type=str, default='data', 
                       help='Path to PIGRGB-Weight dataset directory')
    parser.add_argument('--model_type', type=str, choices=['mlp', 'rf'], default='mlp',
                       help='Model type: mlp (Neural Network) or rf (Random Forest)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data to use for testing (default: 0.2)')
    parser.add_argument('--tune_hyperparams', action='store_true',
                       help='Perform hyperparameter tuning (slower but potentially better)')
    
    args = parser.parse_args()
    
    # Check if data path exists
    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data path '{args.data_path}' does not exist!")
        print("Please make sure the PIGRGB-Weight dataset is in the 'data' directory")
        return
    
    # Run training
    result = train_full_dataset(
        data_path=args.data_path,
        model_type=args.model_type,
        test_size=args.test_size,
        tune_hyperparams=args.tune_hyperparams
    )
    
    if result:
        print(f"\n🎊 Training successful! Check your results.")
    else:
        print(f"\n💥 Training failed. Check the error messages above.")

if __name__ == "__main__":
    main() 