import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re

class PigDatasetLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.image_paths = []
        self.weights = []
        
    def load_dataset(self):
        """Load the PIGRGB-Weight dataset"""
        print("Loading PIGRGB-Weight dataset...")
        
        # Walk through all folds and weight categories
        rgb_path = os.path.join(self.data_path, "RGB_9579")
        
        for fold in os.listdir(rgb_path):
            fold_path = os.path.join(rgb_path, fold)
            if not os.path.isdir(fold_path):
                continue
                
            for weight_category in os.listdir(fold_path):
                category_path = os.path.join(fold_path, weight_category)
                if not os.path.isdir(category_path):
                    continue
                
                # Extract weight from category name (e.g., "73.36_124" -> 73.36)
                weight_match = re.match(r"(\d+\.?\d*)_", weight_category)
                if not weight_match:
                    continue
                
                category_weight = float(weight_match.group(1))
                
                # Load all images in this category
                for img_file in os.listdir(category_path):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(category_path, img_file)
                        
                        # Extract weight from filename (e.g., "73.36kg_1.png" -> 73.36)
                        weight_match = re.match(r"(\d+\.?\d*)kg_", img_file)
                        if weight_match:
                            weight = float(weight_match.group(1))
                        else:
                            weight = category_weight
                        
                        self.image_paths.append(img_path)
                        self.weights.append(weight)
        
        print(f"Loaded {len(self.image_paths)} images with weights ranging from {min(self.weights):.2f}kg to {max(self.weights):.2f}kg")
        return self.image_paths, self.weights
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        """Split dataset into train and test sets"""
        if not self.image_paths:
            self.load_dataset()
        
        return train_test_split(
            self.image_paths, self.weights, 
            test_size=test_size, 
            random_state=random_state,
            stratify=None  # Can't stratify continuous values
        )
    
    def load_image(self, image_path):
        """Load and preprocess a single image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return img
    
    def get_dataset_statistics(self):
        """Get basic statistics about the dataset"""
        if not self.weights:
            self.load_dataset()
        
        stats = {
            'total_images': len(self.image_paths),
            'min_weight': min(self.weights),
            'max_weight': max(self.weights),
            'mean_weight': np.mean(self.weights),
            'std_weight': np.std(self.weights),
            'weight_distribution': pd.Series(self.weights).value_counts().head(10)
        }
        
        return stats 