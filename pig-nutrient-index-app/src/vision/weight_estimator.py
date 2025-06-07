import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from scipy import ndimage

class WeightEstimator:
    def __init__(self, model_type='mlp'):
        """
        Enhanced WeightEstimator using machine learning with image features.
        
        Parameters:
        model_type: 'mlp' for Multi-Layer Perceptron or 'rf' for Random Forest
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Initialize model
        if model_type == 'mlp':
            # BPNN with Trainlm-like configuration
            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='lbfgs',  # Similar to Trainlm
                alpha=0.001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=42
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

    def extract_pig_features(self, image, mask=None):
        """
        Extract morphological features from pig image.
        Based on the research paper's feature extraction method.
        
        Parameters:
        image: RGB image of pig
        mask: Binary mask of pig (if available)
        
        Returns:
        features: Dictionary of extracted features
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        
        if image is None:
            raise ValueError("Invalid image")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create mask if not provided
        if mask is None:
            mask = self.create_pig_mask(image)
        
        # Extract features
        features = {}
        
        # 1. Relative Projection Area (SR)
        total_pixels = image.shape[0] * image.shape[1]
        pig_pixels = np.sum(mask > 0)
        features['relative_projection_area'] = pig_pixels / total_pixels
        
        # 2. Contour-based features
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Contour length (LC)
            features['contour_length'] = cv2.arcLength(largest_contour, True)
            
            # Contour area
            features['contour_area'] = cv2.contourArea(largest_contour)
            
            # Bounding rectangle for body length and width
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Body length and width (Length, LW)
            width, height = rect[1]
            features['body_length'] = max(width, height)
            features['body_width'] = min(width, height)
            
            # Aspect ratio
            features['aspect_ratio'] = features['body_length'] / max(features['body_width'], 1)
            
            # Eccentricity (E)
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                a, b = ellipse[1][0]/2, ellipse[1][1]/2  # Semi-major and semi-minor axes
                if a > 0:
                    eccentricity = np.sqrt(1 - (min(a,b)**2 / max(a,b)**2))
                    features['eccentricity'] = eccentricity
                else:
                    features['eccentricity'] = 0
            else:
                features['eccentricity'] = 0
            
            # Convex hull features
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            features['convexity'] = features['contour_area'] / max(hull_area, 1)
            
            # Solidity
            features['solidity'] = features['contour_area'] / max(hull_area, 1)
            
        else:
            # Default values if no contours found
            features.update({
                'contour_length': 0,
                'contour_area': 0,
                'body_length': 0,
                'body_width': 0,
                'aspect_ratio': 1,
                'eccentricity': 0,
                'convexity': 0,
                'solidity': 0
            })
        
        # 3. Intensity-based features
        pig_region = gray[mask > 0]
        if len(pig_region) > 0:
            features['mean_intensity'] = np.mean(pig_region)
            features['std_intensity'] = np.std(pig_region)
            features['intensity_range'] = np.max(pig_region) - np.min(pig_region)
        else:
            features['mean_intensity'] = 0
            features['std_intensity'] = 0
            features['intensity_range'] = 0
        
        # 4. Texture features using Gray Level Co-occurrence Matrix (simplified)
        features['texture_contrast'] = self.calculate_texture_contrast(gray, mask)
        
        return features

    def create_pig_mask(self, image):
        """
        Create a binary mask for the pig using image processing.
        This is a simplified version - in practice, you'd use SAM2 or similar.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to separate pig from background
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest connected component (assumed to be the pig)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, [largest_contour], 255)
        
        return mask

    def calculate_texture_contrast(self, gray_image, mask):
        """Calculate texture contrast (simplified version)"""
        # Apply mask to get pig region only
        pig_region = gray_image.copy()
        pig_region[mask == 0] = 0
        
        # Calculate local variance as a measure of texture
        if np.sum(mask) > 0:
            # Use Sobel operators to detect edges
            sobel_x = cv2.Sobel(pig_region, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(pig_region, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Return mean edge strength in pig region
            return np.mean(sobel_combined[mask > 0])
        else:
            return 0

    def train(self, image_paths, weights):
        """
        Train the weight estimation model.
        
        Parameters:
        image_paths: List of paths to training images
        weights: List of corresponding weights
        """
        print("Extracting features from training images...")
        features_list = []
        valid_weights = []
        
        for i, (image_path, weight) in enumerate(zip(image_paths, weights)):
            try:
                if i % 100 == 0:
                    print(f"Processing image {i+1}/{len(image_paths)}")
                
                # Extract features
                features = self.extract_pig_features(image_path)
                features_list.append(list(features.values()))
                valid_weights.append(weight)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No valid features extracted from training data")
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(valid_weights)
        
        print(f"Training model with {len(X)} samples and {X.shape[1]} features...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training accuracy
        y_pred = self.model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        print(f"Training completed!")
        print(f"Training MAE: {mae:.2f} kg")
        print(f"Training R²: {r2:.3f}")
        
        return {'mae': mae, 'r2': r2}

    def estimate_weight(self, image_input):
        """
        Estimate weight from image.
        
        Parameters:
        image_input: Image path (string) or numpy array
        
        Returns:
        estimated_weight: Predicted weight in kg
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_pig_features(image_input)
        feature_vector = np.array([list(features.values())])
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict weight
        weight = self.model.predict(feature_vector_scaled)[0]
        
        return max(0, weight)  # Ensure non-negative weight

    def evaluate(self, test_image_paths, test_weights):
        """Evaluate model performance on test data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = []
        actual_weights = []
        
        print("Evaluating model on test data...")
        for i, (image_path, weight) in enumerate(zip(test_image_paths, test_weights)):
            try:
                if i % 50 == 0:
                    print(f"Evaluating image {i+1}/{len(test_image_paths)}")
                
                pred_weight = self.estimate_weight(image_path)
                predictions.append(pred_weight)
                actual_weights.append(weight)
                
            except Exception as e:
                print(f"Error evaluating {image_path}: {e}")
                continue
        
        if predictions:
            mae = mean_absolute_error(actual_weights, predictions)
            r2 = r2_score(actual_weights, predictions)
            
            print(f"Test MAE: {mae:.2f} kg")
            print(f"Test R²: {r2:.3f}")
            
            return {'mae': mae, 'r2': r2, 'predictions': predictions, 'actual': actual_weights}
        else:
            return {'mae': float('inf'), 'r2': 0, 'predictions': [], 'actual': []}

    def save_model(self, filepath):
        """Save trained model and scaler"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model and scaler"""
        if not os.path.exists(filepath):
            raise ValueError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")