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
    def __init__(self, model_type='mlp', tune_hyperparams=False):
        """
        Enhanced WeightEstimator using machine learning with image features.
        
        Parameters:
        model_type: 'mlp' for Multi-Layer Perceptron or 'rf' for Random Forest
        tune_hyperparams: Whether to use hyperparameter tuning
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.tune_hyperparams = tune_hyperparams
        
        # Initialize model with improved hyperparameters
        if model_type == 'mlp':
            if tune_hyperparams:
                # Will be set during training with GridSearchCV
                self.model = None
            else:
                # Improved hyperparameters for Neural Network
                self.model = MLPRegressor(
                    hidden_layer_sizes=(200, 100, 50),  # Deeper network
                    activation='relu',
                    solver='adam',  # Better for larger datasets
                    alpha=0.0001,  # Lower regularization
                    batch_size=32,  # Fixed batch size
                    learning_rate='adaptive',  # Adaptive learning rate
                    learning_rate_init=0.001,
                    max_iter=2000,  # More iterations
                    # Note: early_stopping removed for incremental learning compatibility
                    random_state=42
                )
        else:
            if tune_hyperparams:
                # Will be set during training with GridSearchCV
                self.model = None
            else:
                # Improved hyperparameters for Random Forest
                self.model = RandomForestRegressor(
                    n_estimators=200,  # More trees
                    max_depth=15,  # Deeper trees
                    min_samples_split=5,  # Prevent overfitting
                    min_samples_leaf=2,
                    max_features='sqrt',  # Feature sampling
                    bootstrap=True,
                    oob_score=True,  # Out-of-bag validation
                    n_jobs=-1,  # Use all cores
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
        
        # 5. Additional advanced features for better accuracy
        features.update(self.extract_advanced_features(image, gray, mask))
        
        return features

    def _tune_hyperparameters(self, X, y):
        """Perform hyperparameter tuning using GridSearchCV"""
        from sklearn.model_selection import GridSearchCV
        
        if self.model_type == 'mlp':
            # MLP hyperparameter grid
            param_grid = {
                'hidden_layer_sizes': [
                    (100, 50), (200, 100), (200, 100, 50), (300, 150, 75)
                ],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'batch_size': [32, 64, 128]
            }
            base_model = MLPRegressor(
                activation='relu',
                solver='adam',
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            )
        else:
            # Random Forest hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            base_model = RandomForestRegressor(
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=42
            )
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,  # 3-fold cross-validation
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best hyperparameters: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.2f} MAE")
        
        return grid_search.best_estimator_

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

    def extract_advanced_features(self, image, gray, mask):
        """Extract additional advanced features for improved accuracy"""
        advanced_features = {}
        
        # Color-based features
        if len(image.shape) == 3:
            # HSV color space features
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            pig_region_mask = mask > 0
            
            for i, channel in enumerate(['hue', 'saturation', 'value']):
                channel_data = hsv[:, :, i][pig_region_mask]
                if len(channel_data) > 0:
                    advanced_features[f'{channel}_mean'] = np.mean(channel_data)
                    advanced_features[f'{channel}_std'] = np.std(channel_data)
                else:
                    advanced_features[f'{channel}_mean'] = 0
                    advanced_features[f'{channel}_std'] = 0
        
        # Moments-based features
        if np.sum(mask) > 0:
            moments = cv2.moments(mask)
            if moments['m00'] > 0:
                # Centroid
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                # Normalized central moments (Hu moments)
                hu_moments = cv2.HuMoments(moments).flatten()
                for i, hu in enumerate(hu_moments):
                    advanced_features[f'hu_moment_{i}'] = -np.sign(hu) * np.log10(np.abs(hu)) if hu != 0 else 0
            else:
                for i in range(7):
                    advanced_features[f'hu_moment_{i}'] = 0
        
        # Edge density features
        edges = cv2.Canny(gray, 50, 150)
        pig_edges = edges[mask > 0]
        if len(pig_edges) > 0:
            advanced_features['edge_density'] = np.sum(pig_edges > 0) / len(pig_edges)
        else:
            advanced_features['edge_density'] = 0
        
        # Local Binary Pattern features (simplified)
        if np.sum(mask) > 100:  # Only if pig region is large enough
            radius = 3
            n_points = 8 * radius
            # Simple LBP implementation
            pig_region = gray.copy()
            pig_region[mask == 0] = 0
            
            # Calculate variance of local patterns
            kernel = np.ones((3, 3), np.float32) / 9
            local_mean = cv2.filter2D(pig_region.astype(np.float32), -1, kernel)
            local_var = cv2.filter2D((pig_region.astype(np.float32) - local_mean) ** 2, -1, kernel)
            
            pig_local_var = local_var[mask > 0]
            if len(pig_local_var) > 0:
                advanced_features['texture_variance'] = np.mean(pig_local_var)
            else:
                advanced_features['texture_variance'] = 0
        else:
            advanced_features['texture_variance'] = 0
        
        # Shape complexity features
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Perimeter to area ratio
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            if area > 0:
                advanced_features['perimeter_area_ratio'] = perimeter / np.sqrt(area)
                
                # Compactness (isoperimetric quotient)
                advanced_features['compactness'] = (4 * np.pi * area) / (perimeter ** 2)
            else:
                advanced_features['perimeter_area_ratio'] = 0
                advanced_features['compactness'] = 0
        else:
            advanced_features['perimeter_area_ratio'] = 0
            advanced_features['compactness'] = 0
        
        return advanced_features

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
        
        # Process in smaller batches to prevent memory issues
        batch_size = 50
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]
            batch_weights = weights[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx-1})")
            
            for i, (image_path, weight) in enumerate(zip(batch_paths, batch_weights)):
                try:
                    global_idx = start_idx + i
                    if global_idx % 10 == 0:
                        print(f"  Processing image {global_idx+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                    
                    # Extract features
                    features = self.extract_pig_features(image_path)
                    features_list.append(list(features.values()))
                    valid_weights.append(weight)
                    
                except Exception as e:
                    print(f"  Error processing {image_path}: {e}")
                    continue
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
        
        if not features_list:
            raise ValueError("No valid features extracted from training data")
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(valid_weights)
        
        print(f"Training model with {len(X)} samples and {X.shape[1]} features...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model with optional hyperparameter tuning
        if self.tune_hyperparams:
            print("Performing hyperparameter tuning...")
            self.model = self._tune_hyperparameters(X_scaled, y)
        
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

    def continue_training(self, additional_image_paths, additional_weights):
        """
        Continue training with additional images (only works with MLPRegressor).
        
        Parameters:
        additional_image_paths: List of paths to additional training images
        additional_weights: List of corresponding weights for additional images
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first before continuing training")
        
        if self.model_type != 'mlp':
            raise ValueError("Incremental training only supported for MLPRegressor (mlp). RandomForest requires full retraining.")
        
        print(f"Continuing training with {len(additional_image_paths)} additional images...")
        
        # Extract features from additional images
        features_list = []
        valid_weights = []
        
        # Process in batches
        batch_size = 50
        total_batches = (len(additional_image_paths) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(additional_image_paths))
            batch_paths = additional_image_paths[start_idx:end_idx]
            batch_weights = additional_weights[start_idx:end_idx]
            
            print(f"Processing additional batch {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx-1})")
            
            for i, (image_path, weight) in enumerate(zip(batch_paths, batch_weights)):
                try:
                    global_idx = start_idx + i
                    if global_idx % 10 == 0:
                        print(f"  Processing additional image {global_idx+1}/{len(additional_image_paths)}: {os.path.basename(image_path)}")
                    
                    # Extract features
                    features = self.extract_pig_features(image_path)
                    features_list.append(list(features.values()))
                    valid_weights.append(weight)
                    
                except Exception as e:
                    print(f"  Error processing {image_path}: {e}")
                    continue
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
        
        if not features_list:
            print("No valid features extracted from additional training data")
            return {'mae': None, 'r2': None}
        
        # Convert to numpy arrays
        X_additional = np.array(features_list)
        y_additional = np.array(valid_weights)
        
        print(f"Continuing training with {len(X_additional)} additional samples...")
        
        # Scale additional features using existing scaler
        X_additional_scaled = self.scaler.transform(X_additional)
        
        # Continue training using partial_fit
        self.model.partial_fit(X_additional_scaled, y_additional)
        
        # Calculate performance on additional data
        y_pred_additional = self.model.predict(X_additional_scaled)
        mae_additional = mean_absolute_error(y_additional, y_pred_additional)
        r2_additional = r2_score(y_additional, y_pred_additional)
        
        print(f"Incremental training completed!")
        print(f"Additional data MAE: {mae_additional:.2f} kg")
        print(f"Additional data R²: {r2_additional:.3f}")
        
        return {'mae': mae_additional, 'r2': r2_additional}

    def get_remaining_images(self, all_image_paths, all_weights, used_count):
        """
        Get the remaining images that haven't been used for training yet.
        
        Parameters:
        all_image_paths: Complete list of image paths
        all_weights: Complete list of weights
        used_count: Number of images already used in training
        
        Returns:
        remaining_paths, remaining_weights: Unused images and weights
        """
        if used_count >= len(all_image_paths):
            return [], []
        
        return all_image_paths[used_count:], all_weights[used_count:]