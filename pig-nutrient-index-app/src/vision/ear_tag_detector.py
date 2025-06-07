import cv2
import numpy as np
from collections import defaultdict
import hashlib

class EarTagDetector:
    def __init__(self):
        """
        Ear tag detector for identifying individual pigs.
        Uses color-based detection and simple visual features.
        """
        self.color_ranges = {
            'red': (np.array([0, 50, 50]), np.array([10, 255, 255])),
            'green': (np.array([50, 50, 50]), np.array([70, 255, 255])),
            'blue': (np.array([100, 50, 50]), np.array([130, 255, 255])),
            'yellow': (np.array([20, 50, 50]), np.array([30, 255, 255])),
            'orange': (np.array([10, 50, 50]), np.array([20, 255, 255])),
            'purple': (np.array([130, 50, 50]), np.array([160, 255, 255]))
        }
        
        # Store pig features for consistent ID assignment
        self.pig_database = {}
        self.next_pig_id = 1

    def detect_ear_tags(self, image):
        """
        Detect colored ear tags in the image.
        
        Parameters:
        image: Input image (BGR format)
        
        Returns:
        List of detected ear tag information: [{'color': color, 'center': (x,y), 'area': area}]
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        
        if image is None:
            return []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        detected_tags = []
        
        for color_name, (lower, upper) in self.color_ranges.items():
            # Create mask for this color
            mask = cv2.inRange(hsv, lower, upper)
            
            # Apply morphological operations to clean up
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by size (ear tags should be reasonably sized)
                if 100 < area < 5000:
                    # Get center point
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Check if this tag is in the upper part of image (likely ear region)
                        if cy < image.shape[0] * 0.6:  # Upper 60% of image
                            detected_tags.append({
                                'color': color_name,
                                'center': (cx, cy),
                                'area': area,
                                'contour': contour
                            })
        
        return detected_tags

    def generate_pig_features(self, image, pig_region=None):
        """
        Generate visual features for pig identification when no ear tags are detected.
        
        Parameters:
        image: Input image
        pig_region: Optional mask or bounding box for pig region
        
        Returns:
        Feature dictionary for pig identification
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple features for pig identification
        features = {}
        
        # 1. Average color in different regions
        h, w = image.shape[:2]
        regions = {
            'top': image[0:h//3, :],
            'middle': image[h//3:2*h//3, :],
            'bottom': image[2*h//3:h, :],
            'left': image[:, 0:w//3],
            'center': image[:, w//3:2*w//3],
            'right': image[:, 2*w//3:w]
        }
        
        for region_name, region in regions.items():
            avg_color = np.mean(region.reshape(-1, 3), axis=0)
            features[f'{region_name}_avg_color'] = tuple(avg_color.astype(int))
        
        # 2. Texture features
        # Calculate Local Binary Pattern-like features
        features['texture_variance'] = np.var(gray)
        
        # 3. Shape features (if pig region is provided)
        if pig_region is not None:
            if len(pig_region.shape) == 2:  # It's a mask
                contours, _ = cv2.findContours(pig_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    features['contour_area'] = cv2.contourArea(largest_contour)
                    features['contour_perimeter'] = cv2.arcLength(largest_contour, True)
        
        return features

    def assign_pig_id(self, image, ear_tags=None):
        """
        Assign a consistent ID to a pig based on ear tags or visual features.
        
        Parameters:
        image: Input image
        ear_tags: List of detected ear tags (from detect_ear_tags)
        
        Returns:
        pig_id: String identifier for the pig
        """
        # Method 1: Use ear tag colors if available
        if ear_tags and len(ear_tags) > 0:
            # Sort ear tags by position for consistency
            ear_tags_sorted = sorted(ear_tags, key=lambda x: (x['center'][0], x['center'][1]))
            
            # Create ID based on color combination
            color_combination = "_".join([tag['color'] for tag in ear_tags_sorted])
            pig_id = f"pig_{color_combination}"
            
            return pig_id
        
        # Method 2: Use visual features for identification
        features = self.generate_pig_features(image)
        
        # Create a hash of key features for consistent ID
        feature_string = str(sorted(features.items()))
        feature_hash = hashlib.md5(feature_string.encode()).hexdigest()[:8]
        
        # Try to match with existing pigs
        for existing_id, existing_features in self.pig_database.items():
            # Simple similarity check
            similarity_score = self.calculate_feature_similarity(features, existing_features)
            if similarity_score > 0.8:  # 80% similarity threshold
                return existing_id
        
        # Create new pig ID
        new_pig_id = f"pig_{self.next_pig_id:03d}"
        self.pig_database[new_pig_id] = features
        self.next_pig_id += 1
        
        return new_pig_id

    def calculate_feature_similarity(self, features1, features2):
        """Calculate similarity between two feature sets"""
        if not features1 or not features2:
            return 0.0
        
        # Simple similarity based on color features
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        total_similarity = 0
        for key in common_keys:
            if 'avg_color' in key:
                # Calculate color distance
                color1 = np.array(features1[key])
                color2 = np.array(features2[key])
                distance = np.linalg.norm(color1 - color2)
                # Convert distance to similarity (0-1)
                similarity = max(0, 1 - distance / 255.0)
                total_similarity += similarity
            elif key == 'texture_variance':
                # Normalize texture variance similarity
                var1, var2 = features1[key], features2[key]
                max_var = max(var1, var2, 1)
                similarity = 1 - abs(var1 - var2) / max_var
                total_similarity += similarity
        
        return total_similarity / len(common_keys) if common_keys else 0.0

    def detect_and_identify_pig(self, image):
        """
        Main method to detect and identify a pig in the image.
        
        Parameters:
        image: Input image
        
        Returns:
        pig_info: Dictionary with pig_id and detection details
        """
        # Detect ear tags
        ear_tags = self.detect_ear_tags(image)
        
        # Assign pig ID
        pig_id = self.assign_pig_id(image, ear_tags)
        
        # Prepare result
        result = {
            'pig_id': pig_id,
            'ear_tags': ear_tags,
            'confidence': 1.0 if ear_tags else 0.7,  # Higher confidence with ear tags
            'detection_method': 'ear_tag' if ear_tags else 'visual_features'
        }
        
        return result

    def visualize_detection(self, image, pig_info):
        """
        Draw detection results on the image for visualization.
        
        Parameters:
        image: Input image
        pig_info: Result from detect_and_identify_pig
        
        Returns:
        annotated_image: Image with detection results drawn
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        
        result_image = image.copy()
        
        # Draw ear tags if detected
        if pig_info['ear_tags']:
            for tag in pig_info['ear_tags']:
                center = tag['center']
                color = tag['color']
                
                # Color mapping for visualization
                color_bgr = {
                    'red': (0, 0, 255),
                    'green': (0, 255, 0),
                    'blue': (255, 0, 0),
                    'yellow': (0, 255, 255),
                    'orange': (0, 165, 255),
                    'purple': (128, 0, 128)
                }.get(color, (255, 255, 255))
                
                # Draw circle and label
                cv2.circle(result_image, center, 10, color_bgr, -1)
                cv2.circle(result_image, center, 12, (0, 0, 0), 2)
                cv2.putText(result_image, color, (center[0]-20, center[1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        
        # Draw pig ID
        cv2.putText(result_image, pig_info['pig_id'], (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw confidence and method
        confidence_text = f"Confidence: {pig_info['confidence']:.2f} ({pig_info['detection_method']})"
        cv2.putText(result_image, confidence_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image

    def reset_database(self):
        """Reset the pig identification database"""
        self.pig_database = {}
        self.next_pig_id = 1
        print("Pig identification database reset")

    def get_known_pigs(self):
        """Return list of known pig IDs"""
        return list(self.pig_database.keys()) 