# Configuration settings for the pig nutrient index application

# Model paths for weight estimation and QR detection
WEIGHT_ESTIMATOR_MODEL_PATH = "path/to/weight_estimator/model"
QR_DETECTOR_MODEL_PATH = "path/to/qr_detector/model"

# Thresholds for nutrient index calculations
NUTRIENT_INDEX_LOW_THRESHOLD = 0
NUTRIENT_INDEX_HIGH_THRESHOLD = 100

# Nutrient requirements (example values, adjust as necessary)
NUTRIENT_REQUIREMENTS = {
    "protein": {
        "low": 10,  # grams
        "high": 20  # grams
    },
    "fat": {
        "low": 5,   # grams
        "high": 15  # grams
    },
    "fiber": {
        "low": 3,   # grams
        "high": 10  # grams
    }
}

# Other constants
IMAGE_RESIZE_DIMENSIONS = (224, 224)  # Resize images to this dimension for processing
QR_CODE_DETECTION_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for QR code detection

# Logging configuration
LOGGING_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"  # Format for log messages