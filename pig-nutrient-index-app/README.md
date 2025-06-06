# Pig Nutrient Index Application

## Overview
The Pig Nutrient Index Application is designed to estimate the body weight of pigs using computer vision techniques and to manage their nutrient intake through automated Calan gate feeders. By utilizing unique QR codes imprinted on each pig's back, the application ensures that each pig receives the appropriate amount of food based on its individual nutrient needs.

## Features
- **Weight Estimation**: Estimates the body weight of pigs from images using advanced computer vision algorithms.
- **QR Code Detection**: Identifies and decodes unique QR codes for individual pigs to track their nutrient requirements.
- **Nutrient Index Calculation**: Calculates a nutrient index score that indicates the health and nutrient needs of each pig.
- **Automated Feeding Control**: Integrates with Calan gate feeders to provide precise feeding based on the calculated nutrient index.

## Project Structure
```
pig-nutrient-index-app
├── src
│   ├── main.py                  # Entry point for the application
│   ├── vision                    # Module for vision processing
│   │   ├── __init__.py
│   │   ├── weight_estimator.py   # Weight estimation logic
│   │   └── qr_detector.py        # QR code detection logic
│   ├── feeder_control            # Module for feeder control
│   │   ├── __init__.py
│   │   └── calan_gate.py         # Control logic for Calan gate feeders
│   ├── index_calculator          # Module for nutrient index calculation
│   │   ├── __init__.py
│   │   └── nutrient_index.py     # Nutrient index calculation logic
│   ├── utils                     # Utility functions
│   │   ├── __init__.py
│   │   └── image_processing.py    # Image processing utilities
│   └── config.py                 # Configuration settings
├── tests                         # Unit tests for the application
│   ├── test_weight_estimator.py
│   ├── test_qr_detector.py
│   ├── test_nutrient_index.py
│   └── test_calan_gate.py
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Files to ignore in version control
```

## Setup Instructions
1. **Clone the Repository**: 
   ```
   git clone <repository-url>
   cd pig-nutrient-index-app
   ```

2. **Install Dependencies**: 
   Use the following command to install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Run the Application**: 
   Execute the main application file:
   ```
   python src/main.py
   ```

## Usage Guidelines
- Ensure that the pigs are properly marked with unique QR codes for accurate identification.
- Provide clear images of the pigs for the weight estimation process.
- Monitor the nutrient index scores to adjust feeding as necessary.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Team
Heidi Rinehart
Brent Uyguangco
Eli Young
Dawit Woldemariam

## Overview
The Pig Nutrient Index Application is an end-to-end solution designed to automate and optimize pig feeding using computer vision and individualized identification. The core workflow begins by processing images of pigs, where the system detects and decodes unique QR codes imprinted on each pig’s back. This enables the application to reliably identify individual pigs, ensuring that each animal’s data and feeding regimen are tracked separately. Once a pig is identified, the application uses image analysis techniques to estimate the pig’s body weight. This is achieved through a combination of image processing and contour detection, allowing the system to approximate the size of the pig and convert this measurement into an estimated weight value.

After determining the pig’s weight, the application calculates a nutrient index score on a scale from 0 to 100. This score reflects the pig’s current health and nutritional needs, with 0 indicating a need for urgent nutritional intervention and 100 representing optimal health. The nutrient index is computed based on configurable healthy weight thresholds, making it adaptable to different breeds or growth stages. Using this index, the system then determines the precise amount of feed required for each pig and communicates with Calan gate feeders—automated devices that control access to food on a per-animal basis. This ensures that each pig receives the exact amount of nutrients it needs, reducing the risk of over- or under-feeding and supporting better animal health and farm efficiency.

The codebase is modular and well-structured, with separate components for QR code detection, weight estimation, nutrient index calculation, and feeder control. Each module is accompanied by unit tests to ensure reliability and maintainability. The main pipeline integrates all these components, providing a seamless workflow from image input to automated feeding output. The application is designed to be easily extensible, allowing for future enhancements such as improved segmentation algorithms, integration with farm management systems, or support for additional animal species.

**Disclaimer:** While the logic and integration of this application are based on proven computer vision and automation techniques, we have not been able to test the full pipeline with real images, as there are currently no publicly available datasets of pigs with QR codes imprinted on them. As such, the system has not been validated with real-world data. However, the algorithms and structure are robust, and we are confident that the program will work as intended once suitable data becomes available. This project demonstrates the feasibility and potential impact of individualized, automated animal nutrition management in modern agriculture.