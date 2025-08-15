# Precise Swine & Dine: Automated Pig Weight Estimation and Feeding Control

Computer Vision & ML system for precision livestock management

## Overview
This repository contains an end-to-end system that estimates pig weights from RGB images and controls individualized feeding via a Calan gate integration. It combines YOLO-based detection, morphological feature extraction, and machine learning (MLP neural network and Random Forest) to deliver real-time, contactless weight estimation and feeding decisions.

## Highlights
- **<5 kg MAE**: Morphology-based weight estimation on the PIGRGB-Weight dataset
- **9,579 labeled RGB images**: Trained and evaluated on a large, diverse dataset
- **CV + ML stack**: YOLO detection + scikit-learn regressors (MLP, Random Forest)
- **Feeding control**: Nutrient index computation and Calan gate control logic
- **Team win**: Livestock Monitoring Track, AI Foundry for Ag Applications Hackathon

## Repository Structure
```
AutomaticWeightEstimationAndFeeding/
├── README.md                         # You are here
├── LICENSE
└── pig-nutrient-index-app/           # Main application
    ├── src/
    │   ├── main.py                   # CLI + menu
    │   ├── train_model.py            # Model training
    │   ├── vision/
    │   │   ├── yolo_detector.py
    │   │   ├── weight_estimator.py
    │   │   ├── ear_tag_detector.py
    │   │   └── rfid_detector.py
    │   ├── feeder_control/
    │   │   └── calan_gate.py         # Feeder access simulation/control
    │   ├── index_calculator/
    │   │   └── nutrient_index.py
    │   └── utils/
    │       ├── dataset_loader.py
    │       └── image_processing.py
    ├── tests/
    ├── demo.py
    ├── requirements.txt
    └── DATASET_SETUP.md
```

## Quick Start

### 1) Setup
```bash
git clone <repository-url>
cd AutomaticWeightEstimationAndFeeding/pig-nutrient-index-app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Dataset
Download and extract the PIGRGB-Weight dataset into `pig-nutrient-index-app/data/`.

See detailed instructions: [pig-nutrient-index-app/DATASET_SETUP.md](pig-nutrient-index-app/DATASET_SETUP.md)

Expected structure:
```
pig-nutrient-index-app/
└── data/
    └── RGB_9579/
        ├── fold1/
        ├── fold2/
        ├── fold3/
        ├── fold4/
        └── fold5/
```

### 3) Demo and Checks
```bash
cd pig-nutrient-index-app
python demo.py --check        # Verify dataset and environment
python demo.py --full         # End-to-end demonstration

# Or individual components
python demo.py --train        # Model training demo
python demo.py --inference    # Inference demo
python demo.py --rfid         # RFID demo (simulated)
```

### 4) Train a Model
```bash
python src/train_model.py --data_path data --model_type mlp
# Compare models
python src/train_model.py --compare
# Quick run on subset
python src/train_model.py --max_samples 1000
```

### 5) Run the Application
```bash
python src/main.py                 # Interactive menu
python src/main.py --demo          # Dataset-driven demo
python src/main.py --live          # Live camera mode (if supported)
python src/main.py path/to/img.png # Single image inference
```

## Core Features
- **Advanced weight estimation**: YOLO detection, segmentation-free morphology features (area, length, width, eccentricity) feeding into ML regressors
- **RFID-based identification**: Robust individual pig tracking for camera and feeder zones
- **Automated feeding control**: Nutrient index computation powering Calan gate access and feed dispensing logic
- **Analytics**: Tracking of weight, index, and feeding history

## Dataset: PIGRGB-Weight
- 9,579 RGB images; weights from ~33 kg to ~192 kg
- Multiple behaviors and capture conditions (two heights, natural lighting)

## Technology Stack
- Computer Vision: OpenCV, YOLO
- Machine Learning: scikit-learn (MLPRegressor, RandomForestRegressor)
- Data: pandas, numpy
- Feeder Control: Calan gate access simulation

## Model Details
Extracted features include geometric, contour, and intensity descriptors. Models are evaluated with MAE, R², and MAPE. On held-out data, the system targets **<5 kg MAE**.

Configuration (see `pig-nutrient-index-app/src/config.py`):
```python
WEIGHT_ESTIMATOR_CONFIG = {
    "model_type": "mlp",  # or "rf"
    "hidden_layers": (100, 50, 25),
    "max_iter": 1000,
}

NUTRIENT_INDEX_CONFIG = {
    "min_weight": 30,   # kg
    "max_weight": 120,  # kg
}

FEEDING_CONFIG = {
    "min_feed": 1.0,    # kg
    "max_feed": 5.0,    # kg
}
```

## Testing
```bash
cd pig-nutrient-index-app
python -m pytest tests/
```

## RFID Integration (Concept + Simulation)
- Dual-zone detection: camera zone for weight estimation, feeder zone for access control
- Typical hardware: UHF reader (ThingMagic M6e Nano), passive UHF ear tags, circular polarized antenna
- Advantages vs visual tags: long-range, passive, rugged, tamper-resistant

## Team
- Heidi Rinehart — Team lead, organization
- Brent Uyguangco — Programming, recognition model training
- Eli Young — GitHub, system integration
- Dawit Woldemariam — Agricultural research

## License
MIT License. See `LICENSE`.

## Acknowledgments
- PIGRGB-Weight Dataset: Ji, X., Li, Q., Guo, K., et al. (2025)
- Open source libraries: scikit-learn, OpenCV, YOLO