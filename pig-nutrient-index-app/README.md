# Pig Nutrient Index Application

## Overview
The Pig Nutrient Index Application is an advanced, end-to-end solution for automating and optimizing pig feeding using computer vision and individualized identification. This system combines real-time weight estimation through machine learning with automated feeding controls to personalize nutrition for each pig, addressing the industry need for precision livestock management without manual labor.

## 🚀 Key Features
- **Advanced Weight Estimation**: Machine learning models using morphological features (area, body length, width, eccentricity) trained on the PIGRGB-Weight dataset
- **RFID Pig Identification**: Reliable individual identification using UHF RFID tags for both camera areas and feeder access control
- **Automated Feeding Control**: Integration with Calan gate feeders for precise, individualized feeding based on RFID authorization
- **Real-time Monitoring**: Live camera feed processing with YOLO pig detection and RFID tracking
- **Comprehensive Analytics**: Weight tracking, nutrient index calculation, and feeding optimization

## 🎯 Innovation Highlights

### Problem Solved
In the livestock industry, pigs are typically fed ad libitum or require manual weighing every few days, which is labor-intensive and stressful for animals. Our system provides:

- **Automated weight estimation** without physical contact
- **Individual pig tracking** with reliable RFID identification
- **Real-time feeding decisions** based on estimated nutritional needs
- **Access control at feeders** preventing overfeeding and ensuring fair distribution
- **Cost-effective solution** suitable for medium-scale farms

### Technical Approach
1. **Computer Vision**: Uses RGB images to extract morphological features
2. **Machine Learning**: Neural networks (MLP) and Random Forest models for weight prediction
3. **RFID Identification**: UHF RFID tags provide reliable identification in farm environments
4. **Dual Detection Zones**: Camera area for weight estimation + feeder area for access control
5. **Automated Control**: Calan gate feeders with RFID-controlled access

## 📊 Dataset: PIGRGB-Weight
This application uses the publicly available PIGRGB-Weight dataset:
- **9,579 RGB images** of pigs in free-moving states
- **Weight annotations** ranging from 33.1kg to 192kg
- **Multiple behavioral states**: standing, feeding, walking, drinking
- **Captured conditions**: Two heights (1.88m and 1.78m) with natural lighting

## 🛠️ Technology Stack
- **Computer Vision**: OpenCV, YOLO for pig detection
- **Machine Learning**: scikit-learn (MLPRegressor, RandomForestRegressor)
- **Image Processing**: Advanced morphological feature extraction
- **Hardware Integration**: Calan gate feeder control simulation
- **Data Processing**: pandas, numpy for dataset handling

## 📁 Project Structure
```
pig-nutrient-index-app/
├── src/
│   ├── main.py                     # Main application with menu system
│   ├── train_model.py              # Model training script
│   ├── vision/
│   │   ├── weight_estimator.py     # ML-based weight estimation
│   │   ├── ear_tag_detector.py     # Colored ear tag detection
│   │   └── yolo_detector.py        # YOLO pig detection
│   ├── feeder_control/
│   │   └── calan_gate.py          # Automated feeder control
│   ├── index_calculator/
│   │   └── nutrient_index.py      # Nutrition calculation
│   └── utils/
│       ├── dataset_loader.py       # PIGRGB-Weight dataset loader
│       └── image_processing.py     # Image utilities
├── data/                           # PIGRGB-Weight dataset
├── models/                         # Trained models
├── tests/                          # Unit tests
├── demo.py                         # Comprehensive demonstration
└── requirements.txt                # Dependencies
```

## 🏃‍♂️ Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd pig-nutrient-index-app

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the PIGRGB-Weight dataset and extract it to the `data/` directory:
```
data/
└── RGB_9579/
    ├── fold1/
    ├── fold2/
    ├── fold3/
    ├── fold4/
    └── fold5/
```

### 3. Quick Demo
```bash
# Check dataset setup
python demo.py --check

# Run full demonstration
python demo.py --full

# Or run individual components
python demo.py --train      # Model training demo
python demo.py --inference  # Inference demo
python demo.py --rfid       # RFID detection demo
```

### 4. Train a Model
```bash
# Train with default settings (MLP neural network)
python src/train_model.py --data_path data --model_type mlp

# Compare different models
python src/train_model.py --compare

# Quick training with limited samples
python src/train_model.py --max_samples 1000
```

### 5. Run the Application
```bash
# Interactive menu system
python src/main.py

# Process single image
python src/main.py path/to/pig/image.png

# Demo with dataset
python src/main.py --demo

# Live camera monitoring
python src/main.py --live
```

## 🔬 Machine Learning Features

### Weight Estimation Model
The system extracts 13+ morphological features from pig images:

1. **Geometric Features**:
   - Relative projection area (SR)
   - Body length and width
   - Aspect ratio
   - Eccentricity

2. **Contour Features**:
   - Contour length and area
   - Convexity and solidity
   - Bounding box dimensions

3. **Intensity Features**:
   - Mean and standard deviation of pixel intensities
   - Intensity range
   - Texture contrast (edge detection)

### Model Performance
- **Neural Network (MLP)**: Optimized with multiple hidden layers
- **Random Forest**: Ensemble method for comparison
- **Evaluation Metrics**: MAE (Mean Absolute Error), R², MAPE
- **Expected Performance**: <5kg MAE on test data (based on research literature)

## 🏷️ Pig Identification System

### RFID Tags (Primary Method)
- **UHF RFID Technology**: 860-960 MHz frequency for optimal range and reliability
- **Passive Tags**: No batteries required, powered by reader's RF field
- **Long Range**: 3-6 meter detection range suitable for both camera and feeder areas
- **Environmental Durability**: IP67 rated tags withstand farm conditions
- **Unique IDs**: Each pig has a permanent, tamper-resistant identifier

### Dual Detection Functionality
- **Camera Area Detection**: Identifies which pig is in the weight estimation zone
- **Feeder Access Control**: Authorizes individual pigs at feeding stations
- **Feed History Tracking**: Prevents overfeeding by monitoring recent consumption
- **Real-time Authorization**: Instant decisions on feeding permissions

### Integration with Existing Systems
- **Database Connectivity**: Links RFID to pig records (breed, birth date, health history)
- **Farm Management**: Compatible with existing livestock management software
- **Scalability**: Easy to add new pigs or expand to additional feeding stations
- **Backup Systems**: Visual identification as fallback when RFID fails

## 🎛️ Feeding Control System

### Nutrient Index Calculation
- **Scale**: 0-100 (0 = urgent intervention needed, 100 = optimal health)
- **Weight-based**: Configurable healthy weight thresholds
- **Real-time calculation**: Instant feedback for feeding decisions

### Calan Gate Integration
- **Feed amount calculation**: Linear scaling based on nutrient index
- **Individual control**: Separate gates for each pig
- **Portion control**: 1-5kg feed dispensing range
- **Status monitoring**: Gate open/closed tracking

## 📈 Performance Metrics

### System Capabilities
- **Processing Speed**: Real-time inference on standard hardware
- **Accuracy**: Comparable to manual weighing (±3-5kg typical error)
- **Scalability**: Handles multiple pigs simultaneously
- **Reliability**: Robust to lighting conditions and pig poses

### Hackathon Demonstration
This system demonstrates:
1. **Complete workflow** from image to feeding decision
2. **Real dataset integration** with 9,579+ images
3. **Machine learning pipeline** with training and evaluation
4. **Practical applicability** for real farm deployment
5. **Innovation potential** for precision livestock management

## 🔧 Configuration

### Model Parameters
```python
# In src/config.py
WEIGHT_ESTIMATOR_CONFIG = {
    'model_type': 'mlp',  # or 'rf'
    'hidden_layers': (100, 50, 25),
    'max_iter': 1000
}

NUTRIENT_INDEX_CONFIG = {
    'min_weight': 30,   # kg
    'max_weight': 120   # kg
}

FEEDING_CONFIG = {
    'min_feed': 1.0,    # kg
    'max_feed': 5.0     # kg
}
```

## 🧪 Testing
```bash
# Run unit tests
python -m pytest tests/

# Test individual components
python tests/test_weight_estimator.py
python tests/test_ear_tag_detector.py
python tests/test_nutrient_index.py
```

## 📚 Research Foundation
This implementation is based on recent research in precision livestock farming:

- **SAM2-Pig segmentation** for accurate pig region extraction
- **BPNN with Trainlm** optimization for weight prediction
- **Multi-feature fusion** approach for robust estimation
- **Real-time processing** capabilities for practical deployment

## 🚀 Future Enhancements
- **Advanced segmentation**: Integration with SAM2 for improved pig masks
- **RFID integration**: Support for RFID tags alongside visual detection
- **Cloud connectivity**: Remote monitoring and data analytics
- **Mobile deployment**: TensorFlow Lite for on-device inference
- **Multi-species support**: Adaptation for cattle, sheep, etc.

## 👥 Team
- **Heidi Rinehart**: Team lead, organization
- **Brent Uyguangco**: Computer programming, recognition model training
- **Eli Young**: GitHub, system integration
- **Dawit Woldemariam**: Agricultural research

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- **PIGRGB-Weight Dataset**: Ji, X., Li, Q., Guo, K., et al. (2025)
- **Research Community**: Precision livestock farming researchers
- **Open Source Libraries**: scikit-learn, OpenCV, YOLO contributors

---

**Ready for Production**: This system demonstrates a complete solution for automated pig weight estimation and feeding control, suitable for deployment in modern livestock operations seeking to improve efficiency and animal welfare through precision agriculture technologies.

## 📡 RFID Technology Integration

### **Dual-Zone Detection System**
- **Camera Zone**: RFID reader detects which pig is in the imaging area for weight estimation
- **Feeder Zone**: RFID reader at each Calan gate controls individual pig access to food

### **Hardware Specifications**
- **Reader**: ThingMagic M6e Nano (860-960 MHz UHF, 1-6m range, $200-400)
- **Tags**: Smartrac DogBone RFID (Passive UHF, IP67 rated, $2-5 per tag)
- **Antenna**: Circular polarized panel (6-9 dBi gain, 120° coverage)
- **Integration**: TCP/IP or Serial connectivity to farm management systems

### **Advantages over Visual Tags**
- **Environment-proof**: Works in dirt, mud, rain, and varying lighting
- **Long-range detection**: 3-6 feet detection range
- **No batteries required**: Passive RFID tags
- **Tamper-resistant**: Embedded in durable ear tags
- **Industry standard**: Widely adopted in livestock management