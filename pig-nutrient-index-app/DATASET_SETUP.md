# Dataset Setup Instructions

## PIGRGB-Weight Dataset

This project uses the PIGRGB-Weight dataset for training the pig weight estimation model.

### Dataset Information
- **Total Images**: 9,579 RGB images
- **Weight Range**: 73.36kg - 192.48kg  
- **Mean Weight**: 121.70 ± 32.07kg
- **Image Conditions**: Multiple heights (1.88m, 1.78m), natural lighting
- **Pig Behaviors**: Standing, feeding, walking, drinking

### Download Instructions

1. **Download the dataset** from Google Drive:
   - Link: [Google Drive download](https://drive.google.com/file/d/1AvfUDBlVD6ZAHXYdHquwHBt3bUSjZCvy/view?usp=sharing)

2. **Extract the dataset** to the correct location:
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

3. **Verify the setup**:
   ```bash
   cd pig-nutrient-index-app
   python demo.py --check
   ```

### Expected Directory Structure
After extraction, your data directory should look like:
```
data/
└── RGB_9579/
    ├── fold1/
    │   ├── pig_001_75.2kg.png
    │   ├── pig_002_82.1kg.png
    │   └── ...
    ├── fold2/
    │   └── ...
    ├── fold3/
    │   └── ...
    ├── fold4/
    │   └── ...
    └── fold5/
        └── ...
```

### File Naming Convention
Images are named with the format: `pig_XXX_YYYkg.png`
- `XXX`: Pig identifier
- `YYY`: Weight in kilograms (ground truth)

### Dataset Citation
If you use this dataset, please cite:
```
Ji, X., Li, Q., Guo, K., et al. (2025). 
PIGRGB-Weight: A comprehensive dataset for pig weight estimation using RGB images.
```

### Troubleshooting
- **File not found errors**: Ensure the RGB_9579 folder is directly inside the `data/` directory
- **Permission errors**: Make sure you have read access to all extracted files
- **Large download**: The dataset is ~2GB, ensure you have sufficient disk space 