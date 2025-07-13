# Complete Guide: UCI HAR Dataset for BFRB Detection Project

## Executive Summary

You now have a comprehensive understanding of the UCI HAR dataset! This dataset is an excellent learning resource for your BFRB detection project because it demonstrates the complete pipeline for motion-based activity recognition using smartphone sensors.

## Key Insights from the Analysis

### Dataset Structure
- **10,299 total samples** (7,352 training, 2,947 test)
- **561 features** extracted from IMU sensors (accelerometer + gyroscope)
- **6 activities** (walking, walking upstairs/downstairs, sitting, standing, laying)
- **30 subjects** with subject-independent train/test split

### Most Important Features for Motion Classification
1. **Gravity acceleration features** (tGravityAcc-mean()-X, tGravityAcc-max()-X)
2. **Angle features** (angle(X,gravityMean), angle(Y,gravityMean))
3. **Body acceleration jerk** (tBodyAccJerk-std()-X)
4. **Frequency domain features** (fBodyAcc-mean()-X, fBodyAccJerk-bandsEnergy())
5. **Statistical measures** (mean, std, max, energy, entropy)

### Feature Categories Most Relevant for BFRB:
- **Body acceleration** (292 features) - hand movements
- **Jerk features** (213 features) - sudden movements like scratching
- **Gyroscope features** (213 features) - hand/wrist rotations
- **Entropy features** (33 features) - movement complexity
- **Energy features** (33 features) - movement intensity

## How to Apply This to Your BFRB Project

### 1. Feature Extraction Strategy
```python
# Based on UCI HAR approach, extract these feature types for BFRB:
feature_types = [
    'mean()',     # Average movement patterns
    'std()',      # Movement variability (scratching vs smooth movements)
    'max()',      # Peak movements (sudden gestures)
    'energy()',   # Movement intensity
    'entropy()',  # Movement complexity/randomness
    'jerk',       # Sudden movements (key for BFRB detection)
    'correlation()' # Coordination between hand axes
]
```

### 2. Sensor Utilization Plan
**Phase 1: IMU-Only Model** (for 50% of test set)
- Focus on body acceleration and gyroscope features
- Extract 561 features similar to UCI HAR
- Achieve baseline performance with IMU data only

**Phase 2: Full-Sensor Model** (for remaining 50% of test set)
- Add thermopile features (temperature changes from skin contact)
- Add time-of-flight features (hand-to-face proximity)
- Use sensor fusion techniques

### 3. Classification Architecture
```python
# Hierarchical classification approach:
# Step 1: Binary classification (BFRB vs non-BFRB)
binary_model = RandomForestClassifier()

# Step 2: Multi-class classification (specific BFRB gestures)
multiclass_model = RandomForestClassifier()

# Final score = average of binary F1 and macro F1
```

### 4. Window Strategy
- **Window size**: 2.56 seconds (similar to UCI HAR)
- **Overlap**: 50% (provides more training samples)
- **Sampling rate**: Match your Helios device sampling rate

### 5. Validation Strategy
- **Subject-independent**: Leave-one-subject-out validation
- **Body position aware**: Ensure all positions are represented
- **Separate IMU testing**: Test IMU-only performance independently

## Immediate Next Steps

### 1. Implement UCI HAR Baseline
```bash
# Use the exploration script to understand feature importance
python explore_uci_har.py

# Build a simple classifier to verify understanding
# Start with Random Forest (achieved 91.8% accuracy)
```

### 2. Adapt Feature Extraction
- Use the 561 UCI HAR features as your starting point
- Modify for your specific BFRB gestures
- Add domain-specific features (e.g., hand-to-face proximity)

### 3. Build BFRB-Specific Models
- Start with binary classification (BFRB vs non-BFRB)
- Use similar feature extraction but tune for hand movements
- Incorporate additional sensors gradually

### 4. Handle the Dual Test Set Challenge
Your unique challenge is that 50% of test data has IMU-only:
```python
# Train two models:
imu_only_model = train_model(imu_features)
full_sensor_model = train_model(all_features)

# Ensemble or switch based on available sensors
```

## Technical Implementation Roadmap

### Week 1-2: Learning Phase
- [ ] Run UCI HAR exploration script
- [ ] Implement basic Random Forest classifier
- [ ] Understand feature importance patterns
- [ ] Study time-series windowing techniques

### Week 3-4: Adaptation Phase
- [ ] Adapt UCI HAR features for BFRB gestures
- [ ] Implement sliding window feature extraction
- [ ] Build binary classifier (BFRB vs non-BFRB)
- [ ] Test on IMU-only data

### Week 5-6: Enhancement Phase
- [ ] Add thermopile and time-of-flight features
- [ ] Implement sensor fusion techniques
- [ ] Build multi-class gesture classifier
- [ ] Optimize for the dual evaluation metric

### Week 7-8: Optimization Phase
- [ ] Hyperparameter tuning
- [ ] Cross-validation optimization
- [ ] Handle the split test set challenge
- [ ] Final model ensemble

## Code Templates Based on UCI HAR

### Feature Extraction Template
```python
def extract_features(window_data, sensors=['imu']):
    """Extract UCI HAR-style features from sensor window"""
    features = {}
    
    for sensor in sensors:
        if sensor == 'imu':
            # Extract accelerometer and gyroscope features
            features.update(extract_imu_features(window_data))
        elif sensor == 'thermopile':
            # Extract temperature-based features
            features.update(extract_thermal_features(window_data))
        elif sensor == 'tof':
            # Extract proximity-based features
            features.update(extract_proximity_features(window_data))
    
    return features
```

### Classification Template
```python
def build_bfrb_classifier():
    """Build BFRB classifier based on UCI HAR insights"""
    # Use Random Forest (performed well on UCI HAR)
    binary_clf = RandomForestClassifier(n_estimators=100)
    multiclass_clf = RandomForestClassifier(n_estimators=100)
    
    return binary_clf, multiclass_clf
```

## Success Metrics Based on UCI HAR

- **UCI HAR achieved 91.8% accuracy** with Random Forest
- **Your target**: Beat this performance on binary classification
- **Key insight**: Gravity and angle features were most important for UCI HAR
- **Your focus**: Body acceleration and jerk features for BFRB detection

## Final Recommendations

1. **Start simple**: Begin with the UCI HAR feature set
2. **Validate thoroughly**: Use subject-independent validation
3. **Handle sensor availability**: Build robust IMU-only models
4. **Leverage insights**: Focus on jerk, entropy, and energy features
5. **Iterate quickly**: Use the exploration script to guide feature selection

The UCI HAR dataset has given you a solid foundation for understanding motion-based classification. Now you can confidently adapt these techniques to your BFRB detection challenge!

---

**Files Created:**
- `UCI_HAR_Dataset_Analysis.md` - Comprehensive dataset analysis
- `explore_uci_har.py` - Complete exploration script
- `uci_har_analysis.png` - Visualizations (generated when you run the script)

Run the exploration script anytime to refresh your understanding or test new ideas!
