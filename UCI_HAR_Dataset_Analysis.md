# UCI HAR Dataset Analysis for BFRB Detection Project

## Dataset Overview

The UCI Human Activity Recognition (HAR) dataset is a benchmark dataset for motion-based activity recognition using smartphone sensors. Here's how it relates to your BFRB detection project:

### Key Similarities to Your BFRB Project:
1. **Motion-based classification**: Both datasets use IMU (Inertial Measurement Unit) sensors to detect different types of movements
2. **Feature extraction**: Both require extracting meaningful features from raw sensor data
3. **Multi-class classification**: Both classify different types of activities/gestures
4. **Time-series analysis**: Both deal with temporal sensor data

## Dataset Structure

### Files and Their Purpose:

#### Core Files:
- `README.txt` - Dataset documentation
- `features_info.txt` - Detailed explanation of the 561 features
- `features.txt` - List of all 561 feature names (what you're currently viewing)
- `activity_labels.txt` - Maps activity IDs to names (1-6 activities)

#### Training Data:
- `train/X_train.txt` - Training feature vectors (7,352 samples × 561 features)
- `train/y_train.txt` - Training labels (7,352 activity IDs)
- `train/subject_train.txt` - Subject IDs for training (7,352 subject IDs)

#### Test Data:
- `test/X_test.txt` - Test feature vectors (2,947 samples × 561 features)
- `test/y_test.txt` - Test labels (2,947 activity IDs)
- `test/subject_test.txt` - Subject IDs for test (2,947 subject IDs)

#### Raw Sensor Data (Optional):
- `train/Inertial Signals/` - Raw accelerometer and gyroscope data
- `test/Inertial Signals/` - Raw accelerometer and gyroscope data

## Activities Being Classified

The dataset classifies 6 basic human activities:
1. **WALKING** - Regular walking
2. **WALKING_UPSTAIRS** - Walking upstairs
3. **WALKING_DOWNSTAIRS** - Walking downstairs
4. **SITTING** - Sitting position
5. **STANDING** - Standing position
6. **LAYING** - Lying down

## Sensor Data and Features

### Sensors Used:
- **Accelerometer**: Measures linear acceleration (3-axis: X, Y, Z)
- **Gyroscope**: Measures angular velocity (3-axis: X, Y, Z)
- **Sampling Rate**: 50Hz (50 measurements per second)
- **Window Size**: 2.56 seconds with 50% overlap (128 readings per window)

### Feature Categories (561 total):

#### Time Domain Features (prefix 't'):
- `tBodyAcc-XYZ`: Body acceleration
- `tGravityAcc-XYZ`: Gravity acceleration
- `tBodyAccJerk-XYZ`: Body linear acceleration jerk
- `tBodyGyro-XYZ`: Body angular velocity
- `tBodyGyroJerk-XYZ`: Body angular velocity jerk
- `tBodyAccMag`: Body acceleration magnitude
- `tGravityAccMag`: Gravity acceleration magnitude
- `tBodyAccJerkMag`: Body acceleration jerk magnitude
- `tBodyGyroMag`: Body angular velocity magnitude
- `tBodyGyroJerkMag`: Body angular velocity jerk magnitude

#### Frequency Domain Features (prefix 'f'):
- Same signals as time domain but transformed using Fast Fourier Transform (FFT)

#### Statistical Measures Applied to Each Signal:
- `mean()`: Mean value
- `std()`: Standard deviation
- `mad()`: Median absolute deviation
- `max()`: Maximum value
- `min()`: Minimum value
- `sma()`: Signal magnitude area
- `energy()`: Energy measure
- `iqr()`: Interquartile range
- `entropy()`: Signal entropy
- `arCoeff()`: Autoregression coefficients
- `correlation()`: Correlation coefficient
- `maxInds()`: Index of frequency component with largest magnitude
- `meanFreq()`: Weighted average of frequency components
- `skewness()`: Skewness of frequency domain signal
- `kurtosis()`: Kurtosis of frequency domain signal
- `bandsEnergy()`: Energy of frequency intervals
- `angle()`: Angle between vectors

## How This Relates to Your BFRB Project

### Direct Applications:
1. **Feature Engineering**: The UCI HAR dataset shows how to extract meaningful features from raw IMU data
2. **Classification Pipeline**: Demonstrates end-to-end motion classification workflow
3. **Data Preprocessing**: Shows how to handle sliding windows and overlap
4. **Evaluation Framework**: Provides structure for training/testing splits

### Key Differences to Consider:
1. **Your project has more sensors**: UCI HAR uses only IMU, while your Helios device has thermopiles and time-of-flight sensors
2. **Different gestures**: UCI HAR focuses on basic activities, while you're detecting specific hand movements
3. **Different body positions**: Your project includes various sitting/lying positions
4. **Binary + multi-class**: Your evaluation combines binary classification (BFRB vs non-BFRB) with multi-class gesture recognition

### Learning Opportunities:
1. **Start with IMU-only features**: Since half your test set will be IMU-only, understanding these 561 features is crucial
2. **Feature selection**: Learn which features are most important for motion classification
3. **Windowing strategies**: Understand how to segment time-series data
4. **Cross-validation**: Learn proper evaluation techniques for motion data

## Getting Started with the Data

### Step 1: Load and Explore
```python
import pandas as pd
import numpy as np

# Load feature names
features = pd.read_csv('features.txt', sep=' ', header=None, names=['id', 'feature'])

# Load training data
X_train = pd.read_csv('train/X_train.txt', sep='\s+', header=None)
y_train = pd.read_csv('train/y_train.txt', header=None, names=['activity'])
subject_train = pd.read_csv('train/subject_train.txt', header=None, names=['subject'])

# Load activity labels
activity_labels = pd.read_csv('activity_labels.txt', sep=' ', header=None, names=['id', 'activity'])
```

### Step 2: Analyze Feature Importance
- Study which features are most discriminative
- Understand the difference between time and frequency domain features
- Analyze correlation between features

### Step 3: Build Baseline Model
- Start with simple classifiers (Random Forest, SVM)
- Evaluate performance on the 6-class problem
- Understanding feature importance

### Step 4: Apply Learnings to BFRB Project
- Adapt feature extraction techniques
- Modify classification approach for binary + multi-class evaluation
- Incorporate additional sensor modalities (thermopiles, time-of-flight)

## Recommended Next Steps

1. **Explore the dataset**: Load and visualize the data to understand patterns
2. **Reproduce baseline results**: Implement a simple classifier to verify your understanding
3. **Feature analysis**: Identify which features are most important for motion classification
4. **Adapt techniques**: Apply learned feature extraction and classification methods to your BFRB dataset
5. **Sensor fusion**: Once you master IMU-based classification, work on incorporating additional sensors

This dataset provides an excellent foundation for understanding motion-based classification, which is directly applicable to your BFRB detection project!
