# 🚀 UCI HAR Dataset Learning Journey

A comprehensive beginner-friendly guide to understanding sensor data analysis using the UCI Human Activity Recognition (HAR) dataset.

## 📋 Overview

This repository helps newcomers to sensor data analysis learn the fundamentals using a well-known dataset. Perfect for those transitioning from traditional machine learning datasets to time-series sensor data.

### 🎯 Who This Is For:
- **Beginners** in sensor data analysis
- **ML practitioners** familiar with traditional datasets (CSV with rows/columns)
- **Students** learning about time-series classification
- **Researchers** starting projects with wearable sensor data

### 🎓 What You'll Learn:
- How sensor data differs from traditional datasets
- Feature engineering from raw time-series signals
- Building classification pipelines for sensor data
- Understanding preprocessing techniques for noisy sensor data
- Best practices for evaluation and validation

## 📊 Dataset Information

The UCI HAR dataset contains:
- **30 subjects** performing 6 activities
- **Accelerometer & Gyroscope** data from smartphones
- **561 engineered features** from raw sensor signals
- **10,299 total samples** (7,352 training, 2,947 test)

### Activities:
1. Walking
2. Walking Upstairs  
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn pathlib
```

### Dataset Download
1. Download the UCI HAR Dataset from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
2. Extract to your preferred directory
3. Update the `dataset_path` in the scripts

### Quick Start
```bash
# Clone this repository
git clone https://github.com/yourusername/uci-har-learning-journey.git
cd uci-har-learning-journey

# Run the exploration script
python explore_uci_har.py

# Or open the interactive notebook
jupyter notebook Sensor_Data_Learning_Journey.ipynb
```

## 📁 Repository Structure

```
├── README.md                           # This file
├── explore_uci_har.py                  # Complete analysis script
├── Sensor_Data_Learning_Journey.ipynb  # Interactive learning notebook
├── UCI_HAR_Dataset_Analysis.md         # Detailed dataset analysis
├── BFRB_Project_Guide.md              # Application to BFRB detection
├── requirements.txt                    # Python dependencies
├── examples/                           # Example notebooks and scripts
│   ├── basic_classification.py
│   ├── feature_engineering_demo.py
│   └── visualization_examples.py
└── docs/                              # Additional documentation
    ├── sensor_data_basics.md
    ├── feature_engineering_guide.md
    └── troubleshooting.md
```

## 🎯 Learning Path

### 📚 **Step 1: Understanding (30 minutes)**
Start with the interactive notebook `Sensor_Data_Learning_Journey.ipynb`:
- Dataset structure and organization
- Raw sensor signals vs. engineered features
- Visualization of different activities

### 🔧 **Step 2: Hands-On Analysis (1 hour)**
Run the exploration script:
```python
python explore_uci_har.py
```
This provides comprehensive analysis with:
- Feature importance ranking
- Classification performance
- Visualization of key patterns

### 🧠 **Step 3: Deep Dive (2-3 hours)**
Explore the detailed guides:
- `UCI_HAR_Dataset_Analysis.md` - Complete dataset breakdown
- `BFRB_Project_Guide.md` - Real-world application example

## 📊 Key Results

Our analysis achieves:
- **91.8% classification accuracy** with Random Forest
- **Top features identified**: Gravity and angle measurements
- **561 features** successfully extracted from raw sensor streams

### Most Important Features:
1. `tGravityAcc-mean()-X` (gravity acceleration)
2. `tGravityAcc-max()-X` (gravity peaks)
3. `angle(X,gravityMean)` (orientation angles)
4. `tBodyAccJerk-std()-X` (movement variability)

## 🎯 Real-World Applications

This learning journey prepares you for:
- **Healthcare**: Activity monitoring, fall detection
- **Sports**: Performance analysis, movement tracking
- **Mental Health**: Behavior pattern recognition (like BFRB detection)
- **IoT**: Smart home activity recognition

## 🤝 Contributing

We welcome contributions! Areas where you can help:

### 📝 **Documentation**
- Improve explanations for beginners
- Add more real-world examples
- Translate to other languages

### 💻 **Code Examples**
- Additional visualization techniques
- Different classification algorithms
- Advanced feature engineering methods

### 🐛 **Issues & Improvements**
- Report bugs or unclear explanations
- Suggest new learning exercises
- Share your own sensor data projects

### How to Contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the HAR dataset
- **Original researchers**: Jorge L. Reyes-Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra
- **Community contributors** who help improve this learning resource

## 📞 Contact & Support

- **Issues**: Open a GitHub issue for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: [your-email@example.com] for direct contact

## 🔗 Related Resources

### 📚 **Learning Materials**
- [Andrew Ng's ML Course](https://www.coursera.org/learn/machine-learning) - Machine Learning Fundamentals
- [Time Series Analysis with Python](https://github.com/PacktPublishing/Hands-On-Time-Series-Analysis-with-Python) - Advanced techniques
- [Kaggle Learn](https://www.kaggle.com/learn) - Practical ML skills

### 🎯 **Similar Projects**
- [Human Activity Recognition Papers](https://paperswithcode.com/task/human-activity-recognition)
- [Wearable Computing Datasets](https://www.cis.fordham.edu/wisdm/dataset.php)
- [Sensor Data Analysis Examples](https://github.com/topics/sensor-data-analysis)

---

**⭐ If this helped you learn sensor data analysis, please star the repository!**

**🔄 Share with others who might benefit from this learning journey!**
