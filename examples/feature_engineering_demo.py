#!/usr/bin/env python3
"""
Feature Engineering Demo
This script shows how to manually create features from raw sensor data.
Helps understand the connection between time-series signals and ML features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_raw_sensor_data(dataset_path):
    """Load raw sensor signals for demonstration"""
    print("📡 Loading raw sensor data...")
    
    # Load body acceleration signals
    body_acc_x = pd.read_csv(f"{dataset_path}/train/Inertial Signals/body_acc_x_train.txt", 
                            sep=r'\s+', header=None)
    body_acc_y = pd.read_csv(f"{dataset_path}/train/Inertial Signals/body_acc_y_train.txt", 
                            sep=r'\s+', header=None)
    body_acc_z = pd.read_csv(f"{dataset_path}/train/Inertial Signals/body_acc_z_train.txt", 
                            sep=r'\s+', header=None)
    
    # Load activity labels
    y_train = pd.read_csv(f"{dataset_path}/train/y_train.txt", header=None, names=['activity'])
    
    print(f"✓ Loaded {len(body_acc_x)} signal windows")
    print(f"✓ Each window has {len(body_acc_x.columns)} time points")
    
    return body_acc_x, body_acc_y, body_acc_z, y_train

def extract_statistical_features(signal_window):
    """Extract basic statistical features from a signal window"""
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(signal_window)
    features['std'] = np.std(signal_window)
    features['max'] = np.max(signal_window)
    features['min'] = np.min(signal_window)
    features['median'] = np.median(signal_window)
    
    # Advanced statistics
    features['skewness'] = stats.skew(signal_window)
    features['kurtosis'] = stats.kurtosis(signal_window)
    features['energy'] = np.sum(signal_window ** 2) / len(signal_window)
    features['rms'] = np.sqrt(np.mean(signal_window ** 2))
    
    # Signal variability
    features['range'] = features['max'] - features['min']
    features['iqr'] = np.percentile(signal_window, 75) - np.percentile(signal_window, 25)
    
    return features

def extract_temporal_features(signal_window):
    """Extract time-domain features that capture temporal patterns"""
    features = {}
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(signal_window)) != 0)
    features['zero_crossing_rate'] = zero_crossings / len(signal_window)
    
    # Mean crossing rate (crossings of the mean)
    mean_val = np.mean(signal_window)
    mean_crossings = np.sum(np.diff(np.sign(signal_window - mean_val)) != 0)
    features['mean_crossing_rate'] = mean_crossings / len(signal_window)
    
    # Peak count
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(signal_window)
    features['peak_count'] = len(peaks)
    
    # Slope of trend line
    x = np.arange(len(signal_window))
    slope, _, _, _, _ = stats.linregress(x, signal_window)
    features['trend_slope'] = slope
    
    return features

def demonstrate_feature_engineering(dataset_path):
    """Demonstrate the complete feature engineering process"""
    print("🔧 Demonstrating Feature Engineering Process")
    print("=" * 50)
    
    # Load raw data
    acc_x, acc_y, acc_z, y_train = load_raw_sensor_data(dataset_path)
    
    # Select a few samples for demonstration
    sample_indices = [0, 100, 200]  # Different activities
    activities = y_train.iloc[sample_indices]['activity'].values
    
    print(f"\n📊 Extracting features from sample windows:")
    
    for i, (idx, activity) in enumerate(zip(sample_indices, activities)):
        print(f"\n--- Sample {i+1}: Activity {activity} ---")
        
        # Get raw signal for this window
        signal_x = acc_x.iloc[idx].values
        print(f"Raw signal shape: {signal_x.shape}")
        print(f"Raw signal preview: {signal_x[:5]}...")
        
        # Extract features
        stats_features = extract_statistical_features(signal_x)
        temporal_features = extract_temporal_features(signal_x)
        
        print("\n📈 Statistical Features:")
        for name, value in stats_features.items():
            print(f"  {name}: {value:.4f}")
            
        print("\n⏱️ Temporal Features:")
        for name, value in temporal_features.items():
            print(f"  {name}: {value:.4f}")

def visualize_feature_extraction():
    """Create visualization showing feature extraction process"""
    print("\n📊 Creating Feature Extraction Visualization...")
    
    # Create sample signal
    t = np.linspace(0, 2.56, 128)  # 2.56 seconds, 128 points
    signal = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(128)
    
    # Extract features
    features = extract_statistical_features(signal)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot raw signal
    ax1.plot(t, signal, 'b-', alpha=0.7, label='Raw Signal')
    ax1.axhline(features['mean'], color='r', linestyle='--', label=f"Mean: {features['mean']:.3f}")
    ax1.axhline(features['max'], color='g', linestyle='--', label=f"Max: {features['max']:.3f}")
    ax1.axhline(features['min'], color='orange', linestyle='--', label=f"Min: {features['min']:.3f}")
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Acceleration')
    ax1.set_title('Raw Sensor Signal with Key Features')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot feature summary
    feature_names = list(features.keys())[:8]  # Show first 8 features
    feature_values = [features[name] for name in feature_names]
    
    ax2.bar(feature_names, feature_values)
    ax2.set_title('Extracted Features from Signal')
    ax2.set_ylabel('Feature Value')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('feature_extraction_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved as 'feature_extraction_demo.png'")

def main():
    """Main function to run feature engineering demonstration"""
    # Update this path to your dataset location
    dataset_path = r"d:\Datasets\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset"
    
    try:
        # Demonstrate feature engineering
        demonstrate_feature_engineering(dataset_path)
        
        # Create visualization
        visualize_feature_extraction()
        
        print("\n" + "=" * 50)
        print("🎓 Feature Engineering Demo Complete!")
        print("=" * 50)
        print("Key Takeaways:")
        print("- Raw sensor signals are time-series with 128 points")
        print("- Features summarize important signal characteristics")
        print("- Different features capture different aspects of movement")
        print("- This process transforms time-series → fixed-size feature vectors")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check that the dataset path is correct.")

if __name__ == "__main__":
    main()
