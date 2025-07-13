#!/usr/bin/env python3
"""
UCI HAR Dataset Exploration Script
This script helps you understand the UCI HAR dataset and how it relates to your BFRB detection project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class UCIHARExplorer:
    def __init__(self, data_dir):
        """
        Initialize the UCI HAR dataset explorer
        
        Args:
            data_dir (str): Path to the UCI HAR Dataset directory
        """
        self.data_dir = Path(data_dir)
        self.features = None
        self.activity_labels = None
        self.X_train = None
        self.y_train = None
        self.subject_train = None
        self.X_test = None
        self.y_test = None
        self.subject_test = None
        
    def load_data(self):
        """Load all the UCI HAR dataset files"""
        print("Loading UCI HAR dataset...")
        
        # Load feature names
        features_file = self.data_dir / "features.txt"
        self.features = pd.read_csv(features_file, sep=' ', header=None, names=['id', 'feature'])
        
        # Load activity labels
        activity_file = self.data_dir / "activity_labels.txt"
        self.activity_labels = pd.read_csv(activity_file, sep=' ', header=None, names=['id', 'activity'])
        
        # Load training data
        train_dir = self.data_dir / "train"
        self.X_train = pd.read_csv(train_dir / "X_train.txt", sep=r'\s+', header=None)
        self.y_train = pd.read_csv(train_dir / "y_train.txt", header=None, names=['activity'])
        self.subject_train = pd.read_csv(train_dir / "subject_train.txt", header=None, names=['subject'])
        
        # Load test data
        test_dir = self.data_dir / "test"
        self.X_test = pd.read_csv(test_dir / "X_test.txt", sep=r'\s+', header=None)
        self.y_test = pd.read_csv(test_dir / "y_test.txt", header=None, names=['activity'])
        self.subject_test = pd.read_csv(test_dir / "subject_test.txt", header=None, names=['subject'])
        
        # Set column names for feature matrices
        self.X_train.columns = self.features['feature']
        self.X_test.columns = self.features['feature']
        
        print(f"✓ Loaded {len(self.X_train)} training samples and {len(self.X_test)} test samples")
        print(f"✓ Dataset has {len(self.features)} features and {len(self.activity_labels)} activities")
        
    def explore_dataset_structure(self):
        """Print basic information about the dataset structure"""
        print("\n" + "="*50)
        print("DATASET STRUCTURE ANALYSIS")
        print("="*50)
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Features: {len(self.features)}")
        print(f"Activities: {len(self.activity_labels)}")
        print(f"Subjects: {len(set(self.subject_train['subject'].tolist() + self.subject_test['subject'].tolist()))}")
        
        print("\nActivity Distribution:")
        print("Training set:")
        train_dist = self.y_train['activity'].value_counts().sort_index()
        for activity_id, count in train_dist.items():
            activity_name = self.activity_labels[self.activity_labels['id'] == activity_id]['activity'].iloc[0]
            print(f"  {activity_id}: {activity_name} - {count} samples")
            
        print("\nTest set:")
        test_dist = self.y_test['activity'].value_counts().sort_index()
        for activity_id, count in test_dist.items():
            activity_name = self.activity_labels[self.activity_labels['id'] == activity_id]['activity'].iloc[0]
            print(f"  {activity_id}: {activity_name} - {count} samples")
    
    def analyze_features(self):
        """Analyze the feature types and their distributions"""
        print("\n" + "="*50)
        print("FEATURE ANALYSIS")
        print("="*50)
        
        # Categorize features
        time_features = self.features[self.features['feature'].str.startswith('t')]
        freq_features = self.features[self.features['feature'].str.startswith('f')]
        angle_features = self.features[self.features['feature'].str.startswith('angle')]
        
        print(f"Time domain features: {len(time_features)}")
        print(f"Frequency domain features: {len(freq_features)}")
        print(f"Angle features: {len(angle_features)}")
        
        # Analyze feature statistics
        print("\nFeature Statistics:")
        print(f"Mean feature value: {self.X_train.mean().mean():.4f}")
        print(f"Std feature value: {self.X_train.std().mean():.4f}")
        print(f"Min feature value: {self.X_train.min().min():.4f}")
        print(f"Max feature value: {self.X_train.max().max():.4f}")
        
        # Find most variable features
        feature_vars = self.X_train.var().sort_values(ascending=False)
        print(f"\nTop 10 most variable features:")
        for i, (feature, var) in enumerate(feature_vars.head(10).items(), 1):
            print(f"  {i}. {feature}: {var:.4f}")
    
    def analyze_sensor_types(self):
        """Analyze different sensor signal types"""
        print("\n" + "="*50)
        print("SENSOR SIGNAL ANALYSIS")
        print("="*50)
        
        # Categorize by sensor type
        body_acc_features = self.features[self.features['feature'].str.contains('BodyAcc')]
        gravity_acc_features = self.features[self.features['feature'].str.contains('GravityAcc')]
        gyro_features = self.features[self.features['feature'].str.contains('Gyro')]
        jerk_features = self.features[self.features['feature'].str.contains('Jerk')]
        
        print(f"Body acceleration features: {len(body_acc_features)}")
        print(f"Gravity acceleration features: {len(gravity_acc_features)}")
        print(f"Gyroscope features: {len(gyro_features)}")
        print(f"Jerk (derivative) features: {len(jerk_features)}")
        
        # Analyze statistical measures
        stat_measures = ['mean()', 'std()', 'mad()', 'max()', 'min()', 'sma()', 'energy()', 'iqr()', 'entropy()']
        print(f"\nStatistical measures applied to signals:")
        for measure in stat_measures:
            count = len(self.features[self.features['feature'].str.contains(measure.replace('()', r'\(\)'))])
            print(f"  {measure}: {count} features")
    
    def find_relevant_features_for_bfrb(self):
        """Identify features most relevant for BFRB-like gesture detection"""
        print("\n" + "="*50)
        print("BFRB-RELEVANT FEATURES ANALYSIS")
        print("="*50)
        
        # Features that might be relevant for hand/arm movements
        hand_relevant_keywords = [
            'BodyAcc',      # Body acceleration (hand movements)
            'BodyGyro',     # Body gyroscope (hand rotation)
            'Jerk',         # Sudden movements (scratching, picking)
            'mean()',       # Average movement
            'std()',        # Movement variability
            'max()',        # Peak movements
            'energy()',     # Movement intensity
            'entropy()',    # Movement complexity
            'correlation()', # Coordination between axes
        ]
        
        print("Features potentially relevant for BFRB detection:")
        for keyword in hand_relevant_keywords:
            escaped_keyword = keyword.replace('()', r'\(\)')
            relevant_features = self.features[self.features['feature'].str.contains(escaped_keyword)]
            print(f"\n{keyword} features ({len(relevant_features)} total):")
            for _, row in relevant_features.head(5).iterrows():
                print(f"  - {row['feature']}")
            if len(relevant_features) > 5:
                print(f"  ... and {len(relevant_features) - 5} more")
    
    def create_feature_importance_analysis(self):
        """Simple feature importance analysis using Random Forest"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            
            # Train a simple Random Forest model
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(self.X_train, self.y_train['activity'])
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': self.features['feature'],
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Top 20 most important features for activity classification:")
            for i, row in feature_importance.head(20).iterrows():
                print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
            
            # Cross-validation score
            cv_scores = cross_val_score(rf, self.X_train, self.y_train['activity'], cv=5)
            print(f"\nRandom Forest Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return feature_importance
            
        except ImportError:
            print("scikit-learn not available. Install with: pip install scikit-learn")
            return None
    
    def create_visualization(self):
        """Create visualizations of the dataset"""
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Activity distribution
            activity_counts = self.y_train['activity'].value_counts().sort_index()
            activity_names = [self.activity_labels[self.activity_labels['id'] == i]['activity'].iloc[0] 
                            for i in activity_counts.index]
            
            axes[0, 0].bar(activity_names, activity_counts.values)
            axes[0, 0].set_title('Activity Distribution in Training Set')
            axes[0, 0].set_xlabel('Activity')
            axes[0, 0].set_ylabel('Number of Samples')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Feature value distribution
            sample_features = self.X_train[['tBodyAcc-mean()-X', 'tBodyAcc-std()-X', 
                                           'fBodyAcc-mean()-X', 'fBodyAcc-energy()-X']].values.flatten()
            axes[0, 1].hist(sample_features, bins=50, alpha=0.7)
            axes[0, 1].set_title('Distribution of Sample Feature Values')
            axes[0, 1].set_xlabel('Feature Value')
            axes[0, 1].set_ylabel('Frequency')
            
            # 3. Correlation matrix of sample features
            sample_cols = ['tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z',
                          'tBodyGyro-mean()-X', 'tBodyGyro-mean()-Y', 'tBodyGyro-mean()-Z']
            corr_matrix = self.X_train[sample_cols].corr()
            
            im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            axes[1, 0].set_title('Feature Correlation Matrix (Sample)')
            axes[1, 0].set_xticks(range(len(sample_cols)))
            axes[1, 0].set_yticks(range(len(sample_cols)))
            axes[1, 0].set_xticklabels([col.replace('tBody', '').replace('-mean()', '') for col in sample_cols])
            axes[1, 0].set_yticklabels([col.replace('tBody', '').replace('-mean()', '') for col in sample_cols])
            plt.colorbar(im, ax=axes[1, 0])
            
            # 4. Subject distribution
            subject_counts = pd.concat([self.subject_train, self.subject_test])['subject'].value_counts().sort_index()
            axes[1, 1].bar(subject_counts.index, subject_counts.values)
            axes[1, 1].set_title('Samples per Subject')
            axes[1, 1].set_xlabel('Subject ID')
            axes[1, 1].set_ylabel('Number of Samples')
            
            plt.tight_layout()
            plt.savefig('uci_har_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✓ Visualizations created and saved as 'uci_har_analysis.png'")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    def generate_insights_for_bfrb(self):
        """Generate insights specifically for BFRB detection project"""
        print("\n" + "="*50)
        print("INSIGHTS FOR BFRB DETECTION PROJECT")
        print("="*50)
        
        print("Key Takeaways for your BFRB detection project:")
        print()
        print("1. FEATURE EXTRACTION STRATEGY:")
        print("   - Use sliding windows (2.56 seconds with 50% overlap)")
        print("   - Extract both time and frequency domain features")
        print("   - Focus on statistical measures: mean, std, max, energy, entropy")
        print("   - Include jerk features for sudden movements (scratching, picking)")
        print()
        print("2. SENSOR UTILIZATION:")
        print("   - Start with IMU-only features (since 50% of your test set is IMU-only)")
        print("   - Body acceleration and gyroscope are most informative")
        print("   - Consider magnitude features for orientation-independent detection")
        print()
        print("3. CLASSIFICATION APPROACH:")
        print("   - Build separate models for binary (BFRB vs non-BFRB) and multi-class")
        print("   - Use ensemble methods (Random Forest, XGBoost) for robust performance")
        print("   - Consider hierarchical classification: binary first, then gesture type")
        print()
        print("4. VALIDATION STRATEGY:")
        print("   - Use subject-independent validation (leave-one-subject-out)")
        print("   - Ensure both body positions and gesture types are well represented")
        print("   - Test IMU-only performance separately from full-sensor performance")
        print()
        print("5. ADDITIONAL SENSORS (Thermopiles + Time-of-Flight):")
        print("   - Extract similar statistical features from these sensors")
        print("   - Look for hand-to-face proximity patterns (time-of-flight)")
        print("   - Detect temperature changes from skin contact (thermopiles)")
        print("   - Use early/late fusion techniques to combine sensor modalities")


def main():
    """Main function to run the UCI HAR dataset exploration"""
    # Set the path to your UCI HAR dataset
    # Update this path to match your dataset location
    dataset_path = r"d:\Datasets\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset"
    
    # Create explorer instance
    explorer = UCIHARExplorer(dataset_path)
    
    try:
        # Load the dataset
        explorer.load_data()
        
        # Run all analyses
        explorer.explore_dataset_structure()
        explorer.analyze_features()
        explorer.analyze_sensor_types()
        explorer.find_relevant_features_for_bfrb()
        feature_importance = explorer.create_feature_importance_analysis()
        explorer.create_visualization()
        explorer.generate_insights_for_bfrb()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print("Check the generated 'uci_har_analysis.png' file for visualizations.")
        print("Use the insights above to guide your BFRB detection project.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please check that the dataset path is correct and all files are present.")


if __name__ == "__main__":
    main()
