#!/usr/bin/env python3
"""
Basic Classification Example
This script demonstrates a simple classification pipeline for the UCI HAR dataset.
Perfect for beginners to understand the basic workflow.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_basic_data(dataset_path):
    """Load and prepare the basic UCI HAR data"""
    print("📊 Loading UCI HAR data...")
    
    # Load training data
    X_train = pd.read_csv(f"{dataset_path}/train/X_train.txt", sep=r'\s+', header=None)
    y_train = pd.read_csv(f"{dataset_path}/train/y_train.txt", header=None, names=['activity'])
    
    # Load feature names
    features = pd.read_csv(f"{dataset_path}/features.txt", sep=' ', header=None, names=['id', 'feature'])
    X_train.columns = features['feature']
    
    print(f"✓ Loaded {len(X_train)} samples with {len(X_train.columns)} features")
    return X_train, y_train

def build_simple_classifier(X_train, y_train):
    """Build and evaluate a simple Random Forest classifier"""
    print("🌲 Building Random Forest classifier...")
    
    # Split data
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train, y_train['activity'], test_size=0.2, random_state=42
    )
    
    # Train classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_split, y_train_split)
    
    # Make predictions
    y_pred = rf.predict(X_test_split)
    
    # Evaluate
    accuracy = accuracy_score(y_test_split, y_pred)
    print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return rf, accuracy

def main():
    """Main function to run basic classification example"""
    # Update this path to your dataset location
    dataset_path = r"d:\Datasets\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset"
    
    # Load data
    X_train, y_train = load_basic_data(dataset_path)
    
    # Build classifier
    classifier, accuracy = build_simple_classifier(X_train, y_train)
    
    print("\n✅ Basic classification complete!")
    print(f"📈 This demonstrates how sensor data can achieve {accuracy*100:.1f}% accuracy")
    print("🚀 Ready to try more advanced techniques!")

if __name__ == "__main__":
    main()
