# 📥 How to Download and Setup the UCI HAR Dataset

## 🚨 Important Note
The actual UCI HAR dataset is **NOT included** in this repository because:
- Large file sizes (66MB+) exceed GitHub limits
- Dataset is copyrighted by UCI
- Users should download from the official source
- Keeps the repository lightweight and focused on code

## 📥 Step-by-Step Download Instructions

### Option 1: Direct Download (Recommended)
1. **Visit the official UCI repository**: 
   - Go to: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
   
2. **Download the dataset**:
   - Click on "Data Folder" link
   - Download `UCI HAR Dataset.zip`
   - File size: ~60MB

3. **Extract the dataset**:
   ```bash
   # Extract to your desired location
   unzip "UCI HAR Dataset.zip"
   ```

4. **Update the path in code**:
   ```python
   # In all Python scripts, update this line:
   dataset_path = r"YOUR_PATH_HERE/UCI HAR Dataset/UCI HAR Dataset"
   ```

### Option 2: Command Line Download
```bash
# Download using curl (Unix/Mac/WSL)
curl -o "UCI_HAR_Dataset.zip" "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"

# Or using wget
wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"

# Extract
unzip UCI_HAR_Dataset.zip
```

### Option 3: Python Download (Automated)
```python
import urllib.request
import zipfile
import os

def download_uci_har_dataset(extract_path="./data"):
    """Download and extract UCI HAR dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    zip_path = "UCI_HAR_Dataset.zip"
    
    print("📥 Downloading UCI HAR Dataset...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("📂 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    print("✅ Dataset ready!")
    os.remove(zip_path)  # Clean up zip file
    
    return os.path.join(extract_path, "UCI HAR Dataset")

# Usage
dataset_path = download_uci_har_dataset()
```

## 📁 Expected Directory Structure

After downloading and extracting, you should have:
```
📁 UCI HAR Dataset/
├── 📄 README.txt
├── 📄 features_info.txt
├── 📄 features.txt
├── 📄 activity_labels.txt
├── 📁 train/
│   ├── 📄 X_train.txt
│   ├── 📄 y_train.txt
│   ├── 📄 subject_train.txt
│   └── 📁 Inertial Signals/
│       ├── 📄 body_acc_x_train.txt
│       ├── 📄 body_acc_y_train.txt
│       └── ... (9 total files)
└── 📁 test/
    ├── 📄 X_test.txt
    ├── 📄 y_test.txt
    ├── 📄 subject_test.txt
    └── 📁 Inertial Signals/
        ├── 📄 body_acc_x_test.txt
        ├── 📄 body_acc_y_test.txt
        └── ... (9 total files)
```

## 🔧 Verify Your Setup

Run this code to check if everything is set up correctly:

```python
import os
from pathlib import Path

def verify_dataset(dataset_path):
    """Verify UCI HAR dataset is properly downloaded"""
    required_files = [
        "README.txt",
        "features.txt", 
        "activity_labels.txt",
        "train/X_train.txt",
        "train/y_train.txt",
        "test/X_test.txt",
        "test/y_test.txt"
    ]
    
    dataset_dir = Path(dataset_path)
    
    print("🔍 Checking dataset files...")
    missing_files = []
    
    for file in required_files:
        file_path = dataset_dir / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ Missing {len(missing_files)} files!")
        print("Please check your dataset path and re-download if necessary.")
        return False
    else:
        print("\n🎉 All files found! Dataset is ready to use.")
        return True

# Test your dataset
dataset_path = r"YOUR_PATH_HERE/UCI HAR Dataset/UCI HAR Dataset"
verify_dataset(dataset_path)
```

## 🚨 Troubleshooting

### Problem: "File not found" errors
**Solution**: Check that your `dataset_path` points to the correct location:
```python
# Make sure this path exists and contains the files
dataset_path = r"d:\Datasets\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset"
```

### Problem: "Permission denied" when downloading
**Solution**: 
- Run terminal/command prompt as administrator
- Or download manually from the UCI website

### Problem: "Corrupted zip file"
**Solution**: 
- Delete the zip file and re-download
- Check your internet connection
- Try the manual download option

### Problem: Different file structure
**Solution**: The dataset should extract to a folder called "UCI HAR Dataset" with the structure shown above. If your structure is different, adjust the paths in the code.

## 📊 Dataset Information

- **Total size**: ~60MB compressed, ~250MB uncompressed
- **Samples**: 10,299 (7,352 training + 2,947 test)
- **Features**: 561 per sample
- **Activities**: 6 different activities
- **Subjects**: 30 people

## 🎯 Ready to Start Learning!

Once you've downloaded and verified the dataset:

1. **Update paths** in all Python scripts
2. **Run the verification script** to confirm setup
3. **Start with the interactive notebook**: `Sensor_Data_Learning_Journey.ipynb`
4. **Or run the complete analysis**: `python explore_uci_har.py`

**Happy learning! 🚀**
