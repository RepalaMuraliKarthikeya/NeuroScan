import os
import shutil
import urllib.request
import zipfile

def setup_dataset():
    """
    Instructions for obtaining the Chest X-Ray Pneumonia dataset.
    Since downloading a 1GB+ dataset from Kaggle via script requires Kaggle API tokens,
    this script creates the necessary directory structure and provides instructions.
    """
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset_split')
    
    print("=== Chest X-Ray Dataset Setup ===")
    print("1. Please download the dataset from Kaggle:")
    print("   URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("2. Extract the downloaded archive.")
    print("3. Place the 'train' and 'val' (or 'test') folders into this directory:")
    print(f"   -> {dataset_dir}")
    print("\nThe expected structure should be:")
    print(f"{dataset_dir}/")
    print("  ├── train/")
    print("  │   ├── NORMAL/")
    print("  │   └── PNEUMONIA/")
    print("  └── val/")
    print("      ├── NORMAL/")
    print("      └── PNEUMONIA/")
    print("\n-------------------------")
    print("Creating the directory structure for you now...")
    
    for split in ['train', 'val']:
        for cls in ['NORMAL', 'PNEUMONIA']:
            os.makedirs(os.path.join(dataset_dir, split, cls), exist_ok=True)
            
    print("Directories created successfully!")
    print("Please manually transfer the images from the Kaggle download into these folders before training.")

if __name__ == '__main__':
    setup_dataset()
