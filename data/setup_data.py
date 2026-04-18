"""
Dataset Setup Script for Underwater AUV Project

This script handles unpacking local datasets (like 'Gate Task.zip')
or serves as a placeholder for downloading the Box dataset once access is granted.
"""

import os
import zipfile

def extract_local_dataset(zip_path, extract_to):
    """Extracts a local zip file if it exists."""
    if os.path.exists(zip_path):
        print(f"[INFO] Found local dataset at {zip_path}. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"[SUCCESS] Dataset extracted to {extract_to}")
    else:
        print(f"[WARNING] Local dataset {zip_path} not found.")

def main():
    print("=== CMPE 401: AUV Dataset Setup ===")
    
    # 1. Ensure data directory exists
    data_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(data_dir)
    
    # Define expected paths
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    os.makedirs(os.path.join(images_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(images_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, "val"), exist_ok=True)
    
    print(f"[INFO] Dataset structure ready at {data_dir}")
    
    # 2. Check for Gate Task.zip in project root
    gate_task_zip = os.path.join(project_root, "Gate Task.zip")
    extract_local_dataset(gate_task_zip, data_dir)
    
    # 3. Check for front.zip in project root
    front_zip = os.path.join(project_root, "front.zip")
    extract_local_dataset(front_zip, data_dir)
    
    # 4. Box Dataset Placeholder
    print("\n[INFO] If using the Box dataset, please download the files and place them in:")
    print(f"  - Images: {os.path.join(images_dir, 'train')} and {os.path.join(images_dir, 'val')}")
    print(f"  - Labels: {os.path.join(labels_dir, 'train')} and {os.path.join(labels_dir, 'val')}")
    print("\n[INFO] Once labels are adjusted, update the 'names' field in data/dataset.yaml.")

if __name__ == "__main__":
    main()
