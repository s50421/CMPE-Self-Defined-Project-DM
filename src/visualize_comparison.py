"""
Visual Comparison Script for AUV Project

This script grabs a random image from the validation set, runs the YOLO model on it,
and plots the Ground Truth labels vs the Predicted labels side-by-side for easy visual comparison
in the project report.
"""

import os
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import config

def get_random_val_image(dataset_yaml_path):
    import yaml
    with open(dataset_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    base_path = data.get('path', '')
    val_paths = data.get('val', [])
    if isinstance(val_paths, str):
        val_paths = [val_paths]
        
    all_images = []
    for vp in val_paths:
        full_dir = os.path.join(base_path, vp)
        if os.path.exists(full_dir):
            for img in os.listdir(full_dir):
                if img.endswith(('.jpg', '.png', '.jpeg')):
                    all_images.append(os.path.join(full_dir, img))
                    
    if not all_images:
        return None
    return random.choice(all_images)

def get_ground_truth_labels(img_path):
    # YOLO format: images/val/img.jpg -> labels/val/img.txt
    label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    labels.append((cls_id, cx, cy, w, h))
    return labels

def draw_boxes(img, boxes, class_names, color=(0, 255, 0)):
    # Draw YOLO format boxes on image (cx, cy, w, h normalized)
    h_img, w_img, _ = img.shape
    for cls_id, cx, cy, w, h in boxes:
        x_center, y_center = int(cx * w_img), int(cy * h_img)
        width, height = int(w * w_img), int(h * h_img)
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = class_names.get(cls_id, f"Class {cls_id}")
        cv2.putText(img, label, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    print("=== CMPE 401: Visual Comparison Tool ===")
    
    # Default to the final deployment weights if they exist, otherwise use the best experiment
    weights_path = os.path.join(config.RESULTS_DIR, "okmr_final_deployment", "weights", "best.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(config.RESULTS_DIR, "exp3_yolo11s_25e", "weights", "best.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(config.RESULTS_DIR, "exp1_yolo11n_10e", "weights", "best.pt")
    
    if not os.path.exists(weights_path):
        print(f"[ERROR] Could not find trained weights at {weights_path}. Run experiments first.")
        return
        
    print(f"[INFO] Loading model from {weights_path}")
    model = YOLO(weights_path)
    
    img_path = get_random_val_image(config.DATASET)
    if not img_path:
        print("[ERROR] No validation images found.")
        return
        
    print(f"[INFO] Selected validation image: {img_path}")
    
    # 1. Load Original Image
    img_gt = cv2.imread(img_path)
    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
    img_pred = img_gt.copy()
    
    # 2. Get Ground Truth
    gt_labels = get_ground_truth_labels(img_path)
    draw_boxes(img_gt, gt_labels, model.names, color=(0, 255, 0)) # Green for GT
    
    # 3. Get Predictions
    results = model(img_path)[0]
    pred_boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        # Convert xyxy to cx, cy, w, h normalized
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        h_img, w_img, _ = img_gt.shape
        cx = ((x1 + x2) / 2) / w_img
        cy = ((y1 + y2) / 2) / h_img
        w = (x2 - x1) / w_img
        h = (y2 - y1) / h_img
        pred_boxes.append((cls_id, cx, cy, w, h))
        
    draw_boxes(img_pred, pred_boxes, model.names, color=(255, 0, 0)) # Red for Pred
    
    # 4. Plot Side-by-Side
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_gt)
    plt.title("Ground Truth (Labeled)")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_pred)
    plt.title("YOLO Predictions")
    plt.axis("off")
    
    plt.tight_layout()
    out_path = os.path.join(config.RESULTS_DIR, "visual_comparison.png")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"[SUCCESS] Visual comparison saved to {out_path}")

if __name__ == "__main__":
    main()
