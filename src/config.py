"""
Centralized Configuration for YOLO Training

This file allows easy tweaking of all important variables.
"""

import os

# --- Project Paths ---
# Ensures results always save inside the project folder, regardless of where you run the script from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

import torch

# --- Hardware Acceleration ---
# Auto-detect CUDA. If not available (e.g. forgot to activate venv), fallback to CPU.
if torch.cuda.is_available():
    DEVICE = "0" 
else:
    print("[WARNING] CUDA not detected! Falling back to CPU. If you have a GPU, ensure your virtual environment is active.")
    DEVICE = "cpu"
WORKERS = 2  # Reduced to 2 to prevent RAM overloads

# --- Fast First-Round Training Settings ---
# To make training very fast, you can:
# 1. Reduce EPOCHS (e.g., 3-5)
# 2. Reduce IMG_SIZE (e.g., to 320 or 416)
EPOCHS = 20          # 10 epochs provides a good balance of accuracy and time for RTX
BATCH_SIZE = -1      # AutoBatch to maximize RTX VRAM utilization
IMG_SIZE = 640       # Increased from 416 for better small object detection

# --- Model & Dataset ---
MODEL_NAME = "yolo11n.pt"   # YOLOv11 Nano (Fastest model)
DATASET = os.path.join(DATA_DIR, "dataset.yaml")   # Underwater AUV dataset config
