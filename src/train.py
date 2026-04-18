"""
Unified Training Script for Deep Learning-Based Underwater Object Detection
CMPE 401 Self-Defined Project
"""

import os
import argparse
from ultralytics import YOLO
import config

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for Underwater Object Detection")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME, help="YOLO model architecture to start with")
    parser.add_argument("--name", type=str, default="auv_detection_run", help="Name of the training run")
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE, help="Image size for training")
    parser.add_argument("--export", action="store_true", help="Export model to ONNX after training for edge deployment")
    args = parser.parse_args()

    print("=== CMPE 401: AUV Object Detection Training ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {config.DATASET}")
    print(f"Epochs: {args.epochs}")
    print(f"Image Size: {args.imgsz}")
    print(f"Device: {config.DEVICE}")
    print("="*45)
    
    # Initialize the YOLO model
    model = YOLO(args.model)
    
    # Start training
    model.train(
        data=config.DATASET,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=config.BATCH_SIZE,
        workers=config.WORKERS,
        project=config.RESULTS_DIR,
        name=args.name,
        device=config.DEVICE,
        exist_ok=True
    )
    
    print(f"\n[SUCCESS] Training completed! Results saved to: {os.path.join(config.RESULTS_DIR, args.name)}")
    
    if args.export:
        print("\n[INFO] Exporting model to ONNX format for lightweight edge deployment...")
        # Exporting to ONNX format, common for running inference on Jetson/RPi
        model.export(format="onnx")
        print("[SUCCESS] ONNX export complete.")

if __name__ == "__main__":
    main()
