"""
Inference Script for Visual Demonstration
Runs the trained YOLO model on a set of test images/videos and saves the output.
"""

import os
import argparse
from ultralytics import YOLO
import config

def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on underwater data")
    parser.add_argument("--weights", type=str, default=os.path.join(config.RESULTS_DIR, "exp3_yolo11s_25e", "weights", "best.pt"), help="Path to trained model weights")
    parser.add_argument("--source", type=str, default=os.path.join(config.DATA_DIR, "front", "test", "images"), help="Path to input image, video, or folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    print("=== CMPE 401: AUV Object Detection Inference ===")
    
    if not os.path.exists(args.weights):
        print(f"[ERROR] Weights not found at: {args.weights}")
        return
        
    if not os.path.exists(args.source):
        print(f"[ERROR] Source not found at: {args.source}")
        return

    # Load the trained model
    model = YOLO(args.weights)
    
    # Run inference
    print(f"\n[INFO] Running inference on {args.source}...")
    model.predict(
        source=args.source,
        conf=args.conf,
        save=True,              # Save annotated images/videos
        project=config.RESULTS_DIR,
        name="inference_output",
        exist_ok=True,
        device=config.DEVICE
    )
    
    output_dir = os.path.join(config.RESULTS_DIR, "inference_output")
    print(f"\n[SUCCESS] Inference completed. Visual results saved to {output_dir}")

if __name__ == "__main__":
    main()
