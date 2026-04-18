"""
Final AUV Object Detection Pipeline
CMPE 401 Self-Defined Project

This script is designed to be run ONCE you have analyzed your experimental results
and settled on the optimal hyperparameter configuration.

It performs the complete end-to-end process:
1. Trains the final model using your chosen parameters.
2. Evaluates the model against the project's Success Criteria (>0.85 mAP50, >30 FPS).
3. Generates a visual side-by-side comparison of Ground Truth vs. Predictions.
4. Exports the finalized model to ONNX format for OKMR edge hardware deployment.
"""

import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import config

# =====================================================================
# FINAL MODEL PARAMETERS
# (Update these based on your analysis of the `run_experiments.py` data)
# =====================================================================
FINAL_RUN_NAME = "okmr_final_deployment"
MODEL_ARCH     = "yolo11s.pt"  # YOLO11s proved best for accuracy vs speed
IMG_SIZE       = 640           # 320px drops accuracy significantly, so keep 640
EPOCHS         = 25            # 25 epochs is required for full convergence

# Success Criteria Targets
TARGET_MAP50   = 0.85
TARGET_FPS     = 30.0
# =====================================================================

def evaluate_criteria(model, weights_path):
    print("\n" + "="*50)
    print(" STEP 2: EVALUATING SUCCESS CRITERIA")
    print("="*50)
    
    # Run validation
    metrics = model.val(data=config.DATASET, device=config.DEVICE)
    
    # Extract Metrics
    map50 = metrics.box.map50
    
    # Extract Speed and calculate FPS
    speed_dict = metrics.speed
    total_ms_per_img = sum(speed_dict.values())
    fps = 1000.0 / total_ms_per_img if total_ms_per_img > 0 else 0
    
    print("\n--- FINAL PERFORMANCE REPORT ---")
    
    # Check Accuracy
    print(f"Accuracy (mAP50): {map50:.3f}")
    if map50 >= TARGET_MAP50:
        print(f"  -> [PASS] Meets > {TARGET_MAP50} requirement.")
    else:
        print(f"  -> [FAIL] Falls short of {TARGET_MAP50} requirement.")
        
    # Check Speed
    print(f"Inference Speed:  {fps:.1f} FPS")
    if fps >= TARGET_FPS:
        print(f"  -> [PASS] Meets > {TARGET_FPS} FPS real-time edge requirement.")
    else:
        print(f"  -> [FAIL] Too slow. Fails the {TARGET_FPS} FPS requirement.")
        
    print("-" * 32 + "\n")
    return map50, fps

def generate_visual_comparison(model):
    print("\n" + "="*50)
    print(" STEP 3: GENERATING FINAL VISUAL COMPARISONS")
    print("="*50)
    
    # We will reuse the helper functions from visualize_comparison
    # to avoid duplicating code.
    try:
        from visualize_comparison import get_random_val_image, get_ground_truth_labels, draw_boxes
        
        img_path = get_random_val_image(config.DATASET)
        if not img_path:
            print("[ERROR] No validation image found for visual comparison.")
            return
            
        print(f"[INFO] Evaluating frame: {img_path}")
        
        img_gt = cv2.imread(img_path)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_pred = img_gt.copy()
        
        # Ground Truth
        gt_labels = get_ground_truth_labels(img_path)
        draw_boxes(img_gt, gt_labels, model.names, color=(0, 255, 0))
        
        # Predictions
        results = model(img_path, verbose=False)[0]
        pred_boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            h_img, w_img, _ = img_gt.shape
            cx, cy = ((x1 + x2) / 2) / w_img, ((y1 + y2) / 2) / h_img
            w, h = (x2 - x1) / w_img, (y2 - y1) / h_img
            pred_boxes.append((cls_id, cx, cy, w, h))
            
        draw_boxes(img_pred, pred_boxes, model.names, color=(255, 0, 0))
        
        # Plotting
        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(img_gt)
        plt.title("Ground Truth (Labeled)")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_pred)
        plt.title(f"Final Model Predictions ({FINAL_RUN_NAME})")
        plt.axis("off")
        
        out_path = os.path.join(config.RESULTS_DIR, FINAL_RUN_NAME, "final_visual_comparison.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[SUCCESS] Final visual comparison saved to: {out_path}")
        
    except ImportError:
        print("[WARNING] Could not import visualize_comparison. Ensure it is in the src/ directory.")

def main():
    print(f"\n>>> KICKING OFF THE FINAL PIPELINE: {FINAL_RUN_NAME} <<<")
    
    # ---------------------------------------------------------
    # STEP 1: TRAIN THE FINAL MODEL
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" STEP 1: TRAINING FINAL MODEL")
    print("="*50)
    model = YOLO(MODEL_ARCH)
    
    # We overwrite the project directory so everything neatly saves to results/okmr_final_deployment
    model.train(
        data=config.DATASET,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=config.DEVICE,
        project=config.RESULTS_DIR,
        name=FINAL_RUN_NAME,
        exist_ok=True,
        batch=16, # Standard safe batch size
        workers=4, # Reduced from 8 to save RAM
        cache=False # Disabled RAM caching to prevent memory overload
    )
    
    # Grab the best weights path
    best_weights = os.path.join(config.RESULTS_DIR, FINAL_RUN_NAME, "weights", "best.pt")
    
    # ---------------------------------------------------------
    # STEP 2: EVALUATE PERFORMANCE
    # ---------------------------------------------------------
    # We reload the model from best.pt to ensure we test the best epoch
    final_model = YOLO(best_weights)
    evaluate_criteria(final_model, best_weights)
    
    # ---------------------------------------------------------
    # STEP 3: VISUAL COMPARISONS
    # ---------------------------------------------------------
    generate_visual_comparison(final_model)
    
    # ---------------------------------------------------------
    # STEP 4: EXPORT FOR EDGE DEPLOYMENT
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" STEP 4: EXPORTING MODEL FOR OKMR DEPLOYMENT")
    print("="*50)
    print(f"[INFO] Exporting {best_weights} to ONNX format (imgsz={IMG_SIZE})...")
    
    exported_path = final_model.export(format="onnx", imgsz=IMG_SIZE, dynamic=False)
    print(f"[SUCCESS] Model successfully exported for edge hardware.")
    print(f"          ONNX File Location: {exported_path}")
    print("\n" + "="*50)
    print(">>> FINAL PIPELINE COMPLETE! <<<")
    print(f"Everything your team needs is located in: results/{FINAL_RUN_NAME}/")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
