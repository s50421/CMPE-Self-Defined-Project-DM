"""
My script to extract all the metrics from the trained models.
Instead of manually looking through console logs, this scans the 'results/' folder,
evaluates every model it finds, and dumps everything into a nice CSV file for my report.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import config

def format_row(cols):
    # Quick helper to make markdown table formatting easier
    return "| " + " | ".join(cols) + " |"

def evaluate_all():
    print("\n=== CMPE 401 Self-Defined Project: Automated Model Evaluation ===")
    
    # Bail out if the results directory doesn't exist yet
    if not os.path.exists(config.RESULTS_DIR):
        print(f"[ERROR] Can't find the results directory at {config.RESULTS_DIR}. Did you run training yet?")
        return
    
    # Grab all the subdirectories inside results/ (each one is a training run)
    runs = [d for d in os.listdir(config.RESULTS_DIR) if os.path.isdir(os.path.join(config.RESULTS_DIR, d))]
    
    if not runs:
        print("[WARNING] Results directory is empty.")
        return
        
    print("\nFound the following runs to evaluate:", runs)
    
    # Storing everything in a list of dicts so I can easily convert it to a pandas DataFrame later
    results_data = []
    
    for run in runs:
        # The best weights are always saved here by Ultralytics
        weights_path = os.path.join(config.RESULTS_DIR, run, "weights", "best.pt")
        
        if not os.path.exists(weights_path):
            print(f"[WARNING] No best.pt found for {run}. Skipping...")
            continue
            
        print(f"\n--- Evaluating {run} ---")
        try:
            # Load up the model weights
            model = YOLO(weights_path)
            
            # Run evaluation on the validation set. 
            metrics = model.val(data=config.DATASET, device=config.DEVICE)
            
            # Extract the core metrics I need for the report
            map50_95 = metrics.box.map
            map50 = metrics.box.map50
            
            # Precision and Recall (taking the mean across all classes)
            precision = metrics.box.mp
            recall = metrics.box.mr
            
            # Count the total number of parameters in the model (useful for size comparisons)
            params = sum(p.numel() for p in model.model.parameters())
            
            # Calculate Inference Speed (FPS)
            speed_dict = metrics.speed
            total_ms_per_img = sum(speed_dict.values())
            fps = 1000.0 / total_ms_per_img if total_ms_per_img > 0 else 0
            
            # Append it to my dataset
            results_data.append({
                "Run Name": run,
                "Model Size (Params)": params,
                "mAP50": f"{map50:.3f}",
                "mAP50-95": f"{map50_95:.3f}",
                "Precision": f"{precision:.3f}",
                "Recall": f"{recall:.3f}",
                "Inference FPS": f"{fps:.1f}"
            })
            
        except Exception as e:
            print(f"[ERROR] Something went wrong evaluating {run}: {e}")
            
    if not results_data:
        print("[ERROR] Couldn't successfully evaluate any models.")
        return

    # Shove it all into a DataFrame
    df = pd.DataFrame(results_data)
    
    # 1. Print out a Markdown table so I can easily copy-paste it into my README
    print("\n\n=== Final Comparison Table (Markdown) ===\n")
    headers = list(df.columns)
    print(format_row(headers))
    print(format_row(["---"] * len(headers)))
    for _, row in df.iterrows():
        print(format_row(row.astype(str).tolist()))
    print("\n(Copy and paste this table into the README!)\n")

    # 2. Save it to a CSV file in the /xsheets folder for safekeeping and Excel graphing
    xsheets_dir = os.path.join(config.PROJECT_ROOT, "xsheets")
    os.makedirs(xsheets_dir, exist_ok=True)
    
    csv_path = os.path.join(xsheets_dir, "compiled_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Saved the compiled CSV to: {csv_path}")
    
    # 3. Generate the Inference Speed (FPS) Plot
    try:
        plt.figure(figsize=(10, 6))
        run_names = df["Run Name"].tolist()
        fps_values = [float(x) for x in df["Inference FPS"]]
        
        plt.bar(run_names, fps_values, color='skyblue', edgecolor='black')
        plt.title("Performance Criteria: Inference Speed (FPS)", fontsize=14)
        plt.ylabel("Frames Per Second (Higher is Better)", fontsize=12)
        plt.xlabel("Experimental Runs", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_path = os.path.join(config.RESULTS_DIR, "inference_speed_comparison.png")
        plt.savefig(plot_path)
        print(f"[INFO] Saved FPS comparison plot to: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"[WARNING] Could not generate FPS plot: {e}")

if __name__ == "__main__":
    evaluate_all()
