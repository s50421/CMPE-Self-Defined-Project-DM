"""
Experiment Runner Script
CMPE 401 Self-Defined Project

This script demonstrates how we validate our model build-up. It runs three sequential
experiments to prove that our design choices (model capacity and training duration)
are optimal for detecting morphological features (e.g., shark vs swordfish noses).
"""

import os
import sys
import subprocess
import config

def get_python_executable():
    """Returns the path to the virtual environment python if it exists, else sys.executable."""
    venv_python = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable

def run_experiment(model, epochs, name, imgsz=None, export=False):
    print(f"\n" + "="*50)
    print(f"Kicking off {name}: Model={model}, Epochs={epochs}" + (f", ImgSz={imgsz}" if imgsz else ""))
    print("="*50)
    
    # We use subprocess to run train.py with the correct Python interpreter
    command = [
        get_python_executable(), 
        "src/train.py", 
        "--model", model, 
        "--epochs", str(epochs), 
        "--name", name
    ]
    if imgsz:
        command.extend(["--imgsz", str(imgsz)])
    if export:
        command.append("--export")
    
    # Use the current environment to ensure it runs correctly
    subprocess.run(command, check=True)

def main():
    print("=== CMPE 401: AUV Model Validation Experiments ===")
    print("This will run 3 sequential experiments to build up and prove our model design.")
    
    # Experiment 1: Baseline
    # We start with the smallest model to establish a fast baseline of underfitting/convergence.
    run_experiment("yolo11n.pt", 10, "exp1_yolo11n_10e")
    
    # Experiment 2: Capacity Check
    # We increase the network capacity (Nano -> Small) to see if the added depth/width 
    # improves detection of pointed vs round noses.
    run_experiment("yolo11s.pt", 10, "exp2_yolo11s_10e")
    
    # Experiment 3: Convergence Run
    # We take the best model (YOLO11s) and extend the training duration to reach optimal mAP.
    run_experiment("yolo11s.pt", 25, "exp3_yolo11s_25e")
    
    # Experiment 4: Edge Deployment Optimization
    # We train the smallest model at half resolution (320px) to maximize inference speed (FPS)
    # and export it to ONNX format, which is the standard for deploying on low-power AUV hardware.
    run_experiment("yolo11n.pt", 10, "exp4_yolo11n_320px_onnx", imgsz=320, export=True)
    
    print("\n[SUCCESS] All experiments completed! Run 'python src/evaluate.py' to compile metrics.")

if __name__ == "__main__":
    main()
