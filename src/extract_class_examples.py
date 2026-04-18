import os
import cv2
import glob
import random

def draw_and_save(img_path, label_path, class_id, dataset_name, out_dir):
    img = cv2.imread(img_path)
    if img is None:
        return False
    
    h_img, w_img, _ = img.shape
    found = False
    
    # We will draw ALL boxes of `class_id` in this image
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cid = int(parts[0])
            if cid == class_id:
                found = True
                cx, cy, w, h = map(float, parts[1:5])
                
                # YOLO format to pixel coords
                x_center, y_center = int(cx * w_img), int(cy * h_img)
                width, height = int(w * w_img), int(h * h_img)
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Draw thick red box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                # Add text
                text = f"{dataset_name} | Class {class_id}"
                cv2.putText(img, text, (max(0, x1), max(20, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    if found:
        out_name = f"{dataset_name}_class_{class_id}.jpg"
        out_path = os.path.join(out_dir, out_name.replace(" ", "_"))
        cv2.imwrite(out_path, img)
        return True
    return False

def find_examples(base_path, dataset_name, out_dir):
    labels_dir = os.path.join(base_path, "train", "labels")
    images_dir = os.path.join(base_path, "train", "images")
    
    if not os.path.exists(labels_dir):
        print(f"[WARN] No labels found at {labels_dir}")
        return
        
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    # Shuffle so we get random examples, not just the first video frame
    random.seed(42)
    random.shuffle(label_files)
    
    found_classes = set()
    
    for label_path in label_files:
        basename = os.path.basename(label_path)
        img_name = basename.replace('.txt', '.jpg')
        img_path = os.path.join(images_dir, img_name)
        
        # Some might be .png or .jpeg
        if not os.path.exists(img_path):
            img_name = basename.replace('.txt', '.jpeg')
            img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            img_name = basename.replace('.txt', '.png')
            img_path = os.path.join(images_dir, img_name)
            
        if not os.path.exists(img_path):
            continue
            
        with open(label_path, 'r') as f:
            classes_in_file = set()
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes_in_file.add(int(parts[0]))
                    
        for cid in classes_in_file:
            if cid not in found_classes:
                # Try to draw and save
                success = draw_and_save(img_path, label_path, cid, dataset_name, out_dir)
                if success:
                    found_classes.add(cid)
                    print(f"[INFO] Found example for {dataset_name} Class {cid}")

def main():
    out_dir = os.path.abspath(os.path.join("results", "class_examples"))
    os.makedirs(out_dir, exist_ok=True)
    
    print("Extracting examples from Gate Task...")
    find_examples(os.path.join("data", "Gate Task"), "Gate_Task", out_dir)
    
    print("Extracting examples from front...")
    find_examples(os.path.join("data", "front"), "Front", out_dir)
    
    print(f"\nDone! Examples saved to {out_dir}")

if __name__ == "__main__":
    main()
