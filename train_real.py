import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# --- UNIFIED FEATURE EXTRACTOR (Must match app.py exactly) ---
def extract_features(img):
    if img is None: 
        return None
        
    # 1. Standardize dimensions
    img = cv2.resize(img, (256, 128))

    # 2. Color DNA (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    color_features = hist.flatten()

    # 3. Structural DNA (Grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_features = cv2.resize(gray, (64, 32)).flatten() / 255.0

    # Output: Vector of exactly 2,304 features
    return np.concatenate([color_features, gray_features])

def train_real_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, 'dataset')
    
    # Fix #2: Ensuring strict path compliance for the models folder
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'currency_model.pkl')
    
    folder_map = {
        "ten_new": 0, "ten_old": 1,
        "twenty_new": 2, "twenty_old": 3,
        "fifty_new": 4, "fifty_old": 5,
        "hundred_new": 6, "hundred_old": 7,
        "two_hundred": 8,
        "five_hundred": 9,
        "two_thousand": 10
    }
    
    X = []
    y = []
    
    print("=== Step 1: Scanning Dataset & Applying Augmentation ===")
    if not os.path.exists(dataset_dir):
        print(f"Error: Could not find the folder at {dataset_dir}")
        return

    original_count = 0
    augmented_count = 0

    for folder_name, label_idx in folder_map.items():
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.exists(folder_path):
            continue
            
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    original_count += 1
                    
                    # Fix #3: Rotational Data Augmentation
                    # 0 Degrees (Original)
                    X.append(extract_features(img))
                    y.append(label_idx)
                    
                    # 90 Degrees Clockwise
                    img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    X.append(extract_features(img_90))
                    y.append(label_idx)
                    
                    # 180 Degrees
                    img_180 = cv2.rotate(img, cv2.ROTATE_180)
                    X.append(extract_features(img_180))
                    y.append(label_idx)
                    
                    # 270 Degrees (90 Counter-Clockwise)
                    img_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    X.append(extract_features(img_270))
                    y.append(label_idx)
                    
                    augmented_count += 4

    print(f"Processed {original_count} original images.")
    print(f"Generated {augmented_count} total training samples via augmentation.")
    print("Training Random Forest...")
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    joblib.dump(clf, model_path)
    print(f"Success! Robust AI model saved to: {model_path}")

if __name__ == '__main__':
    train_real_model()