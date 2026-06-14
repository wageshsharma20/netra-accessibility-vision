import os
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Fix #2: Strict path mapping
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'currency_model.pkl')

CURRENCY_VALUES = {
    0: "10", 1: "10",
    2: "20", 3: "20",
    4: "50", 5: "50",
    6: "100", 7: "100",
    8: "200", 
    9: "500", 
    10: "2000"
}

# --- UNIFIED FEATURE EXTRACTOR (Must match train_real.py exactly) ---
def extract_features(img):
    if img is None: 
        return None
        
    img = cv2.resize(img, (256, 128))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    color_features = hist.flatten()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_features = cv2.resize(gray, (64, 32)).flatten() / 255.0

    return np.concatenate([color_features, gray_features])

# Fix #4: Lowered threshold for debugging
def classify_currency(image_path, threshold=0.15):
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model not found at {MODEL_PATH}")
        return None, 0.0, "Model not trained. Run train_real.py first."

    clf = joblib.load(MODEL_PATH)
    
    img = cv2.imread(image_path)
    features = extract_features(img)
    
    if features is None:
        return None, 0.0, "Could not process image."

    feat = features.reshape(1, -1)
    proba = clf.predict_proba(feat)[0]
    print("\n----- DEBUG -----")

    for i in np.argsort(proba)[::-1]:
        print(
        f"Class={i}, Value={CURRENCY_VALUES.get(i)}, Confidence={proba[i]:.4f}"
    )

    print("-----------------\n")
    idx = int(np.argmax(proba))
    confidence = float(proba[idx])
    value = CURRENCY_VALUES.get(idx, "unknown")
    
    # Fix #4: Explicit Terminal Debugging
    print("--------------------------------------------------")
    print(f"DEBUG -> Prediction Class: {idx} (₹{value})")
    print(f"DEBUG -> Confidence Score: {confidence:.4f}")
    print("--------------------------------------------------")
    
    if confidence < threshold:
        return None, confidence, "Not confident. Possible fake or blurry note."
    
    return value, confidence, None

# --- ROUTING ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    tmp = os.path.join(SCRIPT_DIR, "_upload.jpg")
    request.files["image"].save(tmp)

    try:
        value, conf, err = classify_currency(tmp)
        spoken = err if value is None else f"{value} rupees detected."
        result = {
            "value": value, 
            "confidence": conf, 
            "spoken": spoken
        }

        return jsonify(result)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

if __name__ == "__main__":
    app.run(debug=True, port=5000)