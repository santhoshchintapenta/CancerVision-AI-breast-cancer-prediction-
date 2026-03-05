import joblib
import pickle
import os

# Try to load with different protocols
pkl_file = "breast_cancer_pipeline.pkl"

try:
    # Try loading with protocol 4 (more compatible)
    with open(pkl_file, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully with pickle")
except Exception as e:
    print(f"Error with pickle: {e}")
    try:
        # Try with joblib
        model = joblib.load(pkl_file)
        print("Model loaded with joblib")
    except Exception as e2:
        print(f"Error with joblib: {e2}")

# Save with protocol 4 for better compatibility
if 'model' in locals():
    joblib.dump(model, pkl_file, protocol=4)
    print("Model resaved with protocol 4")
