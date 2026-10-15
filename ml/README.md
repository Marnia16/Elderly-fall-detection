# Machine Learning Model & Training Pipeline

This directory contains the reproducible training pipeline and dataset feature cache for the **Smart Elderly Fall Detection System**.

---

## 1. Dataset Overview

- **Source**: Public UR Fall Detection (URFD) dataset (RGB camera stream).
- **Raw Samples**: 160 Fall sequence frames (`fall-01-cam0-rgb`) and 150 Activities of Daily Living (ADL) sequence frames (`adl-01-cam0-rgb`).
- **Valid Detections**: 254 total valid frames where MediaPipe Tasks Pose Landmarker detected a human pose with confidence $\ge 0.5$ (122 Falls, 132 ADL).
- **Extracted Feature Cache**: [`urfd_47_features.csv`](./urfd_47_features.csv) (122 KB). Caching the extracted 47 features allows full, deterministic model training without requiring raw image downloads.

---

## 2. Feature Schema (47 Features)

The feature extraction code in [`backend/app/features.py`](../backend/app/features.py) strictly computes 47 standardized geometric, postural, and spatial features:

1. **Torso Vertical Angle & Bounding Box**: Torso angle to vertical ($y$-axis), bounding box aspect ratio, normalized height, and normalized width.
2. **Head & Shoulder Spatial Displacements**: Vertical descent ratios of nose and shoulders relative to hip center.
3. **Midpoint & Center-of-Mass Coordinates**: Horizontal and vertical positions of shoulders, hips, knees, and ankles.
4. **Key Biomechanical Joint Angles**: Left/right knees, elbows, and hips.
5. **Horizontal Spans**: Shoulder width, hip width, and ankle spread.
6. **Normalized Coordinates ($x, y$) for 12 Key Landmarks**: Nose, left/right shoulders, elbows, wrists, hips, knees, and ankles.

See [`backend/FEATURE_SCHEMA.md`](../backend/FEATURE_SCHEMA.md) for full coordinate definitions.

---

## 3. How to Retrain the Model

To retrain the model from the cached feature CSV:

```bash
# From the repository root using the backend virtual environment:
backend/.venv/bin/python ml/train_fall_detector.py
```

The script will:
1. Load `ml/urfd_47_features.csv`.
2. Perform 5-fold stratified cross-validation.
3. Train a `RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)`.
4. Evaluate on a held-out test split (20%).
5. Save the trained model to [`backend/app/model/random_forest_model.joblib`](../backend/app/model/random_forest_model.joblib) (~191 KB).
6. Export honest evaluation metrics to [`backend/app/model/model_metadata.json`](../backend/app/model/model_metadata.json).

---

## 4. Evaluation Results

- **5-Fold Cross-Validation Accuracy**: $96.45\% \pm 2.31\%$
- **Held-Out Test Set (51 samples)**:
  - **Accuracy**: $98.04\%$
  - **Precision**: $100.0\%$
  - **Recall**: $95.83\%$
  - **F1-Score**: $0.9787$
  - **ROC AUC**: $0.9977$
- **Feature Vector Length**: Exactly 47 features accepted by `model.n_features_in_`.
