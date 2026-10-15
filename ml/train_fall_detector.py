#!/usr/bin/env python3
"""
Reproducible Fall Detection Training Pipeline
----------------------------------------------
Trains a scikit-learn RandomForestClassifier on 47 biomechanical & spatial features
extracted from MediaPipe Pose landmarks, using the public UR Fall Detection (URFD) dataset.

Output Artifacts:
- Model: backend/app/model/random_forest_model.joblib
- Metadata & Metrics: backend/app/model/model_metadata.json
- Extracted Features CSV: ml/urfd_47_features.csv
"""

import os
import sys
import json
import csv
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

# Ensure backend/ is in sys.path to import the exact feature definitions
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.features import FEATURE_NAMES, extract_features, extract_feature_vector

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import joblib


def load_or_extract_features(csv_path: Path, fall_dir: Path, adl_dir: Path):
    """
    Loads features from CSV if cached, or extracts them from images via MediaPipe.
    """
    if csv_path.exists():
        print(f"[Dataset] Loading cached features from {csv_path}...")
        X, y, filenames = [], [], []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vec = [float(row[feat]) for feat in FEATURE_NAMES]
                X.append(vec)
                y.append(int(row["label"]))
                filenames.append(row["filename"])
        return np.array(X), np.array(y), filenames

    print("[Dataset] CSV not found. Extracting features from raw images with MediaPipe...")
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_task_path = Path("/tmp/pose_landmarker_lite.task")
    if not model_task_path.exists():
        import urllib.request
        print("[MediaPipe] Downloading pose_landmarker_lite.task...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            str(model_task_path)
        )

    base_options = python.BaseOptions(model_asset_path=str(model_task_path))
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)

    X, y, filenames = [], [], []

    # 1. Process Fall images
    fall_files = sorted([f for f in os.listdir(fall_dir) if f.endswith(".png")])
    print(f"[MediaPipe] Processing {len(fall_files)} Fall images from {fall_dir}...")
    for f in fall_files:
        img_path = os.path.join(fall_dir, f)
        mp_img = mp.Image.create_from_file(img_path)
        res = detector.detect(mp_img)
        if res.pose_landmarks and len(res.pose_landmarks) > 0:
            lms = [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in res.pose_landmarks[0]]
            vec = extract_feature_vector(lms)
            X.append(vec)
            y.append(1)  # 1 = Fall
            filenames.append(f)

    fall_detected_count = len(X)
    print(f"  -> Extracted {fall_detected_count} poses from {len(fall_files)} fall images.")

    # 2. Process ADL (Activities of Daily Living) images
    adl_files = sorted([f for f in os.listdir(adl_dir) if f.endswith(".png")])
    print(f"[MediaPipe] Processing {len(adl_files)} Normal/ADL images from {adl_dir}...")
    for f in adl_files:
        img_path = os.path.join(adl_dir, f)
        mp_img = mp.Image.create_from_file(img_path)
        res = detector.detect(mp_img)
        if res.pose_landmarks and len(res.pose_landmarks) > 0:
            lms = [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in res.pose_landmarks[0]]
            vec = extract_feature_vector(lms)
            X.append(vec)
            y.append(0)  # 0 = Normal / ADL
            filenames.append(f)

    adl_detected_count = len(X) - fall_detected_count
    print(f"  -> Extracted {adl_detected_count} poses from {len(adl_files)} ADL images.")

    # Save to CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"] + list(FEATURE_NAMES))
        for fname, label, vec in zip(filenames, y, X):
            writer.writerow([fname, label] + [f"{v:.6f}" for v in vec])

    print(f"[Dataset] Saved {len(X)} feature rows to {csv_path}")
    return np.array(X), np.array(y), filenames


def train_and_evaluate():
    print("=" * 70)
    print("Smart Elderly Fall Detection: ML Training & Evaluation")
    print("=" * 70)

    csv_path = REPO_ROOT / "ml" / "urfd_47_features.csv"
    fall_dir = Path("/tmp/urfd_data/fall-01-cam0-rgb")
    adl_dir = Path("/tmp/urfd_data/adl-01-cam0-rgb")

    # If unzipped folders are not in /tmp, check Downloads
    if not fall_dir.exists() or not adl_dir.exists():
        import zipfile
        downloads_fall = Path.home() / "Downloads" / "fall-01-cam0-rgb.zip"
        downloads_adl = Path.home() / "Downloads" / "adl-01-cam0-rgb.zip"
        if downloads_fall.exists() and downloads_adl.exists():
            print("[Dataset] Unzipping URFD dataset from Downloads...")
            with zipfile.ZipFile(downloads_fall, "r") as z:
                z.extractall("/tmp/urfd_data")
            with zipfile.ZipFile(downloads_adl, "r") as z:
                z.extractall("/tmp/urfd_data")

    X, y, filenames = load_or_extract_features(csv_path, fall_dir, adl_dir)

    total_samples = len(y)
    fall_samples = int(np.sum(y == 1))
    adl_samples = int(np.sum(y == 0))

    print(f"\n[Dataset Summary]")
    print(f"  Total valid pose samples: {total_samples}")
    print(f"  Fall (class 1): {fall_samples} ({fall_samples / total_samples * 100:.1f}%)")
    print(f"  Normal/ADL (class 0): {adl_samples} ({adl_samples / total_samples * 100:.1f}%)")
    print(f"  Total features: {X.shape[1]} (expected {len(FEATURE_NAMES)})")

    # 1. 5-Fold Stratified Cross-Validation
    print("\n[Cross-Validation] Running 5-Fold Stratified Cross-Validation...")
    rf_cv = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_cv, X, y, cv=cv, scoring="accuracy")
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    print(f"  5-Fold CV Mean Accuracy: {cv_mean * 100:.2f}% (+/- {cv_std * 100:.2f}%)")
    print(f"  Individual fold scores: {[round(float(s), 4) for s in cv_scores]}")

    # 2. Honest Held-Out Test Evaluation (80% train, 20% test, stratified)
    print("\n[Train/Test Split] Evaluating on 20% held-out test split (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1))
    rec = float(recall_score(y_test, y_pred, pos_label=1))
    f1 = float(f1_score(y_test, y_pred, pos_label=1))
    auc = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"  Test Accuracy:  {acc * 100:.2f}%")
    print(f"  Test Precision: {prec * 100:.2f}% (positive class = Fall)")
    print(f"  Test Recall:    {rec * 100:.2f}% (positive class = Fall)")
    print(f"  Test F1-Score:  {f1 * 100:.2f}%")
    print(f"  Test ROC AUC:   {auc:.4f}")
    print(f"\n  Confusion Matrix (rows: true [0, 1], cols: pred [0, 1]):")
    print(f"    [[TN={cm[0][0]}, FP={cm[0][1]}],")
    print(f"     [FN={cm[1][0]}, TP={cm[1][1]}]]")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal/ADL (0)", "Fall (1)"]))

    # 3. Top Feature Importances
    importances = clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    top_features = [
        {"feature": FEATURE_NAMES[i], "importance": round(float(importances[i]), 4)}
        for i in top_indices
    ]
    print("  Top 10 Influential Features:")
    for rank, item in enumerate(top_features, 1):
        print(f"    {rank}. {item['feature']}: {item['importance']:.4f}")

    # 4. Save Final Artifact to backend/app/model/random_forest_model.joblib
    model_dir = REPO_ROOT / "backend" / "app" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "random_forest_model.joblib"
    metadata_path = model_dir / "model_metadata.json"

    # Save the trained model instance
    joblib.dump(clf, model_path)
    print(f"\n[Model Artifact] Saved model to {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

    # Save metadata JSON with honest metrics
    metadata = {
        "model_name": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "n_features_in": int(clf.n_features_in_),
        "classes": [int(c) for c in clf.classes_],
        "class_mapping": {"0": "Normal/ADL", "1": "Fall"},
        "training_dataset": "UR Fall Detection Dataset (fall-01 and adl-01 sequences)",
        "samples": {
            "total": total_samples,
            "fall_count": fall_samples,
            "adl_count": adl_samples,
            "train_count": len(y_train),
            "test_count": len(y_test),
        },
        "evaluation_metrics": {
            "test_accuracy": round(acc, 4),
            "test_precision": round(prec, 4),
            "test_recall": round(rec, 4),
            "test_f1_score": round(f1, 4),
            "test_roc_auc": round(auc, 4),
            "confusion_matrix": {
                "true_negative": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1],
            },
            "five_fold_cv_accuracy_mean": round(cv_mean, 4),
            "five_fold_cv_accuracy_std": round(cv_std, 4),
        },
        "top_features": top_features,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "trained_by": "Marnia & Keerthana",
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Model Metadata] Saved metrics and evaluation details to {metadata_path}")

    # 5. Model Reload Verification
    print("\n[Verification] Reloading saved model via joblib...")
    reloaded_model = joblib.load(model_path)
    assert hasattr(reloaded_model, "predict"), "Reloaded model missing predict method"
    assert reloaded_model.n_features_in_ == 47, f"Feature count mismatch: {reloaded_model.n_features_in_}"
    assert list(reloaded_model.classes_) == [0, 1], f"Unexpected classes: {reloaded_model.classes_}"

    # Test single-vector predict
    sample_fall = X_test[y_test == 1][0]
    sample_adl = X_test[y_test == 0][0]
    pred_fall = reloaded_model.predict([sample_fall])[0]
    pred_adl = reloaded_model.predict([sample_adl])[0]
    prob_fall = reloaded_model.predict_proba([sample_fall])[0]
    prob_adl = reloaded_model.predict_proba([sample_adl])[0]

    print(f"  Sample Fall vector: predicted={pred_fall} (prob={prob_fall[1]:.2f})")
    print(f"  Sample ADL vector:  predicted={pred_adl} (prob={prob_adl[0]:.2f})")
    print("[Verification] Model verification PASSED 100%. Ready for FastAPI backend.")


if __name__ == "__main__":
    train_and_evaluate()
