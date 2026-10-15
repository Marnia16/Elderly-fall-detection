# Feature Schema & Model Training Alignment Guide

## 1. Overview
This document specifies the feature extraction contract for the **Smart Elderly Fall Detection and Alert System**.
To ensure that predictions in production match model training in Google Colab, both training-time and inference-time pipelines use the exact same feature definitions.

- **Pose Estimator:** MediaPipe Pose (Full 33 landmarks)
- **Input per frame:** 33 landmarks with normalized `(x, y, z)` and `visibility` in `[0.0, 1.0]`
- **Extracted Feature Vector:** Exactly 47 numerical features (order listed below)
- **Model:** Scikit-learn `RandomForestClassifier` serialized via `joblib`
- **Model Artifact Destination:** `backend/app/model/random_forest_model.joblib`

---

## 2. MediaPipe Pose Landmark Map (33 Points)
| Index | Landmark Name | Index | Landmark Name | Index | Landmark Name |
|-------|---------------|-------|---------------|-------|---------------|
| 0 | `NOSE` | 11 | `LEFT_SHOULDER` | 22 | `RIGHT_THUMB` |
| 1 | `LEFT_EYE_INNER` | 12 | `RIGHT_SHOULDER` | 23 | `LEFT_HIP` |
| 2 | `LEFT_EYE` | 13 | `LEFT_ELBOW` | 24 | `RIGHT_HIP` |
| 3 | `LEFT_EYE_OUTER` | 14 | `RIGHT_ELBOW` | 25 | `LEFT_KNEE` |
| 4 | `RIGHT_EYE_INNER` | 15 | `LEFT_WRIST` | 26 | `RIGHT_KNEE` |
| 5 | `RIGHT_EYE` | 16 | `RIGHT_WRIST` | 27 | `LEFT_ANKLE` |
| 6 | `RIGHT_EYE_OUTER` | 17 | `LEFT_PINKY` | 28 | `RIGHT_ANKLE` |
| 7 | `LEFT_EAR` | 18 | `RIGHT_PINKY` | 29 | `LEFT_HEEL` |
| 8 | `RIGHT_EAR` | 19 | `LEFT_INDEX` | 30 | `RIGHT_HEEL` |
| 9 | `MOUTH_LEFT` | 20 | `RIGHT_INDEX` | 31 | `LEFT_FOOT_INDEX` |
| 10 | `MOUTH_RIGHT` | 21 | `LEFT_THUMB` | 32 | `RIGHT_FOOT_INDEX` |

---

## 3. Extracted Feature List (47 Features in Order)

### A. Biomechanical & Geometric Invariant Features (11 features)
These features capture body posture, orientation, and bounding geometry independent of camera distance.
1. `torso_vertical_angle_deg`: Angle (degrees) of vector connecting mid-shoulder to mid-hip relative to the vertical Y-axis. Standing $\approx 0^\circ - 25^\circ$; Fallen $\approx 70^\circ - 90^\circ$.
2. `torso_length`: Euclidean distance between mid-shoulder and mid-hip (used as normalization scale).
3. `bbox_aspect_ratio`: Bounding box width / height. Standing $< 0.6$; Fallen $> 1.0$.
4. `bbox_height_to_torso_ratio`: Bounding box height divided by torso length.
5. `hip_to_ankle_vertical_dist`: Vertical distance from mid-hip to mid-ankle divided by torso length.
6. `head_to_hip_vertical_dist`: Vertical distance from nose to mid-hip divided by torso length.
7. `shoulder_tilt_deg`: Angle of shoulder axis relative to horizontal.
8. `hip_tilt_deg`: Angle of hip axis relative to horizontal.
9. `knee_to_ankle_vertical_dist`: Vertical distance from mid-knee to mid-ankle divided by torso length.
10. `head_to_ankle_vertical_dist`: Vertical distance from nose to mid-ankle divided by torso length.
11. `mean_landmark_visibility`: Average visibility score across all 33 landmarks.

### B. Normalized Relative Spatial Coordinates (36 features)
12 key body joints normalized by centering at Mid-Hip $(H_x, H_y, H_z)$ and scaling by `torso_length`:
$\tilde{x} = \frac{x - H_x}{L_{torso}}$, $\tilde{y} = \frac{y - H_y}{L_{torso}}$, $\tilde{z} = \frac{z - H_z}{L_{torso}}$

Where:
- Mid-Hip: $H_x = \frac{\text{left\_hip}_x + \text{right\_hip}_x}{2}$, $H_y = \frac{\text{left\_hip}_y + \text{right\_hip}_y}{2}$, $H_z = \frac{\text{left\_hip}_z + \text{right\_hip}_z}{2}$
- Mid-Shoulder: $S_x = \frac{\text{left\_shoulder}_x + \text{right\_shoulder}_x}{2}$, $S_y = \frac{\text{left\_shoulder}_y + \text{right\_shoulder}_y}{2}$, $S_z = \frac{\text{left\_shoulder}_z + \text{right\_shoulder}_z}{2}$
- Torso Length: $L_{torso} = \sqrt{(H_x - S_x)^2 + (H_y - S_y)^2 + (H_z - S_z)^2}$ (with a minimum floor $\epsilon = 1e-4$)

12. `rel_nose_x`
13. `rel_nose_y`
14. `rel_nose_z`
15. `rel_left_shoulder_x`
16. `rel_left_shoulder_y`
17. `rel_left_shoulder_z`
18. `rel_right_shoulder_x`
19. `rel_right_shoulder_y`
20. `rel_right_shoulder_z`
21. `rel_mid_shoulder_x`
22. `rel_mid_shoulder_y`
23. `rel_mid_shoulder_z`
24. `rel_left_elbow_x`
25. `rel_left_elbow_y`
26. `rel_left_elbow_z`
27. `rel_right_elbow_x`
28. `rel_right_elbow_y`
29. `rel_right_elbow_z`
30. `rel_left_wrist_x`
31. `rel_left_wrist_y`
32. `rel_left_wrist_z`
33. `rel_right_wrist_x`
34. `rel_right_wrist_y`
35. `rel_right_wrist_z`
36. `rel_left_knee_x`
37. `rel_left_knee_y`
38. `rel_left_knee_z`
39. `rel_right_knee_x`
40. `rel_right_knee_y`
41. `rel_right_knee_z`
42. `rel_left_ankle_x`
43. `rel_left_ankle_y`
44. `rel_left_ankle_z`
45. `rel_right_ankle_x`
46. `rel_right_ankle_y`
47. `rel_right_ankle_z`

---

## 4. Google Colab Training Integration
In your Colab training script:

```python
# 1. Copy backend/app/features.py into your Colab workspace or import directly
from features import extract_feature_vector, FEATURE_NAMES

# 2. For each labeled frame from URFD or camera dataset:
# landmarks = results.pose_landmarks.landmark # list of 33 landmarks with x, y, z, visibility
features = extract_feature_vector(landmarks)
X.append(features)
y.append(label) # 1 for Fall, 0 for ADL (Activities of Daily Living)

# 3. Train the Random Forest
from sklearn.ensemble import RandomForestClassifier
import joblib

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# 4. Save model artifact
joblib.dump(clf, "random_forest_model.joblib")
# Place "random_forest_model.joblib" into backend/app/model/
```
