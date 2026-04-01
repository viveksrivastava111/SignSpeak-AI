import os
import pickle
import numpy as np
from typing import Optional


# Inference 

class GestureClassifier:
   
    def __init__(self, model_path: str = "models/gesture_classifier.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'.\n"
                "Run `python train_model.py` first to train and save the model."
            )
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        self._model  = bundle["model"]
        self._labels = bundle["labels"]   # list[str] — index → gesture name
        print(f"[GestureClassifier] Loaded model with {len(self._labels)} classes: {self._labels}")

    def predict(self, landmarks: np.ndarray) -> tuple[Optional[str], float]:
       
        try:
            X     = landmarks.reshape(1, -1)
            proba = self._model.predict_proba(X)[0]
            idx   = int(np.argmax(proba))
            return self._labels[idx], float(proba[idx])
        except Exception as e:
            print(f"[GestureClassifier] Prediction error: {e}")
            return None, 0.0


# Training
def train(
    data_dir:   str = "data/samples",
    output_path:str = "models/gesture_classifier.pkl",
    test_size:  float = 0.2,
) -> dict:
   
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    X, y, labels = _load_dataset(data_dir)

    if len(X) == 0:
        raise ValueError(f"No training data found in '{data_dir}'. Collect samples first.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators = 200,
            max_depth    = None,
            random_state = 42,
            n_jobs       = -1,
        )),
    ])

    print(f"\n[Training] {len(X_train)} train / {len(X_test)} test samples across {len(labels)} classes")
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    report  = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    accuracy= report["accuracy"]

    print(f"\n[Training] Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, y_pred, target_names=labels))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"model": pipeline, "labels": labels}, f)

    print(f"[Training] Model saved to '{output_path}'")
    return {"accuracy": accuracy, "labels": labels, "n_samples": len(X)}


def _load_dataset(data_dir: str):
    """Walk data_dir and return X (landmarks), y (int labels), label names."""
    X, y = [], []
    labels = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    for idx, label in enumerate(labels):
        label_dir = os.path.join(data_dir, label)
        files = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
        for fname in files:
            landmarks = np.load(os.path.join(label_dir, fname))
            X.append(landmarks)
            y.append(idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), labels
