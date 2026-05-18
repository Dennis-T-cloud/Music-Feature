"""
Task 1: Composer classifier — multi-class prediction from 36-dim MIDI feature vectors.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path

from extract_features import extract_midi_features


class ComposerClassifier:
    """
    Classifies MIDI excerpts by composer using a Random Forest on 36-dim feature vectors.

    Training pipeline:
        1. Call extract_midi_features() on each MIDI file to obtain a 36-dim vector.
        2. Encode composer name labels as integers.
        3. Fit a Random Forest with 500 trees.
    """

    def __init__(self, n_estimators: int = 500, random_state: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=-1,          # use all available CPU cores
            class_weight="balanced",
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def _extract_batch(self, midi_paths: list) -> np.ndarray:
        """Extract features for a list of paths; returns an (N, 36) matrix."""
        features = []
        for path in midi_paths:
            feats = extract_midi_features(str(path))
            features.append(feats)
        return np.array(features, dtype=float)

    def train(self, midi_paths: list, labels: list) -> dict:
        """
        Train the classifier.

        Args:
            midi_paths: List of MIDI file paths.
            labels:     Corresponding composer name strings.

        Returns:
            Dict with cross-validation accuracy statistics.
        """
        X = self._extract_batch(midi_paths)
        y = self.label_encoder.fit_transform(labels)

        # 5-fold cross-validation before fitting on the full training set
        cv_scores = cross_val_score(self.clf, X, y, cv=5, scoring="accuracy")

        self.clf.fit(X, y)
        self.is_trained = True

        return {
            "cv_mean_accuracy": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "n_classes": len(self.label_encoder.classes_),
            "classes": list(self.label_encoder.classes_),
        }

    def predict(self, midi_paths: list) -> list:
        """
        Predict the composer for each MIDI file.

        Args:
            midi_paths: List of MIDI file paths.

        Returns:
            List of composer name strings.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")
        X = self._extract_batch(midi_paths)
        y_pred = self.clf.predict(X)
        return list(self.label_encoder.inverse_transform(y_pred))

    def predict_proba(self, midi_paths: list) -> np.ndarray:
        """Return class probability distributions; shape (N, n_classes)."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")
        X = self._extract_batch(midi_paths)
        return self.clf.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        """Return the Random Forest feature importance scores (36-dim)."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")
        return self.clf.feature_importances_

    def save(self, model_path: str):
        """Serialize the trained model to disk."""
        joblib.dump({"clf": self.clf, "label_encoder": self.label_encoder}, model_path)

    def load(self, model_path: str):
        """Load a serialized model from disk."""
        obj = joblib.load(model_path)
        self.clf = obj["clf"]
        self.label_encoder = obj["label_encoder"]
        self.is_trained = True


if __name__ == "__main__":
    import glob
    paths = glob.glob("../data/**/*.mid", recursive=True)
    if not paths:
        print("No MIDI files found. Place data under the data/ directory.")
    else:
        labels = [Path(p).parent.name for p in paths]
        clf = ComposerClassifier(n_estimators=10)
        result = clf.train(paths, labels)
        print("Training complete:", result)
        preds = clf.predict(paths[:3])
        print("Predictions:", preds)
