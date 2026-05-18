"""
Task 2: Temporal order predictor — determines which of two MIDI excerpts comes first.

Feature construction:
    - 36-dim feature vector for excerpt A
    - 36-dim feature vector for excerpt B
    - 36-dim element-wise difference (A - B)
    - 7-dim boundary pitch features
    ──────────────────────────────────────
    Total: 36 + 36 + 36 + 7 = 115 dims

Binary target: 1 = A precedes B;  0 = B precedes A.
Reverse augmentation: each pair (A, B, 1) generates a symmetric pair (B, A, 0),
doubling the training set and teaching the model temporal directionality.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib

from extract_features import extract_midi_features

try:
    import miditoolkit
except ImportError:
    raise ImportError("miditoolkit is required: pip install miditoolkit")


_FEATURE_DIM = 36
_BOUNDARY_DIM = 7         # see _boundary_features() for layout
_TOTAL_DIM = _FEATURE_DIM * 3 + _BOUNDARY_DIM  # 115


def _boundary_features(midi_path: str) -> tuple:
    """
    Extract boundary pitch information from a MIDI excerpt.
    Returns (first_pitch, last_pitch, first_pc_onehot[0:5]) — 7 dims total.
    First and last pitch are normalized to [0, 1].
    """
    try:
        midi = miditoolkit.MidiFile(str(midi_path))
        notes = []
        for instr in midi.instruments:
            if not instr.is_drum:
                notes.extend(instr.notes)
        notes.sort(key=lambda n: n.start)
        if not notes:
            return tuple([0.0] * _BOUNDARY_DIM)
        first_pitch = notes[0].pitch / 127.0
        last_pitch  = notes[-1].pitch / 127.0
        # One-hot encoding of the first note's pitch class (top 5 classes only)
        pc_onehot = [0.0] * 5
        pc = notes[0].pitch % 12
        if pc < 5:
            pc_onehot[pc] = 1.0
        return (first_pitch, last_pitch) + tuple(pc_onehot)
    except Exception:
        return tuple([0.0] * _BOUNDARY_DIM)


def _build_feature_vector(path_a: str, path_b: str) -> np.ndarray:
    """Concatenate features of two MIDI excerpts into a 115-dim vector."""
    feat_a = np.array(extract_midi_features(path_a))
    feat_b = np.array(extract_midi_features(path_b))
    diff   = feat_a - feat_b
    boundary = np.array(_boundary_features(path_a) + _boundary_features(path_b)[:2])
    # boundary: 7 dims from A + 2 dims (first/last pitch) from B = 9, crop to 7
    combined = np.concatenate([feat_a, feat_b, diff, boundary[:_BOUNDARY_DIM]])
    assert combined.shape[0] == _TOTAL_DIM, f"Feature dimension mismatch: {combined.shape[0]}"
    return combined


class TemporalPredictor:
    """
    Predicts the temporal ordering of two MIDI excerpts within the same piece.

    Label convention:
        1 -> excerpt at path_a precedes excerpt at path_b
        0 -> excerpt at path_b precedes excerpt at path_a
    """

    def __init__(self, n_estimators: int = 500, random_state: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.is_trained = False

    def _augment(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Reverse-pair augmentation: for each (A, B, label) generate (B, A, 1-label).
        The three feature blocks (feat_a, feat_b, diff) are swapped and the diff negated.
        """
        fa  = X[:, :_FEATURE_DIM]
        fb  = X[:, _FEATURE_DIM:_FEATURE_DIM * 2]
        diff = X[:, _FEATURE_DIM * 2:_FEATURE_DIM * 3]
        bnd  = X[:, _FEATURE_DIM * 3:]

        # Swap A/B and negate difference
        X_aug = np.concatenate([fb, fa, -diff, bnd], axis=1)
        y_aug = 1 - y

        X_full = np.vstack([X, X_aug])
        y_full = np.concatenate([y, y_aug])
        return X_full, y_full

    def train(self, pairs: list, labels: list) -> dict:
        """
        Train the predictor.

        Args:
            pairs:  List of (path_a, path_b) tuples.
            labels: List of 0/1 temporal order labels.

        Returns:
            Dict with cross-validation accuracy statistics.
        """
        X = np.array([_build_feature_vector(a, b) for a, b in pairs])
        y = np.array(labels, dtype=int)

        X_aug, y_aug = self._augment(X, y)

        cv_scores = cross_val_score(self.clf, X_aug, y_aug, cv=5, scoring="accuracy")
        self.clf.fit(X_aug, y_aug)
        self.is_trained = True

        return {
            "cv_mean_accuracy": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "n_train_samples": len(y_aug),
        }

    def predict(self, pairs: list) -> list:
        """
        Predict temporal order for a list of excerpt pairs.

        Returns:
            List of integers: 1 (A first) or 0 (B first).
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")
        X = np.array([_build_feature_vector(a, b) for a, b in pairs])
        return list(self.clf.predict(X))

    def predict_proba(self, pairs: list) -> np.ndarray:
        """Return probability distributions; shape (N, 2)."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")
        X = np.array([_build_feature_vector(a, b) for a, b in pairs])
        return self.clf.predict_proba(X)

    def save(self, model_path: str):
        joblib.dump(self.clf, model_path)

    def load(self, model_path: str):
        self.clf = joblib.load(model_path)
        self.is_trained = True
