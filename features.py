"""Shared feature extraction: 21 MediaPipe hand landmarks -> 63 normalized values."""
import numpy as np

FEATURE_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def landmarks_to_features(hand_landmarks, handedness_label=None):
    """Convert MediaPipe landmarks to a position- and scale-invariant vector.

    1. Make the wrist (landmark 0) the origin  -> ignores where the hand is in the frame.
    2. Divide by the largest absolute value    -> ignores distance from the camera.
    3. Mirror left hands                       -> one model works for both hands.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    pts -= pts[0]
    if handedness_label == "Left":
        pts[:, 0] *= -1
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts.flatten()


def get_hand_features(results, index=0):
    """Return (features, landmarks) for the first detected hand, or (None, None)."""
    if not results.multi_hand_landmarks:
        return None, None
    lms = results.multi_hand_landmarks[index]
    label = None
    if results.multi_handedness:
        label = results.multi_handedness[index].classification[0].label
    return landmarks_to_features(lms, label), lms