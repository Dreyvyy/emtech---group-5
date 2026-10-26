
import argparse
import os
import random
import time
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp

from features import get_hand_features

CONF_THRESHOLD = 0.30
HOLD_SECONDS = 1.0
QUIZ_TIMEOUT = 10.0
QUIZ_QUESTIONS = 10


def put(frame, text, pos, scale=0.8, color=(255, 255, 255), thick=2):
    cv2.putText(
        frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
        scale, (0, 0, 0), thick + 3
    )
    cv2.putText(
        frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thick
    )


def load_reference(letter, folder, size=200):
    for ext in ("png", "jpg", "jpeg"):
        path = os.path.join(folder, f"{letter}.{ext}")
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return cv2.resize(img, (size, size))
    return None


def show_reference(frame, letter, folder):
    img = load_reference(str(letter), folder)
    h, w = frame.shape[:2]

    if w < 220 or h < 220:
        return

    if img is not None:
        frame[10:210, w - 210:w - 10] = img
    else:
        cv2.rectangle(
            frame, (w - 210, 10), (w - 10, 210),
            (60, 60, 60), -1
        )
        put(frame, f"No image for {letter}", (w - 205, 110), 0.5)

    put(frame, f"Reference: {letter}", (w - 205, 232), 0.6)


class Smoother:
    def __init__(self, n=7, min_votes=5):
        self.buf = deque(maxlen=n)
        self.min_votes = min_votes

    def update(self, letter):
        self.buf.append(letter)

        if letter is None:
            return None

        common, count = Counter(self.buf).most_common(1)[0]
        return common if common is not None and count >= self.min_votes else None

    def clear(self):
        self.buf.clear()


def main():
    parser = argparse.ArgumentParser(description="SignLearn webcam application")
    parser.add_argument("--model", default="model_landmarks.joblib")
    parser.add_argument("--reference", default="reference")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model file not found: {args.model}")

    saved = joblib.load(args.model)
    label_encoder = None
    if isinstance(saved, dict) and "model" in saved:
        model = saved["model"]
        label_encoder = saved.get("label_encoder")
    else:
        model = saved

    print(f"Loaded model: {args.model}")
    print(f"Model input features: {getattr(model, 'n_features_in_', 'unknown')}")

    if not hasattr(model, "predict_proba"):
        raise SystemExit("The model must support predict_proba().")

    def class_to_letter(class_value):
        # The training script stores a LabelEncoder and the SVC predicts encoded IDs.
        value = label_encoder.inverse_transform([class_value])[0] if label_encoder is not None else class_value
        value = str(value)
        # The downloaded dataset extraction script may save ClassLabel IDs (0-25)
        # instead of letter names. This assumes the dataset orders IDs A, B, ... Z.
        if value.isdigit():
            idx = int(value)
            if 0 <= idx < 26:
                return chr(ord("A") + idx)
        return value.upper()

    letters = sorted({class_to_letter(c) for c in model.classes_})
    if not letters:
        raise SystemExit("The model has no trained classes.")

    expected_features = getattr(model, "n_features_in_", None)

    # SignLearn's current MediaPipe feature extractor is expected to return 63 values.
    # Fail early if the loaded model was trained with a different input format.
    if expected_features is not None and expected_features != 63:
        print(
            f"WARNING: {args.model} expects {expected_features} features, "
            "but SignLearn's hand-landmark pipeline is expected to provide 63. "
            "Retrain the model with data/hand_landmarks.csv before using this app."
        )

    # Normalize string class labels for display and comparison.
    # Keep model.classes_ itself unchanged for probability indexing.
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    mode = "recognize"
    target_idx = 0
    smoother = Smoother()

    quiz = {
        "target": None,
        "start": 0,
        "hold": None,
        "asked": 0,
        "results": [],
        "last_pred": None,
        "done": False,
    }

    def start_quiz():
        quiz.update(
            target=random.choice(letters),
            start=time.time(),
            hold=None,
            asked=0,
            results=[],
            last_pred=None,
            done=False,
        )
        smoother.clear()

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise SystemExit(
            "Cannot open webcam. Check camera permissions or try --camera 1."
        )

    try:
        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as hands:

            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    print("Unable to read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                feats, lms = get_hand_features(results)

                letter = None
                conf = 0.0
                mismatch = False

                if feats is not None:
                    mp_draw.draw_landmarks(
                        frame, lms, mp_hands.HAND_CONNECTIONS
                    )

                    actual_features = len(feats)

                    # Prevent model crashes from mismatched input sizes.
                    if (
                        expected_features is not None
                        and actual_features == expected_features
                    ):
                        probs = model.predict_proba([feats])[0]
                        best = probs.argmax()
                        conf = float(probs[best])
                        print(
                                "Prediction:", model.classes_[best],
                                "| Confidence:", round(conf * 100, 2), "%"
)

                        if conf >= CONF_THRESHOLD:
                            letter = class_to_letter(model.classes_[best])
                    else:
                        mismatch = True

                stable = smoother.update(letter)
                now = time.time()

                if mode == "recognize":
                    put(
                        frame, "Mode: Recognize",
                        (10, 30), 0.7, (0, 255, 255)
                    )
                    put(
                        frame, f"Letter: {stable or '-'}",
                        (10, 80), 1.4, (0, 255, 0), 3
                    )

                    if mismatch:
                        put(
                            frame,
                            f"Feature mismatch: model={expected_features}, webcam={len(feats)}",
                            (10, 155), 0.55, (0, 0, 255)
                        )
                        put(
                            frame,
                            "Run train_model.py on collected landmark samples",
                            (10, 185), 0.5, (0, 0, 255)
                        )
                    else:
                        message = (
                            f"Confidence: {conf:.0%}"
                            if feats is not None
                            else "No hand detected"
                        )
                        put(frame, message, (10, 115), 0.7)

                    if stable is not None:
                        show_reference(frame, stable, args.reference)

                elif mode == "learn":
                    target = letters[target_idx]

                    put(
                        frame, "Mode: Learn (N/P = change letter)",
                        (10, 30), 0.65, (0, 255, 255)
                    )
                    put(
                        frame, f"Target: {target}",
                        (10, 80), 1.2, (255, 200, 0), 3
                    )
                    show_reference(frame, target, args.reference)

                    if mismatch:
                        put(
                            frame, "Model input format does not match webcam",
                            (10, 130), 0.55, (0, 0, 255)
                        )
                    elif stable == target:
                        put(
                            frame, f"Correct! Sign for {target}.",
                            (10, 130), 0.7, (0, 255, 0)
                        )
                    elif stable is not None:
                        put(
                            frame, f"I see {stable}. Compare with reference.",
                            (10, 130), 0.6, (0, 120, 255)
                        )
                    else:
                        put(
                            frame, "Show your hand to the camera",
                            (10, 130), 0.7
                        )

                elif mode == "quiz":
                    if quiz["target"] is None:
                        start_quiz()

                    if quiz["done"]:
                        correct = sum(r["ok"] for r in quiz["results"])
                        put(
                            frame, "Quiz finished!",
                            (10, 50), 1.0, (0, 255, 255), 3
                        )
                        put(
                            frame,
                            f"Score: {correct}/{len(quiz['results'])}",
                            (10, 95), 1.0, (0, 255, 0), 3
                        )
                        missed = sorted({
                            str(r["target"])
                            for r in quiz["results"]
                            if not r["ok"]
                        })
                        practice = ", ".join(missed) if missed else "Great job!"
                        put(
                            frame, f"Practice more: {practice}",
                            (10, 135), 0.6
                        )
                        put(frame, "Press 3 to retry", (10, 170), 0.7)

                    else:
                        elapsed = now - quiz["start"]

                        put(
                            frame,
                            f"Quiz {quiz['asked'] + 1}/{QUIZ_QUESTIONS}",
                            (10, 30), 0.7, (0, 255, 255)
                        )
                        put(
                            frame,
                            f"Sign this letter: {quiz['target']}",
                            (10, 80), 1.0, (255, 200, 0), 3
                        )
                        put(
                            frame,
                            f"Time left: {max(0, QUIZ_TIMEOUT - elapsed):.0f}s",
                            (10, 115), 0.7
                        )

                        if elapsed > 5:
                            show_reference(
                                frame, quiz["target"], args.reference
                            )

                        if stable is not None:
                            quiz["last_pred"] = stable

                        finished = False
                        correct = False

                        if stable == quiz["target"]:
                            if quiz["hold"] is None:
                                quiz["hold"] = now

                            if now - quiz["hold"] >= HOLD_SECONDS:
                                finished = True
                                correct = True
                        else:
                            quiz["hold"] = None

                        if elapsed >= QUIZ_TIMEOUT:
                            finished = True

                        if finished:
                            quiz["results"].append({
                                "target": quiz["target"],
                                "pred": quiz["last_pred"],
                                "ok": correct,
                            })
                            quiz["asked"] += 1

                            if quiz["asked"] >= QUIZ_QUESTIONS:
                                quiz["done"] = True
                            else:
                                quiz.update(
                                    target=random.choice(letters),
                                    start=now,
                                    hold=None,
                                    last_pred=None,
                                )

                            smoother.clear()

                put(
                    frame,
                    "1 Recognize | 2 Learn | 3 Quiz | Q Quit",
                    (10, frame.shape[0] - 20), 0.55
                )

                cv2.imshow("SignLearn", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("1"):
                    mode = "recognize"
                elif key == ord("2"):
                    mode = "learn"
                elif key == ord("3"):
                    mode = "quiz"
                    start_quiz()
                elif key == ord("n"):
                    target_idx = (target_idx + 1) % len(letters)
                elif key == ord("p"):
                    target_idx = (target_idx - 1) % len(letters)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
