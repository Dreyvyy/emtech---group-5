"""Extract normalized MediaPipe hand landmarks for SignLearn training.

Run from the SignLearn project folder:
    python extract_landmarks.py

Expected local Hugging Face dataset folder: data/asl_alphabet_dataset
Output: data/hand_landmarks.csv
"""
from pathlib import Path
import csv

import numpy as np
from datasets import load_from_disk
import mediapipe as mp

from features import FEATURE_COLUMNS, landmarks_to_features

DATASET_DIR = Path("data/asl_alphabet_dataset")
OUTPUT_CSV = Path("data/hand_landmarks.csv")


def get_label(sample, dataset):
    """Convert ClassLabel IDs to class names when the dataset provides them."""
    label = sample["label"]
    label_feature = dataset.features.get("label")
    names = getattr(label_feature, "names", None)
    if names is not None and isinstance(label, (int, np.integer)):
        return str(names[int(label)]).strip()
    return str(label).strip()


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}. "
            "Check DATASET_DIR at the top of this script."
        )

    dataset = load_from_disk(str(DATASET_DIR))
    if hasattr(dataset, "keys") and not isinstance(dataset, list):
        # load_from_disk may return a DatasetDict with a train split.
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            first_split = next(iter(dataset.keys()))
            dataset = dataset[first_split]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    mp_hands = mp.solutions.hands
    written = 0
    skipped = 0

    # Use the same normalized feature function as the webcam app.
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands, OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_COLUMNS + ["label"])

        for i, sample in enumerate(dataset):
            image = sample["image"].convert("RGB")
            image_np = np.asarray(image)
            result = hands.process(image_np)

            if not result.multi_hand_landmarks:
                skipped += 1
                continue

            hand = result.multi_hand_landmarks[0]
            handedness = None
            if result.multi_handedness:
                handedness = result.multi_handedness[0].classification[0].label

            features = landmarks_to_features(hand, handedness)
            label = get_label(sample, dataset)
            writer.writerow([float(v) for v in features] + [label])
            written += 1

            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(dataset)} images; extracted {written} hands")

    print(f"Finished. Saved {written} hand samples to {OUTPUT_CSV}")
    print(f"Skipped {skipped} images where MediaPipe found no hand.")
    print("Features per sample: 63; label column: label")


if __name__ == "__main__":
    main()
