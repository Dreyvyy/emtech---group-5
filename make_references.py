
from pathlib import Path
from datasets import load_from_disk

# Existing dataset folder
DATASET_DIR = Path("data/asl_alphabet_dataset")
OUTPUT_DIR = Path("reference")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset = load_from_disk(str(DATASET_DIR))

# Some saved datasets have a "train" split.
if hasattr(dataset, "keys") and "train" in dataset:
    dataset = dataset["train"]

saved = set()

for sample in dataset:
    label = sample.get("label")
    image = sample.get("image")

    if image is None or label is None:
        continue

    # Convert numeric class IDs to letters when necessary.
    if isinstance(label, int):
        # Check dataset_info/state if IDs don't correspond to A-Z.
        if 0 <= label <= 25:
            letter = chr(ord("A") + label)
        else:
            continue
    else:
        letter = str(label).strip().upper()

    if len(letter) != 1 or not ("A" <= letter <= "Z"):
        continue

    if letter in saved:
        continue

    image.save(OUTPUT_DIR / f"{letter}.png")
    saved.add(letter)
    print(f"Saved {letter}.png")

    if len(saved) == 26:
        break

missing = sorted(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - saved)
print(f"\nCreated {len(saved)} reference images in: {OUTPUT_DIR.resolve()}")

if missing:
    print("Letters not found:", ", ".join(missing))
