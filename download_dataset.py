from datasets import load_dataset

DATASET_NAME = "Marxulia/asl_sign_languages_alphabets_v03"

print("Downloading ASL alphabet dataset...")

dataset = load_dataset(DATASET_NAME, split="train")

print("Total images:", len(dataset))
print("Available columns:", dataset.column_names)
print("First sample:", dataset[0]["label"])
print("Label names:", dataset.features["label"].names)

dataset.save_to_disk("data/asl_alphabet_dataset")

print("Dataset saved successfully!")