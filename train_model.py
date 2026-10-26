
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

CSV_PATH = "data/hand_landmarks.csv"
MODEL_PATH = "model_landmarks.joblib"

# Load landmark data
df = pd.read_csv(CSV_PATH).dropna()

X = df.iloc[:, :63].values
y = df["label"].astype(str).values

# Convert labels into consistent class names
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("Total samples:", len(df))
print("Classes:", list(encoder.classes_))
print("\nSamples per class:")
print(df["label"].value_counts().sort_index())

# Use an unstratified split because some classes may have few samples
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", probability=True, class_weight="balanced")
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("\nTest accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=range(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

# Save model and label encoder together
joblib.dump(
    {"model": model, "label_encoder": encoder},
    MODEL_PATH
)

print(f"\nSaved model to {MODEL_PATH}")
