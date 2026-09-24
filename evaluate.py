import pandas as pd
import torch
import numpy as np
import joblib

from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# 2. LOAD BERT
# ==========================================

print("\nLoading BERT...")

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

bert = BertModel.from_pretrained(
    "bert-base-uncased"
)

bert.to(device)
bert.eval()

print("BERT loaded successfully!")


# ==========================================
# 3. LOAD TRAINED CLASSIFIER
# ==========================================

print("\nLoading Logistic Regression...")

classifier = joblib.load(
    "model/logistic_model/classifier.pkl"
)

print("Classifier loaded successfully!")


# ==========================================
# 4. LOAD TEST DATA
# ==========================================

print("\nLoading test dataset...")

data = pd.read_csv("dataset/test.csv")

print("Test articles:", len(data))


# ==========================================
# 5. EXTRACT BERT FEATURES
# ==========================================

def extract_features(texts):

    features = []

    batch_size = 8

    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch_texts = texts[
            start:start + batch_size
        ]

        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoding[
            "input_ids"
        ].to(device)

        attention_mask = encoding[
            "attention_mask"
        ].to(device)

        with torch.no_grad():

            outputs = bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        cls_features = (
            outputs.last_hidden_state[:, 0, :]
        )

        features.append(
            cls_features.cpu().numpy()
        )

        print(
            f"Processed "
            f"{min(start + batch_size, len(texts))}"
            f"/{len(texts)}"
        )

    return np.vstack(features)


# ==========================================
# 6. EXTRACT TEST FEATURES
# ==========================================

print("\nExtracting BERT features...")

X_test = extract_features(
    data["text"].tolist()
)

y_test = data["label"].values


# ==========================================
# 7. PREDICTION
# ==========================================

print("\nMaking predictions...")

predictions = classifier.predict(
    X_test
)


# ==========================================
# 8. ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Fake",
            "Real"
        ]
    )
)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


print("\nEvaluation completed!")