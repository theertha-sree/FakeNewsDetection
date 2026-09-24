import pandas as pd
import torch
import numpy as np
import joblib

from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ==========================================
# 2. LOAD PRE-TRAINED BERT
# ==========================================

print("\nLoading pre-trained BERT...")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

bert = BertModel.from_pretrained("bert-base-uncased")

bert.to(device)

# IMPORTANT:
# We are NOT training BERT
bert.eval()

print("BERT loaded successfully!")


# ==========================================
# 3. LOAD DATA
# ==========================================

data = pd.read_csv("dataset/train.csv")

# Start with 2,000 articles
data = data.reset_index(drop=True)

print("\nArticles:", len(data))


# ==========================================
# 4. SPLIT DATA
# ==========================================

train_data, val_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

print("Training articles:", len(train_data))
print("Validation articles:", len(val_data))


# ==========================================
# 5. FUNCTION TO EXTRACT BERT FEATURES
# ==========================================

def extract_features(texts):

    features = []

    batch_size = 8

    for start in range(0, len(texts), batch_size):

        batch_texts = texts[start:start + batch_size]

        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():

            outputs = bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # CLS token representation
        cls_features = outputs.last_hidden_state[:, 0, :]

        features.append(
            cls_features.cpu().numpy()
        )

        print(
            f"Processed {min(start + batch_size, len(texts))}/{len(texts)}"
        )

    return np.vstack(features)


# ==========================================
# 6. EXTRACT TRAINING FEATURES
# ==========================================

print("\nExtracting BERT features for training data...")

X_train = extract_features(
    train_data["text"].tolist()
)

y_train = train_data["label"].values


# ==========================================
# 7. EXTRACT VALIDATION FEATURES
# ==========================================

print("\nExtracting BERT features for validation data...")

X_val = extract_features(
    val_data["text"].tolist()
)

y_val = val_data["label"].values


# ==========================================
# 8. TRAIN LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression...")

classifier = LogisticRegression(
    max_iter=1000
)

classifier.fit(
    X_train,
    y_train
)

print("Logistic Regression training completed!")


# ==========================================
# 9. VALIDATION
# ==========================================

predictions = classifier.predict(X_val)

accuracy = accuracy_score(
    y_val,
    predictions
)

print("\n==============================")
print("VALIDATION RESULTS")
print("==============================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        predictions,
        target_names=["Fake", "Real"]
    )
)


# ==========================================
# 10. SAVE CLASSIFIER
# ==========================================

import os

os.makedirs("model/logistic_model", exist_ok=True)

joblib.dump(
    classifier,
    "model/logistic_model/classifier.pkl"
)

print("\nClassifier saved successfully!")

print(
    "Location: model/logistic_model/classifier.pkl"
)

print("\nTraining completed!")