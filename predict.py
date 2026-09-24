import torch
import joblib

from transformers import BertTokenizer, BertModel


# ==========================================
# 1. DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ==========================================
# 2. LOAD PRE-TRAINED BERT
# ==========================================

print("\nLoading BERT...")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

bert = BertModel.from_pretrained("bert-base-uncased")

bert.to(device)

bert.eval()

print("BERT loaded!")


# ==========================================
# 3. LOAD LOGISTIC REGRESSION
# ==========================================

print("Loading classifier...")

classifier = joblib.load(
    "model/logistic_model/classifier.pkl"
)

print("Classifier loaded!")


# ==========================================
# 4. GET NEWS FROM USER
# ==========================================

news = input("\nEnter a news article: ")


# ==========================================
# 5. CONVERT NEWS INTO BERT FEATURES
# ==========================================

encoding = tokenizer(
    news,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

input_ids = encoding["input_ids"].to(device)

attention_mask = encoding["attention_mask"].to(device)


# ==========================================
# 6. EXTRACT BERT FEATURES
# ==========================================

with torch.no_grad():

    outputs = bert(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    # CLS token
    features = outputs.last_hidden_state[:, 0, :]

    features = features.cpu().numpy()


# ==========================================
# 7. PREDICTION
# ==========================================

prediction = classifier.predict(features)[0]

probabilities = classifier.predict_proba(features)[0]


fake_probability = probabilities[0] * 100

real_probability = probabilities[1] * 100


# ==========================================
# 8. RESULT
# ==========================================

if prediction == 0:

    result = "FAKE NEWS"

else:

    result = "REAL NEWS"


print("\n==============================")
print("       PREDICTION")
print("==============================")

print("Result:", result)

print(f"Fake probability: {fake_probability:.2f}%")

print(f"Real probability: {real_probability:.2f}%")