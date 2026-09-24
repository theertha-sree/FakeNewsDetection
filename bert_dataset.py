import pandas as pd
from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load training and testing data
train_data = pd.read_csv("dataset/train.csv")
test_data = pd.read_csv("dataset/test.csv")

# Take a small sample first
sample_texts = train_data["text"].head(5).tolist()

# Tokenize the sample
encoded = tokenizer(
    sample_texts,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

print("Tokenization successful!")
print()

print("Input IDs shape:")
print(encoded["input_ids"].shape)

print()

print("Attention Mask shape:")
print(encoded["attention_mask"].shape)

print()

print("First article input IDs:")
print(encoded["input_ids"][0])

print()

print("First article attention mask:")
print(encoded["attention_mask"][0])