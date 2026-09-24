import pandas as pd
from sklearn.model_selection import train_test_split

# Load our combined dataset
data = pd.read_csv("dataset/news.csv")

# Combine title and news text
data["content"] = data["title"] + " " + data["text"]

# Select input and labels
texts = data["content"]
labels = data["label"]

# Split into training and testing data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Save the training data
train_data = pd.DataFrame({
    "text": train_texts,
    "label": train_labels
})

train_data.to_csv("dataset/train.csv", index=False)

# Save the testing data
test_data = pd.DataFrame({
    "text": test_texts,
    "label": test_labels
})

test_data.to_csv("dataset/test.csv", index=False)

# Display results
print("Data splitting completed!")
print()
print("Total articles:", len(data))
print("Training articles:", len(train_data))
print("Testing articles:", len(test_data))

print()
print("Training label distribution:")
print(train_data["label"].value_counts())

print()
print("Testing label distribution:")
print(test_data["label"].value_counts())