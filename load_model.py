from transformers import BertForSequenceClassification

# Load pretrained BERT
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

print("BERT model loaded successfully!")

print("\nNumber of labels:", model.config.num_labels)
print("Labels:", model.config.id2label)