from transformers import BertTokenizer

# Load the pretrained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Test sentence
text = "This is a fake news article."

# Tokenize the sentence
tokens = tokenizer.tokenize(text)

# Convert tokens to numbers
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Original text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)