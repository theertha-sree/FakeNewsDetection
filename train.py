import pandas as pd

# Load datasets
fake_news = pd.read_csv("dataset/Fake.csv")
real_news = pd.read_csv("dataset/True.csv")

# Add labels
fake_news["label"] = 0
real_news["label"] = 1

# Combine both datasets
news_data = pd.concat([fake_news, real_news], ignore_index=True)

# Keep only the columns we need
news_data = news_data[["title", "text", "label"]]

# Remove missing values
news_data = news_data.dropna()

# Shuffle the dataset
news_data = news_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the combined dataset
news_data.to_csv("dataset/news.csv", index=False)

# Display information
print("Dataset created successfully!")
print()
print(news_data.head())
print()
print("Total articles:", len(news_data))
print()
print("Fake news:", (news_data["label"] == 0).sum())
print("Real news:", (news_data["label"] == 1).sum())