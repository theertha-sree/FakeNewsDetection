# 📰 Fake News Detection using BERT

An AI-based Fake News Detection system that classifies news articles as **Fake News** or **Real News** using **BERT** for text feature extraction and **Logistic Regression** for classification.

The project also includes a **Streamlit web interface** where users can enter a news article and get a prediction with probability scores.

---

## 🚀 Live Demo

Try the deployed application:

👉 https://fakenewsdetection-kzkuao7myye9nctvsq8iya.streamlit.app/

---

## 📌 Project Overview

Fake news is misleading or false information presented as legitimate news. Detecting fake news automatically can help users identify potentially unreliable articles.

In this project:

1. News data is collected and prepared.
2. The dataset is divided into training and testing data.
3. BERT converts news text into meaningful numerical features.
4. Logistic Regression learns to classify those features.
5. The trained classifier predicts whether new articles are Fake or Real.
6. Streamlit provides a simple web interface for users.

### Architecture

```text
                 News Article
                      │
                      ▼
              Text Preprocessing
                      │
                      ▼
              BERT Tokenization
                      │
                      ▼
            Pre-trained BERT Model
                      │
                      ▼
             CLS Feature Extraction
                      │
                      ▼
             Logistic Regression
                      │
              ┌───────┴───────┐
              ▼               ▼
          FAKE NEWS        REAL NEWS
```

## 📂 Project Structure

```text
FakeNewsDetection/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── model/
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── logistic_model/
│       └── classifier.pkl
│
├── app.py
├── train.py
├── prepare_data.py
├── tokenize_data.py
├── bert_dataset.py
├── load_model.py
├── bert_training.py
├── predict.py
├── evaluate.py
├── test_real_world.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Technologies Used

- **Python**
- **BERT**
- **Hugging Face Transformers**
- **PyTorch**
- **Logistic Regression**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- # 📊 Dataset

This project uses the **ISOT Fake News Dataset**.

The dataset contains two CSV files:

```text
Fake.csv
True.csv
