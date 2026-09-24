import streamlit as st
import torch
import joblib
from transformers import BertTokenizer, BertModel


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FAKE NEWS DETECTION AI",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #0b1020;
        color: white;
    }

    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* Header */
    .main-title {
        text-align: center;
        font-size: 52px;
        font-weight: 800;
        margin-bottom: 5px;
        color: white;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 18px;
        margin-bottom: 35px;
    }

    /* Card */
    .glass-card {
        background: #111827;
        border: 1px solid #26324a;
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
    }

    /* Result */
    .result-real {
        background: #062e1b;
        border: 1px solid #16a34a;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
    }

    .result-fake {
        background: #3a0d0d;
        border: 1px solid #dc2626;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
    }

    .result-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .confidence {
        font-size: 20px;
        color: #d1d5db;
    }

    /* Probability cards */
    .prob-card {
        background: #111827;
        border: 1px solid #26324a;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }

    .prob-title {
        color: #9ca3af;
        font-size: 15px;
    }

    .prob-value {
        font-size: 30px;
        font-weight: 700;
        color: white;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #080d1a;
    }

    /* Text area */
    textarea {
        background-color: #0f172a !important;
        color: white !important;
        border-radius: 15px !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-size: 17px;
        font-weight: 700;
        background: #2563eb;
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background: #1d4ed8;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_models():

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    bert = BertModel.from_pretrained(
        "bert-base-uncased"
    )

    bert.eval()

    classifier = joblib.load(
        "model/logistic_model/classifier.pkl"
    )

    return tokenizer, bert, classifier


with st.spinner("Loading AI model..."):

    tokenizer, bert, classifier = load_models()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🛡️ FAKE NEWS DETECTION AI")

    st.markdown("---")

    st.markdown("### 🤖 AI Model")

    st.write("BERT + Logistic Regression")

    st.markdown("### 📊 Classification")

    st.write("Fake News")
    st.write("Real News")

    st.markdown("---")

    st.markdown("### ⚙️ How it works")

    st.write("""
    1. You enter a news article.
    2. BERT analyzes the text.
    3. BERT creates a text representation.
    4. Logistic Regression classifies it.
    5. The system returns the prediction.
    """)

    st.markdown("---")

    st.caption("AI-powered Fake News Detection")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛡️ FAKE NEWS DETECTION AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Fake News Detection using BERT'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INPUT CARD
# =========================================================

st.markdown(
    '<div class="glass-card">',
    unsafe_allow_html=True
)

st.markdown("### 📰 Enter News Article")

news = st.text_area(
    "",
    placeholder="Paste a news article or headline here...",
    height=250
)

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button("🔍 Analyze News"):

    if not news.strip():

        st.warning("⚠️ Please enter a news article first.")

    else:

        with st.spinner("🤖 AI is analyzing the news..."):

            encoding = tokenizer(
                news,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            with torch.no_grad():

                outputs = bert(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"]
                )

                features = outputs.last_hidden_state[:, 0, :]

                features = features.numpy()

            prediction = classifier.predict(features)[0]

            probabilities = classifier.predict_proba(features)[0]

            fake_probability = probabilities[0] * 100
            real_probability = probabilities[1] * 100

            confidence = max(
                fake_probability,
                real_probability
            )


        # =================================================
        # RESULT
        # =================================================

        if prediction == 0:

            st.markdown(
                f"""
                <div class="result-fake">
                    <div class="result-title">
                        ❌ FAKE NEWS
                    </div>
                    <div class="confidence">
                        AI Confidence: {confidence:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-real">
                    <div class="result-title">
                        ✅ REAL NEWS
                    </div>
                    <div class="confidence">
                        AI Confidence: {confidence:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # PROBABILITIES
        # =================================================

        st.markdown("### 📊 Prediction Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                <div class="prob-card">
                    <div class="prob-title">
                        FAKE PROBABILITY
                    </div>

                    <div class="prob-value">
                        {fake_probability:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                int(fake_probability)
            )


        with col2:

            st.markdown(
                f"""
                <div class="prob-card">
                    <div class="prob-title">
                        REAL PROBABILITY
                    </div>

                    <div class="prob-value">
                        {real_probability:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                int(real_probability)
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "<center>🛡️ Fake News Detection AI • BERT-based Fake News Detection</center>",
    unsafe_allow_html=True
)