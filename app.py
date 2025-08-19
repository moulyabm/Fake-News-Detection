import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
import nltk

# Download stopwords (first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = text.split()
    cleaned = [word for word in words if word not in stop_words]
    return " ".join(cleaned)

# Streamlit UI
st.title("📰 Fake News Detection App")
st.subheader("Enter a news headline and check if it's Real or Fake!")

user_input = st.text_input("🔍 News Headline", "")

if st.button("Check"):
    cleaned_input = clean_text(user_input)
    vectorized_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vectorized_input)

    if prediction[0] == 1:
        st.success("✅ This is REAL news.")
    else:
        st.error("❌ This is FAKE news.")