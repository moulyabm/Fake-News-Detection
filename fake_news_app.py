import streamlit as st
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (if not already done)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Set page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detection App")
st.subheader("🔍 Enter a news headline below to verify its authenticity")

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Clean text function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = text.split()
    cleaned = [word for word in words if word not in stop_words]
    return " ".join(cleaned)

# Input box
user_input = st.text_input("Enter News Headline to Verify:")

# Predict on button click
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a news headline.")
    else:
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)

        if prediction[0] == 1:
            st.success("✅ This appears to be *Real News*.")
        else:
            st.error("⚠ This appears to be *Fake News*.")

# Footer / About
st.markdown("---")
st.markdown("### ℹ About")
st.markdown(
    "This Fake News Detection app uses a logistic regression model trained on real and fake news headlines. "
    "It uses TF-IDF for feature extraction and scikit-learn for classification."
)
st.markdown("Made with ❤ using Python and Streamlit")