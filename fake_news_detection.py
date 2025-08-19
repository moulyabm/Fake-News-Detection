import pandas as pd

# Load datasets
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Add label column
fake_df['label'] = 0
true_df['label'] = 1

# Combine the two into one dataset
data = pd.concat([fake_df, true_df], axis=0)

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# Show the first few rows
print(data.head())
import string
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define a function to clean text
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove stopwords
    words = text.split()
    cleaned = [word for word in words if word not in stop_words]
    return " ".join(cleaned)

# Apply cleaning function to the title column
data['cleaned_title'] = data['title'].apply(clean_text)

# Show cleaned data
print(data[['title', 'cleaned_title']].head())
from sklearn.feature_extraction.text import TfidfVectorizer

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the cleaned titles
X = vectorizer.fit_transform(data['cleaned_title'])

# Set the target labels (0 = fake, 1 = real)
y = data['label']

print("TF-IDF Matrix Shape:", X.shape)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
import joblib

# Save the trained model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
# Load model and vectorizer (if you're using it later)
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# New headline
new_title = ["Breaking: Vaccine causes autism, scientists warn"]

# Preprocess and predict
new_title_vectorized = vectorizer.transform(new_title)
prediction = model.predict(new_title_vectorized)

print("Prediction:", "Fake News" if prediction[0] == 1 else "Real News")

