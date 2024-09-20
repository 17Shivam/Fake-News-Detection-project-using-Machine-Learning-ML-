# Fake-News-Detection-project-using-Machine-Learning-ML-

PROBLEM STATEMENT 

The goal of this project is to develop a machine learning model that can accurately predict whether a news article is fake or true. Your model achieved a 94% accuracy, which is quite high, indicating strong performance.

 THE DATASET used  here is used from kaggle dataset 
 Preprocessing Steps:

Tokenization: Split the news articles into words (tokens).
Lowercasing: Convert all words to lowercase to maintain uniformity.
Stop Words Removal: Remove common words (like "the", "is") that don’t carry much meaning.
Lemmatization/Stemming: Reduce words to their base form.


Key Technologiees used :

NLP: For text preprocessing and feature extraction.
ML (Machine Learning): Algorithms for classification (e.g., Logistic Regression, SVM, Random Forest).
Kaggle: Source of datasets.
Python: Core programming language for the entire pipeline.
Anaconda: Managing dependencies and environments.
Blockchain: Optional, for securing news credibility.
Network Analysis: For analyzing connections between news articles and sources.

CODES USED :

# Import libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset (assuming you have a CSV file with 'text' and 'label' columns)
# 'label' column: 1 for real news, 0 for fake news
df = pd.read_csv('fake_news_dataset.csv')  # Replace with your dataset

# Data preprocessing function
def preprocess(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())
    # Remove stopwords and non-alphanumeric characters
    words = [word for word in words if word.isalnum() and word not in stopwords.words('english')]
    return ' '.join(words)

# Apply the preprocessing function to the text data
df['processed_text'] = df['text'].apply(preprocess)

# Split the data into features and labels
X = df['processed_text']
y = df['label']

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display the confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
