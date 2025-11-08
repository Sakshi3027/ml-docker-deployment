"""
Train a sentiment analysis model and save it to disk
Run this file first to create the model
"""

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Expanded training data for better accuracy
positive_texts = [
    "I love this product, it's amazing!",
    "This is the best thing ever!",
    "Absolutely wonderful experience",
    "Great quality and fast shipping",
    "I'm very happy with my purchase",
    "Excellent service and support",
    "This exceeded my expectations",
    "Fantastic product, highly recommend",
    "Outstanding quality and performance",
    "Perfect! Just what I needed",
    "Love it! Will buy again",
    "Impressive and reliable",
    "Superb quality, very satisfied",
    "Amazing value for money",
    "Brilliant product, works perfectly",
    "Delighted with this purchase",
    "Top notch quality",
    "Exactly what I was looking for",
    "Five stars! Highly recommended",
    "Best purchase I've made this year",
    "Incredible product, love it",
    "Very pleased with the results",
    "Exceptional quality and service",
    "Worth every penny",
    "Absolutely fantastic!",
    "Great experience overall",
    "Couldn't be happier",
    "This is awesome!",
    "Really impressed with this",
    "Wonderful product, great value"
]

negative_texts = [
    "Terrible product, waste of money",
    "Very disappointed with this purchase",
    "Poor quality, not worth it",
    "Awful experience, would not recommend",
    "This is the worst thing I've bought",
    "Completely unsatisfied",
    "Bad customer service",
    "Don't buy this, it's horrible",
    "Not as advertised, very misleading",
    "Broke after one use",
    "Total waste of time and money",
    "Cheaply made, fell apart quickly",
    "Would give zero stars if I could",
    "Disappointing and overpriced",
    "Does not work as expected",
    "Regret this purchase completely",
    "Horrible quality, stay away",
    "Not recommended at all",
    "Failed to meet expectations",
    "Useless product, don't waste your money",
    "Very poor quality control",
    "Defective product, very upset",
    "Worst customer service ever",
    "Completely broken on arrival",
    "Not worth the price",
    "Terrible experience from start to finish",
    "Avoid this at all costs",
    "Really disappointed",
    "This is garbage",
    "Absolutely terrible product"
]

# Combine texts and create labels
texts = positive_texts + negative_texts
labels = [1] * len(positive_texts) + [0] * len(negative_texts)

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Positive samples: {sum(labels)}")
print(f"Negative samples: {len(labels) - sum(labels)}\n")

# Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ('classifier', MultinomialNB(alpha=0.1))
])

# Train the model
print("Training the model...")
model_pipeline.fit(X_train, y_train)

# Test the model
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Save the model
import os
os.makedirs('models', exist_ok=True)
model_path = 'models/sentiment_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_pipeline, f)

print(f"✓ Model saved to {model_path}")

# Test predictions
test_texts = [
    "This product is absolutely amazing!",
    "I hate this, it's terrible",
    "Not bad, pretty decent",
    "Best purchase ever!",
    "Waste of money",
    "Pretty good quality",
    "Horrible experience",
    "Love it so much!",
    "Disappointed with quality",
    "Excellent value for money"
]

print("\n" + "="*60)
print("TEST PREDICTIONS")
print("="*60)
for text in test_texts:
    prediction = model_pipeline.predict([text])[0]
    probability = model_pipeline.predict_proba([text])[0]
    sentiment = "Positive ✓" if prediction == 1 else "Negative ✗"
    confidence = max(probability) * 100
    print(f"\n'{text}'")
    print(f"→ {sentiment} (Confidence: {confidence:.1f}%)")

print("\n" + "="*60)
print("✓ Model training complete! You can now run the API.")
print("="*60)