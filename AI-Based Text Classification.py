
# Practical 8: AI-Based Text Classification
# Sentiment Analysis using Naive Bayes

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
sentences = [
    "I love this product",
    "This is an amazing experience",
    "I am very happy",
    "The service is excellent",
    "I hate this product",
    "This is a terrible experience",
    "I am very disappointed",
    "The service is very bad",
    "The product is okay",
    "It is an average experience",
    "Nothing special",
    "The service is normal"
]

# Labels for training data
labels = [
    "Positive", "Positive", "Positive", "Positive",
    "Negative", "Negative", "Negative", "Negative",
    "Neutral", "Neutral", "Neutral", "Neutral"
]

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# Create and train AI model
model = MultinomialNB()
model.fit(X, labels)

print("================================")
print(" AI TEXT CLASSIFICATION SYSTEM ")
print("================================")

# Take input from user
while True:

    user_text = input("\nEnter a sentence (or type exit): ")

    if user_text.lower() == "exit":
        print("Thank you for using the AI system!")
        break

    # Convert input into numerical format
    input_data = vectorizer.transform([user_text])

    # Predict sentiment
    prediction = model.predict(input_data)

    # Display result
    print("Predicted Sentiment:", prediction[0])

    # Explainability
    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities) * 100

    print("Prediction Confidence:", round(confidence, 2), "%")
