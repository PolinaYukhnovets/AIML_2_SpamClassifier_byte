import joblib
import numpy as np
import re

model = joblib.load("models/spam_classifier.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

spam_index = list(model.classes_).index("spam")
feature_names = tfidf.get_feature_names_out()

if spam_index == 1:
    spam_coefficients = model.coef_[0]
else:
    spam_coefficients = -model.coef_[0]

def analyse_message(message, top_n=5):
    message_tfidf = tfidf.transform([message])

    prediction = model.predict(message_tfidf)[0]

    probabilities = model.predict_proba(message_tfidf)[0]
    spam_probability = probabilities[spam_index]

    contributions = message_tfidf.toarray()[0] * spam_coefficients
    top_indices = np.argsort(contributions)[::-1]

    spam_terms = []

    for index in top_indices:
        if contributions[index] > 0:
            spam_terms.append(feature_names[index])

        if len(spam_terms) == top_n:
            break

    return prediction, spam_probability, spam_terms

message = input("Enter an SMS or email message: ")

def highlight_terms(message, terms):
    highlighted_message = message

    for term in terms:
        highlighted_message = re.sub(
            rf"\b({re.escape(term)})\b",
            r"[\1]",
            highlighted_message,
            flags=re.IGNORECASE
        )

    return highlighted_message

prediction, probability, terms = analyse_message(message)
highlighted_message = highlight_terms(message, terms)

print("\n--- Analysis Result ---")
print("Prediction:", prediction.upper())
print(f"Spam probability: {probability:.2%}")

if terms:
    print("Spam-indicative terms:", ", ".join(terms))
else:
    print("Spam-indicative terms: None")

print("Highlighted message:", highlighted_message)
