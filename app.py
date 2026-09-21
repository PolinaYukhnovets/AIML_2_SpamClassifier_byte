import streamlit as st
import joblib
import numpy as np


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


st.title("Spam & Phishing Message Detector")

st.write(
    "Enter an SMS or email message below to check whether it may be spam."
)

message = st.text_area(
    "Message",
    placeholder="Paste your SMS or email message here..."
)

if st.button("Analyse Message"):

    if message.strip():

        prediction, probability, terms = analyse_message(message)

        if prediction == "spam":
            st.error("SPAM DETECTED")

            st.metric(
                "Spam Probability",
                f"{probability:.2%}"
            )

            if terms:
                st.write("**Spam-indicative terms:**")
                st.write(", ".join(terms))

        else:
            st.success("HAM - Message appears legitimate")

            st.metric(
                "Spam Probability",
                f"{probability:.2%}"
            )

            st.write("No spam warning was triggered for this message.")

    else:
        st.warning("Please enter a message to analyse.")