# Project Summary

This project is a machine learning spam classifier that checks SMS messages and predicts whether they are spam or ham.

I used the UCI SMS Spam Collection dataset. The data was cleaned by checking for missing values and removing duplicate messages. TF-IDF was used to convert the message text into numerical features, and Logistic Regression was used to train the classifier.

The model achieved:
- Accuracy: 97.4%
- Precision: 90.6%
- Recall: 88.5%
- F1 Score: 89.6%

The classifier also gives a spam probability and shows the words that contributed to the spam prediction.

The project can be used through the terminal using `predict.py` or through the Streamlit web interface using `app.py`.