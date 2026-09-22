# SMS Spam & Phishing Message Classifier

## Project Overview

This project is a machine learning application that classifies SMS messages as either spam or ham (legitimate messages).

The classifier uses TF-IDF to convert message text into numerical features and Logistic Regression to predict whether a message is spam. In addition to the predicted label, the application provides a spam probability score and identifies terms in the message that contributed most strongly towards the spam prediction.

The project includes a Jupyter Notebook for data preprocessing, model training and evaluation, a command-line prediction tool, and a Streamlit web interface for analysing new messages.

## Dataset

## Dataset

I used the SMS Spam Collection dataset from the UCI Machine Learning Repository. It contains SMS messages labelled as either `ham` or `spam`.

Dataset source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Data Preprocessing

Before training the model, I first checked the dataset for missing values and duplicate messages. Duplicate records were removed so that repeated messages would not affect the training and testing results.

The data was then separated into:
- `X` - the SMS messages
- `y` - the ham or spam labels

I used an 80/20 train-test split, meaning 80% of the data I used for training and 20% was kept for testing. I also, used a stratified split to keep a similar proportion of ham and spam messages in both sets.

Since the model cannot work directly with text, I used TF-IDF to convert the messages into numerical features. The text was converted to lowercase and common English stop words were removed.

## Model Training

I used Logistic Regression to train the spam classifier. The model was trained using the TF-IDF features from the training data.

`class_weight="balanced"` was used because the dataset contains more ham messages than spam messages.

## Model Performance

The model was tested on data that was not used during training.

- Accuracy: 97.4%
- Precision: 90.6%
- Recall: 88.5%
- F1 Score: 89.6%

The confusion matrix and model performance graph are saved in the `images` folder.

## Features

- Classifies messages as spam or ham
- Shows the spam probability
- Shows words that contributed to the spam prediction
- Terminal version
- Streamlit web interface

## Project Structure

```text
SpamClassifier/
│
├── data/
│   └── SMSSpamCollection
├── examples/
│   └── example_predictions.csv
├── images/
│   ├── confusion_matrix.png
│   └── model_metrics.png
├── models/
│   ├── spam_classifier.pkl
│   └── tfidf_vectorizer.pkl
├── app.py
├── predict.py
├── spam_classifier.ipynb
├── requirements.txt
└── README.md
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Classifier

To use the classifier in the terminal:

```bash
python predict.py
```

Enter a message when asked. The program will show the prediction, spam probability and spam-indicative terms.

## Run the Web App

To start the Streamlit interface:

```bash
streamlit run app.py
```

Enter an SMS or email message and click **Analyse Message** to see the result.

## Example Predictions

Ten example messages and their predictions are saved in:

```text
examples/example_predictions.csv
```