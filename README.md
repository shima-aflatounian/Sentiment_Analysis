# Twitter Airline Sentiment Analysis
This project analyzes customer sentiment in tweets mentioning major US airlines like American, United, Delta etc. The goal is to build machine learning models that can accurately predict sentiment in new tweets in real-time.

## Data
The dataset contains around 26,000 tweets extracted using relevant airline hashtags and handles. It includes tweet text and a sentiment label (positive/negative/neutral).

## Technologies
Python
NLTK
Scikit-Learn
Tensorflow/Keras (for deep learning models)
Methodology
Data Preprocessing

## Text cleaning
Tokenization
Stemming
Stopword removal
Feature Extraction

## Word presence
Word count
N-grams
Parts of speech tags
Sentiment lexicons

## Modeling
Naive Bayes
SVM
Neural Networks (LSTM, CNN)
Evaluation

## Train-test split
Hyperparameter tuning
Precision, recall, F1-score
NLTK Library
NLTK is a popular NLP library used for:

Text processing functions like tokenization, stemming, tagging
Corpora and lexical resources for classifiers

## Metrics for evaluation
Tools for analysis like concordance, dispersion plots
This project leverages NLTK for text preprocessing and feature engineering to build robust sentiment analysis models.

## Results
The best deep learning model achieves 80% accuracy on real-time sentiment prediction of newly extracted tweets. This helps airlines monitor brand perception and address customer pain points.
