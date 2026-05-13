# Manglish Sentiment Analysis

This project focuses on sentiment analysis for Manglish (Malay-English code-mixed language) using machine learning techniques in Python.

## Objectives

* Clean and preprocess Manglish text data
* Label sentiments into positive, negative, and neutral categories
* Train sentiment classification models
* Evaluate model performance using machine learning metrics

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* Multinomial Naive Bayes

## Project Structure

```bash id="33yjsm"
data/
│
├── cleaned_manglish.csv
├── labeled_dataset.csv
└── raw_dataManglish_raw.csv

scripts/
│
├── clean_data.py
├── label.py
├── statistic.py
└── train_model.py
```

## Workflow

1. Data collection
2. Data preprocessing
3. Sentiment labeling
4. Feature extraction using TF-IDF
5. Model training
6. Model evaluation

## Current Result

* Accuracy: approximately 62%
* Challenges:

  * Manglish slang
  * code-switching
  * inconsistent spelling
  * weak neutral sentiment prediction

## Future Improvements

* Larger dataset
* Better preprocessing
* Deep learning models
* Transformer-based NLP approaches
* Balanced sentiment classes
