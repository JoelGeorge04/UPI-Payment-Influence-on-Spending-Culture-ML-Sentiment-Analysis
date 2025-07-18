#  UPI Payment Influence on Spending Culture – ML Sentiment Analysis

This project explores how UPI (Unified Payments Interface) is influencing spending habits and behavior in India, using sentiment analysis on text data sourced from real-world user comments.

##  Overview

We built our own dataset by scraping comments and discussions related to UPI and digital payments from:
- **YouTube** (using the YouTube Data API)
- **Reddit** (using the Reddit API via PRAW)

These platforms provided diverse, real-world user opinions which were then cleaned and labeled for sentiment analysis (positive, negative, neutral).

##  Features

- Custom dataset built from live API data (no pre-built datasets used)
- Text preprocessing (cleaning, tokenization, stopword removal)
- Feature extraction using TF-IDF
- Evaluation using multiple ML models
- Sentiment classification into positive, negative.

##  Models Used

We trained and evaluated the dataset using the following machine learning models:

- **LinearSVC**
- **LogisticRegression**
- **RandomForestClassifier**
- **MultinomialNB**

Each model's performance was compared based on accuracy, precision, recall, and F1-score.

##  Results

Our experiments showed that:

- **LinearSVC** and **LogisticRegression** performed best in terms of accuracy.
- **RandomForest** showed slightly better recall but at the cost of speed.
- **MultinomialNB** performed well for simpler, smaller datasets.

##  Technologies Used

- Python
- Scikit-learn
- PRAW (Reddit API)
- YouTube Data API
- Pandas, NumPy
- Matplotlib / Seaborn (for visualizations)

##  Dataset

The dataset was built from scratch using the above APIs. Due to size or license constraints, only a sample may be included here. Instructions to regenerate the dataset are provided in the `data_collection` folder.

##  How to Run

1. Clone this repository
2. Install requirements:
