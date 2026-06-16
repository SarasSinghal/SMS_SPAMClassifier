SMS Spam Classifier

A machine learning pipeline that classifies SMS messages as spam or ham (not spam) using Natural Language Processing and a Multinomial Naive Bayes model.


Project Overview

This project trains a text classification model on a labeled SMS dataset. It covers the full ML workflow — data cleaning, exploratory analysis, NLP preprocessing, model comparison across 11 algorithms, and final model export for deployment.


Files

FileDescriptionmodel__2_.ipynbMain notebook — EDA, preprocessing, training, evaluationspam.csvRaw dataset (5,572 SMS messages labeled ham/spam)model.pklTrained Multinomial Naive Bayes classifiervectorizer.pklFitted TF-IDF vectorizer


Dataset


Source: spam.csv (UCI SMS Spam Collection)
Size: 5,572 messages → 5,169 after deduplication
Class distribution: ~86.6% ham, ~13.4% spam
Columns used: v1 (label), v2 (message text)



Pipeline

1. Data Cleaning


Dropped unused columns (Unnamed: 2, Unnamed: 3, Unnamed: 4)
Renamed v1 → target, v2 → msg
Label-encoded target: ham = 0, spam = 1
Removed duplicate rows


2. Exploratory Data Analysis


Class distribution pie chart
Feature engineering: character count, word count, sentence count per message
Comparative statistics between spam and ham messages
Correlation heatmap
Word clouds for spam and ham
Top-30 most common words in each class (bar charts)


3. Text Preprocessing (transform_text)

Each message is processed through:


Lowercasing
Tokenization (NLTK)
Removal of non-alphanumeric tokens
Stopword and punctuation removal
Porter Stemming


4. Feature Extraction


Vectorizer: TF-IDF (TfidfVectorizer)
Train/Test split: 80/20, random_state=2


5. Model Comparison

Eleven classifiers were evaluated on precision and accuracy:

ModelNotesMultinomial Naive Bayes✅ Selected for deploymentBernoulli Naive BayesGaussian Naive BayesLogistic RegressionL1 penalty, liblinearSVMSigmoid kernelK-Nearest NeighborsDecision Treemax_depth=5Random Forest50 estimatorsAdaBoost50 estimatorsBagging Classifier50 estimatorsExtra Trees50 estimatorsGradient Boosting50 estimatorsXGBoost50 estimators

MultinomialNB was selected as the final model based on its high precision (minimizing false positives is critical for spam detection).
