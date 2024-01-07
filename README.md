# emotion-detection
final project for nyu ece-gy6143
# Emotion Detection from Text

## Overview
This project focuses on detecting emotions from text using machine learning models. It leverages a dataset that comprises 416809 texts, each associated with one of six emotions: joy, sadness, anger, fear, love, and surprise. 

## Dataset
The dataset used is the CARER dataset, which is sourced from Twitter data and is preprocessed for use. It is available at [dair-ai/emotion_dataset](https://github.com/dair-ai/emotion_dataset).

## Text Vectorization
For text vectorization, the TF-IDF (Term Frequency-Inverse Document Frequency) approach is utilized to convert text into a numerical format that machine learning models can process.

## Machine Learning Models
Three machine learning models are experimented with in this project:
- Naive Bayes (MultinomialNB)
- Logistic Regression
- Convolutional Neural Network (CNN)

## Model Training and Testing
Models are trained on the dataset, and their performances are compared. The report details the accuracy and F1-scores for each model. The Logistic Regression model demonstrated the highest accuracy.
## Result
Naive Bayes: 0.76
Logistic Regression: 0.88
CNN: 0.9064
## Application
An application is developed that allows users to input text and receive an emotion prediction along with the probability for each label. The models and tokenizer are saved using `joblib` and `pickle` for this purpose.

## Files
- `merged_training.pkl`: The preprocessed dataset.
- `tfidf_vectorizer.pkl`: The TF-IDF vectorizer.
- `naive_bayes_model.pkl`: The saved Naive Bayes model.
- `logistic_regression_model.joblib`: The saved Logistic Regression model.
- `CNN.h5`: The saved CNN model.
- `tokenizer.pickle`: The tokenizer used for the CNN model.
- `project.ipynb`: Jupyter notebook containing the project code.
- `report.pdf`: The report detailing the project's methodology and results.

## Usage
To use the application, run the `project.ipynb` notebook and follow the instructions for text input.

## Requirements
- Python 3
- scikit-learn
- Keras/TensorFlow
- Pandas
- NumPy
- matplotlib
- seaborn

## License
This project is open-source and available under the MIT License.
