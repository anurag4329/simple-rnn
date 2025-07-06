🎬 IMDB Sentiment Analysis with Simple RNN


This project demonstrates a basic deep learning pipeline for classifying movie reviews from the IMDB dataset as either positive or negative, using a Simple Recurrent Neural Network (RNN) built with Keras. The final model is deployed as an interactive Streamlit web app, allowing users to test custom reviews in real time.


📌 Project Overview
Dataset: IMDB movie review dataset (keras.datasets.imdb)

Model Type: Simple RNN with ReLU activation

Goal: Predict whether a review expresses a positive or negative sentiment

Deployment: Real-time web interface using Streamlit



🧠 Model Architecture
Embedding Layer: Transforms words into 128-dimensional vectors

SimpleRNN Layer: 128 units with ReLU activation

Dense Output Layer: Single neuron with sigmoid activation for binary classification (positive/negative)


🧪 Training Details
Input sequences padded to a max length of 500

Trained on 10,000 most frequent words

Optimizer: Adam

Loss function: Binary Crossentropy

Metrics: Accuracy

Includes EarlyStopping to prevent overfitting (patience = 5)



🚀 Streamlit Web App

* Input any movie review text

* Get sentiment prediction (Positive / Negative)

* View the raw prediction confidence score

* Button to clear inputs and outputs

▶️ Running the App
To launch the web app locally:

`streamlit run app.py
`


🛠 Tech Stack

Python 🐍

Keras / TensorFlow 🔧

Streamlit 🌐

IMDB dataset from keras.datasets