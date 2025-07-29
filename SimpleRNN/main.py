##loading Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from  tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.models import load_model
import streamlit as st

#LOADING DATASET
word_index = imdb.get_word_index()
reverse_word_index = {k: (v + 3) for k, v in word_index.items()} 


#Load Pre-trained Model
model = load_model('simplernn_model.h5')  # Load the saved model weights

#helper functions for text processing
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # 2 is the index for OOV (out of vocabulary)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)  # Pad to the same length as training data
    return padded_review

#Prediction Function
def predict_sentiment(review):
    preprocessed_review = preprocess_text(review)
    model.predict(preprocessed_review)
    sentiment = model.predict(preprocessed_review)[0][0]
    if sentiment > 0.5:
        return "Positive"
    else:
        return "Negative"
    return sentiment 


##Streamlit App
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative).")

#user input for prediction

user_input = st.text_area("Enter your movie review here:")

if st.button("Predict Sentiment"):
    preprocessed_review = preprocess_text(user_input)
    prediction = model.predict(preprocessed_review)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]:.4f}")
else:
    st.write("Please enter a movie review to get the sentiment prediction.")
