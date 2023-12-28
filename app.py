# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer=Tokenizer()

# Load the dataset 
final_dataset=pd.read_csv('final_dataset.csv')
aspect_mapping = {aspect: idx for idx, aspect in enumerate(final_dataset['Aspect Term'].unique())}
sentiment_mapping = {sentiment: idx for idx, sentiment in enumerate(final_dataset['polarity'].unique())}


# Load the saved model
loaded_model = load_model('aspect_sentiment_model.h5')

# Function to preprocess user input and make predictions
def predict_aspect_sentiment(user_input):
    # Tokenize and pad the user input
    user_sequence = tokenizer.texts_to_sequences([user_input])
    user_padded_sequence = pad_sequences(user_sequence, maxlen=78)

    # Make predictions
    predictions = loaded_model.predict(user_padded_sequence)
    predicted_aspect_index = np.argmax(predictions[0])
    predicted_sentiment_index = np.argmax(predictions[0])

    # Map back to aspect and sentiment labels
    predicted_aspect = {v: k for k, v in aspect_mapping.items()}[predicted_aspect_index]
    predicted_sentiment = {v: k for k, v in sentiment_mapping.items()}[predicted_sentiment_index]

    sentiment_labels = {0: "Negative", 1: "Positive", 2: "Neutral"}
    predicted_sentiment = sentiment_labels[predicted_sentiment_index]
    predicted_aspect = predicted_aspect.capitalize()

    return predicted_aspect, predicted_sentiment

# Streamlit app
st.header("Aspect and Sentiment Prediction App")

st.subheader("🏬 Sentiment Analysis on Laptop and Restaurant Reviews 💻")

# User input
st.markdown("Enter the reviews of laptop or restaurant:")
user_input = st.text_area("Text")

# Predict and display results
if st.button("Predict"):
    if user_input:
        aspect, sentiment = predict_aspect_sentiment(user_input)
        st.success(f"Predicted Aspect : {aspect } , Predicted Sentiment : { sentiment}")
    else:
        st.warning("Please enter some text.")
