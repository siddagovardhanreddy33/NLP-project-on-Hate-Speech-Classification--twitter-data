# app.py

import streamlit as st
import tensorflow as tf
import pickle
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences

# PAGE CONFIG
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered"
)

# LOAD MODEL
model = tf.keras.models.load_model('Models/lstm_sentiment_model.h5')

# LOAD TOKENIZER

with open('Models/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# TEXT CLEANING FUNCTION
def clean_text(text):

    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# APP TITLE
st.title("💬 Twitter Sentiment Analysis")
st.write(
    "This NLP application predicts whether a tweet "
    "contains Positive or Negative sentiment using "
    "an LSTM Deep Learning model."
)




# TEXT INPUT

user_input = st.text_area("Enter a Tweet",height=150)

# PREDICTION BUTTON

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:
        # Clean text
        cleaned_text = clean_text(user_input)
        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([cleaned_text])

        # Pad sequence
        padded_sequence = pad_sequences(sequence,maxlen=100)

        # Predict
        prediction = model.predict( padded_sequence)
        prediction_score = prediction[0][0]

        # DISPLAY RESULT

        st.subheader("Prediction Result")
        if prediction_score >= 0.5:
            st.error("🔴 Negative Sentiment")
        else:
            st.success("🟢 Positive Sentiment")

        # DISPLAY SCORE

        st.write(f"Prediction Score: {prediction_score:.4f}")

# SIDEBAR

st.sidebar.title("📌 About Project")

st.sidebar.write(
    """
    ### NLP Sentiment Analysis
    
    This project uses:
    
    - TensorFlow
    - Keras
    - LSTM
    - Streamlit
    
    Features:
    - Text Cleaning
    - Tokenization
    - Sequence Padding
    - Deep Learning Prediction
    """
)

# FOOTER

st.markdown("---")
st.caption("Built using TensorFlow, Keras, and Streamlit")