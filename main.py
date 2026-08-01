#Step1: import libraries and load the model
import numpy as np
import tensorflow as tf
from jsonschema.benchmarks.const_vs_enum import value
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model




#Load the imdb dataset word index
word_index=imdb.get_word_index()
reverse_word_index={value: key for key, value in word_index.items()}

#load the pre-trained model with relu activation
model=load_model('simple_rnn_imdb.keras')

#Step2:helper function
#function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

#function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2) + 3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

import streamlit as st
#streamlit app
st.title('IMDB Movie Reviews Sentiment Analysis')
st.write("Enter a movie review to classify it as positive or negative")

##user input
user_input = st.text_input("Enter a movie review")
if st.button('classify'):
    preprocessed_input = preprocess_text(user_input)

    ## Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    ## Display The Result

    st.write(f'sentiment: {sentiment}')
    st.write(f'prediction_score: {prediction}')

else:
    st.write('Please enter a movie review')