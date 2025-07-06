from keras.datasets import imdb
from tensorflow.keras.models import load_model
import streamlit as st
from keras.src.utils import pad_sequences


# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb_cleaned.h5')
# model.summary()


# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review


### Prediction  function

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment_1 = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment_1, prediction[0][0]


# Step 4: User Input and Prediction


DEFAULT_TEXT = "the movie was awesome"

# Define a function to clear input/output when the button is clicked
def clear_all():
    st.session_state.input_text = ""
    st.session_state.output_sentiment = ""
    st.session_state.output_score = ""

if "input_text" not in st.session_state:
    st.session_state.input_text = DEFAULT_TEXT
if "output_sentiment" not in st.session_state:
    st.session_state.output_sentiment = ""
if "output_score" not in st.session_state:
    st.session_state.output_score = ""
if "clear_flag" not in st.session_state:
    st.session_state.clear_flag = False

st.title('IMDB movie review sentiment analysis using simple RNN')

# Input field tied to session state
input_text = st.text_area(
    'Enter a movie review',
    value=st.session_state.get("input_text", DEFAULT_TEXT),
    key="input_text"
)

# Predict button
if st.button('Analyze'):
    if input_text.strip() == "":
        st.warning("Please enter a valid input.")
    else:
        sentiment,score=predict_sentiment(input_text)
        st.session_state.output_sentiment = f"Sentiment: {sentiment}"
        st.session_state.output_score = f"Score: {score}"

# Clear button — uses a callback function
st.button('Clear', on_click=clear_all)

# Display prediction output
if "output_sentiment" in st.session_state and st.session_state.output_sentiment:
    st.write(st.session_state.output_sentiment)
if "output_score" in st.session_state and st.session_state.output_score:
    st.write(st.session_state.output_score)