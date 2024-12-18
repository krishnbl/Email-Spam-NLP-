import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize the Stemmer
ps = PorterStemmer()

# Function to preprocess text
def tranform_text(txt):
    txt = txt.lower()
    txt = nltk.word_tokenize(txt)
    y = [i for i in txt if i.isalnum()]
    txt = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    txt = [ps.stem(i) for i in txt]
    return " ".join(txt)

# Load the saved models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Add custom background CSS (grey background)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #DCDCDC;  /* Light Grey color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and input area
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = tranform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
