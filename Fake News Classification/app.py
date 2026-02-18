import streamlit as st

st.title("Fake News Classifier")

from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import joblib

model = joblib.load('classify_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def preprocess_text(text_data):
    preprocessed_text = []
    
    for sentence in tqdm(text_data):
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(token.lower()
                                  for token in str(sentence).split()
                                  if token not in stopwords.words('english')))

    return preprocessed_text

text = st.text_area("Enter the news text to classify:", height=200)
if st.button("Classify"):
    if text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        preprocessed_text = preprocess_text([text])
        vectorized_text = vectorizer.transform(preprocessed_text)
        prediction = model.predict(vectorized_text)

        if prediction[0] == 1:
            st.success("The news is classified as: REAL")
        else:
            st.error("The news is classified as: FAKE")