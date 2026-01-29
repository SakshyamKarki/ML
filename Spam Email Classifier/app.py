import streamlit as st
import joblib
import pandas as pd

st.title("Spam Classification App")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

email = st.text_input("Email")


if st.button("Classify Email"):
    input_data = {
        "text": email
    }

    df = pd.DataFrame([input_data])

    text = vectorizer.transform(df["text"])

    prediction = model.predict(text)[0]
    if prediction == "spam":
        st.error(f"Classified Email: {prediction}")
    else:
        st.success(f"Classified Email: {prediction}")

