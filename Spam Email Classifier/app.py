import streamlit as st
import joblib
import numpy as np

# ===== Load Models =====
nb_model = joblib.load("model.pkl")
count_vectorizer = joblib.load("vectorizer.pkl")

svm_model = joblib.load("spam_model.pkl")      # calibrated SVM
tfidf_vectorizer = joblib.load("tfidf.pkl")

# ===== UI =====
st.title("📧 Spam Email Classifier")
st.write("Choose a model and classify an email")

model_choice = st.selectbox(
    "Select Model",
    ["Naive Bayes (CountVectorizer)", "SVM (TF-IDF)"]
)

email_text = st.text_area("Enter Email Text")

# ===== Prediction =====
if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter email text")
    else:
        if model_choice == "Naive Bayes (CountVectorizer)":
            vec = count_vectorizer.transform([email_text])
            prediction = nb_model.predict(vec)[0]
            prob = nb_model.predict_proba(vec)[0]

            confidence = max(prob)

        else:
            vec = tfidf_vectorizer.transform([email_text])
            prediction = svm_model.predict(vec)[0]
            prob = svm_model.predict_proba(vec)[0]

            confidence = max(prob)

        label = "🚫 Spam" if prediction == 1 else "✅ Ham"

        st.subheader("Result")
        st.write("Prediction:", label)
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
