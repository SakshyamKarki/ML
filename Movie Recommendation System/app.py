import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load files
df = pd.read_csv("movies.csv")
tfidf = joblib.load("tfidf.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")

st.title("Movie Recommendation System")

title_input = st.text_input("Enter a movie title:")

if st.button("Recommend"):

    # Clean title
    df['year'] = df['title'].str.extract(r'\((\d{4})\)')
    df['year'] = pd.to_datetime(df['year'], format='%Y', errors='coerce')
    df['title'] = df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
    
    clean_title = title_input.replace(r'\s*\(\d{4}\)', '').strip()

    # Find movie index
    if clean_title not in df['title'].values:
        st.error("Movie not found in database")
    else:
        idx = df[df['title'] == clean_title].index[0]

        # Compute similarity ONLY for selected movie
        sim_scores = cosine_similarity(
            tfidf_matrix[idx], tfidf_matrix
        ).flatten()

        # Get top matches
        top_indices = sim_scores.argsort()[-11:-1][::-1]

        recommendations = df.iloc[top_indices][['title', 'genres']]

        st.subheader("Recommended Movies")
        for _, row in recommendations.iterrows():
            st.write(f"🎬 {row['title']} — {row['genres']}")
