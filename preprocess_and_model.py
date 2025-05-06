import pandas as pd
import numpy as np
import nltk
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords

# Reading the CSV
def load_data(csv_file="imdb_movies_2024.csv"):
    df = pd.read_csv(csv_file)
    # Dropping rows where Movie Name or Storyline are missing
    df.dropna(subset=["Title", "Description"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Text Cleaning
def clean_text(text):
    # Lowercase
    text = text.lower()

    # Removing punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Removing numbers or special chars if needed
    text = re.sub(r"\d+", "", text)

    # Tokenizing
    tokens = text.split()

    # Removing stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Re-joining
    text = " ".join(tokens)
    return text

# Building TF-IDF Matrix
def build_tfidf_matrix(df):
    # Cleaning each storyline
    df["cleaned_storyline"] = df["Description"].apply(clean_text)

    # Useing TF-IDF Vectorizer
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["cleaned_storyline"])

    return tfidf, tfidf_matrix

# Cosine Similarity
def get_cosine_similarity(tfidf_matrix):
    # Calculating the cosine similarity matrix for all movies
    cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cos_sim

# Recommending Movies
def recommend_movies(input_storyline, df, tfidf, tfidf_matrix, top_n=5):
    # Cleaning the input storyline
    cleaned_input = clean_text(input_storyline)

    # Transforming the input text into the existing TF-IDF space
    input_vector = tfidf.transform([cleaned_input])

    # Computing similarity with all movies
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()

    # Getting top N indices (excluding the input itself)
    top_indices = similarity_scores.argsort()[::-1][:top_n]

    # Returning the top N movies
    results = df.iloc[top_indices][["Title", "Description"]]
    results["Similarity Score"] = similarity_scores[top_indices]
    return results

import streamlit as st
import pandas as pd

@st.cache_data
def load_and_prepare_data():
    df = load_data("imdb_movies_2024.csv")
    tfidf, tfidf_matrix = build_tfidf_matrix(df)
    return df, tfidf, tfidf_matrix

def main():
    st.title("IMDb Movie Recommendation System (2024)")

    # Loading and preparing data
    df, tfidf, tfidf_matrix = load_and_prepare_data()

    st.write("""
    **Instructions**:  
    1. Enter a brief storyline or plot description in the text box below.  
    2. Click 'Recommend Movies' to see the top 5 similar movies.  
    """)
    
    user_input = st.text_area("Enter a movie storyline/plot here...")

    if st.button("Recommend Movies"):
        if user_input.strip() == "":
            st.warning("Please enter a storyline first.")
        else:
            recommendations = recommend_movies(user_input, df, tfidf, tfidf_matrix, top_n=5)
            st.subheader("Top 5 Recommended Movies:")
            for i, row in recommendations.iterrows():
                st.markdown(f"**{row['Title']}**") 
                st.write(f"Similarity Score: {row['Similarity Score']:.4f}")
                st.write(f"Storyline: {row['Description']}")
                st.write("---")


if __name__ == "__main__":
    main()