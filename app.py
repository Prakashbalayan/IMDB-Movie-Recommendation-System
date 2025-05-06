import os
import re
import string
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK setup with error handling
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    
    # Download required NLTK data with validation
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
except Exception as e:
    st.error(f"NLTK initialization failed: {str(e)}")
    st.stop()

# Sample data fallback
SAMPLE_MOVIES = [
    ("The Shawshank Redemption", "Two imprisoned men bond over several years, finding solace and eventual redemption through acts of common decency."),
   ("The Godfather", "The aging patriarch of an organized crime dynasty transfers control to his reluctant son."),
   ("Inception", "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into a target's mind."),
    ("Pulp Fiction", "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption."),
    ("Fight Club", "An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more."),
    ("Forrest Gump", "The story of a slow-witted but kind-hearted man from Alabama who witnesses and influences several historical events in 20th-century America."),
   ("The Dark Knight", "Batman sets out to dismantle the remaining criminal organizations in Gotham, but finds himself tested by a rising criminal mastermind known as the Joker."),
   ("The Matrix", "A computer hacker learns that the world around him is a simulated reality and joins a rebellion to free humanity from the machines."),
   ("Gladiator", "A betrayed Roman general fights his way back as a gladiator to avenge his family and challenge the corrupt emperor."),
   ("The Silence of the Lambs", "A young FBI trainee seeks the help of an imprisoned cannibalistic killer to catch another serial killer."),
   ("Interstellar", "A group of explorers travel through a wormhole in space in an attempt to ensure humanity’s survival."),
   ("The Prestige", "Two rival magicians in 19th-century London engage in a bitter competition to create the ultimate stage illusion, with deadly consequences."),
   ("Se7en", "Two detectives hunt a serial killer who uses the seven deadly sins as his modus operandi."),
   ("Whiplash", "A young and ambitious jazz drummer is pushed to his limits by an abusive instructor at a prestigious music conservatory."),
   ("The Departed", "An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in Boston.")

]

def preprocess_text(text):
    """Robust text preprocessing with error handling"""
    if not text or pd.isna(text):
        return ""
    
    try:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        
        # Tokenize with fallback to simple split if tokenizer fails
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Keep important negation words
        keep_words = {'not', 'no', 'but', 'very'}
        stop_words = set(stopwords.words('english')) - keep_words
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        if not tokens:
            return ""
        
        stemmer = PorterStemmer()
        return ' '.join([stemmer.stem(word) for word in tokens])
    except:
        return ""

def get_recommendations(df):
    """Generate recommendations with robust error handling"""
    df['processed'] = df['storyline'].apply(preprocess_text)
    valid_df = df[df['processed'] != ""]
    
    if len(valid_df) < 2:
        st.warning("Not enough valid storylines for recommendations")
        return None
    
    try:
        tfidf = TfidfVectorizer(min_df=1)
        tfidf_matrix = tfidf.fit_transform(valid_df['processed'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return valid_df, cosine_sim
    except Exception as e:
        st.error(f"Recommendation engine failed: {str(e)}")
        return None

def main():
    st.title("🎬 Movie Recommendation System")
    st.write("This system recommends similar movies based on their storylines.")
    
    # Use sample data directly (removed scraping for reliability)
    df = pd.DataFrame(SAMPLE_MOVIES, columns=['title', 'storyline'])
    
    with st.spinner("Analyzing movies..."):
        result = get_recommendations(df)
        if not result:
            st.stop()
            
        valid_df, cosine_sim = result
        
        selected = st.selectbox("Choose a movie:", valid_df['title'])
        
        if st.button("Get Recommendations"):
            try:
                idx = valid_df[valid_df['title'] == selected].index[0]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]  # Top 3
                
                st.subheader("Recommended Movies:")
                for i, (idx, score) in enumerate(sim_scores, 1):
                    movie = valid_df.iloc[idx]['title']
                    with st.container():
                        st.write(f"**{i}. {movie}** (similarity: {score:.2f})")
                        with st.expander("See storyline"):
                            st.write(valid_df.iloc[idx]['storyline'])
            except Exception as e:
                st.error(f"Could not generate recommendations: {str(e)}")

if __name__ == "__main__":
    main()