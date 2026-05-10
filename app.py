import streamlit as st
import pandas as pd
from src.data_processing import load_and_clean_data
from src.hybrid_model import HybridRecommender

# 1. Page Configuration
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

st.title("🎬 Hybrid Movie Recommendation System")
st.markdown("""
This system combines **Content-Based Filtering** (Movie Attributes) and **Collaborative Filtering** (User Behavior) 
to provide personalized movie suggestions.
""")

# 2. Load Data & Model (Using cache for performance)
@st.cache_resource
def prepare_data():
    # Paths to your local CSV files
    movies_path = "data/raw/movies.csv"
    ratings_path = "data/raw/ratings.csv"
    
    # Loading and cleaning
    ratings, movies, _ = load_and_clean_data(movies_path, ratings_path)
    
    # Initializing the Hybrid Engine
    recommender = HybridRecommender(movies_df=movies, ratings_df=ratings)
    return movies, recommender

try:
    movies_df, hybrid_engine = prepare_data()

    # 3. Sidebar UI
    st.sidebar.header("User Settings")
    
    # Select User ID (Sample list, can be dynamic from ratings_df)
    user_list = [1, 10, 50, 100, 150] 
    selected_user = st.sidebar.selectbox("Select User ID:", user_list)

    # Select Movie Title
    movie_list = movies_df['title'].values
    selected_movie = st.sidebar.selectbox("Which movie did you like?", movie_list)

    # Number of recommendations
    num_rec = st.sidebar.slider("Number of Recommendations:", 5, 20, 10)

    # 4. Results Display
    if st.button("Get Recommendations"):
        with st.spinner('Analyzing data and calculating scores...'):
            recommendations = hybrid_engine.get_hybrid_recommendations(
                user_id=selected_user, 
                movie_title=selected_movie, 
                top_n=num_rec
            )
            
            st.subheader(f"✨ Recommended for you based on '{selected_movie}':")
            
            # Displaying results in a clean table
            st.table(recommendations.reset_index(drop=True))
            
            st.success("Recommendations generated successfully!")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please ensure your CSV files are located in 'data/raw/' folder.")