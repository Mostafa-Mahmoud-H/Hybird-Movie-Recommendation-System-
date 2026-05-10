from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_content_engine(movies_df):

    tfidf = TfidfVectorizer(stop_words='english')
    
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim

def get_content_recommendations(movie_title, movies_df, cosine_sim):
    
    if movie_title not in movies_df['title'].values:
        return "Unfortunately, this film is not in the database."

    idx = movies_df[movies_df['title'] == movie_title].index[0]


    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]
    
    return movies_df.iloc[movie_indices][['title', 'genres']]