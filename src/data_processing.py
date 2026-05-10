import pandas as pd

def load_and_clean_data(movies_path, ratings_path):

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path) 
    
    # No missing values or duplications .
    movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
    
    # Merge both tables 
    combined_data = pd.merge(ratings, movies, on='movieId' , how='inner') 
    
    return ratings, movies, combined_data

if __name__ == "__main__":
    ratings_df, movies_df, full_df = load_and_clean_data(r"D:\College\Semmester_6\IntelligentProgramming\Final-Project\movie_recommendation_project\data\raw\movies.csv", r"D:\College\Semmester_6\IntelligentProgramming\Final-Project\movie_recommendation_project\data\raw\ratings.csv")
    
    print("\n---Merged Data---")
    print(full_df[['userId', 'title', 'rating', 'genres']].head())    
    
    