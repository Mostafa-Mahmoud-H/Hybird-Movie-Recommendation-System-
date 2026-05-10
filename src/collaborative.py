from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import pandas as pd

def train_collaborative_model(ratings_df):

    
    reader = Reader(rating_scale=(ratings_df['rating'].min(), ratings_df['rating'].max()))

    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

    trainset = data.build_full_trainset()

    model = SVD()


    model.fit(trainset)
    
    return model

def predict_rating(user_id, movie_id, model):
    """
Predicting the rating that a particular user will give to a particular film.
    """
    prediction = model.predict(user_id, movie_id)
    return prediction.est 