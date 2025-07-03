# movie_recommendation.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.models import load_model

# Load datasets
users = pd.read_csv('ml-1m/users.dat', sep='::', engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='ISO-8859-1')
movies = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='ISO-8859-1')

# Preprocess data
data = pd.merge(ratings, movies, on='MovieID')
user_enc = LabelEncoder()
movie_enc = LabelEncoder()
data['UserID'] = user_enc.fit_transform(data['UserID'])
data['MovieID'] = movie_enc.fit_transform(data['MovieID'])
n_users = data['UserID'].nunique()
n_movies = data['MovieID'].nunique()

# Create utility matrix and apply SVD
utility_matrix = np.zeros((n_users, n_movies))
for row in data.itertuples():
    utility_matrix[row.UserID, row.MovieID] = row.Rating

svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(utility_matrix)
movie_factors = svd.components_.T

# Build recommendation function
def get_recommendations(user_id, top_n=5):
    predicted_ratings = np.dot(user_factors[user_id, :], movie_factors.T)
    top_movie_ids = predicted_ratings.argsort()[::-1][:top_n]
    return movies.iloc[top_movie_ids][['Title', 'Genres']].to_dict(orient="records")
