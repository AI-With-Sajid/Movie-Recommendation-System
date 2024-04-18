from concurrent.futures import ThreadPoolExecutor
from src.data_loader import load_dataset
from src.data_preprocessing import preprocess_data
from src.vectorizer import vectorize_tags
import requests
import streamlit as st
import pickle
import numpy as np


# fetching posters
@st.cache_data
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8e259356ec590251aa2740c6ca8e5700&language=en-US"
    data = requests.get(url).json()
    poster_path = data['poster_path']
    full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return full_path


# recommender function

# loading dataset
df_merged = load_dataset()

## Selection of features
list_selected_cols = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
df_merged = df_merged[list_selected_cols]

# Preprocess data
df_final = preprocess_data(df_merged)

# Obtaining vectors and their similarities
arr_similarity = vectorize_tags(df_final['tags'])


def recommend_movies(str_movie):
    row_movie = df_final[df_final['title'] == str_movie]
    if row_movie.empty:
        print("Movie not found in the dataset.")
        return

    ind_movie = row_movie.index[0]
    arr_distance = arr_similarity[ind_movie]

    # Sorting and selecting top 5 indices
    top_indices = np.argsort(arr_distance)[::-1][1:6]

    # Using vectorized operations to get recommended_movies and movie_ids
    recommended_movies = df_final.iloc[top_indices]['title'].tolist()
    movie_ids = df_final.iloc[top_indices]['movie_id'].tolist()

    # Use ThreadPoolExecutor for parallel API calls to fetch posters
    with ThreadPoolExecutor() as executor:
        posters = list(executor.map(fetch_poster, movie_ids))

    # Filter out None values (errors in fetching posters)
    recommended_movies = [movie for movie, poster in zip(recommended_movies, posters) if poster is not None]
    posters = [poster for poster in posters if poster is not None]

    return recommended_movies, posters